from __future__ import absolute_import
import argparse
import logging
import os, random, shutil
from selector import Selector2,BertAdam
from io import open
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,Dataset)
import jsonlines,json,pickle
from torch.nn.utils.rnn import pad_sequence
from transformers import XLNetTokenizer
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast,GradScaler
from contextlib import nullcontext

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
logger = logging.getLogger(__name__)

class selector2Features(object):
    def __init__(self,tokens,output,token_mask,tobeselected,selected,positives,negatives):
        self.tokens=tokens
        self.output=output
        self.token_mask=token_mask
        self.tobeselected=tobeselected
        self.selected=selected
        self.positives=positives
        self.negatives=negatives

def read_file(file):
    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'rb') as f:
            return pickle.load(f)

def write_file(data, file):
    if file.endswith('jsonl'):
        with jsonlines.open(file, mode='w') as writer:
            writer.write_all(data)
    if file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

class Selector2Dataset(Dataset):

    def __init__(self, args,split='train'):
        model_size='base' if 'base' in args.model else 'large'
        direc = os.path.join(args.input_dir,split+'_selector2'+model_size+'.pkl')
        features=read_file(direc)
        if split=='train':
            self.inputdata=[feature.tokens for feature in features]
            self.outputdata=[feature.output for feature in features]
            self.token_mask=[feature.token_mask for feature in features]
            self.tobeselected=[feature.tobeselected for feature in features]
            self.selected=[feature.selected for feature in features]
            self.positives=[feature.positives for feature in features]
            self.negatives=[feature.negatives for feature in features]
        else:
            # arrange data according to text length to accelerate the evaluation process
            inputdata = [feature.tokens for feature in features]
            outputdata = [feature.output for feature in features]
            token_mask = [feature.token_mask for feature in features]
            tobeselected = [feature.tobeselected for feature in features]
            selected = [feature.selected for feature in features]
            positives = [feature.positives for feature in features]
            negatives = [feature.negatives for feature in features]
            maxlen,minlen=0,1000
            d={}
            for i in range(len(inputdata)):
                length=len(inputdata[i])
                if length>maxlen:
                    maxlen=length
                if length<minlen:
                    minlen=length
                if length not in d.keys():
                    d[length]=[i]
                else:
                    d[length].append(i)
            self.inputdata,self.outputdata,self.token_mask,self.tobeselected,self.selected,self.positives,\
                                                                                self.negatives=[],[],[],[],[],[],[]
            for i in range(minlen,maxlen+1):
                if i in d.keys():
                    l=d[i]
                    for num in l:
                        self.inputdata.append(inputdata[num])
                        self.outputdata.append(outputdata[num])
                        self.token_mask.append(token_mask[num])
                        self.tobeselected.append(tobeselected[num])
                        self.selected.append(selected[num])
                        self.positives.append(positives[num])
                        self.negatives.append(negatives[num])

    def __len__(self):
        return len(self.inputdata)

    def __getitem__(self, item):
        return {'input':torch.LongTensor(self.inputdata[item]),'token_mask':torch.LongTensor(self.token_mask[item]),
                'output':torch.LongTensor(self.outputdata[item]),'tobeselected':self.tobeselected[item],
                'selected':self.selected[item],'positives':torch.LongTensor(self.positives[item]),
                'negatives':self.negatives[item]}

class selector_collate_fn:
    def __init__(self, tokenizer):
        self.pad_ids = tokenizer.pad_token_id

    def __call__(self, items):
        all_sents = pad_sequence([x['input'] for x in items], batch_first=True, padding_value=self.pad_ids)
        all_token_labels=pad_sequence([x['output'].squeeze() for x in items], batch_first=True,padding_value=0)
        all_token_mask=pad_sequence([x['token_mask'].squeeze() for x in items], batch_first=True,padding_value=0)
        attn_mask= (all_sents != self.pad_ids).long()
        selected = [x['selected'] for x in items]
        tobeselected=[x['tobeselected'] for x in items]
        positives = [x['positives'] for x in items]
        negatives = [x['negatives'] for x in items]
        return (all_sents,all_token_labels,all_token_mask,attn_mask,selected,tobeselected,positives,negatives)

def make_output_dir(args):
    os.makedirs(args.output_dir, exist_ok=True)
    tb_dir = os.path.join(args.output_dir, 'log')
    shutil.rmtree(tb_dir, ignore_errors=True)

def save(args, model, tokenizer, model_num):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_dir = os.path.join(args.output_dir, "pytorch_model.bin"+str(model_num))
    torch.save(model_to_save.state_dict(), output_model_dir)

    if model_num==0:
        tokenizer.save_pretrained(args.output_dir)
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)

def evaluate(args, model, eval_dataloader, device):
    model.eval()
    eval_loss, nb_eval_steps, nb_eval_examples, eval_accuracy = 0, 0, 0, 0
    top1_acc, top1_valid, invalid_num, total_num, top2_num, top2_acc, top2_valid = 0, 0, 0, 0, 0, 0, 0
    if device!='rank':
        torch.cuda.set_device(0)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if device=='rank':
            batch = tuple(t.cuda() if type(t) != list else t for t in batch)
        else:
            batch = tuple(t.to(device) if type(t) != list else t for t in batch)
        input_ids, output_ids, token_mask, attn_mask, selected, tobeselected, positives, negatives = batch

        with torch.no_grad():
            loss,output,probs,num1,num2 = model(input_ids, attn_mask, tobeselected, token_mask, selected, positives, negatives, output_ids,
                           return_loss=True,return_selected=True,alpha=args.alpha,top_n=2)
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
            invalid_num += num1
            total_num += num2
            for i in range(len(output)):
                if len(output[i])!=0 and output_ids[i][tobeselected[i][output[i][0]]].data==1:
                    top1_acc += 1
                else:
                    if len(output[i])!=0 and tobeselected[i][output[i][0]] in positives[i]:
                        top1_valid += 1
                    nb_eval_examples += 1

                if len(output[i])>1:
                    top2_num+=1
                    if output_ids[i][tobeselected[i][output[i][1]]].data == 1:
                        top2_acc += 1
                    if tobeselected[i][output[i][1]] in positives[i]:
                        top2_valid += 1

    eval_loss = eval_loss / nb_eval_steps
    top1_valid = float(top1_acc + top1_valid) / (top1_acc + nb_eval_examples)
    if top2_num!=0:
        top2_valid = float(top2_valid) / top2_num
    else:
        top2_valid=0
    top1_acc = float(top1_acc) / (nb_eval_examples + top1_acc)
    logger.info(f'top1_acc:{top1_acc} top1_valid:{top1_valid} top2_valid:{top2_valid} eval_loss:{eval_loss} invalid/total:{invalid_num}/{total_num}')
    return top1_acc

def test(args, model, test_dataloader, device,tokenizer):
    model.eval()
    nb_eval_examples, top1_valid, top2_valid, top1_acc, top2_acc, top2_num = 0, 0, 0, 0, 0, 0
    logger.info(f"writing scores to file {os.path.join(args.output_dir, 'test_result_recording.txt')}")
    invalid_num,total_num=0,0
    for batch in tqdm(test_dataloader, desc="Inference"):
        batch = tuple(t.to(device) if type(t) != list else t for t in batch)
        input_ids, output_ids, token_mask, attn_mask, selected, tobeselected, positives, negatives = batch
        with torch.no_grad():
            output,probs,num1,num2 = model(input_ids, attn_mask, tobeselected, token_mask, selected, positives, negatives, output_ids,
                           return_loss=False,alpha=args.alpha,top_n=2)
            invalid_num+=num1
            total_num+=num2
            for i in range(len(output)):
                if len(output[i])!=0 and output_ids[i][tobeselected[i][output[i][0]]].data==1:
                    top1_acc += 1
                else:
                    if len(output[i])!=0 and tobeselected[i][output[i][0]] in positives[i]:
                        top1_valid+=1
                    nb_eval_examples += 1

                if len(output[i])>1:
                    top2_num+=1
                    if output_ids[i][tobeselected[i][output[i][1]]].data == 1:
                        top2_acc += 1
                    if tobeselected[i][output[i][1]] in positives[i]:
                        top2_valid += 1

    top1_valid = float(top1_acc+top1_valid) / (top1_acc+nb_eval_examples)
    logger.info(f'top1_valid: {top1_valid}')

    top2_valid = float(top2_valid) / (top2_num)
    logger.info(f'top2_valid: {top2_valid}')

    top2_acc = float(top2_acc + top1_acc) / (nb_eval_examples + top1_acc)
    logger.info(f'top2_acc: {top2_acc}')

    top1_acc = float(top1_acc) / (nb_eval_examples + top1_acc)
    logger.info(f'top1_acc: {top1_acc}')

    logger.info(f'invalid_num/total_num: {invalid_num}/{total_num}')
    invalid_ratio = float(invalid_num) / total_num
    logger.info(f'invalid_ratio: {invalid_ratio}')

    logger.info('saving predictions.txt in ' + args.output_dir)
    with open(args.output_dir + '/test_result_recording.txt', 'w+') as writer:
        writer.write(f'use the model saved in {args.model}\n')
        writer.write(f'total_correct: {top1_acc}\n')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12403'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def run(rank, world_size,args):
    setup(rank, world_size)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(rank)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    tokenizer = XLNetTokenizer.from_pretrained(args.model, do_lower_case=True)
    if args.init_weights_dir:
        model_state_dict = torch.load(args.init_weights_dir,map_location='cpu')
        model_state_dict = {k: v for k, v in model_state_dict.items() if "_te" not in k}
        model = Selector2.from_pretrained(args.model, state_dict=model_state_dict)
        del model_state_dict
    else:
        model = Selector2.from_pretrained(args.model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],find_unused_parameters=True)

    collator = selector_collate_fn(tokenizer)
    eval_data = Selector2Dataset(args, 'dev')
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,collate_fn=collator)
    if rank==0:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_data))
        logger.info("  Batch size = %d", args.eval_batch_size)

    if args.do_train:
        train_data = Selector2Dataset(args, 'train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        train_dataloader=DataLoader(train_data,sampler=train_sampler,batch_size=args.train_batch_size,collate_fn=collator)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        scaler = GradScaler()
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        if rank==0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        global_step, best_acc, do_eval, model_num, best_num = 0, 0, False, 0, 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            train_sampler.set_epoch(epoch)
            train_dataloader.sampler.set_epoch(epoch)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.cuda() if type(t) != list else t for t in batch)
                input_ids, output_ids, token_mask, attn_mask, selected, tobeselected, positives, negatives = batch

                my_context = model.no_sync if (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
                with autocast():
                    with my_context():
                        losses=model(input_ids,attn_mask,tobeselected,token_mask,selected,positives,negatives,output_ids,alpha=args.alpha)
                        take_mean = lambda x: x.mean() if x is not None and sum(x.size()) > 1 else x
                        [loss] = list(map(take_mean, [losses]))
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        scaler.scale(loss).backward()

                if (step+1)%(len(train_data)//(args.train_batch_size*world_size*4)+2)==0:
                    do_eval=True
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                if do_eval:
                    do_eval = False
                    torch.distributed.barrier()
                    top1_acc = distributed_evaluate(args, model, eval_dataloader,rank=rank)
                    if top1_acc > best_acc:
                        best_num = epoch
                    best_acc = max(best_acc, top1_acc)
                    if rank==0:
                        save(args, model, tokenizer, model_num)
                    model_num += 1
                    model.train()
                    torch.distributed.barrier()
            do_eval=True

        top1_acc = distributed_evaluate(args, model, eval_dataloader, rank=rank)
        if top1_acc > best_acc:
            best_num = epoch
        best_acc = max(best_acc, top1_acc)
        if rank == 0:
            save(args, model, tokenizer, model_num)
            logger.info("top1_acc = %s", best_acc)
            logger.info("best_epoch = %s", best_num)

def distributed_evaluate(args, model, eval_dataloader, rank=-1):
    model.eval()
    eval_loss, eval_err_sum = 0, 0
    nb_eval_steps, nb_eval_examples,eval_accuracy,top1_acc,top1_valid,invalid_num,total_num,top2_num,top2_acc,top2_valid = 0,0,0,0,0,0,0,0,0,0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.cuda() if type(t) != list else t for t in batch)
        input_ids, output_ids, token_mask, attn_mask, selected, tobeselected, positives, negatives = batch

        with torch.no_grad():
            loss,output,probs,num1,num2 = model(input_ids, attn_mask, tobeselected, token_mask, selected, positives, negatives, output_ids,
                           return_loss=True,return_selected=True,alpha=args.alpha,top_n=2)
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
            invalid_num += num1
            total_num += num2
            for i in range(len(output)):
                if len(output[i])!=0 and output_ids[i][tobeselected[i][output[i][0]]].data==1:
                    top1_acc += 1
                else:
                    if len(output[i])!=0 and tobeselected[i][output[i][0]] in positives[i]:
                        top1_valid += 1
                    nb_eval_examples += 1

                if len(output[i])>1:
                    top2_num+=1
                    if output_ids[i][tobeselected[i][output[i][1]]].data == 1:
                        top2_acc += 1
                    if tobeselected[i][output[i][1]] in positives[i]:
                        top2_valid += 1

    def sum(num):
        num = torch.tensor(num).cuda()
        dist.all_reduce(num, op=dist.ReduceOp.SUM)
        return num.item()

    # Integrate the results of distributed evaluation on multiple gpus
    eval_loss = sum(eval_loss)
    nb_eval_steps = sum(nb_eval_steps)
    top1_acc = sum(top1_acc)
    top1_valid = sum(top1_valid)
    nb_eval_examples = sum(nb_eval_examples)
    top2_valid = sum(top2_valid)
    top2_num = sum(top2_num)
    invalid_num = sum(invalid_num)
    total_num = sum(total_num)

    eval_loss = eval_loss / nb_eval_steps
    top1_valid = float(top1_acc + top1_valid) / (top1_acc + nb_eval_examples)
    if top2_num!=0:
        top2_valid = float(top2_valid) / top2_num
    else:
        top2_valid=0
    top1_acc = float(top1_acc) / (nb_eval_examples + top1_acc)
    if rank==0:
        logger.info(f'top1_acc:{top1_acc} top1_valid:{top1_valid} top2_valid:{top2_valid} eval_loss:{eval_loss} invalid/total:{invalid_num}/{total_num}')
    return top1_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        default='data/3sat/relative_clause/rclause_16_20_21',
                        type=str,
                        help="Dir containing drop examples and features.")
    parser.add_argument("--output_dir",
                        default='out_selector/rclause_16_20_21',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model", default="xlnet-large-cased", type=str)
    parser.add_argument("--init_weights_dir",
                        default='',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run inference on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The initial weight decay for Adam.")
    parser.add_argument('--eps',
                        default=1e-6,
                        type=float,
                        help="Whether to randomly shift position ids of encoder input.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--freeze_encoder',
                        action='store_true',
                        help="Whether to freeze the bert encoder, embeddings.")
    parser.add_argument('--random_shift',
                        action='store_true',
                        help="Whether to randomly shift position ids of encoder input.")
    parser.add_argument('--alpha',
                        type=float,
                        default=0.001,
                        help="scale for contrastive loss")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        world_size = n_gpu
        logger.info("n_gpu: {}, distributed training: {}".format(n_gpu, bool(args.local_rank != -1)))
        args.train_batch_size = args.train_batch_size // world_size
        logger.info('**************** distributed training ******************')
        # distributed training
        mp.spawn(run, args=(world_size, args,), nprocs=world_size)
        exit()

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if args.do_train:
        make_output_dir(args)

    tokenizer = XLNetTokenizer.from_pretrained(args.model, do_lower_case=True)

    if "bin" in args.init_weights_dir:
        logger.info(args.init_weights_dir)
        model_state_dict = torch.load(args.init_weights_dir, map_location=device)
        model = Selector2.from_pretrained(args.model,state_dict=model_state_dict)
        del model_state_dict
    else:
        model = Selector2.from_pretrained(args.model)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model,device_ids=[i for i in range(n_gpu)])

    if args.do_test:
        collator = selector_collate_fn(tokenizer)
        test_data = Selector2Dataset(args, 'test')
        logger.info("***** Running Testing *****")
        logger.info("  Num examples = %d", len(test_data))
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data,sampler=test_sampler,batch_size=args.eval_batch_size,collate_fn=collator)
        test(args, model, test_dataloader, device,tokenizer)
        exit()

    collator = selector_collate_fn(tokenizer)
    eval_data = Selector2Dataset(args, 'dev')
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collator)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.do_train:
        train_data = Selector2Dataset(args, 'train')
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collator)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        scaler = GradScaler()
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        global_step, best_acc, do_eval, model_num, best_num = 0, 0, False, 0, 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) if type(t)!=list else t for t in batch)
                input_ids,output_ids,token_mask,attn_mask,selected,tobeselected,positives,negatives=batch

                with autocast():
                    losses = model(input_ids,attn_mask,tobeselected,token_mask,selected,positives,negatives,output_ids,alpha=args.alpha)
                    take_mean = lambda x: x.mean() if x is not None and sum(x.size()) > 1 else x
                    [loss] = list(map(take_mean, [losses]))
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
                if (step+1) % (len(train_dataloader)//2+1) == 0:
                    do_eval = True
                if do_eval:
                    do_eval = False
                    top1_acc = evaluate(args, model, eval_dataloader, device)
                    save(args, model, tokenizer, model_num)
                    model_num += 1
                    if top1_acc > best_acc:
                        best_num = epoch
                    best_acc = max(best_acc, top1_acc)
                    model.train()
            do_eval = True
        top1_acc = evaluate(args, model, eval_dataloader, device)
        save(args, model, tokenizer, model_num)
        if top1_acc > best_acc:
            best_num = epoch
        best_acc = max(best_acc, top1_acc)
        logger.info("best_eval_acc = %s", best_acc)
        logger.info("best_epoch = %s", best_num)
