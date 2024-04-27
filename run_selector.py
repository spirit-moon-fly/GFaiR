from __future__ import absolute_import
import argparse
import logging
import os, random, shutil
from selector import Selector,BertAdam
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

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
logger = logging.getLogger(__name__)

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

class SelectorDataset(Dataset):

    def __init__(self, args, split='train'):
        model_size='base' if 'base' in args.model else 'large'
        input_direc = os.path.join(args.input_dir,split+'_input_selector'+model_size+'.pkl')
        output_direc = os.path.join(args.input_dir,split + '_output_selector'+model_size+'.pkl')
        tokenmask_direc = os.path.join(args.input_dir,split + '_tokenmask_selector'+model_size+'.pkl')
        if split=='train':
            self.inputdata=read_file(input_direc)
            self.outputdata=read_file(output_direc)
            self.token_mask = read_file(tokenmask_direc)
        else:
            # arrange data according to text length to accelerate the evaluation process
            inputdata = read_file(input_direc)
            outputdata = read_file(output_direc)
            token_mask = read_file(tokenmask_direc)
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
            self.inputdata,self.outputdata,self.token_mask=[],[],[]
            for i in range(minlen,maxlen+1):
                if i in d.keys():
                    l=d[i]
                    for num in l:
                        self.inputdata.append(inputdata[num])
                        self.outputdata.append(outputdata[num])
                        self.token_mask.append(token_mask[num])

    def __len__(self):
        return len(self.inputdata)

    def __getitem__(self, item):
        return {'input':torch.LongTensor(self.inputdata[item]),'token_mask':torch.LongTensor(self.token_mask[item]),
                'output':torch.LongTensor(self.outputdata[item])}

class selector_collate_fn:
    def __init__(self, tokenizer,is_train):
        self.tokenizer = tokenizer
        self.pad_ids = tokenizer.pad_token_id
        self.is_train=is_train

    def __call__(self, items):
        all_sents = pad_sequence([x['input'] for x in items], batch_first=True, padding_value=self.pad_ids)
        all_token_labels=pad_sequence([x['output'].squeeze() for x in items], batch_first=True,padding_value=0)
        if self.is_train is False:
            all_token_mask=[x['token_mask'].squeeze() for x in items]
        else:
            all_token_mask=pad_sequence([x['token_mask'].squeeze() for x in items], batch_first=True,padding_value=0)
        attn_mask= (all_sents != self.pad_ids).long()

        return (all_sents,all_token_labels,all_token_mask,attn_mask)

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

def evaluate(args, model, eval_dataloader, device, n_train, rank=-1):
    model.eval()
    nb_eval_examples, eval_accuracy = 0, 0
    if device!='rank':
        torch.cuda.set_device(0)
    for batch in tqdm(eval_dataloader, desc="Inference"):
        batch = tuple(t.to(device) if type(t) != list else [tensor.to(device) for tensor in t] for t in batch)
        input_ids, output_ids, token_mask, attn_mask = batch
        with torch.no_grad():
            output = model(input_ids, attn_mask, token_mask, output_ids, return_loss=False)
            for i in range(len(output)):
                if output[i][0] == output_ids[i][1] or output[i][0] == output_ids[i][0]:
                    eval_accuracy += 1
                else:
                    nb_eval_examples += 1

    logger.info(f'onebest_correct/total_num: {eval_accuracy}/{nb_eval_examples + eval_accuracy}')
    onebest_acc = float(eval_accuracy) / (nb_eval_examples + eval_accuracy)
    logger.info(f'onebest_acc: {onebest_acc}')
    if rank <= 0:
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w+") as writer:
            logger.info("***** Eval results *****")
            writer.write("onebest_acc = %d\t" % onebest_acc)
            writer.write("n_train = %d\t" % n_train)
            writer.write("\n")
    return onebest_acc

def test(args, model, test_dataloader, device, tokenizer):
    model.eval()
    nb_eval_examples, eval_accuracy = 0,0
    logger.info(f"writing scores to file {os.path.join(args.output_dir, 'test_result_recording.txt')}")
    for batch in tqdm(test_dataloader, desc="Inference"):
        batch = tuple(t.to(device) if type(t)!=list else [tensor.to(device) for tensor in t] for t in batch)
        input_ids, output_ids, token_mask, attn_mask = batch
        with torch.no_grad():
            output = model(input_ids, attn_mask, token_mask, output_ids,return_loss=False)
            for i in range(len(output)):
                if output[i][0] == output_ids[i][1] or output[i][0] == output_ids[i][0]:
                    eval_accuracy+=1
                else:
                    nb_eval_examples += 1

    logger.info(f'onebest_correct/total_num: {eval_accuracy}/{nb_eval_examples + eval_accuracy}')
    onebest_acc = float(eval_accuracy) / (nb_eval_examples + eval_accuracy)
    logger.info(f'onebest_acc: {onebest_acc}')

    logger.info('saving predictions.txt in ' + args.output_dir)
    with open(args.output_dir + '/test_result_recording.txt', 'w+') as writer:
        writer.write(f'use the model saved in {args.model}\n')
        writer.write(f'total_correct: {eval_accuracy}\n')

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
        model = Selector.from_pretrained(args.model, state_dict=model_state_dict)
        del model_state_dict
    else:
        model = Selector.from_pretrained(args.model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    collator = selector_collate_fn(tokenizer, is_train=False)
    eval_data = SelectorDataset(args, 'dev')
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,collate_fn=collator)
    if rank==0:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_data))
        logger.info("  Batch size = %d", args.eval_batch_size)

    if args.do_train:
        collator1 = selector_collate_fn(tokenizer, is_train=True)
        train_data = SelectorDataset(args, 'train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        train_dataloader=DataLoader(train_data,sampler=train_sampler,batch_size=args.train_batch_size,collate_fn=collator1)

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
        global_step, best_acc, do_eval = 0, 0, False
        model_num= 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            train_sampler.set_epoch(epoch)
            train_dataloader.sampler.set_epoch(epoch)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.cuda() for t in batch)
                input_ids, output_ids, token_mask, attn_mask = batch

                with autocast():
                    losses = model(input_ids, attn_mask, token_mask, output_ids)
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
                if do_eval:
                    do_eval = False
                    eval_acc = evaluate(args, model, eval_dataloader, 'rank', len(train_data),rank=rank)
                    if eval_acc > best_acc:
                        if rank == 0:
                            save(args, model, tokenizer, model_num)
                        model_num += 1
                    best_acc = max(best_acc, eval_acc)
                    model.train()
            do_eval=True
        if rank==0:
            logger.info("best_eval_acc = %s", best_acc)

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
    parser.add_argument("--max_seq_length",
                        default=-1,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
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
                        default=64,
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
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
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

    if args.init_weights_dir:
        logger.info(args.init_weights_dir)
        model_state_dict = torch.load(args.init_weights_dir, map_location=device)
        model = Selector.from_pretrained(args.model,state_dict=model_state_dict)
        del model_state_dict
    else:
        model = Selector.from_pretrained(args.model)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_test:
        collator = selector_collate_fn(tokenizer,is_train=False)
        test_data = SelectorDataset(args, 'test')
        logger.info("***** Running Testing *****")
        logger.info("  Num examples = %d", len(test_data))
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data,sampler=test_sampler,batch_size=args.eval_batch_size,collate_fn=collator)
        test(args, model, test_dataloader, device, tokenizer)
        exit()

    collator = selector_collate_fn(tokenizer,is_train=False)
    eval_data = SelectorDataset(args, 'dev')
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collator)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.do_train:
        collator1 = selector_collate_fn(tokenizer, is_train=True)
        train_data = SelectorDataset(args, 'train')
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collator1)

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
        global_step, best_acc,  do_eval = 0, 0.0, False
        model_num, best_num = 0,0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids,output_ids,token_mask,attn_mask=batch

                with autocast():
                    losses = model(input_ids,attn_mask,token_mask,output_ids)
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
                if do_eval:
                    do_eval = False
                    eval_acc = evaluate(args, model, eval_dataloader, device, len(train_data))
                    save(args, model, tokenizer, model_num)
                    model_num += 1
                    if eval_acc > best_acc:
                        best_num = epoch
                    best_acc = max(best_acc, eval_acc)
                    model.train()
            do_eval = True

        eval_acc = evaluate(args, model, eval_dataloader, device, len(train_data))
        save(args, model, tokenizer, model_num)
        if eval_acc > best_acc:
            best_num = epoch
        best_acc = max(best_acc, eval_acc)
        logger.info("best_eval_acc = %s", best_acc)
        logger.info("best_epoch = %s", best_num)

