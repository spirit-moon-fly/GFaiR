from __future__ import absolute_import
import argparse
import logging
import os, random, shutil
from convertor import Convertor
from io import open
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,Dataset)
import jsonlines,json,pickle
from transformers import Adafactor
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer

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

class ConvertorDataset(Dataset):

    def __init__(self, args,split='train'):
        input_direc = os.path.join(args.input_dir,split+'_input_convertor.pkl')
        output_direc = os.path.join(args.input_dir,split + '_output_convertor.pkl')
        self.inputdata=read_file(input_direc)
        self.outputdata=read_file(output_direc)
        self.num_samples=len(self.inputdata)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return {'input':torch.LongTensor(self.inputdata[item]),'output':torch.LongTensor(self.outputdata[item])}

class convertor_collate_fn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_ids = tokenizer.pad_token_id

    def __call__(self, batch):
        x_list = [dic['input'] for dic in batch]
        y_list = [dic['output'] for dic in batch]
        x_padded = pad_sequence(x_list, batch_first=True, padding_value=self.pad_ids)
        y_padded = pad_sequence(y_list, batch_first=True, padding_value=self.pad_ids)
        y_ids = y_padded[:, :-1].contiguous()
        labels = y_padded[:, 1:].clone()
        labels[y_padded[:, 1:] == self.pad_ids] = -100
        return (x_padded,y_padded,(x_padded != self.pad_ids).long(),y_ids,labels)

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

def evaluate(args, model, eval_dataloader, device, n_train, tokenizers):
    model.eval()
    nb_eval_steps, nb_eval_examples,eval_accuracy,eval_loss = 0, 0, 0, 0
    torch.cuda.set_device(0)
    for batch in tqdm(eval_dataloader, desc="Inference"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, output_ids, attn_mask, y_ids, labels = batch
        with torch.no_grad():
            loss = model(input_ids,attn_mask,y_ids,labels).loss
            eval_loss += loss.mean().item()
            nb_eval_steps += 1

        with torch.no_grad():
            output = model.predict_and_decode(input_ids)
            for i in range(output.shape[0]):
                if isequal(output_ids[i], output[i], tokenizers):
                    eval_accuracy += 1
                else:
                    nb_eval_examples += 1

    eval_accuracy = float(eval_accuracy) / (nb_eval_examples + eval_accuracy)
    eval_loss = eval_loss / nb_eval_steps
    result = {'eval_loss': eval_loss, 'eval_acc': eval_accuracy}
    logger.info("***** Eval results *****")
    logger.info(f'eval_loss:{eval_loss} eval_acc:{eval_accuracy}')

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\t" % (key, str(result[key])))
        writer.write("n_train = %d\t" % n_train)
        writer.write("\n")

    return result

def isequal(l1,l2,tokenizer):
    for i in range(len(l1)):
        if l1[i]!=tokenizer.pad_token_id:
            if len(l2)<i+1 or l1[i]!=l2[i]:
                return False
        elif len(l2)>=i+1 and l2[i]!=tokenizer.pad_token_id:
            return False
    return True

def test(args, model, test_dataloader, device,tokenizer):
    model.eval()
    nb_eval_examples, eval_accuracy = 0,0
    logger.info(f"writing scores to file {os.path.join(args.output_dir, 'test_result_recording.txt')}")
    for batch in tqdm(test_dataloader, desc="Inference"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, output_ids, attn_mask, y_ids, labels = batch
        with torch.no_grad():
            output = model.predict_and_decode(input_ids)
            for i in range(output.shape[0]):
                if isequal(output[i],output_ids[i],tokenizer):
                    eval_accuracy += 1
                else:
                    nb_eval_examples += 1

    logger.info(f'total_correct/total_num: {eval_accuracy}/{nb_eval_examples+eval_accuracy}')
    eval_accuracy = float(eval_accuracy) / (nb_eval_examples+eval_accuracy)
    logger.info(f'accuracy: {eval_accuracy}')
    logger.info('saving predictions.txt in ' + args.output_dir)
    with open(args.output_dir + '/test_result_recording.txt', 'a+') as writer:
        writer.write(f'use the model saved in {args.model}\n')
        writer.write(f'total_correct/total_num: {eval_accuracy}/{nb_eval_examples}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        default='data/3sat/relative_clause/rclause_16_20_21',
                        type=str,
                        help="Dir containing drop examples and features.")
    parser.add_argument("--output_dir",
                        default='out_convertor/rclause_16_20_21',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model", default="t5-large", type=str)
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
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
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
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--random_shift',
                        action='store_true',
                        help="Whether to randomly shift position ids of encoder input.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if args.do_train:
        make_output_dir(args)

    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=True)

    if args.init_weights_dir:
        logger.info(args.init_weights_dir)
        model_state_dict = torch.load(args.init_weights_dir, map_location=device)
        model = Convertor(args.model)
        model.load_state_dict(model_state_dict)
        del model_state_dict
    else:
        model = Convertor(args.model)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    collator = convertor_collate_fn(tokenizer)

    if args.do_test:
        test_data = ConvertorDataset(args, 'test')
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data,sampler=test_sampler,batch_size=args.eval_batch_size,collate_fn=collator)
        test(args, model, test_dataloader, device, tokenizer)
        exit()

    eval_data = ConvertorDataset(args, 'dev')
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collator)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.do_train:
        train_data = ConvertorDataset(args, 'train')
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collator)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        optimizer=Adafactor(model.parameters(),lr=args.learning_rate,eps=(1e-30, 1e-3),clip_threshold=1.0,decay_rate=
                        -0.8,beta1=None,weight_decay=0.0,relative_step=False,scale_parameter=False,warmup_init=False)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        model.train()
        global_step, best, do_eval = 0, 1000, False
        model_num, best_num = 0,0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids,output_ids,attn_mask,y_ids,labels=batch

                losses = model(input_ids,attn_mask,y_ids,labels).loss
                take_mean = lambda x: x.mean() if x is not None and sum(x.size()) > 1 else x
                [loss] = list(map(take_mean, [losses]))

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if do_eval:
                    do_eval = False
                    eval_result = evaluate(args, model, eval_dataloader, device, len(train_data),tokenizer)
                    eval_loss = eval_result['eval_loss']
                    if eval_loss < best:
                        save(args, model, tokenizer, model_num)
                        model_num += 1
                        best_num = epoch
                    best = min(best, eval_loss)
                    model.train()
            do_eval = True

        logger.info("best_eval_loss = %s", best)
        logger.info("best_epoch = %s", best_num)

if __name__ == "__main__":
    main()

