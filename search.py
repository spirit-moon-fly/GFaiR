from __future__ import absolute_import
import argparse
import logging
import os, random
from selector import Selector,Selector2
from convertor import Convertor
from io import open
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, Dataset)
import jsonlines,json,pickle
from transformers import XLNetTokenizer,T5Tokenizer
from torch.nn.utils.rnn import pad_sequence


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
logger = logging.getLogger(__name__)

def read_file(filedirec):
    with open(filedirec+'/test.jsonl') as f:
        data=f.readlines()
    return data

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

def convertor_tokenize_batch(tokenizer,contexts,device):
    ruleid2contextid={}
    num=0
    input=[]
    for i in range(len(contexts)):
        context=contexts[i]
        for s in context:
            input.append(s + tokenizer.eos_token)
            ruleid2contextid[num]=i
            num+=1
    tokenized = tokenizer(input, padding=True, return_tensors='pt',add_special_tokens=False)
    input_ids = tokenized['input_ids']
    return input_ids.to(device),ruleid2contextid

def selector_tokenize_batch(tokenizer,contexts,device):
    input_tokens =[torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(
                   tokenizer.cls_token+tokenizer.sep_token.join(context)+tokenizer.sep_token))) for context in contexts]
    input_ids = pad_sequence(input_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = (input_ids != tokenizer.pad_token_id).long()
    token_mask = []
    for tokens in input_ids:
        mask = []
        for i in range(len(tokens)):
            if tokens[i] == tokenizer.sep_token_id:
                mask.append(i)
        token_mask.append(torch.LongTensor(mask).to(device))
    return input_ids.to(device), attn_mask.to(device), token_mask

def selector2_tokenize_batch(tokenizer,contexts, output, device):
    input_tokens=[]
    for i in range(len(contexts)):
        s=tokenizer.cls_token+contexts[i][output[i]]+tokenizer.sep_token
        for j in range(len(contexts[i])):
            if j!=output[i]:
                s+=contexts[i][j]+tokenizer.sep_token
        input_tokens.append(torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))))
    input_ids = pad_sequence(input_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = (input_ids != tokenizer.pad_token_id).long()
    tobeselected, selected = [],[]
    for tokens in input_ids:
        mask = []
        for i in range(len(tokens)):
            if tokens[i] == tokenizer.sep_token_id:
                mask.append(i)
        selected.append(mask[0])
        tobeselected.append(torch.LongTensor(mask[1:]).to(device))
    return input_ids.to(device), attn_mask.to(device), tobeselected, torch.LongTensor(selected).to(device)

def reasoner_tokenize_batch(tokenizer,rule1s,rule2s,device):
    input_tokens = [rule1+rule2 + tokenizer.eos_token for rule1,rule2 in zip(rule1s,rule2s)]
    tokenized = tokenizer(input_tokens, padding=True, return_tensors='pt',add_special_tokens=False)
    input_ids = tokenized['input_ids']
    return input_ids.to(device)

class InferenceDataset(Dataset):

    def __init__(self, args):
        datas=read_file(args.input_dir)
        self.datas={'context':[],'answer':[]}
        for data in datas:
            data = json.loads(data)
            context = data['context'].split(". ")
            for i in range(len(context)):
                context[i] = context[i] + '.'
            self.datas['context'].append(context)
            self.datas['answer'].append(data["answer"])

    def __len__(self):
        return len(self.datas['answer'])

    def __getitem__(self, item):
        return {'context':self.datas['context'][item], 'answer':self.datas['answer'][item]}

class Inference_collate_fn:
    def __init__(self):
        pass
    def __call__(self, items):
        all_context = [x['context'] for x in items]
        all_answer = [x['answer'] for x in items]
        return (all_context,all_answer)

def get_topn(probs,candidates_num,beam_size):
    ret=[]
    probs,indexes=torch.sort(torch.tensor(probs),descending=True)
    for i in range(min(beam_size,len(indexes))):
        if probs[i]<-900000:
            break
        ret.append((indexes[i].data//candidates_num,indexes[i].data%candidates_num))
    return ret,probs[:len(ret)].numpy().tolist()

def beam_search1(output, probs, current_probs, selected, contexts,rule1s,beam_size,rule1s_num,candidates_num,omit_selected_index):
    p,omits,selected_ret,omit_selected_index_ret=[],[0 for _ in range(beam_size)],[],[]
    for i in range(len(output)):
        for j in range(len(output[i])):
            if [rule1s_num[i],output[i][j] if output[i][j]<rule1s_num[i] else output[i][j]+1] in selected[i] or [output[i][j] if output[i][j]<rule1s_num[i] else output[i][j]+1,rule1s_num[i]] in selected[i]:
                p.append(-1000000)
            else:
                p.append(current_probs[i]+probs[i][j])
        for j in range(len(output[i]),candidates_num):
            p.append(-1000000)

    # select the top candidates_num possible reasoning directions
    indexes,p_now=get_topn(p,candidates_num,beam_size)
    rule1s_ret,rule2s_ret,current_probs_ret,contexts_ret=[],[],[],[]
    assert len(indexes)<=beam_size
    for i in range(len(indexes)):
        rule1s_ret.append(rule1s[indexes[i][0]])
        rule2s_ret.append(contexts[indexes[i][0]][output[indexes[i][0]][indexes[i][1]] if output[indexes[i][0]][indexes[i][1]]<rule1s_num[indexes[i][0]] else output[indexes[i][0]][indexes[i][1]]+1])
        contexts_ret.append(contexts[indexes[i][0]])
        current_probs_ret.append(p_now[i])
        l=selected[indexes[i][0]][:]
        l.append([rule1s_num[indexes[i][0]],output[indexes[i][0]][indexes[i][1]] if output[indexes[i][0]][indexes[i][1]]<rule1s_num[indexes[i][0]] else output[indexes[i][0]][indexes[i][1]]+1])
        selected_ret.append(l)
        omit_selected_index_ret.append(omit_selected_index[indexes[i][0]][:])
    for i in range(len(indexes),beam_size):
        omits[i]=1
        rule1s_ret.append(rule1s[i])
        rule2s_ret.append(contexts[i][0])
        contexts_ret.append(contexts[i][:])
        current_probs_ret.append(-1000000)
        selected_ret.append([0])
        omit_selected_index_ret.append([])
    return rule1s_ret,rule2s_ret,contexts_ret,current_probs_ret,omits,selected_ret,omit_selected_index_ret

def beam_search(output,probs,current_probs,selected,contexts,batch_size,beam_size,rule1s,rule1s_num,candidates_num,omit_selected_index):
    rule1s_ret,rule2s_ret,contexts_ret,current_probs_ret,omits_ret,selected_ret,omit_selected_index_ret=[],[],[],[],[],[],[]
    for i in range(batch_size):
        rule1s_tmp,rule2s_tmp,contexts_tmp,current_probs_tmp,omits_tmp,selected_tmp,omit_selected_index_tmp=beam_search1(output[i*beam_size:(i+1)*beam_size],
            probs[i*beam_size:(i+1)*beam_size],current_probs[i*beam_size:(i+1)*beam_size],selected[i*beam_size:(i+1)*beam_size],contexts[i*beam_size:(i+1)*beam_size],
            rule1s[i*beam_size:(i+1)*beam_size],beam_size,rule1s_num[i*beam_size:(i+1)*beam_size],candidates_num,omit_selected_index[i*beam_size:(i+1)*beam_size])
        for j in range(beam_size):
            rule1s_ret.append(rule1s_tmp[j])
            rule2s_ret.append(rule2s_tmp[j])
            contexts_ret.append(contexts_tmp[j][:])
            current_probs_ret.append(current_probs_tmp[j])
            selected_ret.append(selected_tmp[j][:])
            omit_selected_index_ret.append(omit_selected_index_tmp[j][:])
        omits_ret.append(omits_tmp)

    return rule1s_ret,rule2s_ret,contexts_ret,current_probs_ret,omits_ret,selected_ret,omit_selected_index_ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        default='data/3sat/relative_clause/rclause_16_20_21',
                        type=str,
                        help="Dir containing drop examples and features.")
    parser.add_argument("--output_dir",
                        default='out_inference/rclause_16_20_21',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--convertor_init_weights_dir",
                        default='out_convertor/rclause_16_20_21/pytorch_model.bin0',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")
    parser.add_argument("--selector_init_weights_dir",
                        default='out_selector/rclause_16_20_21/pytorch_model.bin7',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")
    parser.add_argument("--selector2_init_weights_dir",
                        default='out_selector2/rclause_16_20_21/pytorch_model.bin3',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")
    parser.add_argument("--reasoner_init_weights_dir",
                        default='out_reasoner/rclause_16_20_21/pytorch_model.bin7',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")
    parser.add_argument("--inference_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--beam_size',
                        type=int,
                        default=2,
                        help="random seed for initialization")
    parser.add_argument('--candidates_num',
                        type=int,
                        default=2,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    beam_size,candidates_num=args.beam_size,args.candidates_num
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    XLNET_tokenizer = XLNetTokenizer.from_pretrained('../../model/xlnet', do_lower_case=True)
    T5_Tokenizer= T5Tokenizer.from_pretrained('../../model/T5',do_lower_case=True)

    model_state_dict = torch.load(args.selector_init_weights_dir, map_location=device)
    selectormodel = Selector.from_pretrained('../../model/xlnet',state_dict=model_state_dict)

    model_state_dict = torch.load(args.selector2_init_weights_dir, map_location=device)
    selector2model = Selector2.from_pretrained('../../model/xlnet', state_dict=model_state_dict)

    model_state_dict = torch.load(args.convertor_init_weights_dir, map_location=device)
    convertormodel = Convertor('../../model/T5')
    convertormodel.load_state_dict(model_state_dict)

    model_state_dict = torch.load(args.reasoner_init_weights_dir, map_location=device)
    reasonermodel = Convertor('../../model/T5')
    reasonermodel.load_state_dict(model_state_dict)
    del model_state_dict

    selectormodel.to(device)
    selector2model.to(device)
    convertormodel.to(device)
    reasonermodel.to(device)
    if n_gpu > 1:
        reasonermodel = torch.nn.DataParallel(reasonermodel)
        selectormodel = torch.nn.DataParallel(selectormodel)
        selector2model = torch.nn.DataParallel(selectormodel)
        convertormodel = torch.nn.DataParallel(convertormodel)
    selectormodel.eval()
    selector2model.eval()
    reasonermodel.eval()
    convertormodel.eval()

    collator = Inference_collate_fn()
    test_data = InferenceDataset(args)
    test_epoch = len(test_data)//args.inference_batch_size
    reservenum = len(test_data) - test_epoch * args.inference_batch_size
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,sampler=test_sampler,batch_size=args.inference_batch_size,collate_fn=collator)

    eval_examples, eval_accuracy, proof_acc, proof_allnum = 0, 0, 0, 0
    # [predicted label, gold label]
    resultlist=[[0 for _ in range(3)] for _ in range(3)]
    d={'unknown':2,'yes':1,'no':0}
    if 'grounded_rule_lang' in args.input_dir:
        max_step={'5var':25,'8var':50,'10var':60,'12var':75}[args.input_dir.split('/')[-1]]
    else:
        max_step=50

    for step, batch in enumerate(tqdm(test_dataloader, desc="Inference")):
        orig_contexts,answers=batch
        if step == test_epoch:
            orig_contexts=orig_contexts[:reservenum]
            answers=answers[:reservenum]
            args.inference_batch_size=reservenum

        # convert
        input_ids,ruleid2contextid=convertor_tokenize_batch(T5_Tokenizer, orig_contexts, device)

        contexts=[[] for _ in range(len(orig_contexts))]
        with torch.no_grad():
            output = convertormodel.predict_and_decode(input_ids,is_inference=True)

        for i in range(len(output)):
            if output[i] not in contexts[ruleid2contextid[i]]:
                contexts[ruleid2contextid[i]].append(output[i])

        stop=False
        inf_time=0
        result=['unknown' for _ in range(len(orig_contexts))]
        current_probs=[0 for _ in range(args.inference_batch_size*beam_size)]
        selected=[[] for _ in range(args.inference_batch_size*beam_size)]
        omit_selected_index=[[] for _ in range(args.inference_batch_size*beam_size)]

        # select one theory in rule1s
        input_ids, attn_mask, token_mask = selector_tokenize_batch(XLNET_tokenizer, contexts, device)
        with torch.no_grad():
            output = selectormodel(input_ids, attn_mask, token_mask, return_loss=False, top_n=1)
            rule1s = []
            for i in range(len(output)):
                for _ in range(beam_size):
                    rule1s.append(contexts[i][output[i]])

        # select another theory in rule2s
        input_ids, attn_mask, tobeselected, selector1_selected = selector2_tokenize_batch(XLNET_tokenizer, contexts, output, device)

        with torch.no_grad():
            output2, probs = selector2model(input_ids, attn_mask, tobeselected, selected=selector1_selected, return_loss=False, top_n=beam_size)

        rule2s,omit=[],[[0 for _ in range(beam_size)] for _ in range(args.inference_batch_size)]
        for i in range(len(output2)):
            for j in range(len(output2[i])):
                rule2s.append(contexts[i][output2[i][j] if output2[i][j]<output[i] else output2[i][j]+1])
                current_probs[i*beam_size+j]=probs[i][j]
                selected[i*beam_size+j].append([output[i],output2[i][j] if output2[i][j]<output[i] else output2[i][j]+1])
            for j in range(len(output2[i]),beam_size):
                rule2s.append('')
                # if there are no enough theories that can form a valid theory pair with the former selected theory. set the corresponding label to 1 in omit
                omit[i][j]=1

        # compose two theories to derive an intermediate conclusion
        input_ids = reasoner_tokenize_batch(T5_Tokenizer, rule1s, rule2s, device)
        with torch.no_grad():
            pred = reasonermodel.predict_and_decode(input_ids, is_inference=True)

        tmp=[]
        for i in range(args.inference_batch_size):
            if result[i] != 'unknown':
                continue
            for j in range(beam_size):
                tmp.append(contexts[i][:])
                if omit[i][j]==1:
                    continue
                if pred[i * beam_size + j] == '' or pred[i * beam_size + j] == '.':
                    result[i] = 'no'
                    break
                elif pred[i * beam_size + j] in contexts[i]:
                    omit_selected_index[i * beam_size + j].append(len(selected[i*beam_size+j])-1)
                else:
                    tmp[i*beam_size+j].append(pred[i*beam_size+j])
        contexts = tmp
        inf_time += 1
        while stop is False and inf_time<max_step:
            inf_time+=1
            # select a theory in rule1s
            input_ids, attn_mask, token_mask = selector_tokenize_batch(XLNET_tokenizer,contexts,device)
            with torch.no_grad():
                output=selectormodel(input_ids, attn_mask, token_mask, return_loss=False,top_n=1)
                rule1s=[]
                for i in range(len(output)):
                    rule1s.append(contexts[i][output[i]])

            # select another theory in rule2s
            input_ids,attn_mask,tobeselected, selector1_selected = selector2_tokenize_batch(XLNET_tokenizer,contexts, output, device)
            with torch.no_grad():
                output2,probs=selector2model(input_ids,attn_mask,tobeselected,selected=selector1_selected,return_loss=False,top_n=candidates_num)
                # use beam search according to validity and possibility
                rule1s,rule2s,contexts,current_probs,omit,selected,omit_selected_index=beam_search(output2, probs, current_probs, selected, contexts,
                                            args.inference_batch_size, beam_size,rule1s,output,candidates_num,omit_selected_index)

            # compose two theories to derive an intermediate conclusion
            input_ids = reasoner_tokenize_batch(T5_Tokenizer,rule1s,rule2s,device)
            with torch.no_grad():
                pred=reasonermodel.predict_and_decode(input_ids,is_inference=True)

            for i in range(args.inference_batch_size):
                if result[i] != 'unknown':
                    continue
                flag=0
                for j in range(beam_size):
                    if omit[i][j] == 1:
                        continue
                    flag = 1
                    if pred[i*beam_size+j]=='' or pred[i*beam_size+j]=='.':
                        result[i]='no'
                        break
                    elif pred[i*beam_size+j] in contexts[i*beam_size+j]:
                        omit_selected_index[i * beam_size + j].append(len(selected[i*beam_size+j])-1)
                    else:
                        contexts[i*beam_size+j].append(pred[i*beam_size+j])
                if flag==0:
                    result[i]='yes'
            if 'unknown' not in result:
                stop=True
        for i in range(len(result)):
            resultlist[d[result[i]]][d[answers[i]]]+=1
            # If we cannot infer a contradiction within max_step, we will assume that there is no contradiction in this theory set
            if result[i]=='unknown':
                result[i]='yes'
            if result[i]==answers[i]:
                eval_accuracy+=1
            eval_examples+=1
        if (step+1)%int(test_epoch*0.05)==0:
            logger.info(f'correct/num: {eval_accuracy}/{eval_examples}')
            logger.info(f'proof_correct/proof_allnum: {proof_acc}/{proof_allnum}')
            logger.info(resultlist)

    logger.info(f'total_correct/total_num: {eval_accuracy}/{eval_examples}')
    eval_num=eval_accuracy
    eval_accuracy = float(eval_accuracy) / (eval_examples)
    logger.info(f'accuracy: {eval_accuracy}')
    logger.info('saving predictions.txt in ' + args.output_dir)
    logger.info(resultlist)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_dir + '/test_result_recording.txt', 'w+') as writer:
        writer.write(f'total_correct/total_num: {eval_num}/{eval_examples}\n')
        writer.write(f'accuracy: {eval_accuracy}')
        writer.write(f'resultlist: {resultlist}')

if __name__ == "__main__":
    main()