from __future__ import absolute_import
import argparse
import logging
import os, random
from selector import Selector2
from convertor import Convertor
from io import open
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler,Dataset)
import jsonlines,json, pickle
from transformers import XLNetTokenizer,T5Tokenizer
from torch.nn.utils.rnn import pad_sequence
from data.logic1 import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',level = logging.INFO)
logger = logging.getLogger(__name__)

def FOL2NL(follist):
    l=[]
    if follist == ['False']:
        return ''
    for i,fol in enumerate(follist):
        flag = 0 if fol[:3]=='Not' else 1
        fol = fol[4:-1] if fol[:3]=='Not' else fol
        fol= fol.split('(')
        predicate=fol[0].lower()
        liters=fol[1][:-1].split(',')
        if len(liters)==1:
            liter=liters[0]
            if liter[:2] == '$x' or liter[:2] == '$y':
                liter='everything'
            elif liter.startswith("skolem$"):
                liter='something'
            s=liter+' is not ' +predicate if flag==0 else liter+' is '+predicate
            l.append(s)
        elif len(liters)==2:
            liter1,liter2=liters[0],liters[1]
            if liter1[:2] == '$x' or liter1[:2] == '$y':
                liter1 = 'everything'
            elif liter1.startswith("skolem$"):
                liter1 = 'something'

            if liter2[:2] == '$x' or liter2[:2] == '$y':
                liter2 = 'everything'
            elif liter2.startswith("skolem$"):
                liter2 = 'something'
            s = liter1+' does not '+predicate+' '+liter2 if flag==0 else liter1+' '+predicate+'s '+liter2
            l.append(s)
    ret=l[0]
    for i in range(1,len(l)):
        ret += ' or '+l[i]
    return ret+'.'

def read_file(filedirec,split):
    with open(filedirec+'/'+split+'.jsonl') as f:
        data=f.readlines()
    with open(filedirec+'/FOL'+split+'.json') as f:
        data1=json.load(f)
    return data, data1

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

def upper(strB):
    return strB[0].upper() + strB[1:].lower()

class InferenceDataset(Dataset):

    def __init__(self, args):
        datas,fol_datas=read_file(args.input_dir,args.split)
        self.datas={'context':[],'answer':[],'query':[],'fols':[],'dep':[],'id':[]}
        for data in datas:
            data = json.loads(data)
            id = data["id"]
            self.datas['id'].append(id)
            context = data['context'].split(". ")
            r, q = context[-1].split(' $query$ ')
            context[-1]=r
            for i in range(len(context)):
                context[i] = context[i] + '.'
            for fol_data in fol_datas:
                id1 = fol_data['id']
                if id == id1:
                    self.datas['fols'].append([eval(fol) for fol in fol_data['fols']])
                    fol_q = eval([key for key in fol_data['query'].keys()][0])
                    self.datas['query'].append([q,fol_q,fol_q.arg if fol_q.isa(Not) else Not(fol_q)])
            self.datas['context'].append(context)
            self.datas['answer'].append(data["answer"])
            if "depth" in data.keys():
                self.datas['dep'].append(data["depth"] if data['answer']!='unknown' else 100)
            else:
                self.datas['dep'].append(0)

    def __len__(self):
        return len(self.datas['answer'])

    def __getitem__(self, item):
        return {'context':self.datas['context'][item], 'answer':self.datas['answer'][item],'query':self.datas['query'][item],
                'fols':self.datas['fols'][item], 'dep':self.datas['dep'][item],'id':self.datas['id'][item]}

class Inference_collate_fn:
    def __init__(self):
        pass
    def __call__(self, items):
        all_context = [x['context'] for x in items]
        all_answer = [x['answer'] for x in items]
        all_fols = [x['fols'] for x in items]
        all_query = [x['query'][0] for x in items]
        all_query_fol = [x['query'][1] for x in items]
        all_query_fol_reverse = [x['query'][2] for x in items]
        all_dep = [x['dep'] for x in items]
        all_id = [x['id'] for x in items]
        return (all_context,all_answer,all_query,all_query_fol,all_query_fol_reverse,all_fols,all_dep,all_id)

def convertor_tokenize_batch(tokenizer,contexts,device,q,reverse):
    ruleid2contextid={}
    num=0
    input=[]
    q_nums=[]
    for i in range(len(contexts)):
        context=contexts[i]
        for s in context:
            input.append(s + tokenizer.eos_token)
            ruleid2contextid[num]=i
            num+=1
        input.append(q[i]+tokenizer.eos_token if reverse==False else q[i] + ' reverse: ' + tokenizer.eos_token)
        ruleid2contextid[num] = i
        q_nums.append(num)
        num += 1
    tokenized = tokenizer(input, padding=True, return_tensors='pt',add_special_tokens=False)
    input_ids = tokenized['input_ids']
    return input_ids.to(device),ruleid2contextid,q_nums

def selector_tokenize_batch(tokenizer,contexts,device):
    input_tokens = []
    for i in range(len(contexts)):
        num=len(contexts[i])-1
        s = tokenizer.cls_token + contexts[i][num] + tokenizer.sep_token
        for j in range(len(contexts[i])):
            if j != num:
                s += contexts[i][j] + tokenizer.sep_token
        input_tokens.append(torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))))
    input_ids = pad_sequence(input_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = (input_ids != tokenizer.pad_token_id).long()
    tobeselected,selected = [],[]
    for tokens in input_ids:
        mask = []
        for i in range(len(tokens)):
            if tokens[i] == tokenizer.sep_token_id:
                mask.append(i)
        selected.append(mask[0])
        tobeselected.append(torch.LongTensor(mask[1:]).to(device))
    return input_ids.to(device), attn_mask.to(device), tobeselected, torch.LongTensor(selected).to(device)

def reasoner_tokenize_batch(tokenizer,rule1s,rule2s,device):
    input_tokens = [rule1.strip(' ') + ' ' + rule2.strip(' ') + ' ' + tokenizer.eos_token for rule1,rule2 in zip(rule1s,rule2s)]
    tokenized = tokenizer(input_tokens, padding=True, return_tensors='pt',add_special_tokens=False)
    input_ids = tokenized['input_ids']
    return input_ids.to(device)

def get_topn(probs,candidates_num,beam_size):
    ret=[]
    probs,indexes=torch.sort(torch.tensor(probs),descending=True)
    for i in range(min(beam_size,len(indexes))):
        if probs[i]<-100000:
            break
        ret.append((indexes[i].data//candidates_num,indexes[i].data%candidates_num))
    return ret,probs[:len(ret)].numpy().tolist()

def beam_search1(output, probs, current_probs, selected, contexts,rule1s,beam_size,rule1s_num,candidates_num,resultlen,omit_selected_index):
    p,omits,selected_ret,omit_selected_index_ret=[],[0 for _ in range(beam_size)],[],[]
    for i in range(len(output)):
        for j in range(len(output[i])):
            if [rule1s_num[i],output[i][j] if output[i][j]<rule1s_num[i] else output[i][j]+1] in selected[i] or [output[i][j] if output[i][j]<rule1s_num[i] else output[i][j]+1,rule1s_num[i]] in selected[i]:
                p.append(-1000000)
            else:
                p.append(current_probs[i]+probs[i][j])
        for j in range(len(output[i]),candidates_num):
            p.append(-1000000)
    indexes,p_now=get_topn(p,candidates_num,beam_size)
    rule1s_ret,rule2s_ret,current_probs_ret,contexts_ret,resultlen_ret=[],[],[],[],[]
    for i in range(len(indexes)):
        rule1s_ret.append(rule1s[indexes[i][0]])
        rule2s_ret.append(contexts[indexes[i][0]][output[indexes[i][0]][indexes[i][1]] if output[indexes[i][0]][indexes[i][1]]<rule1s_num[indexes[i][0]] else output[indexes[i][0]][indexes[i][1]]+1])
        contexts_ret.append(contexts[indexes[i][0]])
        resultlen_ret.append(resultlen[indexes[i][0]])
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
        resultlen_ret.append(0)
        omit_selected_index_ret.append([])
    return rule1s_ret,rule2s_ret,contexts_ret,current_probs_ret,omits,selected_ret,resultlen_ret,omit_selected_index_ret

def beam_search(output,probs,current_probs,selected,contexts,batch_size,beam_size,rule1s,rule1s_num,candidates_num,resultlen_inbatch,omit_selected_index):
    rule1s_ret,rule2s_ret,contexts_ret,current_probs_ret,omits_ret,selected_ret,resultlen_ret,omit_selected_index_ret=[],[],[],[],[],[],[],[]
    for i in range(batch_size):
        rule1s_tmp,rule2s_tmp,contexts_tmp,current_probs_tmp,omits_tmp,selected_tmp,resultlen_tmp,omit_selected_index_tmp=beam_search1(output[i*beam_size:(i+1)*beam_size],
            probs[i*beam_size:(i+1)*beam_size],current_probs[i*beam_size:(i+1)*beam_size],selected[i*beam_size:(i+1)*beam_size],contexts[i*beam_size:(i+1)*beam_size],
            rule1s[i*beam_size:(i+1)*beam_size],beam_size,rule1s_num[i*beam_size:(i+1)*beam_size],candidates_num,resultlen_inbatch[i*beam_size:(i+1)*beam_size],
                                                                                                               omit_selected_index[i*beam_size:(i+1)*beam_size])
        for j in range(beam_size):
            rule1s_ret.append(rule1s_tmp[j])
            rule2s_ret.append(rule2s_tmp[j])
            contexts_ret.append(contexts_tmp[j][:])
            current_probs_ret.append(current_probs_tmp[j])
            selected_ret.append(selected_tmp[j][:])
            resultlen_ret.append(resultlen_tmp[j])
            omit_selected_index_ret.append(omit_selected_index_tmp[j][:])
        omits_ret.append(omits_tmp[:])

    return rule1s_ret,rule2s_ret,contexts_ret,current_probs_ret,omits_ret,selected_ret,resultlen_ret,omit_selected_index_ret

def testbatch(q,T5_Tokenizer,orig_contexts,device,convertormodel,XLNET_tokenizer,selector2model,reasonermodel,reverse,
              beam_size,candidates_num,input_dir,orig_fols,fol_q):
    batch_size=len(orig_contexts)
    # convert
    input_ids, ruleid2contextid,q_nums = convertor_tokenize_batch(T5_Tokenizer, orig_contexts, device, q, reverse)
    contexts,fols = [[] for _ in range(len(orig_contexts))],[[] for _ in range(len(orig_contexts))]
    tmp_fols = []
    for i in range(len(orig_fols)):
        for fol in orig_fols[i]:
            tmp_fols.append(fol)
        tmp_fols.append(fol_q[i])
    assert len(input_ids)==len(tmp_fols)

    yes_num=[0 for _ in range(len(orig_contexts))]
    with torch.no_grad():
        output = convertormodel.predict_and_decode(input_ids, is_inference=True)
        for i in range(len(output)):
            if output[i] not in contexts[ruleid2contextid[i]]:
                contexts[ruleid2contextid[i]].append(output[i])
                fols[ruleid2contextid[i]].append(tmp_fols[i])
            elif i in q_nums:
                yes_num[ruleid2contextid[i]]=1
                contexts[ruleid2contextid[i]].append(output[i])
                fols[ruleid2contextid[i]].append(tmp_fols[i])

    startlen=[len(contexts[i]) for i in range(len(contexts))]
    midprove,midprove_num = [[] for _ in range(len(contexts))],[[] for _ in range(len(contexts))]
    inf_time = 0
    result = ['unknown' if yes_num[i]==0 else 'yes' for i in range(batch_size)]
    resultlen = [0 for _ in result]
    resultlen_inbatch=[0 for _ in range(beam_size * batch_size)]
    current_probs = [0 for _ in range(batch_size * beam_size)]
    selected = [[] for _ in range(batch_size * beam_size)]
    omit_selected_index = [[] for _ in range(batch_size * beam_size)]

    # select the last theory in rule1s according to the inference strategies
    rule1s = []
    for i in range(batch_size):
        for _ in range(beam_size):
            rule1s.append(contexts[i][-1])

    # select another theory in rule2s
    input_ids, attn_mask, tobeselected, selector1_selected = selector_tokenize_batch(XLNET_tokenizer, contexts, device)
    with torch.no_grad():
        output,probs = selector2model(input_ids, attn_mask, tobeselected,selected=selector1_selected, return_loss=False, top_n=candidates_num)

    rule2s, omit = [], [[0 for _ in range(beam_size)] for _ in range(batch_size)]
    for i in range(len(output)):
        for j in range(len(output[i])):
            rule2s.append(contexts[i][output[i][j]])
            current_probs[i * beam_size + j] = probs[i][j]
            selected[i * beam_size + j].append([len(contexts[i])-1, output[i][j]])
        for j in range(len(output[i]), beam_size):
            rule2s.append('')
            omit[i][j] = 1

    # compose two theories to derive an intermediate conclusion
    input_ids = reasoner_tokenize_batch(T5_Tokenizer, rule1s, rule2s, device)
    with torch.no_grad():
        pred = reasonermodel.predict_and_decode(input_ids, is_inference=True)

    tmp=[]
    for i in range(batch_size):
        for _ in range(beam_size):
            tmp.append(contexts[i][:])
        if result[i] != 'unknown':
            continue
        for j in range(beam_size):
            if omit[i][j] == 1:
                continue
            if pred[i * beam_size + j] == '' or pred[i * beam_size + j] == '.':
                result[i] = 'no'
                resultlen[i]=resultlen_inbatch[i*beam_size+j]+1
                # get an intermediate reasoning step
                midprove[i],midprove_num[i] = getmidprove(selected[i * beam_size + j], startlen[i], contexts[i],omit_selected_index[i * beam_size + j])
                break
            elif pred[i * beam_size + j] in contexts[i]:
                omit_selected_index[i * beam_size + j].append(len(selected[i * beam_size + j]) - 1)
            else:
                tmp[i * beam_size + j].append(pred[i * beam_size + j])
                resultlen_inbatch[i* beam_size + j] += 1
    contexts=tmp
    inf_time += 1
    if input_dir=='depth-5':
        max_time = 20
    else:
        max_time = 10 if 'sat' in input_dir else 40
    while inf_time < max_time and 'unknown' in result:
        inf_time += 1
        # select the last theory in rule1s according to the inference strategies
        rule1s, rule1s_num = [], []
        for i in range(len(contexts)):
            rule1s.append(contexts[i][-1])
            rule1s_num.append(len(contexts[i]) - 1)

        # select another theory in rule2s
        input_ids, attn_mask, tobeselected, selector1_selected = selector_tokenize_batch(XLNET_tokenizer, contexts, device)
        with torch.no_grad():
            output,probs = selector2model(input_ids, attn_mask, tobeselected,selected=selector1_selected, return_loss=False,top_n=candidates_num)
            # use beam search according to validity and possibility
            rule1s, rule2s, contexts, current_probs, omit, selected, resultlen_inbatch,omit_selected_index = beam_search(output, probs, current_probs, selected, contexts, batch_size, beam_size,
                                                                              rule1s, rule1s_num, candidates_num, resultlen_inbatch,omit_selected_index)

        # compose two theories to derive an intermediate conclusion
        input_ids = reasoner_tokenize_batch(T5_Tokenizer, rule1s, rule2s, device)
        with torch.no_grad():
            pred = reasonermodel.predict_and_decode(input_ids, is_inference=True)

        for i in range(batch_size):
            if result[i] != 'unknown':
                continue
            flag = 0
            for j in range(beam_size):
                if omit[i][j] == 1:
                    continue
                flag = 1
                if pred[i * beam_size + j] == '' or pred[i * beam_size + j] == '.':
                    result[i] = 'no'
                    resultlen[i]=resultlen_inbatch[i * beam_size + j]+1
                    # get an intermediate reasoning step
                    midprove[i],midprove_num[i] = getmidprove(selected[i * beam_size + j], startlen[i], contexts[i * beam_size + j],omit_selected_index[i * beam_size + j])
                    break
                elif pred[i * beam_size + j] in contexts[i * beam_size + j]:
                    omit_selected_index[i * beam_size + j].append(len(selected[i * beam_size + j]) - 1)
                else:
                    contexts[i * beam_size + j].append(pred[i * beam_size + j])
                    resultlen_inbatch[i * beam_size + j] += 1
            if flag == 0:
                result[i] = 'yes'
    # If we cannot infer a contradiction within max_step, we will assume that there is no contradiction in this theory set
    for i in range(len(result)):
        if result[i]=='unknown':
            result[i]='yes'
    return result,resultlen,midprove,contexts,midprove_num,fols

def merge(result1,result2,resultlen1,resultlen2,midprove1,midprove2,contexts1,contexts2,midprove_num1,midprove_num2,fols1,fols2):
    midprove,contexts,midprove_num=[[] for _ in range(len(midprove1))],[[] for _ in range(len(midprove1))],[[] for _ in range(len(midprove1))]
    fols=[[] for _ in range(len(fols1))]
    result=['' for _ in result2]
    for i in range(len(result1)):
        if result1[i]=='yes' and result2[i]=='no':
            result[i]='true'
            midprove[i]=midprove2[i]
            contexts[i]=contexts2[i]
            fols[i]=fols2[i]
            midprove_num[i]=midprove_num2[i]
        elif result1[i]=='no' and result2[i]=='yes':
            result[i]='false'
            midprove[i] = midprove1[i]
            contexts[i] = contexts1[i]
            fols[i] = fols1[i]
            midprove_num[i] = midprove_num1[i]
        elif result1[i]=='yes' and result2[i]=='yes':
            result[i]='unknown'
        else:
            # If contradictions are inferred in both theory sets, the label will be determined by the party with fewer
            # reasoning steps, because the more reasoning steps there are, the more likely the inference is to go wrong.
            if resultlen1[i]<resultlen2[i]:
                result[i]='false'
                midprove[i] = midprove1[i]
                fols[i] = fols1[i]
                contexts[i] = contexts1[i]
                midprove_num[i] = midprove_num1[i]
            else:
                result[i]='true'
                midprove[i] = midprove2[i]
                contexts[i] = contexts2[i]
                fols[i] = fols2[i]
                midprove_num[i] = midprove_num2[i]
    return result,midprove,contexts,midprove_num,fols

def getmidprove(selected,origlen,contexts,omit_selected_index):
    mid_proves,mid_provenums=[],[]
    for i in range(len(selected)):
        if i in omit_selected_index:
            continue
        r1,r2=selected[i]
        mid_provenums.append([r1, r2])
        if origlen==len(contexts):
            mid_proves.append([contexts[r1][:], contexts[r2][:], ''])
        else:
            mid_proves.append([contexts[r1][:],contexts[r2][:],contexts[origlen][:]])
        origlen+=1
    return mid_proves,mid_provenums

def check(midnlprove, midprove_num1, fols1):
    resolution=ResolutionRule()
    ToCNF=ToCNFRule()
    for i in range(len(fols1)):
        fols1[i]=ToCNF.applyRule(fols1[i])[0]
    for i in range(len(midprove_num1)):
        r1=fols1[midprove_num1[i][0]]
        r2=fols1[midprove_num1[i][1]]
        midfolprove = resolution.applyRule(r1, r2)
        if len(midfolprove)==0:
            return False
        elif midfolprove[0]==AtomFalse:
            if midnlprove[i][-1]=='' or midnlprove[i][-1]=='.':
                return True
            else:
                return False
        else:
            mid_goldnls=[]
            for fol in midfolprove:
                # get all the possible gold NL corresponding to gold fol for comparation
                mid_goldnl=getgoldnls(flattenOr(fol))
                mid_goldnls.append(mid_goldnl)
            flag=0
            for j in range(len(mid_goldnls)):
                flag=0
                for goldnl in mid_goldnls[j]:
                    if goldnl==midnlprove[i][-1]:
                        flag=1
                        fols1.append(midfolprove[j])
                        break
                if flag==1:
                    break
            if flag==0:
                return False
    return False

fullarrangedict={}
def getgoldnls(fol):
    # The atomic formulas connected by 'or' can be arranged in any order
    n=len(fol)
    if n not in fullarrangedict.keys():
        fullarrangedict[n]=fullarrange(n)
    NLs=create_goldnl(fol,fullarrangedict)
    return NLs

def create_goldnl(fol,d):
    num=len(fol)
    l=d[num]
    NLs=[]
    for i in range(len(l)):
        fol_tmp=[]
        for j in range(len(l[i])):
            fol_tmp.append(str(fol[l[i][j]]))
        NLs.append(FOL2NL(fol_tmp))
    return NLs

def fullarrange(n):
    t = []
    sum = 1
    for i in range(1, n+1):
        sum = sum * i
        t.append(str(i-1))
    s = set()
    while len(s) < sum:
        random.shuffle(t)
        s.add("".join(t))
    s = sorted(s)
    ret=[]
    for i in range(len(s)):
        str1=s[i]
        tmp=[]
        for j in range(len(str1)):
            tmp.append(int(str1[j]))
        ret.append(tmp)
    assert len(ret)==len(s)
    return ret

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
                        default='out_convertor/ruletaker/pytorch_model.bin0',
                        type=str,
                        help="The directory where init model wegihts an config are stored.")
    parser.add_argument("--selector_init_weights_dir",
                        default='out_selector/rclause_16_20_21/pytorch_model.bin13',
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
    parser.add_argument('--split',
                        default='test',
                        type=str,
                        help="random seed for initialization")
    parser.add_argument('--beam_size',
                        type=int,
                        default=2,
                        help="random seed for initialization")
    parser.add_argument('--candidates_num',
                        type=int,
                        default=2,
                        help="random seed for initialization")
    parser.add_argument('--maxdep',
                        type=int,
                        default=-1,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    XLNET_tokenizer = XLNetTokenizer.from_pretrained('../../model/xlnet', do_lower_case=True)
    T5_Tokenizer= T5Tokenizer.from_pretrained('../../model/T5',do_lower_case=True)

    model_state_dict = torch.load(args.selector_init_weights_dir, map_location=device)
    selectormodel = Selector2.from_pretrained('../../model/xlnet',state_dict=model_state_dict)
    model_state_dict = torch.load(args.convertor_init_weights_dir, map_location=device)
    convertormodel = Convertor('../../model/T5')
    convertormodel.load_state_dict(model_state_dict)
    model_state_dict = torch.load(args.reasoner_init_weights_dir, map_location=device)
    reasonermodel = Convertor('../../model/T5')
    reasonermodel.load_state_dict(model_state_dict)
    del model_state_dict

    selectormodel.to(device)
    convertormodel.to(device)
    reasonermodel.to(device)
    if n_gpu > 1:
        reasonermodel = torch.nn.DataParallel(reasonermodel)
        selectormodel = torch.nn.DataParallel(selectormodel)
        convertormodel = torch.nn.DataParallel(convertormodel)
    selectormodel.eval()
    reasonermodel.eval()
    convertormodel.eval()

    collator = Inference_collate_fn()
    test_data = InferenceDataset(args)
    test_epoch = len(test_data)//args.inference_batch_size
    reservenum = len(test_data) - test_epoch * args.inference_batch_size
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,sampler=test_sampler,batch_size=args.inference_batch_size,collate_fn=collator)
    eval_examples, eval_accuracy, proof_allnum, proof_acc, faithful = [0 for _ in range(args.maxdep+2)], \
        [0 for _ in range(args.maxdep+2)], [0 for _ in range(args.maxdep+2)], [0 for _ in range(args.maxdep+2)], [0 for _ in range(args.maxdep+2)]
    eval_examples_total, eval_accuracy_total, proof_allnum_total, proof_acc_total, faithful_total = 0,0,0,0,0
    # [depth, predicted label, gold label]
    resultlist=[[[0 for _ in range(3)] for _ in range(3)] for _ in range(args.maxdep+2)]
    d={'unknown':2,'true':1,'false':0}
    for step, batch in enumerate(tqdm(test_dataloader, desc="Inference")):
        orig_contexts,answers,q,fol_q,fol_q_reverse,fols,deps,ids=batch
        if step == test_epoch:
            orig_contexts=orig_contexts[:reservenum]
            answers=answers[:reservenum]
            q=q[:reservenum]
            fol_q = fol_q[:reservenum]
            fol_q_reverse = fol_q_reverse[:reservenum]
            fols = fols[:reservenum]
            deps = deps[:reservenum]
        # Unknown responds to the args.maxdep+1
        deps = [deps[i] if deps[i]!=100 else -1 for i in range(len(deps))]
        # reason for the theory set composing NL Theory and the hypothesis
        result1,resultlen1,midprove1,contexts1,midprove_num1,fols1=testbatch(q,T5_Tokenizer,orig_contexts,device,convertormodel,XLNET_tokenizer,
            selectormodel,reasonermodel,False,args.beam_size,args.candidates_num,args.input_dir,fols,fol_q)
        # reason for the theory set composing NL Theory and the negation of the hypothesis
        result2,resultlen2,midprove2,contexts2,midprove_num2,fols2=testbatch(q,T5_Tokenizer,orig_contexts,device,convertormodel,XLNET_tokenizer,
            selectormodel,reasonermodel,True,args.beam_size,args.candidates_num,args.input_dir,fols,fol_q_reverse)
        # get the final label and the reasoning procedure according to the reasoning results of previous two theory sets
        result,midprove,contexts,midprove_num,fols = merge(result1,result2,resultlen1,resultlen2,midprove1,midprove2,contexts1,contexts2,midprove_num1,midprove_num2,fols1,fols2)
        for i in range(len(result1)):
            resultlist[deps[i]][d[result[i]]][d[answers[i]]] += 1
            if result[i] == answers[i]:
                eval_accuracy_total += 1
                eval_accuracy[deps[i]] += 1
                if answers[i]=='true' or answers[i]=='false':
                    proof_allnum_total+=1
                    proof_allnum[deps[i]] += 1
                    if check(midprove[i],midprove_num[i],fols[i]) is True:
                        proof_acc_total+=1
                        proof_acc[deps[i]] += 1
            eval_examples_total += 1
            eval_examples[deps[i]] += 1
        if (step+1)%int(test_epoch*0.05)==0:
            logger.info(f'correct_total/num_total: {eval_accuracy_total}/{max(eval_examples_total,1e-3)}')
            logger.info(f'proof_acc_total: {(proof_acc_total+eval_accuracy_total-proof_allnum_total)/max(eval_examples_total,1e-3)}')
            logger.info(f'faithful_total: {(proof_acc_total+eval_accuracy_total-proof_allnum_total)/max(eval_accuracy_total,1e-3)}')
            for i in range(args.maxdep+2):
                logger.info(resultlist[i])
                logger.info(f'correct/num_dep{i}: {eval_accuracy[i]}/{max(eval_examples[i],1e-3)}')
                logger.info(f'proof_acc_dep{i}: {(proof_acc[i] + eval_accuracy[i] - proof_allnum[i]) / max(eval_examples[i],1e-3)}')
                logger.info(f'faithful_dep{i}: {(proof_acc[i] + eval_accuracy[i] - proof_allnum[i]) / max(eval_accuracy[i],1e-3)}')
                logger.info('')

    logger.info(f'correct_total/num_total: {eval_accuracy_total}/{max(eval_examples_total,1e-3)}')
    logger.info(f'proof_acc_total: {(proof_acc_total + eval_accuracy_total - proof_allnum_total) / max(eval_examples_total,1e-3)}')
    logger.info(f'faithful_total: {(proof_acc_total + eval_accuracy_total - proof_allnum_total) / max(eval_accuracy_total,1e-3)}')
    for i in range(args.maxdep + 2):
        logger.info(resultlist[i])
        logger.info(f'correct/num_dep{i}: {eval_accuracy[i]}/{max(eval_examples[i],1e-3)}')
        logger.info(f'proof_acc_dep{i}: {(proof_acc[i] + eval_accuracy[i] - proof_allnum[i]) / max(eval_examples[i],1e-3)}')
        logger.info(f'faithful_dep{i}: {(proof_acc[i] + eval_accuracy[i] - proof_allnum[i]) / max(eval_accuracy[i],1e-3)}')
        logger.info('')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_dir + '/test_result_recording.txt', 'a+') as writer:
        writer.write(f'correct_total/num_total: {eval_accuracy_total}/{eval_examples_total}')
        writer.write(f'proof_acc_total: {(proof_acc_total + eval_accuracy_total - proof_allnum_total) / eval_examples_total}')
        writer.write(f'faithful_total: {(proof_acc_total + eval_accuracy_total - proof_allnum_total) / eval_accuracy_total}')
        for i in range(args.maxdep + 2):
            writer.write(f'correct/num_(dep{i}): {eval_accuracy[i]}/{max(eval_examples[i],1e-3)}')
            writer.write(f'proof_acc_dep{i}: {(proof_acc[i] + eval_accuracy[i] - proof_allnum[i]) / max(eval_examples[i],1e-3)}')
            writer.write(f'faithful_dep{i}: {(proof_acc[i] + eval_accuracy[i] - proof_allnum[i]) / max(eval_accuracy[i],1e-3)}')
            writer.write('')

if __name__ == "__main__":
    main()

