import json, logging, pickle, argparse
from tqdm import tqdm
from logic1 import *
from transformers import T5Tokenizer,XLNetTokenizer
from FOLReasoning import reversefact, FOL2NL
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

START_TOK, END_TOK, SPAN_SEP, IGNORE_IDX = '@', '\\', ';', 0

class selector2Features(object):
    def __init__(self,tokens,output,token_mask,tobeselected,selected,positives,negatives):
        self.tokens=tokens
        self.output=output
        self.token_mask=token_mask
        self.tobeselected=tobeselected
        self.selected=selected
        self.positives=positives
        self.negatives=negatives

def read_jsonl(filedirec):
    with open(filedirec + '/test.jsonl') as f:
        return f.readlines()

def read_file(file):
    with open(file, encoding='utf8') as f:
        return json.load(f)

def read_pkl(file):
    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'rb') as f:
            return pickle.load(f)

def write_file(data, file):
    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def makeseletordata(args,split):
    orig_datas = read_file(args.direc+split+'_withmidprove.json')
    tokenizer = XLNetTokenizer.from_pretrained(args.model, do_lower_case=True)
    cls,sep=tokenizer.cls_token,tokenizer.sep_token
    input_data=[]
    output_data=[]
    token_masks=[]
    for orig_data in tqdm(orig_datas):
        mid_proves = orig_data['gold_mid_proves']
        s=cls
        for i,mid_prove in enumerate(mid_proves):
            if len(mid_prove)==2:
                s += mid_prove[1][0].strip('.')+'.'+ sep
            if len(mid_prove)==4:
                r1_num,r2_num,_,new_r=mid_prove
                new_r = new_r[0].strip('.')
                tokens=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
                input_data.append(tokens)
                num=-1
                if split=='train':
                    output = [0 for _ in range(len(tokens))]
                    for j in range(len(tokens)):
                        if tokens[j] == tokenizer.sep_token_id:
                            num += 1
                            if num == r1_num or num == r2_num:
                                output[j] = 1
                    token_mask=[1 if token == tokenizer.sep_token_id else 0 for token in tokens]
                else:
                    token_mask=[]
                    for j in range(len(tokens)):
                        if tokens[j] == tokenizer.sep_token_id:
                            token_mask.append(j)
                    output=(r1_num,r2_num)
                output_data.append(output)
                s += new_r + '.' + sep
                token_masks.append(token_mask)

    write_file(input_data,args.direc+split+'_input_selector'+args.model_size+'.pkl')
    write_file(output_data, args.direc + split + '_output_selector'+args.model_size+'.pkl')
    write_file(token_masks, args.direc + split + '_tokenmask_selector'+args.model_size+'.pkl')

def unify2(s1,s2):
    predicate1 = s1.split('(')[0]
    predicate2 = s2.split('(')[0]
    liter1 = s1.split('(')[1][:-1]
    liter2 = s2.split('(')[1][:-1]
    if (predicate1[0]=='-' and predicate2[0]!='-' and predicate1[1:]==predicate2) or \
            predicate1[0]!='-' and predicate2[0]=='-' and predicate1==predicate2[1:]:
        if liter1=='x' or liter2=='x':
            return True
        elif liter1==liter2:
            return True
    return False

def unify1(s1,s2):
    if (s1[0]=='-' and s2[0]!='-' and s1[1:]==s2) or s1[0]!='-' and s2[0]=='-' and s1==s2[1:]:
        return True
    return False

def getposi_neg_propositional(l1,num1):
    fol=l1[num1].split(' | ')
    l=[]
    for i in range(len(l1)):
        if i != num1:
            l.append(l1[i])
    posi,neg=[],[]
    for i in range(len(l)):
        atoms=l[i].split(' | ')
        flag=0
        for atom in atoms:
            for item in fol:
                if unify1(atom,item):
                    posi.append(i)
                    flag=1
                    break
            if flag==1:
                break
        if i not in posi:
            neg.append(i)
    return posi,neg

def getposi_neg(l1,num1):
    fol=l1[num1].split(' | ')
    l=[]
    for i in range(len(l1)):
        if i != num1:
            l.append(l1[i])
    posi,neg=[],[]
    for i in range(len(l)):
        atoms=l[i].split(' | ')
        flag=0
        for atom in atoms:
            for item in fol:
                if unify2(atom,item):
                    posi.append(i)
                    flag=1
                    break
            if flag==1:
                break
        if i not in posi:
            neg.append(i)
    return posi,neg

def getposi_neg_rt(l1,num1):
    fol=l1[num1]
    l=[]
    for i in range(len(l1)):
        if i != num1:
            l.append(l1[i])
    posi,neg=[],[]
    reso=ResolutionRule()
    for i in range(len(l)):
        ret=reso.applyRule(l[i],fol)
        if ret!=[]:
            posi.append(i)
        if i not in posi:
            neg.append(i)
    return posi,neg

def createdata(l,l1,num1,num2,tokenizer,func=getposi_neg):
    s=tokenizer.cls_token+l[num1]+tokenizer.sep_token
    for i in range(len(l)):
        if i!=num1:
            s+=l[i]+tokenizer.sep_token
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    num=-1
    if num1>num2:
        num2+=1
    output = [0 for _ in range(len(tokens))]
    token_mask=[0 for _ in range(len(tokens))]
    selected,tobeselected=0,[]
    # positives represents theories that can form a valid theory pair with the theory at the position of num1
    positives,negatives=func(l1,num1)
    for j in range(len(tokens)):
        if tokens[j] == tokenizer.sep_token_id:
            num += 1
            if num==0:
                selected=j
            else:
                token_mask[j]=1
                tobeselected.append(j)
            if num == num2:
                output[j] = 1

    positives=[tobeselected[i] for i in positives]
    negatives=[tobeselected[i] for i in negatives]
    return selector2Features(tokens,output,token_mask,tobeselected,selected,positives,negatives)

def makeseletor2data(args,split):
    orig_datas = read_file(args.direc+split+'_withmidprove.json')
    tokenizer = XLNetTokenizer.from_pretrained(args.model, do_lower_case=True)
    features=[]
    func = getposi_neg if 'relative_clause' in args.direc else getposi_neg_propositional
    for orig_data in tqdm(orig_datas):
        mid_proves = orig_data['gold_mid_proves']
        l,l1=[],[]
        for i,mid_prove in enumerate(mid_proves):
            if len(mid_prove)==2:
                l.append(mid_prove[1][0].strip('.')+'.')
                l1.append(mid_prove[1][1])
            if len(mid_prove)==4:
                r1_num,r2_num,_,new_r=mid_prove
                feature=createdata(l,l1,r1_num,r2_num,tokenizer,func=func)
                features.append(feature)
                feature = createdata(l, l1, r2_num, r1_num, tokenizer,func=func)
                features.append(feature)
                l.append(new_r[0].strip('.')+'.')
                l1.append(new_r[1])

    write_file(features,args.direc+split+'_selector2'+args.model_size+'.pkl')

def createdata1(l, l1, num1, num2, tokenizer):
    s = tokenizer.cls_token + l[num1] + tokenizer.sep_token
    for i in range(len(l)):
        if i != num1:
            s += l[i] + tokenizer.sep_token
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    num = -1
    if num1 > num2:
        num2 += 1

    output = [0 for _ in range(len(tokens))]
    token_mask = [0 for _ in range(len(tokens))]
    selected, tobeselected = 0, []
    # positives represents theories that can form a valid theory pair with the theory at the position of num1
    positives, negatives = getposi_neg_rt(l1, num1)
    for j in range(len(tokens)):
        if tokens[j] == tokenizer.sep_token_id:
            num += 1
            if num == 0:
                selected = j
            else:
                token_mask[j] = 1
                tobeselected.append(j)
            if num == num2:
                output[j] = 1

    positives = [tobeselected[i] for i in positives]
    negatives = [tobeselected[i] for i in negatives]

    return selector2Features(tokens, output, token_mask, tobeselected, selected, positives, negatives)

def makeseletordata_rt(args,split):
    orig_datas = read_pkl(args.direc+split+'_withmidprove.pkl')
    tokenizer = XLNetTokenizer.from_pretrained(args.model, do_lower_case=True)
    features=[]
    max_depth=0
    depth0_num=0
    for orig_data in tqdm(orig_datas):
        mid_proves = orig_data['gold_mid_proves']
        l,l1=[],[]
        num = 0
        for i,mid_prove in enumerate(mid_proves):
            if len(mid_prove)==2:
                l.append(mid_prove[1][0].strip('.') + '.')
                l1.append(mid_prove[1][1])
            if len(mid_prove)==4:
                num+=1
                r1_num,r2_num,_,new_r=mid_prove
                feature = createdata1(l[:], l1[:], r1_num, r2_num, tokenizer)
                if feature not in features:
                    features.append(feature)
                l.append(new_r[0].strip('.') + '.')
                l1.append(new_r[1])
        if num>max_depth:
            max_depth=num
        if num==1:
            depth0_num+=1
    print(max_depth)
    dir = args.direc
    write_file(features,dir+split+'_selector2'+args.model_size+'.pkl')

def makereasonerdata(args,split):
    orig_datas = read_file(args.direc + split + '_withmidprove.json')
    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=True)
    input_data = []
    output_data = []
    for orig_data in tqdm(orig_datas):
        mid_proves = orig_data['gold_mid_proves']
        proves,fols=[],[]
        for i, mid_prove in enumerate(mid_proves):
            if len(mid_prove) == 2:
                proves.append(mid_prove[1][0].strip('.'))
            if len(mid_prove) == 4:
                r1_num, r2_num, _, new_r = mid_prove
                new_r=new_r[0].strip('.')
                proves.append(new_r)
                tmp=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(proves[r1_num]+'.'+proves[r2_num]+'.'+tokenizer.eos_token))
                if tmp not in input_data:
                    input_data.append(tmp)
                    # output format: <pad> conclusion </s>
                    output_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.pad_token+new_r+'.'+tokenizer.eos_token)))
    write_file(input_data, args.direc+split+'_input_reasoner.pkl')
    write_file(output_data, args.direc + split + '_output_reasoner.pkl')

def makereasonerdata_rt(args,split):
    orig_datas = read_pkl(args.direc + split + '_withmidprove.pkl')
    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=True)
    input_data = []
    output_data = []
    reso = ResolutionRule()
    num,total_num=0,0
    for orig_data in tqdm(orig_datas):
        mid_proves = orig_data['gold_mid_proves']
        proves,fols=[],[]
        for i, mid_prove in enumerate(mid_proves):
            if len(mid_prove) == 2:
                proves.append(mid_prove[1][0].strip('.'))
                fols.append(mid_prove[1][1])
            if len(mid_prove) == 4:
                r1_num, r2_num, _, new_r = mid_prove
                fols.append(new_r[1])
                new_r=new_r[0].strip('.')
                proves.append(new_r)
                # input format: rule1 rule2 </s>
                tmp=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(proves[r1_num]+'. '+proves[r2_num]+'. '+tokenizer.eos_token))
                assert 'skolem' not in new_r
                if tmp not in input_data:
                    total_num += 1
                    rule1=reso.applyRule(fols[r2_num],fols[r1_num])
                    if rule1[0]==AtomFalse:
                        new_r1=''
                    else:
                        equal_flag=0
                        if len(rule1)>1:
                            num+=1
                        for item in rule1:
                            if equal_fol(item,fols[i]):
                                equal_flag=1
                        assert equal_flag==1
                        if equal_flag==0:
                            print(fols[r2_num],fols[r1_num],rule1[0],fols[i])
                            exit(0)
                        rule1 = flattenOr(fols[i])
                        rule1=[str(i) for i in rule1]
                        new_r1=FOL2NL(rule1)
                        assert 'skolem' not in new_r1
                    input_data.append(tmp)
                    # change the order of input
                    input_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(proves[r2_num]+'. '+proves[r1_num]+'. '+tokenizer.eos_token)))
                    # output format: <pad> conclusion </s>
                    output_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.pad_token+new_r+'.'+tokenizer.eos_token)))
                    output_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.pad_token + new_r1.strip('.') + '.' + tokenizer.eos_token)))
    print(num/total_num)
    write_file(input_data,args.direc+split+'_input_reasoner.pkl')
    write_file(output_data, args.direc + split + '_output_reasoner.pkl')

def makeconvertordata(args,split):
    orig_datas = read_file(args.direc+split+'_prover9CNFformat.json')
    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=True)
    input_data = []
    d=set()
    output_data = []
    for orig_data in tqdm(orig_datas):
        context2FOLs = orig_data["context2FOL"]
        for key in context2FOLs.keys():
            value=context2FOLs[key]
            if value[1] not in d:
                d.add(value[1])
                # input format: rule </s>
                input_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value[1] + tokenizer.eos_token)))
                # output format: <pad> conclusion </s>
                output_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.pad_token + value[0] + tokenizer.eos_token)))
    write_file(input_data,args.direc+split+'_input_convertor.pkl')
    write_file(output_data, args.direc + split + '_output_convertor.pkl')

def makeconvertordata_rt(args,split):
    orig_datas = read_file(args.direc+split+'_prover9CNFformat.json')
    tokenizer = T5Tokenizer.from_pretrained(args.model, do_lower_case=True)
    input_data = []
    d=set()
    output_data = []
    for orig_data in tqdm(orig_datas):
        context2FOLs = orig_data["context2FOL"]
        for key in context2FOLs.keys():
            value=context2FOLs[key]
            if value[1] not in d:
                d.add(value[1])
                # input format: rule </s>
                input_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value[1] + tokenizer.eos_token)))
                # output format: <pad> conclusion </s>
                output_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.pad_token + value[0] + tokenizer.eos_token)))
        query=orig_data["query"]
        for key in query.keys():
            value=query[key]
            if value[1] not in d:
                d.add(value[1])
                # input format: rule </s>
                input_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value[1] + tokenizer.eos_token)))
                # output format: <pad> conclusion </s>
                output_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tokenizer.pad_token + value[0] + tokenizer.eos_token)))
            nl,_ = reversefact(value[1])
            # input format: rule reverse: </s>
            input_data.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(value[1] + ' reverse: ' + tokenizer.eos_token)))
            # output format: <pad> conclusion </s>
            output_data.append(tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(tokenizer.pad_token + nl + tokenizer.eos_token)))

    write_file(input_data,args.direc+split+'_input_convertor.pkl')
    write_file(output_data, args.direc + split + '_output_convertor.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='reasoner',choices=['selector', 'convertor', 'reasoner','all'])
    parser.add_argument('--model_size', default='large', choices=['base', 'large'])
    parser.add_argument('--direc', default='3sat/grounded_rule_lang/12var/', type=str)
    args = parser.parse_args()
    random.seed(42)
    if args.model_type == 'selector' or args.model_type =='all':
        #args.model = 'xlnet-' + args.model_size+'-cased'
        args.model = '../../../model/xlnet'
        if 'ruletaker' in args.direc:
            makeseletordata_rt(args, split='train')
            makeseletordata_rt(args, split='dev')
            makeseletordata_rt(args, split='test')
        else:
            makeseletor2data(args, split='train')
            makeseletor2data(args, split='dev')
            makeseletor2data(args, split='test')
            makeseletordata(args, split='train')
            makeseletordata(args, split='dev')
            makeseletordata(args, split='test')
    if args.model_type == 'convertor' or args.model_type =='all':
        args.model = '../../../model/T5'
        if 'ruletaker' in args.direc:
            makeconvertordata_rt(args,split='train')
            makeconvertordata_rt(args, split='dev')
            makeconvertordata_rt(args, split='test')
        else:
            makeconvertordata(args, split='train')
            makeconvertordata(args, split='dev')
            makeconvertordata(args, split='test')
    if args.model_type == 'reasoner' or args.model_type =='all':
        args.model = '../../../model/T5'
        if 'ruletaker' in args.direc:
            makereasonerdata_rt(args, split='train')
            makereasonerdata_rt(args, split='dev')
            makereasonerdata_rt(args, split='test')
        else:
            makereasonerdata(args, split='train')
            makereasonerdata(args, split='dev')
            makereasonerdata(args, split='test')




