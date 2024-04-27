from logic1 import *
import json
from tqdm import tqdm
import func_timeout
import re
import multiprocessing
import argparse
import pickle

def read_file(file):
    with open(file, encoding='utf8') as f:
        return json.load(f)

def read_pkl(file):
    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'rb') as f:
            return pickle.load(f)

def readdirec(filedirec):
    train_datas=read_file(filedirec+'/FOLtrain.json')
    dev_datas=read_file(filedirec+'/FOLdev.json')
    test_datas=read_file(filedirec+'/FOLtest.json')
    return train_datas,dev_datas,test_datas

def write_file(data, file):
    if any([file.endswith(ext) for ext in ['pkl', 'pickle', 'pck', 'pcl']]):
        with open(file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def reverseq(q):
    if q.isa(Not):
        return q.arg
    elif q.isa(Atom) or q.isa(Exists) or q.isa(Forall):
        return Not(q)
    else:
        raise('error')

def reverse(context,s1,s2,s3):
    # (.*) (is|are) (.*)
    if s3[:3]=='not':
        s3=s3[4:]
    else:
        s3='not ' + s3

    if s1=='Everything':
        s1='something'
    elif s1=='Something':
        s1='everything'
    return s1+' '+s2+' '+s3

def reverse1(context,s1,s2,s3):
    #  (.*) (need|needs|eat|eats|see|sees|visit|visits|like|likes|chase|chases) (.*)'
    if len(s1)>8 and s1[-8:]=='does not':
        s1=s1[:-9]
        if s2[-1] != 's':
            s2=s2+'s'
    elif len(s1.split(' '))==2:
        s1=s1+' does not'
        if s2[-1] == 's':
            s2 = s2[:-1]
    else:
        raise('error')
    return s1+' '+s2+' '+s3

def reversefact(fact):
    pattern = '(.*) (is|are) (.*)'
    pattern1 = '(.*) (need|needs|eat|eats|see|sees|visit|visits|like|likes|chase|chases) (.*)'
    m = re.match(pattern, fact, re.I)
    if m:
        assert len(fact.split(' ')) <= 5
        NL = reverse(fact, m.group(1), m.group(2), m.group(3))
    else:
        m1 = re.match(pattern1, fact, re.I)
        if m1:
            assert len(fact.split(' ')) <= 7
            NL = reverse1(fact, m1.group(1), m1.group(2), m1.group(3))
        else:
            print(fact)
            raise('error')
    return NL.lower().replace('the ',''),NL

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

def getprove(deriv,l,num):
    if len(deriv.children)!=0:
        if deriv.num == -1:
            a,num1=getprove(deriv.children[0],l,num)
            b,num2=getprove(deriv.children[1],l,num1)
            deriv.num=num2
            num2+=1
            num=num2
            ret = flattenOr(deriv.form) if deriv.form is not AtomFalse else [AtomFalse]
            ret.append(deriv.num)
            ret = [str(i) for i in ret]
            l.append([int(a[-1]),int(b[-1]),int(ret[-1]),[FOL2NL(ret[:-1]),deriv.form]])
        else:
            ret = flattenOr(deriv.form) if deriv.form is not AtomFalse else [AtomFalse]
            ret.append(deriv.num)
            ret = [str(i) for i in ret]
    else:
        ret = flattenOr(deriv.form) if deriv.form is not AtomFalse else [AtomFalse]
        ret.append(deriv.num)
        ret = [str(i) for i in ret]
    return ret,num

def check(datas,split_num,direc,split):
    d={'true':ENTAILMENT,'false':CONTRADICTION,'unknown':NEURAL}
    skip_num=0
    converteddata=[]
    conflictnum=0
    qs=[]
    preid=''
    for traindata in tqdm(datas):
        standardlization = ToCNFRule()
        KB=createResolutionKB()
        result=d[traindata["answer"]]
        q=traindata['query']
        ql=[key for key in q.keys()]
        if result==NEURAL:
            continue
        q=eval(ql[0])
        # KB first infer the entailment relationship, the reverse operation is to accelerate reasoning speed
        q=q if result==ENTAILMENT else reverseq(q)
        id = traindata['id']
        id = id.split('_')[0] if '-' in id else id
        if id == preid:
            if q in qs:
                continue
            else:
                qs.append(q)
        else:
            qs = [q]
            preid = id

        result=ENTAILMENT if result==CONTRADICTION else result
        num = 0
        num2context={}
        for key in traindata['context2FOL'].keys():
            num2context[num]=[traindata['context2FOL'][key][0],OrList(standardlization.applyRule(eval(key)))]
            KB.tell(eval(key),num)
            num += 1
        try:
            f,KBres = KB.ask(q,num)
            q_origin=traindata['query']
            if d[traindata["answer"]] == ENTAILMENT:
                # If the answer is entailment, then the representation of q in KB will be the negation of q
                q_NL = [reversefact(q_origin[ql[0]][1])[0], OrList(standardlization.applyRule(eval('Not(' + ql[0] + ')')))]
            else:
                q_NL = [q_origin[ql[0]][0], OrList(standardlization.applyRule(eval(ql[0])))]
            num2context[num] = q_NL
            if KBres.status != result:
                print(result, KBres.status)
                print(ql[0])
                conflictnum+=1
                print('conflictnum',conflictnum)
                deriv = None
                for key in KB.derivations.keys():
                    if key == AtomFalse:
                        deriv = KB.derivations[key]
                if deriv != None:
                    l = [[i, num2context[i]] for i in range(len(num2context))]
                    getprove(deriv, l, num + 1)
                    for prove in l:
                        print(prove)
                else:
                    l = [[i, num2context[i]] for i in range(len(num2context))]
                    for prove in l:
                        print(prove)
            else:
                deriv = None
                for key in KB.derivations.keys():
                    if key == AtomFalse:
                        deriv = KB.derivations[key]
                if deriv != None:
                    l = [[i,num2context[i]] for i in range(len(num2context))]
                    # get intermediate reasoning steps in natural language
                    getprove(deriv, l, num + 1)
                    for i in l:
                        assert(i[-1][1] is False or isinstance(i[-1][1],Expression))
                    converteddata.append({"gold_mid_proves": l,"id":traindata['id']})
                else:
                    raise('error')
        except func_timeout.exceptions.FunctionTimedOut:
            skip_num+=1
    print('***********'+str(skip_num)+'************')
    print('***********' + str(conflictnum) + '************')
    write_file(converteddata,direc+'/'+split+'_'+str(split_num)+'_withmidprove.pkl')

def parallel(direc,split):
    datas = read_file(direc+'/FOL'+split+'.json')
    n = 40
    procs = []
    n_cpu = min(multiprocessing.cpu_count(),n)
    chunk_size = int(n / n_cpu)
    shape=len(datas)
    for i in range(0, n_cpu):
        min_i = chunk_size * i
        if i < n_cpu - 1:
            max_i = chunk_size * (i + 1)
        else:
            max_i = n
        digits = []
        for digit in range(min_i, max_i):
            digits.append(digit)
        print("digits:",digits)
        print("CPU:",i)
        procs.append(multiprocessing.Process(target=check,args=(datas[shape*i//n_cpu:shape*(i+1)//n_cpu],i,direc,split)))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    return n_cpu,shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direc", default='ruletaker_3ext_sat', type=str)
    parser.add_argument("--split", default='train', choices=['dev', 'train','test'],
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()
    direc=args.direc
    split=args.split
    print(direc,split)
    n,shape=parallel(direc,split)

    print('********************merging********************')
    datas=[]
    for i in range(n):
        tmp=read_pkl(direc+'/'+split+'_'+str(i)+'_withmidprove.pkl')
        for data in tmp:
            datas.append(data)
    print(len(datas),shape)
    write_file(datas,direc+'/'+split+'_withmidprove.pkl')
