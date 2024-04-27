from nltk import *
import json
from tqdm import tqdm

def write_file(data, file):
    if file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

def read_file(file):
    with open(file, encoding='utf8') as f:
        return json.load(f)

def readdirec(filedirec):
    train_datas=read_file(filedirec+'/train_prover9CNFformat.json')
    dev_datas=read_file(filedirec+'/dev_prover9CNFformat.json')
    test_datas=read_file(filedirec+'/test_prover9CNFformat.json')
    return train_datas,dev_datas,test_datas

def lower(strB):
    return strB.lower()

def FOL2NL(mid_fol):
    if mid_fol=='':
        return ''
    l=mid_fol.split(' | ')
    if l[0][0]=='-':
        NL='no '+l[0][1:]
    else:
        NL=l[0]
    for i in range(1,len(l)):
        s=l[i]
        if s[0]=='-':
            NL+=' or no '+s[1:]
        else:
            NL+=' or '+s
    return lower(NL+'.')

def check(datas,direc,split='train'):
    read_expr = Expression.fromstring
    new_datas=[]
    for traindata in tqdm(datas):
        follist=[key for key in traindata['context2FOL'].keys()]
        gold_mid_proves = [(i, follist[i][7:-1]) if follist[i][:5]=='all x' else (i, follist[i]) for i in range(len(follist))]
        prover = Prover9Command([], [read_expr(key.strip('.')) for key in traindata['context2FOL'].keys()])
        ans=prover.prove()
        if ans==True:
            proof=prover.proof()
            proof=proof.split('\n')[18:-2]
            mid_prove=[]
            assum2origin={}

            for i in range(len(proof)):
                l1=proof[i].split(' ')
                num,annotation=l1[0],l1[-1][1:-2]
                middle=proof[i][len(num)+1:-len(annotation)-6]
                if middle[0]=='(':
                    middle=middle[1:-1]
                if annotation=='assumption':
                    for j in range(len(follist)):
                        a=logicequals(follist[j], middle)
                        if a:
                            assum2origin[num]=str(j)
                annotation = processops(annotation)
                mid_prove.append([num,middle,annotation])
                if annotation[0][:8]=='clausify':
                    clausnum=annotation[0].split('(')[1][:-1]
                    assum2origin[num]=assum2origin[clausnum]
            # process the proof derived from prover9 and check the proof
            gold_mid_proves=getmidprove(mid_prove,assum2origin,gold_mid_proves)
            assert gold_mid_proves is not False
            for i in range(len(gold_mid_proves)):
                mid_fol=gold_mid_proves[i][-1]
                s=FOL2NL(mid_fol)
                if len(gold_mid_proves[i])==4:
                    gold_mid_proves[i]=(gold_mid_proves[i][0],gold_mid_proves[i][1],gold_mid_proves[i][2],[s,mid_fol])
                else:
                    gold_mid_proves[i]=(gold_mid_proves[i][0],[s,mid_fol])
            new_data={}
            new_data['gold_mid_proves']=gold_mid_proves
            new_data["answer"]='no'
            new_data['id']=traindata['id']
            new_datas.append(new_data)

    write_file(new_datas, direc + '/'+split+'_withmidprove.json')
    print(len(new_datas))

def processops(s):
    ops=s.split(')')
    if '(' in ops[0]:
        ops[0]=ops[0]+')'
    for i in range(1,len(ops)-1):
        ops[i] = ops[i][1:]
        if '(' in ops[i]:
            ops[i]=ops[i]+')'
    if len(ops)!=1 and ')' in ops[-2]:
        ops=ops[:-1]
    ret_ops=[]
    for op in ops:
        if op[:5]!='merge':
            ret_ops.append(op)
    return ret_ops

def logicequals(str1,str2):
    l1 = str1.split(' | ')
    l2 = str2.split(' | ')
    num1 = [0 for _ in range(len(l1))]
    num2 = [0 for _ in range(len(l2))]
    for i in range(len(l1)):
        for j in range(len(l2)):
            if l1[i]==l2[j]:
                num1[i]=1
                num2[j]=1
    for i in num1:
        if i!=1:
            return False
    for i in num2:
        if i!=1:
            return False
    return True

def getmidprove(mid_proves,assum2origin,gold_mid_prove):
    # assumption clausify resolve copy merge back_unit_del unit_del ur
    # mid_prove list: [num,rule,annotation]
    # assum2origin dict[num of rule in prove] = num of rule in premise
    num2rule = {}
    num2rule_start=99999
    mid_rule_num=-1
    for mid_prove in mid_proves:
        ops=mid_prove[2]
        mid_rule=''
        ops_len=len(ops)
        for op_num,op in enumerate(ops):
            if op=='assumption' or op[:8]=='clausify' or op[:4]=='copy':
                num2rule[mid_prove[0]]=mid_prove[1]
                if op[:4]=='copy':
                    assum2origin[mid_prove[0]]=assum2origin[op.split('(')[1][:-1]]
                mid_rule=mid_prove[1]
                break

            if op[:7]=='resolve':
                l=op.split(',')
                assert len(l)==4
                n1=l[0].split('(')[1]
                n2=l[2]
                mid_rule=resolution(num2rule[n1],num2rule[n2])
                if n1 in assum2origin.keys():
                    n1=assum2origin[n1]
                if n2 in assum2origin.keys():
                    n2=assum2origin[n2]

                if op_num==ops_len-1:
                    assum2origin[mid_prove[0]]=str(len(gold_mid_prove))
                    num2rule[mid_prove[0]]=mid_rule
                else:
                    num2rule[str(num2rule_start)]=mid_rule
                    assum2origin[str(num2rule_start)]=str(len(gold_mid_prove))
                    mid_rule_num=num2rule_start
                    num2rule_start-=1
                gold_mid_prove.append((int(n1),int(n2),len(gold_mid_prove),mid_rule))

            if op[:2]=='ur':
                l = op.split(',')
                params=[]
                for i in range(len(l)):
                    if i==0:
                        s = l[0].split('(')[1]
                    elif i==len(l)-1:
                        s = l[-1][:-1]
                    else:
                        s=l[i]
                    if s.isdigit():
                        params.append(s)
                mid_rule=resolution(num2rule[params[0]],num2rule[params[1]])
                if params[0] in assum2origin.keys():
                    params[0]=assum2origin[params[0]]
                if params[1] in assum2origin.keys():
                    params[1]=assum2origin[params[1]]

                if len(params)==2 and op_num==ops_len-1:
                    assum2origin[mid_prove[0]] = str(len(gold_mid_prove))
                    num2rule[mid_prove[0]] = mid_rule
                else:
                    num2rule[str(num2rule_start)] = mid_rule
                    assum2origin[str(num2rule_start)] = str(len(gold_mid_prove))
                    mid_rule_num = num2rule_start
                    num2rule_start -= 1
                gold_mid_prove.append((int(params[0]), int(params[1]), len(gold_mid_prove), mid_rule))
                for i in range(2,len(params)):
                    mid_rule=resolution(mid_rule,num2rule[params[i]])
                    if params[i] in assum2origin.keys():
                        params[i] = assum2origin[params[i]]
                    if str(mid_rule_num) in assum2origin.keys():
                        mid_rule_num = assum2origin[str(mid_rule_num)]
                    if len(params) == i+1 and op_num == ops_len - 1:
                        assum2origin[mid_prove[0]] = str(len(gold_mid_prove))
                        num2rule[mid_prove[0]] = mid_rule
                        gold_mid_prove.append((int(mid_rule_num), int(params[i]), len(gold_mid_prove), mid_rule))
                    else:
                        num2rule[str(num2rule_start)] = mid_rule
                        assum2origin[str(num2rule_start)] = str(len(gold_mid_prove))
                        gold_mid_prove.append((int(mid_rule_num), int(params[i]), len(gold_mid_prove), mid_rule))
                        mid_rule_num = num2rule_start
                        num2rule_start -= 1

            if op[:13]=='back_unit_del':
                mid_rule_num=op[14:-1]
                mid_rule=num2rule[mid_rule_num]

            if op[:8]=='unit_del':
                num1=op.split(',')[1][:-1]
                mid_rule=resolution(mid_rule,num2rule[num1])
                if num1 in assum2origin.keys():
                    num1=assum2origin[num1]
                if str(mid_rule_num) in assum2origin.keys():
                    mid_rule_num=assum2origin[str(mid_rule_num)]
                if op_num==ops_len-1:
                    assum2origin[mid_prove[0]] = str(len(gold_mid_prove))
                    num2rule[mid_prove[0]] = mid_rule
                    gold_mid_prove.append((int(mid_rule_num), int(num1), len(gold_mid_prove), mid_rule))
                else:
                    num2rule[str(num2rule_start)] = mid_rule
                    assum2origin[str(num2rule_start)] = str(len(gold_mid_prove))
                    gold_mid_prove.append((int(mid_rule_num), int(num1), len(gold_mid_prove), mid_rule))
                    mid_rule_num = num2rule_start
                    num2rule_start -= 1

        if mid_rule!='' and logicequals(mid_rule,mid_prove[1])==False:
            print(mid_proves)
            print(mid_rule)
            print(mid_prove[1])
            raise('error')
        if mid_rule=='' and mid_prove[1]=='$F':
            return gold_mid_prove
        if (mid_rule!='' and mid_prove[1]=='$F') or (mid_rule=='' and mid_prove[1]!='$F'):
            return False
    return False

def resolution(str1,str2):
    l1 = str1.split(' | ')
    l2 = str2.split(' | ')
    l=resolist(l1,l2)
    if l==False:
        return False
    return ' | '.join(l)

def resolist(l1,l2):
    l=[]
    num1=[0 for _ in range(len(l1))]
    num2=[0 for _ in range(len(l2))]
    for i in range(len(l1)):
        for j in range(len(l2)):
            if (l1[i]==l2[j][1:] and l2[j][0]=='-') or (l2[j]==l1[i][1:] and l1[i][0]=='-'):
                num1[i]=1
                num2[j]=1
    if 1 not in num1:
        return False
    total_num=0
    for i in range(len(l1)):
        if num1[i]!=1:
            if l1[i] not in l:
                l.append(l1[i])
        else:
            total_num+=1
    for i in range(len(l2)):
        if num2[i]!=1:
            if l2[i] not in l:
                l.append(l2[i])
        else:
            total_num+=1
    if total_num!=2:
        return False
    return l

if __name__ == "__main__":
    direcs=['3sat/grounded_rule_lang/5var','3sat/grounded_rule_lang/8var',
            '3sat/grounded_rule_lang/10var','3sat/grounded_rule_lang/12var']
    for i,direc in enumerate(direcs):
        train_datas,dev_datas,test_datas=readdirec(direc)
        check(train_datas, direc, split='train')
        check(dev_datas, direc, split='dev')
        check(test_datas,direc,split='test')