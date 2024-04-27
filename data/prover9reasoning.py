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

def check(datas,predicate2NL,direc,split='train'):
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
            # prover9 may go wrong in few cases, so flag1 is utilized to indicate
            # whether an error occurs in the reasoning process
            flag1=0
            for i in range(len(proof)):
                l1=proof[i].split(' ')
                num,annotation=l1[0],l1[-1][1:-2]
                middle=proof[i][len(num)+1:-len(annotation)-6]
                if middle[0]=='(':
                    middle=middle[1:-1]
                if annotation=='assumption':
                    flag=0
                    for j in range(len(follist)):
                        if logicequals_expr(follist[j],middle):
                            assum2origin[num]=str(j)
                            flag=1
                            break
                    if flag==0:
                        flag1=1
                        break

                annotation = processops(annotation)
                mid_prove.append([num,middle,annotation])
                if annotation[0][:8]=='clausify':
                    clausnum=annotation[0].split('(')[1][:-1]
                    assum2origin[num]=assum2origin[clausnum]

            if flag1==0:
                # process the proof derived from prover9 and check the proof
                gold_mid_proves=getmidprove(mid_prove,assum2origin,gold_mid_proves)
                assert gold_mid_proves is not False
                for i in range(len(gold_mid_proves)):
                    mid_fol=gold_mid_proves[i][-1].strip('.')
                    s=FOL2NL(mid_fol,predicate2NL)
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

def reverse(fol):
    num=0
    if fol[:3]=='all':
        fol=fol[7:-1]
        num=1
    elif fol[:3]=='exists':
        fol=fol[10:-1]
        num=2

    fol='-'+fol if fol[0]!='-' else fol[1:]
    if num==1:
        return 'exists x ('+fol+')'
    elif num==2:
        return 'all x ('+fol+')'
    else:
        return fol

def FOL2NL(mid_fol,predicate2NL):
    if mid_fol=='':
        return ''
    mid_fol=mid_fol.split(' | ')
    if '(x)' in mid_fol[0]:
        sub='Everyone'
    else:
        sub=mid_fol[0].split('(')[1][:-1]

    if mid_fol[0].split('(')[0][0]=='-':
        s = sub+' is not ' + predicate2NL[mid_fol[0].split('(')[0][1:]]
    else:
        s = sub + ' is ' + predicate2NL[mid_fol[0].split('(')[0]]
    for i in range(1,len(mid_fol)):
        if mid_fol[i].split('(')[0][0] == '-':
            s = s + ' or not ' + predicate2NL[mid_fol[i].split('(')[0][1:]]
        else:
            s = s + ' or ' + predicate2NL[mid_fol[i].split('(')[0]]
    return s.strip('.')

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

def logicequals_expr(str1,str2):
    l1 = str1.split(' | ')
    l2 = str2.split(' | ')
    num1 = [0 for _ in range(len(l1))]
    num2 = [0 for _ in range(len(l2))]
    if l1[0][:7]=='all x (':
        l1[0] = l1[0][7:]
        l1[-1] = l1[-1][:-1]
    if l2[0][:7] == 'all x (':
        l2[0] = l2[0][7:]
        l2[-1] = l2[-1][:-1]
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
    # operations include assumption clausify resolve copy merge back_unit_del unit_del ur
    # mid_prove: list of [num,rule,annotation]
    # assum2origin: dict[num of rule in prove]=num of rule in premise
    num2rule = {}
    num2rule_start=99999
    mid_rule_num=-1
    for mid_prove in mid_proves:
        ops=mid_prove[2]
        mid_rule=''
        ops_len=len(ops)
        for op_num,op in enumerate(ops):
            # There must have been operations or 'copy' before merge, so there's no need to handle it absolutely
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

        if mid_rule!='' and logicequals_expr(mid_rule,mid_prove[1])==False:
            for mid in mid_proves:
                print(mid)
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
    for i in range(len(l1)):
        for j in range(len(l2)):
            d=unify(l1[i], l2[j])
            if d is not False:
                l1_tmp = l1[0:i] + l1[i + 1:]
                l2_tmp = l2[0:j] + l2[j + 1:]
                if d is True:
                    l = mergelist(l1_tmp, l2_tmp)
                else:
                    l1_tmp = substitude(l1_tmp, d)
                    l2_tmp = substitude(l2_tmp, d)
                    l = mergelist(l1_tmp, l2_tmp)
                if l is not False:
                    return ' | '.join(l)
    return False

def mergelist(l1,l2):
    l=[]
    for i in range(len(l1)):
        reverse='-'+l1[i] if l1[i][0]!='-' else l1[i][1:]
        if reverse in l:
            return False
        if l1[i] not in l:
            l.append(l1[i])
    for i in range(len(l2)):
        reverse = '-' + l2[i] if l2[i][0] != '-' else l2[i][1:]
        if reverse in l:
            return False
        if l2[i] not in l:
            l.append(l2[i])
    return l

def unify(s1,s2):
    predicate1 = s1.split('(')[0]
    predicate2 = s2.split('(')[0]
    liter1 = s1.split('(')[1][:-1]
    liter2 = s2.split('(')[1][:-1]
    if (predicate1[0]=='-' and predicate2[0]!='-' and predicate1[1:]==predicate2) or \
            predicate1[0]!='-' and predicate2[0]=='-' and predicate1==predicate2[1:]:
        if liter1=='x':
            if liter2=='x':
                return {'x':'x'}
            else:
                return {'x':liter2}
        elif liter2=='x':
            return {'x':liter1}
        elif liter1==liter2:
            return True
        else:
            return False
    return False

def substitude(l,d):
    for i in range(len(l)):
        l[i]=l[i].replace('(x)','('+d['x']+')')
    return l

if __name__ == "__main__":
    predicate2NL=read_file('3sat/predicate2NL.json')
    direcs=['3sat/relative_clause/rclause_16_20_21','3sat/relative_clause/rclause_25_28_32',
            '3sat/relative_clause/rclause_35_42_48','3sat/relative_clause/rclause_60_64_70']
    for i,direc in enumerate(direcs):
        train_datas,dev_datas,test_datas=readdirec(direc)
        check(train_datas, predicate2NL, direc, split='train')
        check(dev_datas, predicate2NL, direc, split='dev')
        check(test_datas,predicate2NL,direc,split='test')


