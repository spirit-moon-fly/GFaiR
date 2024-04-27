import json
import re
from tqdm import tqdm

def readdirec(filedirec):
    with open(filedirec+'/train.jsonl') as f:
        train_datas = f.readlines()
    with open(filedirec+'/dev.jsonl') as f:
        dev_datas = f.readlines()
    with open(filedirec+'/test.jsonl') as f:
        test_datas = f.readlines()
    return train_datas,dev_datas,test_datas

def write_file(data, file):
    if file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

def upper(strB):
    return strB[0].upper() + strB[1:].lower()

def lower(strB):
    return strB.lower()

def convert(context,s1,s2):
    # (.*) (is|are) (.*)
    s1=lower(s1).replace('the ','')
    if s1=='something' or s1=='everything':
        subject='x'
    else:
        subject=s1
    #print(subject)
    if s2[:3]=='not':
        fol='-'+upper(s2[4:-1])+'('+lower(subject)+')'
    else:
        fol= upper(s2[:-1]) + '(' + lower(subject) + ')'\

    if s1=='something':
        fol='exists x ('+fol+')'
    elif s1=='everything':
        fol='all x ('+fol+')'

    return fol

def convert1(context,s1,s2,s3):
    #  (.*) (need|needs|eat|eats|see|sees|visit|visits|like|likes|chase|chases) (.*)'
    if s2[-1]=='s':
        s2=s2[:-1]
    if len(s1)>8 and s1[-8:]=='does not':
        fol='-'+upper(s2)+'('+lower(s1[:-9].split(' ')[1]+','+s3.split(' ')[1][:-1])+')'
    elif len(s1.split(' '))==2:
        fol = upper(s2) + '(' + lower(s1.split(' ')[1] + ',' + s3.split(' ')[1][:-1]) + ')'
    else:
        print(s1)
        print(s2)
        print(s3)
        print(context)
        raise('error')
    return fol

def matchfact(fact):
    pattern = '(.*) (is|are) (.*)'
    pattern1 = '(.*) (need|needs|eat|eats|see|sees|visit|visits|like|likes|chase|chases) (.*)'
    m = re.match(pattern, fact, re.I)
    if m:
        assert len(fact.split(' ')) <= 5
        FOL = convert(fact, m.group(1),m.group(3))
    else:
        m1 = re.match(pattern1, fact, re.I)
        if m1:
            #print(fact)
            assert len(fact.split(' ')) <= 7
            FOL = convert1(fact, m1.group(1), m1.group(2), m1.group(3))
        else:
            #print(fact)
            raise('error')
    return lower(fact).replace('the ',''),FOL

def process(l):
    if len(l[2])==2:
        fol=upper(l[1])+'('+lower(l[2][0])+','+lower(l[2][1])+')'
        if l[2][0]=='x':
            if l[0]==0:
                nl = 'everything does not ' + lower(l[1])+' ' + lower(l[2][1])
            else:
                nl = 'everything ' + lower(l[1]) + 's ' + lower(l[2][1])
        elif l[2][1]=='x':
            if l[0] == 0:
                nl = lower(l[2][0]) + ' does not ' + lower(l[1])+' everything'
            else:
                nl = lower(l[2][0]) + ' ' + lower(l[1]) + 's everything'
        else:
            if l[0] == 0:
                nl = lower(l[2][0]) + ' does not ' + lower(l[1]) + ' ' + lower(l[2][1])
            else:
                nl = lower(l[2][0]) + ' ' + lower(l[1]) + 's ' + lower(l[2][1])
    else:
        fol = upper(l[1]) + '(' + lower(l[2][0]) + ')'
        if l[2][0]=='x':
            if l[0]==0:
                nl = 'everything is not ' + lower(l[1])
            else:
                nl = 'everything is ' + lower(l[1])
        else:
            if l[0] == 0:
                nl = lower(l[2][0]) + ' is not '+lower(l[1])
            else:
                nl = lower(l[2][0]) + ' is '+lower(l[1])
    if l[0]==0:
        fol = '-'+ fol
    return fol,nl.strip('.')

def rematch(datas):
    convert_datas = []
    for data in tqdm(datas):
        fols=[]
        data = json.loads(data)
        context = data['context'].split(". ")
        r,q=context[-1].split(' $query$ ')
        context[-1]=r
        id = data['id'].split('_')[0]
        if "orig_rules" not in data['meta'].keys():
            d=convert_datas[-1]
            assert d['id'].split('_')[0]==id
            context2FOL=d['context2FOL'].copy()
            CNF_NL, FOL = matchfact(q)
            convert_datas.append({'context2FOL':context2FOL,"answer":data["answer"],"id":data['id'],"fols":d['fols'].copy(),
                                  'query':{FOL.strip('.'): [CNF_NL.strip('.')+'.',q.strip('.')+'.']}})
            continue
        orig_rules = data['meta']["orig_rules"]
        context2FOL = dict()
        facts, rules, rule_fact = [], [], []
        for i in range(len(context)):
            context[i] = context[i] + '.'
            flag=0
            for j in range(len(orig_rules)):
                if orig_rules[j][-1]+'.' == context[i]:
                    rules.append((orig_rules[j][0],orig_rules[j][1]))
                    rule_fact.append(0)
                    flag=1
                    break
            if flag==0:
                facts.append(context[i])
                rule_fact.append(1)
        rule_fols,fact_fols=[],[]
        for rule in rules:
            if len(rule[0])==3:
                FOL0,NL0  = process(rule[0][0])
                FOL1,NL1  = process(rule[0][1])
                FOL2,NL2  = process(rule[0][2])
                FOL = FOL0 + ' | ' + FOL1 +' | ' + FOL2
                if '(x' in FOL or 'x)' in FOL:
                    FOL='all x ('+ FOL+')'
                CNF_NL = NL0 + ' or ' + NL1 + ' or ' + NL2
                if 'thing ' in rule[1] or 'something ' in rule[1] or 'things ' in rule[1] or 'everything ' in rule[1] or 'anything ' in rule[1]:
                    CNF_NL=CNF_NL.replace('everyone','everything')
            else:
                FOL0, NL0 = process(rule[0][0])
                FOL1, NL1 = process(rule[0][1])
                FOL = FOL0 + ' | ' + FOL1
                if '(x' in FOL or 'x)' in FOL:
                    FOL='all x ('+ FOL+')'
                CNF_NL = NL0 + ' or ' + NL1
                if 'thing ' in rule[1] or 'something ' in rule[1] or 'things ' in rule[1] or 'everything ' in rule[1] or 'anything ' in rule[1]:
                    CNF_NL=CNF_NL.replace('everyone','everything')
            context2FOL[FOL.strip('.')] = [CNF_NL.strip('.')+'.', rule[1].strip('.')+'.']
            rule_fols.append(FOL.strip('.'))

        for fact in facts:
            CNF_NL,FOL=matchfact(fact)
            context2FOL[FOL.strip('.')] = [CNF_NL.strip('.')+'.', fact.strip('.')+'.']
            fact_fols.append(FOL.strip('.'))

        CNF_NL, FOL = matchfact(q)
        factnum,rulenum=0,0
        for i in range(len(rule_fact)):
            if rule_fact[i]==0:
                fols.append(rule_fols[rulenum])
                rulenum+=1
            else:
                fols.append(fact_fols[factnum])
                factnum += 1
        convert_data={'context2FOL': context2FOL, "answer": data["answer"], "id": data["id"],"fols":fols,
                              'query': {FOL.strip('.'): [CNF_NL.strip('.')+'.',q.strip('.')+'.']}}
        if 'depth' in data.keys():
            convert_data['dep']=data['depth']
        convert_datas.append(convert_data)
    return convert_datas

def convertis(s1,reverse=False):
    l=s1.split(' ')
    if l[0] in ['someone','something','everyone','everything','anyone','anything','it','they']:
        sub='x'
    else:
        sub=lower(l[0])
    if len(l) == 4:
        fol = '-'+upper(l[3])+'('+sub+')'
    elif len(l) == 3:
        fol = upper(l[2]) + '(' + sub + ')'
    else:
        raise('error')
    if reverse is True:
        fol = '-' +fol if fol[0]!='-' else fol[1:]
        l=[l[0],l[1],'not',l[2]] if fol[0]=='-' else [l[0],l[1],l[3]]
        NL = ' '.join(l)
    else:
        NL=s1
    return NL,fol,sub

def convert2(context, s1, s2, s3):
    # if (.*) and (.*) then (.*)
    NL1, fol1, sub1 = convertis(s1,reverse=True)

    if len(s2.split(' '))>2:
        assert s2.split(' ')[0]==s1.split(' ')[0] and s2.split(' ')[1]=='is' and len(s2.split(' '))<=4
        l2=s2.split(' ')[2:]
    else:
        l2=s2.split(' ')
    if l2[0]=='not':
        fol2=upper(l2[1])+'('+lower(sub1)+')'
        NL2=sub1+' is '+l2[1] if sub1!='x' else 'everything is '+l2[1]
    else:
        fol2='-'+upper(l2[0])+'('+lower(sub1)+')'
        NL2=sub1+' is not '+l2[0] if sub1!='x' else 'everything is not '+l2[0]

    l3=s3.split(' ')
    if len(l3)>2:
        assert (l3[0]==s1.split(' ')[0] or (l3[0] in ['it','they'] and '(x)' in fol1)) and l3[1] == 'is' and len(l3)<=4
        l3 = l3[2:]
    if l3[0]=='not':
        fol3 = '-'+upper(l3[1])+'('+lower(sub1)+')'
        NL3 = sub1+' is not '+l3[1] if sub1!='x' else 'everything is not '+l3[1]
    else:
        fol3 = upper(l3[0])+'('+lower(sub1)+')'
        NL3 = sub1+' is '+l3[0] if sub1!='x' else 'everything is '+l3[0]
    NL=NL1+' or '+NL2 + ' or ' + NL3
    fol=fol1+' | '+fol2+' | '+fol3
    if '(x)' in fol:
        fol = 'all x (' + fol + ')'
    return NL,fol

def convert3(context, s1, s2):
    # if (.*) then (.*)
    NL1, fol1, sub1 = convertis(s1, reverse=True)
    l2=s2.split(' ')
    if len(l2) > 2:
        assert (l2[0] == s1.split(' ')[0] or (l2[0] in ['it','they'] and '(x)' in fol1)) and l2[1] == 'is' and len(l2) <= 4
        l2 = l2[2:]

    if l2[0] == 'not':
        fol2 = '-' +upper(l2[1]) + '(' + lower(sub1) + ')'
        NL2 = sub1+' is not ' +l2[1] if sub1!='x' else 'everything is not '+l2[1]
    else:
        fol2 = upper(l2[0]) + '(' + lower(sub1) + ')'
        NL2 = sub1 + ' is '+l2[0] if sub1!='x' else 'everything is '+l2[0]
    NL=NL1+' or '+NL2
    fol = fol1 + ' | ' + fol2
    if '(x)' in fol:
        fol = 'all x (' + fol + ')'
    return NL,fol

def convert4(context, s1, s2):
    # (.*) things is (.*)
    l1=s1.split(', ')
    assert len(l1)<=2
    if len(l1)==2:
        fol1='-'+upper(l1[0])+'(x) | -'+upper(l1[1])+'(x)'
        NL1= 'everything is not '+l1[0]+' or everything is not '+l1[1]
    else:
        fol1 = '-' + upper(l1[0]) + '(x)'
        NL1 = 'everything is not ' + l1[0]
    l2=s2.split(' ')
    if l2[0] == 'not':
        fol2 = '-' +upper(l2[1]) + '(x)'
        NL2 = 'everything is not ' +l2[1]
    else:
        fol2 = upper(l2[0]) + '(x)'
        NL2 = 'everything is '+l2[0]
    fol=fol1+' | '+fol2
    NL=NL1+' or '+NL2
    if '(x)' in fol:
        fol = 'all x (' + fol + ')'
    return NL,fol

def convert5(context, s1, s2, s3):
    # something is (.*) or (.*) or (.*)
    fol1 = upper(s1)+'(x)' if s1[:4]!='not ' else '-' +upper(s1[4:]) + '(x)'
    NL1 ='something is '+s1
    fol2 = upper(s2) + '(x)' if s2[:4]!='not ' else '-' +upper(s2[4:]) + '(x)'
    NL2 = 'something is ' + s2
    fol3 = upper(s3) + '(x)' if s3[:4]!='not ' else '-' +upper(s3[4:]) + '(x)'
    NL3 = 'something is ' + s3
    fol = 'exists x ('+fol1+' | '+fol2+' | '+fol3+')'
    NL = NL1+' or '+NL2+' or '+NL3
    return NL,fol

def convert6(context, s1, s2):
    # something is (.*) or (.*)
    fol1 = upper(s1)+'(x)' if s1[:4]!='not ' else '-' +upper(s1[4:]) + '(x)'
    NL1 ='something is '+s1
    fol2 = upper(s2) + '(x)' if s2[:4]!='not ' else '-' +upper(s2[4:]) + '(x)'
    NL2 = 'something is ' + s2
    fol = 'exists x ('+fol1+' | '+fol2+')'
    NL = NL1+' or '+NL2
    return NL,fol

def convert7(context, s1, s2):
    # (everything|something) is (.*)
    fol1 = upper(s2) + '(x)' if s2[:4] != 'not ' else '-' + upper(s2[4:]) + '(x)'
    if s1=='everything':
        fol= 'all x ('+fol1+')'
    else:
        fol = 'exists x (' + fol1 + ')'
    return context,fol

def convert8(context, s1, s2):
    # (.*) is (.*)
    if s2[:3]=='not':
        fol='-'+upper(s2[4:])+'('+lower(s1)+')'
    else:
        fol = upper(s2) + '(' + lower(s1) + ')'
    if '(x)' in fol:
        fol = 'all x (' + fol + ')'
    return context,fol

def rematchhardRT(direc,split):
    with open(direc+'/'+split+'.jsonl') as f:
        datas = f.readlines()
    pattern2 = 'if (.*) and (.*) then (.*)'
    pattern3 = 'if (.*) then (.*)'
    pattern4 = '(.*) things is (.*)'
    pattern5 = "something is (.*) or (.*) or (.*)"
    pattern6 = "something is (.*) or (.*)"
    pattern7 = "(everything|something) is (.*)"
    pattern8 = '(.*) is (.*)'

    convert_datas = []
    num = 0
    for data in tqdm(datas):
        fols = []
        data = json.loads(data)
        context = data['context'].split(". ")
        r, q = context[-1].split(' $query$ ')
        context[-1] = r
        context2FOL = dict()
        id = data["id"]
        num+=1
        for i in range(len(context)):
            ci= lower(context[i][:]).replace('all ','').replace('are','is').replace('the ','')
            m2 = re.match(pattern2, ci, re.I)
            if m2:
                CNF_NL,FOL = convert2(ci, m2.group(1),m2.group(2),m2.group(3))
            else:
                m3 = re.match(pattern3, ci, re.I)
                if m3:
                    CNF_NL,FOL = convert3(ci, m3.group(1), m3.group(2))
                else:
                    m4=re.match(pattern4, ci, re.I)
                    if m4:
                        CNF_NL,FOL = convert4(ci, m4.group(1), m4.group(2))
                    else:
                        m5 = re.match(pattern5, ci, re.I)
                        if m5:
                            CNF_NL, FOL = convert5(ci, m5.group(1), m5.group(2), m5.group(3))
                        else:
                            m6 = re.match(pattern6, ci, re.I)
                            if m6:
                                CNF_NL, FOL = convert6(ci, m6.group(1), m6.group(2))
                            else:
                                m7 = re.match(pattern7, ci, re.I)
                                if m7:
                                    CNF_NL, FOL = convert7(ci, m7.group(1), m7.group(2))
                                else:
                                    m8 = re.match(pattern8, ci, re.I)
                                    if m8:
                                        CNF_NL,FOL = convert8(ci, m8.group(1), m8.group(2))
                                    else:
                                        raise('error')

            if 'exists ' not in FOL and 'something ' in CNF_NL:
                CNF_NL=CNF_NL.replace('something ', 'everything ')
            if 'someone ' in CNF_NL or 'anyone ' in CNF_NL or 'something ' in CNF_NL or 'anything ' in CNF_NL:
                CNF_NL=CNF_NL.replace('someone ','everything ').replace('anyone ','everything ').replace('anything ', 'everything ')
            assert 'it ' not in CNF_NL or 'rabbit ' in CNF_NL
            if 'they ' in CNF_NL or 'everyone ' in CNF_NL:
                CNF_NL=CNF_NL.replace('they ','everything ').replace('everyone ', 'everything ')

            context2FOL[FOL.strip('.')] = [CNF_NL.strip('.') + '.', context[i].strip('.') + '.']
            fols.append(FOL.strip('.'))

        CNF_NL, FOL = matchfact(q)
        convert_datas.append({'context2FOL': context2FOL, "answer": data["answer"],'id':id,'fols':fols,
                              'query': {FOL.strip('.'): [CNF_NL.strip('.') + '.', q.strip('.') + '.']}})
    write_file(convert_datas,direc+'/'+split+'_prover9CNFformat.json')

direc='myhard_ruletaker'
rematchhardRT(direc,'train')
rematchhardRT(direc,'dev')
rematchhardRT(direc,'test')
rematchhardRT('hard_ruletaker','test')
rematchhardRT('hard_ruletaker','dev')
rematchhardRT('ruletaker_exist','train')
rematchhardRT('ruletaker_exist','dev')
rematchhardRT('ruletaker_exist','test')

train_datas,dev_datas,test_datas=readdirec('ruletaker_3ext_sat')
convert_train_data=rematch(train_datas)
write_file(convert_train_data,'ruletaker_3ext_sat/train_prover9CNFformat.json')
convert_dev_data=rematch(dev_datas)
write_file(convert_dev_data,'ruletaker_3ext_sat/dev_prover9CNFformat.json')
convert_test_data=rematch(test_datas)
write_file(convert_test_data,'ruletaker_3ext_sat/test_prover9CNFformat.json')

with open('depth-5/test.jsonl') as f:
    test_datas = f.readlines()
convert_test_data=rematch(test_datas)
write_file(convert_test_data,'depth-5/test_prover9CNFformat.json')






