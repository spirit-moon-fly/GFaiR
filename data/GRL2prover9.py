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

def convert1(m1,m2,m3):
    # If (.*) and (.*) then (.*)
    if m1[:2]=='no':
        fol=upper(m1[3:])
        nl=m1[3:]
    else:
        fol='-'+upper(m1)
        nl='no '+m1
    fol += ' | '
    nl += ' or '
    if m2[:2]=='no':
        fol+=upper(m2[3:])
        nl+=m2[3:]
    else:
        fol+='-'+upper(m2)
        nl+='no '+m2
    fol += ' | '
    nl += ' or '
    if m3[:2]=='no':
        fol += '-'+upper(m3[3:])
        nl += 'no '+m3[3:]
    else:
        fol += upper(m3)
        nl += m3
    nl+='.'
    return lower(nl),fol

def rematch(datas):
    pattern1 = 'If (.*) and (.*) then (.*)'
    convert_datas = []
    for data in tqdm(datas):
        fols=[]
        data=json.loads(data)
        context=data['context'].split(". ")
        for i in range(len(context)):
            context[i]=context[i]+'.'
        context2FOL=dict()
        for i in range(len(context)):
            m = re.match(pattern1,context[i],re.I)
            if m:
                CNF_NL,FOL=convert1(m.group(1),m.group(2),m.group(3).strip('.'))
            else:
                print(context[i])
                raise('error')
            context2FOL[FOL]=[CNF_NL,context[i]]
            fols.append(FOL)
        convert_datas.append({'context2FOL':context2FOL,"answer":data["answer"],"id":data["id"],"fols":fols})
    return convert_datas

direcs=['3sat/grounded_rule_lang/5var','3sat/grounded_rule_lang/8var',
        '3sat/grounded_rule_lang/10var','3sat/grounded_rule_lang/12var']
for direc in direcs:
    train_datas,dev_datas,test_datas=readdirec(direc)
    convert_train_data=rematch(train_datas)
    convert_dev_data=rematch(dev_datas)
    convert_test_data=rematch(test_datas)
    write_file(convert_train_data,direc+'/train_prover9CNFformat.json')
    write_file(convert_dev_data,direc+'/dev_prover9CNFformat.json')
    write_file(convert_test_data,direc+'/test_prover9CNFformat.json')


