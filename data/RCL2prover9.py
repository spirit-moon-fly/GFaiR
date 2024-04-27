import json
import re

def readdirec(filedirec):
    with open(filedirec+'/train.jsonl') as f:
        train_datas = f.readlines()
    with open(filedirec+'/dev.jsonl') as f:
        dev_datas = f.readlines()
    with open(filedirec+'/test.jsonl') as f:
        test_datas = f.readlines()
    return train_datas,dev_datas,test_datas

def read_file(file):
    with open(file, encoding='utf8') as f:
        return json.load(f)

def write_file(data, file):
    if file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

def upper(strB):
    return strB[0].upper() + strB[1:].lower()

def lower(strB):
    return strB.lower()

def convertis(str1):
    l=str1.split(' ')
    if len(l)==1:
        return upper(l[0])+'(@)'
    if len(l)==2:
        if l[0]=='not':
            return '-'+upper(l[1])+'(@)'
        else:
            if l[0]=='a' or l[0]=='an':
                return upper(l[1])+'(@)'
            else:
                raise("error")
    if len(l)==3:
        return '-'+upper(l[2])+'(@)'
    else:
        raise 'words longer than 3'

def convertnoun(str1):
    l=str1.split(' ')
    if len(l)==1:
        return upper(l[0])+'(@)'
    if len(l)==2:
        if l[1]=='person':
            return upper(l[0])+'(@)'
        raise('error')
    else:
        raise('error')

def convert(context,m1,m2,m3):
    # Everyone who is (.*) and (.*) is (.*)
    atom1 = convertis(m1)
    atom2 = convertis(m2)
    atom3 = convertis(m3)
    atom1=  atom1.replace('(@)','(x)')
    atom2 = atom2.replace('(@)', '(x)')
    atom3 = atom3.replace('(@)', '(x)')

    if atom1[0]=='-':
        atom1 = atom1[1:]
        m1 = 'is ' + m1[4:]
    else:
        atom1 = '-' + atom1
        m1 = 'is not ' + m1

    if atom2[0]=='-':
        atom2 = atom2[1:]
        m2 = m2[4:]
    else:
        atom2 = '-' + atom2
        m2 = 'not ' + m2
    fol = 'all x ('+atom1 + ' | ' + atom2 + ' | ' + atom3+')'
    nl = 'Everyone ' + m1 + ' or ' + m2 + ' or ' + m3+ '.'
    return nl,fol

def convert1(context,m1,m2,m3,m4,exist_predicate2NL):
    # (Every|No) (.*) who is (.*) is (.*)
    atom2 = convertnoun(m2)
    atom3 = convertis(m3)
    atom4 = convertis(m4)
    atom2 = atom2.replace('(@)', '(x)')
    atom3 = atom3.replace('(@)', '(x)')
    atom4 = atom4.replace('(@)', '(x)')

    if atom3[0]=='-':
        atom3 = atom3[1:]
        m3 = m3[4:]
    else:
        atom3 = '-' + atom3
        m3 = 'not ' + m3

    if atom2[0]=='-':
        raise('error')
    else:
        if 'person' in m2:
            m2 = 'is not ' + m2.split()[0]
        else:
            m2 = 'is not ' + exist_predicate2NL[upper(m2)]
        atom2 = '-' + atom2

    if m1=='No':
        if atom4[0]=='-':
            atom4 = atom4[1:]
            m4 = m4[4:]
        else:
            atom4='-'+atom4
            m4 = 'not ' + m4
    fol = 'all x (' + atom2 + ' | ' + atom3 + ' | ' + atom4 + ')'
    nl = 'Everyone ' + m2 + ' or ' + m3 + ' or ' + m4+'.'
    return nl,fol


def convert2(context,m1,m2,m3,m4):
    # (.*) is (.*) or (.*) or (.*)
    atom2 = convertis(m2)
    atom3 = convertis(m3)
    atom4 = convertis(m4)
    atom2 = atom2.replace('(@)', '('+lower(m1)+')')
    atom3 = atom3.replace('(@)', '('+lower(m1)+')')
    atom4 = atom4.replace('(@)', '('+lower(m1)+')')
    result = atom2 + ' | ' + atom3 + ' | ' + atom4
    return context,result

def rematch(datas,exist_predicate2NL):
    pattern = 'Everyone who is (.*) and (.*) is (.*)'
    pattern1 = '(Every|No) (.*) who is (.*) is (.*)'
    pattern2 = '(.*) is (.*) or (.*) or (.*)'
    num=0
    convert_datas=[]
    for data in datas:
        fols=[]
        num+=1
        data=json.loads(data)
        context=data['context'].split(". ")
        for i in range(len(context)):
            context[i]=context[i]+'.'
        context2FOL=dict()
        for i in range(len(context)):
            m = re.match(pattern,context[i],re.I)
            if m:
                CNF_NL,FOL=convert(context[i],m.group(1),m.group(2),m.group(3)[:-1])
            else:
                m1 = re.match(pattern1, context[i], re.I)
                if m1:
                    CNF_NL,FOL = convert1(context[i],m1.group(1),m1.group(2),m1.group(3),m1.group(4)[:-1],exist_predicate2NL)
                else:
                    m2 = re.match(pattern2, context[i], re.I)
                    if m2:
                        CNF_NL,FOL = convert2(context[i],m2.group(1),m2.group(2),m2.group(3),m2.group(4)[:-1])
                    else:
                        print(context[i])
                        raise(context[i])
            fols.append(FOL)
            context2FOL[FOL]=[CNF_NL,context[i]]
        convert_datas.append({'context2FOL':context2FOL,"answer":data["answer"],"id":data["id"],'fols':fols})
    return convert_datas



predicate2NL=read_file('3sat/predicate2NL.json')
direcs=['3sat/relative_clause/rclause_16_20_21','3sat/relative_clause/rclause_25_28_32',
        '3sat/relative_clause/rclause_35_42_48','3sat/relative_clause/rclause_60_64_70']
for direc in direcs:
    train_datas,dev_datas,test_datas=readdirec(direc)
    convert_train_data=rematch(train_datas,predicate2NL)
    convert_dev_data=rematch(dev_datas,predicate2NL)
    convert_test_data=rematch(test_datas,predicate2NL)
    write_file(convert_train_data,direc+'/train_prover9CNFformat.json')
    write_file(convert_dev_data,direc+'/dev_prover9CNFformat.json')
    write_file(convert_test_data,direc+'/test_prover9CNFformat.json')


