import json
from tqdm import tqdm

def read_file(file):
    with open(file, encoding='utf8') as f:
        return json.load(f)

def write_file(data, file):
    if file.endswith('json'):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

def process(s):
    l=s.split('(')
    predicate=l[0]
    vars=l[1][:-1].split(',')
    if predicate[0]=='-':
        is_not=True
        predicate=predicate[1:]
    else:
        is_not=False
    for i in range(len(vars)):
        if vars[i]=='x':
            vars[i]='$x'
    return is_not,predicate,vars

def convert(s):
    l=s.split(' | ')
    all,exist=False,False
    if l[0][:7]=='all x (':
        all=True
        l[0]=l[0][7:]
        l[-1]=l[-1][:-1]
    elif l[0][:10]=='exists x (':
        exist = True
        l[0] = l[0][10:]
        l[-1] = l[-1][:-1]

    atoms=[]
    for i in range(len(l)):
        is_not,predicate,vars=process(l[i])
        atom='Atom(\''+predicate+'\''
        for j in range(len(vars)):
            atom+=',\''+vars[j]+'\''
        atom+=')'
        if is_not is True:
            atom='Not('+atom+')'
        atoms.append(atom)
    formula=atoms[0]
    for i in range(1,len(atoms)):
        formula='Or('+formula+','+atoms[i]+')'
    if all is True:
        formula='Forall(\'$x\','+formula+')'
    elif exist is True:
        formula='Exists(\'$x\','+formula+')'
    return formula

def tofol(direc,split='dev'):
    datas = read_file(direc+'/'+split+'_prover9CNFformat.json')
    rclstyle_datas=[]
    for data in tqdm(datas):
        c = dict()
        context2FOL=data['context2FOL']
        answer=data['answer']
        q=data['query']
        fols=data['fols']
        fols=[convert(key) for key in fols]
        for key in context2FOL.keys():
           c[convert(key)]=context2FOL[key]
        qd=dict()
        for key in q.keys():
           qd[convert(key)]=q[key]
        if 'id' in data.keys():
            if 'dep' in data.keys():
                rclstyle_datas.append({'answer':answer, 'query':qd, 'context2FOL':c, 'id':data["id"], 'fols':fols, 'dep':data['dep']})
            else:
                rclstyle_datas.append({'answer': answer, 'query': qd, 'context2FOL': c, 'id': data["id"], 'fols': fols})
        elif 'dep' in data.keys():
            rclstyle_datas.append({'answer': answer, 'query': qd, 'context2FOL': c, 'fols':fols, 'dep':data['dep']})
        else:
            rclstyle_datas.append({'answer': answer, 'query': qd, 'context2FOL': c, 'fols': fols})
    write_file(rclstyle_datas,direc+'/FOL'+split+'.json')

tofol('depth-5','test')
tofol('hard_ruletaker','dev')
tofol('hard_ruletaker','test')
direc='myhard_ruletaker'
tofol(direc,'train')
tofol(direc,'dev')
tofol(direc,'test')
tofol('ruletaker_3ext_sat','train')
tofol('ruletaker_3ext_sat','dev')
tofol('ruletaker_3ext_sat','test')
tofol('ruletaker_exist','train')
tofol('ruletaker_exist','dev')
tofol('ruletaker_exist','test')






