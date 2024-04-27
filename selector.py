import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from transformers import XLNetPreTrainedModel,XLNetModel

class Selector(XLNetPreTrainedModel):

    def __init__(self,config):
        super(Selector, self).__init__(config)
        self.transformer = XLNetModel(config)
        out_dim = config.hidden_size
        self.classifier=nn.Linear(out_dim, 1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self,input_ids, attn_mask,token_mask, targets=None, return_loss=True,top_n=2):
        last_hidden_state = self.transformer(input_ids=input_ids, attention_mask=attn_mask)['last_hidden_state']
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state).squeeze()
        if return_loss==True:
            loss_not_reduced = F.binary_cross_entropy_with_logits(logits.reshape((input_ids.shape[0],-1)), targets.float(), reduction='none')
            loss_masked = loss_not_reduced * token_mask
            loss_reduced = loss_masked.sum() / token_mask.sum()
            return loss_reduced
        else:
            if top_n==1:
                return self.predictfirst(logits.reshape((input_ids.shape[0],input_ids.shape[1])),token_mask)
            elif top_n==2:
                return self.predict(logits.reshape((input_ids.shape[0],input_ids.shape[1])),token_mask)
            else:
                raise('error')

    def predict(self,logits,token_mask):
        output=[]
        for i in range(logits.shape[0]):
            logit=logits[i][token_mask[i]]
            max1,max2,max1_num,max2_num=-10000,-10000,-1,-1
            for j in range(logit.shape[0]):
                if logit[j]>max1:
                    max2=max1
                    max2_num=max1_num
                    max1=logit[j]
                    max1_num=j
                elif logit[j]>max2:
                    max2_num=j
                    max2=logit[j]
            output.append((max1_num,max2_num))
        return output

    def predictfirst(self,logits,token_mask):
        output=[]
        for i in range(logits.shape[0]):
            logit=logits[i][token_mask[i]]
            max1,max1_num=-10000,-1
            for j in range(logit.shape[0]):
                if logit[j]>max1:
                    max1=logit[j]
                    max1_num=j
            output.append(max1_num)
        return output

class Selector2(XLNetPreTrainedModel):

    def __init__(self,config):
        super(Selector2, self).__init__(config)
        self.transformer = XLNetModel(config)
        out_dim = config.hidden_size
        self.classifier=nn.Linear(out_dim, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.transform=nn.Linear(out_dim, out_dim)

    def forward(self,input_ids,attn_mask,tobeselected,token_mask=None,selected=None,positives=None,negetives=None,
                targets=None,return_loss=True,return_selected=False,alpha=0.1,top_n=1):
        last_hidden_state = self.transformer(input_ids=input_ids, attention_mask=attn_mask)['last_hidden_state']
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state).squeeze()
        false_num, total_num = 0, 0
        if return_loss==True:
            loss_not_reduced = F.binary_cross_entropy_with_logits(logits.reshape((input_ids.shape[0],-1)), targets.float(), reduction='none')
            loss_masked = loss_not_reduced * token_mask
            loss_reduced = loss_masked.sum() / token_mask.sum()
            transformed = self.transform(last_hidden_state)
            contrastive_loss,num=0,0
            for i in range(len(transformed)):
                if len(negetives[i])==0 or len(positives[i])==0:
                    continue
                num+=1
                posi=transformed[i][positives[i]]
                nega=transformed[i][negetives[i]]
                tmp=transformed[i][selected[i]].reshape(1,-1)
                nega_totalsimi = torch.exp(F.cosine_similarity(tmp,nega)).sum()
                posi_simis = F.cosine_similarity(tmp,posi)
                for posi_simi in posi_simis:
                    contrastive_loss-=alpha*torch.log(torch.exp(torch.minimum(posi_simi,torch.tensor(0.8)))/nega_totalsimi)/posi.shape[0]
            if num!=0:
                loss_reduced+=contrastive_loss/num
            if return_selected==False:
                return loss_reduced
            for i in range(len(transformed)):
                tmp = transformed[i][selected[i]].reshape(1, -1)
                if len(positives[i])!=0:
                    total_num+=len(positives[i])
                    posi = transformed[i][positives[i]]
                    posi_simi = F.cosine_similarity(tmp, posi)
                    for j in posi_simi:
                        if j<0:
                            false_num+=1
                if len(negetives[i]) != 0:
                    total_num+=len(negetives[i])
                    nega = transformed[i][negetives[i]]
                    nega_simi=F.cosine_similarity(tmp,nega)
                    for j in nega_simi:
                        if j>0:
                            false_num+=1
            output=self.predict(logits.reshape((input_ids.shape[0], -1)), tobeselected,last_hidden_state,selected,top_n=top_n)
            return loss_reduced,output[0],output[1],false_num,total_num
        else:
            if positives is None:
                return self.predict(logits.reshape((input_ids.shape[0],input_ids.shape[1])),tobeselected,last_hidden_state,selected,top_n=top_n)
            transformed = self.transform(last_hidden_state)
            for i in range(len(transformed)):
                tmp = transformed[i][selected[i]].reshape(1, -1)
                if len(positives[i])!=0:
                    total_num+=len(positives[i])
                    posi = transformed[i][positives[i]]
                    posi_simi = F.cosine_similarity(tmp, posi)
                    for j in posi_simi:
                        if j<0:
                            false_num+=1
                if len(negetives[i]) != 0:
                    total_num+=len(negetives[i])
                    nega = transformed[i][negetives[i]]
                    nega_simi=F.cosine_similarity(tmp,nega)
                    for j in nega_simi:
                        if j>0:
                            false_num+=1
            out=self.predict(logits.reshape((input_ids.shape[0],input_ids.shape[1])),tobeselected,last_hidden_state,selected,top_n=top_n)
            return out[0],out[1],false_num,total_num

    def predict(self,logits,tobeselected,last_hidden_state,selected,top_n=1):
        transformed = self.transform(last_hidden_state)
        output, probs = [], []
        for i in range(logits.shape[0]):
            tmp = transformed[i][selected[i]].reshape(1, -1)
            tobeselected_logits = transformed[i][tobeselected[i]]
            tobeselected_simi = F.cosine_similarity(tmp, tobeselected_logits)
            logit = torch.sigmoid(logits[i][tobeselected[i]])
            for j in range(len(tobeselected_simi)):
                if tobeselected_simi[j]<0:
                    logit[j] = -1
            sorted, indices = torch.sort(logit, descending=True)
            valid_num=top_n
            for j in range(len(sorted)):
                if sorted[j]==-1:
                    valid_num=j
                    break
            output.append(indices[:min(valid_num, top_n)].cpu().numpy().tolist())
            probs.append(sorted[:min(valid_num, top_n)].cpu().numpy().tolist())
        return output, probs

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {'warmup_linear':warmup_linear}

class BertAdam(Optimizer):
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['next_m'] = torch.zeros_like(p.data)
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                next_m.mul_(beta1).add_(other=grad,alpha=1 - beta1)
                next_v.mul_(beta2).addcmul_(grad, grad,value=1 - beta2)
                update = next_m / (next_v.sqrt() + group['e'])

                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss