import torch
from torch import nn
from transformers import T5Tokenizer,T5ForConditionalGeneration

class Convertor(nn.Module):

    def __init__(self,model_name):
        super().__init__()
        self.convertor = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def addtokenizer(self,tokenizer):
        self.tokenizer = tokenizer

    def forward(self,input_ids,attn_mask,y_ids,labels):
        outputs = self.convertor(input_ids=input_ids, attention_mask=attn_mask, decoder_input_ids=y_ids, labels=labels)
        return outputs

    def predict(self, input, output):
        max_length = output.size(1)
        out = self.convertor.generate(input, num_beams=1, min_length=1, max_length=max_length)
        preds = torch.zeros_like(output)
        preds[:, :out.shape[1]] = out
        return preds

    def decode(self, preds):
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in preds]

    def predict_and_decode(self, input_ids,is_inference=False):
        pred = self.convertor.generate(input_ids, max_length=64, num_beams=1, min_length=1)
        if is_inference:
            pred = self.decode(pred)
        return pred
