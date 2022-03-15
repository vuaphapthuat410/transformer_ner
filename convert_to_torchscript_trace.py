import numpy as np
import torch
import torch.nn as nn

from transformers import XLMRobertaTokenizerFast, AutoConfig, BertModel
from models import BertCRF

class NerWrapper(nn.Module):
  def __init__(self, model):
    super(NerWrapper, self).__init__()
    self.model = model

  def eval(self):
    self.model.eval()

  def forward(self, input_ids):
    return self.model(input_ids)

if __name__ == "__main__":
  enc = XLMRobertaTokenizerFast.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384", do_lower_case=False)

  # Tokenizing input text
  text = "<s> Anh ấy là ai ? </s> Anh ấy là nhân viên công ty ICOMM </s>"
  tokenized_text = enc.tokenize(text)

  # Masking one of the input tokens
  masked_index = 8
  tokenized_text[masked_index] = '<mask>'
  indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
  segments_ids = np.full(20, 0).tolist()

  # Creating a dummy input
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])
  dummy_input = [tokens_tensor, segments_tensors]

  # Initializing the model with the torchscript flag
  # Flag set to True even though it is not necessary as this model does not have an LM Head.
  config = AutoConfig.from_pretrained("./", torchscript=True)

  # Instantiating the model
  model = BertCRF(config)

  # Init the wrapper
  wrapper = NerWrapper(model)

  # The model/wrapper needs to be in evaluation mode
  wrapper.eval()

  # Creating the trace
  traced_model = torch.jit.trace(wrapper, tokens_tensor, check_trace=True)
  # print(torch.argmax(model(tokens_tensor)[0], dim=2))
  # print(torch.argmax(traced_model(tokens_tensor, segments_tensors)[0], dim=2))
  print(model(tokens_tensor)[0])
  print(traced_model(tokens_tensor)[0])
  torch.jit.save(traced_model, "traced_minilm.pt")