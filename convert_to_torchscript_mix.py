import numpy as np
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from torchcrf import CRF

class BertCRF(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            log_likelihood, tags = self.crf(logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        # tags = torch.Tensor(tags)  # change this to convert model
        # ensure tags is a tensor
        # if torch.is_tensor(tags):
        #     tags = tags
        # else:
        #     tags = torch.Tensor(tags) # convert to tensor

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags

from transformers import XLMRobertaTokenizerFast, AutoConfig, BertModel

if __name__ == "__main__":
  enc = XLMRobertaTokenizerFast.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384", do_lower_case=False)

  # Tokenizing input text
  text = "Anh ấy là nhân viên công ty ICOMM."
  tokenized_text = enc.tokenize(text)

  # Masking one of the input tokens
  indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
  segments_ids = np.full(20, 0).tolist()

  # Creating a dummy input
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])
  dummy_input = [tokens_tensor, segments_tensors]

  # Initializing the model with the torchscript flag
  # Flag set to True even though it is not necessary as this model does not have an LM Head.
  config = AutoConfig.from_pretrained("/content/drive/MyDrive/NLP/MiniLM-CRF-sLR", torchscript=True)

  # Instantiating the model
  model = BertCRF.from_pretrained("/content/drive/MyDrive/NLP/MiniLM-CRF-sLR", config = config)
  # inputs = enc(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
  inputs = enc(text, max_length=50, padding='max_length', truncation=True, return_tensors='pt')

  # Init the wrapper
  # wrapper = NerWrapper(model)

  # The model/wrapper needs to be in evaluation mode
  model.eval()

  # Creating the trace
  traced_model = torch.jit.trace(model, tokens_tensor, check_trace=True)
  # print(torch.argmax(model(tokens_tensor)[0], dim=2))
  # print(torch.argmax(traced_model(tokens_tensor, segments_tensors)[0], dim=2))
  print(inputs['input_ids'])
  print(model(inputs['input_ids']))
  print(traced_model(inputs['input_ids']))
  torch.jit.save(traced_model, "traced_minilm.pt")