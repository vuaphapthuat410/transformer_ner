import numpy as np
import torch

from transformers import XLMRobertaTokenizerFast
from models import BertCRF

from pathlib import Path
from transformers.convert_graph_to_onnx import convert

model = BertCRF.from_pretrained('../drive/MyDrive/NLP/MiniLM/MiniLM-CRF-sLR', num_labels=9)
tokenizer = XLMRobertaTokenizerFast.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384", do_lower_case=False)
id2label = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# Exported onnx model path.
saved_onnx_path = "./exported_model/minilm-crf.onnx"
convert("pt", model, Path(saved_onnx_path), 11, tokenizer)