

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Conv1D


def get_specific_layer_names(model):
    result = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            result.append('.'.join(name.split('.')[-2:]))
    return result

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
print(model)
print()
print()
print(list(set(get_specific_layer_names(model))))