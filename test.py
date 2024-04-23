import torch

logits = torch.tensor([1, 2, 3, 4], dtype=torch.float)
temperature = 1
probs = torch.softmax(logits / temperature, dim=-1)
print(probs)
temperature = 0.8
probs = torch.softmax(logits / temperature, dim=-1)
print(probs)
