import torch
import sys
d = torch.load(sys.argv[1],  map_location=torch.device("cpu"))
del d['optimizer0']
torch.save(d, sys.argv[1])

