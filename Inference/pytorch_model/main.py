import torch
import rcan
from option import args
import numpy as np
from reprod_log import ReprodLogger
reprod_log = ReprodLogger()

model = rcan.RCAN(args)
print('Loading Model...')
model.load_state_dict(torch.load('../../experiment/model/RCAN_BIX4.pt', 'cpu'))
print('Success!')
model.eval()

np.random.seed(1)
np_input = np.random.randn(16,3,48,48)
pytorch_input = torch.tensor(np_input, dtype=torch.float32)
pytorch_output = model(pytorch_input)
pytorch_output = pytorch_output.detach().cpu().numpy()

print('Complete inference!')

reprod_log.add("np_seed0_input", np_input)
reprod_log.add("output", pytorch_output)
reprod_log.save("../forward_pytorch.npy")