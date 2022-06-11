import paddle
import rcan
from option import args
import numpy as np
from reprod_log import ReprodLogger
reprod_log = ReprodLogger()

model = rcan.RCAN(args)
print('Loading Model...')
model.set_state_dict(paddle.load('../../experiment/model/RCAN_BIX4.pdparams'))
print('Success!')
model.eval()

np.random.seed(1)
np_input = np.random.randn(16,3,48,48)
paddle_input = paddle.to_tensor(np_input, dtype=paddle.float32)
paddle_output = model(paddle_input)
paddle_output = paddle_output.cpu().numpy()
print('Complete inference!')

reprod_log.add("np_seed0_input", np_input)
reprod_log.add("output", paddle_output)
reprod_log.save("../forward_paddle.npy")

