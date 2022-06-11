import numpy as np
from reprod_log import ReprodDiffHelper

diff_helper = ReprodDiffHelper()

info_paddle = diff_helper.load_info("./forward_paddle.npy")
info_pytorch = diff_helper.load_info("./forward_pytorch.npy")

diff_helper.compare_info(info_paddle, info_pytorch)

diff_helper.report(
    diff_method="mean", diff_threshold=1e-6, path="./forward_diff_log.txt")
