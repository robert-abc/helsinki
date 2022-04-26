from torch.utils.tensorboard import SummaryWriter
from utils.dip import *
from utils.tools import *

writer = SummaryWriter('runs/test')

dtype = torch.FloatTensor
input_type = 'noise'
input_depth = 32
pad = 'reflection'
NET_TYPE = 'skip'

deblur_input = get_noise(input_depth,input_type,
                    (512,512)).type(dtype).detach()

deblur_net = get_net(input_depth, NET_TYPE, pad,
                skip_n33d=128,
                skip_n33u=128,
                skip_n11=4,
                n_channels=1,
                num_scales=5,
                upsample_mode='bilinear').type(dtype)

writer.add_graph(deblur_net, deblur_input)
writer.close()