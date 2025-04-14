'''
Our model state is reserved. Please contact us if you need to use it.
'''
# Demo file to verify if model is working correctly (Loads trained model and applies it to new FHR data to get predictions)
# To load a trained CNN model and predict the risk index for a given 10-minute segment of FHR data
import numpy as np
import torch
from net1d import *

# Define CNN model with the parameters
net = Net1D(
    in_channels=1,
    base_filters=256,
    ratio=1,
    filter_list=[256,512,512,1024,1024],
    m_blocks_list=[2,3,3,2,2],
    kernel_size=16,
    stride=2,
    n_classes=1,
    use_bn=True,
    use_do=True,
    verbose=False,
    groups_width=16)

# Load pretrained weights 
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load('model_state',map_location=device_str))

'''
a piece of FHR data
'''
data_to_process=np.load('your data path')
val_x=torch.tensor(np.expand_dims(data_to_process,1)).to('cuda',torch.float32)
print(f'Risk index for the segment id {net(val_x)}')





