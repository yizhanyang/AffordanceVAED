import torch
from affordance_vaed import Decoder, Encoder, AffordanceVAED
from matplotlib import pyplot as plt
from blender_loader import BlenderLoader
import torch.optim as optim
import argparse
from tools import affordance_to_array
from PIL import Image
import numpy as np

encoder = Encoder(20, 3)
decoder = Decoder(20, 2)

use_cuda = True #torch.cuda.is_available()

if use_cuda:
    print('GPU works!')
else:
    print('YOU ARE NOT USING GPU')

device = torch.device('cuda' if use_cuda else 'cpu')


model = AffordanceVAED(encoder, decoder, device, beta=1).to(device)

model.load_state_dict(torch.load('/home/yizhan/AffordanceVAED/perception_results/r3/model_epoch_63.pth.tar',map_location = 'cpu'))
model.to(device)
model.eval()

optimizer = optim.Adam(model.parameters(), lr=1.0e-3)

data_paths = ['/home/yizhan/real_images']
dataset = BlenderLoader(batch_size=64, num_processes=18, include_depth=False, data_paths=data_paths, split=False)

dataloader = dataset.get_iterator(True)

for i, batch in enumerate(dataloader):
    output = encoder(batch[0].to(device))[0]
    output = decoder(output).detach().cpu().numpy()
    print(output.shape)

    for j, sample in enumerate(output):
        print(sample.shape)
        im = (affordance_to_array(sample)*255).astype(np.uint8).transpose(1,2,0)

        print(im.shape, im.dtype)
        im = Image.fromarray(im)
        fname = "affordance_%d.png" % (i*dataset.batch_size+j)
        im.save(fname)
        print("saved")


















#z = torch.randn(100,20)

#output = decoder(z).detach().numpy()

#for i in range(100):
    #plt.imshow(output[i,1])
    #plt.show()



