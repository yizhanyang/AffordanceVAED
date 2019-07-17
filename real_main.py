import torch
from affordance_vaed import Decoder, Encoder, AffordanceVAED
from matplotlib import pyplot as plt
from blender_loader import BlenderLoader
import torch.optim as optim
from real_loader import KinectEvaluationLoader

encoder = Encoder(20, 3)
decoder = Decoder(20, 2)

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('GPU works!')
else:
    print('YOU ARE NOT USING GPU')

device = torch.device('cuda' if use_cuda else 'cpu')


model = AffordanceVAED(encoder, decoder, device, beta=1).to(device)

model.load_state_dict(torch.load('/home/spacemaster/AffordanceVAED/perception_results/r3/model_epoch_63.pth.tar',map_location = 'cpu'))
model.eval()

optimizer = optim.Adam(model.parameters(), lr=1.0e-3)


dataloader = KinectEvaluationLoader(256, 128, False, '/home/spacemaster/real_images', debug=False)

output = decoder(dataloader).detach().numpy()

for i in range(100):
    plt.imshow(output[i,1])
    plt.show()















#z = torch.randn(100,20)

#output = decoder(z).detach().numpy()

#for i in range(100):
    #plt.imshow(output[i,1])
    #plt.show()



