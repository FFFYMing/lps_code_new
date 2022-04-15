

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
import os

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #print(x.shape)
        #x_copy = x.clone()
        x_ft = torch.fft.rfft2(x)
        #helper_one = torch.ones(x.shape).to(x.device)
        #helper_ft = torch.fft.rfft2(helper_one)
        
        #print(x_ft.shape)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        #out_helper_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,device=x.device)
        #print(x_ft[:, :, :self.modes1, :self.modes2].shape)
        #print(self.weights1.shape)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        #out_helper_ft[:, :, :self.modes1, :self.modes2] = \
        #    self.compl_mul2d(helper_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        #out_helper_ft[:, :, -self.modes1:, :self.modes2] = \
        #    self.compl_mul2d(helper_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        #helper = torch.fft.irfft2(out_helper_ft, s=(x.size(-2), x.size(-1)))
        return x #- x_copy*helper

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, nlayer):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.nlayer = nlayer
        self.fc00 = nn.Linear(5, self.width)


        self.convlayer11 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2).cuda() 
        self.w11 = nn.Conv1d(self.width, self.width, 1).cuda() 



        #for _ in range(self.nlayer-1):
        #    self.convlayer.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2).cuda())
        #    self.w.append(nn.Conv1d(self.width, self.width, 1).cuda())

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)
        #self.m = nn.Threshold(0.3,0)

    @staticmethod
    def tanh_general(x, alpha):
        return (torch.exp(alpha * x) - torch.exp(-alpha * x)) / (torch.exp(alpha * x) + torch.exp(-alpha * x))


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        damage = x[:,:,:,0:1].clone()
        x = self.fc00(x)


        u11 = x.permute(0, 3, 1, 2)



        for layer in range(self.nlayer-1):
            u11_1 = self.convlayer11(u11)
            u11_2 = self.w11(u11.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            u11 = F.relu(u11_1 + u11_2)/self.nlayer + u11

        u11_1 = self.convlayer11(u11)
        u11_2 = self.w11(u11.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        u11 = (u11_1 + u11_2)/self.nlayer + u11

        x = u11.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x += bd
        return damage + self.tanh_general(x[:,:,:,0:1],0.1), torch.cat([x[:,:,:,1:2],x[:,:,:,2:3]],dim=1)
        #return torch.cat([x[:,:,:,0],x[:,:,:,1]],dim=1),x[:,:,:,2:3]




################################################################
# configs
################################################################

TRAIN_PATH20 = '../LPS_data_mat/KIC_20_6_70.mat'
TEST_PATH20 = '../LPS_data_mat/KIC_20_6_70.mat'


ntrain = 55
ntest = 5

batch_size = 5
learning_rate = 0.005

epochs = 500
step_size = 100
gamma = 0.5

modes = 15
width = 32

s = 200
r = 1

################################################################
# load data and data normalization
################################################################
s1 = 200
s2 = 200
reader = MatReader(TRAIN_PATH20)
start = 0
step_train = 5
#x_train = reader.read_field('E')[:ntrain,::r,::r].reshape(ntrain,s,s,1)
u_train = reader.read_field('u')[start+1:ntrain+1,::r,::r].reshape(ntrain-start,s1,s2)
v_train = reader.read_field('v')[start+1:ntrain+1,::r,::r].reshape(ntrain-start,s1,s2)
damage_train = reader.read_field('damage')[start+1:ntrain+1,::r,::r].reshape(ntrain-start,s1,s2,1)

dlast_train = reader.read_field('damage')[start:ntrain-step_train,::r,::r].reshape(ntrain-start-step_train,s1,s2,1)
ulast_train = reader.read_field('u')[start:ntrain-step_train,::r,::r].reshape(ntrain-start-step_train,s1,s2,1)
vlast_train = reader.read_field('v')[start:ntrain-step_train,::r,::r].reshape(ntrain-start-step_train,s1,s2,1)

ustep_train = torch.zeros((ntrain-step_train,s1,s2,step_train))
vstep_train = torch.zeros((ntrain-step_train,s1,s2,step_train))
damagestep_train = torch.zeros((ntrain-step_train,s1,s2,step_train))

for i in range(ntrain-step_train):
    for j in range(step_train):
        ustep_train[i,:,:,j] = u_train[i+j,:,:]
        vstep_train[i, :, :, j] = v_train[i + j, :, :]
        damagestep_train[i, :, :, j] = damage_train[i + j, :, :,0]

reader = MatReader(TEST_PATH20)
#x_test = reader.read_field('E')[ntrain:ntrain+ntest,::r,::r].reshape(ntest,s,s,1)
u_test = reader.read_field('u')[1+ntrain:ntrain+ntest+1,::r,::r].reshape(ntest,s1,s2)
v_test = reader.read_field('v')[1+ntrain:ntrain+ntest+1,::r,::r].reshape(ntest,s1,s2)
damage_test = reader.read_field('damage')[1+ntrain:1+ntrain+ntest,::r,::r].reshape(ntest,s1,s2,1)
dlast_test = reader.read_field('damage')[ntrain:ntrain+ntest,::r,::r].reshape(ntest,s1,s2,1)
ulast_test = reader.read_field('u')[ntrain:ntrain+ntest,::r,::r].reshape(ntest,s1,s2,1)
vlast_test = reader.read_field('v')[ntrain:ntrain+ntest,::r,::r].reshape(ntest,s1,s2,1)

ntrain = ustep_train.shape[0]



"""
ustep_train = ustep_train[:,40:140,50:150,:]
vstep_train = vstep_train[:,40:140,50:150,:]
damagestep_train = damagestep_train[:,40:140,50:150,:]
#ulast_train = ulast_train[:,40:140,50:150]
#vlast_train = vlast_train[:,40:140,50:150]
dlast_train = dlast_train[:,40:140,50:150,:]

#x_test = x_test[:,40:60,90:110,:]
u_test = u_test[:,40:140,50:150]
v_test = v_test[:,40:140,50:150]
damage_test = damage_test[:,40:140,50:150,:]
#ulast_test = ulast_test[:,40:140,50:150]
#vlast_test = vlast_test[:,40:140,50:150]
dlast_test = dlast_test[:,40:140,50:150,:]

s = 100

"""
plt.imshow(damage_test[0,:,:,0])
plt.show()
plt.imshow(damage_test[4,:,:,0])
plt.show()




dis_train = torch.cat([ustep_train,vstep_train],dim=1)
dis_test = torch.cat([u_test,v_test],dim=1)

dislast_train = torch.cat([ulast_train,vlast_train],dim=1)
dislast_test = torch.cat([ulast_test,vlast_test],dim=1)


##normalize u and v and damage


dis_normalizer = UnitGaussianNormalizer(dis_train)
#dis_train = dis_normalizer.encode(dis_train)
#dis_test = dis_normalizer.encode(dis_test)


d_normalizer = UnitGaussianNormalizer(damage_train)
#damage_train = d_normalizer.encode(damage_train)

v_normalizer = UnitGaussianNormalizer(v_train)
v_train = v_normalizer.encode(v_train)


dlast_normalizer = UnitGaussianNormalizer(dlast_train)
#dlast_train = dlast_normalizer.encode(dlast_train)
#dlast_test = dlast_normalizer.encode(dlast_test)



dis_train = dis_train.to(device)
dis_test = dis_test.to(device)
dislast_train = dislast_train.to(device)
dislast_test = dislast_test.to(device)
dlast_train = dlast_train.to(device)
dlast_test = dlast_test.to(device)

#dislast_normalizer = UnitGaussianNormalizer(dislast_train)
#dislast_train = dislast_normalizer.encode(dislast_train)
#dislast_test = dislast_normalizer.encode(dislast_test)



grids = []
grids.append(np.linspace(0, 1.99, s))
grids.append(np.linspace(0, 1.99, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)
grid = grid.to(device)

print(dislast_train.shape)
x_train = torch.cat([dlast_train,dislast_train[:,:s,:,:],dislast_train[:,s:,:,:],grid.repeat(ntrain,1,1,1)], dim=3)
#x_test = torch.cat([dlast_test,dis_test[:,:s,:,0:1],dis_test[:,s:,:,0:1], grid.repeat(ntest,1,1,1)], dim=3)



train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, dis_train,damagestep_train), batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, dis_test,damage_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
def scheduler(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
def LR_schedule(learning_rate,steps,scheduler_step,scheduler_gamma):
    #print(steps//scheduler_step)
    return learning_rate*np.power(scheduler_gamma,(steps//scheduler_step))



base_dir = './LPS_FNO_NKN_nnn/shatodeep_NKN_currentuvd_s200'
if not os.path.exists(base_dir):
    os.makedirs(base_dir);


myloss = LpLoss(size_average=False)
d_normalizer.cuda()
dis_normalizer.cuda()
v_normalizer.cuda()
test_u_l2 = 0.0
test_d_l2 = 0.0
for nlayer in range(3):
    nb = 2**nlayer
    print("nlayer: %d" % nb)
    model = FNO2d(modes, modes, width, nb).cuda()
    print(count_params(model))

    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    model_filename= '%s/FNO_NKN_depth%d.ckpt' % (base_dir, nb)
    if nb!=1:
        model_filename_restart = '%s/FNO_NKN_depth%d.ckpt' % (base_dir, nb//2)
        model.load_state_dict(torch.load(model_filename_restart))

    train_l2_min = 1e10


    for ep in range(epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep, step_size, gamma))
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_mse = 0.0
        train_l2_u = 0.0
        train_l2_d = 0.0
        for x, dis,d in train_loader:
            x, dis,d = x.cuda(), dis.cuda(),d.cuda()
            optimizer.zero_grad()
            loss = 0.0
            loss_d = 0.0
            loss_u = 0.0
            for step in range(step_train):
                if step == 0:
                    d_out,dis_out = model(x)
                else:
                    x = torch.cat([d_out,dis_out[:,:s,:,:],dis_out[:,s:,:,:],grid.repeat(step_train,1,1,1)], dim=3)
                    d_out,dis_out = model(x)

                loss_d += myloss(d_out.view(x.shape[0], -1), d[:,:,:,step:step+1].view(x.shape[0], -1))
                loss_u += myloss(dis_out.view(x.shape[0], -1), dis[:,:,:,step:step+1].view(x.shape[0], -1))
                #dis_out = dislast_normalizer.encode(dis_out)
            loss = loss_d + loss_u
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()/step_train
            train_l2_d += loss_d.item()/step_train
            train_l2_u += loss_u.item() / step_train

        train_l2 /= ntrain
        train_l2_u /= ntrain
        train_l2_d /= ntrain
        model.eval()
        if train_l2 < train_l2_min:
            train_l2_min = train_l2
            torch.save(model.state_dict(), model_filename)
            test_u_l2 = 0.0
            test_d_l2 = 0.0
            with torch.no_grad():
                for step in range(ntest):
                    if step == 0:
                        x = torch.cat(
                            [dlast_test[0:1],dislast_test[0:1, :s, :,:], dislast_test[0:1, s:, :,:],
                                grid.repeat(1, 1, 1, 1)],
                            dim=3).to(device)
                    else:
                        x = torch.cat(
                                [im_d.reshape(1,s,s,1),im_dis[:,:s,:,:],im_dis[:,s:,:,:],
                                grid.repeat(1, 1, 1, 1)], dim=3).to(device)

                    im_d,im_dis = model(x)
                    #im_out = dis_normalizer.decode(im_dis.reshape(1,2*s,s))
                    # im_out = im.clone()
                    dis = dis_test[step:step + 1].to(device)
                    d = damage_test[step:step+1].to(device)

                    test_d_l2 += myloss(im_d.view(x.shape[0], -1), d.view(x.shape[0], -1))
                    test_u_l2 += myloss(im_dis.view(x.shape[0], -1), dis.reshape(1,2*s,s,1).view(x.shape[0], -1))
                    # = dislast_normalizer.encode(im_dis)
                    #print(im_d)
                    #print(myloss(im_d.view(x.shape[0], -1), d.view(x.shape[0], -1)))
            test_u_l2 /= ntest
            test_d_l2 /= ntest

        t2 = default_timer()
        print("depth: %d, epoch: %d, current training l2: %f,best training l2: %f, training u: %f training d:%f test_u_l2: %f, test_d_l2: %f" %(nb, ep,train_l2,train_l2_min, train_l2_u, train_l2_d,test_u_l2,test_d_l2))













