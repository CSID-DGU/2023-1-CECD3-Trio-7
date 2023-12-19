import torch
import numpy as np
#import pytorch_fft.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pretrainedmodels
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn import Parameter
from torchsummaryX import summary
import torchvision
from fftNd import *
import TDN
import ssl
import os


def f_transform(y,y_pred):

    #FFT
    f_y = np.fft.fft2(y)
    f_y_shift = np.fft.fftshift(f_y)

    f_pred = np.fft.fft2(y_pred)
    f_pred_shift = np.fft.fftshift(f_pred)
    #AMP,PHASE 구함
    y_amp = np.abs(f_y_shift)
    y_phase = np.angle(f_y_shift)

    y_pred_amp = np.abs(f_pred_shift)
    y_pred_phase = np.angle(f_pred_shift)

    return np.sum(np.abs((y_amp - y_pred_amp)) + np.abs((y_phase - y_pred_phase)))


def f_transform_np(y,y_pred):

    #FFT
    f_y = np.fft.fft2(y)
    f_pred = np.fft.fft2(y_pred)

    #AMP,PHASE 구함
    y_amp = torch.abs(f_y)
    y_phase = torch.angle(f_y)

    y_pred_amp = torch.abs(f_pred)
    y_pred_phase = torch.angle(f_pred)

    return torch.sum(torch.abs((y_amp - y_pred_amp)) + torch.abs((y_phase - y_pred_phase)))


def brelu(y,y_pred):
    m = nn.Hardtanh(0., 1.)
    o_i = m(y)
    x = torch.log(torch.abs(o_i - y_pred))

    return torch.sum(x)
    
def spatial_loss_f(input_img, label_img):
    loss_func = nn.L1Loss()
    loss = loss_func(input_img, label_img)
    return loss
    

def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])





train_transform = transforms.Compose([ 
    transforms.ToTensor()
])

pil_transform = transforms.Compose([ 
    transforms.ToPILImage()
])



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform = None):
        self.data_dir = data_dir
        self.transform = train_transform
        self.data_dir_input = self.data_dir + '/hazy'
        self.data_dir_label = self.data_dir + '/GT'
        lst_data_input = os.listdir(self.data_dir_input)
        lst_data_label = os.listdir(self.data_dir_label)

        self.lst_input = lst_data_input
        self.lst_label = lst_data_label
        self.transform = transform

    def __len__(self):
        return len(self.lst_label)


    def __getitem__(self,index):
        #hazy_images = [os.path.join(self.data_dir_input, x) for x in os.listdir(lst_data_input) if is_image_file(x)]
        #GT_images = [os.path.join(lst_data_label, x) for x in os.listdir(lst_data_label) if is_image_file(x)]
        input = Image.open(os.path.join(self.data_dir_input, self.lst_input[index]))
        label = Image.open(os.path.join(self.data_dir_label, self.lst_label[index]))

        input_img = self.transform(input)
        label_img = self.transform(label)
        

        return input_img, label_img
        #return input,label



class TestDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform = None):
        self.data_dir = data_dir
        self.transform = train_transform
        self.data_dir_input = self.data_dir + '/hazy'
        self.data_dir_label = self.data_dir + '/GT'
        lst_data_input = os.listdir(self.data_dir_input)
        lst_data_label = os.listdir(self.data_dir_label)

        self.lst_input = lst_data_input
        self.lst_label = lst_data_label
        self.transform = transform

    def __len__(self):
        return len(self.lst_label)


    def __getitem__(self,index):
        #hazy_images = [os.path.join(self.data_dir_input, x) for x in os.listdir(lst_data_input) if is_image_file(x)]
        #GT_images = [os.path.join(lst_data_label, x) for x in os.listdir(lst_data_label) if is_image_file(x)]
        input = Image.open(os.path.join(self.data_dir_input, self.lst_input[index]))
        label = Image.open(os.path.join(self.data_dir_label, self.lst_label[index]))

        input_img = self.transform(input)
        label_img = self.transform(label)

        return input_img, label_img



train_path = '/home/yj/Desktop/hd/test41/image/train'
test_path = '/home/yj/Desktop/hd/test41/image/test'


train_dataset = TrainDataset(train_path, train_transform)
test_dataset = TestDataset(test_path, train_transform)

train_loader = DataLoader(train_dataset,
                        batch_size = 40,
                        shuffle = False,
                        num_workers = 0
                        )

test_loader = DataLoader(test_path,
                        batch_size = 40,
                        shuffle = False,
                        num_workers = 0
                        )

#images, labels = next(iter(train_loader))
#print(images[0].shape)


#model
model = Net(pretrained = True)
model.to("cuda")

#for param in model.parameters():
#    param.requires_grad = False

#epochs,batch_size
epochs = 10
batch_size = 40


#print(model)


#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-04, betas = (0.9,0.999), eps = 1e-08)

#scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,55,80], gamma=0.5)

#model.load_state_dict(torch.load(os.path.join("/home/yj/Desktop/hd/test41", "TDN_NTIRE2020_Dehazing.pt")))

#summary(model, (1, 1200, 1600), batch_size)


# Training loop
with torch.no_grad():
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
    
        for batch_idx, (input_img, label_img) in enumerate(train_loader):
            input_img, label_img = input_img.to('cuda'), label_img.to('cuda')
    
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(input_img)
    
            # Compute the loss
            spatial_loss = spatial_loss_f(outputs,label_img)
            frequency_loss = f_transform(outputs,label_img)
            threshole_loss = brelu(outputs,label_img)
            loss = 0.5 * spatial_loss + 0.5 * frequency_loss + threshole_loss
    
            # Backpropagation
            loss.backward()
            optimizer.step()
    
            running_loss += loss
    
            # Print statistics
            if batch_idx % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

# Save the trained model
torch.save(model.state_dict(), 'your_trained_model.pth')


'''
def model_train(model, data_loader):
    model.train()
    prograss_bar = tqdm(data_loader)

    for input_img, label_img in prograss_bar:
        input_img = pil_transform(input_img)
        label_img = pil_transform(label_img)
        frequency_loss = f_transform(input_img,label_img)

        input_img = train_transform(input_img)
        label_img = train_transform(label_img)
        input_img, label_img = input_img.ToTensor(), label_img.ToTensor()
        input_img, label_img = input_img.to('cuda'), label_img.to('cuda')
        #f_loss = torch.tensor(frequency_loss)
        #f_loss.to('cuda')
        optimizer.zero_grad()
        spatial_loss = spatial_loss_f(input_img,label_img)

        threshole_loss = brelu(input_img,label_img)
  
        loss = 0.5 * spatial_loss + 0.5 * frequency_loss + threshole_loss

        output = model(imput_img)
        loss.backward()
        optimizer.step()
        scheduler.step()


    
for epoch in range(epochs):
    model_train(model, train_loader)
'''
'''
def model_train(model, data_loader):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch_idx, (input_img, label_img) in enumerate(train_loader):
            input_img, label_img = input_img.to('cuda'), label_img.to('cuda')

            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(input_img)
    
            # Compute the loss
            spatial_loss = spatial_loss_f(outputs,label_img)
            frequency_loss = f_transform(outputs,label_img)
            threshole_loss = brelu(outputs,label_img)
            loss = 0.5 * spatial_loss + 0.5 * frequency_loss + threshole_loss
    
            # Backpropagation
            loss.backward()
            optimizer.step()
    
            running_loss += loss
    
            # Print statistics
            if batch_idx % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0


model_train(model, train_loader)
'''


#test
'''
img = Image.open("/home/yj/Desktop/hd/test41/image/train/hazy/13_hazy.png")
img1 = ToTensor()(img)
img1 = Variable(img1).cuda().unsqueeze(0)
#print(img1)
#print(img1.shape)
trans = transforms.ToPILImage() 
img2 = model(img1)
print(img2)
print(img2.shape)
img2 = img2.cpu().data
#img2 = img2.cpu().data
img3 = trans(img2[0])
img3.save("/home/yj/Desktop/hd/test41/image/train/hazy/13_hazy1.png")

'''
    












