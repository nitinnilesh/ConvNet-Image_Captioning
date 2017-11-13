get_ipython().magic('matplotlib inline')
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as fun
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import numpy as np
import os
import matplotlib.pyplot as plt


# In[25]:


data = pd.read_csv('Captions/mt.csv', header=None)
data.drop(data.columns[2], axis=1, inplace=True)
data.dropna(inplace=True)
data = data.reset_index(drop=True)
data.to_csv('captions.csv', index=False)


# In[26]:


path = '/media/nishant/Fantasy/MS/Courses/SMAI/Project/Flickr8k_text'
for root, dirs, files in os.walk(path):
    for f in files:
        if f == 'Flickr8k.token.txt':
            with open(os.path.join(root, f), "r") as infile:
                f = infile.read()
                f.replace('-', ' ')
                f = f.replace(", ", "")
                f = f.replace('"', '')
                f = f.replace(";", "")
                f = f.replace(": ", "")
                f = f.replace("!", "")
                #f = re.sub("\S*\d\S*", "", f).strip()
                l = k = f.split()
                #l = [x for x in l if len(x)>5 and len(x)<11]
                l = set(l)
                print(len(l))
                word_list = [w for w in l if w.isalpha()]
                print(len(word_list))


# In[27]:


class Flicker(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.captions_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.captions_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.captions_frame.iloc[idx, 0])
        
        img_name = img_name[:-2]
        
        image = io.imread(img_name)
        
        captions = self.captions_frame.iloc[idx, 1].strip().split()
        if captions[-1] != '.':
            captions.append('.')
        #print(captions)
        #plt.imshow(image)
        #plt.show()
        
        sample = {'image': image, 'captions': captions}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[28]:


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, captions = sample['image'], sample['captions']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for captions because for images,
        # x and y axes are axis 1 and 0 respectively
        #captions = captions * [new_w / w, new_h / h]

        return {'image': img, 'captions': captions}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, captions = sample['image'], sample['captions']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #cap = list()
        #for word in captions:
            #cap.append(one_hot(word))
        #captions = np.array(cap)
        #print(captions.shape)
        image = torch.Tensor(image)
        return {'image': image.type(dtype),
                'captions': captions}


# In[29]:


transformed_dataset = Flicker(csv_file='Captions/captions.csv',
                                           root_dir='Flickr8k_Dataset/Flicker8k_Dataset',
                                           transform=transforms.Compose([
                                               Rescale((224, 224)),
                                               ToTensor(), 
                                               #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                           ]))


# In[30]:


dataloader = DataLoader(transformed_dataset, batch_size=1,
                        shuffle=True, num_workers=1)


# In[31]:


use_cuda = False #torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


# In[32]:


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          len(sample_batched['captions']))

    # observe 4th batch and stop.
    if i_batch == 3:
        break


# In[33]:


class CNN(nn.Module):
    def __init__(self, embed_dim):
        super(CNN, self).__init__()
        self.squeeze = models.squeezenet1_0(pretrained=True)
        self.features = nn.Linear(1000, embed_dim)
        
    def forward(self, x):
        feat = self.squeeze(x)
        feat = self.features(feat)
        return feat        


# In[34]:


class LSTM(nn.Module):
    def __init__(self, embed_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = 9000
        self.embed_dim = 300
        self.hx = Variable(torch.randn(1, self.hidden_dim), requires_grad=True)
        self.cx = Variable(torch.randn(1, self.hidden_dim), requires_grad=True)
        self.lstm = nn.LSTMCell(300, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.embed_dim)
        
    def forward(self, x):
        gen_cap = Variable(torch.zeros(self.hidden_dim, 1), requires_grad=True)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        for i in range(20):
            output = self.out(self.hx.view(1, -1))
            #print(gen_cap)
            #print(self.hx)
            self.hx, self.cx = self.lstm(output, (self.hx, self.cx))
            gen_cap = torch.cat((gen_cap, self.hx.view(-1, 1)), 1)
        return gen_cap


# In[35]:


embed_dim = 300
num_epochs = 5


# In[36]:


cnn = CNN(embed_dim)
lstm = LSTM(embed_dim)


# In[37]:


if use_cuda:
    cnn.cuda()
    lstm.cuda()


# In[38]:


parameters = list(cnn.parameters()) + list(lstm.parameters())
optimizer = optim.Adam(parameters, lr=0.001)


# In[39]:


def emd2wd(arr, word_list):
    word = []
    maxs, indices = torch.max(arr, 0)
    indices = indices.data.numpy().tolist()
    for i in indices:
        word.append(word_list[i])
    return word


# In[40]:


def loss_fun(outputs, captions):
    loss = Variable(torch.zeros(1, 1), requires_grad=True)
    for i, word in enumerate(captions):
        try:
            index = word_list.index(word)
            loss = loss + outputs.data[index, i]
        except:
            return loss
    return loss


# In[42]:


for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs, captions = data['image'], data['captions']
        inputs, captions = Variable(inputs, requires_grad=True), captions
        optimizer.zero_grad()
        cnn.zero_grad()
        lstm.zero_grad()
        features = cnn(inputs)
        outputs = lstm(features)
        print(emd2wd(outputs, word_list))
        loss = loss_fun(outputs, captions)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if i % 2 == 1:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
print('Finished Training')


# In[43]:


features = cnn(inputs)
outputs = lstm(features)
print(emd2wd(outputs, word_list))
