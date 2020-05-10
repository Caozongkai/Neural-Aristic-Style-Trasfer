import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.utils.data

from PIL import Image

class GramMatrix(nn.Module):
    def forward(self, x):
        a, b, c, d = x.size()
        f = x.view(a * b, c * d)
        G = torch.mm(f, f.t())
        
        return G.div(a * b * c * d)

class ArtCNN():
    def __init__(self, style, content, result):
        super(ArtCNN, self).__init__()
        
        self.style = style
        self.content = content
        self.result = nn.Parameter(result.data)
        
        self.content_layers = [4]
        self.style_layers = [1, 2, 3, 4, 5]
        self.content_weight = 1
        self.style_weight = 1000
        
        self.network = models.vgg19(pretrained=True)
        
        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.LBFGS([self.result])
        
        if torch.cuda.is_available():
            self.network.cuda()
            self.gram.cuda()
            
    def train(self):
        def helper():
            self.optimizer.zero_grad()
          
            result = self.result.clone()
            result.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()
            
            contentLoss = 0
            styleLoss = 0
            
            #n-th block of conv layers
            convBlock = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.network.features):
                if isinstance(layer, nn.ReLU):
                    layer = nn.ReLU(inplace=False)
                if torch.cuda.is_available():
                    layer.cuda()
                    
                result = layer(result)
                content = layer(content)
                style = layer(style)
                
                if isinstance(layer, nn.Conv2d):
                    if convBlock in self.content_layers:
                        contentLoss += self.loss(result * self.content_weight, content.detach() * self.content_weight)
                    
                    if convBlock in self.style_layers:
                        result_g, style_g = self.gram(result), self.gram(style)
                        styleLoss += self.loss(result_g * self.style_weight, style_g.detach() * self.style_weight)
                
                # Update the convBlock when go to the next block of conv layers
                if isinstance(layer, nn.ReLU):
                    convBlock += 1
            
            loss = contentLoss + styleLoss
            loss.backward()
            
            return loss
    
        self.optimizer.step(helper)
        return self.result

imsize = int(sys.argv[3])

transform = transforms.Compose([
             transforms.Resize((imsize, imsize)),
             transforms.ToTensor()
         ])

save = transforms.ToPILImage()

def loadImage(filename):
    img = Image.open(filename)
    img = Variable(transform(img))
    img = img.unsqueeze(0)
    return img
  
def saveImage(img, filename):
    img = img.data.clone().cpu()
    img = img.view(3, imsize, imsize)
    img = save(img)
    img.save(filename)

if torch.cuda.is_available():
    floatType = torch.cuda.FloatTensor
else:
    floatType = torch.FloatTensor


styleFile = sys.argv[1]
contentFile = sys.argv[2]
style = loadImage(styleFile).type(floatType)
content = loadImage(contentFile).type(floatType)

result = content.clone().type(floatType)
result.data = torch.randn(result.data.size()).type(floatType)

art = ArtCNN(style, content, result)
for i in range(200):
    result = art.train()

    if i % 10 == 0:
        print("Iteration: %d" % (i))

        path = "outputs/%d.png" % (i)
        result.data.clamp_(0, 1)
        saveImage(result, path)

result.data.clamp_(0, 1)
saveImage(result, "final.png")
