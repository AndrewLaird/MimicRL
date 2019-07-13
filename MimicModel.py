import torch
import torch.nn as nn



class MimicNet(torch.nn.Module):
    
    def __init__(self):
        
        super(MimicNet, self).__init__()
        
        # Num input channels, num output channels, size of window
        self.conv1 = nn.Conv2d(1, 1024, 4)
        self.conv2 = nn.Conv2d(1024, 2048, 6)
        self.conv3 = nn.Conv2d(2048, 4092, 10)

        self.pool1 = nn.AvgPool2d(79)
        
        # gonna compress it down to a 1x2048 vector
        self.reverse_conv1 = nn.Conv2d(4092,2048,10)
        self.reverse_conv2 = nn.Conv2d(2048,1024,6)
        self.reverse_conv3 = nn.Conv2d(1024,3,4)
    
    def encode(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = x.view(-1,4092)
        return x
        



        

        
