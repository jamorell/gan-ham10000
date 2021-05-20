import torch.nn as nn
import torch
#https://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid
#https://pytorch.org/vision/stable/models.html
import torchvision.models as models
import torch.nn.functional as F
class Generator(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10, gen_noise_input: int = 100):#(self):
        super().__init__()
        
        self.image_size = image_size

        self.channels = channels

        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.gen_noise_input = gen_noise_input
        
        self.model = nn.Sequential(
            nn.Linear(self.gen_noise_input + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, channels * image_size * image_size), #nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, input, labels): #def forward(self, z, labels): #def forward(self, input: torch.Tensor, labels: list = None) -> torch.Tensor:
        #z = z.view(z.size(0), self.gen_noise_input)
        #c = self.label_emb(labels)
        #x = torch.cat([z, c], 1)
        #out = self.model(x)
        #return out.view(x.size(0), 28, 28)
        conditional_input = torch.cat([input, self.label_emb(labels)], dim=-1)
        out = self.model(conditional_input)
        out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)
        #print("out.shape = " + str(out.shape))
        return out

class Discriminator(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10):
        super().__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            #nn.Linear(784 + num_classes, 1024), #nn.Linear(794, 1024),
            nn.Linear(channels * image_size * image_size + num_classes, 1024), #nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input, labels): # def forward(self, input: torch.Tensor, labels: list = None) -> torch.Tensor:
        #input = input.view(input.size(0), 784)

        input = torch.flatten(input, 1)
        conditional = self.label_emb(labels)
        input = torch.cat([input, conditional], 1)
        out = self.model(input)
        return out.squeeze()


################INCEPTIONV3
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
