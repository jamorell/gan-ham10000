#https://colab.research.google.com/drive/16gtZtUyV8Zus9xoc7jCAjh2TS55uHrhL

#https://github.com/arturml/mnist-cgan/blob/master/mnist-cgan.ipynb
#https://github.com/Lornatang/CGAN-PyTorch

from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image

from nnmodels import *
from mywriter import MyWriter
from mydatasets import MyDatasets
from getdatasets import *
from tensorboardX import SummaryWriter
from checkpoints import *
from mainloop import *
from fid import *

import matplotlib.pyplot as plt
import numpy as np

dataset_name = "ham10000_gray_normal"
dataset_url = "http://mallba3.lcc.uma.es/jamorell/datasets/ham10000/gray/normal/base_dir.zip"
dataset_path = "./" + dataset_name + "/base_dir/" 
runs_url = "./output_cpu/runs"
models_url = "./output_cpu/models"
images_url = "./output_cpu/images"

try:
    os.makedirs(models_url)
except Exception as e:
    print(e)

try:
    os.makedirs(images_url)
except Exception as e:
    print(e)


image_size = 28
n_channels = 1
total_labels = 7
gen_noise_input = 100
batch_size = 64

print("IS_CUDA = " + str(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model_fid = InceptionV3([block_idx])
model_fid = model_fid.to(device)


get_dataset_from_url(dataset_name, dataset_url)

data_loader = MyDatasets.get_ham10000_gray_normal(dataset_path, batch_size)
print(data_loader)

generator = Generator(image_size, n_channels, total_labels, gen_noise_input).to(device)
discriminator = Discriminator(image_size, n_channels, total_labels).to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)


writer = SummaryWriter(runs_url)
mywritter = MyWriter(writer)



num_epochs = 5000 #20
n_critic = 50
display_step = 500 # 10
loaded = load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, models_url) #,epoch
epoch = loaded[0]
saved_real = False
step = 0
saved_step = loaded[1]
save_model_each_n_epochs = 5

######################LOOP

print("total epochs = " + str(num_epochs))
for epoch in range(epoch, num_epochs):
    print('Starting epoch {}...'.format(epoch), end=' ')
    print("\n")
    for i, (images, labels) in enumerate(data_loader):
        print("images.shape = " + str(images.shape))

        step = epoch * len(data_loader) + i + 1
        if (step < saved_step or images.shape[0] < batch_size):
          continue

        mywritter.set_step(step)
        print("STEP = " + str(step))
        real_images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        generator.train()
        
        if (not saved_real):
          grid = make_grid(real_images, nrow=3, normalize=True)
          save_image(grid, images_url + '/real.png')
          writer.add_image('real_image', grid, step)
          saved_real = True


        #d_loss = discriminator_train_step(len(real_images), discriminator,
        #                                  generator, d_optimizer, criterion,
        #                                  real_images, labels)
        ############### DISCRIMINATOR TRAIN STEP
        d_optimizer.zero_grad()
        # train with real images
        real_validity = discriminator(real_images, labels)
        real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
        # train with fake images
        z = Variable(torch.randn(batch_size, gen_noise_input)).to(device)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, total_labels, batch_size))).to(device)
        fake_images = generator(z, fake_labels)
        fake_validity = discriminator(fake_images, fake_labels)
        fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
        mywritter.add_scalar('d_loss_real', real_loss)
        mywritter.add_scalar('d_loss_fake', fake_loss)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        d_loss = d_loss.item()
        ###############

        #g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
        ############### GENERATOR TRAIN STEP
        g_optimizer.zero_grad()
        z = Variable(torch.randn(batch_size, gen_noise_input)).to(device)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, total_labels, batch_size))).to(device)
        fake_images = generator(z, fake_labels)
        validity = discriminator(fake_images, fake_labels)
        g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
        g_loss.backward()
        g_optimizer.step()
        g_loss = g_loss.item()
        ###############       

        ############### FID SCORE
        if step % display_step == 0:
          print("before fretchet_dist")
          fretchet_dist = calculate_fretchet(real_images, fake_images, model_fid, batch_size, image_size, device) 
          print("type(fretchet_dist) = " + str(type(fretchet_dist)))
          print("fretchet_dist = " + str(fretchet_dist))
          mywritter.add_scalar('fretchet_dist', fretchet_dist)   
        ###############
        print("g_loss = " + str(g_loss) + " d_loss = " + str(d_loss))
        mywritter.add_scalar('g_loss', g_loss)
        mywritter.add_scalar('d_loss', d_loss)
                 

        #writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)  
        mywritter.write()
        if step % display_step == 0:
          n_img = 9
          z = Variable(torch.randn(n_img, gen_noise_input)).to(device)
          labels = Variable(torch.LongTensor(np.random.randint(0, total_labels, n_img))).to(device)
          sample_images = generator(z, labels)
          print("sample_images.shape = " + str(sample_images.shape))
          grid = make_grid(sample_images, nrow=3, normalize=True)
          #grid = make_grid(sample_images, nrow=3, normalize=True)
          writer.add_image('sample_image', grid, step)   
          print("fake_image saved!")
          save_image(grid, images_url + '/_' + str(epoch) + '_' + str(step) + '.png')
          
    if (epoch % save_model_each_n_epochs == 0):
      print("saving epoch = " + str(epoch) + " step = " + str(step)) 
      save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, g_loss, d_loss, models_url,  epoch, step)
      #SAVE_IMAGE
      z = Variable(torch.randn(total_labels * total_labels, gen_noise_input)).to(device)
      labels = torch.LongTensor([i for i in range(total_labels) for _ in range(total_labels)]).to(device)
      images = generator(z, labels)
      grid = make_grid(images, nrow=total_labels, normalize=True)
      save_image(grid, images_url + '/_' + str(epoch) + '_' + str(step) + '.png')
      #

    writer.flush()
    writer.close()
    print('Done!')
######################


#END REMOVE DATASET
remove_dataset_folder("./" + dataset_name + "/")
