#https://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid
################FID SCORE
import torch
import numpy as np
from torch.autograd import Variable
from scipy import linalg

def calculate_activation_statistics(images,model,device,batch_size=128,dims=2048):
    model.eval()
    act=np.empty((len(images), dims))
    '''
    if cuda:
        batch=images.to(device)
    else:
        batch=images
    '''
    batch=images.to(device)
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, device, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
    
#Convert 1-channel data to 3-channel data when calculating Perceptual Loss
#https://www.programmersought.com/article/45027317971/
def calculate_fretchet(images_real, images_fake, model, batch_size, image_size, device):
    #print("z1")
    dims = 2048 #int(image_size * image_size / 2)
    #print("z2")
    '''
    print("images_real")
    print("images_real.shape " + str(images_real.shape))
    print("images_fake.shape " + str(images_fake.shape))
    print("batch_size " + str(batch_size))
    print("image_size " + str(image_size)) 
    '''
    #1 channel to 3 channels   
    #img2 = torch.from_numpy(np.zeros_like(images_real.cpu()))
    img2 = torch.from_numpy(np.zeros( (batch_size, 3, image_size, image_size)))
    #print("z3")    
    #print("images_real.shape = " + str(images_real.shape ))
    #print("img2.shape = " + str(img2.shape ))
    img2[:,0,:,:] = np.squeeze(images_real.data.cpu()) 
    #print("z4")    
    img2[:,1,:,:] = np.squeeze(images_real.data.cpu()) 
    #print("z5")
    img2[:,2,:,:] = np.squeeze(images_real.data.cpu())
    #print("z6")    
    images_real = Variable(img2.to(device)).float()
    #print("z7")    
    #img2 = torch.from_numpy(np.zeros_like(images_fake.cpu()))
    img2 = torch.from_numpy(np.zeros( (batch_size, 3, image_size, image_size) ))
    #print("z8")    
    #print("images_fake.shape = " + str(images_fake.shape ))
    #print("img2.shape = " + str(img2.shape ))
    img2[:,0,:,:] = np.squeeze(images_fake.data.cpu()) 
    #print("z9")    
    img2[:,1,:,:] = np.squeeze(images_fake.data.cpu()) 
    #print("z10")    
    img2[:,2,:,:] = np.squeeze(images_fake.data.cpu())  
    #print("z11")    
    images_fake = Variable(img2.to(device)).float()
    #print("z12")    
    ####################
    mu_1,std_1=calculate_activation_statistics(images_real,model, device, batch_size=batch_size, dims=dims)
    #print("z13")
    #print("images_fake")
    mu_2,std_2=calculate_activation_statistics(images_fake,model, device, batch_size=batch_size, dims=dims)
    #print("z14")    
    #print("calculated_fretchet")

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2, device)
    print("z15")    
    return fid_value
################
