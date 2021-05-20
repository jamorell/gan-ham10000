import torch
import json

def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, model_url, epoch = None, step = None):
  try:
    if (epoch == None):
      filename = model_url  + "/state.txt"
      with open(filename) as json_file:
        data = json.load(json_file)
        print(data)
        epoch = data['epoch']
        step = data['step']
    print("Loading epoch " + str(epoch))
    checkpoint_generator = torch.load(model_url + "/generator_" + str(epoch) + "_" + str(step) + ".pth")
    generator.model.load_state_dict(checkpoint_generator['model_state_dict'])
    g_optimizer.load_state_dict(checkpoint_generator['optimizer_state_dict'])

    checkpoint_discriminator = torch.load(model_url + "/discriminator_" + str(epoch) + "_" + str(step) + ".pth")
    discriminator.model.load_state_dict(checkpoint_discriminator['model_state_dict'])
    d_optimizer.load_state_dict(checkpoint_discriminator['optimizer_state_dict'])  


    return epoch, step, generator, discriminator, g_optimizer, d_optimizer
  except Exception as e:
    print("Exception loading checkpoint")
    print(e)
    return 0, 1

def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, g_loss, d_loss, model_url, epoch, step):
  #Generator Save
  epoch = epoch + 1
  step = step + 1
  torch.save({
    'epoch': epoch,
    'model_state_dict': generator.model.state_dict(),
    'optimizer_state_dict': g_optimizer.state_dict(),
    'g_loss': g_loss,
  }, model_url + "/generator_" + str(epoch) + "_" + str(step) + ".pth")
  print("generator saved!")
  #Discriminator Save
  torch.save({
    'epoch': epoch,
    'model_state_dict': discriminator.model.state_dict(),
    'optimizer_state_dict': d_optimizer.state_dict(),
    'd_loss': d_loss,
  }, model_url + "/discriminator_" + str(epoch) + "_" + str(step) + ".pth")
  print("discriminator saved!")
  mydict = {}
  mydict['epoch'] = epoch
  mydict['step'] = step
  filename = model_url  + "/state.txt"
  with open(filename, 'w') as outfile:
      json.dump(mydict, outfile)
  print("state saved! " + str(mydict))
