def start():
  print("total epochs = " + str(num_epochs))
  for epoch in range(epoch, num_epochs):
      print('Starting epoch {}...'.format(epoch), end=' ')
      print("\n")
      for i, (images, labels) in enumerate(data_loader):
          print("images.shape = " + str(images.shape))
          #print("(images.shape[0]) = " + str((images.shape[0])))
          #print("len(images[0]) = " + str(len(images[0])))

          step = epoch * len(data_loader) + i + 1
          if (step < saved_step or images.shape[0] < batch_size):
            continue

          mywritter.set_step(step)
          print("STEP = " + str(step))
          real_images = Variable(images).cuda()
          labels = Variable(labels).cuda()
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
          real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())
          # train with fake images
          z = Variable(torch.randn(batch_size, gen_noise_input)).cuda()
          fake_labels = Variable(torch.LongTensor(np.random.randint(0, total_labels, batch_size))).cuda()
          fake_images = generator(z, fake_labels)
          fake_validity = discriminator(fake_images, fake_labels)
          fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())
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
          z = Variable(torch.randn(batch_size, gen_noise_input)).cuda()
          fake_labels = Variable(torch.LongTensor(np.random.randint(0, total_labels, batch_size))).cuda()
          fake_images = generator(z, fake_labels)
          validity = discriminator(fake_images, fake_labels)
          g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
          g_loss.backward()
          g_optimizer.step()
          g_loss = g_loss.item()
          ###############       

          ############### FID SCORE
          if step % display_step == 0:
            print("before fretchet_dist")
            fretchet_dist = calculate_fretchet(real_images, fake_images, model_fid, batch_size, image_size) 
            print("type(fretchet_dist) = " + str(type(fretchet_dist)))
            print("fretchet_dist = " + str(fretchet_dist))
            mywritter.add_scalar('fretchet_dist', fretchet_dist)   
          ###############

          mywritter.add_scalar('g_loss', g_loss)
          mywritter.add_scalar('d_loss', d_loss)
                   

          #writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)  
          mywritter.write()
          if step % display_step == 0:
            n_img = 9
            z = Variable(torch.randn(n_img, gen_noise_input)).cuda()
            labels = Variable(torch.LongTensor(np.random.randint(0, total_labels, n_img))).cuda()
            sample_images = generator(z, labels)
            print("sample_images.shape = " + str(sample_images.shape))
            grid = make_grid(sample_images, nrow=3, normalize=True)
            #grid = make_grid(sample_images, nrow=3, normalize=True)
            writer.add_image('sample_image', grid, step)   
            print("fake_image saved!")
            save_image(grid, images_url + '/_' + str(epoch) + '_' + str(step) + '.png')
            
      if (epoch % save_model_each_n_epochs == 0):
        save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, g_loss, d_loss,  epoch, step)
        #SAVE_IMAGE
        z = Variable(torch.randn(total_labels * total_labels, gen_noise_input)).cuda()
        labels = torch.LongTensor([i for i in range(total_labels) for _ in range(total_labels)]).cuda()
        images = generator(z, labels)
        grid = make_grid(images, nrow=total_labels, normalize=True)
        save_image(grid, images_url + '/_' + str(epoch) + '_' + str(step) + '.png')
        #

      writer.flush()
      writer.close()
      print('Done!')
