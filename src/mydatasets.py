import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

class MyDatasets:
    @staticmethod
    def get_dataset(dataset_path, batch_size):
        return MyDatasets.get_ham10000_gray_normal(dataset_path, batch_size)

    @staticmethod
    def get_ham10000_gray_normal(dataset_path, batch_size):
        ####MY IMAGES FOLDER
        #### CONSTANTS
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5)) #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = ImageFolder(root=dataset_path, transform=transform)



        print("BEFORE total samples = " + str(len(dataset.samples)))
        print("BEFORE total labels = " + str(len(dataset.targets)))
        dataset_size = len(dataset)
        classes = dataset.classes
        num_classes = len(dataset.classes)
        img_dict = {}
        for i in range(num_classes):
            img_dict[classes[i]] = 0

        for i in range(dataset_size):
            img, label = dataset[i]
            img_dict[classes[label]] += 1
        print(img_dict)

        #Oversampling INIT
        x_train = dataset.samples
        y_train = dataset.targets
        sampling_seed = 0        

        print("before")
        print(type(x_train))
        print(type(x_train[0]))
        print(type(y_train))
        print(type(y_train[0]))
        print(x_train[0])
        print(y_train[0])

        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=sampling_seed)
        #print(type(dataset.samples[0]))
        #print(type(dataset.targets[0]))
        x_train, y_train = sampler.fit_resample(x_train,y_train)
        x_train = list(map(lambda x: tuple(x), x_train))
        #y_train = y_train.tolist()
        print("\n\nafter")
        print(type(x_train))
        print(type(x_train[0]))
        print(type(y_train))
        print(type(y_train[0]))
        print(x_train[0])
        print(y_train[0])
       
        dataset.samples = x_train
        dataset.targets = y_train
        #Oversampling END

        print("total samples = " + str(len(dataset.samples)))
        print("total labels = " + str(len(dataset.targets)))
        dataset_size = len(dataset)
        classes = dataset.classes
        num_classes = len(dataset.classes)
        img_dict = {}
        for i in range(num_classes):
            img_dict[classes[i]] = 0

        for i in range(dataset_size):
            img, label = dataset[i]
            img_dict[classes[int(label)]] += 1
        print(img_dict)
        #exit(0)
        ### NEW TRANSFORMS INIT        
        #dataset.transform = transforms.Compose([ #transforms.Resize((input_size,input_size)),
        #                              transforms.Grayscale(num_output_channels=1),
        #                              transforms.RandomHorizontalFlip(),
        #                              transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
        #                              transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        #                                transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])        
        #
        ### NEW TRANSFORMS END

       



        data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=8,#num_workers=0,
                pin_memory=True, # TODO -> Cuidado puede dar error
                shuffle=True
        )
        return data_loader
