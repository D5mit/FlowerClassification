#!/usr/bin/env python
# Imports here
import torch
from torch import nn
# import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models  # we wil be using a pretrained model
import matplotlib.pyplot as plt
import seaborn as sns
import json

import numpy as np
from collections import OrderedDict
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image

# font colors
GS = '\033[90m'
GE = '\033[0m'


def create_dataloaders(data_dir):
    # Define transforms for the training, validation, and testing sets

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(45),
                                     transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),

        'test': transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]),

        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    }

    # Load the datasets with ImageFolder
    print(GS + '  Training data directory: ' + train_dir)
    print('  Testing data directory: ' + test_dir)
    print('  Validation data directory: ' + valid_dir + GE)

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
    }

    return dataloaders, image_datasets


def set_device(gpu):
    # set the device to GPU if GPU is selected and available
    if gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(GS + '  Start training on ' + str(device) + '...' + GE)
    return device


def create_model(arch, lr=0.001, device='cpu', hidden_units=256):
    # Add own classifier
    outputSize = 102

    if arch == 'vgg16':

        iModel = models.vgg16(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dro1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(hidden_units, outputSize)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    elif arch == 'alexnet':

        iModel = models.alexnet(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9216, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dro1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(hidden_units, outputSize)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    # default model is vgg16
    else:

        iModel = models.vgg16(pretrained=True)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dro1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(hidden_units, outputSize)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    # Freeze parameters so we don't back prop through them
    for param in iModel.parameters():
        param.requires_grad = False

    iModel.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(iModel.classifier.parameters(), lr=lr)

    iModel.to(device)

    return iModel, criterion, optimizer


def validation(model, validloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def modelTrain(model, trainloader, validloader, criterion, optimizer, epochs=1, print_every=1, device='cpu'):
    # device = set_device(gpu)

    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        model.to(device)
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, validloader, criterion, device)

                print("  Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss / len(validloader)),
                      "Valid Accuracy: {:.3f}".format(valid_accuracy / len(validloader)))

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()


def modelTest(iModel, dataLoader, device):
    print('  Start Testing on ' + str(device))

    # test the model against training data
    test_loss = 0
    testImages, testLabels = next(iter(dataLoader))

    testImages, testLabels = testImages.to(device), testLabels.to(device)

    print('')
    print('    INPUT')
    print('    -----')
    print('    Test Images Shape: ' + str(testImages.shape))
    print('    Test Labels Shape: ' + str(testLabels.shape))

    # -test Model
    iModel.eval()  # model is now used to testing, nothing new to be learnt
    with torch.no_grad():
        testImages, testLabels = testImages.to(device), testLabels.to(device)

        # print('')
        # print('CHECK ONE ITEM PREDICTION')
        # print('-------------------------')
        # print(testLabels.data[0])

        output = iModel.forward(testImages)
        ps = torch.exp(output)
        predicted = ps.max(dim=1)[1]
        # print(predicted[0])

        equality = (testLabels.data == ps.max(dim=1)[1])

        iCorrect = torch.sum(equality).item()

        print(iCorrect)
        print(equality.shape[0])
        accuracy = iCorrect / equality.shape[0]

        print('')
        print('   CHECK ITEMS IN TEST SET')
        print('   -----------------------')
        print('   ' + str(equality))

        print('')
        print('   ACCURACY')
        print('   -----------------------')
        print('   Model accuracy is: ' + str(accuracy))


def saveModel(iModel, arch, image_dataset, save_dir):
    # Save the checkpoint to drive
    iModel.class_to_idx = image_dataset.class_to_idx
    iModel.cpu()
    torch.save({'arch': arch,
                'state_dict': iModel.state_dict(),
                'class_to_idx': iModel.class_to_idx},
               save_dir)
    print('  Checkpoint saved')
    print('')


# Process a PIL image for use in a PyTorch model
def process_image(image, statusOutput=False):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    new_width = 224
    new_height = 224

    # load the image

    iImage = Image.open(image)
    if statusOutput:
        print(iImage)

    # Resize the image
    if iImage.size[0] > iImage.size[1]:
        iImage.thumbnail((99999, 256))  # constrain the height to be 256
    else:
        iImage.thumbnail((256, 99999))  # constrain the width to 256
    if statusOutput:
        print(iImage)

        # crop the image
    width, height = iImage.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    iImage = iImage.crop((left, top, right, bottom))
    if statusOutput:
        print(iImage)

        # convert image to numpy image
    np_image = np.array(iImage) / 255
    if statusOutput:
        print(np_image[0, 0, 2])

    # normalize the image to networks format
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    if statusOutput:
        print(np_image.shape)

    # PyTorch expects the color channel to be the first dimension but it's the third dimension
    np_image = np_image.transpose((2, 0, 1))
    if statusOutput:
        print(np_image.shape)

    tensor_image = torch.from_numpy(np_image)
    return tensor_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def load_model(checkpoint_path, poutput=False):
    chpt = torch.load(checkpoint_path)

    if poutput:
        print(chpt)

    #     model = models.densenet121(pretrained=True)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = chpt['class_to_idx']

    outputSize = 102
    hidden_units = 256

    iModel = models.vgg16(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dro1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, outputSize)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Put the classifier on the pretrained network
    model.classifier = classifier

    model.load_state_dict(chpt['state_dict'])

    return model


def predict(image_path, checkpoint, topk=5, categories='cat_to_name.json', gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    if gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    # load the model
    lmodel = load_model(checkpoint)

    #     print(type(lmodel))

    # set model to evaluate and also to GPU if possible
    lmodel.eval()
    lmodel.to(device=device)

    # prepare image for model input
    img = process_image(image_path)
    ten_img = torch.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
    ten_img[0] = img
    ten_img = ten_img.to(device)

    # do prediction
    with torch.no_grad():
        loutput = lmodel.forward(ten_img)
    predicted_classes = torch.exp(loutput)

    # Top class (it was not easy figuring this out...)
    top_probs, top_classes = predicted_classes.topk(topk)
    top_probs = np.around(top_probs, decimals=4)

    # move data to CPU if not already in CPU
    if device == 'CPU':
        top_probs = top_probs.detach().numpy().tolist()[0]
        top_classes = top_classes.detach().numpy().tolist()[0]
    else:
        top_probs = top_probs.cpu().detach().numpy().tolist()[0]
        top_classes = top_classes.cpu().detach().numpy().tolist()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in lmodel.class_to_idx.items()}
    top_classes = [idx_to_class[lab] for lab in top_classes]

    #
    # Set up plot
    #     plt.figure(figsize=(6, 10))
    #     ax = plt.subplot(2, 1, 1)

    #     imshow(img, ax, title='title');
    # Plot flower
    # Plot bar chart
    #     plt.subplot(2, 1, 2)

    # idx_to_class = {val: key for key, val in lmodel.class_to_idx.items()}
    # print(idx_to_class)

    with open(categories, 'r') as f:
        cat_to_name = json.load(f)

    top_flowers = [cat_to_name[lab] for lab in top_classes]
    top_probs = np.around(top_probs, 2)

    print('Top Flowers probability:')
    i = 0
    while i < topk:
        print(top_flowers[i] + ': ' + str(top_probs[i]))
        i = i + 1
    #     print(top_flowers[0])

    #     print(top_classes)

    #     sns.barplot(x=top_probs, y=top_flowers, color=sns.color_palette()[0]);
    #     plt.show()

    return top_probs, top_classes


def create_and_train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Create dataLoader
    dataLoaders, image_datasets = create_dataloaders(data_dir)

    # Set device as GPU or CPU depending on available
    device = set_device(gpu)

    # Create the network, define the criterion and optimizer
    iModel, criterion, optimizer = create_model(arch, learning_rate, device, hidden_units)

    # train the model
    modelTrain(iModel, dataLoaders['train'], dataLoaders['valid'], criterion, optimizer, epochs, 32, device)

    # test model
    modelTest(iModel, dataLoaders['test'], device)

    # save model
    saveModel(iModel, arch, image_datasets['train'], save_dir)

    print('THE END')

