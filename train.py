# Imports here
import torch
import torchvision 
from torch import nn,optim
from torchvision import datasets, transforms, models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import seaborn as sns
from PIL import Image
import json
import matplotlib.pyplot as plt
import argparse



def main():
    in_arg=get_input_args()
    alexnet = models.alexnet(pretrained=True)
    vgg13 = models.vgg13(pretrained=True)

    modela = {'alexnet': alexnet, 'vgg': vgg13}
    
    data_dir = in_arg.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms =transforms.Compose([transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                         transforms.RandomRotation(40),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valTest_transforms=transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data=datasets.ImageFolder(train_dir,transform=train_transforms)

    valid_data=datasets.ImageFolder(valid_dir,transform=valTest_transforms)

    test_data=datasets.ImageFolder(test_dir,transform=valTest_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)

    validloader =torch.utils.data.DataLoader(valid_data,batch_size=64)

    testloader=torch.utils.data.DataLoader(test_data,batch_size=64)
    
    
    model=modela[in_arg.arch]
    
    for p in model.parameters():
        p.requierd_grad=False
    input_f=None
    if in_arg.arch== 'vgg':
        input_f=25088
    else:
        input_f=9216

    classifier=nn.Sequential(nn.Linear(input_f,in_arg.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.4),
                            nn.Linear(in_arg.hidden_units,102),
                           nn.LogSoftmax(dim=1))
    model.classifier=classifier
    device = in_arg.gpu
    
    criterion = nn.NLLLoss()
    lr=in_arg.learning_rate
    optimizer=optim.Adam(model.classifier.parameters(),lr=lr)
    
    model.to(device)
    
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/(print_every):.3f}.. "
                      f"validation loss: {val_loss/len(validloader):.3f}.. "
                      f"validation accuracy: {(accuracy/len(validloader))*100:.1f}%")
                running_loss = 0
                model.train()
    
    
    checkpoint = {'classifier': model.classifier,
              'learning_rate': lr,
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx,
              'optimizer_dict': optimizer.state_dict()}
    torch.save(checkpoint, in_arg.save_dir+'/checkpoint0.pth')
def get_input_args():
    parser = argparse.ArgumentParser(description='Image Classifier Command Line Application')
    # code to read all arguments here
    parser.add_argument('data_directory', type=str,
                        help='give data dirctory')
    parser.add_argument('--save_dir', type=str, default='ImageClassifier', 
                        help='Set directory to save checkpoints')
    
    parser.add_argument('--arch',type=str, default = 'alexnet', 
                       help='Choose architecture')
    parser.add_argument('--learning_rate',type=float, default = 0.003, 
                       help='Set learning_rate')
    parser.add_argument('--hidden_units',type=int, default = 4096, 
                       help='set number of hidden_units ')
    parser.add_argument('--epochs',type=int, default = 2, 
                       help='set number of epochs  ')
    parser.add_argument('--gpu',type=str, default = 'cuda', 
                       help='Use GPU for training  ')
    return parser.parse_args()

if __name__ == "__main__":
    main()