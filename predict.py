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
    with open(in_arg.category_name, 'r') as f:
        cat_to_name = json.load(f)
    
    optimizer,model=loadCheckpoint(in_arg.checkpoint)
    model.to(in_arg.gpu)
    top_p, top_class = predict(in_arg.img,model,in_arg.top_k)
    labels = []
    print(top_class)
    for i in top_class:
        labels.append(cat_to_name[i])
    print("Flower name:",labels[0])
    top_p=top_p.to('cpu')
    top_p=top_p.numpy()
    if(len(labels)>1):
        for i in range(len(labels)):
            print("top classes:",labels[i], f"prbablility: {top_p[i]:.3f}")
    else:
        print("top classes",labels[0], f"prbablility{top_p[0]:.3f}")
    #'flowers/valid/100/image_07904.jpg'



def get_input_args():
    parser = argparse.ArgumentParser(description='Image Classifier Command Line Application')
    # code to read all arguments here
    parser.add_argument('img', type=str,
                        help='give iamge path')
    parser.add_argument('checkpoint', type=str, 
                        help='give checkpoint directory')
    
    parser.add_argument('--top_k',type=int, default = 5, 
                       help='Return top KK most likely classes')

    parser.add_argument('--category_name',type=str, default = 'ImageClassifier/cat_to_name.json', 
                       help='mapping of categories to real names file')
    parser.add_argument('--gpu',type=str, default = 'cuda', 
                       help='Use GPU for training  ')
    return parser.parse_args()

def loadCheckpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model=models.alexnet(pretrained=True)
    for p in model.parameters():
        p.requierd_grad=False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)    
    optimizer.load_state_dict(checkpoint['optimizer_dict'])


    return optimizer, model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    np_image = transform(pil_image)
        
    return np_image

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

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    
    image=process_image(image_path)
    image=torch.tensor(image)
    image=image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        image=image.to(device)
        outputs = model.forward(image)
        ps = torch.exp(outputs)
        probs, indices = ps.topk(topk)
        probs = probs.squeeze()
        classes = [model.idx_to_class[idx] for idx in indices[0].tolist()]
    return probs, classes
'''def image_display(image, model):
    
    #creating plots
    plt.figure(figsize=(6,10))
    ax = plt.subplot(2,1,1)
    
    #ploting flower
    flower_name = image.split('/')[2]
    flower_title = str(cat_to_name[flower_name])
    
    top_p, top_class = predict(image,model,in_arg.top_k)
    labels = []
    for i in top_class:
        labels.append(cat_to_name[i])
    img = process_image(image)
    imshow(img, ax, title=flower_title)
    
    plt.subplot(2,1,2) #2nd subplot
    sns.barplot(x=top_p, y=labels, color=sns.color_palette()[0]);
    plt.show() '''
if __name__ == "__main__":
    main()