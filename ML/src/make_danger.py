'''
This script uses two datasets publicly available on github.
One of these datasets is a collection of images of dangerous objects. Some of these classes are not a threat in our context (eg. alcohol, blood, cigarette), while others are (eg. gun, knife, insulting gesture).
This can be found at: https://github.com/poori-nuna/HOD-Benchmark-Dataset/tree/main/dataset/class
The other dataset is just neutral images of people (originally intended for pose estimation). This can be found at:
https://github.com/VikramShenoy97/Human-Segmentation-Dataset/tree/master/Training_Images

These datasets are several GBs in size, and so they will not be included in the repository, since it is shared for multiple projects.
If it is desired to run this script, the datasets must be downloaded from the links given and placed into the directories specified in the script.
For the github repository, the final `.pt` files will be included for use in the model.
'''
from torchvision import transforms
from PIL import Image
import torch
import os
import random

base_path = '../data/danger/raw'
classes_path = os.path.join(base_path, 'classes')
neutral_path = os.path.join(base_path, 'neutral')

# There are 6 classes in the danger dataset, but only 3 are relevant for substantiating a threat a a door. The other 3 will be reapproriated as negative examples.
danger_roots = ['insulting_gesture', 'gun', 'knife']
safe_roots = ['alcohol', 'blood', 'cigarette']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def get_images_in_directory(directory):
    images = []
    if not os.path.exists(directory):
        print(f"Warning: Directory does not exist: {directory}")
        return images

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):  
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path).convert("RGB")  
                    img = transform(img)
                    images.append(img)
                except Exception as e:
                    print(f"Failed to process image {file_path}: {e}")
    return images

def get_danger_images():
    images = []
    for root in danger_roots:
        danger_path = os.path.join(classes_path, root)
        images.extend(get_images_in_directory(danger_path))
    print(f'Number of danger images: {len(images)}')
    return images

def get_safe_images():
    images = []
    for root in safe_roots:
        safe_path = os.path.join(classes_path, root)
        images.extend(get_images_in_directory(safe_path))
    print(f'Number of safe images: {len(images)}')
    return images

def get_neutral_images():
    images = get_images_in_directory(neutral_path)
    print(f'Number of neutral images: {len(images)}')
    return images

def make_binary_dataset(danger_images, safe_images):
    danger_images = torch.stack(danger_images)
    danger_labels = torch.ones(danger_images.shape[0])

    safe_images = torch.stack(safe_images)
    safe_labels = torch.zeros(safe_images.shape[0])

    random.shuffle(danger_images)
    random.shuffle(safe_images)
    
    train_images = torch.cat((danger_images[:int(0.8 * len(danger_images))], safe_images[:int(0.8 * len(safe_images))]))
    train_labels = torch.cat((danger_labels[:int(0.8 * len(danger_images))], safe_labels[:int(0.8 * len(safe_images))]))

    test_images = torch.cat((danger_images[int(0.8 * len(danger_images)):], safe_images[int(0.8 * len(safe_images)):]))
    test_labels = torch.cat((danger_labels[int(0.8 * len(danger_images)):], safe_labels[int(0.8 * len(safe_images)):]))

    return train_images, train_labels, test_images, test_labels
    

def main():
    danger_images = get_danger_images()
    safe_images = get_safe_images()

    neutral_images = get_neutral_images()
    safe_images.extend(neutral_images)

    random.shuffle(safe_images)
    safe_images = safe_images[:len(danger_images)]

    print(f'Final number of danger images: {len(danger_images)}')
    print(f'Final number of safe images: {len(safe_images)}')

    train_images, train_labels, test_images, test_labels = make_binary_dataset(danger_images, safe_images)
    print(f'Number of training images: {train_images.shape[0]}')
    print(f'Number of test images: {test_images.shape[0]}')

    torch.save(train_images, os.path.join(base_path, 'train.pt'))
    torch.save(train_labels, os.path.join(base_path, 'train_labels.pt'))
    torch.save(test_images, os.path.join(base_path, 'test.pt'))
    torch.save(test_labels, os.path.join(base_path, 'test_labels.pt'))

if __name__ == '__main__':
    main()
