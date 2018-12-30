from collections import OrderedDict, defaultdict
import json
import logging
import os

import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from IPython import embed

writer = SummaryWriter()
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

NUM_EPOCHS = 5
BATCH_SIZE = 20
DROPOUT = 0.5
NUM_CLASSES = 102

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomVerticalFlip(0.5),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

validation_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def rename_classes_dirs(data_directory, category_name_mapping):
    root_directory = os.path.abspath(data_directory)
    for directory in os.listdir(data_directory):
        class_name = category_name_mapping.get(directory)
        if class_name:
            new_directory_name = class_name.replace(' ', '-').lower()
            os.rename(os.path.join(root_directory, directory), os.path.join(root_directory, new_directory_name))


def get_sampler_for_data_loader(dataset: datasets.ImageFolder):
    classes = [c[1] for c in dataset.imgs]
    _, class_count = np.unique(classes, return_counts=True)
    weights = 1. / torch.Tensor(class_count)
    weights /= weights.sum()  # Normalization, so weights sum to 1.0
    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(dataset.imgs))


with open('cat_to_name.json', 'r') as f:
    category_to_name_mapping = json.load(f)

rename_classes_dirs(valid_dir, category_to_name_mapping)
rename_classes_dirs(train_dir, category_to_name_mapping)

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

train_sampler = get_sampler_for_data_loader(train_dataset)
validation_sampler = get_sampler_for_data_loader(validation_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

model = models.vgg19(pretrained=True)

# Freeze features parameters (gradients will not be calculated)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Dropout(p=DROPOUT),
    nn.Linear(1024, 500),
    nn.ReLU(),
    nn.Linear(500, NUM_CLASSES)
)

# Attach new classifier
model.classifier = classifier
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters())

batch_number = 0
step_number = 0
total_training_batches = int(len(train_dataset) / BATCH_SIZE)

train_losses, validation_losses = [], []
for e in range(NUM_EPOCHS):
    running_loss = 0
    for images, labels in train_loader:
        if batch_number % 10 == 0:
            logging.info('Batch number {}/{}...'.format(batch_number, total_training_batches))
        batch_number += 1
        step_number += 1

        # Pass this computations to selected device
        images = images.cuda()
        labels = labels.cuda()

        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()

        # Forwards pass, then backward pass, then update weights
        probabilities = model(images)
        loss = criterion(probabilities, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        validation_loss = 0
        min_validation_loss = 1000
        accuracy = 0

        # Turn off gradients for testing
        with torch.no_grad():
            # set model to evaluation mode
            model.eval()
            for images, labels in validation_loader:
                # Pass this computations to selected device
                images = images.cuda()
                labels = labels.cuda()

                probabilities = model(images)
                validation_loss += criterion(probabilities, labels)

                # Get the class probabilities
                ps = torch.softmax(probabilities, dim=1)

                # Get top probabilities
                top_probability, top_class = ps.topk(1, dim=1)

                # Comparing one element in each row of top_class with
                # each of the labels, and return True/False
                equals = top_class == labels.view(*top_class.shape)

                # Number of correct predictions
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            # Set model to train mode
        model.train()

        train_losses.append(running_loss / len(train_loader))
        validation_losses.append(validation_loss / len(validation_loader))

        # Get minimum validation loss
        min_validation_loss = min(validation_losses)

        # Counting the losses
        training_loss = running_loss / len(validation_loader)
        validation_loss = validation_loss / len(validation_loader)

        writer.add_scalar('data/train_loss', training_loss, step_number)
        writer.add_scalar('data/validation_loss', validation_loss, step_number)
        writer.add_scalar('data/validation_accuracy', (accuracy / len(validation_loader)) * 100, step_number)

        logging.info("Epoch: {}/{}.. ".format(e + 1, NUM_EPOCHS))
        logging.info("Training Loss: {:.3f}.. ".format(training_loss))
        logging.info("Validation Loss: {:.3f}.. ".format(validation_loss))
        logging.info("Validation Accuracy: {:.3f}%".format((accuracy / len(validation_loader)) * 100))

        # Save model if validation loss have decreased
        if validation_loss <= min_validation_loss:
            logging.info("Validation has decreased, saving model")
            torch.save(model.state_dict(), 'model.pt')
            min_validation_loss = validation_loss

        batch_number = 0

writer.close()
