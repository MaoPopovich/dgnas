import torch
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os

class EmbeddingDataset:
    """
    Takes in CLIP embeddings (INPUTS), labels, and CLIP text embedding (TEXT_EMB of shape (num_domains, clip emb shape)).
    Weakly labels the domain using the text embeddings 
    TODO: try softlabels
    """
    def __init__(self, inputs, labels, groups, dom_gt, use_dom_gt=False, text_emb=[]):
        self.inputs, self.labels, self.groups = inputs, labels, groups
        self.text_emb = text_emb
        _, self.embedding_dim = inputs.shape
        if use_dom_gt:
            print("==> Using domain GT labels")
            self.domain_labels = dom_gt
        else:
            print("==> Using domain CLIP labels")
            self.domain_labels = self.get_labels(self.text_emb, self.inputs) if len(self.text_emb) > 0 else np.array([0 for i in range(len(self.inputs))])
            print("domain labels", np.unique(np.array(self.domain_labels)))
        self.num_classes, self.num_domains = len(set(self.labels)), len(set(self.domain_labels))
        # get class weights for upweighting
        self.class_weights = self.get_counts(self.labels)
        self.dom_weights = self.get_counts(self.domain_labels)
        print(len(self.inputs), len(self.labels), len(self.domain_labels))
        assert len(self.inputs) == len(self.labels) == len(self.domain_labels), f"input, label, and domain label lengths don't match"

    @staticmethod
    def get_counts(labels):
        values, counts = np.unique(labels, return_counts=True)
        sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
        values, counts = [ list(tuple) for tuple in  sorted_tuples]
        fracs   = 1 / torch.Tensor(counts)
        return fracs / torch.max(fracs)

    @staticmethod
    def get_labels(text_emb, inputs):
        """ Gets weak domain labels given CLIP text embeddings """
        if len(text_emb.shape) == 3:
            text_emb = torch.mean(text_emb, dim = 0)
            print("new text emb ", text_emb.shape)
        similarity = (100.0 * torch.Tensor(inputs).to('cuda').float() @ text_emb.T.to('cuda').float()).softmax(dim=-1)
        values, indices = similarity.topk(1)
        return [i[0].item() for i in indices]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class VLCS(Dataset):
    def __init__(self, base_path, site, transform=None):
        self.dir_path = base_path + "/VLCS/{}".format(site) + "/full"
        # Prepare Pytorch train/test Datasets
        self.dataset = ImageFolder(self.dir_path, transform=transform)

        self.transform = transform

        # img_path, label = self.dataset.imgs[666]
        # image = Image.open(img_path)
        # image = self.transform(image)

        # plt.imshow(image.permute(2, 1, 0))
        # plt.savefig("11.png")
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.imgs[idx]
        image = Image.open(img_path)

        image = self.transform(image)
        image = convert_to_rgb(image)

        return image, label

def prepare_VLCS_embedding(embedding_path, source_domain):
    augmented_path = embedding_path + "/VLCS" + source_domain + "/augmented_LADS_small"
    augmented_features = np.load(os.path.join(augmented_path, "aug_features.npy"))
    augmented_labels = np.load(os.path.join(augmented_path, "aug_labels.npy"))
    augmented_domains = np.load(os.path.join(augmented_path, "aug_domains.npy"))
    augmented_groups = np.load(os.path.join(augmented_path, "aug_groups.npy"))
    source_features = np.load(os.path.join(augmented_path, "source_features.npy"))
    source_labels = np.load(os.path.join(augmented_path, "source_labels.npy"))
    source_domains = np.load(os.path.join(augmented_path, "source_domains.npy"))
    source_groups = np.load(os.path.join(augmented_path, "source_groups.npy"))

    augmented_dataset = EmbeddingDataset(augmented_features, augmented_labels, augmented_domains, augmented_groups)
    source_dataset = EmbeddingDataset(source_features, source_labels, source_domains, source_groups)
    return source_dataset, augmented_dataset

def prepare_VLCS(data_base_path, train_data_ratio=0.1):
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
    ])
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # Prepare Pytorch train/test Datasets

    caltech_dataset = VLCS(data_base_path, "CALTECH", transform)
    labelme_dataset = VLCS(data_base_path, "LABELME", transform)
    pascal_dataset = VLCS(data_base_path, "PASCAL", transform)
    sun_dataset = VLCS(data_base_path, "SUN", transform)
    # # Check dataset sizes
    # print(f"CALTECH Dataset: {len(caltech_dataset)}")
    # print(f"LABELME Dataset: {len(labelme_dataset)}")
    # print(f"PASCAL Dataset: {len(pascal_dataset)}")
    # print(f"SUN Dataset: {len(sun_dataset)}")

    # split dataset (train and test)
    caltech_train, caltech_testset = split_train_test_dataset(caltech_dataset, train_data_ratio)
    labelme_train, labelme_testset = split_train_test_dataset(labelme_dataset, train_data_ratio)
    pascal_train, pascal_testset = split_train_test_dataset(pascal_dataset, train_data_ratio)
    sun_train, sun_testset = split_train_test_dataset(sun_dataset, train_data_ratio)

    caltech_train.transform = train_transform
    labelme_train.transform = train_transform
    pascal_train.transform = train_transform
    sun_train.transform = train_transform

    # return train_datasets, val_datasets, full_datasets
    train_datasets = [caltech_train, labelme_train, pascal_train, sun_train]
    val_datasets = [caltech_testset, labelme_testset, pascal_testset, sun_testset]
    full_datasets = [caltech_dataset, labelme_dataset, pascal_dataset, sun_dataset]
    return train_datasets, val_datasets, full_datasets

def convert_to_rgb(image):
    if image.shape[0] == 1:  # Check if the image is grayscale
        image = image.repeat(3, 1, 1)  # Convert to RGB by repeating the channel
    return image

def split_train_test_dataset(dataset, train_data_ratio):
    data_len = len(dataset)
    test_len = int(data_len * (1 - train_data_ratio))
    train_len = len(dataset) - test_len

    # split dataset  (train and test)
    trainset, testset = torch.utils.data.random_split(dataset, [train_len, test_len])
    return trainset, testset

if __name__ == '__main__':
    base_path = "/home/qinghua_mao/work/DomainBed/domainbed/data"
    bs = 32
    train_dataset, test_dataset, full_dataset = prepare_VLCS(base_path, bs)