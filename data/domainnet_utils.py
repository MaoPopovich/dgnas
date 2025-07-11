import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets
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

class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, transform=None):
        self.dir_path = base_path + "/DomainNet/{}".format(site)
        # Prepare Pytorch train/test Datasets
        self.dataset = ImageFolder(self.dir_path, transform=transform)

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.base_path, self.paths[idx][10:])
        # # img_path = os.path.join(self.base_path, self.paths[idx])
        # label = self.labels[idx]
        # image = Image.open(img_path)
        img_path, label = self.dataset.imgs[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def prepare_domainnet_embedding(embedding_path, source_domain):
    augmented_path = embedding_path + "/DomainNet" + source_domain + "/augmented_LADS_small"
    source_features = np.load(os.path.join(augmented_path, "source_features.npy"))
    source_labels = np.load(os.path.join(augmented_path, "source_labels.npy"))
    source_domains = np.load(os.path.join(augmented_path, "source_domains.npy"))
    source_groups = np.load(os.path.join(augmented_path, "source_groups.npy"))
    augmented_features = np.load(os.path.join(augmented_path, "aug_features.npy"))
    augmented_labels = np.load(os.path.join(augmented_path, "aug_labels.npy"))
    augmented_domains = np.load(os.path.join(augmented_path, "aug_domains.npy"))
    augmented_groups = np.load(os.path.join(augmented_path, "aug_groups.npy"))

    augmented_dataset = EmbeddingDataset(augmented_features, augmented_labels, augmented_domains, augmented_groups)
    source_dataset = EmbeddingDataset(source_features, source_labels, source_domains, source_groups)
    return source_dataset, augmented_dataset


def prepare_domainnet(data_base_path, train_data_ratio=0.1):
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # clipart
    clipart_dataset = DomainNetDataset(data_base_path, 'clipart', transform=transform)
    infograph_dataset = DomainNetDataset(data_base_path, 'infograph', transform=transform)
    painting_dataset = DomainNetDataset(data_base_path, 'painting', transform=transform)
    quickdraw_dataset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform)
    real_dataset = DomainNetDataset(data_base_path, 'real', transform=transform)
    sketch_dataset = DomainNetDataset(data_base_path, 'sketch', transform=transform)
    # # check dataset sizes
    # print(f"Clipart Dataset: {len(clipart_dataset)}")
    # print(f"Infograph Dataset: {len(infograph_dataset)}")
    # print(f"Painting Dataset: {len(painting_dataset)}")
    # print(f"Quickdraw Dataset: {len(quickdraw_dataset)}")
    # print(f"Real Dataset: {len(real_dataset)}")
    # print(f"Sketch Dataset: {len(sketch_dataset)}")

    # split dataset (train and test)
    clipart_train, clipart_testset = split_train_test_dataset(clipart_dataset, train_data_ratio)
    infograph_train, infograph_testset = split_train_test_dataset(infograph_dataset, train_data_ratio)
    painting_train, painting_testset = split_train_test_dataset(painting_dataset, train_data_ratio)
    quickdraw_train, quickdraw_testset = split_train_test_dataset(quickdraw_dataset, train_data_ratio)
    real_train, real_testset = split_train_test_dataset(real_dataset, train_data_ratio)
    sketch_train, sketch_testset = split_train_test_dataset(sketch_dataset, train_data_ratio)

    clipart_train.transform = train_transform
    infograph_train.transform = train_transform
    painting_train.transform = train_transform
    quickdraw_train.transform = train_transform
    real_train.transform = train_transform
    sketch_train.transform = train_transform

    # return train_datasets, val_datasets, full_datasets
    train_datasets = [clipart_train, infograph_train, painting_train, quickdraw_train, real_train, sketch_train]
    val_datasets = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]
    full_datasets = [clipart_dataset, infograph_dataset, painting_dataset, quickdraw_dataset, real_dataset, sketch_dataset]
    return train_datasets, val_datasets, full_datasets

def split_train_test_dataset(dataset, train_data_ratio):
    data_len = len(dataset)
    test_len = int(data_len * (1 - train_data_ratio))
    train_len = len(dataset) - test_len

    # split dataset  (train and test)
    trainset, testset = torch.utils.data.random_split(dataset, [train_len, test_len])
    return trainset, testset