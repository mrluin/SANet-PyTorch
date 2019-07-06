import torch
from torch.utils.data import DataLoader

def mean_std_calculator(dataset):

    # mean and std all both from the data used for training, excluding the test data

    loader = DataLoader(
        dataset,
        batch_size = 64,
        num_workers = 1,
        shuffle = False
    )

    mean = 0.
    std = 0.
    num_samples = 0.

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1) # N C HxW
        mean += data.mean(2).sum(0) # channel-wise independent
        std += data.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    return mean, std

def unbalanced_weight_calculator(dataset):

    """
     Return the unbalanced_weight of the given dataset
    """
    # after mapping
    loader = DataLoader(
        dataset,
        batch_size = 10,
        num_workers = 1,
        shuffle = False
    )
    # calculate class_samples_count
    class_samples_count = torch.tensor([0, 0, 0, 0, 0, 0])
    for _, target in loader:

        # here target from data.Dataset, the type of target is tensor
        for t in torch.unique(target, sorted=True):
            class_samples_count[t] += (target == t).sum()

    weight = 1. / class_samples_count.float()
    return class_samples_count, weight

def class_weight_calcularot(samples_count):

    p_class = torch.tensor([samples_count[k].float() / samples_count.sum().float() for k in range(6)])
    print(p_class)
    balanced_weight = 1. / torch.log(p_class + 1.12)

    return balanced_weight









