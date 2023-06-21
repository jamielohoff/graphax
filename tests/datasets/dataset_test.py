from tqdm import tqdm

from torch.utils.data import DataLoader
from graphax.dataset import GraphDataset

dataset = GraphDataset("./data")
print(len(dataset))
loader = DataLoader(dataset, 768, shuffle=True, drop_last=True)
for content in tqdm(loader):
    print(content)

