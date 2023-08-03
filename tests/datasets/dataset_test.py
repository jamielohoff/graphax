from tqdm import tqdm

from torch.utils.data import DataLoader
from graphax.dataset import GraphDataset

dataset = GraphDataset("/Users/grieser/Projects/alphagrad/src/alphagrad/data/_samples")
print(len(dataset))
loader = DataLoader(dataset, 2048, num_workers=16, shuffle=False, drop_last=True)
for content in tqdm(loader):
    print(content)

