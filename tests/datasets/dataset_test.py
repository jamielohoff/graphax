from tqdm import tqdm

from torch.utils.data import DataLoader
from graphax.dataset import GraphDataset

dataset = GraphDataset("/Users/grieser/Projects/graphax/tests/datasets", shape=[10, 50, 10])
print(len(dataset))
loader = DataLoader(dataset, 2, num_workers=2, shuffle=False, drop_last=True)
for content in tqdm(loader):
    print(content)

