import tensorstore as ts
import numpy as np
import time

dataset = ts.open(
    {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': '/Users/seth.hinz/Documents/source/czpeedy/big/output'},
     'metadata': {'compressor': {'id': 'blosc', 'cname': 'blosclz', 'shuffle': 0, 'clevel': 1}, 'dtype': '<u2',
                  'shape': (5, 512, 512, 512), 'chunks': [1, 128, 128, 128]}, 'create': True, 'delete_existing': True}).result()

for i in range(5):
    print(f"writing data {i}:")
    data = np.random.randint(0, 9, (512, 512, 512), dtype=np.uint16) + 10 * i
    dataset[i].write(data).result()
    time.sleep(6)