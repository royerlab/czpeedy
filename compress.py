# import napari
# import zarr
# from ome_zarr.io import parse_url
# from ome_zarr.writer import write_image
import numpy as np
import tensorstore as ts
# import blosc
import time

# Define the shape of the tensor based on known dimensions
shape = (1928, 1440, 2048)

# Load raw data into numpy array
with open('T_14.V_0.(1928x1440x2048).raw', 'rb') as f:
    data = np.fromfile(f, dtype=np.uint16).reshape(shape)

print(data.shape)
# viewer = napari.view_image(data, ndisplay=3)
# napari.run()

# # OME-Zarr
# path = "output.zarr"
# store = parse_url(path, mode="w").store
# root = zarr.group(store=store)
# write_image(image=data, group=root, axes="zyx")

# # Compress with Blosc
# bytes_array = data.tobytes()
# print(len(bytes_array))
# print("start")
# start_time = time.time()
# result_size = 0
# for i in range(0, int(len(bytes_array) / 2147483631)):
#     compressed = blosc.compress(bytes_array[(i * 2147483631):min(len(bytes_array), (i + 1) * 2147483631)], typesize=2, shuffle=blosc.BITSHUFFLE)
#     result_size += len(compressed)
# elapsed = time.time() - start_time
# print("Compressed")
# print(elapsed)
# print(result_size)
# print(result_size / len(bytes_array))
# # ~1.3GBps!
# # not much speed diff as a function of clevel. keep clevel 9 for smallest result size
# # Using bitshuffle takes us from 0.58 to 0.35 compression. very solid, can probably do better if we
# # drop from 16 bits to 12 bits

# Write to disk with tensorstore
# dataset = ts.open({
#     'driver': 'zarr',
#     'kvstore': {
#         'driver': 'file',
#         'path': 'D:\\Seth\\tmp/',
#     },
#     'metadata': {
#         'compressor': {
#             'id': 'blosc',
#             'cname': 'lz4',
#             'shuffle': 2
#         },
#         'dtype': '>u2',
#         'shape': shape,
#         # 'blockSize': [100, 100, 100],
#     },
#     'create': True,
#     'delete_existing': True,
#     # 'dtype': 'uint16'
# }).result()

print("ts open")
# around 30 s
# dataset = ts.open({
#     'driver': 'zarr3',
#     'kvstore': {
#         'driver': 'file',
#         'path': 'D:\\Seth\\tmp/',
#     },
#     'metadata': {
#         "shape": shape,
#         "chunk_grid": {
#             "name": "regular",
#             "configuration": {"chunk_shape": [400, 400, 1024]}
#             # "configuration": {"chunk_shape": [482, 480, 512]} # doesn't write at all?????
#         },
#         "chunk_key_encoding": {"name": "default"},
#         # "codecs": [{"name": "blosc", "configuration": {"cname": "lz4", "shuffle": "bitshuffle"}}],
#         "data_type": "uint16",
#     },
#
#     # 'metadata': {
#     #     'compressor': {
#     #         'id': 'blosc',
#     #         'cname': 'lz4',
#     #         'shuffle': 2
#     #     },
#     #     'dtype': '>u2',
#     #     'shape': shape,
#     #     # 'blockSize': [100, 100, 100],
#     # },
#     'create': True,
#     'delete_existing': True,
#     # 'dtype': 'uint16'
#     },
#     # Due to a (i think) bug in tensorstore, you have to put the codec separate from the other metadata
#     # or it fails to merge the expected codecs ([]) and the given codecs (not empty array)
#     codec=ts.CodecSpec({
#       "codecs": [{"name": "blosc", "configuration": {"cname": "lz4", "shuffle": "bitshuffle"}}],
#       'driver': 'zarr3',
#     })).result()

# # 16 s (zarr2 is faster?)
# dataset = ts.open({
#     'driver': 'zarr',
#     'kvstore': {
#         'driver': 'file',
#         'path': 'D:\\Seth\\tmp/',
#     },
#
#     'metadata': {
#         'compressor': {
#             'id': 'blosc',
#             'cname': 'lz4',
#             'shuffle': 2
#         },
#         'dtype': '>u2',
#         'shape': shape,
#         "chunks": [482, 480, 1024],
#         # 'blockSize': [100, 100, 100],
#     },
#     'create': True,
#     'delete_existing': True,
#     # 'dtype': 'uint16'
# }).result()

# # 13 s (zarr2 is faster and byte shuffle is faster)
# dataset = ts.open({
#     'driver': 'zarr',
#     'kvstore': {
#         'driver': 'file',
#         'path': 'D:\\Seth\\tmp/',
#     },
#
#     'metadata': {
#         'compressor': {
#             'id': 'blosc',
#             'cname': 'lz4',
#             'shuffle': 1
#         },
#         'dtype': '>u2',
#         'shape': shape,
#         "chunks": [482, 480, 2048],
#         # 'blockSize': [100, 100, 100],
#     },
#     'create': True,
#     'delete_existing': True,
#     # 'dtype': 'uint16'
# }).result()

# # 8-10 s (zarr2 / byte shuffle / lz4->blocslz / clevel 2 for worse compression faster)
# dataset = ts.open({
#     'driver': 'zarr',
#     'kvstore': {
#         'driver': 'file',
#         'path': 'D:\\Seth\\tmp/',
#     },
#
#     'metadata': {
#         'compressor': {
#             'id': 'blosc',
#             'cname': 'blosclz',
#             'shuffle': 1,
#             'clevel': 2
#         },
#         'dtype': '>u2',
#         'shape': shape,
#         "chunks": [482, 480, 2048],
#         # 'blockSize': [100, 100, 100],
#     },
#     'create': True,
#     'delete_existing': True,
#     # 'dtype': 'uint16'
# }).result()

# 7 s (zarr2 / byte shuffle / blocslz / clevel 4 / native endianness)
dataset = ts.open({
    'driver': 'zarr',
    'kvstore': {
        'driver': 'file',
        'path': 'D:\\Seth\\tmp/',
    },

    'metadata': {
        'compressor': {
            'id': 'blosc',
            'cname': 'blosclz',
            'shuffle': 1,
            'clevel': 4
        },
        'dtype': '<u2',
        'shape': shape,
        "chunks": [482, 480, 2048],
        # 'blockSize': [100, 100, 100],
    },
    'create': True,
    'delete_existing': True,
    # 'dtype': 'uint16'
}).result()

print("ts start write")
now = time.time()
dataset.write(data).result()
elapsed = time.time() - now

print(elapsed)
# Tensorstore is super fast geez
# I tried most of the compressors - lz4 seems to be a good compromise between cpu usage, speed, and filesize.
# There is not a huge difference between the options, but having shuffle: 2 (bitshuffle) is very important
# for both speed and resultant size.
# On the ssd it takes about 15s to compress and save this 11gb stack - realized i'm using ssd not HD but the write
# speeds were only like 500MBps so probably comparable
