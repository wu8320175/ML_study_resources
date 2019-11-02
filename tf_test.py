def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
file='./Datas/cifar-10-batches-py/data_batch_1'
datas=unpickle(file)
print(datas.keys())
# print(datas[b'data'].shape)
# print(datas[b'labels'],len(datas[b'labels']))
# print(datas[b'filenames'],len(datas[b'filenames']))
print(datas[b'batch_label'])