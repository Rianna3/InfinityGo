import tempfile
import h5py
# import keras
import os
import torch

# from keras.models import load_model, save_model

def save_model_to_hdf5_group(model,f):
    '''
    Save the pytorch model to an HDF5 file group. 
    - save the model to a temporary file
    - embed the content of this temporary file into the given HDF5 file group
    '''
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-pytorchmodel')
    tempfname += '.h5'
    try:
        os.close(tempfd)
        # save_model(model, tempfname)
        torch.save(model, tempfname)
        serialized_model = h5py.File(tempfname, 'r')
        root_item = serialized_model.get('/')
        serialized_model.copy(root_item,f,'pytorchmodel')
        serialized_model.close()
    finally:
        os.unlink(tempfname)

def load_model_from_hdf5_group(f):
    '''
    Load model from hdf5 files
    - extract the model into a temporary file.
    - use Pytorch load_model to read it
    '''
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-pytorchmodel')
    tempfname += '.h5'
    try:
        os.close(tempfd)
        serialized_model = h5py.File(tempfname,'w')
        print('The file is ',f)
        root_item = f.get('pytorchmodel')
        print('The root item is ',root_item)
        for attr_name, attr_value in root_item.attrs.items():
            serialized_model.attrs[attr_name] = attr_value
        print('The serialized model is ',serialized_model)
        print('The key of root_item ',root_item.keys())
        for k in root_item.keys():
            f.copy(root_item.get(k), serialized_model, k)
        serialized_model.close()
        print('tempfname:',tempfname)
        return torch.load(tempfname)
    finally:
        os.unlink(tempfname)

# def set_gpu_memory_target(frac):
#     '''
#     To limit Tensorflow's GPU memory usage to a specified fraction of the total avalible GPU memory
#     '''
#     if keras.backend.backend() != 'tensorflow':
#         return 
#     import tensorflow as tf
#     config = tf.ConfigProto(device_count={'GPU':1})
#     sess = tf.Session(config=config)


