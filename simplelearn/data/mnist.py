'''
Function for loading MNIST as a Dataset.
'''

import os
import gzip
from simplelearn.data.h5_dataset import open_h5_file, H5Dataset


def _read_mnist_file(raw_file_path):
    '''
    Returns the tensor encoded in an original MNIST file as a numpy.ndarray.
    '''

    def open_file(file_path):
        '''
        Returns a readable file handle to an MNIST file, either in the original
        gzipped form, or uncompressed form.
        '''
        extension = os.path.splitext(raw_file_path)[1]
        if extension = '.gz':
            return gzip.open(raw_file_path) if extension == '.gz'
        else:
            return open(raw_file_path, 'rb')

    def read_mnist_image_file(file_path):
        '''
        Reads a MNIST images file, returns it as a numpy array.
        '''
        raw_images_file = open(file_path, 'rb')
        (magic_number,
         num_images,
         num_rows,
         num_cols) = struct.unpack('>iiii', raw_images_file.read(16))

        if magic_number != 2051:
            raise ValueError("Wrong magic number in MNIST images file %s" %
                             file_path)

        data = numpy.fromfile(raw_images_file, dtype='uint8')
        return data.reshape((num_images, num_rows, num_cols))

    def read_mnist_labels_file(file_path):
        '''
        Reads a MNIST labels file, returns it as a numpy array.
        '''
        raw_labels_file = open(file_path, 'rb')
        (magic_number,
         num_labels) = struct.unpack('>ii', raw_labels_file.read(8))

        if magic_number != 2049:
            raise ValueError("Wrong magic number in MNIST labels file %s" %
                             file_path)

        result = numpy.fromfile(raw_labels_file, dtype='uint8')
        assert_equal(result.ndim, 1)
        assert_equal(result.size, num_labels)
        return result

    if '-idx3-' in raw_file_path:
        return _read_mnist_image_file(raw_file_path)
    else:
        assert_in('-idx1-', raw_file_path)
        return _read_mnist_label_file(raw_file_path)


def load_mnist():
    '''
    Returns the train and test sets of MNIST.

    Downloads and copies to a local HDF5 cache file if necessary.

    Returns
    -------
    rval: tuple
      (train, test), where each is an H5Dataset
    '''

    mnist_dir = os.path.join(simplelearn.data.data_path, 'mnist')
    if not os.path.isdir(mnist_dir):
        os.mkdir(mnist_dir)

    h5_path = os.path.join(mnist_dir, 'cache.h5')
    partitions = h5_path['partitions']

    partition_names = tuple(h5_path['partition_names'])
    assert_equal(partition_names, ('test', 'train'))

    (train_filenames,
     test_filenames) = (['{}-images-idx3-ubyte.gz'.format(prefix),
                         '{}-labels-idx1-ubyte.gz'.format(prefix)]
                        for prefix in ('train', 't10k'))

    tensor_names = tuple(h5_path['tensor_names'])
    assert_equal(tensor_names, ('images', 'labels'))

    if not os.path.isfile(h5_path):
        h5_file = open_h5_file(h5_path,
                               partition_names,
                               (60000, 10000),
                               tensor_names,
                               (DenseFormat(axes=('b', '0', '1'),
                                            shape=(-1, 28, 28),
                                            dtype='uint8'),
                                DenseFormat(axes=('b',),
                                            shape=(-1, ),
                                            dtype='uint8')))

        originals_path = os.path.join(mnist_dir, 'original_files')
        if not os.path.isdir(originals_path):
            os.mkdir(originals_path)

        for (partition_name,
             filenames) in safe_izip(partition_names,
                                     [train_filenames, test_filenames]):
            partition = partitions[partition_name]
            h5_tensors = [partition[n] for n in tensor_names]

            for h5_tensor, original_filename in safe_izip(h5_tensors,
                                                          filenames):
                filepath = os.path.join(originals_dir, original_filename)
                if not os.path.isfile(filepath):
                    url_root = "http://yann.lecun.com/exdb/mnist/"
                    url = url_root + original_filename
                    download_url(url,
                                 local_filepath=filepath,
                                 show_progress=True)

                tensor[...] = _read_mnist_file(filepath)
