'''
Functions and classes for loading, saving, and iterating through NORB datasets.
'''

from simplelearn.nodes import Node
import simplelearn.data.Hdf5Dataset


def _read_norb_file(filepath):
    '''
    Reads a NORB file and returns a numpy.ndarray with its contents.

    Parameters
    ----------
    A raw NORB file, of '.mat' or '.mat.gz' format.

    Returns
    -------
    rval: numpy.ndarray
      A tensor with the shape and dtype determined by the NORB file.
    '''
    if not filepath.endswith('.mat') or filepath.endswith('.mat.gz'):
        raise ValueError("Expected filename extension '.mat' or '.mat.gz', "
                         "but got %s." filepath)

    def read_numbers(file_handle, dtype, count):
        '''
        Reads some numbers from a binary file and returns them as a
        numpy.ndarray.

        Parameters
        ----------

        file_handle : file handle
          The file handle from which to read the numbers.

        num_type : str, numpy.dtype
          The dtype of the numbers.

        count : int
          Reads off this many numbers.
        '''

        num_bytes = count * numpy.dtype(num_type).itemsize
        string = file_handle.read(num_bytes)
        return numpy.fromstring(string, dtype=dtype)

    def read_header(file_handle):
        '''
        Reads the header of a NORB file, describing the tensor dtype and shape.

        Returns
        -------
        rval: tuple
          (dtype, shape), where dtype is the numpy.dtype of the tensor,
          and shape is the tuple of ints describing its shape.
        '''

        # Maps dtype keys to dtypes and their NORB isizes in bytes.
        # This is missing one mapping: 0x1E3D4C52 -> 'packed matrix'.
        # While this is mentioned in the NORB spec, it's never actually
        # used by the NORB datasets, so it's not implemented here.
        dtype_dict = {0x1E3D4C51: ('float32', 4),
                      0x1E3D4C53: ('float64', 8),
                      0x1E3D4C54: ('int32', 4),
                      0x1E3D4C55: ('uint8', 1),
                      0x1E3D4C56: ('int16', 2)}

        dtype_key = read_numbers(file_handle, 'int32', 1)
        assert_equal(dtype_key.shape, (1, ))
        dtype, dsize = dtype_dict[dtype_key[0]]

        ndim = read_numbers(file_handle, 'int32', 1)[0]
        assert_less_equal(ndim, 3)
        shape = read_numbers(file_handle, 'int32', ndim)  # max(ndim, 3))

        return dtype, shape

    file_handle = (open(filepath, 'rb')
                   if filepath.endswith('.mat')
                   else gzip.open(filepath))

    dtype, shape = read_header(file_handle)
    size = numpy.prod(shape)
    return read_numbers(file_handle, dtype, size).reshape(shape)


def _create_hdf(which_norb):
    '''
    Creates a big_norb_cache.h5 or a small_norb_cache.h5 file.

    Parameters
    ----------
    which_norb: 'big' or 'small'

    Returns
    -------
    rval: None
    '''

    assert_in(which_norb, ('big', 'small'))

    norb_directory = os.path.join(simplelearn.data_path,
                                  '%s_norb' % which_norb)
    if not os.path.isdir(norb_directory):
        os.mkdir(norb_directory)
        raise IOError("Created missing designated directory %s. "
                      "Please copy the raw NORB files into it.")

    def get_norb_filenames(which_norb, which_set):
        '''
        Returns the norb filenames if they're in the default directory.

        If one or more are missing, this throws an IOError.

        This checks for both expected file suffixes (.mat, .mat.gz).

        Returns
        -------
        rval: tuple
          (dat_files, cat_files, info_files), where each is a
          list of filenames.
        '''
        def get_norb_filenames_gz(which_norb, which_set):
            '''
            Returns the expected list of norb file paths, using the
            .mat.gz suffix. Does not check whether the files exist.

            Parameters
            ----------
            which_norb: 'big' or 'small'
            which_set: 'test' or 'train'

            Returns
            -------
            rval: tuple
              (dat_files, cat_files, info_files), where each is a
              list of filenames.
            '''
            if which_norb == 'small':
                prefix = 'smallnorb'
                dim = 96
                file_num_strs = ['']
            else:
                prefix = 'norb'
                dim = 108
                file_nums = range(1, 3 if which_set == 'test' else 11)
                file_num_strs = ['-%02d' % n for n in file_nums]

            suffix = 'mat.gz'

            dat_files = []
            cat_files = []
            info_files = []

            for (file_type,
                 file_list) in safe_izip(('dat', 'cat', 'info'),
                                         (dat_files, cat_files, info_files)):
                for file_num_str in file_num_strs:
                    filename = ('{prefix}-5x{instances}x9x18x6x2x{dim}x{dim}'
                                '{file_num_str}{file_type}.{suffix}'
                                % {'prefix': prefix,
                                   'instances': instances,
                                   'dim': dim,
                                   'file_type': file_type,
                                   'suffix': suffix})

            return dat_files, cat_files, info_files

        dat_files, cat_files, info_files = get_norb_filenames_gz(which_norb,
                                                                 'train')

        def all_files_exist(dat_files, cat_files, info_files):
            for files in (dat_files, cat_files, info_files):
                if not all(os.path.isfile(f) for f in files):
                    return False

            return True

        # If the .mat.gz files aren't there, check for .mat files
        if not all_files_exist(dat_files, cat_files, info_files):
            # chop off the .gz suffix
            dat_files, cat_files, info_files = ([f[:-3] for f in files]
                                                for files in (dat_files,
                                                              cat_files,
                                                              info_files))
            if not all_files_exist(dat_files, cat_files, info_files):
                raise IOError("Couldn't find the raw NORB files in %s, in "
                              "either .mat or .mat.gz format." %
                              norb_directory)

        return (dat_files, cat_files, info_files)

    def add_dataset(which_norb, which_set, hdf_file):
        assert_in(which_norb, ('big', 'small'))
        assert_in(which_set, ('train', 'test'))

        group = hdf_file.create_group(which_set)

        def get_num_examples(which_norb, which_set):
            if which_norb == 'small':
                examples_per_file = 24300
                num_files = 2
            else:
                examples_per_file = 29160
                num_files = 10 if which_set == 'train' else 2

            return examples_per_file * num_files

        num_examples = get_num_examples(which_norb, which_set)

        image_dim = 96 if which_norb == 'small' else 108

        images = add_tensor('images',
                            (num_examples, 2, image_dim, image_dim),
                            'uint8',
                            ('b', 's', '0', '1'),
                            group)

        label_size = 5 if which_norb == 'small' else 11

        labels = add_tensor('labels',
                            (num_examples, label_size),
                            'int32',
                            ('b', 'f'),
                            group)

        cat_files, dat_files, info_files = get_norb_filenames(which_norb,
                                                              which_set)

        index = 0
        for cat_file, dat_file, info_file in safe_izip(cat_files,
                                                       dat_files,
                                                       info_files):
            index_range = slice(index, index + examples_per_file)

            with _read_norb_file(dat_file) as dat:
                assert_equal(dat.dtype, images.dtype)
                images[index_range, :, :, :] = dat

            with _read_norb_file(cat_file) as cat:
                assert_equal(cat.dtype, labels.dtype)
                labels[index_range, 0] = cat

            with _read_norb_file(info_file) as info:
                assert_equal(info.dtype, labels.dtype)
                labels[index_range, 1:] = info

    hdf_path = os.path.join(norb_filepath, '%s_norb_cache.h5' % which_norb)
    with h5py.File(hdf_path, 'w') as hdf_file:  # w for overwrite, w- for no
        add_dataset('train', hdf_file)
        add_dataset('test', hdf_file)
