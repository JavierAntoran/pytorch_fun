from __future__ import print_function
import ConfigParser
import collections
import h5py, sys
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

# ----------------------------------------------------------------------------------------------------------------------
def read_config(file_cfg, cfg_item_list, config=None):
    if config is None:
        config = collections.OrderedDict()  # config = {}
    try:
        print("read config %s ..." % file_cfg)
        parser = ConfigParser.RawConfigParser()
        parser.read(file_cfg)
    except:
        print("Error reading %s" % file_cfg)

    print("config parsing %s ..." % file_cfg)
    for line in cfg_item_list:
        level, item, type = line
        try:
            if type == 'str':
                config[item] = parser.get(level, item)
            elif type == 'int':
                config[item] = parser.getint(level, item)
            elif type == 'boolean':
                config[item] = parser.getboolean(level, item)
            elif type == 'float':
                config[item] = parser.getfloat(level, item)
        except:
            cprint('r', 'cfg param not found: %s' % item)
    return config

# ----------------------------------------------------------------------------------------------------------------------
suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes)
    return '%s%s' % (f, suffixes[i])


# ----------------------------------------------------------------------------------------------------------------------
def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()

# ----------------------------------------------------------------------------------------------------------------------
def save_list_array(list, filename):
    with h5py.File(filename, 'w') as hf:
        for i in range(len(list)):
            if isinstance(list[i], np.ndarray):
                hf.create_dataset('_array_%d' % i, data=list[i])
            else:
                hf.create_dataset('_array_%d' % i, data=list[i].get_value())


# ----------------------------------------------------------------------------------------------------------------------
def load_list_array(list, filename):
    with h5py.File(filename, 'r') as hf:
        # print('List of items in the base directory:', hf.items())
        for i in range(len(list)):
            if isinstance(list[i], np.ndarray):
                list[i] = np.array(hf.get('_array_%d' % i))
            else:
                list[i].set_value(np.array(hf.get('_array_%d' % i)))


# ----------------------------------------------------------------------------------------------------------------------
def save_obj(obj, filename):
    with open(filename, 'w') as f:
        pickle.dump(obj, f, protocol=2)


# ----------------------------------------------------------------------------------------------------------------------
def load_obj(filename):
    with open(filename, 'r') as f:
        return pickle.load(f)


def get_num_batches(nb_samples, batch_size, roundup=True):
    if roundup:
        return (nb_samples + (-nb_samples % batch_size)) / batch_size  # roundup division
    else:
        return nb_samples / batch_size


# ----------------------------------------------------------------------------------------------------------------------
def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(nb_samples)
    for i in range(get_num_batches(nb_samples, batch_size, roundup)):
        yield ind[i * batch_size: (i + 1) * batch_size]


# ----------------------------------------------------------------------------------------------------------------------
def generate_dataset(X, Y, batch_size=1, random=True, scale=1.0,
                     one_hot=False, nb_class=None, **kwargs):
    nb_samples = X.shape[0]
    print("nb_samples: %d" % nb_samples)
    for ind_sample in generate_ind_batch(nb_samples, batch_size, random):
        Xn = X[ind_sample, :]
        Xn /= scale
        if one_hot:
            Yn = np.zeros((len(ind_sample), nb_class))
            Yn[np.arange(len(ind_sample)), Y[ind_sample]] = 1
        else:
            Yn = Y[ind_sample, :]
        yield [Xn, Yn]


# ----------------------------------------------------------------------------------------------------------------------
def one_hot(y, nb_classes):
    nb_samples = y.shape[0]
    Y = np.zeros((nb_samples, nb_classes))
    Y[np.arange(nb_samples), y] = 1
    return Y


def flip_ndarray(x, chanel_left=True):
    flip = np.random.randint(2, size=x.shape[0])
    y = np.copy(x)
    for n in range(x.shape[0]):
        if flip[n] == 1:
            if chanel_left:
                y[n, :, :, :] = y[n,: , :, ::-1]  # (n, c, y, x)
            else:
                y[n, :, :, :] = y[n, :, ::-1, :]  # (n, y, x, c)
    return y


def crop_ndarray(x, pad=1, chanel_left=True):
    pad_nx = np.random.randint(2 * pad + 1, size=x.shape[0])
    pad_ny = np.random.randint(2 * pad + 1, size=x.shape[0])
    y = np.copy(x)
    if chanel_left:  # (n, c, x, y)
        for n in range(x.shape[0]):
            for c in range(x.shape[1]):
                x_pad = np.pad(x[n, c], pad_width=((pad, pad), (pad, pad)), mode='constant')
                y[n, c, :, :] = x_pad[pad_nx[n]:pad_nx[n] + x.shape[2], pad_ny[n]:pad_ny[n] + x.shape[3]]
    else:  # (n, x, y, c)
        for n in range(x.shape[0]):
            for c in range(x.shape[3]):
                x_pad = np.pad(x[n, :, :, c], pad_width=((pad, pad), (pad, pad)), mode='constant')
                y[n, :, :, c] = x_pad[pad_nx[n]:pad_nx[n] + x.shape[1], pad_ny[n]:pad_ny[n] + x.shape[2]]
    return y
