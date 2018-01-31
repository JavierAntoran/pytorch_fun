from __future__ import print_function
import ConfigParser
import collections
import pickle
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
def read_config(file_cfg, cfg_item_list, config=None):
    if config is None:
        config = collections.OrderedDict() #config = {}
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
def generate_dataset(X,Y, batch_size=1,  random=True, scale=1.0,
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

