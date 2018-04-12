from __future__ import print_function
import collections
import h5py, pickle, sys, os, gzip
import numpy as np

def fg_color(color):
    r,g,b = color
    color = 16 + (r * 36) + (g * 6) + b
    return '38;5;%dm' % color

def bg_color(color):
    r,g,b = color
    color = 16 + (r * 36) + (g * 6) + b
    return '48;5;%dm' % color

def cprint2(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    print("\033[%s%s%s\033[0m" % (pre_code, fg_color(color), text), **kwargs)

def cprint3(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    print("\033[%s%s%s\033[0m" % (pre_code, bg_color(color), text), **kwargs)

from matplotlib import cm
def view_image(x, max=-1, min=-1):
    if len(x.shape) == 3:
        return view_imagec(x)
        
    print(x.shape)
    if max == -1:
        x1 = x.max()
    else:
        x1 = max

    if min == -1:
        x0 = x.min()
    else:
        x0 = min

    xn = (x - x0)/(x1 - x0 + 1e-3)
    xn = np.asarray(xn*256,dtype=np.int)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            v = int(xn[r,c])
            c = cm.jet(v)
            color = (int(c[0]*5), int(c[1]*5), int(c[2]*5))
            cprint3( color, ' ', end='')
        print('')


def view_imagec(x):   
    for r in range(x.shape[1]):
        for c in range(x.shape[2]):
            c = x[:,r,c]            
            color = (int(c[0]*5), int(c[1]*5), int(c[2]*5))
            cprint3( color, ' ', end='')
        print('')


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

from collections import defaultdict, OrderedDict
def table2dict_id_data(table):
    b2a = OrderedDict()
    a2b = defaultdict(list)
    for a, b in table:
        b2a[b] = a
        a2b[a].append(b)
    return a2b, b2a

def table2dict_data_id(table):
    b2a = OrderedDict()
    a2b = defaultdict(list)
    for b, a in table:
        b2a[b] = a
        a2b[a].append(b)
    return a2b, b2a


def eval_cos_distance(x1, x2):
    x1n = np.linalg.norm(x1, axis=1).reshape(-1, 1)
    x2n = np.linalg.norm(x2, axis=1).reshape(-1, 1)
    return np.dot(x1, x2.T) / np.dot(x1n, x2n.T)

def eval_square_distance(x1, x2):
    return -np.mean((x1 - x2)**2)

def eval_square_norm_distance(x1, x2):
    x1 = x1/np.linalg.norm(x1, 2)
    x2 = x2/np.linalg.norm(x2, 2)
    return -np.mean((x1 - x2)**2)



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
#def save_obj(obj, filename):
#    with open(filename, 'w') as f:
#        pickle.dump(obj, f, protocol=2)


# ----------------------------------------------------------------------------------------------------------------------
#def load_obj(filename):
#    with open(filename, 'r') as f:
#        return pickle.load(f)

def save_obj(obj, file):
    if not isinstance(file,str):
        pickle.dump(obj, file, protocol=2)
        return

    root,ext = os.path.splitext(file)
    if ext == '.gz':
        with gzip.open(file, 'w') as f:
            pickle.dump(obj, f, protocol=2)
    else:
        with open(file, 'w') as f:
            pickle.dump(obj, f, protocol=2)

# ----------------------------------------------------------------------------------------------------------------------
def load_obj(file):
    if not isinstance(file,str):
        return pickle.load(f)

    root,ext = os.path.splitext(file)
    if ext == '.gz':
        with gzip.open(file, 'r') as f:
            return pickle.load(f)
    else:
        with open(file, 'r') as f:
            return pickle.load(f)

# ----------------------------------------------------------------------------------------------------------------------
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
def generate_ind_batch_loop(nb_samples, batch_size, random=True, roundup=True):
    while(True):
        for ind in generate_ind_batch(nb_samples, batch_size, random, roundup):
            yield ind
        print('gen reset')

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
