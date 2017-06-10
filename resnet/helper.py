import numpy as np
import os
import shutil


def split_train_val(X, y, train_size):
    '''Split dataset for training and validation.

    Args:
        X: A 1-D numpy array containing pathes of images.
        y: A 1-D numpy array containing labels.
        train_size: Size of training data to split.
    Returns:
        1-D numpy array having the same definition with X and y.
    '''

    total_size = len(X)
    # shuffle data
    shuffle_indices = np.random.permutation(np.arange(total_size))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    # split training data
    train_indices = np.random.choice(total_size, train_size, replace=False)
    X_train = X[train_indices]
    y_train = y[train_indices]

    # split validation data
    val_indices = [i for i in xrange(total_size) if i not in train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, y_train, X_val, y_val


def write_to_file(data, file_to_output):
    '''Write X_train/y_train/X_val/y_val/X_infer to file for further 
       processing (e.g. make input queue of tensorflow).

    Args:
        data: A 1-D numpy array, e.g, X_train/y_train/X_val/y_val/X_infer.
        file_to_output: A file to store data.
    '''
    # with open('X_train.csv','a') as f_handle:
    #     np.savetxt(f_handle, X_train, fmt='%s', delimiter=",")

    # write list to file
    with open(file_to_output, 'w') as f:
        for item in data.tolist():
            f.write(str(item) + '\n')


def load_data(file_to_read):
    '''Load X_train/y_train/X_val/y_val/X_infer for further 
       processing (e.g. make input queue of tensorflow).

    Args:
        file_to_read: 
    Returns:
        X_train/y_train/X_val/y_val/X_infer.
    '''

    data = np.recfromtxt(file_to_read)
    data = np.asarray(data)

    return data


################ ML ####################

def ml_load_labels(file):
    labels = list(open(file).readlines())
    labels = [s.strip() for s in labels]
    labels = [s.split() for s in labels]

    labels.sort(key=lambda x: x[0])
    
    labels_dict = dict(labels)
    
    labels = np.asarray(labels, dtype=int)
    labels = labels[:, 1]
    
    return labels, labels_dict


def ml_load_ima_path(images_path):
    file_names = [images_path+s for s in os.listdir(images_path)]
    
    file_names.sort()
    
    file_names = np.asarray(file_names)

    return file_names


def cp_file(file_list, labels_dict):
    for file_path in file_list:
        filename = os.path.basename(file_path).split('.')[0]

        label = labels_dict[filename]
        
        dst = '/home/yhx/ml/ic-data/' + label
        shutil.copy(file_path, dst)


if __name__ == '__main__':
    labels_path = '/home/yhx/ml/ic-data/train.label'
    labels, labels_dict = ml_load_labels(labels_path)

    # images_path = '/home/yhx/ml/ic-data/train/'
    # image_path_list = ml_load_ima_path(images_path)

    # X_train, y_train, X_val, y_val = split_train_val(image_path_list, labels, 2250)
    # write_to_file(X_train, "/home/yhx/ml/ic-data/X_train.txt")
    # write_to_file(y_train, "/home/yhx/ml/ic-data/y_train.txt")
    # write_to_file(X_val, "/home/yhx/ml/ic-data/X_val.txt")
    # write_to_file(y_val, "/home/yhx/ml/ic-data/y_val.txt")


    file_list = list(open('/home/yhx/ml/ic-data/X_train.txt').readlines())
    file_list = [s.strip() for s in file_list]
    # print(file_list)

    cp_file(file_list, labels_dict)

