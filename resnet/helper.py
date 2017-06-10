import numpy as np
import os
import pandas as pd


def load_img_path(data_path, num_img):
    '''Load images pathes
    
    Args:
        data_path: A folder containing images. e.g, ./home/.../cifar10/
        num_img: Number of images.
    Returns:
        Pathes of images.

        e.g.
            [./home/.../cifar10/1.png
            ./home/.../cifar10/2.png
            ...,
            ./home/.../cifar10/num_img.png]
    '''

    img_path_list = []
    for i in xrange(num_img):
        img_path = os.path.join(data_path, '%d.png' % (i+1, ))
        img_path_list.append(img_path)
    
    img_path_list = np.asarray(img_path_list)
    
    # print(img_path_list)
    # print(img_path_list.shape)

    return img_path_list


def load_labels(labels_file):
    '''Convert string labels to numberic format.

    Args:
        labels_file: A file containing labels. e.g, ./home/.../cifar10/trainLabels.csv
    Returns:
        Labels. A 1-D numpy array of shape (N, ).

        e.g.
            [c1, c2, ..., cn]
    '''

    labels_dict = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, \
                    'frog':6, 'horse':7, 'ship':8, 'truck':9}
    
    # labels = np.genfromtxt(labels_path, delimiter=',')
    labels = np.recfromcsv(labels_file, delimiter=',')
    labels = [list(x) for x in labels]
    labels = np.asarray(labels)
    labels = labels[:,1].tolist()

    # map to indices
    labels = [labels_dict[x] for x in labels]
    label_list = np.asarray(labels)

    # print(label_list)
    # print(label_list.shape)

    return label_list


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
    # print(X_train.shape)
    # print(X_train[4])
    # print(y_train[4])

    # split validation data
    val_indices = [i for i in xrange(total_size) if i not in train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    # print(X_val.shape)

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
    
    # print(data)
    # print(data.shape)


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


def decode_labels(file_to_read, file_to_output):
    '''Decode numberic labels to string.

    Args:
        file_to_read: A file of which a line is a label(interger format) for an image.
        file_to_output: A csv file to output the decoded labels. Having the format bellow
            
        e.g.
            id,label
            1,deer
            2,cat
            ...
    '''

    labels_dict = {'0':'airplane', '1':'automobile', '2':'bird', '3':'cat', '4':'deer', '5':'dog', 
        '6':'frog', '7':'horse', '8':'ship', '9':'truck'}
    
    labels = list(open(file_to_read).readlines())
    labels = [s.strip() for s in labels]
    id_labels = [labels_dict[s] for s in labels]
    # id_labels = np.asarray(id_labels)
    
    idx = pd.Int64Index(np.arange(1, 300001).tolist())
    id_labels = pd.DataFrame(index = idx, data = {'label': id_labels})
    id_labels.index.name = 'id'
    id_labels.to_csv(file_to_output)

    print(id_labels)



if __name__ == '__main__':
    test_img_path = '/home/yhx/kaggle/cifar10/test/'

    # train_img_path = '/home/yang/Downloads/FILE/CODE/kaggle/cifar-10/cifar10/train/'
    # num_img = 50000
    # labels_path = '/home/yang/Downloads/FILE/CODE/kaggle/cifar-10/cifar10/trainLabels.csv'
    # num_class = 10

    # img_path_list = load_img_path(train_img_path, num_img)
    # label_list = load_labels(labels_path)

    # X_train, y_train, X_val, y_val = split_train_val(img_path_list, label_list, 45000)

    # write_to_file(X_train, "./cifar10/X_train.txt")
    # write_to_file(y_train, "./cifar10/y_train.txt")
    # write_to_file(X_val, "./cifar10/X_val.txt")
    # write_to_file(y_val, "./cifar10/y_val.txt")
    
    ################ test data ################
    img_path_list = load_img_path(test_img_path, 300000)
    write_to_file(img_path_list, "./cifar10/test.txt")

    # X_test = load_data('/home/yhx/kaggle/cifar10/test.txt')
    X_test = load_data('/home/yang/Downloads/FILE/CODE/kaggle/cifar-10/cifar10/test.txt')
    print(X_test)

    ################ decode labels ################
    # decode_labels("/home/yang/Downloads/FILE/CODE/kaggle/ws/predict.txt", '/home/yang/Downloads/FILE/CODE/kaggle/ws/predict.csv')


