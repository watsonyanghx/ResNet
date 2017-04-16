"""CIFAR dataset input module.
"""

import tensorflow as tf


def build_input(X, y, hps, mode):
    """Build image and labels.

    Args:
        X: Pathes of images.
           
           e.g.
               ['/home/.../infer/1.png'
                '/home/.../infer/2.png'
                ...,
                '/home/.../infer/300000.png']
        y: Image labels.
           
           e.g.
               [1 6 2 4 ... 7]
        hps:  Hyperparameters.
        mode: Either 'train' or 'eval'.
    Returns:
        images: Batches of images. [batch_size, image_size, image_size, 3]
        labels: Batches of labels. [batch_size, num_classes]
    """
    # image_size = 32
    # num_classes = 10
    # depth = 3
    batch_size = hps.batch_size
    image_size = hps.image_size
    depth = hps.depth
    num_classes = hps.num_classes

    # convert to tensor
    X = tf.convert_to_tensor(X, dtype=tf.string)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([X, y], shuffle=True)
    
    image = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image, channels=3)
    label = tf.reshape(input_queue[1], [1])

    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(image, image_size+4, image_size+4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
        # image = tf.image.random_brightness(image, max_delta=63. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 16
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, labels


def build_infer_input(X, y, hps):
    '''
    Args:
        X: Pathes of Images.
        
           e.g.
               ['/home/.../infer/1.png'
                '/home/.../infer/2.png'
                ...,
                '/home/.../infer/300000.png']
        y: Image labels.
           
           e.g.
               [1 6 2 4 ... 7]
        hps: Hyperparameters.
    Returns:
        images: Batches of images. [batch_size, image_size, image_size, 3]
        labels: Batches of labels. [batch_size, num_classes]
    '''

    batch_size = hps.batch_size
    image_size = hps.image_size
    depth = hps.depth
    num_classes = hps.num_classes

    X = tf.convert_to_tensor(X, dtype=tf.string)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([X, y], shuffle=False)
    
    image = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image, channels=3)
    label = tf.reshape(input_queue[1], [1])


    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
    image = tf.image.per_image_standardization(image)

    example_queue = tf.FIFOQueue(
        3 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[image_size, image_size, depth], [1]])
    num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    return images, labels

