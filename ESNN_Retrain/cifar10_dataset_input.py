import pickle
import numpy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class CIFAR10_Dataset_Input(object):
    def __init__(self, FLAGS, mode):
        self.FLAGS = FLAGS
        self.mode = mode
        self.load_cifar10()

    # grab the next batch of data from a dataset
    def next_batch(self):
        ids = np.arange(len(self.images))
        np.random.shuffle(ids)
        xs = [self.images[i] for i in ids]
        ys = [self.labels[i] for i in ids]
        for i in range(len(xs)):
            yield xs[i], ys[i]

    # generate dataset iterator
    def input_fn(self):
        if self.mode == 'train':
            dataset = tf.data.Dataset.from_generator(self.next_batch, (tf.float32, tf.int32))
            dataset = dataset.map(self.input_process)
            dataset = dataset.shuffle(self.FLAGS.batch_size, seed=self.FLAGS.seed).batch(self.FLAGS.batch_size)
            dataset = dataset.repeat()
        else:
            dataset = tf.data.Dataset.from_generator(self.next_batch, (tf.float32, tf.int32))
            dataset = dataset.map(self.input_process).batch(self.FLAGS.batch_size)
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return {'feature': features}, labels

    # read the original images from the CIFAR10 dataset
    def load_cifar10(self):
        images, labels = [], []
        if self.mode == 'train':
            for filename in ['%s/data_batch_%d' % (self.FLAGS.data_dir, j) for j in range(1, 6)]:
                with open(filename, 'rb') as fo:
                    cifar10 = pickle.load(fo, encoding='bytes')
                for i in range(len(cifar10[b"labels"])):
                    image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                    image = numpy.transpose(image, (1, 2, 0))
                    image = image.astype(float)
                    images.append(image)
                labels += cifar10[b"labels"]
            self.images = numpy.array(images, dtype='float')
            self.labels = numpy.array(labels, dtype='int')
        else:
            for filename in ['%s/test_batch' % (self.FLAGS.data_dir)]:
                with open(filename, 'rb') as fo:
                    cifar10 = pickle.load(fo, encoding='bytes')
                for i in range(len(cifar10[b"labels"])):
                    image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                    image = numpy.transpose(image, (1, 2, 0))
                    image = image.astype(float)
                    images.append(image)
                labels += cifar10[b"labels"]
            self.images = numpy.array(images, dtype='float')
            self.labels = numpy.array(labels, dtype='int')


    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # original images preprocessing
    def input_process(self, img, labels):
        if self.mode == 'train':
            img = tf.reshape(img, [32, 32, 3])
            img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
            img = tf.random_crop(img, [32, 32, 3], seed=self.FLAGS.seed)
            img = tf.image.random_flip_left_right(img, seed=self.FLAGS.seed)
        else:
            img = tf.reshape(img, [32, 32, 3])

        return img, labels