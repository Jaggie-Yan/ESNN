import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def Indicator(x):
    return tf.stop_gradient(tf.to_float(x >= 0) - tf.sigmoid(x)) + tf.sigmoid(x)

class KernelMasked(tf.keras.Model):
    def __init__(self, kernel, strides, dropout_rate=None):
        super(KernelMasked, self).__init__()
        self.kernel = kernel
        self.strides = strides
        self.dropout_rate = tf.stop_gradient(dropout_rate)
        self._build()

    def _build(self):
        self.kernel_shape = self.kernel.shape

        # back to normal conv type
        if self.kernel_shape[0] != 5:
            self.kernel_masked = self.kernel
        else:
            assert self.kernel_shape[1] == 5

            # superkernel size selection thresholds
            self.t3x3 = self.add_weight(shape=(1,),
                                        initializer=tf.random_normal_initializer(
                                            mean=(np.linalg.norm(np.random.normal(loc=0.0,
                                                                                  scale=np.sqrt(2.0 / int(
                                                                                      self.kernel_shape[0] * self.kernel_shape[1] *
                                                                                      self.kernel_shape[3])),
                                                                                  size=(9 * self.kernel_shape[2] *
                                                                                        self.kernel_shape[3])))),
                                            stddev=0),
                                        name="t3x3")
            self.t5x5 = self.add_weight(shape=(1,),
                                  initializer=tf.random_normal_initializer(
                                    mean=(np.linalg.norm(np.random.normal(loc=0.0,
                                                                          scale=np.sqrt(2.0 / int(
                                                                            self.kernel_shape[0] * self.kernel_shape[1] *
                                                                            self.kernel_shape[3])),
                                                                          size=(16 * self.kernel_shape[2] * self.kernel_shape[3])))),
                                    stddev=0),
                                  name="t5x5")

            # create masks based on kernel_shape
            center_3x3 = np.zeros(self.kernel_shape)
            center_3x3[1:4, 1:4, :, :] = 1.0
            self.mask3x3 = tf.convert_to_tensor(center_3x3,
                                                dtype=self.t5x5.dtype)
            center_5x5 = np.ones(self.kernel_shape) - center_3x3
            self.mask5x5 = tf.convert_to_tensor(center_5x5,
                                                dtype=self.t5x5.dtype)

            # make indicator results "accessible" as separate vars
            kernel_3x3 = self.kernel * self.mask3x3
            kernel_5x5 = self.kernel * self.mask5x5
            self.norm3x3 = tf.norm(kernel_3x3)
            self.norm5x5 = tf.norm(kernel_5x5)

            x3x3 = self.norm3x3 - self.t3x3
            self.indicator3x3 = Indicator(x3x3)
            if all(s == 1 for s in self.strides):
                # noise to add
                if self.dropout_rate is not None:
                    self.d3x3 = tf.nn.dropout(Indicator(x3x3), rate=self.dropout_rate)
                else:
                    self.d3x3 = Indicator(x3x3)
            else:
                self.d3x3 = 1.0  # you cannot drop all layers!

            x5x5 = self.norm5x5 - self.t5x5
            self.indicator5x5 = Indicator(x5x5)
            # noise to add
            if self.dropout_rate is not None:
                self.d5x5 = tf.nn.dropout(Indicator(x5x5), rate=self.dropout_rate)
            else:
                self.d5x5 = Indicator(x5x5)

            self.kernel_masked = self.d3x3 * (kernel_3x3 + self.d5x5 * kernel_5x5)

    def call(self):
        return self.kernel_masked

