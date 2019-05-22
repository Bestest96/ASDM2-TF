import tensorflow as tf

from Dataset import Dataset


class Preprocessor:

    def preprocess(self, data, input_dim, output_dim, device='/cpu:0', **kwargs):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.device(device):
            input_t, output_t = self._prepare(data, input_dim, output_dim, **kwargs)
            input_prep_op = self._preprocess_input(input_t, **kwargs)
            output_prep_op = self._preprocess_output(output_t, **kwargs)
        with tf.Session(config=config) as sess:
            preprocessed_input, preprocessed_output = sess.run([input_prep_op, output_prep_op])
        return self._create_dataset(preprocessed_input, preprocessed_output)

    @staticmethod
    def _prepare(data, input_dim, output_dim, **kwargs):
        sample_size = len(data[0])
        begin = sample_size - output_dim - input_dim
        _, input_t, output_t = tf.split(tf.convert_to_tensor(data), [begin, input_dim, output_dim], 1)
        return input_t, output_t

    @staticmethod
    def _preprocess_input(input_t, **kwargs):
        prep_input_t = input_t
        normalization = kwargs.get('normalization')
        shape = kwargs.get('reshape')
        transpose = kwargs.get('transpose')
        if shape is not None:
            prep_input_t = tf.map_fn(lambda x: tf.reshape(x, shape), prep_input_t)
        if transpose is not None:
            prep_input_t = tf.map_fn(lambda x: tf.transpose(x, transpose), prep_input_t)
        if normalization is not None:
            prep_input_t = normalization(prep_input_t)
        return prep_input_t

    @staticmethod
    def _preprocess_output(output_t, **kwargs):
        return output_t

    @staticmethod
    def _create_dataset(input_t, output_t):
        return Dataset(input_t, output_t)


def classes_dict(classes_list):
    cl_dict = {}
    for i, cl in zip(range(len(classes_list)), classes_list):
        cl_dict[cl] = i
    return cl_dict


def reduce_classes(table, classes=None):
    if classes is None:
        labels = table[:, -1]
        unique_labels = set(labels)
        labels_dict = {e: i for e, i in zip(unique_labels, range(len(unique_labels)))}
    else:
        labels_dict = classes_dict(classes)
    for row in table:
        row[-1] = labels_dict[row[-1]]
    return table


class ClassificationPreprocessor(Preprocessor):

    @staticmethod
    def _prepare(data, input_dim, output_dim, **kwargs):
        classes = kwargs.get('classes')
        if classes is not None:
            reduce_classes(data, classes)
        return super(ClassificationPreprocessor, ClassificationPreprocessor)._prepare(data,
                                                                                      input_dim,
                                                                                      output_dim,
                                                                                      **kwargs)

    @staticmethod
    def _preprocess_output(output_t, **kwargs):
        to_one_hot = kwargs.get('to_one_hot')
        if to_one_hot is None:
            to_one_hot = True
        int_output = tf.cast(output_t, tf.int32)
        classes_nb = tf.reduce_max(int_output) + 1
        if to_one_hot:
            prep_output = tf.reshape(tf.one_hot(int_output, classes_nb), shape=(tf.shape(int_output)[0], classes_nb))
        else:
            prep_output = int_output
        return prep_output


class AutoencoderPreprocessor(Preprocessor):

    @staticmethod
    def _preprocess_output(output_t, **kwargs):
        return output_t

    @staticmethod
    def _create_dataset(input_t, output_t):
        return Dataset(input_t, input_t)
