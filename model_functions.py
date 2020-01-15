import tensorflow as tf
import keras.backend as K
import numpy as np

#Input is 40 vectors from 4 restaurants
#Takes average for every restaurant
#Returns 40 vectors where every vector in average vector for his restaurants(only 4 vectors are "independent")
def groupwise_average(inp):
    import tensorflow as tf
    
    class_matrix = tf.reshape(tf.transpose(K.one_hot(tf.cast(inp[1], dtype=tf.int32), num_classes=4)), [4,tf.shape(inp[0])[0]])
    
    nums_vector = tf.cast(tf.math.count_nonzero(class_matrix, 1, keepdims=True), dtype=tf.float32)
    
    mean_matrix = class_matrix/nums_vector
    
    general_mean_matrix = tf.transpose(tf.keras.backend.dot(mean_matrix, inp[0]))
    
    return tf.transpose(tf.keras.backend.dot(general_mean_matrix, class_matrix))


def bp_mll_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

    # get true and false labels
    shape = tf.shape(y_true)
    y_i = tf.equal(y_true, tf.ones(shape))
    y_i_bar = tf.not_equal(y_true, tf.ones(shape))

    # get indices to check
    truth_matrix = tf.cast(pairwise_and(y_i, y_i_bar), dtype=tf.float32)

    # calculate all exp'd differences
    sub_matrix = pairwise_sub(y_pred, y_pred)
    exp_matrix = tf.exp(tf.negative(sub_matrix))

    # check which differences to consider and sum them
    sparse_matrix = tf.multiply(exp_matrix, truth_matrix)
    sums = tf.reduce_sum(sparse_matrix, axis=[1,2])

    # get normalizing terms and apply them
    y_i_sizes = tf.reduce_sum(tf.cast(y_i, dtype=tf.float32), axis=1)
    y_i_bar_sizes = tf.reduce_sum(tf.cast(y_i_bar, dtype=tf.float32), axis=1)
    normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)
    results = tf.divide(sums, normalizers)

    # average error
    return tf.reduce_mean(results)


def pairwise_sub(first_tensor: tf.Tensor, second_tensor: tf.Tensor) -> tf.Tensor:

    column = tf.expand_dims(first_tensor, 2)
    row = tf.expand_dims(second_tensor, 1)
    
    return tf.subtract(column, row)


def pairwise_and(first_tensor: tf.Tensor, second_tensor: tf.Tensor) -> tf.Tensor:

    column = tf.expand_dims(first_tensor, 2)
    row = tf.expand_dims(second_tensor, 1)
    
    return tf.logical_and(column, row)


def recall_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (possible_positives + K.epsilon())
    
    return recall


def precision_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    
    return precision
    
    
def f1_m(y_true, y_pred):
    
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
