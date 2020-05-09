from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import json
from sklearn.metrics import confusion_matrix
from Models import conv_edge, conv_road, vgg

def model_selection(model_name):

    if model_name.startswith('vgg'):
        print(model_name)
        model = vgg(n_classes=6, IMG_SIZE = 128)

    elif model_name.startswith('edge'):
        print(model_name)
        model = conv_edge(n_classes= 6, IMG_SIZE=128)

    elif model_name.startswith('road'):
        print(model_name)
        model = conv_road(n_classes = 6, IMG_SIZE = 128)

    else:
        raise('Please select a relevant model name.')

    return model

def main():
    BATCH_SIZE = 32
    IMG_SIZE = 128
    n_epochs = 100
    learning_rate = 0.01
    best_acc = 0
    resume = False
    # Select the model here: edge, road, vgg
    model_name = 'edge'

    # Configure the GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TF to only use one GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[7], 'GPU')
        except RuntimeError as e:
            print(e)
    # Data loading
    train_dir = '../dataset/split/train200_2'
    test_dir   = '../dataset/split/test'

    train_gen = ImageDataGenerator(rescale = 1./255)
    val_gen   = ImageDataGenerator(rescale = 1./255)

    train_data = train_gen.flow_from_directory(directory = train_dir,
                                               batch_size = BATCH_SIZE,
                                               shuffle = True,
                                               target_size=(IMG_SIZE, IMG_SIZE))
    val_data = val_gen.flow_from_directory(directory = test_dir,
                                           batch_size = BATCH_SIZE,
                                           target_size = (IMG_SIZE, IMG_SIZE))

    # Select model
    model = model_selection(model_name)

    # Define loss function and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)
    loss_fn   = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


    # Checkpoint details
    checkpoint_path = '../Checkpoints/{}/'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = tf.train.Checkpoint(optimizer = optimizer, model = model)
    manager = tf.train.CheckpointManager(checkpoint, directory= checkpoint_path.format(model_name), max_to_keep=1)

    if resume:
        # Load the last best model
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights('../Checkpoints/{}/'.format(model_name))
        # Evaluate on the test set to get best acc
        best_acc, cm = evaluate(val_data, model, loss_fn)

        fp,fn,tp,tn = cm
        precision = tp / (tp + fn)
        recall = tp / (tp + fn)
        fpr = fp / (tn + fp)
        fnr = fn / (tp + fn)
        f1 = 2. / (1./precision + 1./recall)

        print("------Metrics Reports -------")
        print("Accuracy: {}, Precision: {}, Recall: {},"
              "Fpr: {}, Fnr: {}, F1-Score: {}".format(best_acc, precision.mean(),
                                               recall.mean(),fpr.mean(), fnr.mean(), f1.mean())

    for epoch in range(n_epochs):

        # Adjust the learning rate after every 30 epochs
        step_based_decay(optimizer, epoch)
        # Train for one epoch
        train(train_data, model, loss_fn, optimizer, epoch)

        # Evaluate on validation set
        acc, _ = evaluate(val_data, model, loss_fn)

        # Record checkpoints if test accuracy is better than
        if acc > best_acc:
            best_acc = acc
            # manager.save()
            model.save_weights(checkpoint_path.format(model_name), overwrite = True)




def train(train_loader, model, loss_function, optimizer, epoch):
    batches = 0
    batch_time = Average()
    acc = Average()
    loss = Average()

    end = time.time()
    for batch_x, batch_y in train_loader:

        # Stop the data generator since it goes infinitely
        batches += 1
        if batches >= len(train_loader):
            break

        with tf.GradientTape() as tape:
            # compute output
            logits = model(batch_x, training = True)
            loss_value = loss_function(batch_y, logits)
        # Calculate batch accuracy
        batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                                         tf.argmax(batch_y, axis=1)),
                                                    dtype=tf.float64)).numpy()

        # Compute gradient and do back-prop
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Record accuracy and loss
        loss.update(loss_value.numpy(), batch_x.shape[0])
        acc.update(batch_accuracy, batch_x.shape[0])
        # Measure computation time
        batch_time.update(time.time() - end)
    # End of an epoch
    print("Epoch {:03d} --->, Loss: {:.5f}, Accuracy: {:.3f}, Batch Time: {:.3f} ".format(epoch + 1,
                                                                                 loss.avg,
                                                                                 acc.avg,
                                                                                 batch_time.avg))

def evaluate(val_loader, model, loss_function):
    batches = 0
    acc = Average()
    loss = Average()
    batch_time = Average()
    fps = Average()
    fns = Average()
    tps = Average()
    tns = Average()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):

        batches += 1
        if batches >= len(val_loader):
            break

        logits = model(images, training = False)
        loss_value = loss_function(logits, labels)
        # Record batch accuracy
        predictions = tf.argmax(logits, axis = 1)
        correct_predictions = tf.equal(predictions, tf.argmax(labels, axis =1))
        batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype = tf.float64)).numpy()

        # Record accuracy and loss
        loss.update(loss_value, images.shape[0])
        acc.update(batch_accuracy, images.shape[0])

        # Record batch time and set the new counter
        batch_time.update(time.time() - end)
        end = time.time()
        # Create confusion matrix
        cm = confusion_matrix(y_true = tf.argmax(labels, axis =1), y_pred = predictions,labels = [*range(6)])

        # Calculate other metrics
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - fp - fn - tp

        ## Record these metrics
        fps.update(fp, images.shape[0])
        fns.update(fn, images.shape[0])
        tps.update(tp, images.shape[0])
        tns.update(tn, images.shape[0])

    print("Test Epoch ---> Loss: {:.5f}, Accuracy: {:.3f}, Batch Time: {:.3f} ".format(loss.avg,
                                                                                       acc.avg,
                                                                                       batch_time.avg))
    return acc.avg, (fps.sum, fns.sum, tps.sum, tns.sum)



def step_based_decay(optimizer, epoch):
    """Sets a new learning after fixed number of epochs.
    Fixed epoch size : 30
    Decrease Factor: 0.1
    """
    alpha = optimizer.lr.numpy()
    exp = np.floor((1+epoch)/30)
    lr = alpha * (0.1 ** exp)
    optimizer.lr.assign(lr)
    print('New Learning Rate: {}'.format(lr))

def polynomialdecay(optimizer, epoch, n_epochs):
    """ Compute the new learning rate based on polynomial decay"""
    alpha = optimizer.lr.numpy()
    power = 1.0 # linear decay
    decay = (1 - (epoch / float(n_epochs))) ** power
    lr = alpha * decay
    optimizer.lr.assign(lr)
    print('New Learning Rate: {}'.format(lr))


def learning_rate_decay(optimizer, epoch, n_epochs):
    """Decays the learning rate after every epoch"""
    alpha = optimizer.lr.numpy()
    decay = alpha/n_epochs
    # Set the new learning rate
    lr = alpha * 1.0/(1.0 + (decay*epoch))
    optimizer.lr.assign(lr)
    print('New Learning Rate: {}'.format(lr))

class Average(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
