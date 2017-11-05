from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from snapshot import SnapshotCallbackBuilder
# dimensions of our images.
img_width, img_height = 299, 299


train_data_dir = 'kvasir-dataset'
validation_data_dir = 'test'
nb_train_samples = 450*8
nb_validation_samples = 50*8
epochs = 200
batch_size = 16




def base_model_finetuning():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    predictions = Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    """f = open('structure.json', 'w')
    f.write(model.to_json())
    f.close()"""
    # Freeze inception layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        rescale=1. / 255,
        zoom_range=(0.8, 1.1),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    datagen2 = ImageDataGenerator(
        rescale=1. / 255
    )
    gen1 = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16)

    gen2 = datagen2.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                        batch_size=batch_size)
    snapshot_ens=SnapshotCallbackBuilder(nb_epochs=20,nb_snapshots=5)


    model.fit_generator(gen1,

                        steps_per_epoch=nb_train_samples // 16,
                        epochs=20,
                        validation_data=gen2,
                        validation_steps=nb_validation_samples // 16,
                        callbacks=snapshot_ens.get_callbacks(model_prefix='inception_ens')
                        )
def test_ensembles():
    from keras.models import model_from_json
    m_prec=0
    m_rec=0
    m_fmeas=0
    m_acc=0
    for name_file in os.listdir('weights/'):
        f = open('backend_inceptionv2.json', 'r')
        model=None
        model = model_from_json(f.read())
        f.close()
        model.load_weights('weights/'+name_file)
        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        datagen2 = ImageDataGenerator(
            rescale=1. / 255
        )
        generator = datagen2.flow_from_directory(
            'test/',
            target_size=(img_width, img_height),
            batch_size=50,
            class_mode=None,
            shuffle=False)
        np.set_printoptions(suppress=True)
        predictions = model.predict_generator(generator, 8)
        index = 0
        confusion_matrix = np.zeros((8, 8))
        for i in predictions:
            true_class = index // 50
            confusion_matrix[np.argmax(i)][true_class] += 1
            index += 1
        tps = confusion_matrix.diagonal()
        fps = np.sum(confusion_matrix, (0))
        fps -= tps
        fns = np.sum(confusion_matrix, (1))
        fns -= tps
        precision = tps / (np.sum(confusion_matrix, (1)))
        recall = tps / (np.sum(confusion_matrix, (0)))
        accuracy = np.sum(tps) / (np.sum(confusion_matrix))
        f_measure = (2 * precision * recall) / (precision + recall)
        m_prec+=np.mean(precision)
        m_rec+=np.mean(recall)
        m_fmeas+=np.mean(f_measure)
        m_acc+=accuracy
        print('p:',end='')
        print(np.mean(precision))
        print('r:', end='')
        print(np.mean(recall))
        print('fm:', end='')
        print(np.mean(f_measure))
        print('a:', end='')
        print(accuracy)
        print('-------------')
    print('final precision ' + str(m_prec / 5))
    print('final recall ' + str(m_rec / 5))
    print('final fmeas ' + str(m_fmeas / 5))
    print('final accura ' + str(m_acc / 5))


test_ensembles()
#base_model_finetuning()
