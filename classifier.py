import tensorflow as tf
import os 
import cv2 as cv
import imghdr
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


class EmotionClassifier:
    def __init__(self, data_dir='data', model_path='model/emotionModel.keras' ):
        self.image_extensions = ['jpeg','jpg','png','bmp']
        self.data_dir = data_dir
        self.model_path = model_path
        self.model_path = model_path
        self.train = self.val = self.test = None
        self.model = None

    # control the used memory to be minimal
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    def filterFiles(self):
        # filter images by type in data folder
        for image_class in os.listdir(self.data_dir):
            for image in os.listdir(os.path.join(self.data_dir,image_class)):
                image_path = os.path.join(self.data_dir,image_class,image)
                try:
                    img = cv.imread(image_path)
                    type = imghdr.what(image_path)
                    if type not in self.image_extensions:
                        print("Image's type is not permitted")
                        os.remove(image_path)
                except Exception as e:
                    print("Issue with image")
                    os.remove(image_path)        
            print(f"Successfuly filtered directory {image_class}!")
    
    def dataPipeline(self):
        #generate data with labels and preprocess them automatically
        data = tf.keras.utils.image_dataset_from_directory(self.data_dir)
        # access data from pipeline
        data_iterator = data.as_numpy_iterator()
        return data_iterator,data
    
    def normalizeBatch(self):
        fig, ax = plt.subplots(ncols=8, figsize=(20,20))
        data_iterator,_ = self.dataPipeline()
        batch = data_iterator.next()# there is 2 parts : images and their labels
        for idx, img in enumerate(batch[0][:8]):
            ax[idx].imshow(img.astype(int))
            # above images 2(sad), 0(happy)
            label = int(batch[1][idx]) 
            ax[idx].set_title(str(label))
        plt.show()
    
    def splitData(self):
        _,data = self.dataPipeline()
        data_norm = data.map(lambda x, y: (x / 255.0, y))
        train_size = int(len(data_norm)*0.7)
        val_size = int(len(data_norm)*0.2)+1
        test_size = int(len(data_norm)*0.1)+1
        self.train = data_norm.take(train_size)
        self.val = data_norm.skip(train_size).take(val_size)
        self.test = data_norm.skip(train_size+val_size).take(test_size)

    def buildModel(self):
        self.model = Sequential()
        #extract low_level features (vision of a model)
        self.model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
        #reduce spatial dimensions
        self.model.add(MaxPooling2D())

        #more filters
        self.model.add(Conv2D(32, (3,3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        # extract more abstract features
        self.model.add(Conv2D(16, (3,3), 1, activation='relu'))
        self.model.add(MaxPooling2D())

        #unfold features map to 1D vector for Dense
        self.model.add(Flatten())
        #fully connected layer with 256 neurons for object classifications
        self.model.add(Dense(256, activation='relu'))
        #output layer -- output 0 or 1
        self.model.add(Dense(1, activation='sigmoid'))

        #optimizer
        self.model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
        print(self.model.summary()) # param == 0 --> non-learnable layer

    def loadModel(self):
        self.model = load_model(self.model_path)

    def trainModel(self):
        logdir = 'logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
        hist = self.model.fit(self.train, epochs=20, validation_data=self.val, callbacks=[tensorboard_callback])
        return hist
    
    def plot_history(self):
        # Get dictionary of metrics from training
        history = self.trainModel().history
        # Plot training & validation accuracy values
        if 'accuracy' in history:
            plt.plot(history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Validation Accuracy')

        # Plot training & validation loss values
        if 'loss' in history:
            plt.plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')

        plt.title('Model Accuracy and Loss')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
    
    def saveModel(self):
        if self.model is None:
            raise ValueError("Model not found. Build and train the model before saving.")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

    
    def predict(self,image_path):
        img = cv.imread(image_path)
        resize = tf.image.resize(img, (256, 256))
        input_tensor = np.expand_dims(resize / 255.0, axis=0)
        prediction = self.model.predict(input_tensor)

        if prediction > 0.6:
            print(f"Predicted class is Sad ({prediction[0][0]:.4f})")
        else:
            print(f"Predicted class is Happy ({prediction[0][0]:.4f})")

if __name__ == "__main__":
    classifier = EmotionClassifier()

    #classifier.filterFiles()
    #classifier.splitData()
    #classifier.buildModel()
    #classifier.trainModel()
    #classifier.saveModel()
    #classifier.plot_history()
    classifier.loadModel()
    classifier.predict(r'D:\emotion_classifier\1-2.jpg')
    classifier.predict(r'D:\emotion_classifier\170404-happy-workers-feature.jpg')  


