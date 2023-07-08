from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.training_set = None
        self.test_set = None    
 
    def train_valid_generator(self):

        train_datagen_kwargs = dict(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
        )

        training_set_kwargs = dict(
                            
                                target_size = (64, 64),
                                batch_size=self.config.params_batch_size,
                                class_mode = 'binary'
        )
        test_datagen_kwargs = dict(rescale = 1./255)

        test_set_kwargs = dict(
                            target_size = (64, 64),
                            batch_size=self.config.params_batch_size,
                            class_mode = 'binary')


        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **test_datagen_kwargs
        )

        self.test_set = test_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **test_set_kwargs
        )
        


        if self.config.params_is_augmentation:
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                **train_datagen_kwargs
            )
        else:
            train_datagen = test_datagen

        self.training_set = train_datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **training_set_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

  

    def train(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=self.config.params_image_size))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        
        self.model.fit(x=self.training_set, validation_data=self.test_set, epochs=self.config.params_epochs)

        self.save_model(
                    path=self.config.trained_model_path,
                    model=self.model
                    )

    

