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

        datagenerator_kwargs = dict(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   validation_split=0.20,
                                   horizontal_flip = True
        )

        dataflow_kwargs = dict(
                                target_size = self.config.params_image_size[:-1],
                                batch_size=self.config.params_batch_size,
                                class_mode = 'binary'
        )
        

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        


        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            
                **datagenerator_kwargs
            )
        else:
             train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
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

        
        #self.model.fit(x=self.training_set, validation_data=self.test_set, epochs=self.config.params_epochs)
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
        )

        self.save_model(
                    path=self.config.trained_model_path,
                    model=self.model
                    )

    



    

