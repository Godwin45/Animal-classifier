from cnnClassifier.entity.config_entity import EvaluationConfig
import tensorflow as tf
from pathlib import Path
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.score = None

    
    def _valid_generator(self):
       
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
        

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    