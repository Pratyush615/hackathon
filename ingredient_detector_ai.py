import cv2
import tensorflow as tf
import numpy as np

class IngredientDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def detect_ingredients(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize for model input
        image = np.expand_dims(image, axis=0) / 255.0  # Normalize
        predictions = self.model.predict(image)
        detected_ingredients = self._post_process(predictions)
        return detected_ingredients

    def _post_process(self, predictions):
        # Convert model output to ingredient names
        ingredient_labels = ["apple", "carrot", "eggs", "pork"]  # Example
        return [ingredient_labels[i] for i, p in enumerate(predictions[0]) if p > 0.5]