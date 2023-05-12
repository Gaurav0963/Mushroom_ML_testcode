import os
import pickle
import pandas as pd


class Prediction:
    @staticmethod
    def decode_prediction(prediction) -> str:
        """
        :param prediction: Encoded Prediction value
        :return: Decoded Prediction
        """
        if prediction == 1:
            return " : Mushroom is Edible"
        elif prediction == 0:
            return " : Mushroom is Poisonous"

    def model_predict(self, features: pd.DataFrame):
        """
        :param features: Input provided by user is collected as Pandas DataFrame
        :return: Predicted Value by Model is returned
        """
        # pre_processor_path = os.path.join(os.getcwd(), 'models', os.listdir("models")[1])
        # model_path = os.path.join(os.getcwd(), 'models', os.listdir("models")[0])

        pre_processor = load_object(file_path="preprocessor.pkl")
        model = load_object(file_path="model.pkl")

        scaled_features = pre_processor.transform(features)

        predictions_val = model.predict(scaled_features)

        predictions = self.decode_prediction(predictions_val)

        return predictions


class CustomData:
    def __init__(self, cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size,
                 gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
                 stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type,
                 spore_print_color, population, habitat):
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        :return: Input data as Pandas DataFrame
        """
        custom_data_dict = {
            "cap-shape": [self.cap_shape],
            "cap-surface": [self.cap_surface],
            "cap-color": [self.cap_color],
            "bruises": [self.bruises],
            "odor": [self.odor],
            "gill-attachment": [self.gill_attachment],
            "gill-spacing": [self.gill_spacing],
            "gill-size": [self.gill_size],
            "gill-color": [self.gill_color],
            "stalk-shape": [self.stalk_shape],
            "stalk-root": [self.stalk_root],
            "stalk-surface-above-ring": [self.stalk_surface_above_ring],
            "stalk-surface-below-ring": [self.stalk_surface_below_ring],
            "stalk-color-above-ring": [self.stalk_color_above_ring],
            "stalk-color-below-ring": [self.stalk_color_below_ring],
            "veil-type": [self.veil_type],
            "veil-color": [self.veil_color],
            "ring-number": [self.ring_number],
            "ring-type": [self.ring_type],
            "spore-print-color": [self.spore_print_color],
            "population": [self.population],
            "habitat": [self.habitat]
        }

        return pd.DataFrame(custom_data_dict)


def load_object(file_path: str):
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)

