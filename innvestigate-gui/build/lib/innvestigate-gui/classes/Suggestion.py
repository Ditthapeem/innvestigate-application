class Suggestion:
    def __init__(self, kind=None, input_images=None, models_correct="", models_mean_distance_from_input_class="",
                 input_images_class_name="", model_predictions="", layer_mean_activations="", neuron_activations="",
                 models_top_pred_mean_probs=""):
        
        self.kind = kind
        self.input_images = input_images
        self.models_correct = models_correct
        self.models_mean_distance_from_input_class = models_mean_distance_from_input_class
        self.input_images_class_name = input_images_class_name
        self.model_predictions = model_predictions
        self.layer_mean_activations = layer_mean_activations
        self.neuron_activations = neuron_activations
        self.models_top_pred_mean_probs = models_top_pred_mean_probs
        
    def toJSON(self):
        return self.__dict__
