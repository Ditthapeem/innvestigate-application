class Visualization:
    def __init__(self, output_images, max_activations_neuron_indexes_per_image, predictions):
        self.output_images = output_images
        self.max_activations_neuron_indexes_per_image = max_activations_neuron_indexes_per_image
        self.predictions = predictions
        
    def toJSON(self):
        return self.__dict__
