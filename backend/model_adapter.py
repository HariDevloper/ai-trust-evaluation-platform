# Model Adapter

class ModelAdapter:
    def __init__(self, model):
        self.model = model
    
    def predict(self, input_data):
        return self.model.predict(input_data)
