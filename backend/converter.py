import pickle
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import onnx
import os

def load_model(model_path):
    """Load the pre-trained scikit-learn model from a pickle file."""
    with open(model_path, 'rb') as file:
        model_dict = pickle.load(file)
    return model_dict['model']

def convert_to_onnx(model, output_path):
    """Convert a scikit-learn model to ONNX format and save it."""
    initial_type = [("float_input", FloatTensorType([None, model.n_features_in_]))]
    onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
    
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model saved as ONNX at: {output_path}")

if __name__ == "__main__":
    model_path = "modelislv17.p"  # Update with your actual model path
    output_path = "random_forest.onnx"
    
    if not os.path.exists(model_path):
        print("Error: Model file not found!")
    else:
        model = load_model(model_path)
        convert_to_onnx(model, output_path)
