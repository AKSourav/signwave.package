import pickle
import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

def convert_sklearn_to_onnx(pickle_file, onnx_file, input_dimensions):
    """
    Convert a scikit-learn model from pickle format to ONNX format.
    Handles cases where the pickle file contains a dictionary of models.
    
    Parameters:
    pickle_file (str): Path to the pickle file containing the model or model dictionary
    onnx_file (str): Path where the ONNX model will be saved
    input_dimensions (tuple): The shape of the input tensor
    """
    # Load the pickle model
    print(f"Loading model from {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Model data type: {type(model_data).__name__}")
    
    # If the model is a dictionary, we need to identify the actual model
    if isinstance(model_data, dict):
        print(f"Model is a dictionary with keys: {list(model_data.keys())}")
        
        # Try to find the model in the dictionary
        # Common keys include 'model', 'classifier', 'estimator', etc.
        model_candidates = ['model', 'classifier', 'estimator', 'rf', 'randomforest', 
                           'random_forest', 'decision_tree', 'dt', 'pipeline']
        
        model = None
        model_key = None
        
        # First check if any key exactly matches our candidates
        for key in model_candidates:
            if key in model_data:
                model = model_data[key]
                model_key = key
                break
        
        # If not found, try case-insensitive matching or partial matching
        if model is None:
            for key in model_data:
                key_lower = key.lower()
                for candidate in model_candidates:
                    if candidate in key_lower:
                        model = model_data[key]
                        model_key = key
                        break
                if model is not None:
                    break
        
        # If still not found, try to identify scikit-learn objects
        if model is None:
            for key, value in model_data.items():
                module_name = getattr(value.__class__, '__module__', '')
                if 'sklearn' in module_name:
                    model = value
                    model_key = key
                    break
        
        # If still not found, let the user choose
        if model is None:
            print("Couldn't automatically identify which key contains the model.")
            print("Available keys:")
            for i, key in enumerate(model_data.keys()):
                value_type = type(model_data[key]).__name__
                print(f"{i+1}. {key} (type: {value_type})")
            
            # For automated scripts, we'll try the first key that seems to be a model
            for key, value in model_data.items():
                try:
                    # Check if it has predict method which most sklearn models have
                    if hasattr(value, 'predict'):
                        model = value
                        model_key = key
                        print(f"Selected '{key}' as it has a predict method.")
                        break
                except:
                    continue
    else:
        # The pickle file directly contained a model
        model = model_data
        model_key = "main_model"
    
    if model is None:
        raise ValueError("Could not identify a valid model in the pickle file.")
    
    print(f"Using model from key '{model_key}' with type: {type(model).__name__}")
    
    # Define the input type
    initial_type = [('float_input', FloatTensorType(input_dimensions))]
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_file) if os.path.dirname(onnx_file) else '.', exist_ok=True)
    
    # Convert the model to ONNX
    print("Converting model to ONNX format...")
    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Save the model
        print(f"Saving ONNX model to {onnx_file}...")
        with open(onnx_file, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Verify the model can be loaded
        print("Verifying the model...")
        sess = rt.InferenceSession(onnx_file)
        input_name = sess.get_inputs()[0].name
        print(f"Model input name: {input_name}")
        output_name = sess.get_outputs()[0].name
        print(f"Model output name: {output_name}")
        
        print("Conversion completed successfully!")
        return True
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        if "MissingShapeCalculator" in str(e):
            print("\nIt seems the model type is not directly supported by skl2onnx.")
            print("You might need to register a custom converter for this specific model type.")
        return False

if __name__ == "__main__":
    # Paths
    pickle_file = "modelislv17.p"  # Your pickle model path
    onnx_file = "modelislv17.onnx"  # Output ONNX model path
    
    # Based on your previous code, it seems the input has 183 features:
    # data_aux = pose_data + lh + rh where:
    # pose_data is 99 elements, lh and rh are 42 elements each
    # 99 + 42 + 42 = 183
    input_shape = (None, 183)  # (batch_size, features)
    
    # Convert and save the model
    convert_sklearn_to_onnx(pickle_file, onnx_file, input_shape)