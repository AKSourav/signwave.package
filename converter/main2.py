import pickle
import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

def convert_sklearn_to_onnx(pickle_file, onnx_file, input_dimensions):
    """
    Convert a scikit-learn model from pickle format to ONNX format.
    Handles cases where the pickle file contains a dictionary of models.
    Addresses issues with complex output types.
    
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
    
    # Extract the model from dictionary if needed
    if isinstance(model_data, dict):
        print(f"Model is a dictionary with keys: {list(model_data.keys())}")
        
        # Common model keys
        model_candidates = ['model', 'classifier', 'estimator', 'rf', 'randomforest', 
                           'random_forest', 'decision_tree', 'dt', 'pipeline']
        
        model = None
        model_key = None
        
        # Try exact matches first
        for key in model_candidates:
            if key in model_data:
                model = model_data[key]
                model_key = key
                break
        
        # Try case-insensitive/partial matching
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
        
        # Try to identify scikit-learn objects
        if model is None:
            for key, value in model_data.items():
                module_name = getattr(value.__class__, '__module__', '')
                if 'sklearn' in module_name:
                    model = value
                    model_key = key
                    break
        
        # Try to find anything with a predict method
        if model is None:
            for key, value in model_data.items():
                try:
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
    
    # Check model type and adapt conversion approach
    model_type = type(model).__name__
    print(f"Model specific type: {model_type}")
    
    # Define the input type
    initial_type = [('float_input', FloatTensorType(input_dimensions))]
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_file) if os.path.dirname(onnx_file) else '.', exist_ok=True)
    
    # Convert the model to ONNX
    print("Converting model to ONNX format...")
    try:
        # Try different conversion options based on model type
        options = {
            'zipmap': False,  # Disable zipmap which often creates map outputs
            'nocl': True      # Avoid using custom loops which can lead to sequence outputs
        }
        
        onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
        
        # Save the model
        print(f"Saving ONNX model to {onnx_file}...")
        with open(onnx_file, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Verify the model can be loaded
        print("Verifying the model...")
        sess = rt.InferenceSession(onnx_file)
        input_name = sess.get_inputs()[0].name
        print(f"Model input name: {input_name}")
        for i, output in enumerate(sess.get_outputs()):
            print(f"Model output {i} name: {output.name}, type: {output.type}")
        
        print("Conversion completed successfully!")
        return True
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        
        if "MissingShapeCalculator" in str(e):
            print("\nIt seems the model type is not directly supported by skl2onnx.")
            print("Attempting alternative conversion approach...")
            
            try:
                # Try with custom options
                from skl2onnx.common.data_types import Int64TensorType, StringTensorType
                
                # Try different output types
                options = {
                    'zipmap': False,
                    'nocl': True,
                    'output_class_labels': False  # Don't output class labels as strings
                }
                
                # Register shape calculator if needed
                # This part may need customization based on model type
                from skl2onnx.shape_calculators.classifier import calculate_classifier_output_shapes
                
                # Try conversion with updated options
                onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)
                
                # Save the model
                print(f"Saving ONNX model to {onnx_file} (alternate method)...")
                with open(onnx_file, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
                print("Conversion completed successfully with alternate method!")
                return True
            except Exception as e2:
                print(f"Alternative conversion also failed: {str(e2)}")
                print("\nSuggestions:")
                print(" - If this is a custom model, you may need to register a custom shape calculator")
                print(" - Try simplifying the model or wrapping it in a simpler sklearn compatible interface")
                print(" - Consider using a different model type that is better supported by ONNX")
                return False
        else:
            return False

if __name__ == "__main__":
    # Paths
    pickle_file = "modelislv17.p"  # Your pickle model path
    onnx_file = "modelislv17-2.onnx"  # Output ONNX model path
    
    # Based on your previous code, using 183 features:
    # data_aux = pose_data + lh + rh where:
    # pose_data is 99 elements, lh and rh are 42 elements each
    # 99 + 42 + 42 = 183
    input_shape = (None, 183)  # (batch_size, features)
    
    # Convert and save the model
    convert_sklearn_to_onnx(pickle_file, onnx_file, input_shape)