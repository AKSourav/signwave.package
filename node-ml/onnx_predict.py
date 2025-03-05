
import onnxruntime as ort
import numpy as np
import json
import sys
import os

def run_inference(input_file, model_path):
    # Load input data
    print(f"Loading input data from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract filename without extension to get expected label
    base_filename = os.path.basename(input_file)
    expected_label = None
    if '-' in base_filename:
        expected_label = base_filename.split('-')[1].split('.')[0]
    
    input_data = np.array([data['input_data']], dtype=np.float32)
    
    print("Loading ONNX model...")
    # Load ONNX session
    session = ort.InferenceSession(model_path)
    
    # Print input and output info
    input_name = session.get_inputs()[0].name
    print(f"Model input name: {input_name}")
    
    output_names = [output.name for output in session.get_outputs()]
    print(f"Model output names: {output_names}")
    
    print(f"Running inference with input shape: {input_data.shape}")
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    
    # Process results
    result_dict = {}
    
    for i, output_name in enumerate(output_names):
        if isinstance(outputs[i], np.ndarray):
            output_value = outputs[i].tolist()
        else:
            output_value = outputs[i]
        
        result_dict[output_name] = output_value
    
    # Get the label (assuming first output is label for classification)
    predicted_label = None
    if "output_label" in result_dict:
        predicted_label = result_dict["output_label"]
        if isinstance(predicted_label, list):
            predicted_label = predicted_label[0]
    
    # Create result object
    result = {
        "input_file": input_file,
        "expected_label": expected_label,
        "predicted_label": predicted_label,
        "prediction_data": result_dict
    }
    
    # Print summary
    print(f"\nInput file: {input_file}")
    print(f"Expected label: {expected_label}")
    print(f"Predicted label: {predicted_label}")
    
    if expected_label and predicted_label:
        match = expected_label == predicted_label
        print(f"Match: {match}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python onnx_predict.py <input_file> <model_path>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    model_path = sys.argv[2]
    
    result = run_inference(input_file, model_path)
    
    # Write result to stdout as JSON
    print("\nRESULT_JSON_START")
    print(json.dumps(result))
    print("RESULT_JSON_END")
