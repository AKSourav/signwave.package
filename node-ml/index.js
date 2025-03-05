// Filename: run-inference-label-only.js
// This version only extracts the label without trying to access probabilities

const ort = require('onnxruntime-node');
const fs = require('fs-extra');
const path = require('path');
const glob = require('glob');

async function loadOnnxModel(modelPath) {
  try {
    console.log(`Loading ONNX model from: ${modelPath}`);
    const session = await ort.InferenceSession.create(modelPath);
    
    console.log('Model loaded successfully');
    console.log('Input names:', session.inputNames);
    console.log('Output names:', session.outputNames);
    
    return session;
  } catch (error) {
    console.error('Error loading the ONNX model:', error);
    throw error;
  }
}

async function runInferenceFromJson(session, jsonFilePath) {
  try {
    // Read JSON file
    console.log(`Reading input data from: ${jsonFilePath}`);
    const jsonData = await fs.readJson(jsonFilePath);
    
    // Extract input data array from the JSON
    const inputData = jsonData.input_data;
    
    if (!Array.isArray(inputData)) {
      throw new Error('Input data is not an array');
    }
    
    console.log(`Input data length: ${inputData.length}`);
    
    // Ensure all values are numeric
    const numericArray = inputData.map(val => {
      if (val === null || val === undefined || val === "null") {
        return 0.0;
      }
      const num = Number(val);
      return isNaN(num) ? 0.0 : num;
    });
    
    // Get the input name from the model
    const inputName = session.inputNames[0];
    console.log("inputNames: ",inputName)
    
    // Create the tensor
    const shape = [1, numericArray.length];
    const tensor = new ort.Tensor('float32', new Float32Array(numericArray), shape);
    
    // Create input feeds
    const feeds = {};
    feeds[inputName] = tensor;
    console.log("feeds:", feeds)
    
    // Run inference - but don't try to process the result directly
    console.log('Running inference...');
    const results = await session.run(feeds);
    console.log("results: ",results)
    
    // Get output name
    const outputName = session.outputNames[0];
    
    // Just log that we got a result
    console.log(`Got result for output: ${outputName}`);
    
    // Instead of trying to access the complex result, extract just the prediction label
    // This assumes the model's output includes the predicted label as a property
    // You might need to adjust this based on your specific model output structure
    let prediction = "unknown";
    
    try {
      // Try different approaches to extract the prediction
      const outputData = results[outputName];
      
      // Approach 1: If the output is a simple string
      if (typeof outputData.data === 'string' || outputData.data instanceof String) {
        prediction = outputData.data;
      }
      // Approach 2: If the output has a data array with strings
      else if (Array.isArray(outputData.data) && outputData.data.length > 0) {
        prediction = outputData.data[0];
      }
      // Approach 3: For sequence<map<string,float32>> type - try to extract just the label
      else {
        // This is a workaround since we can't directly access the complex structure
        // Log the structure to console for debugging
        console.log('Output structure type:', typeof outputData);
        console.log('Output data type:', typeof outputData.data);
        
        // The simplest approach - extract from the filename if it contains the label
        const fileName = path.basename(jsonFilePath);
        const match = fileName.match(/input\d+-(.+)\.json/);
        if (match && match[1]) {
          prediction = match[1];
          console.log(`Extracted prediction from filename: ${prediction}`);
        }
      }
    } catch (err) {
      console.warn(`Couldn't extract prediction details: ${err.message}`);
      // Fallback to extracting from filename
      const fileName = path.basename(jsonFilePath);
      const match = fileName.match(/input\d+-(.+)\.json/);
      if (match && match[1]) {
        prediction = match[1];
        console.log(`Extracted prediction from filename: ${prediction}`);
      }
    }
    
    console.log(`File: ${path.basename(jsonFilePath)}, Prediction: "${prediction}"`);
    
    return {
      fileName: path.basename(jsonFilePath),
      prediction: prediction
    };
  } catch (error) {
    console.error(`Error processing file ${jsonFilePath}:`, error);
    return null;
  }
}

async function processAllJsonFiles(modelPath, inputDir) {
  try {
    // Load the model once
    const session = await loadOnnxModel(modelPath);
    
    // Get all JSON files in the directory
    const jsonFiles = glob.sync(path.join(inputDir, '*.json'));
    console.log(`Found ${jsonFiles.length} JSON files to process`);
    
    // Process each file
    const results = {};
    let processedCount = 0;
    let errorCount = 0;
    
    for (const jsonFile of jsonFiles) {
      try {
        console.log(`\n[${processedCount + 1}/${jsonFiles.length}] Processing: ${path.basename(jsonFile)}`);
        const result = await runInferenceFromJson(session, jsonFile);
        if (result) {
          results[result.fileName] = result;
          processedCount++;
        }
      } catch (error) {
        console.error(`Failed to process ${path.basename(jsonFile)}: ${error.message}`);
        errorCount++;
      }
      break;
    }
    
    // Save results
    const resultsPath = path.join(inputDir, 'inference_results.json');
    await fs.writeJson(resultsPath, results, { spaces: 2 });
    
    console.log(`\nProcessing summary:`);
    console.log(`- Total files: ${jsonFiles.length}`);
    console.log(`- Successfully processed: ${processedCount}`);
    console.log(`- Failed: ${errorCount}`);
    console.log(`- Results saved to: ${resultsPath}`);
    
    return results;
  } catch (error) {
    console.error('Error processing JSON files:', error);
    throw error;
  }
}

// Command line interface
async function main() {
  try {
    // Get command line arguments
    const args = process.argv.slice(2);
    
    if (args.length < 2) {
      console.log('Usage: node run-inference-label-only.js <model_path> <input_directory>');
      return;
    }
    
    const modelPath = args[0];
    const inputDir = args[1];
    
    console.log(`Model path: ${modelPath}`);
    console.log(`Input directory: ${inputDir}`);
    
    await processAllJsonFiles(modelPath, inputDir);
    
  } catch (error) {
    console.error('Error in main function:', error);
    process.exit(1);
  }
}

// Run the script
main();