// class-mapping.js
// This script creates a mapping between numeric class indices and string labels

// Define your class labels here - replace with your actual class names
const classLabels = [
    "hello",     // 0
    "thanks",    // 1
    "iloveyou",  // 2
    "yes",       // 3
    "no",        // 4
    "please",    // 5
    "sorry",     // 6
    "goodbye",   // 7
    "help",      // 8
    "love"       // 9
    // Add more class labels as needed
  ];
  
  // Save to a JSON file for use in the inference script
  const fs = require('fs');
  
  fs.writeFileSync('class-mapping.json', JSON.stringify({
    labels: classLabels
  }, null, 2));
  
  console.log(`Created class mapping with ${classLabels.length} labels`);
  console.log('Saved to class-mapping.json');