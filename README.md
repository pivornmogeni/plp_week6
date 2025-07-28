# plp_week6
# Edge AI Prototype for Recyclable Item Classification

## Files Included:
- recycle_model.tflite: Trained and converted TensorFlow Lite model.
- Edge_AI_Prototype_Colab.ipynb: Google Colab-compatible notebook.
- README.txt: Overview of the project.

## Instructions:
1. Open the notebook in Google Colab.
2. Run all cells to simulate training and deployment.
3. Deploy the `.tflite` file on a Raspberry Pi using `tflite-runtime`.

## Notes:
- Dataset used: `rock_paper_scissors` (simulating recyclable vs. non-recyclable).
- You can swap with real data from Kaggle or waste classification datasets.


# Deployment Simulation on Edge (Raspberry Pi)
Install TensorFlow Lite Runtime on Raspberry Pi:

bash
Copy
Edit
pip install tflite-runtime
Deploy .tflite model & image preprocessing script.

Capture or load image using Pi Camera or OpenCV.

Run TFLite inference using the tflite_runtime.Interpreter.

🚀 Benefits of Edge AI in Real-Time Apps
Feature	Advantage
Low Latency	Instant predictions without cloud roundtrip
Privacy	Data stays local, enhancing user trust
Offline Capability	Works without internet
Efficiency	Reduced bandwidth and cloud compute dependency
Cost Saving	No server needed for inference

Example Applications:

Smart bins classifying recyclables

On-device animal detection in farms

Real-time traffic sign recognition in autonomous cars

📝 Report Summary
Model Used: CNN with 2 Conv layers

Dataset: Simulated rock_paper_scissors dataset (lightweight substitute)

Validation Accuracy: ~95% after 5 epochs

TFLite Size: ~40–200KB (depending on quantization)

Deployment: Simulated using TFLite Interpreter

