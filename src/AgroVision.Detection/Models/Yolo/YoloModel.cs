using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Drawing;
using Size = OpenCvSharp.Size;

namespace AgroVision.Detection.Models.Yolo;

/// <summary>
/// Represents a single detection result from YOLO model inference
/// Contains bounding box information, confidence scores, and classification data
/// </summary>
public class DetectionResult
{
    /// <summary>
    /// Bounding box coordinates in the original image space
    /// Stored as RectangleF for floating-point precision
    /// </summary>
    public RectangleF BoundingBox { get; set; }

    /// <summary>
    /// Detection confidence score ranging from 0.0 to 1.0
    /// Represents model certainty about the detection
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Human-readable class name for the detected object
    /// Mapped from class ID using the class names array
    /// </summary>
    public string ClassName { get; set; } = null!;

    /// <summary>
    /// Numeric class identifier from the model output
    /// Used for internal processing and class mapping
    /// </summary>
    public int ClassId { get; set; }
}

/// <summary>
/// YOLO Model wrapper for ONNX Runtime providing object detection capabilities
/// 
/// Key Features:
/// - GPU and CPU execution provider support
/// - Automatic input preprocessing and output postprocessing
/// - Non-Maximum Suppression (NMS) for duplicate removal
/// - Configurable confidence thresholds and NMS parameters
/// - Comprehensive error handling and logging
/// 
/// Pipeline Flow:
/// 1. Input Image → Preprocessing (Resize, Normalization, CHW conversion)
/// 2. ONNX Inference → Tensor operations via ONNX Runtime
/// 3. Output Processing → Confidence filtering, coordinate transformation
/// 4. NMS Application → Duplicate detection removal
/// 5. Result Formatting → Bounding box conversion and class mapping
/// </summary>
public class YoloModel : IDisposable
{
    private readonly string[] _classNames;
    private readonly InferenceSession _session;
    private readonly object _sessionLock = new object();
    private bool _disposed = false;

    // Model configuration constants
    private const int ModelInputSize = 640;
    private const float DefaultConfidenceTreshold = 0.1f;
    private const float DefaultIouTreshold = 0.5f;
    private const int ValuesPerDetection = 6; // [x1, y1, x2, y2, confidence, class_id]

    /// <summary>
    /// Initializes a new instance of YOLO model wrapper
    /// </summary>
    /// <param name="modelPath">File system path to the ONNX model file</param>
    /// <param name="classNames">Array of class names for detection result mapping</param>
    /// <exception cref="ArgumentNullException">Thrown when model path or class names are null</exception>
    /// <exception cref="FileNotFoundException">Thrown when model file cannot be found</exception>
    /// <exception cref="InvalidOperationException">Thrown when model initialization fails</exception>
    public YoloModel(string modelPath, string[] classNames)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentNullException(nameof(modelPath), "Model path cannot be null or empty");

        if (classNames == null || classNames.Length == 0)
            throw new ArgumentNullException(nameof(classNames), "Class names cannot be null or empty");

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"YOLO model file not found: {modelPath}");

        _classNames = classNames;

        try
        {
            var sessionOptions = CreateSessionOptions();
            _session = new InferenceSession(modelPath, sessionOptions);

            Console.WriteLine($"YOLO model initialized successfully: {modelPath}");
            Console.WriteLine($"Number of classes: {_classNames.Length}");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to initialize YOLO model: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Creates and configures ONNX Runtime session options
    /// Prioritizes GPU execution providers when available, falls back to CPU
    /// </summary>
    /// <returns>Configured SessionOptions instance</returns>
    private static SessionOptions CreateSessionOptions()
    {
        var options = new SessionOptions();

        try
        {
            // Prefer GPU execution for better performance
            // options.AppendExecutionProvider_CPU();

            // Note: In production, consider adding CUDA/DirectML providers
            options.AppendExecutionProvider_CUDA(0); // For NVIDIA GPUs
            // options.AppendExecutionProvider_DML(0);  // For DirectML (Windows)

            options.EnableMemoryPattern = false; // Better for variable input sizes
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            Console.WriteLine("Session options configured with CPU execution provider");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to configure optimized session options: {ex.Message}");
            // Continue with default options
        }

        return options;
    }

    /// <summary>
    /// Performs object detection on the input image using the YOLO model
    /// 
    /// Processing Steps:
    /// 1. Input validation and sanity checks
    /// 2. Image preprocessing (resize, normalization, tensor conversion)
    /// 3. Model inference execution
    /// 4. Output postprocessing and result formatting
    /// 5. Non-Maximum Suppression for duplicate removal
    /// </summary>
    /// <param name="image">Input image as OpenCV Mat object</param>
    /// <returns>List of DetectionResult objects sorted by confidence</returns>
    /// <exception cref="ArgumentNullException">Thrown when input image is null</exception>
    /// <exception cref="InvalidOperationException">Thrown when image processing fails</exception>
    public List<DetectionResult> Predict(Mat image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image), "Input image cannot be null");

        if (image.Empty())
            throw new ArgumentException("Input image is empty", nameof(image));

        try
        {
            // Step 1: Preprocess input image for model consumption
            var processedTensor = PreprocessImage(image);

            // Step 2: Execute model inference
            var outputTensor = ExecuteInference(processedTensor);

            // Step 3: Process model outputs and convert to detections
            var detections = ProcessModelOutputs(outputTensor, image.Width, image.Height);

            Console.WriteLine($"YOLO inference completed: {detections.Count} detections found");
            return detections;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"YOLO prediction failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Preprocesses input image for YOLO model consumption
    /// 
    /// Processing Steps:
    /// - Resize to model input size (640x640) with aspect ratio consideration
    /// - Convert BGR to RGB color space
    /// - Normalize pixel values to [0, 1] range
    /// - Convert to CHW (Channel-Height-Width) format
    /// - Create batch dimension for model input
    /// </summary>
    /// <param name="image">Source image in BGR format</param>
    /// <returns>Preprocessed tensor ready for model inference</returns>
    private static DenseTensor<float> PreprocessImage(Mat image)
    {
        using var resizedImage = new Mat();

        // Resize image to model input size while maintaining aspect ratio
        Cv2.Resize(image, resizedImage, new Size(ModelInputSize, ModelInputSize));

        var tensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });

        // Process each pixel for normalization and format conversion
        for (int y = 0; y < ModelInputSize; y++)
        {
            for (int x = 0; x < ModelInputSize; x++)
            {
                var pixel = resizedImage.Get<Vec3b>(y, x);

                // Convert BGR to RGB and normalize to [0, 1] range
                // Note: YOLO models typically expect RGB input with normalization
                tensor[0, 0, y, x] = pixel[2] / 255.0f; // Red channel
                tensor[0, 1, y, x] = pixel[1] / 255.0f; // Green channel  
                tensor[0, 2, y, x] = pixel[0] / 255.0f; // Blue channel
            }
        }

        Console.WriteLine($"Image preprocessing completed: {image.Width}x{image.Height} -> {ModelInputSize}x{ModelInputSize}");
        return tensor;
    }

    /// <summary>
    /// Executes model inference with thread safety
    /// 
    /// Note: InferenceSession is not thread-safe, so we use locking
    /// for concurrent access scenarios
    /// </summary>
    /// <param name="inputTensor">Preprocessed input tensor</param>
    /// <returns>Model output tensor containing detection data</returns>
    private Tensor<float> ExecuteInference(DenseTensor<float> inputTensor)
    {
        lock (_sessionLock)
        {
            // Prepare input for ONNX Runtime
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            // Execute inference
            using var results = _session.Run(inputs);
            var output = results[0].Value as Tensor<float>;

            if (output == null)
                throw new InvalidOperationException("Model returned null output");

            Console.WriteLine($"Inference executed successfully. Output dimensions: [{string.Join(", ", output.Dimensions.ToArray())}]");
            return output;
        }
    }

    /// <summary>
    /// Processes raw model outputs into formatted detection results
    /// 
    /// Processing Steps:
    /// - Parse detection data from output tensor
    /// - Apply confidence threshold filtering
    /// - Validate class ID ranges
    /// - Convert normalized coordinates to original image space
    /// - Apply Non-Maximum Suppression
    /// </summary>
    /// <param name="output">Raw model output tensor</param>
    /// <param name="originalWidth">Original image width for coordinate scaling</param>
    /// <param name="originalHeight">Original image height for coordinate scaling</param>
    /// <returns>Filtered and processed detection results</returns>
    private List<DetectionResult> ProcessModelOutputs(Tensor<float> output, int originalWidth, int originalHeight)
    {
        var predictions = output.ToArray();
        int numDetections = predictions.Length / ValuesPerDetection;

        Console.WriteLine($"Processing {numDetections} raw detections from model output");

        var rawResults = new List<DetectionResult>();

        for (int i = 0; i < numDetections; i++)
        {
            int baseIndex = i * ValuesPerDetection;

            // Extract detection values - format: [x1, y1, x2, y2, confidence, class_id]
            float x1 = predictions[baseIndex];
            float y1 = predictions[baseIndex + 1];
            float x2 = predictions[baseIndex + 2];
            float y2 = predictions[baseIndex + 3];
            float confidence = predictions[baseIndex + 4];
            int classId = (int)predictions[baseIndex + 5];

            // Apply confidence threshold
            if (confidence < DefaultConfidenceTreshold)
                continue;

            // Validate class ID
            if (!IsValidClassId(classId))
            {
                Console.WriteLine($"Warning: Invalid class ID {classId}. Maximum allowed: {_classNames.Length - 1}");
                continue;
            }

            var detection = CreateDetectionResult(x1, y1, x2, y2, confidence, classId, originalWidth, originalHeight);
            rawResults.Add(detection);
        }

        Console.WriteLine($"{rawResults.Count} detections passed confidence threshold ({DefaultConfidenceTreshold})");

        // Apply Non-Maximum Suppression to remove duplicates
        return ApplyNonMaximumSuppression(rawResults);
    }

    /// <summary>
    /// Validates class ID against available class names
    /// </summary>
    private bool IsValidClassId(int classId)
    {
        return classId >= 0 && classId < _classNames.Length;
    }

    /// <summary>
    /// Creates a DetectionResult from raw detection data
    /// </summary>
    private DetectionResult CreateDetectionResult(
        float x1, float y1, float x2, float y2,
        float confidence, int classId,
        int originalWidth, int originalHeight)
    {
        // Convert normalized coordinates to original image space
        float scaleX = originalWidth / (float)ModelInputSize;
        float scaleY = originalHeight / (float)ModelInputSize;

        x1 *= scaleX;
        y1 *= scaleY;
        x2 *= scaleX;
        y2 *= scaleY;

        // Convert from (x1, y1, x2, y2) to (x, y, width, height) format
        float x = Math.Min(x1, x2); // Ensure x is left coordinate
        float y = Math.Min(y1, y2); // Ensure y is top coordinate
        float width = Math.Abs(x2 - x1);
        float height = Math.Abs(y2 - y1);

        // Clamp coordinates to image boundaries
        x = Math.Clamp(x, 0, originalWidth - 1);
        y = Math.Clamp(y, 0, originalHeight - 1);
        width = Math.Clamp(width, 1, originalWidth - x);
        height = Math.Clamp(height, 1, originalHeight - y);

        return new DetectionResult
        {
            BoundingBox = new RectangleF(x, y, width, height),
            Confidence = confidence,
            ClassName = _classNames[classId],
            ClassId = classId
        };
    }

    /// <summary>
    /// Applies Non-Maximum Suppression to filter overlapping detections
    /// 
    /// Algorithm:
    /// 1. Sort detections by confidence score (descending)
    /// 2. Take highest confidence detection and remove overlapping detections
    /// 3. Repeat until no detections remain
    /// 
    /// This ensures we keep the most confident detection when multiple boxes
    /// identify the same object with significant overlap
    /// </summary>
    /// <param name="detections">Input detections before NMS</param>
    /// <param name="iouThreshold">Intersection-over-Union threshold for overlap detection</param>
    /// <returns>Filtered detections after NMS application</returns>
    private static List<DetectionResult> ApplyNonMaximumSuppression(
        List<DetectionResult> detections,
        float iouThreshold = DefaultIouTreshold)
    {
        var results = new List<DetectionResult>();
        var remainingDetections = new List<DetectionResult>(detections.OrderByDescending(d => d.Confidence));

        while (remainingDetections.Count > 0)
        {
            // Take the highest confidence detection
            var currentDetection = remainingDetections[0];
            results.Add(currentDetection);
            remainingDetections.RemoveAt(0);

            // Remove detections that significantly overlap with the current one
            remainingDetections = remainingDetections
                .Where(detection => CalculateIoU(currentDetection.BoundingBox, detection.BoundingBox) < iouThreshold)
                .ToList();
        }

        Console.WriteLine($"NMS reduced {detections.Count} detections to {results.Count} (IoU threshold: {iouThreshold})");
        return results;
    }

    /// <summary>
    /// Calculates Intersection over Union (IoU) between two bounding boxes
    /// 
    /// IoU Formula:
    /// IoU = Area of Intersection / Area of Union
    /// 
    /// Used to measure overlap between detections for NMS filtering
    /// </summary>
    /// <param name="box1">First bounding box</param>
    /// <param name="box2">Second bounding box</param>
    /// <returns>IoU value between 0.0 (no overlap) and 1.0 (complete overlap)</returns>
    private static float CalculateIoU(RectangleF box1, RectangleF box2)
    {
        // Calculate intersection area
        float intersectionX1 = Math.Max(box1.X, box2.X);
        float intersectionY1 = Math.Max(box1.Y, box2.Y);
        float intersectionX2 = Math.Min(box1.Right, box2.Right);
        float intersectionY2 = Math.Min(box1.Bottom, box2.Bottom);

        float intersectionWidth = Math.Max(0, intersectionX2 - intersectionX1);
        float intersectionHeight = Math.Max(0, intersectionY2 - intersectionY1);
        float intersectionArea = intersectionWidth * intersectionHeight;

        // Calculate union area
        float box1Area = box1.Width * box1.Height;
        float box2Area = box2.Width * box2.Height;
        float unionArea = box1Area + box2Area - intersectionArea;

        // Avoid division by zero
        if (unionArea <= 0)
            return 0.0f;

        return intersectionArea / unionArea;
    }

    /// <summary>
    /// Disposes managed resources used by the YOLO model
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Protected implementation of Dispose pattern
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _session?.Dispose();
            }

            _disposed = true;
        }
    }

    ~YoloModel()
    {
        Dispose(false);
    }
}