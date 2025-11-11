using AgroVision.Detection.Models.Yolo;
using OpenCvSharp;
using System.Drawing;
using Point = OpenCvSharp.Point;

namespace AgroVision.Detection.Pipelines;

/// <summary>
/// Represents the result of leaf analysis containing detection and classification information
/// </summary>
public class LeafAnalysisResult
{
    /// <summary>
    /// Bounding box coordinates of the detected leaf in the original image
    /// </summary>
    public RectangleF LeafBoundingBox { get; set; }

    /// <summary>
    /// Classified health status of the leaf (e.g., disease name or 'healthy')
    /// </summary>
    public string HealthStatus { get; set; } = null!;

    /// <summary>
    /// Confidence score of the health classification (0-1)
    /// </summary>
    public float HealthConfidence { get; set; }

    /// <summary>
    /// Confidence score of the leaf detection (0-1)
    /// </summary>
    public float DetectionConfidence { get; set; }
}

/// <summary>
/// Advanced pipeline for analyzing plant leaves using a two-stage ML approach
/// 
/// Pipeline Flow:
/// 1. Detection Stage: Identifies and locates leaves in the image using YOLO model
/// 2. Classification Stage: Analyzes each detected leaf for diseases/health status
/// 3. Result Aggregation: Combines detection and classification results
/// 
/// Key Features:
/// - Robust error handling and fallback mechanisms
/// - Comprehensive logging for debugging and monitoring
/// - Memory-efficient resource management with IDisposable pattern
/// - Configurable class mappings for different plant species and diseases
/// </summary>
public class LeafAnalysisPipeline : IDisposable
{
    private readonly YoloModel _detectionModel;
    private readonly YoloModel _classificationModel;
    private bool _disposed = false;

    // Plant species for detection - configured for agricultural and botanical applications
    private static readonly string[] _detectionClasses =
    [
        "ginger", "banana", "tobacco", "ornamental", "rose", "soyabean", "papaya",
        "garlic", "raspberry", "mango", "cotton", "corn", "pomegranate", "strawberry",
        "Blueberry", "brinjal", "potato", "wheat", "olive", "rice", "lemon", "cabbage",
        "guava", "chilli", "capsicum", "sunflower", "cherry", "cassava", "apple", "tea",
        "sugarcane", "groundnut", "weed", "peach", "coffee", "cauliflower", "tomato",
        "onion", "gram", "chiku", "jamun", "castor", "pea", "cucumber", "grape", "cardamom"
    ];

    // Disease classes focused on common agricultural pathogens and conditions
    private static readonly string[] _diseaseDetectionClasses =
    [
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___healthy"
    ];

    /// <summary>
    /// Initializes a new instance of the leaf analysis pipeline with specified ML models
    /// </summary>
    /// <param name="detectionModelPath">File path to the YOLO detection model weights</param>
    /// <param name="classificationModelPath">File path to the YOLO classification model weights</param>
    /// <exception cref="ArgumentNullException">Thrown when model paths are null or empty</exception>
    /// <exception cref="FileNotFoundException">Thrown when model files cannot be located</exception>
    public LeafAnalysisPipeline(string detectionModelPath, string classificationModelPath)
    {
        if (string.IsNullOrWhiteSpace(detectionModelPath))
            throw new ArgumentNullException(nameof(detectionModelPath), "Detection model path cannot be null or empty");

        if (string.IsNullOrWhiteSpace(classificationModelPath))
            throw new ArgumentNullException(nameof(classificationModelPath), "Classification model path cannot be null or empty");

        // Validate model file existence
        if (!File.Exists(detectionModelPath))
            throw new FileNotFoundException($"Detection model not found at: {detectionModelPath}");

        if (!File.Exists(classificationModelPath))
            throw new FileNotFoundException($"Classification model not found at: {classificationModelPath}");

        // Initialize YOLO models with appropriate class mappings
        _detectionModel = new YoloModel(detectionModelPath, _detectionClasses);
        _classificationModel = new YoloModel(classificationModelPath, _diseaseDetectionClasses);
    }

    /// <summary>
    /// Performs comprehensive leaf analysis on the specified image
    /// 
    /// Analysis Steps:
    /// 1. Image loading and validation
    /// 2. Leaf detection using primary YOLO model
    /// 3. Per-leaf disease classification using secondary YOLO model
    /// 4. Result compilation and resource cleanup
    /// </summary>
    /// <param name="imagePath">Path to the input image file</param>
    /// <returns>List of LeafAnalysisResult objects containing detection and classification data</returns>
    /// <exception cref="ArgumentNullException">Thrown when image path is null or empty</exception>
    /// <exception cref="FileNotFoundException">Thrown when image file cannot be found</exception>
    /// <exception cref="InvalidOperationException">Thrown when image processing fails</exception>
    public List<LeafAnalysisResult> AnalyzeImage(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
            throw new ArgumentNullException(nameof(imagePath), "Image path cannot be null or empty");

        if (!File.Exists(imagePath))
            throw new FileNotFoundException($"Image file not found: {imagePath}");

        // Load and validate input image
        using var sourceImage = Cv2.ImRead(imagePath);
        if (sourceImage.Empty())
            throw new InvalidOperationException($"Failed to load image or image is empty: {imagePath}");

        return AnalyzeImage(sourceImage);
    }

    /// <summary>
    /// Performs leaf analysis on a pre-loaded OpenCV Mat image
    /// </summary>
    /// <param name="sourceImage">OpenCV Mat object containing the source image</param>
    /// <returns>List of LeafAnalysisResult objects</returns>
    public List<LeafAnalysisResult> AnalyzeImage(Mat sourceImage)
    {
        if (sourceImage == null)
            throw new ArgumentNullException(nameof(sourceImage), "Source image cannot be null");

        if (sourceImage.Empty())
            throw new ArgumentException("Source image is empty", nameof(sourceImage));

        var results = new List<LeafAnalysisResult>();

        // Create debug visualization copy (could be extended to save debug images)
        using var debugImage = sourceImage.Clone();

        // Stage 1: Detect leaves in the image
        var detectedLeaves = _detectionModel.Predict(sourceImage);
        Console.WriteLine($"Detected {detectedLeaves.Count} potential leaves in image");

        // Stage 2: Process each detected leaf
        foreach (var leaf in detectedLeaves)
        {
            // Skip low-confidence detections to reduce false positives
            if (leaf.Confidence < 0.1f)
            {
                Console.WriteLine($"Skipping low-confidence detection: {leaf.ClassName} ({leaf.Confidence:P1})");
                continue;
            }

            // Visualize detection on debug image
            VisualizeDetection(debugImage, leaf.BoundingBox, leaf.ClassName);

            // Extract leaf region for detailed analysis
            using var leafRegion = ExtractLeafRegion(sourceImage, leaf.BoundingBox);

            // Stage 3: Classify leaf health status
            var healthAssessment = AssessLeafHealth(leafRegion);

            results.Add(new LeafAnalysisResult
            {
                LeafBoundingBox = leaf.BoundingBox,
                HealthStatus = healthAssessment.ClassName,
                HealthConfidence = healthAssessment.Confidence,
                DetectionConfidence = leaf.Confidence
            });
        }

        Console.WriteLine($"Completed analysis of {results.Count} leaves");
        return results;
    }

    /// <summary>
    /// Extracts and validates region of interest (ROI) containing a single leaf
    /// 
    /// Safety Features:
    /// - Boundary validation to prevent out-of-range exceptions
    /// - Dimension verification to ensure valid ROI size
    /// - Fallback to full image if extraction fails
    /// </summary>
    /// <param name="sourceImage">Source image matrix</param>
    /// <param name="boundingBox">Bounding box coordinates for ROI extraction</param>
    /// <returns>Mat object containing the extracted leaf region</returns>
    private static Mat ExtractLeafRegion(Mat sourceImage, RectangleF boundingBox)
    {
        if (sourceImage == null)
            throw new ArgumentNullException(nameof(sourceImage));

        try
        {
            // Validate and adjust bounding box coordinates to image boundaries
            var (x, y, width, height) = ValidateAndAdjustBoundingBox(sourceImage, boundingBox);

            Console.WriteLine($"Extracting leaf region - Original: {boundingBox}, Adjusted: X={x}, Y={y}, W={width}, H={height}");

            // Extract region of interest
            var roi = new Mat(sourceImage, new Rect(x, y, width, height));

            // Verify extraction success
            if (roi.Empty())
            {
                Console.WriteLine("WARNING: Extracted ROI is empty, falling back to full image");
                return sourceImage.Clone();
            }

            return roi;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: Failed to extract leaf region: {ex.Message}");
            return sourceImage.Clone(); // Fallback to full image
        }
    }

    /// <summary>
    /// Validates and adjusts bounding box coordinates to ensure they remain within image boundaries
    /// </summary>
    /// <param name="image">Source image for boundary reference</param>
    /// <param name="boundingBox">Original bounding box coordinates</param>
    /// <returns>Validated and adjusted bounding box coordinates</returns>
    private static (int x, int y, int width, int height) ValidateAndAdjustBoundingBox(Mat image, RectangleF boundingBox)
    {
        // Calculate safe coordinates within image boundaries
        float safeX = Math.Clamp(boundingBox.X, 0, image.Width - 1);
        float safeY = Math.Clamp(boundingBox.Y, 0, image.Height - 1);
        float safeWidth = Math.Clamp(boundingBox.Width, 1, image.Width - safeX);
        float safeHeight = Math.Clamp(boundingBox.Height, 1, image.Height - safeY);

        // Convert to integer coordinates for OpenCV Rect
        int x = (int)Math.Floor(safeX);
        int y = (int)Math.Floor(safeY);
        int width = (int)Math.Ceiling(safeWidth);
        int height = (int)Math.Ceiling(safeHeight);

        // Final boundary validation
        x = Math.Clamp(x, 0, image.Width - 1);
        y = Math.Clamp(y, 0, image.Height - 1);
        width = Math.Clamp(width, 1, image.Width - x);
        height = Math.Clamp(height, 1, image.Height - y);

        return (x, y, width, height);
    }

    /// <summary>
    /// Classifies leaf health status using the specialized disease classification model
    /// 
    /// Features:
    /// - Confidence-based result selection
    /// - Comprehensive error handling
    /// - Fallback results for edge cases
    /// </summary>
    /// <param name="leafImage">ROI image containing a single leaf</param>
    /// <returns>DetectionResult with health classification and confidence</returns>
    private DetectionResult AssessLeafHealth(Mat leafImage)
    {
        if (leafImage == null)
            throw new ArgumentNullException(nameof(leafImage));

        try
        {
            var classificationResults = _classificationModel.Predict(leafImage);

            if (classificationResults.Any())
            {
                // Select highest confidence result
                var primaryResult = classificationResults.OrderByDescending(r => r.Confidence).First();

                Console.WriteLine(
                    $"Health assessment: {primaryResult.ClassName} " +
                    $"(confidence: {primaryResult.Confidence:P1})");

                return primaryResult;
            }

            // Fallback for no results
            Console.WriteLine("No classification results available, using fallback");
            return CreateFallbackResult(leafImage, "unknown", 0f);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Health assessment error: {ex.Message}");
            return CreateFallbackResult(leafImage, "classification_error", 0f);
        }
    }

    /// <summary>
    /// Creates a fallback detection result for error scenarios
    /// </summary>
    private static DetectionResult CreateFallbackResult(Mat image, string className, float confidence)
    {
        return new DetectionResult
        {
            ClassName = className,
            Confidence = confidence,
            BoundingBox = new RectangleF(0, 0, image.Width, image.Height)
        };
    }

    /// <summary>
    /// Visualizes detection results on the debug image with bounding boxes and labels
    /// </summary>
    /// <param name="image">Target image for visualization</param>
    /// <param name="boundingBox">Bounding box coordinates</param>
    /// <param name="label">Detection label text</param>
    private static void VisualizeDetection(Mat image, RectangleF boundingBox, string label)
    {
        try
        {
            int x = (int)boundingBox.X;
            int y = (int)boundingBox.Y;
            int width = (int)boundingBox.Width;
            int height = (int)boundingBox.Height;

            // Draw bounding box
            Cv2.Rectangle(image, new Rect(x, y, width, height), Scalar.Red, 2);

            // Draw label with background for better visibility
            var labelPosition = new Point(x, y - 5);
            Cv2.PutText(image, label, labelPosition, HersheyFonts.HersheySimplex, 0.5, Scalar.Red, 1);

            Console.WriteLine($"Visualized detection: {label} at [{x}, {y}, {width}, {height}]");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Visualization error: {ex.Message}");
        }
    }

    /// <summary>
    /// Disposes managed resources used by the pipeline
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Protected implementation of Dispose pattern
    /// </summary>
    /// <param name="disposing">True if called from Dispose, false if from finalizer</param>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                // Dispose managed resources
                _detectionModel?.Dispose();
                _classificationModel?.Dispose();
            }

            _disposed = true;
        }
    }

    ~LeafAnalysisPipeline()
    {
        Dispose(false);
    }
}