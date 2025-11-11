using AgroVision.Detection.Pipelines;

using var pipeline = new LeafAnalysisPipeline(
    detectionModelPath: @"Resources\Weights\leaves-detection.onnx",
    classificationModelPath: @"Resources\Weights\leaves-disease-detection.onnx");

var predictions = pipeline
    .AnalyzeImage(@"Resources\Images\potatoes.jpg");

foreach (var prediction in predictions)
{
    Console.WriteLine("Found leaf with health status: {0}, with confidence: {1}, with bb: {2}", 
        prediction.HealthStatus,
        prediction.HealthConfidence,
        prediction.LeafBoundingBox);
}