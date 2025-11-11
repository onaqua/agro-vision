using AgroVision.YoloConverter.Tools;

namespace AgroVision;

static class Program
{
    static void Main(string[] args)
    {
        //CsvConverter.Convert(
        //    @"F:\source\x-argro\datasets\leaf-disease-detection\test_labels.csv", 
        //    @"F:\source\x-argro\datasets\leaf-disease-detection\labels", 
        //    new Dictionary<string, int>
        //    {
        //        { "Cherry leaf", 0 },
        //        { "Peach leaf", 1 },
        //        { "Corn leaf blight", 2 },
        //        { "Apple rust leaf", 3 },
        //        { "Potato leaf late blight", 4 },
        //        { "Strawberry leaf", 5 },
        //        { "Corn rust leaf", 6 },
        //        { "Tomato leaf late blight", 7 },
        //        { "Tomato mold leaf", 8 },
        //        { "Potato leaf early blight", 9 },
        //        { "Apple leaf", 10 },
        //        { "Tomato leaf yellow virus", 11 },
        //        { "Blueberry leaf", 12 },
        //        { "Tomato leaf mosaic virus", 13 },
        //        { "Raspberry leaf", 14 },
        //        { "Tomato leaf bacterial spot", 15 },
        //        { "Squash Powdery mildew leaf", 16 },
        //        { "grape leaf", 17 },
        //        { "Tomato Early blight leaf", 18 },
        //        { "Apple Scab Leaf", 19 },
        //        { "Tomato Septoria leaf spot", 20 },
        //        { "Tomato leaf", 21 },
        //        { "Soyabean leaf", 22 },
        //        { "Corn Gray leaf spot", 23 },
        //        { "Bell_pepper leaf spot", 24 },
        //        { "Bell_pepper leaf", 25 },
        //        { "grape leaf black rot", 26 },
        //        { "Potato leaf", 27 },
        //        { "Tomato two spotted spider mites leaf", 28 }
        //    });
        //// Конвертируем все файлы
        //DatasetConverter.ConvertDataset(@"../../../../../datasets/leaf-disease-detection/test", @"../../../../../datasets/leaf-disease-detection/test/labels");

        //if (!Directory.Exists("../../../../../datasets/leaf-disease-detection/test/validation"))
        //{
        //    Directory.CreateDirectory("../../../../../datasets/leaf-disease-detection/test/validation");
        //}

        ////DatasetValidator.ValidateAnnotation(
        ////    imagePath: "../../../../../datasets/leaf-disease-detection/test/_1030395.JPG.jpg",
        ////    txtPath: "../../../../../datasets/leaf-disease-detection/test/labels/_1030395.JPG.txt",
        ////    outputPath: "../../../../../datasets/leaf-disease-detection/test/labels/validation/_1030395.JPG.jpg");

        ////Console.WriteLine("Конвертация завершена!");

        ////Directory
        ////    .GetFiles("../../../../../datasets/leaf-disease-detection/test")
        ////    .Where(file => Path.GetExtension(file) == ".jpg")
        ////    .ToList()
        ////    .ForEach(image =>
        ////    {
        ////        var oldPath = Path.GetFullPath(image);
        ////        var newPath = Path.Combine("../../../../../datasets/leaf-disease-detection/test", "images", Path.GetFileName(image));

        ////        File.Move(oldPath, newPath);

        ////        Console.WriteLine("Файл перемещен в {0}", newPath);
        ////    });

        foreach (var file in Directory.GetFiles("F:\\source\\x-argro\\src\\AgroVision.DiseaseDetection\\datasets\\leaf-disease-detection\\labels\\val"))
        {
            var text = File
                .ReadAllLines(file)
                .Select(line => line.TrimEnd().Replace(",", "."));

            File.WriteAllLines(file, text);
        }
    }
}