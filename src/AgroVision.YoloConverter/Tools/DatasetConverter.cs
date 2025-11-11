using System.Xml;

namespace AgroVision.YoloConverter.Tools;

public static class DatasetConverter
{
    private static Dictionary<string, int> _mappings = new Dictionary<string, int>();

    private static void ConvertXmlToYoloTxt(string xmlFilePath, string outputTxtPath)
    {
        try
        {
            // Загружаем XML документ
            var xmlDoc = new XmlDocument();
            xmlDoc.Load(xmlFilePath);

            // Получаем размеры изображения
            var sizeNode = xmlDoc.SelectSingleNode("annotation/size");
            int width = int.Parse(sizeNode.SelectSingleNode("width").InnerText);
            int height = int.Parse(sizeNode.SelectSingleNode("height").InnerText);

            // Получаем все объекты
            var objectNodes = xmlDoc.SelectNodes("annotation/object");

            using (var writer = new StreamWriter(outputTxtPath))
            {
                foreach (XmlNode objNode in objectNodes)
                {
                    var className = objNode.SelectSingleNode("name").InnerText;

                    // Получаем ID класса
                    if (!_mappings.TryGetValue(className, out int classId))
                    {
                        _mappings.Add(className, _mappings.Count);

                        Console.WriteLine($"{_mappings.Last().Value}: {_mappings.Last().Key}");
                    }

                    // Получаем bounding box
                    var bndbox = objNode.SelectSingleNode("bndbox");
                    int xmin = int.Parse(bndbox.SelectSingleNode("xmin").InnerText);
                    int ymin = int.Parse(bndbox.SelectSingleNode("ymin").InnerText);
                    int xmax = int.Parse(bndbox.SelectSingleNode("xmax").InnerText);
                    int ymax = int.Parse(bndbox.SelectSingleNode("ymax").InnerText);

                    // Конвертируем в YOLO формат
                    double xCenter = (xmin + xmax) / 2.0 / width;
                    double yCenter = (ymin + ymax) / 2.0 / height;
                    double w = (xmax - xmin) / (double)width;
                    double h = (ymax - ymin) / (double)height;

                    // Записываем в файл
                    writer.WriteLine($"{classId} {xCenter:F6} {yCenter:F6} {w:F6} {h:F6}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка при обработке файла {xmlFilePath}: {ex.Message}");
        }
    }

    public static void ConvertDataset(string xmlDir, string txtDir)
    {
        // Создаем директорию для выходных файлов если не существует
        if (!Directory.Exists(txtDir))
        {
            Directory.CreateDirectory(txtDir);
        }

        // Получаем все XML файлы
        var xmlFiles = Directory.GetFiles(xmlDir, "*.xml");

        foreach (string xmlFile in xmlFiles)
        {
            string fileName = Path.GetFileNameWithoutExtension(xmlFile);
            string txtFile = Path.Combine(txtDir, fileName + ".txt");

            //Console.WriteLine($"Конвертируем: {fileName}");

            ConvertXmlToYoloTxt(xmlFile, txtFile);
        }

        //Console.WriteLine($"Обработано файлов: {xmlFiles.Length}");
    }
}
