namespace AgroVision.YoloConverter.Tools;

public static class CsvConverter
{
    public static void Convert(string csvPath, string outputDirectory, IDictionary<string, int>? classes = null)
    {
        // Чтение CSV файла
        string[] csvLines = File.ReadAllLines(csvPath);

        // Пропуск заголовка и парсинг данных
        var records = csvLines.Skip(1)
            .Select(line => line.Split(','))
            .Where(parts => parts.Length == 8)
            .Select(parts => new
            {
                FileName = parts[0],
                Width = int.Parse(parts[1]),
                Height = int.Parse(parts[2]),
                Class = parts[3],
                XMin = int.Parse(parts[4]),
                YMin = int.Parse(parts[5]),
                XMax = int.Parse(parts[6]),
                YMax = int.Parse(parts[7])
            })
            .Where(r => r.Width > 0 && r.Height > 0) // Игнорируем записи с нулевыми размерами
            .GroupBy(r => r.FileName);

        // Создание маппинга классов в ID
        var uniqueClasses = records.SelectMany(g => g.Select(r => r.Class)).Distinct().ToList();

        classes ??= uniqueClasses
            .Select((className, index) => new { className, index })
            .ToDictionary(x => x.className, x => x.index);

        if (!Directory.Exists(outputDirectory))
        {
            // Создание папки для выходных файлов
            Directory.CreateDirectory(outputDirectory);
        }

        var invalidChars = Path.GetInvalidPathChars();

        // Обработка каждой группы (изображения)
        foreach (var group in records)
        {
            string labelFileName = Path.ChangeExtension(group.Key, ".txt");
            string labelPath = Path.Combine(outputDirectory, labelFileName);

            labelPath = labelPath.Replace("?", "");

            try
            {

                using StreamWriter writer = new StreamWriter(labelPath);

                foreach (var record in group)
                {
                    // Расчет нормализованных координат YOLO
                    double xCenter = (record.XMin + record.XMax) / 2.0 / record.Width;
                    double yCenter = (record.YMin + record.YMax) / 2.0 / record.Height;
                    double width = (record.XMax - record.XMin) / (double)record.Width;
                    double height = (record.YMax - record.YMin) / (double)record.Height;

                    // Обеспечение попадания координат в диапазон [0,1]
                    xCenter = Math.Max(0, Math.Min(1, xCenter));
                    yCenter = Math.Max(0, Math.Min(1, yCenter));
                    width = Math.Max(0, Math.Min(1, width));
                    height = Math.Max(0, Math.Min(1, height));

                    int classId = classes[record.Class];
                    writer.WriteLine($"{classId} {xCenter:F6} {yCenter:F6} {width:F6} {height:F6}");
                }
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Ignore: {ex.Message}");
            }
        }

        // Сохранение файла с именами классов
        File.WriteAllLines("classes.txt", classes.Keys);

        Console.WriteLine("Conversion completed!");
        Console.WriteLine($"Found {classes.Count} classes");

        foreach (var item in classes)
        {
            Console.WriteLine($"{item.Value}: {item.Key}");
        }
    }
}
