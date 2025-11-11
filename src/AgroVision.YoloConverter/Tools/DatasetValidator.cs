using System.Drawing;

namespace AgroVision.YoloConverter.Tools;

public static class DatasetValidator
{
    public static void ValidateAnnotation(string imagePath, string txtPath, string outputPath)
    {
        try
        {
            // Загружаем изображение
            using (var image = new Bitmap(imagePath))
            using (var graphics = Graphics.FromImage(image))
            {
                var pen = new Pen(Color.Red, 3);

                // Читаем аннотации
                var lines = File.ReadAllLines(txtPath);

                foreach (var line in lines)
                {
                    var parts = line.Split(' ');
                    if (parts.Length >= 5)
                    {
                        double xCenter = double.Parse(parts[1]);
                        double yCenter = double.Parse(parts[2]);
                        double width = double.Parse(parts[3]);
                        double height = double.Parse(parts[4]);

                        // Денормализуем координаты
                        int x1 = (int)((xCenter - width / 2) * image.Width);
                        int y1 = (int)((yCenter - height / 2) * image.Height);
                        int x2 = (int)((xCenter + width / 2) * image.Width);
                        int y2 = (int)((yCenter + height / 2) * image.Height);

                        // Рисуем bounding box
                        graphics.DrawRectangle(pen, x1, y1, x2 - x1, y2 - y1);

                        // Добавляем текст с классом
                        graphics.DrawString($"Class: {parts[0]}", new System.Drawing.Font("Arial", 12), Brushes.Red, x1, y1 - 20);
                    }
                }

                // Сохраняем результат
                image.Save(outputPath);
                Console.WriteLine($"Валидационное изображение сохранено: {outputPath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Ошибка при валидации: {ex.Message}");
        }
    }
}
