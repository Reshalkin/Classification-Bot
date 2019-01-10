from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models


# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.
# Это всего лишь заготовка, поэтому не стесняйтесь менять имена функций, добавлять аргументы, свои классы и
# все такое.
class ClassPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Сюда необходимо перенести всю иницализацию, вроде загрузки свеерточной сети и т.д.
        self.model = models.vgg16(pretrained=True).to(self.device).train(False)
        # этот параметр стоит вынести в конфиг файл
        num_features = 25088
        self.model.classifier = nn.Linear(num_features, 2)
        # ЗДЕСЬ НУЖНО ЗАГРУЗИТЬ ВЕСА, КОТОРЫЕ ВЫ ПОЛУЧИЛИ ПРИ ТРЕНИРОВКЕ

        # Это просто скопированные трансформации, которые использовались для валидации
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])



    def predict(self, img_stream):
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
        # Не забудьте перенести все трансофрмации, которые вы использовали при тренировке
        # Для этого будет удобно сохранить питоновский объект с ними в виде файла с помощью pickle,
        # а потом загрузить здесь.

        # Обработка картинки сейчас производится в методе process image, а здесь мы должны уже применить нашу
        # модель и вернуть вектор предсказаний для нашей картинки

        # Для наглядности мы сначала переводим ее в тензор, а потом обратно
        return self.model(self.process_image(img_stream))[0]

    # В predict используются некоторые внешние функции, их можно добавить как функции класса
    # Если понятно, что функция является служебной и снаружи использоваться не должна, то перед именем функции
    # принято ставить _ (выглядит это так: def _foo() )
    # ниже пример того, как переносить методы
    def process_image(self, img_stream):
        image = Image.open(img_stream)
        # Добавляем лишнее измерение, чтобы получить батч с одной картинкой
        image = self.transforms(image).unsqueeze(0)
        return image.to(self.device, torch.float)
