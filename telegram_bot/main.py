from model import ClassPredictor
from telegram_token import token
import torch
from config import reply_texts
import numpy as np
from PIL import Image
from io import BytesIO


model = ClassPredictor()


def send_prediction_on_photo(bot, update):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    image_stream = BytesIO()
    image_file.download(out=image_stream)

    output = model.predict(image_stream)
    _, class_index = torch.max(output, 0)

    # теперь отправим результат
    update.message.reply_text(reply_texts['pred_answer'].format(class_index))
    print("Sent Answer to user")


if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    # используем прокси, так как без него у меня ничего не работало(
    updater = Updater(token=token,  request_kwargs={'proxy_url': 'socks4://168.195.171.42:44880'})
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, send_prediction_on_photo))
    updater.start_polling()
