import tkinter as tk

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 299, height = 299, bg='black')
canvas1.pack()

def start(): 
    import cv2
    import os
    import re
    import asyncio
    from aiogram import Bot, Dispatcher, types
    import time
    from datetime import datetime, timedelta
    from transliterate import translit
    import face_recognition
    import numpy as np
    import glob

    class SimpleFacerec:
        def __init__(self):
            self.known_face_encodings = []
            self.known_face_names = []
            self.frame_resizing = 0.25

        def load_encoding_images(self, images_path):
            images_path = glob.glob(os.path.join(images_path, "*.*"))

            print("{} encoding images found.".format(len(images_path)))

            for img_path in images_path:
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                basename = os.path.basename(img_path)
                (filename, ext) = os.path.splitext(basename)
                img_encoding = face_recognition.face_encodings(rgb_img)[0]

                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
            print("Encoding images loaded")

        def detect_known_faces(self, frame):
            if frame is None or len(frame) == 0:
                return np.array([]), []

            small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)

            if not face_locations:
                return np.array([]), []

            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)

            face_locations = np.array(face_locations)
            face_locations = face_locations / self.frame_resizing
            return face_locations.astype(int), face_names

        def get_face_encoding_path(self, name):
            return os.path.join("images", f"{name}.jpg")

    TOKEN_API = "Здесь_Токен" #Объявляем токен бота (заменить "Здесь_токен" на дейстивтельный токен бота)
    YOUR_CHAT_ID = "Здесь_ID" #Объявляем айди чата, в который бот будет отправлять сообщения (заменить "Здесь_ID" на действительное айди чата) 
    data_folder = "images/" #Указываем путь к папке с фотографиями учеников

    sfr = SimpleFacerec()
    sfr.load_encoding_images(data_folder) #Загружаем фотографии

    cameras = [cv2.VideoCapture(0)] #Указываем источник(-и) захвата картинки

    bot = Bot(TOKEN_API)
    dp = Dispatcher(bot)

    cooldown_time_bot = 10  # Лимит отправки сообщений от бота (в секундах)
    name_last_sent_time = {}  # Словарь для хранения временных меток последних отправок имен

    # Время, в которое бот должен отправить список пришедших
    global target_time; 
    target_time = datetime.now().replace(hour=9, minute=5, second=00, microsecond=0)

    async def send_lists(valid_names):
        valid_names = [name.split("_", 1)[0] for name in valid_names] #Форматирование всех имён для надлежащего обновления списков пришедших и отсутствующих

        classes = {}
        class_pattern = re.compile(r'(\d+[A-Z])') #В имени каждого файла ищем закономерность: "цифра + большая буква от A до Z"

        for filename in os.listdir(data_folder): #Сканируем и создаём список учеников и классов с помощью названий фалов в директории "images/"
            match = class_pattern.search(filename)
            if match:
                person_name = filename.split("_")[0] #Форматируем
                
                class_name = match.group(1)
                if class_name in classes:
                    classes[class_name].append(person_name)
                else:
                    classes[class_name] = [person_name]
        
        for class_name, students in sorted(classes.items()):

            present_students = (set(students) & set(valid_names)) #Создаём список пришедших
            absent_students = (set(students) - set(valid_names)) #Создаём список отсутствующих

            #Переводим с латинских букв
            present_students = sorted([translit(name, 'ru') for name in present_students])
            absent_students = sorted([translit(name, "ru") for name in absent_students])

            #Объединяем в один список 
            present_students = "\n".join(present_students)
            absent_students = "\n".join(absent_students)

            class_name = translit(class_name, "ru").replace("09", "9") #Форматируем имя класса

            if present_students: #Условие заполненности списка присутствующих
                message = (f"В {class_name} пришли: \n{present_students}").replace("Ь", "ь") #Модель сообщения о наличии учеников
                print(message) #Вывод сообщения в консоль. Для тестов и наблюдений. Можно удалить
                await bot.send_message(chat_id=YOUR_CHAT_ID, text=message) #Отправка ботом сообщения
            
            if absent_students: #Условие заполненности списка отсутствующих
                message = (f"Из {class_name} отсутствуют: \n{absent_students}").replace("Ь", "ь") #Модель сообщения об отсутствии учеников
                print(message) #Вывод сообщения в консоль. Для тестов и наблюдений. Можно удалить
                await bot.send_message(chat_id=YOUR_CHAT_ID, text=message) #Отправка ботом сообщения

    async def bot_polling():
        await dp.start_polling() #Начало работы бота

    async def process_frames():
        global target_time  # Объявляем, что используем глобальную переменную
        detected_names = set()  # Множество для хранения имен всех распознанных

        while True:
            for i, cap in enumerate(cameras):
                ret, frame = cap.read()
                face_locations, face_names = sfr.detect_known_faces(frame)
                valid_names = [name for name in face_names if name != "Unknown"] #Исключение неопознанных людей из списков пришедших и отсутствующих 
                detected_names.update(valid_names)  # Обновляем множество всех распознанных имен
                valid_names = [name.split("_", 1)[0] for name in valid_names] #Форматирование имён

                current_time_bot = time.time() #Получение времени
                for name in valid_names:
                    last_sent_time = name_last_sent_time.get(name, 0)

                    # Используем valid_names в асинхронной функции
                    if valid_names and (current_time_bot - last_sent_time) >= cooldown_time_bot:
                        for name in valid_names:
                            name_last_sent_time[name] = current_time_bot

                for face_loc, name in zip(face_locations, valid_names):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3] #Получение координат лица в кадре
                    name = (translit(name.split("_", 1)[0], "ru")).replace("Ь", "ь") #Формат имени для отображения 

                    #Вывод на экран рамки и имени распознанного человека
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 2) 
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

                cv2.imshow(f"Frame Camera {i}", frame) #Вывод кадров

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif datetime.now() >= target_time or key == 32:
                await send_lists(detected_names)
                target_time = datetime.now().replace(hour=9, minute=5, second=0, microsecond=0) + timedelta(days=1)

        for cap in cameras:
            cap.release()
        cv2.destroyAllWindows()

    async def main():
        await asyncio.gather(bot_polling(), process_frames())

    asyncio.run(main())

button1 = tk.Button(text='Включить',command=start, bg='white',fg='black')
canvas1.create_window(150, 150, window=button1)

root.mainloop()