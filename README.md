# LCT MISIS Boriskin AI Hackathon

Привет! Это репозиторий нашего back end проекта для хакатона. Мы создали это с использованием некоторых из самых мощных инструментов: FastAPI для быстрого создания API, PyTorch для глубокого обучения, SQLAlchemy для работы с базой данных и Postgres как наш основной сервер базы данных.

Ссылка на репозиторий фронта:
[https://github.com/SlDo/lct](https://github.com/SlDo/lct)

## Основные особенности:

- **FastAPI**: Современный, быстрый (высокопроизводительный) веб-фреймворк для построения API с Python 3.7+.
- **PyTorch**: Открытая библиотека машинного обучения, используемая для приложений, таких как компьютерное зрение и глубокое обучение.
- **SQLAlchemy**: Популярный инструмент для работы с базами данных в Python.
- **Postgres**: Надежная и мощная система управления базами данных.
- **Docker**: Контейнеризация приложений.
- **Redis**: Быстрая база данных, которая может служить в качестве кэша, базы данных и очереди сообщений.

## Как запустить

Для запуска проекта удостоверьтесь, что у вас установлен Docker и Docker Compose, а затем выполните следующие шаги:

1. Клонировать репозиторий:
```
git clone https://github.com/redpower5x5/lct_back
```

2. Перейдите в директорию проекта бэкенда:
```
cd lct_worker
```

3. Запустите проект с помощью Docker Compose:
```
docker-compose up
```

После выполнения этих команд, API будет доступен по адресу `http://localhost:3333`.

## Доступ

вы можете авторизоваться на нашей платформе [https://lct.onixx.ru](https://lct.onixx.ru)
Логин: stepovoy.dev@gmail.com
Пароль: admin

## Документация

Вы можете просмотреть полную документацию к нашему API, используя Swagger по адресу: [https://lctapi.onixx.ru/docs](https://lctapi.onixx.ru/docs).