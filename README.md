# pixel=gen

Тестирую модели генерирующие пиксельные картинки.

## Как работать

Склонируй репо.

Запуск стека утилитой `make`

- `make check` линтинг, mypy и тесты

## Установка зависимостей в рабочую область

Каждый из проектов использует свой интерпретатор. Если нуцжен линтинг - надо создать окружение для питона в папке каждого из двух генераторов.

1. Перейти в область проекта pdfgenerator/aipgenerator (открыть новый терминал и выбрать область)
2. Установить `poetry` окружение и подготовьте линтер. Для этого используй команду `poetry install --with dev --no-root`. pyproject.toml находится в корне проекта.
3. Указать путь к bнтерпретатору в `.venv`. Перезапустить IDE.

Лучший способ работы с проектом - подтянуть версию python через pyenv и сохранить настройки poetry в файле конфигураций внутри проекта. Тогда зависимости окружения будут сохранены в локальной папке внутри проекта. К примеру так:

```txt
# .python-version

3.12.5
```

```toml
# poetry.toml

virtualenvs.create = true
virtualenvs.prefer-active-python = true
# ... some other variables
```

## Run

`generate --count=10 -p --prompt="The small agressive robot with tentacles, pixelart"`

Current used: `stabilityai/stable-diffusion-xl-base-1.0` + `nerijs/pixel-art-xl`. [pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl).

## Research

- [Pixel-art icon model](https://huggingface.co/azyg0s/pixelart-50m) A small text-to-image model that draws 32×32 RGBA pixel-art game icons with transparent backgrounds.
- [Pixel Party XL](https://huggingface.co/pixelparty/pixel-party-xl) This is a full model training for better pixel art adherence based on SDXL. Feel free to use this model for your own projects, but please do not host it. `stabilityai/stable-diffusion-xl-base-1.0` + `nerijs/pixel-party-xl`
- [pixel-art-3.5L](https://huggingface.co/nerijs/pixel-art-3.5L) LoRa aimed to produce pixel perfect and grid aligned 128x128 outputs for Stable Diffusion 3.5 Large.
- [pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5](https://huggingface.co/artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5)
- [more models](https://huggingface.co/models?other=pixel+art)
