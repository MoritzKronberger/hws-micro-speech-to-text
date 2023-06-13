# Micro Text-To-Speech – Live Transcription

Includes setup for strict typechecking with [mypy](https://mypy.readthedocs.io/en/stable/index.html) and linting with [flake8](https://flake8.pycqa.org/en/latest/index.html) + settings for the [VSCode Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

## Installation

__*Uses Python 3.10*__

Create a virtual environment:

```bash
python -m venv venv
```

Active the virtual environment:

```bash
./venv/Scripts/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Create .env file

Create the .env scaffold using

```bash
bash init_dotenv.sh
```

and enter your configuration.

## Run Application

Start live transcription:

```bash
python -m app
```

## Run Typechecks and Linter

Run mypy for typechecks:

```bash
mypy -m app
```

Run flake8 for linting:

```bash
flake8
```
