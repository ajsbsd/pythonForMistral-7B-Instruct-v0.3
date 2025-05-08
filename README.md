# pythonForMistral-7B-Instruct-v0.3

A Python-based interface for interacting with the Mistral-7B-Instruct-v0.3 language model.

## Overview

This project provides a simple Python script to interact with the Mistral-7B-Instruct-v0.3 model. It allows users to input prompts and receive generated responses, which are then logged for future reference or fine-tuning purposes.

## Features

- **Interactive Prompting**: Input prompts directly via the command line.
- **Response Generation**: Utilize the Mistral-7B-Instruct-v0.3 model to generate responses.
- **Logging**: Automatically logs each prompt and its corresponding response with a timestamp to a CSV file.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - `mistral_inference`
  - `mistral_common`

Ensure that the Mistral model and tokenizer files are available locally.

## Installation

```bash
git clone https://github.com/ajsbsd/pythonForMistral-7B-Instruct-v0.3.git
cd pythonForMistral-7B-Instruct-v0.3
pip install -r requirements.txt

