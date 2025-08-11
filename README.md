# ResMe

ResMe is a resume optimization tool that leverages LLMs to generate optimized resumes and cover letters. Outputs are saved in the `outputs/` folder.

## Features
- Optimizes resumes using LLMs
- Generates cover letters
- Example outputs included in `outputs/`

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and running scripts. Do **not** use `pip` or `venv`.

1. Install [uv](https://github.com/astral-sh/uv):
   ```powershell
   pip install uv
   ```
2. Install dependencies:
   ```powershell
   uv pip install -r requirements.txt
   ```

## Usage

To run the main script:
```powershell
uv run .\version0.py
```

Outputs will be saved in the `outputs/` folder. Example outputs are provided for reference; you may delete them if you wish, but the folder is required for saving new results.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT
