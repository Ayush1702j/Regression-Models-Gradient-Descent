# GitHub Upload — Quick Guide

## 1. Create GitHub repository

Repository name:

`Regression-Gradient-Descent`

## 2. Open terminal inside this project folder

```bash
cd Regression_Gradient_Descent
```

## 3. Run and verify the project

```bash
pip install -r requirements.txt
python main.py
```

Check that `results/` and `graphs/` contain generated files.

## 4. Upload to GitHub

```bash
git init
git add .
git commit -m "Regression models and Gradient Descent implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Regression-Gradient-Descent.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## 5. Important

Do not upload:

- `.venv/`
- `__pycache__/`
- `.env`
- editor temporary files

These are already excluded by `.gitignore`.
