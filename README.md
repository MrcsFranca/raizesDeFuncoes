# 🧮 Numerical Methods Calculator
---
Aiming to facilitate the resolution of complex mathematical problems through a graphical interface, I created the Numerical Methods Calculator.

This is a software developed in Python that solves Systems of Linear Equations (using direct and iterative methods) and finds Roots of Functions. It supports data entry via text files or manually through the interface.

## 🚀 Features

### Linear Systems
* **Direct Methods:** Gauss Elimination (Simple, Partial Pivoting, Complete Pivoting), LU Decomposition, Cholesky Factorization.
* **Iterative Methods:** Gauss-Jacobi, Gauss-Seidel.
* **Validation:** Automatic convergence checks (Sassenfeld and Row Criteria).

### Roots of Functions
* **Methods:** Bisection, Fixed Point (MPF), Newton-Raphson, Secant, Regula Falsi.

## 👨‍💻 How to install and run

This project was made to run with Python 3.

First, you will need to clone my GitHub repository.

After that, you need to install the `numpy` library, which is required for the matrix calculations:

```bash
pip install numpy
```
Run the interface:

```bash
python3 interface.py
```

## 📝 License

This project has the [MIT](https://choosealicense.com/licenses/mit/) license.
