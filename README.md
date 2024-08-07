# Numerical-Analysis
A comprehensive Numerical Analysis web application developed using Django. It includes implementations of various numerical methods such as Bisection, Newton's Method, Gauss-Seidel, and more, providing an interactive platform for users to perform numerical calculations and visualize results.

## Numerical Analysis Web Application

This project is a web-based application that implements various numerical methods for solving mathematical problems. It is built using the Django framework and offers an interactive interface for users to input data, perform calculations, and visualize results.

## Features

- **Bisection Method**: A root-finding method for continuous functions.
- **Newton's Method**: A powerful method for finding roots of a real-valued function.
- **Gauss-Seidel Method**: An iterative technique for solving systems of linear equations.
- **Secant Method**: A numerical method for finding roots of a function.
- **Lagrange Interpolation**: A method of constructing a polynomial that passes through a given set of points.
- **Cubic and Linear Splines**: Techniques for interpolation of a function using piecewise polynomials.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/itssam0/Numerical-Analysis.git
   cd Numerical-Analysis
   
2. **Set up the virtual environment**:
    ```bash
    python3 -m venv env
    source env/bin/activate
    
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Run migrations**:
    ```bash
    python manage.py migrate

5. **Run the development server**:
    ```bash
    python manage.py runserver

## Usage

1. Open your web browser and navigate to http://127.0.0.1:8000/.
2. Select the numerical method you want to use from the available options.
3. Input the necessary parameters and click on the "Calculate" button.
4. The results will be displayed on the screen, with options to visualize the computation if applicable.

## Project Structure

- 'calculator/': Contains the Django application with views, URLs, and templates for different numerical methods.
- 'templates/': HTML files for rendering the web pages.
- 'static/': Static files like CSS and JavaScript.
- 'proyecto/': Project configuration files including settings, URLs, and WSGI.

## Contributing

Contributions are welcome! If you would like to add new features, improve existing functionality, or fix bugs, please feel free to submit a pull request.
