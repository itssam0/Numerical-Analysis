# calculator/views.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import io
import math
import sympy as sp
import urllib, base64

def base(request):
    return render(request, 'base.html')

def metodos(request):
    return render(request, 'metodos.html')

# Biseccion             // Linea 35
# Newton                // Linea 139
# Punto Fijo            // Linea 233
# Regla Falsa           // Linea 320
# Raices Multiples      // Linea 413
# Secante               // Linea 506
# Gauss Seidel y Jacobi // Linea 599
# SOR                   // Linea 681
# Lagrange              // Linea 762
# Newton Interpolante   // Linea 816
# Spline Lineal         // Linea 883
# Spline Cubico         // Linea 948
# Vandermonde           // Linea 1039

# ---------------------------------------------------------------------------------------------------------------------------------
# CAPITULO 1
def biseccion(request):
    if request.method == 'POST':
        try:
            Xi = float(request.POST['xi'])
            Xs = float(request.POST['xs'])
            Tol = float(request.POST['tol'])
            Niter = int(request.POST['iteraciones'])
            Fun = request.POST['func'].replace('^', '**')  # Reemplaza ^ con **
            
            # Validar la función ingresada
            x = Xi
            eval(Fun)  # Prueba si la función es válida
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap1/biseccion.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que la función es válida."
            })

        fm = []
        E = []
        iterations = []
        x = Xi
        fi = eval(Fun)
        x = Xs
        fs = eval(Fun)

        if fi == 0:
            s = Xi
            E = 0
            resultado = f"{Xi} es raiz de f(x)"
        elif fs == 0:
            s = Xs
            E = 0
            resultado = f"{Xs} es raiz de f(x)"
        elif fs * fi < 0:
            c = 0
            Xm = (Xi + Xs) / 2
            x = Xm                 
            fe = eval(Fun)
            fm.append(fe)
            E.append(100)
            iterations.append((c, Xi, Xm, Xs, fe, 100))
            while E[c] > Tol and fe != 0 and c < Niter:
                if fi * fe < 0:
                    Xs = Xm
                    x = Xs                 
                    fs = eval(Fun)
                else:
                    Xi = Xm
                    x = Xi
                    fs = eval(Fun)
                Xa = Xm
                Xm = (Xi + Xs) / 2
                x = Xm 
                fe = eval(Fun)
                fm.append(fe)
                Error = abs(Xm - Xa)
                E.append(Error)
                c = c + 1
                iterations.append((c, Xi, Xm, Xs, fe, Error))
            if fe == 0:
                s = x
                resultado = f"{s} es raiz de f(x)"
            elif Error < Tol:
                s = x
                resultado = f"{s} es una aproximacion de una raiz de f(x) con una tolerancia {Tol}"
            else:
                s = x
                resultado = f"Fracaso en {Niter} iteraciones"
        else:
            resultado = "El intervalo es inadecuado"

        # Crear la gráfica
        plt.plot(fm, label='f(x)')
        plt.xlabel('Iteraciones')
        plt.ylabel('f(x)')
        plt.title('Convergencia del Método de Bisección')
        plt.legend()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        # Crear DataFrame
        df = pd.DataFrame(iterations, columns=['Iteration', 'a', 'xi', 'b', 'f_xi', 'Error'])

        # Codificar CSV en base64
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return render(request, 'Cap1/biseccion.html', {
            'resultado': resultado,
            'grafica': uri,
            'data': df.to_dict(orient='records'),
            'csv': csv_base64,
        })
        
    return render(request, 'Cap1/biseccion.html')

def newton(request):
    if request.method == 'POST':
        try:
            X0 = float(request.POST['x0'])
            Tol = float(request.POST['tolerancia'])
            Niter = int(request.POST['niter'])
            Fun = request.POST['funcion'].replace('^', '**')

            # Definir la variable simbólica y la función
            x = sp.symbols('x')
            func = sp.sympify(Fun)
            deriv_func = sp.diff(func, x)

            # Convertir la función y su derivada a funciones lambda
            func_lambda = sp.lambdify(x, func, modules=["numpy", "math"])
            deriv_lambda = sp.lambdify(x, deriv_func, modules=["numpy", "math"])

            # Validar la función ingresada
            func_lambda(X0)
            deriv_lambda(X0)
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap1/newton.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que la función es válida."
            })

        fn = []
        xn = []
        E = []
        N = []
        x_val = X0
        f = func_lambda(x_val)
        derivada = deriv_lambda(x_val)
        c = 0
        Error = 100
        fn.append(f)
        xn.append(x_val)
        E.append(Error)
        N.append(c)

        while Error > Tol and f != 0 and derivada != 0 and c < Niter:
            x_val = x_val - f / derivada
            derivada = deriv_lambda(x_val)
            f = func_lambda(x_val)
            fn.append(f)
            xn.append(x_val)
            c += 1
            Error = abs(xn[c] - xn[c-1])
            N.append(c)
            E.append(Error)

        if f == 0:
            resultado = f"{x_val} es raíz de f(x)"
        elif Error < Tol:
            resultado = f"{x_val} es una aproximación de una raíz de f(x) con una tolerancia {Tol}"
        else:
            resultado = f"Fracaso en {Niter} iteraciones"

        # Crear la gráfica
        plt.plot(xn, fn, label='f(x)')
        plt.xlabel('Iteraciones')
        plt.ylabel('f(x)')
        plt.title('Convergencia del Método de Newton')
        plt.legend()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        # Crear DataFrame
        df = pd.DataFrame({
            'Iteration': N,
            'x': xn,
            'f_x': fn,  
            'Error': E
        })

        # Codificar CSV en base64
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return render(request, 'Cap1/newton.html', {
            'resultado': resultado,
            'grafica': uri,
            'data': df.to_dict(orient='records'),
            'csv': csv_base64,
        })
    return render(request, 'Cap1/newton.html')

def pf(request):
    if request.method == 'POST':
        try:
            X0 = float(request.POST['x0'])
            Tol = float(request.POST['Tol'])
            Niter = int(request.POST['niter'])
            g = request.POST['funcg'].replace('^', '**')

            # Definir f(x) como g(x) - x
            Fun = f"{g} - x"

            # Validar la función g(x)
            x = X0
            eval(g)    # Prueba si la función g es válida
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap1/pf.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que las funciones son válidas."
            })

        fn = []
        xn = []
        E = []
        N = []
        x = X0
        f = eval(Fun)
        c = 0
        Error = 100
        fn.append(f)
        xn.append(x)
        E.append(Error)
        N.append(c)

        while Error > Tol and f != 0 and c < Niter:
            x = eval(g)
            fe = eval(Fun)
            fn.append(fe)
            xn.append(x)
            c += 1
            Error = abs(xn[c] - xn[c-1])
            N.append(c)
            E.append(Error)

        if fe == 0:
            resultado = f"{x} es raíz de f(x)"
        elif Error < Tol:
            resultado = f"{x} es una aproximación de una raíz de f(x) con una tolerancia de {Tol}"
        else:
            resultado = f"Fracaso en {Niter} iteraciones"

        # Crear la gráfica
        plt.plot(N, fn, label='f(x)')
        plt.xlabel('Iteraciones')
        plt.ylabel('f(x)')
        plt.title('Convergencia del Método de Punto Fijo')
        plt.legend()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        # Crear DataFrame
        df = pd.DataFrame({
            'Iteration': N,
            'xi': xn,
            'f_xi': fn,  # Cambiamos 'f(xi)' a 'f_xi' para ser compatible con Django
            'Error': E
        })

        # Codificar CSV en base64
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return render(request, 'Cap1/pf.html', {
            'resultado': resultado,
            'grafica': uri,
            'data': df.to_dict(orient='records'),
            'csv': csv_base64,
        })
        
    return render(request, 'Cap1/pf.html')

def rf(request):
    if request.method == 'POST':
        try:
            x0 = float(request.POST['x0'])
            x1 = float(request.POST['x1'])
            Tol = float(request.POST['Tol'])
            Niter = int(request.POST['niter'])
            Fun = request.POST['func'].replace('^', '**')

            # Validar la función ingresada
            x = x0
            eval(Fun)  # Prueba si la función es válida
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap1/regla_falsa.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que la función es válida."
            })

        iterations = []
        xi = x0
        xs = x1
        fxi = eval(Fun.replace('x', str(xi)))
        fxs = eval(Fun.replace('x', str(xs)))
        if fxi == 0:
            resultado = f"{xi} es raíz de f(x)"
        elif fxs == 0:
            resultado = f"{xs} es raíz de f(x)"
        elif fxi * fxs < 0:
            xm = xi - (fxi * (xs - xi) / (fxs - fxi))
            fxm = eval(Fun.replace('x', str(xm)))
            iterations.append((0, xi, xs, xm, fxi, fxs, fxm, None))
            c = 1
            error = Tol + 1

            while error > Tol and fxm != 0 and c < Niter:
                if fxi * fxm < 0:
                    xs = xm
                    fxs = fxm
                else:
                    xi = xm
                    fxi = fxm
                x_prev = xm
                xm = xi - (fxi * (xs - xi) / (fxs - fxi))
                fxm = eval(Fun.replace('x', str(xm)))
                error = abs(xm - x_prev)
                iterations.append((c, xi, xs, xm, fxi, fxs, fxm, error))
                c += 1

            if fxm == 0:
                resultado = f"{xm} es raíz de f(x)"
            elif error < Tol:
                resultado = f"{xm} es una aproximación de una raíz de f(x) con una tolerancia {Tol}"
            else:
                resultado = f"Fracaso en {Niter} iteraciones"
        else:
            resultado = "El intervalo es inadecuado"

        # Crear la gráfica
        x_vals = np.linspace(min(x0, x1), max(x0, x1), 400)
        y_vals = [eval(Fun.replace('x', str(x))) for x in x_vals]
        plt.plot(x_vals, y_vals, label='f(x)')
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Método de la Regla Falsa')
        plt.legend()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        # Crear DataFrame
        df = pd.DataFrame(iterations, columns=['Iteration', 'xi', 'xs', 'xm', 'fx_i', 'fx_s', 'fx_m', 'Error'])

        # Codificar CSV en base64
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return render(request, 'Cap1/rf.html', {
            'resultado': resultado,
            'grafica': uri,
            'data': df.to_dict(orient='records'),
            'csv': csv_base64,
        })
    return render(request, 'Cap1/rf.html')

def rm(request):
    if request.method == 'POST':
        try:
            x0 = float(request.POST['x0'])
            Tol = float(request.POST['tol'])
            Niter = int(request.POST['iteraciones'])
            Fun = request.POST['func'].replace('^', '**')

            # Derivadas de la función
            Deriv1 = f"{Fun}.diff(x)"
            Deriv2 = f"{Deriv1}.diff(x)"

            # Validar la función y sus derivadas
            x = x0
            eval(Fun)
            eval(Deriv1)
            eval(Deriv2)
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap1/rm.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que las funciones son válidas."
            })

        fn = []
        xn = []
        E = []
        N = []
        x = x0
        f = eval(Fun)
        f_prime = eval(Deriv1)
        f_double_prime = eval(Deriv2)
        c = 0
        Error = 100
        fn.append(f)
        xn.append(x)
        E.append(Error)
        N.append(c)

        while Error > Tol and f != 0 and f_prime != 0 and c < Niter:
            x = x - (f * f_prime) / (f_prime**2 - f * f_double_prime)
            f = eval(Fun)
            f_prime = eval(Deriv1)
            f_double_prime = eval(Deriv2)
            fn.append(f)
            xn.append(x)
            c += 1
            Error = abs(xn[c] - xn[c-1])
            N.append(c)
            E.append(Error)

        if f == 0:
            resultado = f"{x} es raíz de f(x)"
        elif Error < Tol:
            resultado = f"{x} es una aproximación de una raíz de f(x) con una tolerancia {Tol}"
        else:
            resultado = f"Fracaso en {Niter} iteraciones"

        # Crear la gráfica
        plt.plot(xn, fn, label='f(x)')
        plt.xlabel('Iteraciones')
        plt.ylabel('f(x)')
        plt.title('Convergencia del Método de Raíces Múltiples')
        plt.legend()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        # Crear DataFrame
        df = pd.DataFrame({
            'Iteration': N,
            'xi': xn,
            'f_xi': fn,
            'Error': E
        })

        # Codificar CSV en base64
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return render(request, 'Cap1/rm.html', {
            'resultado': resultado,
            'grafica': uri,
            'data': df.to_dict(orient='records'),
            'csv': csv_base64,
        })
    return render(request, 'Cap1/rm.html')

def secante(request):
    if request.method == 'POST':
        try:
            x0 = float(request.POST['x0'])
            x1 = float(request.POST['x1'])
            Tol = float(request.POST['Tol'])
            Niter = int(request.POST['niter'])
            Fun = request.POST['func'].replace('^', '**')

            # Validar la función ingresada
            x = x0
            eval(Fun)
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap1/secante.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que la función es válida."
            })

        fn = []
        xn = []
        E = []
        N = []
        x_ant = x0
        x_sig = x1
        f_ant = eval(Fun.replace('x', 'x_ant'))
        f_sig = eval(Fun.replace('x', 'x_sig'))
        c = 0
        Error = 100
        fn.append(f_sig)
        xn.append(x_sig)
        E.append(Error)
        N.append(c)

        while Error > Tol and f_sig != 0 and c < Niter:
            x = x_sig - f_sig * (x_sig - x_ant) / (f_sig - f_ant)
            f_ant = f_sig
            x_ant = x_sig
            x_sig = x
            f_sig = eval(Fun.replace('x', 'x_sig'))
            fn.append(f_sig)
            xn.append(x_sig)
            c += 1
            Error = abs(xn[c] - xn[c-1])
            N.append(c)
            E.append(Error)

        if f_sig == 0:
            resultado = f"{x} es raíz de f(x)"
        elif Error < Tol:
            resultado = f"{x} es una aproximación de una raíz de f(x) con una tolerancia {Tol}"
        else:
            resultado = f"Fracaso en {Niter} iteraciones"

        # Crear la gráfica
        plt.plot(xn, fn, label='f(x)')
        plt.xlabel('Iteraciones')
        plt.ylabel('f(x)')
        plt.title('Convergencia del Método de la Secante')
        plt.legend()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        # Crear DataFrame
        df = pd.DataFrame({
            'Iteration': N,
            'xn': xn,
            'f_xn': fn,
            'Error': E
        })

        # Codificar CSV en base64
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return render(request, 'Cap1/secante.html', {
            'resultado': resultado,
            'grafica': uri,
            'data': df.to_dict(orient='records'),
            'csv': csv_base64,
        })
    return render(request, 'Cap1/secante.html')

# ---------------------------------------------------------------------------------------------------------------------------------
# CAPITULO 2


def gauss_seidel(request):
    if request.method == 'POST':
        try:
            filas = int(request.POST.get('filas', 0))  # Use get() to avoid KeyError
            matrizA = np.array(eval(request.POST['matrizA']))
            vectorb = np.array(eval(request.POST['vectorb']))
            vectorx0 = np.array(eval(request.POST['vectorx0']))
            tolerancia = float(request.POST['tolerancia'])
            iteraciones = int(request.POST['niter'])
            metodo = int(request.POST['met'])

            if metodo not in [0, 1]:
                raise ValueError("Método inválido")
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap2/gauss-seidel.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que los valores son válidos."
            })

        D = np.diag(np.diag(matrizA))
        L = -np.tril(matrizA, -1)
        U = -np.triu(matrizA, 1)

        x0 = vectorx0
        c = 0
        error = tolerancia + 1
        E = [error]
        resultados = [{'Iteracion': c, 'x': x0.tolist(), 'Error': error}]
        
        while error > tolerancia and c < iteraciones:
            if metodo == 0:
                T = np.linalg.inv(D).dot(L + U)
                C = np.linalg.inv(D).dot(vectorb)
            elif metodo == 1:
                T = np.linalg.inv(D - L).dot(U)
                C = np.linalg.inv(D - L).dot(vectorb)
            
            x1 = T.dot(x0) + C
            error = np.linalg.norm((x1 - x0) / x1, np.inf)
            x0 = x1
            c += 1
            resultados.append({'Iteracion': c, 'x': x0.tolist(), 'Error': error})
            E.append(error)

        radio_espectral = max(abs(np.linalg.eigvals(T)))

        df = pd.DataFrame(resultados)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        if error < tolerancia:
            resultado = f"{x0} es una aproximación de la solución del sistema con una tolerancia {tolerancia}"
        else:
            resultado = f"Fracasó en {iteraciones} iteraciones"

        # Crear la gráfica
        iteraciones = df['Iteracion']
        errores = df['Error']
        plt.plot(iteraciones, errores, marker='o', label='Error')
        plt.xlabel('Iteraciones')
        plt.ylabel('Error')
        plt.title('Convergencia del Método de Gauss-Seidel/Jacobi')
        plt.legend()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        return render(request, 'Cap2/gauss-seidel.html', {
            'resultado': resultado,
            'grafica': uri,
            'radio': radio_espectral,
            'data': df.to_dict(orient='records'),
            'csv': csv_base64,
        })
    return render(request, 'Cap2/gauss-seidel.html')

def sor(request):
    if request.method == 'POST':
        try:
            filas = int(request.POST.get('filas', 0))  # Use get() to avoid KeyError
            matrizA = np.array(eval(request.POST['matrizA']))
            vectorb = np.array(eval(request.POST['matrizB']))
            vectorx0 = np.array(eval(request.POST['matrizX0']))
            tolerancia = float(request.POST['tol'])
            iteraciones = int(request.POST['niter'])
            w = float(request.POST['w'])
            tipo_error = int(request.POST.get('error', 0))  # Default to absolute error if not provided

            if tipo_error not in [0, 1]:
                raise ValueError("Tipo de error inválido")
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap2/sor.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que los valores son válidos."
            })

        D = np.diag(np.diag(matrizA))
        L = -np.tril(matrizA, -1)
        U = -np.triu(matrizA, 1)

        x0 = vectorx0
        c = 0
        error = tolerancia + 1
        E = [error]
        resultados = [{'Iteracion': c, 'x': x0.tolist(), 'Error': error}]

        while error > tolerancia and c < iteraciones:
            T = np.linalg.inv(D - w * L).dot((1 - w) * D + w * U)
            C = w * np.linalg.inv(D - w * L).dot(vectorb)
            x1 = T.dot(x0) + C
            if tipo_error == 1:
                error = np.linalg.norm((x1 - x0) / x1, np.inf)
            else:
                error = np.linalg.norm((x1 - x0), np.inf)
            x0 = x1
            c += 1
            resultados.append({'Iteracion': c, 'x': x0.tolist(), 'Error': error})
            E.append(error)

        radio_espectral = max(abs(np.linalg.eigvals(T)))

        df = pd.DataFrame(resultados)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        if error < tolerancia:
            resultado = f"{x0} es una aproximación de la solución del sistema con una tolerancia {tolerancia}"
        else:
            resultado = f"Fracasó en {iteraciones} iteraciones"

        # Crear la gráfica
        iteraciones = df['Iteracion']
        errores = df['Error']
        plt.plot(iteraciones, errores, marker='o', label='Error')
        plt.xlabel('Iteraciones')
        plt.ylabel('Error')
        plt.title('Convergencia del Método SOR')
        plt.legend()

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        return render(request, 'Cap2/sor.html', {
            'resultado': resultado,
            'grafica': uri,
            'radio': radio_espectral,
            'data': df.to_dict(orient='records'),
            'csv': csv_base64,
        })
    return render(request, 'Cap2/sor.html')

def lagrange(request):
    if request.method == 'POST':
        try:
            vectorx = np.array(eval(request.POST['vectorx']))
            vectory = np.array(eval(request.POST['vectory']))

            if len(vectorx) != len(vectory):
                raise ValueError("Los vectores x e y deben tener la misma longitud")
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap3/lagrange.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que los valores son válidos."
            })

        n = len(vectorx)
        pol = np.zeros(n)

        for i in range(n):
            Li = np.poly1d([1])
            den = 1
            for j in range(n):
                if i != j:
                    Li *= np.poly1d([1, -vectorx[j]])
                    den *= (vectorx[i] - vectorx[j])
            pol += vectory[i] * Li / den

        polinomio_string = str(np.poly1d(pol))

        # Crear un conjunto de puntos para graficar el polinomio
        x_vals = np.linspace(min(vectorx), max(vectorx), 1000)
        y_vals = np.polyval(pol, x_vals)

        # Graficar el polinomio resultante
        plt.plot(x_vals, y_vals, 'r', label='Polinomio de Lagrange')
        plt.plot(vectorx, vectory, 'bo', label='Puntos de entrada')
        plt.title('Polinomio de Lagrange')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        # Guardar la gráfica en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        return render(request, 'Cap3/lagrange.html', {
            'polinomio': polinomio_string,
            'grafica': uri,
        })
    return render(request, 'Cap3/lagrange.html')

def newtonint(request):
    if request.method == 'POST':
        try:
            vectorx = np.array(eval(request.POST['vectorx']))
            vectory = np.array(eval(request.POST['vectory']))

            if len(vectorx) != len(vectory):
                raise ValueError("Los vectores x e y deben tener la misma longitud")
        except (ValueError, SyntaxError) as e:
            return render(request, 'Cap3/newtonint.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que los valores son válidos."
            })

        n = len(vectorx)
        tabla = np.zeros((n, n + 1))
        tabla[:, 0] = vectorx
        tabla[:, 1] = vectory

        for j in range(2, n + 1):
            for i in range(j - 1, n):
                tabla[i, j] = (tabla[i, j - 1] - tabla[i - 1, j - 1]) / (tabla[i, 0] - tabla[i - j + 1, 0])

        coef = np.diag(tabla[:, 1:])
        pol = np.poly1d([coef[0]])
        acum = np.poly1d([1])

        for i in range(1, n):
            acum *= np.poly1d([1, -vectorx[i - 1]])
            pol += coef[i] * acum

        polinomio_string = str(pol)

        x_vals = np.linspace(min(vectorx), max(vectorx), 1000)
        y_vals = np.polyval(pol, x_vals)

        plt.plot(x_vals, y_vals, 'r', label='Polinomio de Newton')
        plt.plot(vectorx, vectory, 'bo', label='Puntos de entrada')
        plt.title('Polinomio de Newton')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        columnas = list(range(2, n + 1))

        df = pd.DataFrame(tabla, columns=['X', 'Y'] + [f'Diferencia Dividida {i}' for i in columnas])
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return render(request, 'Cap3/newtonint.html', {
            'polinomio': polinomio_string,
            'grafica': uri,
            'tabla': tabla.tolist(),
            'columnas': columnas,
            'csv': csv_base64,
        })
    return render(request, 'Cap3/newtonint.html')

def spline_lineal(request):
    if request.method == 'POST':
        try:
            x = list(map(float, request.POST['x'].split(',')))
            y = list(map(float, request.POST['y'].split(',')))

            if len(x) != len(y):
                raise ValueError("Los vectores x e y deben tener la misma longitud")
            if sorted(x) != x:
                raise ValueError("El vector x debe estar en orden ascendente")
        except ValueError as e:
            return render(request, 'Cap3/splinelineal.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que los valores son válidos y que el vector x está en orden ascendente."
            })

        n = len(x)
        A = np.zeros((2 * (n - 1), 2 * (n - 1)))
        b = np.zeros(2 * (n - 1))

        for i in range(n - 1):
            A[i, 2 * i] = x[i]
            A[i, 2 * i + 1] = 1
            b[i] = y[i]

            A[n - 1 + i, 2 * i] = x[i + 1]
            A[n - 1 + i, 2 * i + 1] = 1
            b[n - 1 + i] = y[i + 1]

        coef = np.linalg.solve(A, b)
        tabla = coef.reshape((n - 1, 2))

        x_vals = np.linspace(min(x), max(x), 1000)
        y_vals = np.piecewise(x_vals, [((x_vals >= x[i]) & (x_vals <= x[i + 1])) for i in range(n - 1)],
                              [lambda x, i=i: tabla[i, 0] * x + tabla[i, 1] for i in range(n - 1)])

        plt.plot(x_vals, y_vals, label='Spline Lineal')
        plt.plot(x, y, 'ro', label='Puntos de entrada')
        plt.title('Spline Lineal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        plt.close()

        df = pd.DataFrame(tabla, columns=['a', 'b'])
        df.index.name = 'Segmento'
        csv_buf = io.StringIO()
        df.to_csv(csv_buf)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        return render(request, 'Cap3/splinelineal.html', {
            'grafica': uri,
            'tabla': tabla.tolist(),
            'columnas': ['a', 'b'],
            'csv': csv_base64,
        })
    return render(request, 'Cap3/splinelineal.html')

def spline_cubico(request):
    if request.method == 'POST':
        try:
            x = list(map(float, request.POST['x'].split(',')))
            y = list(map(float, request.POST['y'].split(',')))
            d = 3  # Degree for cubic spline
        except ValueError:
            return render(request, 'Cap3/splinecubico.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que los valores son válidos."
            })

        # Ensure x and y have the same length
        if len(x) != len(y):
            return render(request, 'Cap3/splinecubico.html', {
                'error': "Los conjuntos de x y y deben tener la misma longitud."
            })

        # Spline Cúbico
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        h = np.diff(x)
        
        alpha = np.zeros(n)
        for i in range(1, n-1):
            alpha[i] = (3/h[i] * (y[i+1] - y[i])) - (3/h[i-1] * (y[i] - y[i-1]))
        
        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)

        for i in range(1, n-1):
            l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

        l[-1] = 1
        z[-1] = 0

        b = np.zeros(n-1)
        c = np.zeros(n)
        d = np.zeros(n-1)

        for j in range(n-2, -1, -1):
            c[j] = z[j] - mu[j] * c[j+1]
            b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
            d[j] = (c[j+1] - c[j]) / (3 * h[j])

        # Create the table for coefficients
        a = y[:-1]
        coef_table = np.vstack((a, b, c[:-1], d)).T

        # Create DataFrame for CSV
        df = pd.DataFrame(coef_table, columns=['a', 'b', 'c', 'd'])
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        # Plot the splines
        plt.figure()
        for i in range(n-1):
            xs = np.linspace(x[i], x[i+1], 100)
            ys = (a[i] + b[i] * (xs - x[i]) + c[i] * (xs - x[i])**2 + d[i] * (xs - x[i])**3)
            plt.plot(xs, ys, label=f'Segment {i}')
        
        plt.scatter(x, y, color='red')
        plt.title('Spline Cúbico')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        # Convert coef_table to list of lists for template rendering
        coef_table_list = coef_table.tolist()

        return render(request, 'Cap3/splinecubico.html', {
            'grafica': img_base64,
            'tabla': coef_table_list,
            'columnas': ['a', 'b', 'c', 'd'],
            'csv': csv_base64,
        })

    return render(request, 'Cap3/splinecubico.html')

def vandermonde(request):
    if request.method == 'POST':
        try:
            cantidad_puntos = int(request.POST['cantidadPuntos'])
            x = [float(request.POST[f'x{i}']) for i in range(cantidad_puntos)]
            y = [float(request.POST[f'y{i}']) for i in range(cantidad_puntos)]
        except ValueError:
            return render(request, 'Cap3/vandermonde.html', {
                'error': "Por favor ingrese todos los campos correctamente y asegúrese de que los valores son válidos."
            })

        # Ensure x and y have the same length
        if len(x) != len(y):
            return render(request, 'Cap3/vandermonde.html', {
                'error': "Los conjuntos de x y y deben tener la misma longitud."
            })

        # Método de Vandermonde
        x = np.array(x)
        y = np.array(y)
        V = np.vander(x)
        Polinomio = np.linalg.solve(V, y)
        Polinomio = Polinomio[::-1]  # Revertir el orden para usar con np.polyval

        # Generar la tabla de coeficientes
        coef_table = Polinomio
        df = pd.DataFrame([coef_table], columns=[f'a{i}' for i in range(len(coef_table)-1, -1, -1)])
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        csv_base64 = base64.b64encode(csv_buf.getvalue().encode()).decode()

        # Graficar la función
        plt.figure()
        x_plot = np.linspace(min(x), max(x), 100)
        y_plot = np.polyval(Polinomio, x_plot)
        plt.plot(x, y, 'ro', label='Datos')
        plt.plot(x_plot, y_plot, 'b-', label='Interpolación')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        return render(request, 'Cap3/vandermonde.html', {
            'grafica': img_base64,
            'tabla': coef_table,
            'columnas': [f'a{i}' for i in range(len(coef_table)-1, -1, -1)],
            'csv': csv_base64,
        })

    return render(request, 'Cap3/vandermonde.html')