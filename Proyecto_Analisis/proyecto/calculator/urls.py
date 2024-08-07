# calculator/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.metodos, name='metodos'),
    path('base/', views.base, name='base'),
    path('biseccion/', views.biseccion, name='biseccion'),
    path('newton/', views.newton, name='newton'),
    path('pf/', views.pf, name='pf'),
    path('rf/', views.rf, name='rf'),
    path('rm/', views.rm, name='rm'),
    path('secante/', views.secante, name='secante'), 
    path('gauss_seidel/', views.gauss_seidel, name='gauss_seidel'),
    path('sor/', views.sor, name='sor'),
    path('lagrange/', views.lagrange, name='lagrange'),
    path('newtonint/', views.newtonint, name='newtonint'),
    path('spline_lineal/', views.spline_lineal, name='spline_lineal'),
    path('spline_cubico/', views.spline_cubico, name='spline_cubico'), 
    path('vandermonde/', views.vandermonde, name='vandermonde'),             
]
