from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload, name='upload'),
    path('status/', views.status, name='status'),
    path('chat/', views.chat, name='chat'),
    path('health/', views.health, name='health'),
]
