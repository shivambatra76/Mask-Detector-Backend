from django.urls import path
from . import views

urlpatterns = [
    path("detect_mask", views.home, name="home")
]
