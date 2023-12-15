# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.document_similarity, name='document_similarity'),
# ]

from django.urls import path
from similarity_app import views
from similarity_app.views import document_similarity
from django.shortcuts import redirect
from .views import document_similarity, result, download_report, show_report
urlpatterns = [
    path("", views.document_similarity, name='document_similarity'),
    # path('', redirect('document_similarity'), name='root'),
    path('similarity/', views.document_similarity, name='document_similarity'),
    path('result/', views.result, name='result'),
    path('download_report', download_report, name='download_report'),
    path('show-report/', show_report, name='show_report'),
]
