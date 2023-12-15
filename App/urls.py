"""Final URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
# from django.urls import path

# urlpatterns = [
#     path("admin/", admin.site.urls),
# ]
from django.contrib import admin
from django.urls import include, path
from similarity_app import views
from similarity_app.views import document_similarity
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('similarity_app.urls')),
    path('document_similarity/', views.document_similarity,
         name='document_similarity'),
    path('doc.html', views.doc_view, name='doc'),
    path('upload.html', include('similarity_app.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
