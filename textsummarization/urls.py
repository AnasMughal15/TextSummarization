from django.urls import path
from app.views import summarize_text

urlpatterns = [
    path('api/summarize_text/', summarize_text, name='TextSummarizer'),
]
