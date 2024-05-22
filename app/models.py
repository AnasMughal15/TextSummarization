from django.db import models

class UploadedFile(models.Model):
    pdf_file = models.FileField(upload_to='uploads/')
