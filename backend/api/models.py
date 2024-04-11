from django.db import models
from django.utils.translation import gettext_lazy as _
from django.conf import settings


def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT / user_<id>/<filename>
    return 'user_{0}/{1}'.format(instance.user.id, filename)


class DownloadModel(models.Model):

    class Status(models.TextChoices):
        IN_PROGRESS = 'PROG', _('IN_PROGRESS')
        DONE = 'DONE', _('DONE')

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    photo = models.ImageField(upload_to=user_directory_path)
    created = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=4,
        choices=Status.choices,
        default=Status.IN_PROGRESS,
    )


class ParamTypeModel(models.Model):
    name = models.CharField(max_length=255, unique=True)


class DocumentsTypeModel(models.Model):
    name = models.CharField(max_length=255, unique=True)


class ParamTypeDocumentsTypeModel(models.Model):
    param_type = models.ForeignKey(ParamTypeModel, on_delete=models.CASCADE)
    document_type = models.ForeignKey(DocumentsTypeModel, on_delete=models.CASCADE)


class ResultModel(models.Model):
    download = models.ForeignKey(DownloadModel, on_delete=models.CASCADE)
    document_type = models.ForeignKey(DocumentsTypeModel, on_delete=models.CASCADE)


class ParamValueModel(models.Model):
    param_type = models.ForeignKey(ParamTypeModel, on_delete=models.CASCADE)
    value = models.CharField(max_length=255)
    result = models.ForeignKey(ResultModel, on_delete=models.CASCADE)
