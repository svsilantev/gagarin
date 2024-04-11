import django_filters.rest_framework
from rest_framework import status
from rest_framework import generics
from rest_framework import permissions
from rest_framework.response import Response

from . import serializers
from . import models


class DownloadModelListView(generics.ListCreateAPIView):
    model = models.DownloadModel
    queryset = models.DownloadModel.objects.all()
    serializer_class = serializers.DownloadModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'user__id': ["in", "exact", "icontains"],
        'status': ["in", "exact", "icontains"],
    }
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request, *args, **kwargs):
        serializer = serializers.CreateDownloadModelSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        serializer.save(user=self.request.user)

        headers = self.get_success_headers(serializer.data)

        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class ParamTypeModelListView(generics.ListAPIView):
    queryset = models.ParamTypeModel.objects.all()
    serializer_class = serializers.ParamTypeModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'name': ["in", "exact", "icontains"]
    }
    permission_classes = [permissions.IsAuthenticated]


class DocumentsTypeModelListView(generics.ListAPIView):
    queryset = models.DocumentsTypeModel.objects.all()
    serializer_class = serializers.DocumentsTypeModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'name': ["in", "exact", "icontains"]
    }
    permission_classes = [permissions.IsAuthenticated]


class ParamTypeDocumentsTypeModelListView(generics.ListAPIView):
    queryset = models.ParamTypeDocumentsTypeModel.objects.all()
    serializer_class = serializers.ParamTypeDocumentsTypeModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'param_type__id': ["in", "exact", "icontains"],
        'document_type__id': ["in", "exact", "icontains"],
    }
    permission_classes = [permissions.IsAuthenticated]


class ResultModelListView(generics.ListAPIView):
    queryset = models.ResultModel.objects.all()
    serializer_class = serializers.ResultModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'download__id': ["in", "exact", "icontains"],
        'document_type__id': ["in", "exact", "icontains"],
    }
    permission_classes = [permissions.IsAuthenticated]


class ParamValueModelListView(generics.ListAPIView):
    queryset = models.ParamValueModel.objects.all()
    serializer_class = serializers.ParamValueModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'param_type__id': ["in", "exact", "icontains"],
        'value': ["in", "exact", "icontains"],
        'result__id': ["in", "exact", "icontains"],
    }
    permission_classes = [permissions.IsAuthenticated]
