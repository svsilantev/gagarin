import copy
from pathlib import Path

import django_filters.rest_framework
from rest_framework import status
from rest_framework import generics
from rest_framework import permissions
from rest_framework.response import Response

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np

from . import serializers
from . import models


def predict(model, test_loader):
    RUN_DEVICE = "cpu"
    with torch.no_grad():
        logits = []
        filenames = []
        for images, img_names in test_loader:
            images = images.to(RUN_DEVICE)
            model.eval()
            outputs = model(images).cpu()
            logits.append(outputs)
            filenames += img_names.cpu()
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    predicted_classes = np.argmax(probs, axis=1)
    # predicted_classes = [class_names[idx] for idx in class_indices]
    filenames = np.array([int(tens.numpy()) for tens in filenames])
    # filenames = [class_names[idx] for idx in filenames]

    return probs, predicted_classes, filenames


def get_net(pth_file_path=None, freezing=False):
    RUN_DEVICE = "cpu"
    AMOUNT_OF_CLASSES = 4
    resnet = torchvision.models.resnet152(pretrained=True)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, AMOUNT_OF_CLASSES)
    if pth_file_path is not None:
        resnet.load_state_dict(torch.load(pth_file_path, map_location=RUN_DEVICE))

    if freezing:
        counter = 0
        for child in resnet.children():
            if counter < 18:  # заморозка первых 18 слоев
                for param in child.parameters():
                    param.requires_grad = False
                    counter += 1
            # print(iresnet_finetuned)

    resnet = resnet.to(RUN_DEVICE)
    return resnet


class BipDataset(Dataset):  # класс датасета
    def __init__(self, files, mode=None):
        super().__init__()
        # список файлов для загрузки
        # режим работы
        # self.mode = mode

        # if self.mode not in DATA_MODES:
        #     print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
        #     raise NameError

        # self._amount_of_augmentations = AMOUNT_OF_AUGMENTATIONS
        self.files = self._get_files(files)  # sorted(files)
        self.labels = self._get_labels()
        self.len_ = len(self.files)
        # self.augmentations = augmentations

    def _get_files(self, files):
        raw_files = sorted(files)
        # if self.mode != 'test':
        #     files = [file for file in raw_files for _ in range(self._amount_of_augmentations)]
        # else:
        #     files = raw_files
        files = raw_files
        return files

    def _get_labels(self):
        raw_labels = [path.parent.name for path in self.files]  # получаем имя директории. Имя директории = имя файла
        labels_numbers = [
            0 if string == "Drivers" else 1 if string == "Passports" else 2 if string == "PTS" else 3 if string == "STS" else -1
            for string in raw_labels]

        # СЕРЕЖАААА! Нужно вот это переделать. Преобразуем классы в инты. Если паспорт, то 0, если тс, то 1 и тд. Реализуй это пж
        # if self.mode != 'test':
        #     labels = [label for label in labels_numbers for _ in range(self._amount_of_augmentations)]
        # else:
        #     labels = labels_numbers
        labels = labels_numbers
        return labels

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        file_copied = copy.deepcopy(file)
        image = Image.open(file_copied).convert('RGB')
        image.load()
        return image

    def __getitem__(self, index):
        TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        RESCALE_SIZE = (224, 224)
        image = self.load_sample(self.files[index])
        image = image.resize(RESCALE_SIZE)
        image = np.array(image)

        image = np.array(image / 255, dtype='float32')
        image = TRANSFORM(image)
        # if self.mode != 'test' and not(self.augmentations is None):
        #     image = self.augmentations(image)
        label = self.labels[index]
        sample = image, label

        return sample


resnet = get_net(pth_file_path='./api/models/Resnet152_ep_6_from_10.pth', freezing=False)


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
        class_names = ["Drivers", "Passports", "PTS", "STS"]
        serializer = serializers.CreateDownloadModelSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        serializer.save(user=self.request.user)
        print(Path('./'+str(serializer.data["photo"])))
        dataset = BipDataset([Path('./'+str(serializer.data["photo"])), ])
        data = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        classes_probs, predicted_classes, filenames = predict(resnet, data)
        models.DocumentsTypeModel.objects.get_or_create(name=class_names[predicted_classes[0]])
        headers = self.get_success_headers(serializer.data)

        return Response({
            'type': class_names[predicted_classes[0]],
            'confidence': max(classes_probs[0]),
        }, status=status.HTTP_200_OK, headers=headers)


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
