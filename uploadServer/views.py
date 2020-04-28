from django.shortcuts import render
from django.http import JsonResponse
from io import BytesIO
from .controller import request_handler
from .controller import uploader


def index(request):
    return JsonResponse({"message": "SUCCESS"})


def upload(request):
    if request.method == 'POST':
        output, _hash = request_handler.img_process_opencv(
            request.FILES['file'].read())  # 进行图像处理
        uploader.upload(output, _hash)
        return JsonResponse({"message": "SUCCESS", 'status_code': 200, 'url': 'https://img-uploaded.oss-cn-shenzhen.aliyuncs.com/'+str(_hash)+'.jpg'})
    else:
        return JsonResponse({"message": "FAILURE", 'status_code': 400})
