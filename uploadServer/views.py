from django.shortcuts import render
from django.http import JsonResponse
from .controller import request_handler


def index(request):
    return JsonResponse({"message": "SUCCESS"})


def upload(request):
    if request.method == 'POST':
        request_handler.handle_uploaded_file(request.FILES['file'])
        request_handler.img_process_opencv()
        return JsonResponse({"message": "SUCCESS", 'status_code': 200})
    else:
        return JsonResponse({"message": "FAILURE"})
