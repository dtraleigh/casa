from django.contrib.auth.decorators import login_required
from django.shortcuts import render

@login_required
def wemo_main(request):
    return render(request, 'wemo/wemo_main.html')
