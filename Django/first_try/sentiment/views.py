'''
Author: 21308004-Yao Yuan
Date: 2023-5-20
'''

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render,redirect

def login(request):
  if request.method == 'POST':
     username = request.POST.get('username')
     passowrd = request.POST.get('password')
     if username=='yaoyuan' and passowrd =='12345678':
        return redirect('/index')
     else:
        return render(request,'login.html',{"error":"用户名或密码错误"})

  return render(request,'login.html')

def index(request):
  #if request.method == 'POST':
    #return redirect('/pie')
  return render(request,'index.html')

def bar_y_sentiment(request):
  return render(request,'bar-y-sentiment.html')

def pie_sentiment(request):
  return render(request,'pie-sentiment.html')