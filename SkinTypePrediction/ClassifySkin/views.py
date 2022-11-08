from django.http import HttpResponse
from django.shortcuts import render, redirect
from . import forms
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from scipy.fftpack import ifft
from scipy import fft
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
from itertools import chain
from sklearn.model_selection import train_test_split
from . import trainmodel

# Create your views here.

def plots_view(request):
    return render(request, 'ClassifySkin/plots.html')

def index_view(request):
    if request.method == 'POST':
        form = forms.SkinForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return redirect('classification_view')
    else:
        form = forms.SkinForm()
    return render(request, 'ClassifySkin/index.html', {'form' : form})

def classification_view(request):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    IM_SIZE = 90
    from tensorflow import keras
    model = keras.models.load_model('static/model')

    def get_img2(folder):
        X=[]
        filename=os.listdir(folder)
        print(filename)
        img=cv2.imread(folder + '/' + filename[0],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(IM_SIZE,IM_SIZE))
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        msd = np.std(magnitude_spectrum)
        new_image = cv2.Laplacian(img,cv2.CV_64F)
        lpvar = abs(np.max(new_image) - np.min(new_image))/np.max(new_image)
        #flatten the image
        flatten_list = list(chain.from_iterable(img))
        #filtering
        sos = signal.iirfilter(3, Wn=0.01, rs=0.5 ,fs=100,btype='lp',output='sos',
           analog=False, ftype='cheby2')
        filtered = signal.sosfilt(sos, flatten_list)
        #power Spectral density
        _, psd = signal.welch(filtered)
        #find peaks of PSD
        peaks, _  = signal.find_peaks(psd)
        maxPeaks  = trainmodel.kLargest(peaks, k=6)
        #mean and trainmodel.rms
        Mean = np.mean(flatten_list)
        Rms = trainmodel.rms(filtered)
        # trainmodel.autocorrelation
        auto= trainmodel.autocorrelation(filtered)
        maxauto = trainmodel.kLargest(auto, k=5)
        #fft
        invfft = fft.fft(filtered)
        vfl = np.std(flatten_list)
        invfft_r_peaks, _  = signal.find_peaks(invfft.real)
        invfft_imag_peaks, _  = signal.find_peaks(invfft.imag)
        maxinvfft_r_peaks  = trainmodel.kLargest(invfft_r_peaks, k=6)
        maxinvfft_imag_peaks  = trainmodel.kLargest(invfft_imag_peaks, k=6)
        #peaks of periodogram filtered
        _, Pxx_den = signal.periodogram(filtered,100)
        Perio_Peaks, _  = signal.find_peaks(Pxx_den)

        maxPerio_Peaks  = trainmodel.kLargest(Perio_Peaks, k=6)
        total = maxPeaks + [trainmodel.rms,Mean,lpvar,msd,vfl]
        total = total + maxPerio_Peaks
        total = total + maxinvfft_r_peaks
        total = total + maxinvfft_imag_peaks
        total = total + maxPeaks
        total = total + maxauto
        X.append(total)
        return np.array(X)

    #Make the prediction
    npath = "media/images"
    uX= get_img2(npath)
    print("ux1",uX)
    # mpath = "static/images/dataset/"
    # IM_SIZE = 90
    # X,y = get_img(mpath)
    # X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size = 0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(trainmodel.X_train)
    X_test = scaler.transform(trainmodel.X_test)

    mmscaler = MinMaxScaler()
    X_train_01 = mmscaler.fit_transform(trainmodel.X_train)
    X_test_01 = mmscaler.transform(trainmodel.X_test)

    uX = scaler.fit_transform(uX)
    X_test_01 = mmscaler.transform(uX)
    y_pred = model.predict(X_test_01)
    print(y_pred)
    if y_pred[0][0]>=y_pred[0][1]:
        result="Dry"
    else:
        result="Oily"
    filename=os.listdir(npath)
    for i in range(0,len(filename)):
        os.remove(npath + '/' +filename[i])
    return render(request, 'ClassifySkin/success.html', {'result' : result})
