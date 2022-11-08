# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:07:07 2022

@author: Srushti
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from scipy.fftpack import ifft
from scipy import fft
import cv2
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
from itertools import chain 
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")
# Reading images in grayscale
oilydata = cv2.imread("oily(6).jpeg",cv2.IMREAD_GRAYSCALE)
drydata = cv2.imread("dry(35).jpg",cv2.IMREAD_GRAYSCALE)
fig , axs = plt.subplots(1,2,figsize=(10,10))
axs[0].imshow(drydata,cmap='gray')
axs[0].set_title('dry')
axs[1].imshow(oilydata,cmap='gray')
axs[1].set_title('oily')
plt.show()

# To transform images into signals we reduce image sizes 
# and used a flatten fucntion to convert them from 2-D into 1-D array

resize_dimg = cv2.resize(drydata,(50,50))
resize_oimg = cv2.resize(oilydata,(50,50))
dflatten_list = list(chain.from_iterable(resize_dimg)) 
oflatten_list = list(chain.from_iterable(resize_oimg)) 
fig,axs = plt.subplots(2,1,figsize=(12,4),sharex=True)
axs[0].plot(dflatten_list)
axs[0].set_title('DRY')
axs[1].plot(oflatten_list)
axs[1].set_title('OILY')
plt.show()

# comparing probability distributions
fig,axs = plt.subplots(1,2,figsize=(12,4),sharex=True)
sns.distplot(dflatten_list,ax=axs[0],color='Red').set_title('DRY')
sns.distplot(oflatten_list,ax=axs[1],color='Green').set_title('OILY')
plt.show()

# We are looking for small hiden informations on these 
# images so the ideal type of filter is LOWPASS FILTER.
# PSD: Power Spectral Density
dfreqs, dpsd = signal.welch(dflatten_list)
ofreqs, opsd = signal.welch(oflatten_list)
fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)
axs[0].semilogx(dfreqs, dpsd,color ='r')
axs[1].semilogx(ofreqs, opsd,color ='g')
axs[0].set_title('PSD: DRY')
axs[1].set_title('PSD: OILY')
axs[0].set_xlabel('Frequency')
axs[1].set_xlabel('Frequency')
axs[0].set_ylabel('Power')
axs[1].set_ylabel('Power')
plt.show()
sos = signal.iirfilter(3, Wn=0.01, rs=0.06 ,fs=100,btype='lp',output='sos',
                       analog=False, ftype='cheby2')
w, h = signal.sosfreqz(sos, worN=100)

# Freq response
plt.subplot(2, 1, 1)
db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
plt.plot(w, db)
plt.ylim(-75, 5)
plt.grid(True)
plt.yticks([0, -20, -40, -60])
plt.ylabel('Gain [dB]')
plt.title('Frequency Response')
plt.subplot(2, 1, 2)
plt.plot(w/np.pi, np.angle(h))
plt.grid(True)
plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
           [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.ylabel('Phase [rad]')
plt.xlabel('Normalized frequency (1.0 = Nyquist)')
plt.show()

# Step response
t, s = signal.step(sos)
fig,axs = plt.subplots(1,1,figsize=(7,3),sharey=True)
axs.semilogx(t, s,color ='g')
axs.set_title('PSD')
axs.set_xlabel('Frequency')
axs.set_ylabel('Power')
plt.show()

# Applying the filter to Signals
fig, axs = plt.subplots(2, 2,figsize=(12,4), sharey=True,sharex=True)
dfiltered = signal.sosfilt(sos, dflatten_list)
ofiltered = signal.sosfilt(sos, oflatten_list)
axs[0,0].plot(dflatten_list)
axs[0,0].set_title('DRY')
axs[0,1].plot(dfiltered)
axs[0,1].set_title('DRY After 0.01 Hz low-pass filter')
axs[1,0].plot(oflatten_list)
axs[1,0].set_title('OILY')
axs[1,1].plot(ofiltered)
axs[1,1].set_title('OILY After 0.01 Hz low-pass filter')
plt.show()

# Power Spectral Density of filtered Signals
dfreqs, dpsd = signal.welch(dfiltered)
ofreqs, opsd = signal.welch(ofiltered)
fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)
axs[0].semilogx(dfreqs, dpsd,color ='r')
axs[1].semilogx(ofreqs, opsd,color ='g')
axs[0].set_title('PSD: DRY FILTERED')
axs[1].set_title('PSD: OILY FILTERED')
axs[0].set_xlabel('Frequency')
axs[1].set_xlabel('Frequency')
axs[0].set_ylabel('Power')
axs[1].set_ylabel('Power')
plt.show()

# Periodogram of filtered Signsls
fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)
pf, Pxx_den = signal.periodogram(dfiltered, 50)
f, nxx_den = signal.periodogram(ofiltered, 50)
axs[0].semilogy(pf, Pxx_den)
axs[1].semilogy(f, nxx_den)
axs[0].set_title('DRY')
axs[1].set_title('OILY')
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
plt.show()

# Helper functions to find autocorrelation,
# Root Mean Square and max values of an array

def autocorrelation(x):
    xp = np.fft.fftshift((x - np.average(x))/np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = fft.fft(xp)
    p = np.absolute(f)**2
    pi = ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

def kLargest(arr, k): 
    max_ks = []
    # Sort the given array arr in reverse  
    # order. 
    arr = np.sort(arr)[::-1]
    #arr.sort(reverse = True) 
    # Print the first kth largest elements 
    for i in range(k):
        max_ks.append(arr[i])
        
    return max_ks

def rms(x):
    return np.sqrt(np.mean(x**2))

# Autocorrelation
print(f'Tree max values of autocorrelation\noily FILTERED:   {kLargest(autocorrelation(ofiltered), 3)}\n\ndry FILTERED:{kLargest(autocorrelation(dfiltered), 3)}')

# Finding Peaks
ppeaks, _  = signal.find_peaks(dpsd)
npeaks, _  = signal.find_peaks(opsd)
print(f'Ten max peaksof filtered PSD\ndry:   {kLargest(ppeaks,10)}\noily peaks:{kLargest(npeaks,10)}')

# Root Mean Square and Mean of filtered signals
print(f'Root Mean Square\n--------------------\noily : {rms(ofiltered)}\ndry : {rms(dfiltered)}')
print('--'*10)
print(f'Mean \n--------------------\noily : {np.mean(dflatten_list)}\ndry : {np.mean(ofiltered)}')

# Inverse Fast Fourier Transform
fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)
offt = fft.fft(oflatten_list)
dfft = fft.fft(dflatten_list)
axs[0].plot(dfft.real, 'b-')
axs[0].plot(dfft.imag, 'r--')
axs[0].set_title('DRY')
axs[1].plot(offt.real, 'b-')
axs[1].plot(offt.imag, 'r--')
axs[1].set_title('OILY')
plt.legend(('real', 'imaginary'))
plt.show()

# Magnitude Spectrum for oily
dft = cv2.dft(np.float32(oilydata),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.figure(figsize=(11,6))
plt.subplot(121),plt.imshow(oilydata, cmap = 'gray')
plt.title('ORIGINAL'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# Magnitude Spectrum for dry
dft = cv2.dft(np.float32(drydata),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.figure(figsize=(11,6))
plt.subplot(121),plt.imshow(drydata, cmap = 'gray')
plt.title('ORIGINAL'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# Apply Signal Processing to all images
mpath = "dataset/"
IM_SIZE = 90
def get_img(folder):
    X = []
    y = []
    for xr in os.listdir(folder):
        if not xr.startswith('.'):
            if xr in ['oily']:
                label = 0
            elif xr in ['dry']:
                label = 1
            for filename in tqdm(os.listdir(folder + xr)):
                im_array = cv2.imread(folder + xr +'/'+ filename,cv2.IMREAD_GRAYSCALE)
                if im_array is not None:
                    img = cv2.resize(im_array,(IM_SIZE,IM_SIZE))
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
                    maxPeaks  = kLargest(peaks, k=6)
                    #mean and rms
                    Mean = np.mean(flatten_list)
                    Rms = rms(filtered)
                    # autocorrelation
                    auto= autocorrelation(filtered)
                    maxauto = kLargest(auto, k=5)
                    #fft
                    invfft = fft.fft(filtered)
                    vfl = np.std(flatten_list)
                    invfft_r_peaks, _  = signal.find_peaks(invfft.real)
                    invfft_imag_peaks, _  = signal.find_peaks(invfft.imag)
                    maxinvfft_r_peaks  = kLargest(invfft_r_peaks, k=6)
                    maxinvfft_imag_peaks  = kLargest(invfft_imag_peaks, k=6)
                    #peaks of periodogram filtered
                    _, Pxx_den = signal.periodogram(filtered,100)
                    Perio_Peaks, _  = signal.find_peaks(Pxx_den)
                    
                    maxPerio_Peaks  = kLargest(Perio_Peaks, k=6)
                    total = maxPeaks + [Rms,Mean,lpvar,msd,vfl] 
                    total = total + maxPerio_Peaks
                    total = total + maxinvfft_r_peaks
                    total = total + maxinvfft_imag_peaks
                    total = total + maxPeaks
                    total = total + maxauto
                    X.append(total)
                    y.append(label)    
    y = np.array(y)
    return np.array(X),y
X,y = get_img(mpath)

X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size = 0.2)
#Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)    
X_test = scaler.transform(X_test)
mmscaler = MinMaxScaler()
X_train_01 = mmscaler.fit_transform(X_train)    
X_test_01 = mmscaler.transform(X_test)

# Training the model
import tensorflow as tf  
model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  
model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax)) 

model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])  
model.fit(X_train_01, y_train, epochs=200)  

val_loss, val_acc = model.evaluate(X_test_01, y_test) 
print(f'Validation loss: {val_loss}')  
print(f'Validation accuracy: {val_acc}') 

train_loss, train_acc = model.evaluate(X_train_01, y_train) 
print(f'Training loss: {train_loss}')  
print(f'Training accuracy: {train_acc}') 

def get_img2(folder):
    X=[]
    filename=os.listdir(folder)
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
    maxPeaks  = kLargest(peaks, k=6)
    #mean and rms
    Mean = np.mean(flatten_list)
    Rms = rms(filtered)
    # autocorrelation
    auto= autocorrelation(filtered)
    maxauto = kLargest(auto, k=5)
    #fft
    invfft = fft.fft(filtered)
    vfl = np.std(flatten_list)
    invfft_r_peaks, _  = signal.find_peaks(invfft.real)
    invfft_imag_peaks, _  = signal.find_peaks(invfft.imag)
    maxinvfft_r_peaks  = kLargest(invfft_r_peaks, k=6)
    maxinvfft_imag_peaks  = kLargest(invfft_imag_peaks, k=6)
    #peaks of periodogram filtered
    _, Pxx_den = signal.periodogram(filtered,100)
    Perio_Peaks, _  = signal.find_peaks(Pxx_den)
    
    maxPerio_Peaks  = kLargest(Perio_Peaks, k=6)
    total = maxPeaks + [Rms,Mean,lpvar,msd,vfl] 
    total = total + maxPerio_Peaks
    total = total + maxinvfft_r_peaks
    total = total + maxinvfft_imag_peaks
    total = total + maxPeaks
    total = total + maxauto
    X.append(total)
    return np.array(X)

#Make the prediction
npath = "sample"
uX= get_img2(npath)
# print(uX)
uX = scaler.fit_transform(uX)       
X_test_01 = mmscaler.transform(uX)
y_pred = model.predict(X_test_01)
print(y_pred)
if y_pred[0][0]>=y_pred[0][1]:
    print("Dry")
else:
    print("Oily")