import numpy as np
from scipy.fftpack import fft2, ifft2,fftshift,ifftshift
import cv2
import matplotlib.pyplot as plt



def FourierPtychographicMicroscopy(target,height,NA,wavelen,H,times,task):
    max_freq = int(NA / wavelen * 2 * np.pi)
    times = int(times)
    task = int(task)
    height = int(height)
    spectrum = np.zeros((int(height), int(height)), dtype=complex)
    target = np.array(target, dtype=complex)
    spectrumiif = fftshift(fft2(target))
    # 循环不同LED位置与角度
    for i in range(times):

        print(i)

        # 检查程序运行状况
        for j in range(times):

            #print(i,j)
            # 不同角度的LED
            thetai = (( height/2-max_freq)*(i*2/times)*(i<=times/2)-( height/2-max_freq)*((i-times/2)*2/times)*(i>times/2))
            thetaj = (( height/2-max_freq)*(j*2/times)*(j<=times/2)-( height/2-max_freq)*((j-times/2)*2/times)*(j>times/2))



            b = 2*np.pi/wavelen*H
            x = np.linspace(-thetai * b, thetai * b, height, dtype=complex)
            y = np.linspace(-thetaj * b, thetaj * b, height, dtype=complex)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-1j * (X+Y))
            psf = np.array(target, dtype=complex)*Z
            #拼接
            spectrumii = fftshift(fft2(psf))
            spectrumi = np.zeros((int(height), int(height)), dtype=complex)

            # 频谱平移拼接
            x_pose_n = int((height/2-max_freq+( height/2-max_freq)*i*2/times)*(i<times/2)+(height/2-max_freq-( height/2-max_freq)*(i-times/2)*2/times)*(i>times/2))
            y_pose_n = int((height/2-max_freq+( height/2-max_freq)*j*2/times)*(j<times/2)+(height/2-max_freq-( height/2-max_freq)*(j-times/2)*2/times)*(j>times/2))
            spectrumi[int(x_pose_n):int(x_pose_n+2*max_freq) , int(y_pose_n):int(y_pose_n+2*max_freq)] = spectrumii[int(height/2-max_freq):int(height/2+max_freq) , int(height/2-max_freq):int(height/2+max_freq)]

            spectrum += spectrumi
    # 将最终的频谱进行逆傅里叶变换得到频谱数据
    rec = np.abs(ifft2(ifftshift(spectrum)))
    rec = rebuild(rec**2)

    if task == 1:
        return rec,spectrum
    else:
        I = np.sum(np.sum(rec))-np.sum(np.sum(target))
        var = (np.max(rec)-np.min(rec))/(np.max(rec)+np.min(rec))
        err = np.sum(np.sum((rec - target ** 2) ** 2) )/ height / height
        return err,var,I


def rebuild(norm_int):
    maxi = np.max(norm_int)
    mini = np.min(norm_int)
    rec = ((norm_int - mini) / (maxi - mini))
    rec = norm_int * 255
    rec = rec.astype("uint8")
    return rec
def standardize(origin):
    maxi = np.max(origin)
    mini = np.min(origin)
    norm = ((origin - mini) / (maxi - mini))
    return norm
def save_recon(rec, rec_name):
    cv2.imwrite("img_FPM/{}.bmp".format(rec_name), rec)
    cv2.imshow("Reconstruction", rec)
    cv2.waitKey(0)


# 初始化参数
targetr = cv2.imread("target.png")
height, width = targetr.shape[:2]
target_name = "target"
maxi = np.max(targetr)
mini = np.min(targetr)
target = ((targetr - mini) / (maxi - mini))
target = standardize((standardize(target) + 0.1*np.random.random((512,512)) ))
cv2.imwrite("img_FPM/{}.bmp".format(target_name), rebuild(target))
targetf = target**(1/2)

NA0 = NA = 0.1
wavelenth0 = 632e-9
H = 86E-3

recon,spectrum = FourierPtychographicMicroscopy(targetf,targetr,height,NA0,wavelenth0,122,1)
save_recon(recon,"rebuildion")


max_freq = int(NA0 / wavelenth0 * 2 * np.pi)
timesmax = int(height / 2 / max_freq)
print(timesmax)

#设定循环次数
iter = np.linspace(1,timesmax,int(timesmax/2),dtype=int)
err = []
var = []
I = []
err = np.array(err)
var = np.array(var)
I = np.array(I)

#循环成像
for i in iter:
erri ,vari,Ii = FourierPtychographicMicroscopy(targetf,targetr,height,NA0,wavelenth0,i,2)
print(i)
err = np.append(err,erri)
var = np.append(var ,vari)
I = np.append(I,Ii)

err = np.array(err)
var = np.array(var)
I = np.array(I)


plt.plot(iter,err)
plt.title("error")
plt.show()

plt.plot(iter,var)
plt.title("contrast")
plt.show()

plt.plot(iter,I)
plt.title("mean iuminance")
plt.show()


#绘图

    #设定循环次数
iter = np.linspace(100120,300,50,dtype=int)
err = []
var = []
I = []
err = np.array(err)
tar = np.array(var)
I = np.array(I)

#循环成像
for i in iter:
    NA = i*10**(-9)
    max_freq = int(NA / wavelenth0 * 2 * np.pi)
    timesmax = int(height / 2 / max_freq)
    print(timesmax)
    erri ,vari,Ii = FourierPtychographicMicroscopy(targetf,targetr,height,NA,wavelenth0,timesmax,2)
    print(i)
    err = np.append(err,erri)
    var = np.append(var ,vari)
    I = np.append(I,Ii)

err = np.array(err)
var = np.array(var)
I = np.array(I)

plt.plot(iter/512,err)
plt.title("error")
plt.show()

plt.plot(iter/512,var)
plt.title("contrast")
plt.show()

plt.plot(iter/512,I)
plt.title("mean iuminance")
plt.show()


