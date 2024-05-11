import numpy as np
from scipy.fftpack import fft2, ifft2,fftshift,ifftshift
import cv2
import matplotlib.pyplot as plt

"""本程序用于实现傅里叶叠层成像，样本与最终成像的数组单位为m"""
"""本程序中，傅里叶叠层成像的实现，是通过先傅里叶变换，再直接频谱平移拼接，来等价于倾斜照明导致的频谱平移"""

def normalization(origin):

    maxi = np.max(origin)
    mini = np.min(origin)
    norm = ((origin - mini) / (maxi - mini))
    return norm

def reconstruct( norm_int ):
    norm_int = normalization(norm_int)
    rec = norm_int * 255
    rec = rec.astype("uint8")
    return rec

def get_target(photo_name):
    # 获取的是目标成像的振幅分布
    target = cv2.imread(photo_name,0)
    height, width = target.shape[:2]
    target_name = "target"

    targetr = normalization(target)
    target = normalization((normalization(target) + 0.1*np.random.random((512,512)) ))

    cv2.imwrite("img_FPM/{}.bmp".format(target_name), reconstruct(target))
    #cv2.imshow("target",target)
    #cv2.waitKey(0)

    targetf = target**(1/2)
    return targetr,targetf,height

def save_recon(rec,rec_name):
    cv2.imwrite("img_FPM/{}.bmp".format(rec_name), rec)
    cv2.imshow("Reconstruction", rec)
    cv2.waitKey(0)



def FPM1(targetr,targetf,height,NA,wavelen,times,task):
    """选择任务：
    task = 1：输出固定参数的最终成像
    task = 2, 输出此参数下的方差、对比度、平均亮度，用于循环不同的times、NA
    """
    # 处理参数
    max_freq = int(NA / wavelen * 2 * np.pi)
    times = int(times)
    task = int(task)
    height = int(height)


    # 初始化低数值孔径成像数组与最终频谱
    spectrum = np.zeros((int(height), int(height)), dtype=complex)
    targetf = np.array(targetf, dtype=complex)
    targetr = np.array(targetr, dtype=complex)
    spectrumiif = fftshift(fft2(targetf))
    # 循环不同LED位置与角度
    for i in range(times):

        # 检查程序运行状况
        for j in range(times):

            #print(i,j)
            # 不同角度的LED
            x_pose_n = int((height / 2 - max_freq + (height / 2 - max_freq) * i * 2 / times) * (i < times / 2) + (height / 2 - max_freq - (height / 2 - max_freq) * (i - times / 2) * 2 / times) * (i > times / 2))

            y_pose_n = int((height / 2 - max_freq + (height / 2 - max_freq) * j * 2 / times) * (j < times / 2) + (height / 2 - max_freq - (height / 2 - max_freq) * (j - times / 2) * 2 / times) * (j > times / 2))

            # 得到不同位置的低分辨率成像并进行拼接
            spectrumi = np.zeros((int(height), int(height)), dtype=complex)

            # 不同的LED角度的成像
            spectrumi[int(x_pose_n):int(x_pose_n+2*max_freq) , int(y_pose_n):int(y_pose_n+2*max_freq)] = spectrumiif[int(x_pose_n):int(x_pose_n+2*max_freq) , int(y_pose_n):int(y_pose_n+2*max_freq)]

            # 频谱平移拼接
            #spectrum = np.where( ( spectrumi )**2 > ( spectrum )**2, spectrumi , spectrum )
            spectrum = spectrumi+spectrum

    # 将最终的频谱进行逆傅里叶变换得到频谱数据
    rec = np.abs(ifft2(ifftshift(spectrum)))**2
    #rec = np.abs(ifft2(spectrum))


    if task == 1:
        rec = reconstruct(rec)
        return rec,spectrum
    else:
        I = (np.sum(np.sum(rec)) - np.sum(np.sum(targetr)))/ height / height
        var = (np.max(rec) - np.min(rec)) / (np.max(rec) + np.min(rec))
        err = np.sum(np.sum((rec - targetr ** 2) ** 2)) / height / height
        return err, var, I



def printerr_var_i_due2_times(targetf,targetr,height1,NA0,wavelenth0):
    max_freq = int(NA0 / wavelenth0 * 2 * np.pi)
    timesmax = int(height1 / 2 / max_freq)
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
        erri ,vari,Ii = FPM1(targetf,targetr,height1,NA0,wavelenth0,i,2)
        print(i)
        err = np.append(err,erri)
        var = np.append(var ,vari)
        I = np.append(I,Ii)

    err = np.array(err)
    var = np.array(var)
    I = np.array(I)
    """np.savetxt('FPMerr.txt',err)
    np.savetxt('FPMvar.txt',var)
    np.savetxt('FPMI.txt',I)"""

    plt.plot(iter,err)
    plt.title("error")
    plt.show()

    plt.plot(iter,var)
    plt.title("contrast")
    plt.show()

    plt.plot(iter,I)
    plt.title("mean iuminance")
    plt.show()

def printerr_var_i_due2_NA(targetf,targetr,height1,wavelenth0):


    #设定循环次数
    iter = np.linspace(100,300,50,dtype=int)
    err = []
    var = []
    I = []
    err = np.array(err)
    var = np.array(var)
    I = np.array(I)

    #循环成像
    for i in iter:
        NA = i*10**(-9)
        max_freq = int(NA / wavelenth0 * 2 * np.pi)
        timesmax = int(height1 / 2 / max_freq)
        print(timesmax)
        erri ,vari,Ii = FPM1(targetf,targetr,height1,NA,wavelenth0,timesmax,2)
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

targetr,targetf,height1 = get_target("GStarget.png")
# 初始化参数
NA0 = NA = 200E-9
wavelenth0 = 500e-9


#recon,spectrum = FPM1(targetf,targetr,height1,NA0,wavelenth0,122,1)
#save_recon(recon,"reconstruction")

#printerr_var_i_due2_times(targetf,targetr,height1,NA0,wavelenth0)
printerr_var_i_due2_NA(targetf,targetr,height1,wavelenth0)
