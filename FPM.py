import numpy as np
from scipy.fftpack import fft2, ifft2,fftshift,ifftshift
import cv2
import matplotlib.pyplot as plt

"""本程序用于实现傅里叶叠层成像，样本与最终成像的数组单位为um(10**-6m)"""

#归一化
def normalization(origin):

    maxi = np.max(origin)
    mini = np.min(origin)
    norm = ((origin - mini) / (maxi - mini))
    return norm

#重构
def reconstruct( norm_int ):
    norm_int = normalization(norm_int)
    rec = norm_int * 255
    rec = rec.astype("uint8")
    return rec

#获取目标函数并叠加随机噪声
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

#保存重构图像
def save_recon(rec,rec_name):
    cv2.imwrite("img_FPM/{}.bmp".format(rec_name), rec)
    cv2.imshow("Reconstruction", rec)
    cv2.waitKey(0)




#叠层成像
def FPM(target,height,NA,wavelen,H,times,task):

    """选择任务：
    task = 1：输出固定参数的最终成像
    task = 2, 输出此参数下的方差、对比度
    """
    # 处理参数
    max_freq = int(NA / wavelen * 2 * np.pi)
    times = int(times)
    task = int(task)
    height = int(height)


    # 初始化低数值孔径成像数组与最终频谱
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


            #b = height*wavelen/4/np.pi*10**(-6)# 比例系数
            b = 2*np.pi/wavelen*H
            x = np.linspace(-thetai * b, thetai * b, height, dtype=complex)
            y = np.linspace(-thetaj * b, thetaj * b, height, dtype=complex)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-1j * (X+Y))

            # 不同的LED角度的成像
            psf = np.array(target, dtype=complex)*Z
            # 得到不同位置的低分辨率成像并进行拼接
            spectrumii = fftshift(fft2(psf))
            spectrumi = np.zeros((int(height), int(height)), dtype=complex)

            # 频谱平移拼接
            x_pose_n = int((height/2-max_freq+( height/2-max_freq)*i*2/times)*(i<times/2)+(height/2-max_freq-( height/2-max_freq)*(i-times/2)*2/times)*(i>times/2))
            y_pose_n = int((height/2-max_freq+( height/2-max_freq)*j*2/times)*(j<times/2)+(height/2-max_freq-( height/2-max_freq)*(j-times/2)*2/times)*(j>times/2))
            spectrumi[int(x_pose_n):int(x_pose_n+2*max_freq) , int(y_pose_n):int(y_pose_n+2*max_freq)] = spectrumii[int(height/2-max_freq):int(height/2+max_freq) , int(height/2-max_freq):int(height/2+max_freq)]

            spectrum += spectrumi
    # 将最终的频谱进行逆傅里叶变换得到频谱数据
    rec = np.abs(ifft2(ifftshift(spectrum)))
    rec = reconstruct(rec**2)

    if task == 1:
        return rec,spectrum
    else:
        I = np.sum(np.sum(rec))-np.sum(np.sum(target))
        var = (np.max(rec)-np.min(rec))/(np.max(rec)+np.min(rec))
        err = np.sum(np.sum((rec - target ** 2) ** 2) )/ height / height
        return err,var,I


#绘图模块，绘制倾斜照明次数不同带来的误差、对比度、平均亮度
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
        erri ,vari,Ii = FPM(targetf,targetr,height1,NA0,wavelenth0,i,2)
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


#绘图模块，绘制数值孔径不同带来的误差、对比度、平均亮度
def printerr_var_i_due2_NA(targetf,targetr,height1,wavelenth0):
    #设定循环次数
    iter = np.linspace(100120,300,50,dtype=int)
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
        erri ,vari,Ii = FPM(targetf,targetr,height1,NA,wavelenth0,timesmax,2)
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

targetr,targetf,height1 = get_target("laodeng.png")
# 初始化参数
NA0 = NA = 0.1
wavelenth0 = 632e-9
H = 86E-3

recon,spectrum = FPM(targetf,targetr,height1,NA0,wavelenth0,122,1)
save_recon(recon,"reconstruction")
printerr_var_i_due2_times(targetf,targetr,height1,NA0,wavelenth0)
printerr_var_i_due2_NA(targetf,targetr,height1,wavelenth0)





