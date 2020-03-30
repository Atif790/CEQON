import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageEnhance
from PIL import ImageFilter
from scipy.signal import convolve2d
from torchvision import transforms
import pytorch_ssim
import torchvision.transforms.functional as TF
from scipy.stats import kstest, ks_2samp
import numpy as np
from math import log10, sqrt
from scipy import stats
import math

org_scanned = 'Images/scaned_bars.jpg'
original = 'Images/bars.png'

org_img = cv2.imread(original, 0)
sc_img = cv2.imread(org_scanned, 0)

sc_sliced = sc_img[320:1640, 275:1793]
composed = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480, 571), interpolation=Image.NEAREST)])
org_img = composed(org_img)
sc_img = composed(sc_sliced)

org_img = transforms.ToTensor()(org_img)
sc_img = transforms.ToTensor()(sc_img)


def print_scanned_image():
    printed_scanned = "Images/printed_and_scanned_from_scanned_image_bars.jpg"
    p_s_img = cv2.imread(printed_scanned, 0)
    p_s_sliced = p_s_img[350:1720, 342:1747]
    p_img = transforms.ToPILImage()(p_s_sliced)
    r_image = transforms.Resize((480, 571))(p_img)
    rotated_img = TF.rotate(r_image, -2, fill=(0,))
    plt.imsave("Printed_scanned.jpg", rotated_img)
    # plt.show()
    rotated_img = transforms.ToTensor()(rotated_img)
    return rotated_img


def MSE(org_img, sc_img):
    org_img = org_img.numpy()
    sc_img = sc_img.numpy()
    err = np.sum((org_img.astype("float") - sc_img.astype("float")) ** 2)
    err /= float(org_img.shape[0] * sc_img.shape[1])
    return err


def cal_ssim(org, scanned):
    org = org.unsqueeze(0)
    scanned = scanned.unsqueeze(0)
    ssim = pytorch_ssim.ssim(org, scanned)
    return ssim


def cal_contrast(img):
    img = img.squeeze(0)
    height, width = img.shape
    total_pixels = height * width
    img = img.unsqueeze(0)
    white = cv2.countNonZero(np.float32(img))
    black = total_pixels - white
    contrast = abs(white - black)
    return contrast


def Covariance(x, y):
    x = x.squeeze(0)
    y = y.squeeze(0)
    x = x.numpy()
    y = y.numpy()
    x = np.cov(x)
    y = np.cov(y)
    x = x.mean()
    y = y.mean()
    m = (x + y) / 2
    return m


def PSNR(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def estimate_noise(I):
    I = I.squeeze(0)
    print(I.shape)
    H, W = I.shape
    M = [[1, -2, 1],  # High numbers means low noise.
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))
    return sigma


def joint_histogram(img1, img2):
    # img1 = img1.numpy()
    # img2 = img2.numpy()
    # h = cv2.calcHist( [img1, img2], [0, 1], None,  [180, 256], [0, 180, 0, 256] )
    h1 = torch.histc(img1, bins=4, min=0, max=571)
    h2 = torch.histc(img2, bins=4, min=0, max=571)
    plt.subplot(2, 2, 1)
    plt.plot(h1)
    plt.subplot(2, 2, 3)
    plt.plot(h2)
    plt.show()
    # print(hist)


def cal_variance(img):
    var = torch.var(img)
    print(var.item())


def cal_Values(org_img, scanned_print_scanned):
    mse_value = MSE(org_img, scanned_print_scanned)  # 0 in MSE indicates perfactly similar
    manhattan_distance = torch.dist(org_img, scanned_print_scanned, 1)
    eculidian_dist = torch.dist(org_img, scanned_print_scanned, 2)
    ssim_val = cal_ssim(org_img, scanned_print_scanned)
    org_img_contrast = cal_contrast(org_img)
    sc_img_contrast = cal_contrast(scanned_print_scanned)
    contrast_diff = abs(org_img_contrast - sc_img_contrast)
    mean_val_orig = torch.mean(org_img)
    mean_val_sc = torch.mean(scanned_print_scanned)
    Mean_diff = abs(mean_val_orig - mean_val_sc)
    psnr = PSNR(org_img, scanned_print_scanned)
    n_org = estimate_noise(org_img)
    n_sc = estimate_noise(scanned_print_scanned)

    # jointhistogram(org_img, sc_img)

    print("Manhattan Distance: ", manhattan_distance.item())  # Direct distance
    print("Eculidian Distance:", eculidian_dist.item())  # Measure by pythogoros theorm
    print("Mean Square Error: ", mse_value)
    print("Structural Similarity (SSIM): ", ssim_val.item())  # Between -1 to 1. 1 is for most similar
    print("Co-Variance: ", Covariance(org_img, scanned_print_scanned))
    print("Contrast Difference: ", contrast_diff)
    print("Mean Difference: ", Mean_diff.item())
    print("PSNR: ", psnr)
    print("Origianl Images Noice: ", n_org)
    print("Scanned Images Noice: ", n_sc)


def k_s_test(img1, img2):
    img1 = img1.numpy()
    img2 = img2.numpy()
    print(img1)
    # return ks_2samp(img1, img2)


def filters(img):
    # org_image= transforms.ToTensor()(img)
    composed = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480, 571), interpolation=Image.NEAREST)])
    trans_org_img = composed(img)

    enh_bri = ImageEnhance.Brightness(trans_org_img)
    brightness = 1.0
    image_brightened = enh_bri.enhance(brightness)

    enh_col = ImageEnhance.Color(image_brightened)
    color = 1.5
    image_colored = enh_col.enhance(color)

    enh_con = ImageEnhance.Contrast(image_colored)
    contrast = 1.0
    image_contrasted = enh_con.enhance(contrast)

    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    sharpness = 1.5
    image_sharped = enh_sha.enhance(sharpness)

    image_sharped = transforms.ToTensor()(image_sharped)
    return image_sharped


def display(img):
    img = img.squeeze(0)
    plt.imshow(img)
    plt.show()


def linear_filter(img):  # Linear Filter
    composed = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480, 571), interpolation=Image.NEAREST)])
    img = composed(img)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    dst = cv2.filter2D(np.float32(img), -1, kernel)
    cv2.imshow("Org", np.float32(img))
    cv2.imshow("Filtered", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def region_discontinuities(img):  # Linear Filter
    composed = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480, 571), interpolation=Image.NEAREST)])
    img = composed(img)
    kernel = np.array([[-1, 2, -1],
                       [-1, 2, -1],
                       [-1, 2, -1]])
    dst = cv2.filter2D(np.float32(img), -1, kernel)
    cv2.imshow("Org", np.float32(img))
    cv2.imshow("Filtered", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def remove_extranous_marks():
    #img = cv2.imread('Images/sample_noise.jpeg')
    img = cv2.imread('Images/sample_noise_2.png')
    img_bw = 255 * (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    mask = np.dstack([mask, mask, mask]) / 255
    out = img * mask
    cv2.imshow('Output', out)
    cv2.imshow("Original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def median_filter():
    img = cv2.imread('Images/sample_noise_2.png',0)
    median = cv2.medianBlur(img, 5)

    compare = np.concatenate((img, median), axis=1) #side by side comparison

    cv2.imshow('img', compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


median_filter()
#remove_extranous_marks()
#filter_org = region_discontinuities(org_img)
# filter_scanned = filter_2(sc_img)
# filter_printed_scanned = filter_2(print_scanned_image())

'''
cal_Values(filter_scanned,filter_printed_scanned)'''

# Extranous Marks is salt and peper noice
