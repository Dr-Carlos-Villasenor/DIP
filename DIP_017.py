import cv2
from matplotlib import pyplot as plt
import numpy as np

def gaussLowPassFilter(shape, radius=10):  # Gaussian low pass filter
    u, v = np.mgrid[-1:1:2.0 / shape[0], -1:1:2.0 / shape[1]]
    D = np.sqrt(u ** 2 + v ** 2)
    D0 = radius / shape[0]
    kernel = np.exp(- (D ** 2) / (2 * D0 ** 2))
    return kernel

def butterworthNRFilter(shape, radius=9, uk=60, vk=80, n=2):  # Butterworth notch band stop filter
    M, N = shape[1], shape[0]
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    Dm = np.sqrt((u - M // 2 - uk) ** 2 + (v - N // 2 - vk) ** 2)
    Dp = np.sqrt((u - M // 2 + uk) ** 2 + (v - N // 2 + vk) ** 2)
    D0 = radius
    n2 = n * 2
    kernel = (1 / (1 + (D0 / (Dm + 1e-6)) ** n2)) * (1 / (1 + (D0 / (Dp + 1e-6)) ** n2))
    return kernel

def imgFrequencyFilter(img, lpTyper="GaussLP", radius=10):
    normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)

    # (1) Edge fill
    imgPad = np.pad(img, ((0, img.shape[0]), (0, img.shape[1])), mode="reflect")
    # (2) Centralization: F (x, y) * - 1 ^ (x + y)
    mask = np.ones(imgPad.shape)
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    imgPadCen = imgPad * mask
    # (3) Fourier transform
    fft = np.fft.fft2(imgPadCen)

    # (4) Construct the frequency domain filter transfer function:
    if lpTyper == "GaussLP":
        print(lpTyper)
        freFilter = gaussLowPassFilter(imgPad.shape, radius=60)
    elif lpTyper == "GaussHP":
        freFilter = gaussLowPassFilter(imgPad.shape, radius=60)
    elif lpTyper == "ButterworthNR":
        print(lpTyper)
        freFilter = butterworthNRFilter(imgPad.shape, radius=9, uk=60, vk=80, n=2)  # Butterworth notch band stop filter
    elif lpTyper == "MButterworthNR":
        print(lpTyper)
        BNRF1 = butterworthNRFilter(imgPad.shape, radius=9, uk=60, vk=80, n=2)  # Butterworth notch band stop filter
        BNRF2 = butterworthNRFilter(imgPad.shape, radius=9, uk=-60, vk=80, n=2)
        BNRF3 = butterworthNRFilter(imgPad.shape, radius=9, uk=60, vk=160, n=2)
        BNRF4 = butterworthNRFilter(imgPad.shape, radius=9, uk=-60, vk=160, n=2)
        freFilter = BNRF1 * BNRF2 * BNRF3 * BNRF4
    else:
        print("Error of unknown filter")
        return -1

    # (5) Modify Fourier transform in frequency domain: Fourier transform point multiplication filter transfer function
    freTrans = fft * freFilter
    # (6) Inverse Fourier transform
    ifft = np.fft.ifft2(freTrans)
    # (7) Decentralized inverse transform image
    M, N = img.shape[:2]
    mask2 = np.ones(imgPad.shape)
    mask2[1::2, ::2] = -1
    mask2[::2, 1::2] = -1
    ifftCenPad = ifft.real * mask2
    # (8) Intercept the upper left corner, the size is equal to the input image
    imgFilter = ifftCenPad[:M, :N]
    imgFilter = np.clip(imgFilter, 0, imgFilter.max())
    imgFilter = np.uint8(normalize(imgFilter) * 255)
    return imgFilter


# Using notch filtering to delete moire patterns in digital printed images
# (1) Read original image
img = cv2.imread('DataSet/car.png', flags=0)  # flags=0 read as grayscale image
fig = plt.figure(figsize=(10, 5))
plt.subplot(141), plt.title("Original"), plt.axis('off'), plt.imshow(img, cmap='gray')

# (2) Image Gaussian low pass filtering
imgGLPF = imgFrequencyFilter(img, lpTyper="GaussLP", radius=30)  # Image Gaussian low pass filtering
plt.subplot(142), plt.title("GaussLP filter"), plt.axis('off'), plt.imshow(imgGLPF, cmap='gray')

# (3) Image Butterworth notch band stop filtering
imgBNRF = imgFrequencyFilter(img, lpTyper="ButterworthNR", radius=9)
plt.subplot(143), plt.title("ButterworthNR filter"), plt.axis('off'), plt.imshow(imgBNRF, cmap='gray')

# (4) Superimposed Butterworth notch band stop filtering
imgSBNRF = imgFrequencyFilter(img, lpTyper="MButterworthNR", radius=9)
plt.subplot(144), plt.title("Superimposed BNRF"), plt.axis('off'), plt.imshow(imgSBNRF, cmap='gray')

plt.tight_layout()
plt.show()