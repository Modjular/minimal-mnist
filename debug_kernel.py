'''
Ported from:
https://github.com/google/skia/blob/main/src/effects/imagefilters/SkMatrixConvolutionImageFilter.cpp
'''

import numpy as np
TOLERANCE = 1 / (1 << 12)

def SkScalarNearlyZero(x):
    return abs(x) < TOLERANCE

def SkScalarRoundToInt(float):
    return round(float)


def create_kernel_bitmap(fWidth, fHeight, kernel):

    innerBias = 0
    innerGain = 1
    length = fWidth * fHeight

    min = kernel[0]
    max = kernel[0]
    for i in range (1, length):
        if kernel[i] < min:
            min = kernel[i]
        if kernel[i] > max:
            max = kernel[i]


    innerGain = max - min
    innerBias = min

    # Treat a near-0 gain (i.e. box blur) as 1 and let innerBias move everything to final value.
    if (SkScalarNearlyZero(innerGain)):
        innerGain = 1.0


    kernelBM = np.zeros((fWidth, fHeight))

    for y in range(fHeight):
        for x in range(fWidth):
            i = y * fWidth + x
            kernelBM[x, y] = SkScalarRoundToInt(255 * (kernel[i] - min) / innerGain)

    return kernelBM, innerGain, innerBias

weights = "8 -14 13 13 -11 -3 -16 -13 7 -16 8 -9 -5 13 1 -11 17 -5 6 10 17 -12 -7 -14 -4 10 3 -10 4 -6 14 -4 16 -9 2 -1 5 15 -1 27 19 15 9 -3 11 9 -4 15 -12 -2 -12 14 2 5 -7 -7 4 -12 -2 3 -6 -15 -3 -7 34 12 15 22 55 9 0 -12 -15 16 14 14 35 4 17 -9 -5 18 11 10 8 17 12 8 -4 -10 10 -4 10 17 28 19 -1 8 -21 -7 -9 28 20 20 -8 5 2 -1 25 9 -14 4 -3 -17 5 16 2 -33 -46 -22 -53 -37 -21 7 -14 -23 -19 12 -23 38 -5 -43 -51 -53 -47 -10 -11 -24 -1 8 -6 11 -4 -8 -26 -64 -78 -49 -48 -42 -20 29 38 40 21 -25 32 85 33 31 -1 -20 -43 -36 -23 22 -12 -14 -17 -3 8 4 -37 -77 -82 -52 -59 -90 -52 1 9 22 73 66 23 39 -9 42 23 8 -9 -31 -3 -12 -30 -16 11 1 -3 -19 -36 -47 -45 -62 -17 0 50 79 6 6 30 60 45 49 71 6 -17 33 7 27 44 7 -30 13 3 -1 -6 -4 -46 -44 -47 -48 -31 52 52 84 100 70 52 46 101 42 33 42 45 48 40 153 93 14 14 9 15 14 9 14 -21 -23 -30 -69 -27 17 77 60 73 94 19 -157 -33 -9 27 64 35 32 54 169 137 16 7 -20 10 3 17 13 8 5 -38 -15 -10 72 77 111 98 50 37 -141 -65 -39 44 39 60 68 103 189 135 28 8 -1 9 7 -9 8 -37 6 29 44 8 10 21 59 -1 27 -18 -193 -104 -82 -23 18 17 90 93 162 144 14 9 16 -8 -5 4 0 -16 -9 74 18 2 32 -27 14 -28 22 -97 -160 -83 -18 -57 -5 14 61 98 167 170 19 -3 -2 3 7 19 -6 -17 1 24 18 -17 -10 39 29 -13 0 -54 -42 -100 -99 -16 10 -37 5 90 158 115 25 -4 8 18 -14 9 9 -35 5 16 10 -33 -8 44 84 6 63 -10 -32 -88 -43 -35 -83 -21 11 -2 82 89 2 -1 -7 -3 11 9 -8 -24 27 -22 36 19 26 44 -29 8 0 -58 -51 -108 -6 0 -96 -33 -39 -74 33 56 8 7 -12 14 -7 0 -30 -43 34 13 54 82 59 32 21 30 -18 -69 -30 -43 24 -45 -71 -42 -49 -44 -35 -29 -12 -10 18 12 3 15 -24 -35 29 39 58 62 136 85 92 3 -70 -67 -25 27 24 -46 -44 -11 -66 -30 -57 -12 15 8 -13 -14 14 -16 -44 -32 34 70 100 93 141 143 39 7 -84 -96 -8 20 29 3 -54 -45 -48 2 -23 -5 -12 9 6 -10 17 -9 -31 -21 58 55 54 97 83 88 33 58 -88 -71 -44 -19 44 40 -43 0 -8 37 -16 -2 5 1 -12 7 7 16 -38 -26 96 41 48 49 37 72 26 101 42 -24 24 -23 -4 7 -14 -42 -14 -11 17 -3 -4 14 9 18 15 -21 -13 -12 14 30 67 -9 43 4 97 80 64 39 29 -44 14 -38 -53 -54 -8 45 46 -6 9 3 12 4 5 7 2 -34 -8 7 26 37 52 21 67 113 166 54 36 -16 -60 -42 -42 -43 0 20 31 -6 6 14 12 -6 2 11 12 -26 -66 -51 -12 51 61 63 82 105 106 116 83 38 -20 -18 -15 -10 -8 32 -3 -17 -16 -17 10 8 2 0 -26 -40 -96 -111 -83 -47 16 -5 25 97 99 68 84 101 63 55 17 -14 -25 -11 -7 25 -14 1 0 -1 -6 12 -1 -5 -49 -65 -91 -114 -86 -59 -19 -55 -7 4 61 73 58 29 16 -26 2 -4 9 -11 10 -12 -4 9 -13 9 14 12 2 -21 1 -16 -50 -35 -1 -2 -9 -11 -6 -17 -16 -7 6 -7 -20 4 17 -12 18 -6 -3 -16 14 2 17 -2 -2 13 5 13 16 7 12 0 -16 -5 13 -13 -7 15 -13 3 15 9 -19 11 9 -6 -3"

kernel = [1, 2, 1, 0, 0, 0, -1, -2, -1]
kernel = [float(s)/512 for s in weights.split(' ')]

bm, gain, bias = create_kernel_bitmap(28, 28, kernel)

print(bm)
print(gain, bias)