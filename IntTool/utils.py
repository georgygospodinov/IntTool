import os
import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase
from scipy.stats import multivariate_normal
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
from tqdm import tqdm


def fourier(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    spectrum = np.fft.fftshift(dft)
    return spectrum


def apply_mask(spectrum, max_id, size, complex_mask=True, gauss=False):
    rows = spectrum.shape[0]
    cols = spectrum.shape[1]

    if gauss:
        x = np.linspace(0, cols, cols, endpoint=False)
        g_x = multivariate_normal.pdf(x, mean=max_id[1], cov=size[1])
        g_x = g_x.reshape(1, cols)

        y = np.linspace(0, rows, rows, endpoint=False)
        g_y = multivariate_normal.pdf(y, mean=max_id[0], cov=size[0])
        g_y = g_y.reshape(rows, 1)
        gf = np.dot(g_y, g_x)
        gf /= np.max(gf)

        if complex_mask:
            mask = np.zeros((rows, cols, 2))
            mask[:, :, 0] = gf
            mask[:, :, 1] = gf
        else:
            mask = gf

    else:
        if complex_mask:
            mask = np.zeros((rows, cols, 2), np.uint8)
        else:
            mask = np.zeros((rows, cols), np.uint8)
        mask[max_id[0] - size[0]: max_id[0] + size[0] + 1, max_id[1] - size[1]: max_id[1] + size[1] + 1] = 1

    return spectrum * mask


def inverse_fourier(spectrum):
    f_ishift = np.fft.ifftshift(spectrum)
    img_back = cv2.idft(f_ishift)
    return img_back


def complex_abs(spectrum):
    return cv2.magnitude(spectrum[:, :, 0], spectrum[:, :, 1])


def phase(spectrum):
    return cv2.phase(spectrum[:, :, 0], spectrum[:, :, 1])


def soft(y, n=7):
    start = n // 2
    stop = y.size - n // 2
    begin = [np.mean(y[:2 * i + 1]) for i in range(start)]
    end = [np.mean(y[-2 * i - 1:]) for i in range(start)][::-1]
    medium = [np.mean(y[i - n // 2: i + n // 2 + n % 2]) for i in range(start, stop)]
    return np.array(begin + medium + end)


def read_image(path):
    return cv2.imread(path, -1)


def symmetrize(phase, center_index):
    length = phase.size
    if center_index < length / 2:
        centered_phase = np.zeros(2 * center_index + 1)
        for i in range(2 * center_index + 1):
            centered_phase[i] = phase[i]
    elif center_index > length / 2:
        centered_phase = np.zeros(2 * (length - 1 - center_index) + 1)
        i_start = center_index - (length - 1 - center_index)
        i_stop = length
        for i in range(i_start, i_stop):
            centered_phase[i - i_start] = phase[i]
    else:
        centered_phase = np.copy(phase[:-1])

    symmetrical_phase = (np.flip(centered_phase, axis=0) + centered_phase) / 2

    return symmetrical_phase


def symmetrize_2d(phase, center_index):
    row_count = symmetrize(phase[:, 0], center_index).size

    symmetrical_phase = np.zeros((row_count, phase.shape[1]))
    for i in range(symmetrical_phase.shape[1]):
        symmetrical_phase[:, i] = symmetrize(phase[:, i], center_index)
        symmetrical_phase[:, i] -= symmetrical_phase[:, i][0]
    return symmetrical_phase


def fwhm(x, y):
    spline = UnivariateSpline(x, y - np.max(y) / 2, s=0)
    roots = spline.roots()
    return roots


def critical_density(wavelength):
    # wavelength in mcm
    # return N_critical in cm^(-3)
    c = 299792458
    eps0 = 8.85e-12
    e = 1.6e-19
    m = 9.1e-31
    w = 2 * np.pi * c / (wavelength * 1e-6)
    return (w * w * eps0 * m) / (e * e) * 1e-6


def remove_plane(data):
    y_size, x_size = data.shape
    x = np.arange(x_size)
    y = np.arange(y_size)

    X1, X2 = np.meshgrid(x, y)

    # Regression
    X = np.hstack((np.reshape(X1, (x_size * y_size, 1)), np.reshape(X2, (x_size * y_size, 1))))
    X = np.hstack((np.ones((x_size * y_size, 1)), X))
    YY = np.reshape(data, (x_size * y_size, 1))

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    plane = np.reshape(np.dot(X, theta), (y_size, x_size))

    cleared_data = data - plane

    return cleared_data - cleared_data.min()


def phase_from_plasma_background(interferogram, background, mask, mask_center, mask_size, mask_gauss=False, remove=True, unwrap=True):
    y1, y2, x1, x2 = mask
    img_phase = phase(
        inverse_fourier(
            apply_mask(fourier(interferogram), mask_center, mask_size, gauss=mask_gauss)
        )
    )[y1:y2, x1:x2]
    back_phase = phase(
        inverse_fourier(
            apply_mask(fourier(background), mask_center, mask_size, gauss=mask_gauss)
        )
    )[y1:y2, x1:x2]
    if unwrap:
        reconstructed_phase = unwrap_phase(img_phase - back_phase)
    else:
        reconstructed_phase = (np.pi + img_phase - back_phase) % (2 * np.pi) + np.pi
    if remove:
        reconstructed_phase = remove_plane(reconstructed_phase)
    return reconstructed_phase - reconstructed_phase.min()


def remove_linear_trend(data):
    linear_trend = data[0] + (data[-1] - data[0]) * np.arange(data.size) / data.size
    return data - linear_trend


def rotate_image(img, angle, y1, y2, x1, x2):
    return rotate(img, angle)[y1:y2, x1:x2]


def h_n(n, y, R, step=0.0001):
    if n == 0:
        return (R ** 2 - y ** 2) ** 0.5
    r = np.arange(0, (R ** 2 - y ** 2) ** 0.5 + step, step)
    f = 1 - (-1) ** n * np.cos(np.pi * n * (y ** 2 + r ** 2) ** 0.5 / R)
    return np.trapz(f, r)


def abel_inversion_2d(data, n_min=1, n_max=20, path_to_save='./abel_coefs/', section_id=-1, two_gauss=False,
                      verbose=False, p0=None):
    if two_gauss:
        two_gauss_abel_inversion(
            data[:, section_id],
            n_min=n_min,
            n_max=n_max,
            path_to_save=path_to_save,
            verbose=True,
            p0=p0
        )

        return np.array([
            two_gauss_abel_inversion(
                data[:, i],
                n_min=n_min,
                n_max=n_max,
                path_to_save=path_to_save,
                verbose=verbose,
                p0=p0
            )
            for i in tqdm(range(data.shape[1]))
        ]).T

    else:
        abel_inversion(data[:, section_id], n_min=n_min, n_max=n_max, path_to_save=path_to_save, verbose=True)

        return np.array([
            abel_inversion(data[:, i], n_min=n_min, n_max=n_max, path_to_save=path_to_save, verbose=verbose)[1]
            for i in tqdm(range(data.shape[1]))
        ]).T


def line_form(x, x0, a, b):
    return abs(a) * np.exp(-(x - x0) ** 2 / b ** 2)


def fit_function(x, x1, a1, b1, x2, a2, b2):
    return abs(a1) * np.exp(-(x - x1) ** 2 / b1 ** 2) + abs(a2) * np.exp(-(x - x2) ** 2 / b2 ** 2)


def r_2(y, fitted_y):
    return 1 - np.sum((y - fitted_y) ** 2) / np.sum((y - y.mean()) ** 2)


def two_gauss_abel_inversion(h_y, n_min=1, n_max=20, path_to_save='./abel_coefs/', verbose=True, r_2_threshold=0.85,
                             p0=None):
    """
    TODO: return H_y
    """
    x = np.arange(h_y.size) - h_y.size / 2

    popt, _ = curve_fit(fit_function, x, h_y, maxfev=1000000, p0=p0)

    fitted_y = fit_function(x, *popt)

    R_2 = r_2(h_y, fitted_y)

    if R_2 < r_2_threshold:
        return abel_inversion(h_y, n_min=n_min, n_max=5, path_to_save=path_to_save, verbose=verbose)[1]

    if verbose:
        print(f'r_2: {R_2}')

    fit = {
        "left": line_form(x, *popt[:3]),
        "right": line_form(x, *popt[3:])
    }

    if verbose:
        plt.figure(figsize=(18, 6))
        plt.subplot(121)
        plt.plot(x, h_y, label='initial')
        plt.plot(x, fitted_y, label='fitted')
        plt.xlabel('r, пиксели', fontsize=14)
        plt.ylabel('фаза, радиан', fontsize=14)
        plt.grid()
        plt.legend(fontsize=12)

        plt.subplot(122)
        plt.xlabel('r, пиксели', fontsize=14)
        plt.ylabel('фаза, радиан', fontsize=14)
        plt.plot(x, h_y, label='initial')
        plt.plot(x, fit["left"], label='fitted')
        plt.plot(x, fit["right"], label='fitted')
        plt.grid()
        plt.legend(fontsize=12)
        plt.show()

    line = {
        "left": line_form(x, *popt[:3]),
        "right": line_form(x, *popt[3:])
    }

    dist = line["right"].argmax() - line["left"].argmax()

    centered_line = {
        "left": symmetrize(line["left"], line["left"].argmax()),
        "right": symmetrize(line["right"], line["right"].argmax())
    }

    abel = {
        "left": abel_inversion(
            centered_line["left"],
            n_min=n_min,
            n_max=n_max,
            path_to_save=path_to_save,
            verbose=verbose
        )[1],
        "right": abel_inversion(
            centered_line["right"],
            n_min=n_min,
            n_max=n_max,
            path_to_save=path_to_save,
            verbose=verbose
        )[1]
    }

    if verbose:
        polar = phi_rotate(abel['left'][abel['left'].size // 2:], x0=-dist / 2, size=1000) + phi_rotate(
            abel["right"][abel["right"].size // 2:], x0=dist / 2, size=1000)
        plt.imshow(polar, cmap='jet')
        plt.savefig('./rotated_distribution_gauss.png', dpi=300)
        plt.show()

    abel_with_zeros = {
        "left": np.concatenate((
            abel["left"],
            [0] * (x.size - abel["left"].size)
        )),
        "right": np.concatenate((
            [0] * (x.size - abel["right"].size),
            abel["right"]
        ))
    }

    return abel_with_zeros["left"] + abel_with_zeros["right"]


def abel_inversion(h_y, n_min=1, n_max=20, path_to_save='./abel_coefs/', verbose=True, N_MAX=20):
    """
    TODO: draw learning curve for n in [1, N_MAX]
    """
    h_y -= h_y[0]
    R = h_y.size // 2

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
        print(f'dir {path_to_save} was created')

    coefficients_path = os.path.join(path_to_save, f'R={R}')

    I = {}

    if os.path.exists(coefficients_path):
        with open(coefficients_path, 'rb') as handle:
            I.update(pickle.load(handle))
    else:
        for n in tqdm(range(N_MAX + 1)):
            for y in np.arange(0, R + 1, 1):
                I[(n, y)] = h_n(n, y, R, step=1e-3)
                I[(n, -y)] = I[(n, y)]

        with open(coefficients_path, 'wb') as handle:
            pickle.dump(I, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X = np.zeros((h_y.size, n_max - n_min + 1))

    for i, y in enumerate(np.arange(-R, R + 1)):
        for n in range(n_min, n_max + 1):
            X[i, n - n_min] = 2 * I[(n, y)]

    A = np.linalg.inv(X.T @ X) @ (X.T @ h_y)

    r = np.arange(-R, R + 1)
    f_r = A @ np.array([(1 - (-1) ** n * np.cos(np.pi * n * r / R)) for n in range(n_min, n_max + 1)])

    H_y = X @ A

    if verbose:
        errors = []
        for n in range(1, X.shape[1] + 1):
            x = X[:, :n]
            A = np.linalg.inv(x.T @ x) @ (x.T @ h_y)
            error = ((h_y - x @ A) ** 2).mean()
            errors.append(error)
        plt.plot(range(n_min, n_max + 1), errors)
        plt.yscale('log')
        plt.ylabel('MSE')
        plt.xlabel('n')
        plt.grid()
        plt.show()

        plt.figure(figsize=(18, 5))
        plt.subplot(131)
        plt.plot(r, h_y, label='исходные данные')
        plt.plot(r, H_y, label='реконструкция')
        plt.legend()
        plt.grid()

        plt.subplot(132)
        plt.title('ошибка восстановления')
        plt.plot(r, h_y - H_y)
        plt.grid()

        plt.subplot(133)
        plt.title('результат обратного преобразования Абеля')
        plt.plot(r, f_r)
        plt.grid()
        plt.show()

    return H_y, f_r


def phi_rotate(r, x0=0, y0=0, size=None):
    R = r.size - 1
    if size is None or size < 2 * R + 1:
        size = 2 * R + 1
    out = np.zeros((size, size))
    for i, y in enumerate(range(-size // 2, size // 2 + 1)):
        for j, x in enumerate(range(-size // 2, size // 2 + 1)):
            index = int(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5)
            if index < R:
                out[i, j] = r[index]
    return out


def interferogram_describe(interferogram, mask_size, area_to_show, gauss_mask=False, shift_x=800, shift_y=1200):
    plt.imshow(interferogram, cmap='gray')
    plt.title('interferogram')
    plt.axis('off')
    plt.show()

    spectrum = complex_abs(fourier(interferogram))

    plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plt.imshow(np.log(spectrum), cmap='jet')
    plt.title('spectrum')

    shifted_spectrum = spectrum[:shift_y, :shift_x]

    plt.subplot(132)
    plt.imshow(np.log(shifted_spectrum), cmap='jet')
    plt.title('shifted spectrum')

    max_id = np.array(np.unravel_index(shifted_spectrum.argmax(), shifted_spectrum.shape))

    plt.subplot(133)
    plt.imshow(
        np.log(spectrum[
               max_id[0] - area_to_show[0]: max_id[0] + area_to_show[0] + 1,
               max_id[1] - area_to_show[1]: max_id[1] + area_to_show[1] + 1
               ]),
        cmap='jet'
    )
    plt.title('detected maximum')
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plt.imshow(
        apply_mask(np.log(spectrum), max_id, mask_size, complex_mask=False, gauss=gauss_mask)[
            max_id[0] - area_to_show[0]: max_id[0] + area_to_show[0] + 1,
            max_id[1] - area_to_show[1]: max_id[1] + area_to_show[1] + 1
        ],
        cmap='jet'
    )
    plt.title('maximum with filter')

    plt.subplot(132)
    plt.imshow(
        apply_mask(
            np.log(spectrum),
            max_id,
            mask_size,
            complex_mask=False,
            gauss=gauss_mask
        ),
        cmap='jet'
    )
    plt.title('spectrum with mask')

    plt.subplot(133)
    plt.imshow(
        complex_abs(
            inverse_fourier(apply_mask(fourier(interferogram), max_id, mask_size, gauss=gauss_mask))
        ),
        cmap='gray')
    plt.title('Inverse fourier after mask')
    plt.axis('off')
    plt.show()

    return max_id
