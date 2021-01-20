import cv2
import numpy as np

def gaussian_filter(input, output, l = 5, sig = 50):
    center = (l - 1) // 2
    ax = np.linspace(-center, center, l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = kernel / np.sum(kernel)

    for x in range(center, input.shape[0] - center):
        for y in range(center, input.shape[1] - center):
            output[x, y] = np.sum(kernel * input[x - center:x + center + 1, y - center:y + center + 1])

    return output


#
# #median filtering

def median_filter(input, output, l = 5):
    center = (l - 1) // 2
    img = np.zeros((input.shape[0] + 2*center, input.shape[1] + 2*center))
    img[center:input.shape[0] + center, center:input.shape[1] + center] = input
    for x in range(center, img.shape[0] - center):
        for y in range(center, img.shape[1] - center):
            kernel_median = np.copy(img[x-center:x+center+1,y-center:y+center+1])
            kernel_median = kernel_median.flatten()
            kernel_median = sorted(kernel_median)
            median = kernel_median[(len(kernel_median)-1)//2]
            output[x - center,y - center] = median
    return output

def find_text_area(input, output):
    ret, thresh1 = cv2.threshold(input, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #font specs
    number_box = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (0,0,255)
    lineType = 2

    contours.reverse()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        if cv2.contourArea(cnt) > 3000:
            rect = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output, str(number_box),
                    (x + 20, y  + 40),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

            number_box = number_box + 1

    return thresh1, contours

def calc_text_area(img, min_size=10):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    pixel_size = 0
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if (sizes[i] < min_size):
            img[output == i + 1] = 0
        else:
            pixel_size += sizes[i]
    print("Area (px): ", pixel_size)

def word_count(img):

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilation = cv2.dilate(img, rect_kernel, iterations=1)

    _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    words = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        area = cv2.contourArea(cnt)
        if area > 200:
            words += 1
    print("Number of words: ", words)

def integral_gray(int_img, x ,y , w ,h):

    img_area = int_img[y + h - 1, x + w - 1] + int_img[y - 1, x  - 1] - int_img[y + h - 1, x - 1 ] - int_img[y - 1, x + w - 1]
    gray_mean = img_area / (w*h)
    print("Mean gray-level value in bounding box: ", gray_mean)

def hw1(original, filename):

    original = cv2.imread(original)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img1 = gray.copy()
    # median_filter(gray, img1)
    thresh, contours = find_text_area(img1, original)
    intergral_image = cv2.integral(img1)

    i = 1
    for cnt in  contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = thresh[y:y + h, x:x + w]
        if cv2.contourArea(cnt) > 3000:
            print("---- Region ", i, ": ----")
            calc_text_area(cropped)
            print("Bounding Box Area (px): ", cv2.contourArea(cnt))
            word_count(cropped)
            integral_gray(intergral_image, x, y, w, h)
            i += 1

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', original)
    cv2.waitKey(0)

original = '2_original.png'
filename = '2_original.png'
hw1(original, filename)

