import cv2
import skimage
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_nd
import os
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import datetime
from pydicom.dataset import FileDataset, FileMetaDataset, validate_file_meta
from pydicom.uid import UID, generate_uid
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
import io
import copy


def convert_image_to_ubyte(img):
    return img_as_ubyte(rescale_intensity(normalize_image(img), out_range=(0.0, 1.0)))


def normalize_image(image):
    image = np.maximum(image, 0)
    image = (image / np.quantile(image, 0.99)).astype(np.float64)
    image = np.minimum(image, 255)
    return image


def save_dicom(imageData, patient_name, comments):
    ds = pydicom.dcmread("dummy.dcm")
    ds.PixelData = imageData.tobytes()
    ds.Rows, ds.Columns = imageData.shape

    ds.PatientName = patient_name
    ds.ImageComments = comments
    dt = datetime.datetime.now()
    ds.AcquisitionDate = dt.strftime('%Y%m%d')

    ds.save_as("result.dcm", write_like_original=False)


def generate_points_between_2(coordinates, width):
    """
    Generates a list of coordinates(points) for a line segment given its starting and ending points

    Args:
        coordinates (list): A list of four elements containing the x and y coordinates
            of two points in the format [x1, y1, x2, y2].
        width (int): The width of the line in pixels.

    Returns:
        list: A list of points in the format [(x1, y1), (x2, y2), ...] that make
            up the line with the given width.
    """
    # convert the input coordinates (a list or tuple of four integers) into four separate integers
    x1, y1, x2, y2 = map(int, coordinates)
    """
    Calculating the absolute difference(length of the line segment) and sign between the x and y coordinates of the 
    two points (direction)  is a way to determine the direction and magnitude of the line segment that connects them.
    Direction will be either 1 or -1, depending on whether the x and y coordinates of the second point are 
    greater than or less than those of the first point.
    """
    abs_x, abs_y = np.abs([x2 - x1, y2 - y1])
    dif_x, dif_y = np.sign([x2 - x1, y2 - y1])
    """
    The final line filters the generated coordinates to only include those that fall within a circle with radius equal 
    to width / 2 and centered at (width / 2, width / 2). This is achieved by using a list comprehension that checks if
    the distance between each coordinate and the center of the circle is less than or equal to width / 2.
    """
    # generates a list of coordinates by generate_points_for_line
    result = [(x, y) for x, y in genetate_points_for_line(x1, y1, x2, y2, abs_x, abs_y, dif_x, dif_y)
              if ((x - width / 2) ** 2 + (y - width / 2) ** 2) <= (width / 2) ** 2]
    return result


def genetate_points_for_line(x1, y1, x2, y2, abs_x, abs_y, dif_x, dif_y):
    """
    Generate a list of points for a line.

    Args:
        x1 (int): The x-coordinate of the first point.
        y1 (int): The y-coordinate of the first point.
        abs_x (int): The absolute difference between the x-coordinates of the two points.
        abs_y (int): The absolute difference between the y-coordinates of the two points.
        dif_x (int): The direction of the x-coordinates (either 1 or -1).
        dif_y (int): The direction of the y-coordinates (either 1 or -1).

    Returns:
        list: A list of points in the format [(x1, y1), (x2, y2), ...] that make
            up the line.
    """
    # initializes a list containing the starting point of the line segment
    result = [(x1, y1)]
    # this value will be used to determine which direction to move along the line segment as new points are generated.
    # error bound
    e = abs_x - abs_y
    # starting from the initial point and adding one point at a time until the final point is reached
    # Bresenham's line algorithm
    while (x1 != x2) or (y1 != y2):
        e2 = e * 2
        if e2 > -abs_y:
            e -= abs_y
            x1 += dif_x
        if e2 < abs_x:
            e += abs_x
            y1 += dif_y
        result.append((x1, y1))
    return result


def check_points_coordinates(angle, bias, width):
    """
    Calculate the positions of two points on a circle (opposite, in I and III) with the given `angle`, `bias`, and `width`.
    this code appears to define a function that generates the coordinates of a rectangle with a certain width,
    rotated at a certain angle and translated by a certain distance.

    Args:
        angle (float): The angle in radians of the points relative to the positive x-axis.
        bias (float): The distance from the center of the circle to the line connecting the points.
        width (float): The diameter of the circle.

    Returns:
        List[float]: A list of four floats representing the x and y coordinates of the two points,
        in the order (x1, y1, x2, y2).
    """
    # calculates the radius of the rectangle
    radius = width / 2
    """
    calculate the x and y coordinates of a point on the circumference of a circle with radius equal to the "bias"
    input parameter, at an angle that is 90 degrees clockwise from the angle input parameter
    This point will be used to translate the rectangle later.
    """
    x_coordinate = bias * np.cos(angle + np.pi / 2)
    y_coordinate = bias * np.sin(angle + np.pi / 2)
    """
    use the radius and angle input parameters, along with the previously calculated point, to determine the four 
    corners of the rectangle. The first two coordinates correspond to the top right corner of the rectangle,
    while the last two coordinates correspond to the bottom left corner of the rectangle.
    """
    return (
        radius * np.cos(angle) + x_coordinate + radius,
        radius * np.sin(angle) + y_coordinate + radius,
        -radius * np.cos(angle) + x_coordinate + radius,
        -radius * np.sin(angle) + y_coordinate + radius,
    )


def value_of_line(line, width, height, image_read, additive_or_substractive):
    """
    The code takes a line represented as a list of (x, y) coordinates and reads the pixel values of an image at
    those coordinates. The sum of the pixel values is normalized by the total number of coordinates in the line.
    This normalization makes the contrast measure more robust. Finally, the exponential of the negative of the
    normalized value is calculated to create a measure of the contrast of the line. The exponential function
    is used to transform the contrast measure into a more intuitive and interpretable range, where higher
    values indicate higher contrast. The negative sign is used to invert the contrast measure so that
    higher values indicate higher contrast, consistent with most other image processing functions.

    Args:
    line (list): A list of tuples representing the coordinates of the points
    that make up the line.
    width (int): The width of the image.
    height (int): The height of the image.
    image_read (np.ndarray): A numpy array representing the image.
    additive_or_substractive (bool): A boolean value indicating whether to use an experimental
    method of calculating the line value.

    Returns:
    float or np.ndarray: The calculated value of the line.
    """
    result = []  # initialize an empty list to store pixel values

    count_elements = 0
    sum_elements = 0
    for x, y in line:
        if 0 <= x < width and 0 <= y < height:  # check if the x and y coordinates are within the bounds of the image
            # if the x and y coordinates are within the bounds of the image, read the pixel value from the image and
            # add it to the result list
            result.append(image_read[x, y])
            sum_elements += image_read[x, y]
            count_elements += 1

    if additive_or_substractive:
        if len(line) > 0:
            # calculate the contrast measure by taking the negative exponential of the average pixel value of the line
            # normalized by 255 and the length of the line
            normalize_sum = sum_elements / (255 * len(line))
            return np.exp(-normalize_sum)
        else:
            return 0  # if there are no elements in the line, return 0
    else:
        if count_elements > 0:
            # return an array of pixel values of the line if there are any
            return sum_elements / count_elements
        else:
            return 0


def sinogram_creation(image_read, width, height, angle, number_of_emitters, thickness,
                      additive_or_substractive):
    sinogram_array, lines_array = [], []

    """
    The purpose of this lines of code is to generate a sequence of angles at which to take projections of the 
    input image. The angle variable represents the step size between each angle in degrees, and in this case is 
    set to 4. The np.linspace() function is then used to generate an array of angles between 0 and 360 (exclusive)
    with a step size of angle
    """
    print(angle)
    angulation = np.linspace(0, 360, num=360 // angle)
    print(angulation)
    """
    If tmp_for_biases and tmp_for_emitters are provided, they will be used to set the number of biases and emitters,
    respectively. If they are not provided (i.e., they are set to None), the default value will be the height 
    of the image_read

    Setting the number of biases and emitters is important because it affects the quality of the 
    sinogram image. Having more biases and emitters will result in a higher resolution sinogram, which can 
    lead to a better output image
    """

    if thickness is None:
        thickness = image_read.shape[0]

    if number_of_emitters is None:
        number_of_emitters = image_read.shape[0]

    """
    This line of code creates an array biases with values that represent the positions of the sensors or 
    detectors relative to the center of the object being imaged.

    The np.arange() function returns evenly spaced values within a given interval. In this case, the interval is 
    from -space_biases/2 to space_biases/2, and the spacing between each value is space_biases/space_emitters.

    The purpose of creating this biases array is to use it in the generation of the sinogram, 
    which is a two-dimensional array that contains the projections of the image onto the sensors at different
    angles. Each row of the sinogram corresponds to a different angle of projection, and each column 
    corresponds to a different position of the sensors or detectors.
    """

    biases = np.arange(-thickness / 2, thickness / 2, thickness / number_of_emitters)
    print(biases)
    iter_sinograms = []
    for angle_temp in angulation:
        """
        Here, angulation[i] represents the angle at which to take the projection at the i-th iteration
        of the loop. The angle is then converted from degrees to radians and used to calculate the points 
        at which to take the projection, as seen in the check_points_coordinates() function.
        """
        angle = angle_temp * np.pi / 180
        sinogram_array_row, lines_array_row = [], []

        for bias in biases:
            """
            bias(position of the sensors or detectors) variable is used to calculate the coordinates of the 
            line in the image
            """
            # returns the coordinates of some key points on the object
            points = check_points_coordinates(angle, bias, width)
            x1, y1, x2, y2 = points
            # takes the points returned by the previous function and generates a line or curve that connects them
            line_to_zip = line_nd((x1, y1), (x2, y2))
            line = list(zip(line_to_zip[0], line_to_zip[1]))
            # calculates property of the line based on its position and the image data
            color = value_of_line(line, width, height, image_read, additive_or_substractive)
            # this array is being used to accumulate information about the image
            sinogram_array_row.append(color)
            # this array is being used to keep track of the lines generated during the image processing task
            lines_array_row.append(line)
        sinogram_array.append(sinogram_array_row)
        lines_array.append(lines_array_row)
        iter_sinograms.append(copy.deepcopy(sinogram_array[::-1]))


    """
    This is done to create a negative image, where dark areas in the original image become light 
    and light areas become dark
    """
    if additive_or_substractive:
        # Scale the sin array to the range [0, 255]. This is done to ensure that all the values
        # in the array are within the valid range for pixel values in an 8-bit image
        sinogram_array = (sinogram_array - np.min(sinogram_array)) * (
                256 / (np.max(sinogram_array) - np.min(sinogram_array)))
        # Invert the values in the sin array
        sinogram_array = 255 - sinogram_array

    # Reverse the sinogram
    reversed_array = sinogram_array[::-1]
    sinogram_image = np.array(reversed_array)

    # This line is used to generate reconstructed images at n-th scan iteration
    # Radon_transform_reverse(sinogram_image, width, height, lines_array, iterations=75)

    sinogram_filtered = filter_sinogram(sinogram_image, 45)
    # output_image_filtered, copy_image_n_iteration = radon_transform_reverse(sinogram_filtered, width, height,
    #                                                                         lines_array)

    output_image_filtered, image_iterations = radon_transform_reverse(sinogram_filtered, width, height,
                                                                      lines_array, iterations=50)

    plt.subplot(161)
    plt.imshow(image_read, cmap='gray')
    plt.title('Input Image')

    plt.subplot(162)
    plt.imshow(sinogram_image, cmap='gray')
    plt.title('Sinogram')

    output_image, copy_image_n_iteration = radon_transform_reverse(sinogram_image, width, height, lines_array)
    plt.subplot(163)
    plt.imshow(output_image, cmap='gray')
    plt.title('Output Image')

    plt.subplot(164)
    plt.imshow(sinogram_filtered, cmap='gray')
    plt.title('Filtered Sinogram')

    output_image_filtered = convert_image_to_ubyte(output_image_filtered)

    plt.subplot(165)
    plt.imshow(output_image_filtered, cmap='gray')
    plt.title('Filtered Output Image')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=2,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig('result.png', format="png", bbox_inches='tight')

    print(f"Mean square error: {mean_square_error(image_read, output_image_filtered)}")
    return output_image_filtered, image_iterations, iter_sinograms


def save_image_at_iteration_2(reconstructed_image, n):
    # create the "stages" subdirectory if it does not exist
    os.makedirs("stages", exist_ok=True)

    # save the image to the "stages" subdirectory
    filename = f"stages/iteration_{n}.png"
    plt.imshow(convert_image_to_ubyte(reconstructed_image), cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')


def save_image_at_iteration_sinogram(sinogram_image, n):
    # create the "stages" subdirectory if it does not exist
    os.makedirs("stages", exist_ok=True)

    # save the image to the "stages" subdirectory
    filename = f"stages/iteration_sinogram_{n}.png"
    plt.imshow(convert_image_to_ubyte(sinogram_image), cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')


def radon_transform_reverse(sinogram, width, height, lines_array, iterations=None):
    # first initializes a 2D array with zeros
    reconstructed_image = np.zeros((width, height))

    copy_image_n_iteration = []

    """
    Using nested loops we iterate over each element of the sinogram and the lines array. For each line in the lines 
    array, the function iterates over each pixel in the line and adds the corresponding sinogram value to the 
    corresponding pixel in the reconstructed_image array
    """
    for line_idx in range(sinogram.shape[0]):
        for angle_idx in range(sinogram.shape[1]):
            for pixel_idx in range(len(lines_array[line_idx][angle_idx])):
                pixel_x, pixel_y = lines_array[line_idx][angle_idx][pixel_idx]
                if 0 <= pixel_x < width and 0 <= pixel_y < height:
                    reconstructed_image[pixel_x, pixel_y] += sinogram[line_idx][angle_idx]
                pixel_idx += 1
            angle_idx += 1
        line_idx += 1

        # save the reconstructed image at the specified iteration
        # if iterations and len(copy_image_n_iteration) == iterations:
        #     save_image_at_iteration(np.flipud(reconstructed_image), iterations)
        #     return np.flipud(reconstructed_image), copy_image_n_iteration

        # append a copy of the current state of the reconstructed image to the copy_image_n_iteration list
        copy_image_n_iteration.append(np.copy(np.flipud(reconstructed_image)))

    # finally reverse an image
    reversed_image = np.flipud(reconstructed_image)

    return reversed_image, copy_image_n_iteration


def filter_sinogram(sinogram, mask_size=11):
    # make a copy of the input sinogram array
    sinogram_filtered = np.copy(sinogram)

    # calculate the index of the middle element of the mask
    element_in_the_middle = (mask_size - 1) // 2

    # create an array of zeros with dimensions equal to the mask size
    root = np.zeros((mask_size, mask_size))

    # set the middle element to 1
    root[element_in_the_middle, element_in_the_middle] = 1

    # loop over half the mask size, starting from 1
    for i in range(1, element_in_the_middle + 1):
        # if i is odd, set the corresponding elements in the root array
        """
        The if i % 2 == 1 statement is used to create a specific pattern in the filter root. It sets certain values 
        in the root to a specific value, while leaving others at 0. This pattern is used to create a specific shape
        of the root, which is used to apply a specific type of filter to the sinogram. This way we remove certain types
        of noise from the sinogram.
        """
        if i % 2 == 1:
            root[element_in_the_middle + i, element_in_the_middle] = root[element_in_the_middle - i,
            element_in_the_middle] = (-4 / np.pi ** 2) / (
                    i ** 2)

    # applying a convolution operation on each row of the sinogram array using the root
    sinogram_filtered = np.apply_along_axis(lambda x: np.convolve(x, root[:, element_in_the_middle], mode='same'),
                                            axis=1, arr=sinogram_filtered)

    return sinogram_filtered


def mean_square_error(image_original, image_filtered):
    assert image_original.shape == image_filtered.shape, "Images have different shapes!"

    # Find the difference between the original and filtered images
    diff = image_original - image_filtered

    # Square the difference and calculate the mean
    mse = np.mean(np.square(diff))

    # Calculate the square root of the mean squared error (MSE) to get the root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    return rmse


if __name__ == '__main__':
    image = 'Images\kolko.png'

    # read an image in grayscale mode
    image_read = cv2.imread(image, 0)

    width, height = np.shape(image_read)
    print("Width {w}\t Height {h}".format(w=width, h=height))

    sinogram_creation(image_read, width, height, 1, 220, 220, additive_or_substractive=True)
