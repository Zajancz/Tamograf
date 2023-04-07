import matplotlib.pyplot as plt
import numpy as np
import cv2


def generate_points_between_2(coordinates, width):
    """
    Generate a list of points for a line given two points and a width.

    Args:
        coordinates (list): A list of four elements containing the x and y coordinates
            of two points in the format [x1, y1, x2, y2].
        width (int): The width of the line in pixels.

    Returns:
        list: A list of points in the format [(x1, y1), (x2, y2), ...] that make
            up the line with the given width.
    """
    x1, y1, x2, y2 = map(int, coordinates)
    abs_x, abs_y = np.abs([x2 - x1, y2 - y1])
    dif_x, dif_y = np.sign([x2 - x1, y2 - y1])
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
    result = [(x1, y1)]
    e = abs_x - abs_y
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

    Args:
        angle (float): The angle in radians of the points relative to the positive x-axis.
        bias (float): The distance from the center of the circle to the line connecting the points.
        width (float): The diameter of the circle.

    Returns:
        List[float]: A list of four floats representing the x and y coordinates of the two points,
        in the order (x1, y1, x2, y2).
    """
    radius = width / 2
    x_coordinate = bias * np.cos(angle + np.pi / 2)
    y_coordinate = bias * np.sin(angle + np.pi / 2)
    return [
        radius * np.cos(angle) + x_coordinate + radius,
        radius * np.sin(angle) + y_coordinate + radius,
        -radius * np.cos(angle) + x_coordinate + radius,
        -radius * np.sin(angle) + y_coordinate + radius,
    ]


def value_of_line(line, width, height, image_read, additive_or_substractive):
    """
    Calculate the value of a single line on an image.

    Args:\n
    line (list): A list of tuples representing the coordinates of the points
    that make up the line.\n
    width (int): The width of the image.\n
    height (int): The height of the image.\n
    image_read (np.ndarray): A numpy array representing the image.\n
    additive_or_substractive (bool): A boolean value indicating whether to use an experimental
    method of calculating the line value.\n

    Returns:
    float or np.ndarray: The calculated value of the line.
    """
    result = []
    i = 0
    while i < len(line):
        x, y = line[i]
        if 0 <= x < width and 0 <= y < height:
            result.append(image_read[y, x])
        i += 1

    if additive_or_substractive:
        if len(line) > 0:
            return np.exp(-(sum(result) / (255 * len(line))))
        else:
            return np.exp(-0)
    else:
        return np.array(result, dtype=np.float32) if len(result) > 0 else np.zeros(1, dtype=np.float32)


def sinogram_creation(image_read, sinogram_image, width, height):
    """
    Create a sinogram matrix from an image.

    Args:
    image_read (np.ndarray): A numpy array representing the image.
    sinogram_image (np.ndarray): A numpy array representing the sinogram image.
    width (int): The width of the image.
    height (int): The height of the image.

    Returns:
    np.ndarray: A numpy array representing the sinogram image.
    """
    if sinogram_image is None:
        if image_read is None:
            raise Exception("Sorry, the picture wasn`t loaded")
        additive_or_substractive = True
        sinogram_array, lines_array = [], []
        angle = 1
        angulation = np.linspace(0, 360 - angle, num=360 // angle)
        print(angulation)
        tmp_for_biases = None
        tmp_for_emitters = None

        if tmp_for_biases is not None:
            space_biases = tmp_for_biases
        else:
            space_biases = image_read.shape[0]

        if tmp_for_emitters is not None:
            space_emitters = tmp_for_emitters
        else:
            space_emitters = image_read.shape[0]

        # evenly spaced numbers
        biases = np.arange(-space_biases / 2, space_biases / 2, space_biases / space_emitters)
        print(biases)
        i = 0
        while i < len(angulation):
            angle = angulation[i] * np.pi / 180
            sinogram_array_row, lines_array_row = [], []
            j = 0
            while j < len(biases):
                bias = biases[j]
                points = check_points_coordinates(angle, bias, width)
                line = generate_points_between_2(points, width)

                color = value_of_line(line, width, height, image_read, additive_or_substractive)
                sinogram_array_row.append(color)
                lines_array_row.append(line)
                j += 1
            sinogram_array.append(sinogram_array_row)
            lines_array.append(lines_array_row)
            i += 1

        if additive_or_substractive:
            # Scale the sin array to the range [0, 255]
            sinogram_array = (sinogram_array - np.min(sinogram_array)) * (
                        256 / (np.max(sinogram_array) - np.min(sinogram_array)))
            # Invert the values in the sin array
            sinogram_array = 255 - sinogram_array

        print(sinogram_array)
        reversed_array = sinogram_array[::-1]
        sinogram_image = np.array(reversed_array)
        plt.subplot(122)
        plt.imshow(sinogram_image, cmap='gray', aspect='0.3')
        plt.title('Sinogram')
        plt.show()

        return print(sinogram_image)
    print("For future use")


if __name__ == '__main__':
    image = 'Images\kolko.png'
    sinogram_image = None
    # read an image in grayscale mode
    image_read = cv2.imread(image, 0)

    plt.subplot(121)
    plt.imshow(image_read, cmap='gray')
    plt.title('Input Image')

    width, height = np.shape(image_read)
    print("Width:{w}\t Height:{h}".format(w=width, h=height))

    if width != height:
        raise Exception("Sorry, the picture isn`t squared")

    sinogram_creation(image_read, sinogram_image, width, height)
