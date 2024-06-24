# PIV
#Particle image velocimetry
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim

# Load the two images
frame_a = imread('E005_1.tif',as_gray=True)
frame_b = imread('E005_2.tif',as_gray=True)


win_size = 128
print(f"{win_size=}")

# Calculate the mean squared error between the two windows
a_win = frame_a[:win_size, :win_size].copy()
b_win = frame_b[:win_size, :win_size].copy()

mse = np.mean((a_win - b_win) ** 2)

# Print the mean squared error
print("Mean Squared Error: {:.2f}".format(mse))

# Display the two windows side by side
plt.subplot(232)
plt.imshow(a_win, cmap='gray')
plt.title('Window from frame A')

plt.subplot(231)
plt.imshow(b_win, cmap='gray')
plt.title('Window from frame B')


mse = np.mean((frame_a - frame_b) ** 2)

# Print the result
print(f"The mean squared error between the two frames is: {mse}")

without_shift = b_win - a_win

shifted = b_win - np.roll(a_win, (5, 0), axis=(0, 1))


shifted_a = np.roll(a_win, (10, 0), axis=(0, 1))
plt.subplot(233)
plt.imshow(shifted_a, cmap=plt.cm.gray)
plt.plot("Shifted A")
plt.subplot(234)
plt.imshow(without_shift, cmap=plt.cm.gray)
plt.plot("Without_shift")
plt.subplot(235)
plt.imshow(shifted, cmap=plt.cm.gray)
plt.plot("With_shift")
# This function calculates the correlation between two 2D arrays, x and y, using the Fast Fourier Transform (FFT) method.
import numpy as np

def correlate(x, y, method='fft'):
    """
    Calculate the correlation between two arrays `x` and `y`.

    Parameters
    ----------
    x : array_like
        The first array to correlate. Can be 1D or 2D.
    y : array_like
        The second array to correlate. Must have the same number of dimensions as `x`.
    method : str, optional
        The correlation method to use. Can be either 'fft' or 'direct'. Default is 'fft'.

    Returns
    -------
    corr : ndarray
        The correlation between `x` and `y`. Returns both the real and imaginary parts.
    """

    # Check if the specified method is supported
    if method not in ['fft', 'direct']:
        raise ValueError("Method not supported. Must be either 'fft' or 'direct'.")

    # Check if `x` and `y` have the same dimensions
    if x.ndim != y.ndim:
        raise ValueError("Arrays must have the same number of dimensions.")

    # Compute the size and shape of the output array
    if x.ndim == 1:
        size = x.size + y.size - 1
        shape = (size,)
    else:
        size = (x.shape[0] + y.shape[0] - 1, x.shape[1] + y.shape[1] - 1)
        shape = size

    # Initialize the output array with zeros
    corr = np.zeros(shape, dtype=np.complex128)

    # Calculate the correlation using the specified method
    if method == 'fft':
        # Compute the 2D FFT of x and y
        f = np.fft.fft2(x)
        g = np.fft.fft2(np.flipud(np.fliplr(y)))
        # Calculate the correlation between x and y using the convolution theorem
        # and return the real and imaginary parts
        
        corr = np.fft.ifft2(f * g)
    else:
        # Calculate the correlation using the direct definition
        if x.ndim == 1:
            for i in range(size):
                corr[i] = np.sum(x * y[i:size-i])
        else:
            for i in range(size[0]):
                for j in range(size[1]):
                    corr[i, j] = np.sum(x * y[i:, j:][:size[0]-i, :size[1]-j])

    # Return the real and imaginary parts of the correlation
    return corr.real

cross_corr = correlate(b_win - b_win.mean(), a_win - a_win.mean(), method="fft")

# print(cross_corr[:,:,3])

# Find the peak in the cross-correlation map
max_val = cross_corr.max()
max_idx = np.argmax(cross_corr)
y, x = np.unravel_index(max_idx, cross_corr.shape)
print(max_val,x,y)

# # Calculate the shift from the peak position
best_shift = (y - cross_corr.shape[0] // 2, x - cross_corr.shape[1] // 2)

# # Calculate the distance from the peak position
best_dist = np.sqrt(np.sum((b_win - b_win.mean()) ** 2))

# Plot the shifted window and the template
plt.subplot(236)
plt.imshow(np.roll(a_win, best_shift, axis=(0, 1)), cmap='gray')
plt.imshow(b_win, cmap='gray')

# Plot the correlation map
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
Y, X = np.meshgrid(np.arange(cross_corr.shape[0]), np.arange(cross_corr.shape[1]))
ax.plot_surface(Y, X, cross_corr, cmap='jet', linewidth=0.2) # type: ignore
plt.title("Correlation map -- peak is the most probable shift")

    

# Print the result
print(f"Best distance: {best_dist}")
print(f"Best shift: {best_shift}")
# ===============================================



# getting the velocity field
for i in range(1,7):
    # Load the two images
    print("===")
    print('image/B{0:03d}_1.tif'.format(i))
    print('image/B{0:03d}_2.tif\n === '.format(i))
    # frame_a = imread('frames/frame_{0:02d}_delay-0.05s.tiff'.format(i),as_gray=True)
    # frame_b = imread('frames/frame_{0:02d}_delay-0.05s.tiff'.format(i+1),as_gray=True)
    frame_a = imread('image/B{0:03d}_1.tif'.format(i),as_gray=True)
    frame_b = imread('image/B{0:03d}_2.tif'.format(i),as_gray=True)
    from scipy.signal import correlate

    def velocityField(initialFrame,NextFrame,win_size = 32):
        '''
        Calculates and returns the velocity field 
        of all the particles in the given image.
        Each vector in the velocity field is the mean velocity 
        of the particles in a section of given win_size of the image 
        '''
        posy = np.arange(0,initialFrame.shape[0],win_size)  
        posx = np.arange(0,initialFrame.shape[1],win_size)   
        velx = np.zeros((len(posy),len(posx)))   
        vely = np.zeros((len(posy),len(posx)))   
        for iy,y in enumerate(posy):
            for ix,x in enumerate(posx):
                initial_small_win = initialFrame[y : y + win_size, x : x + win_size]
                final_small_win = NextFrame[y : y + win_size, x : x + win_size]
                cross_corr = correlate(final_small_win - final_small_win.mean(),initial_small_win - initial_small_win.mean(),
                                    method="fft")
                max_ind = np.unravel_index(np.argmax(cross_corr),cross_corr.shape)
                vely[iy,ix], velx[iy,ix] = max_ind - np.array([win_size,win_size]) + 1
                
        # the starting position of each velocity vector will be from center of the given screen
        posx += win_size//2
        posy += win_size//2
        return posx,posy,velx,vely

    posx,posy,velx,vely = velocityField(frame_a,frame_b,32)
    norm_of_velocity = np.sqrt(velx ** 2 + vely ** 2)
    # final_velocity = np.array([np.sum(velx),np.sum(vely)])
    # print(final_velocity)


    fig,ax = plt.subplots(figsize = (6,6))
    # plt.gca().invert_yaxis()
    ax.quiver( posx, posy[::-1],velx,-vely,norm_of_velocity,cmap="plasma",angles = "xy",scale_units="xy")
    ax.set_title("Image B{0:03d}_1.tif and B{0:03d}_1.tif".format(i,i))
    ax.set_aspect("equal")
    # plt.show()
    plt.show(block = False)
    plt.pause(3)
    plt.close()
    


