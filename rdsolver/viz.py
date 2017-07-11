import itertools
import warnings

import numpy as np

import skimage

import bokeh.io
import bokeh.layouts
import bokeh.models
import bokeh.plotting

import ipywidgets


def display_notebook(c, time_points, plot_height=400):
    """
    Build display of results of RD simulation.

    Parameters
    ----------
    c : ndarray
        Output of rd.solve(), a 4D array.
        Index 0: Species
        Index 1: x-coordinate
        Index 2: y-coordinate
        Index 3: time coodinate
    time_points : ndarray
        Timepoints
    plot_height : int, default 400
        Height of plot, in pixels.

    Notes
    -----
    .. *ONLY* works in Jupyter notebooks.

    .. Your browswer might have problems displaying the data.
       If this is the case, you might want to try launching Jupyter
       as jupyter notebook --browser=safari --NotebookApp.iopub_data_rate_limit=10000000
    """

    # If a single image, convert
    if len(c.shape) == 3:
        c = c.reshape((1, *c.shape))
    if len(c.shape) != 4:
        raise RuntimeError(
                'c must be n_species x nx x ny x n_time_points array.')

    # Make sure number of time points matches concentration dimensions
    if c.shape[3] != len(time_points):
        raise RuntimeError('Number of time points must equal c.shape[3].')

    # Set any negative concentrations to zero
    if (c < 0).any():
        warnings.warn('Negative concentrations detected: setting to zero.',
                      RuntimeWarning)
        c[c<0] = 0.0

    # Determine maximal concentration in each channel
    c_max = c.max(axis=(1, 2, 3))

    # Get shape of domain
    n, m = c.shape[1:3]

    # Set up figure with appropriate dimensions
    plot_width = int(m/n * plot_height)
    p = bokeh.plotting.figure(plot_height=plot_height, plot_width=plot_width,
                              x_range=[0, m], y_range=[0, n],
                              tools='pan,box_zoom,wheel_zoom,reset,resize')

    # If single channel, display with viridis
    if c.shape[0] == 1:
        color = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(256),
                                               low=0, high=c_max[0])
        im_bokeh = p.image(image=[c[0,:,:,0]], x=0, y=0, dw=m, dh=n,
                           color_mapper=color)
    else:
        im_disp = make_cmy_image(c[:,:,:,0], *c_max)
        im_bokeh = p.image_rgba(image=[im_disp], x=0, y=0, dw=m, dh=n)

    # Show the plot
    bokeh.io.show(p, notebook_handle=True)

    def update(time=0.0):
        ind = np.searchsorted(time_points, time)

        if c.shape[0] == 1:
            im_bokeh.data_source.data['image'] = [c[0,:,:,ind]]
        else:
            im_disp = make_cmy_image(c[:,:,:,ind], *c_max)
            im_bokeh.data_source.data['image'] = [im_disp]
        bokeh.io.push_notebook()

    ipywidgets.interact(update,
        time=ipywidgets.FloatSlider(min=time_points[0], max=time_points[-1],
                                    value=time_points[0]))


def display_single_frame(c, i=-1, plot_height=400, notebook=True,
                         output_file='rd_result.html', title='RD result',
                         browser=None):
    """
    Display the concentration field of a single time point.

    Parameters
    ----------
    c : ndarray
        Output of rd.solve(), a 4D array.
        Index 0: Species
        Index 1: x-coordinate
        Index 2: y-coordinate
        Index 3: time coodinate
    i : int, default -1
        Index of time point you want displayed.
    plot_height : int, default 400
        Height of plot, in pixels.
    notebook : bool, default True
        If True, display in notebook. Otherwise, create and display
        as HTML in a new tab in the browser.
    output_file : str, default 'rd_result.html'
        Name of output file for HTML. Ignored if `notebook` is True.
    title : str, default 'RD result'
        Title of HTML page. Ignored is `notebook` is True.
    browser : str, default None
        Name of browser to use to display HTML. Ignored if `notebook`
        is True.

    Returns
    -------
    output : bokeh plotting object
        The plot.
    """

    if len(c.shape) == 3:
        c = c.reshape((1, *c.shape))
    if len(c.shape) != 4:
        raise RuntimeError(
                'c must be n_species x nx x ny x n_time_points array.')

    # Get shape
    n, m = c.shape[1:3]

    # Set up figure with appropriate dimensions
    plot_height = plot_height
    plot_width = int(m/n * plot_height)
    p = bokeh.plotting.figure(plot_height=plot_height, plot_width=plot_width,
                              x_range=[0, m], y_range=[0, n],
                              tools='pan,box_zoom,wheel_zoom,reset,resize')

    # If single channel, display with viridis
    if c.shape[0] == 1:
        color = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(256))
        im_bokeh = p.image(image=[c[:,:,:,i]], x=0, y=0, dw=m, dh=n,
                           color_mapper=color)
    else:
        im_disp = make_cmy_image(c[:,:,:,i])
        im_bokeh = p.image_rgba(image=[im_disp], x=0, y=0, dw=m, dh=n)

    if notebook:
        bokeh.io.output_notebook()
        bokeh.io.show(p)
    else:
        bokeh.io.output_file(output_file, title=title)
        bokeh.io.show(p, browser=browser)

    return p


def make_cmy_image(c, im_cyan_max=None, im_mag_max=None, im_yell_max=None):
    """
    Make an RGBA CMY image from a concentration field.
    """
    if c.shape[0] == 2:
        im = im_merge_cmy(c[0,:,:], c[1,:,:], None, im_cyan_max, im_mag_max,
                          im_yell_max)
    elif c.shape[0] == 3:
        im = im_merge_cmy(c[0,:,:], c[1,:,:], c[2,:,:], im_cyan_max,
                          im_mag_max, im_yell_max)
    else:
        raise RuntimeError('Too many channels. Select up to three to display.')

    return rgb_to_rgba32(im)


def im_merge_cmy(im_cyan, im_mag, im_yell=None, im_cyan_max=None,
                 im_mag_max=None, im_yell_max=None):
    """
    Merge channels to make RGB image that has cyan, magenta, and
    yellow.
    Parameters
    ----------
    im_cyan: array_like
        Image represented in cyan channel.  Must be same shape
        as `im_magenta` and `im_yellow`.
    im_mag: array_like
        Image represented in magenta channel.  Must be same shape
        as `im_yellow` and `im_yellow`.
    im_yell: array_like, default None
        Image represented in yellow channel.  Must be same shape
        as `im_cyan` and `im_magenta`.
    im_cyan_max : float, default max of inputed cyan channel
        Maximum value to use when scaling the cyan channel
    im_mag_max : float, default max of inputed magenta channel
        Maximum value to use when scaling the magenta channel
    im_yell_max : float, default max of inputed yellow channel
        Maximum value to use when scaling the yellow channel

    Returns
    -------
    output : array_like, dtype float, shape (*im_cyan.shape, 3)
        RGB image the give CMY coloring of image
    """

    # Compute max intensities if needed
    if im_cyan_max is None:
        im_cyan_max = im_cyan.max()
    if im_mag_max is None:
        im_mag_max = im_mag.max()
    if im_yell_max is None and im_yell is not None:
        im_yell_max = im_yell.max()
    else:
        im_yell_max = 0

    # Make sure maxes are ok
    if im_cyan_max < im_cyan.max() or im_mag_max < im_mag.max() \
            or (im_yell is not None and im_yell_max < im_yell.max()):
        raise RuntimeError('Inputted max of channel > max of inputted channel.')

    if (im_cyan < 0).any() or (im_max < 0).any() \
            or (im_yell is not None and (im_yell < 0).any()):
        raise RuntimeError('Negative pixel values encountered.')

    # Scale the images
    im_c = im_cyan / im_cyan_max
    im_m = im_mag / im_mag_max
    if im_yell is None:
        im_y = np.zeros_like(im_cyan)
    else:
        im_y = im_yell / im_yell_max

    # Convert images to RGB with magenta, cyan, and yell channels
    im_rgb = np.empty((*im_c.shape, 3))
    im_rgb[:,:,0] = 1 - im_c
    im_rgb[:,:,1] = 1 - im_m
    im_rgb[:,:,2] = 1 - im_y

    return im_rgb


def rgb_to_rgba32(im):
    """
    Convert an RGB image to a 32 bit-encoded RGBA image.

    Parameters
    ----------
    im : ndarray, shape (nrows, ncolums, 3)
        Input image. All pixel values must be between 0 and 1.

    Returns
    -------
    output : ndarray, shape (nros, ncolumns), dtype np.uint32
        Image decoded as a 32 bit RBGA image.
    """
    # Ensure it has three channels
    if len(im.shape) != 3 or im.shape[2] !=3:
        raise RuntimeError('Input image is not RGB.')

    # Make sure all entries between zero and one
    if (im < 0).any() or (im > 1).any():
        raise RuntimeError('All pixel values must be between 0 and 1.')

    # Get image shape
    n, m, _ = im.shape

    # Convert to 8-bit, which is expected for viewing
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        im_8 = skimage.img_as_ubyte(im)

    # Add the alpha channel, which is expected by Bokeh
    im_rgba = np.stack((*np.swapaxes(im_8, 0, 2),
                       255*np.ones((n, m), dtype=np.uint8)), axis=2)

    # Reshape into 32 bit. Must flip up/down for proper orientation
    return np.flipud(im_rgba.view(dtype=np.int32).reshape((n, m)))
