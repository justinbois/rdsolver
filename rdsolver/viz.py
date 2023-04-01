import itertools
import warnings

import numpy as np
import scipy.interpolate

import bokeh.io
import bokeh.layouts
import bokeh.models
import bokeh.plotting
import bokeh.application
import bokeh.application.handlers


def display(time_points, c, frame_height=400, pyapp=False):
    """
    Display the solution to a 2D RD system

    Parameters
    ----------
    time_points : ndarray
        Time points where concentrations were sampled.
    c : ndarray
        Output of rd.solve(), a 4D array.
        Index 0: Species
        Index 1: x-coordinate
        Index 2: y-coordinate
        Index 3: time coodinate
    frame_height : int, default 400
        Height of plot, in pixels.
    pyapp : bool, default False
        If True, return an instance of a Bokeh app that uses Python
        to access the entries in the array c. This will reduce the
        size of a notebook you may embed the plot in. If False, then
        the display is rendered with a slider where all images are
        preloaded and accessed with JavaScript callbacks.

    Returns
    -------
    output : Either Bokeh app or Bokeh layout
        To display the output, use `bokeh.io.show(output)`.
    """
    if pyapp:
        return _display_app(time_points, c, frame_height=frame_height)
    else:
        return _display_js(time_points, c, frame_height=frame_height)


def _display_js(time_points, c, frame_height=400):
    # If a single image, convert
    if len(c.shape) == 3:
        c = c.reshape((1, *c.shape))
    if len(c.shape) != 4:
        raise RuntimeError("c must be n_species x nx x ny x n_time_points array.")

    # Make sure number of time points matches concentration dimensions
    if c.shape[3] != len(time_points):
        raise RuntimeError("Number of time points must equal c.shape[3].")

    # Determine maximal and minimal concentrations in each channel
    c_max = c.max(axis=(1, 2, 3))
    c_min = c.min(axis=(1, 2, 3))

    # Add a dummy yellow channel if we have two species for convenience
    if len(c_max) == 2:
        c_max = np.concatenate((c_max, (0.0,)))
        c_min = np.concatenate((c_min, (0.0,)))

    # Get shape of domain
    n, m = c.shape[1:3]

    # Set up figure with appropriate dimensions
    frame_width = int(m / n * frame_height)

    # Build the plot
    p = bokeh.plotting.figure(
        frame_height=frame_height,
        frame_width=frame_width,
        x_range=[0, m],
        y_range=[0, n],
    )

    # Build the images to display
    if c.shape[0] == 1:
        ims = np.moveaxis(c[0, :, :, :], [0, 1, 2], [1, 2, 0])
        color = bokeh.models.LinearColorMapper(
            bokeh.palettes.viridis(256), low=c_min[0], high=c_max[0]
        )
        disp_fun = p.image
        disp_kwargs = dict(color_mapper=color)
    else:
        ims = [
            _make_cmy_image(c[:, :, :, i], *c_max, *c_min) for i in range(c.shape[3])
        ]
        color = None
        disp_fun = p.image_rgba
        disp_kwargs = dict()

    cds = bokeh.models.ColumnDataSource(dict(image=[ims[0]]))
    disp_fun(image="image", x=0, y=0, dw=m, dh=n, source=cds, **disp_kwargs)

    slider_format = bokeh.models.CustomJSTickFormatter(
        args=dict(time_points=time_points),
        code="return time_points[Math.round(tick)].toFixed(3)",
    )

    slider = bokeh.models.Slider(
        start=0,
        end=len(time_points) - 1,
        value=0,
        step=1,
        title="time",
        format=slider_format,
    )

    callback = bokeh.models.CustomJS(
        args=dict(cds=cds, slider=slider, ims=ims, m=m, n=n),
        code="""
    // Extract data from source and sliders
    let im = cds.data['image'][0];
    let new_im = ims[slider.value];

    for (let i = 0; i < n * m; i++) im[i] = new_im[i];

    cds.change.emit();
    """,
    )

    slider.js_on_change("value", callback)

    layout = bokeh.layouts.column(
        bokeh.layouts.row(bokeh.layouts.Spacer(width=10), slider), p
    )

    return layout


def _display_app(time_points, c, frame_height=400):
    """
    Build display of results of RD simulation.

    Parameters
    ----------
    time_points : ndarray
        Time points where concentrations were sampled.
    c : ndarray
        Output of rd.solve(), a 4D array.
        Index 0: Species
        Index 1: x-coordinate
        Index 2: y-coordinate
        Index 3: time coodinate
    frame_height : int, default 400
        Height of plot, in pixels.

    Notes
    -----
    .. To display in a notebook hosted, e.g., at `localhost:8888`, do
       `bokeh.io.show(display_notebook(time_points, c),
                      notebook_url='localhost:8888')`
    """
    # If a single image, convert
    if len(c.shape) == 3:
        c = c.reshape((1, *c.shape))
    if len(c.shape) != 4:
        raise RuntimeError("c must be n_species x nx x ny x n_time_points array.")

    # Make sure number of time points matches concentration dimensions
    if c.shape[3] != len(time_points):
        raise RuntimeError("Number of time points must equal c.shape[3].")

    # Determine maximal and minimal concentrations in each channel
    c_max = c.max(axis=(1, 2, 3))
    c_min = c.min(axis=(1, 2, 3))

    # Add a dummy yellow channel if we have two species for convenience
    if len(c_max) == 2:
        c_max = np.concatenate((c_max, (0.0,)))
        c_min = np.concatenate((c_min, (0.0,)))

    # Get shape of domain
    n, m = c.shape[1:3]

    # Set up figure with appropriate dimensions
    frame_width = int(m / n * frame_height)

    p = bokeh.plotting.figure(
        frame_height=frame_height,
        frame_width=frame_width,
        x_range=[0, m],
        y_range=[0, n],
    )

    # Add the image to the plot
    if c.shape[0] == 1:
        color = bokeh.models.LinearColorMapper(
            bokeh.palettes.viridis(256), low=c_min[0], high=c_max[0]
        )
        source = bokeh.models.ColumnDataSource(data={"image": [c[0, :, :, 0]]})
        p.image(image="image", x=0, y=0, dw=m, dh=n, source=source, color_mapper=color)
    else:
        im_disp = _make_cmy_image(c[:, :, :, 0], *c_max, *c_min)
        source = bokeh.models.ColumnDataSource(data={"image": [im_disp]})
        p.image_rgba(image="image", x=0, y=0, dw=m, dh=n, source=source)

    def _callback(attr, old, new):
        i = np.searchsorted(time_points, slider.value)

        if c.shape[0] == 1:
            im_disp = c[0, :, :, i]
        else:
            im_disp = _make_cmy_image(c[:, :, :, i], *c_max, *c_min)

        source.data = {"image": [im_disp]}

    slider = bokeh.models.Slider(
        start=time_points[0],
        end=time_points[-1],
        value=time_points[0],
        step=1 / len(time_points) * (time_points[1] - time_points[0]),
        title="time",
    )
    slider.on_change("value", _callback)

    layout = bokeh.layouts.column(
        bokeh.layouts.row(bokeh.layouts.Spacer(width=10), slider), p
    )

    def _app(doc):
        doc.add_root(layout)

    return _app


def display_single_frame(c, i=-1, frame_height=400):
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
    frame_height : int, default 400
        Height of plot, in pixels.
    notebook : bool, default True
        If True, display in notebook. Otherwise, create and display

    Returns
    -------
    output : bokeh plotting object
        The plot.
    """

    if len(c.shape) == 2:
        c = c.reshape((1, *c.shape, 1))
        if i not in [0, -1]:
            raise RuntimeError("For single image, cannot specify i.")
    elif len(c.shape) == 3:
        c = c.reshape((1, *c.shape))
        if i not in [0, -1]:
            raise RuntimeError("For single image, cannot specify i.")
    if len(c.shape) != 4:
        raise RuntimeError("c must be n_species x nx x ny x n_time_points array.")

    # Get shape
    n, m = c.shape[1:3]

    # Set up figure with appropriate dimensions
    frame_height = frame_height
    frame_width = int(m / n * frame_height)
    p = bokeh.plotting.figure(
        frame_height=frame_height,
        frame_width=frame_width,
        x_range=[0, m],
        y_range=[0, n],
        tools="pan,box_zoom,wheel_zoom,save,reset",
    )

    # If single channel, display with viridis
    if c.shape[0] == 1:
        color = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(256))
        p.image(image=[c[0, :, :, i]], x=0, y=0, dw=m, dh=n, color_mapper=color)
    else:
        p.image_rgba(image=[_make_cmy_image(c[:, :, :, i])], x=0, y=0, dw=m, dh=n)

    return p


def _make_cmy_image(
    c,
    im_cyan_max=None,
    im_mag_max=None,
    im_yell_max=None,
    im_cyan_min=None,
    im_mag_min=None,
    im_yell_min=None,
):
    """
    Make an RGBA CMY image from a concentration field.
    """

    if c.shape[0] == 2:
        im = _im_merge_cmy(
            c[0, :, :],
            c[1, :, :],
            None,
            im_cyan_max,
            im_mag_max,
            im_yell_max,
            im_cyan_min,
            im_mag_min,
            im_yell_min,
        )
    elif c.shape[0] == 3:
        im = _im_merge_cmy(
            c[0, :, :],
            c[1, :, :],
            c[2, :, :],
            im_cyan_max,
            im_mag_max,
            im_yell_max,
            im_cyan_min,
            im_mag_min,
            im_yell_min,
        )
    else:
        raise RuntimeError("Too many channels. Select up to three to display.")

    return _rgb_to_rgba32(im)


def _im_merge_cmy(
    im_cyan,
    im_mag,
    im_yell=None,
    im_cyan_max=None,
    im_mag_max=None,
    im_yell_max=None,
    im_cyan_min=None,
    im_mag_min=None,
    im_yell_min=None,
):
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
    im_cyan_min : float, default min of inputed cyan channel
        Maximum value to use when scaling the cyan channel
    im_mag_min : float, default min of inputed magenta channel
        Minimum value to use when scaling the magenta channel
    im_yell_min : float, default min of inputed yellow channel
        Minimum value to use when scaling the yellow channel

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
    if im_yell is not None and im_yell_max is None:
        im_yell_max = im_yell.max()

    # Compute min intensities if needed
    if im_cyan_min is None:
        im_cyan_min = im_cyan.min()
    if im_mag_min is None:
        im_mag_min = im_mag.min()
    if im_yell is not None and im_yell_min is None:
        im_yell_min = im_yell.min()

    # Make sure maxes are ok
    if (
        im_cyan_max < im_cyan.max()
        or im_mag_max < im_mag.max()
        or (im_yell is not None and im_yell_max < im_yell.max())
    ):
        raise RuntimeError("Inputted max of channel < max of inputted channel.")

    # Make sure mins are ok
    if (
        im_cyan_min > im_cyan.min()
        or im_mag_min > im_mag.min()
        or (im_yell is not None and im_yell_min > im_yell.min())
    ):
        raise RuntimeError("Inputted min of channel > min of inputted channel.")

    # Scale the images
    im_c = (im_cyan - im_cyan_min) / (im_cyan_max - im_cyan_min)
    im_m = (im_mag - im_mag_min) / (im_mag_max - im_mag_min)
    if im_yell is None:
        im_y = np.zeros_like(im_cyan)
    else:
        im_y = (im_yell - im_yell_min) / (im_yell_max - im_yell_min)

    # Convert images to RGB with magenta, cyan, and yellow channels
    im_c = np.stack((np.zeros_like(im_c), im_c, im_c), axis=2)
    im_m = np.stack((im_m, np.zeros_like(im_m), im_m), axis=2)
    im_y = np.stack((im_y, im_y, np.zeros_like(im_y)), axis=2)
    im_rgb = im_c + im_m + im_y
    for i in [0, 1, 2]:
        im_rgb[:, :, i] /= im_rgb[:, :, i].max()

    return im_rgb


def _rgb_to_rgba32(im):
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
    if im.ndim != 3 or im.shape[2] != 3:
        raise RuntimeError("Input image is not RGB.")

    # Make sure all entries between zero and one
    if (im < 0).any() or (im > 1).any():
        raise RuntimeError("All pixel values must be between 0 and 1.")

    # Get image shape
    n, m, _ = im.shape

    # Convert to 8-bit, which is expected for viewing
    im_8 = np.round(im * 255).astype(np.uint8)

    # Add the alpha channel, which is expected by Bokeh
    im_rgba = np.stack(
        (*np.rollaxis(im_8, 2), 255 * np.ones((n, m), dtype=np.uint8)), axis=2
    )

    # Reshape into 32 bit. Must flip up/down for proper orientation
    return np.flipud(im_rgba.view(dtype=np.uint32).reshape((n, m)))


def _interpolate_2d(a, n_interp_points=(200, 200)):
    """
    Interplate a 2D array.

    Parameters
    ----------
    a : 2D ndarray
        Array to be interplated
    n_interp_points : 2-tuple of ints
        Number of interpolation points in the row and column
        dimension, respectively.

    Returns
    -------
    output : ndarray, shape n_interp_points
        Interpolated array.
    """

    # Set up grids
    x = np.arange(a.shape[0])
    y = np.arange(a.shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    x_interp = np.linspace(x[0], x[-1], n_interp_points[0])
    y_interp = np.linspace(y[0], y[-1], n_interp_points[1])
    xx_interp, yy_interp = np.meshgrid(x_interp, y_interp, indexing="ij")

    # Perform B-spline interpolation
    spline = scipy.interpolate.RectBivariateSpline(x, y, a, s=0)
    a_flat = spline.ev(xx_interp.flatten(), yy_interp.flatten())
    return a_flat.reshape(n_interp_points)


def interpolate_concs(c, n_interp_points=(200, 200)):
    """
    Performs interpolation of all concentration fields.

    Parameters
    ----------
    c : 4D ndarray
        Solution of RD equations.
    n_interp_points : 2-tuple of ints
        Number of interpolation points in the row and column
        dimension, respectively.

    Returns
    -------
    output : ndarray, shape (c.shape[0], *n_interp_points, c.shape[3])
        Interpolated concentrations.
    """
    # Set up output array
    c_interp = np.empty((c.shape[0], *n_interp_points, c.shape[-1]))

    # Interpolate each species for each time point.
    for i in range(c.shape[-1]):
        for j in range(c.shape[0]):
            c_interp[j, :, :, i] = _interpolate_2d(c[j, :, :, i], n_interp_points)

    return c_interp
