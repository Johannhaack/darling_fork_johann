import time

import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.ndimage

import darling


class _Visualizer(object):
    # TODO: some of this should probably be in the properties module...

    def __init__(self, dset_reference):
        self.dset = dset_reference
        self.xlabel = "Detector row index"
        self.ylabel = "Detector column index"

    def mean(self):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 2, figsize=(9, 6), sharex=True, sharey=True)
        fig.suptitle(
            "Mean Map \nfirst moment around motor coordinates",
            fontsize=22,
        )

        im_ratio = self.dset.mean.shape[0] / self.dset.mean.shape[1]
        for i in range(2):
            im = ax[i].imshow(self.dset.mean[:, :, i], cmap="jet")
            fig.colorbar(im, ax=ax[i], fraction=0.046 * im_ratio, pad=0.04)
            ax[i].set_title(
                "Mean in motor " + self.dset.reader.scan_params["motor_names"][i],
                fontsize=14,
            )
            ax[i].set_xlabel(self.xlabel, fontsize=14)
            if i == 0:
                ax[i].set_ylabel(self.ylabel, fontsize=14)
        plt.tight_layout()
        return fig, ax

    def covariance(self, mask=None):
        """
        Plot the covariance matrix of the data set. Using RGBA colormap to plot the covariance matrix with transparency.

        Args:
            mask (:obj:`numpy array`): A binary mask with the same shape as the data set. If provided, the
                covariance matrix will be plotted where the mask = 1. Defaults to None.
        """
        plt.style.use("dark_background")
        fig, ax = plt.subplots(2, 2, figsize=(18, 18), sharex=True, sharey=True)
        fig.suptitle(
            "Covariance Map \nsecond moment around motor coordinates",
            fontsize=22,
        )
        im_ratio = self.dset.covariance.shape[0] / self.dset.covariance.shape[1]

        for i in range(2):
            for j in range(2):
                data = self.dset.covariance[:, :, i, j]

                _data = np.where(mask, data, np.nan) if mask is not None else data

                im = ax[i, j].imshow(_data, interpolation="nearest", cmap="magma")
                fig.colorbar(im, ax=ax[i, j], fraction=0.046 * im_ratio, pad=0.04)

                ax[i, j].set_title(
                    f"Covar[{self.dset.reader.scan_params['motor_names'][i]}, {self.dset.reader.scan_params['motor_names'][j]}]",
                    fontsize=8,
                )
                if j == 0:
                    ax[i, j].set_ylabel(self.ylabel, fontsize=14)
                if i == 1:
                    ax[i, j].set_xlabel(self.xlabel, fontsize=14)

        plt.tight_layout()
        return fig, ax

    def kam(self):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(9, 9), sharex=True, sharey=True)
        fig.suptitle(
            "(Projected) KAM Map \nlocal variation in orientation",
            fontsize=22,
        )
        if self.dset.kam is None:
            _ = self.dset.kernel_average_misorientation()
        kam = np.full_like(self.dset.kam, fill_value=np.nan)
        a, b = self.dset.kam_kernel_size
        kam[
            (a // 2) : -(a // 2),
            (b // 2) : -(b // 2),
        ] = self.dset.kam[
            (a // 2) : -(a // 2),
            (b // 2) : -(b // 2),
        ]
        im_ratio = kam.shape[0] / kam.shape[1]
        im = ax.imshow(
            kam,
            cmap="jet",
        )
        fig.colorbar(im, ax=ax, fraction=0.046 * im_ratio, pad=0.04)
        ax.set_xlabel(self.xlabel, fontsize=14)
        ax.set_ylabel(self.ylabel, fontsize=14)
        plt.tight_layout()
        return fig, ax

    def misorientation(self):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(9, 9), sharex=True, sharey=True)
        fig.suptitle(
            "Misorientation Map \nL2 norm of mean map after median subtraction",
            fontsize=22,
        )
        mean = self.dset.mean.copy()
        mean[:, :, 0] -= np.median(mean[:, :, 0].flatten())
        mean[:, :, 1] -= np.median(mean[:, :, 1].flatten())
        misori = np.linalg.norm(mean, axis=-1)
        im_ratio = misori.shape[0] / misori.shape[1]
        im = ax.imshow(misori, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046 * im_ratio, pad=0.04)
        ax.set_xlabel(self.xlabel, fontsize=14)
        ax.set_ylabel(self.ylabel, fontsize=14)
        plt.tight_layout()
        return fig, ax

    def mosaicity(self, norm="dynamic"):
        """
        Plot the mosaicity map. This takes the motor limits or data ranges for normalization.
        Sets the blue channel to make the mosaicity map more readable. The colormap is plotted
        on the right based on the selected scaling method.

        Args:
            use_motors (:obj:`bool`): If True, scales the mosaicity map using motor limits. If False, uses data ranges.
            mask (:obj:`numpy array`): A 2D binary mask with the same shape as the data set. If provided, it scales the mosaicity map
                where the mask == 1. Defaults to None.
        """
        rgbmap, colorkey, colorgrid = darling.properties.rgb(
            self.dset.mean, norm=norm, coordinates=self.dset.motors
        )

        plt.style.use("dark_background")
        fig, ax = plt.subplots(
            1, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [3, 1]}
        )
        fig.suptitle(
            "Mosaicity Map \n maps motors to a cylindrical HSV colorspace",
            fontsize=22,
        )

        ax[0].imshow(rgbmap)
        ax[0].set_xlabel(self.xlabel, fontsize=14)
        ax[0].set_ylabel(self.ylabel, fontsize=14)

        ax[1].pcolormesh(*colorgrid, colorkey, shading="auto")
        a = np.max(colorgrid[0]) - np.min(colorgrid[0])
        b = np.max(colorgrid[1]) - np.min(colorgrid[1])
        ax[1].set_aspect(a / b)

        ax[1].set_xlabel(self.dset.reader.scan_params["motor_names"][0], fontsize=14)
        ax[1].set_ylabel(self.dset.reader.scan_params["motor_names"][1], fontsize=14)
        ax[1].set_title(r"Color Map", fontsize=14)
        plt.tight_layout()
        return fig, ax


class DataSet(object):
    """A DFXM data-set.

    This is the master data class of darling. Given a reader the DataSet class will read data form
    arbitrary layers, process, threshold, compute moments, visualize results, and compile 3D feature maps.

    Args:
        reader (:obj: `darling.reader.Reader`): A file reader implementing, at least, the functionallity
            specified in darling.reader.Reader().

    Attributes:
        reader (:obj: `darling.reader.Reader`): A file reader implementing, at least, the functionallity
            specified in darling.reader.Reader().

    """

    def __init__(self, reader):
        self.reader = reader
        self.plot = _Visualizer(self)
        self.mean, self.covariance = None, None
        self.mean_3d, self.covariance_3d = None, None
        self.kam = None
        self.kam_kernel_size = None

    def load_scan(self, scan_id, roi=None):
        """Load a scan into RM.

        NOTE: Input args should match the darling.reader.Reader used, however it was implemented.

        Args:
            scan_id (:obj:`str`): scan id to load from, these are internal keys to diffirentiate
                layers.
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded. The roi refers to the detector
                dimensions.

        """
        self.data, self.motors = self.reader(scan_id, roi)

    def subtract(self, value):
        """Subtract a fixed integer value form the data. Protects against uint16 sign flips.

        Args:
            value (:obj:`int`): value to subtract.

        """
        self.data.clip(value, None, out=self.data)
        self.data -= value

    def estimate_background(self):
        """Automatic background correction based on image statistics.

        a set of sample data is extracted from the data block. The median and standard deviations are iteratively
        fitted, rejecting outliers (which here is diffraction signal). Once the noise distirbution has been established
        the value corresponding to the 99.99% percentile is returned. I.e the far tail of the noise is returned.

        """
        sample_size = 40000
        index = np.random.permutation(sample_size)
        sample = self.data.flat[index]
        sample = np.sort(sample)
        noise = sample.copy()
        for i in range(20):
            mu = np.median(noise)
            std = np.std(noise)
            noise = noise[np.abs(noise) < mu + 2 * 3.891 * std]  # 99.99% confidence
        background = np.max(noise)
        return background

    def moments(self):
        """Compute first and second moments.

        The internal attributes self.mean and self.covariance are set when this function is run.

        Returns:
            (:obj:`tupe` of :obj:`numpy array`): mean and covariance maps of shapes (a,b,2) and (a,b,2,2)
                respectively with a=self.data.shape[0] and b=self.data.shape[1].
        """
        self.mean, self.covariance = darling.properties.moments(self.data, self.motors)
        return self.mean, self.covariance

    def kernel_average_misorientation(self, size=(5, 5)):
        """Compute the KAM (Kernel Average Misorientation) map.

        KAM is compute by sliding a kernel across the image and for each voxel computing
        the average misorientation between the central voxel and the surrounding voxels.

        NOTE: This is a projected KAM in the sense that the rotation the full rotation
        matrix of the voxels are unknown. I.e this is a computation of the misorientation
        between diffraction vectors Q and not orientation elements of SO(3).

        Args:
            size (:obj:`tuple`): The size of the kernel to use for the KAM computation.
                Defaults to (3, 3).

        Returns:
            :obj:`numpy array` : The KAM map of shape=(a, b). (same units as input.)
        """
        self.kam = darling.properties.kam(self.mean, size)
        self.kam_kernel_size = size
        return self.kam

    def integrate(self, axis=None, dtype=np.float32):
        """Return the summed data stack along the specified axes, avoiding data stack copying.

        If no axis is specified, the integration is performed over all dimensions
        except the first two, which are assumed to be the detector dimensions.

        Args:
            axis (:obj:`int` or :obj:`tuple`, optional): The axis or axes along which to integrate.
                If None, integrates over all axes except the first two.
            dtype (:obj:`numpy.dtype`, optional): The data type of the output array.
                Defaults to np.float32.

        Returns:
            :obj:`numpy.ndarray`: Integrated frames, a 2D numpy array of reduced
                shape and dtype `dtype`.
        """
        if axis is None:
            out = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=dtype)
            integrated_frames = np.sum(
                self.data, axis=tuple(range(2, self.data.ndim)), out=out
            )
        else:
            shape = list(self.data.shape)
            for ax in sorted(np.atleast_1d(axis), reverse=True):
                shape.pop(ax)
            out = np.zeros(shape, dtype=dtype)
            integrated_frames = np.sum(self.data, axis=axis, out=out)
        return integrated_frames

    def estimate_mask(
        self,
        threshold=200,
        erosion_iterations=3,
        dilation_iterations=25,
        fill_holes=True,
    ):
        """Segment the sample diffracting region based on summed intensity along motor dimensions.

        Args:
            threshold (:obj:`int`):  a summed count value above which the sample is defined.
            erosion_iterations (:obj:`int`): Number of times to erode the mask using a 2,2 structure.
            dilation_iterations (:obj:`int`): Number of times to dilate the mask using a 2,2 structure.
            fill_holes (:obj:`bool`):  Fill enclosed holes in the final mask.

        Returns:
            (:obj:`numpy array`): Returns: a binary 2D mask of the sample.

        """
        mask = self.integrate() > threshold
        mask = scipy.ndimage.binary_erosion(
            mask, structure=np.ones((2, 2)), iterations=erosion_iterations
        )
        mask = scipy.ndimage.binary_dilation(
            mask, structure=np.ones((2, 2)), iterations=dilation_iterations
        )
        if fill_holes:
            mask = scipy.ndimage.binary_fill_holes(mask)
        return mask

    def compile_layers(self, scan_ids, threshold=None, roi=None, verbose=False):
        """Sequentially load a series of scans and assemble the 3D moment maps.

        this loads the mosa data array with shape a,b,m,n,(o) where a,b are the detector dimension and
        m,n,(o) are the motor dimensions as ordered in the `self.motor_names`.

        NOTE: This function will load data sequentially and compute moments on the fly. While all
        moment maps are stored and concatenated, only one scan (the raw 4d or 5d data) is kept in
        memory at a time to enhance RAM performance.

        Args:
            scan_ids (:obj:`str`): scan ids to load, e.g 1.1, 2.1 etc...
            threshold (:obj:`int` or :obj:`str`): background subtraction value or string 'auto' in which
                case a default background estimation is performed and subtracted.
                Defaults to None, in which case no background is subtracted.
            roi (:obj:`tuple` or :obj:`int`): row_min row_max and column_min and column_max, defaults to None,
                in which case all data is loaded
            verbose (:obj:`bool`): Print loading progress or not.

        """
        mean_3d = []
        covariance_3d = []
        tot_time = 0
        for i, scan_id in enumerate(scan_ids):
            t1 = time.perf_counter()

            if verbose:
                print(
                    "\nREADING SCAN: "
                    + str(i + 1)
                    + " out of totally "
                    + str(len(scan_ids))
                    + " scans"
                )
            self.load_scan(scan_id, roi)

            if threshold is not None:
                if threshold == "auto":
                    if verbose:
                        print(
                            "    Subtracting estimated background for scan id "
                            + str(scan_id)
                            + " ..."
                        )
                    _threshold = self.estimate_background()
                    self.threshold(_threshold)
                else:
                    if verbose:
                        print(
                            "    Subtracting fixed background value="
                            + str(threshold)
                            + " for scan id "
                            + str(scan_id)
                            + " ..."
                        )
                    self.threshold(threshold)

            if verbose:
                print("    Computing moments for scan id " + str(scan_id) + " ...")

            mean, covariance = self.moments()

            if verbose:
                print(
                    "    Concatenating to 3D volume for scan id "
                    + str(scan_id)
                    + " ..."
                )
            mean_3d.append(mean)
            covariance_3d.append(covariance)

            t2 = time.perf_counter()
            tot_time += t2 - t1

            estimated_time_left = str((tot_time / (i + 1)) * (len(scan_ids) - i - 1))
            if verbose:
                print("    Estimated time left is : " + estimated_time_left + " s")

        self.mean_3d = np.array(mean_3d)
        self.covariance_3d = np.array(covariance_3d)

        if verbose:
            print("\ndone! Total time was : " + str(tot_time) + " s")

        return self.mean_3d, self.covariance_3d
    
    def transform_to_global_coordinate_system(self, pixel_size = [1,1,1], scan_ids = None, translation_vector = None, rotation_vector = None, aspect = None):

        """This function takes the (x,y,z) postion [center positions] of the hexapod and the angular orientation mu, omega and center of chi,phi 
        and gives back the transformed coordinates space of the full field of view registered on the detector
        We expect that only the z position if multiple scan_ids are provided

        Args: 
            pixel_size (:obj:`list`): The pixel size of the detector in microns, defaults to [1,1,1] for both 2D and 3D Data.
            scan_ids (:obj:`list`): List of scan ids, if the coordinate system is a layer mosa scan, if scan ids we get the scan ids from a metadata function 
            translation_vector (:obj:`numpy.ndarray`): The translation vector of the hexapod in microns, defaults to None.
            rotation_vector (:obj:`numpy.ndarray`): The rotation vector of the hexapod in radians, defaults to None.
            aspect (:obj:`float`): The aspect ratio of the detector, defaults to None.

        Returns:
            :obj:`numpy.ndarray`: The transformed coordinates of the full field of view registered on the detector.

        """

        if translation_vector is not None and  type(translation_vector) != np.ndarray :
            try:
                translation_vector = np.array(translation_vector).astype(float)
            except:
                raise TypeError("translation vector should be a numpy array or a list of floats")

        if rotation_vector is not None and type(rotation_vector) != np.ndarray:
            try:
                rotation_vector = np.array(rotation_vector).astype(float)
            except:
                raise TypeError("translation vector should be a numpy array or a list of floats")
        if self.data is None:
            detector_size = 2048,2048
        else:
            detector_size = self.data.shape[0],self.data.shape[1]

        self.global_coord_system = darling.multigrain_visualizer.transform_to_global_coordinate_system(self.reader,detector_size, pixel_size, scan_ids, translation_vector, rotation_vector, aspect)

        return self.global_coord_system

    def to_paraview(self, file):
        """Write moment maps to paraview readable format for 3D visualisation.

        The written data array will have attributes as:

            cov_11, cov_12, (cov_13), cov_22, (cov_23, cov_33) : Elements of covariance matrix.
            mean_1, mean_2, (mean_3) : The first moments in each dimension.

        Here 1 signifies the self.motors[0] dimension while 2 is in self.motors[2], (and
        3 in self.motors[3], when the scan is 3D)

        NOTE: Requires that 3D moment maps have been compiled via compile_layers().

        Args:
            file (:obj:`string`): Absolute path ending with desired filename.

        """

        dim = np.array(self.mean_3d.shape)[-1]
        s, a, b = np.array(self.mean_3d.shape)[0:3]
        sg = np.linspace(0, s, s)
        ag = np.linspace(0, a, a)
        bg = np.linspace(0, b, b)
        mesh = np.meshgrid(sg, ag, bg, indexing="ij")
        points = np.array([x.flatten() for x in mesh])
        N = points.shape[1]
        cells = [("vertex", np.array([[i] for i in range(N)]))]

        if len(file.split(".")) == 1:
            filename = file + ".xdmf"
        else:
            filename = file

        point_data = {}
        for i in range(dim):
            point_data["mean_" + str(i + 1)] = self.mean_3d[:, :, :, i].flatten()
            for j in range(i, dim):
                point_data["cov_" + str(i + 1) + str(j + 1)] = self.covariance_3d[
                    :, :, :, i, j
                ].flatten()

        meshio.Mesh(
            points.T,
            cells,
            point_data=point_data,
        ).write(filename)


if __name__ == "__main__":
    pass
