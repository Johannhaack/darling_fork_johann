"""Collection of pre-implemneted h5 readers developed for id03 format.

NOTE: In general the file reader is strongly dependent on data collection scheme and it is therefore the
purpose of darling to allow the user to subclass Reader() and implement their own specific data structure.

Once the reader is implemented in darling format it is possible to interface the DataSet class and use
all features of darling.

"""

import re

import h5py
import numpy as np

import darling


class Reader(object):
    """Parent class for readers.

    Args:
        abs_path_to_h5_file (:obj: `str`): Absolute file path to data.

    Attributes:
        abs_path_to_h5_file (:obj: `str`): Absolute file path to data.

    """

    def __init__(self, abs_path_to_h5_file):
        self.abs_path_to_h5_file = abs_path_to_h5_file

    def __call__(self, scan_id, roi=None):
        """Method to read a single 2D scan

        NOTE: This method is meant to be purpose implemented to fit the specific data aqusition
            scheme used.

        Args:
            scan_id (:obj:`str`): scan id to load from, these are internal kayes to diffirentiate
                layers.
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded. The roi refers to the detector
                dimensions.

        Returns:
            data (:obj:`numpy array`) of shape=(a,b,m,n) and type np.uint16 and motors
            (:obj:`tuple` of :obj:`numpy array`) of shape=(m,) and shape=(n,) and type
            np.float32. a,b are detector dimensions while m,n are scan dimensions over
            which teh motor settings vary.

        """
        pass


class MosaScan(Reader):
    """Load a 2D mosa scan. This is a id03 specific implementation matphing a specific beamline mosa scan macro.

    NOTE: This reader was specifically written for data collection at id03. For general purpose reading of data you
    must implement your own reader class. The exact reding of data is strongly dependent on data aqusition scheme and
    data structure implementation.

    Args:
        abs_path_to_h5_file str (:obj:`str`): absolute path to the h5 file with the diffraction images.
    """

    def __init__(
        self,
        abs_path_to_h5_file,
    ):
        self.abs_path_to_h5_file = abs_path_to_h5_file
        self.config = darling.metadata.ID03(abs_path_to_h5_file)
        self.scan_params = None

    def __call__(self, scan_id, roi=None):
        """Load a scan

        this loads the mosa data array with shape N,N,m,n where N is the detector dimension and
        m,n are the motor dimensions as ordered in the self.motor_names. You may view the implemented darling readers as example templates for implementing
        your own reader.

        Args:
            scan_id (:obj:`str`):scan id to load from, e.g 1.1, 2.1 etc...
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded

        Returns:
            data, motors : data of shape=(a,b,m,n) and the
                motor arrays as 2d meshgrids, shape=(k,m,n)
                where k is the number of motors used in the
                scan (typically 1,2 or 3).

        """

        self.scan_params = self.config(scan_id)

        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            # Read in motrs
            motors = [
                h5f[scan_id][mn][...].reshape(*self.scan_params["scan_shape"])
                for mn in self.scan_params["motor_names"]
            ]
            motors = np.array(motors).astype(np.float32)

            # read in data and reshape
            if roi:
                r1, r2, c1, c2 = roi
                data = h5f[scan_id][self.scan_params["data_name"]][:, r1:r2, c1:c2]
            else:
                data = h5f[scan_id][self.scan_params["data_name"]][:, :, :]

            data = data.reshape(
                (*self.scan_params["scan_shape"], data.shape[-2], data.shape[-1])
            )
            data = data.swapaxes(0, -2)
            data = data.swapaxes(1, -1)

        return data, motors


class Darks(MosaScan):
    """Load a series of motorless images. This is a id03 specific implementation matphing aspecific beamline mosa scan macro.

    Typically used to red dark images collected with a loopscan.

    NOTE: This reader was specifically written for data collection at id03. For general purpose reading of data you
    must implement your own reader class. The exact reding of data is strongly dependent on data aqusition scheme and
    data structure implementation.

    Args:
        abs_path_to_h5_file str (:obj:`str`): absolute path to the h5 file with the diffraction images.
    """

    def __call__(self, scan_id, roi=None):
        """Load a scan

        this loads the static scan data array with shape a,b,m where a,b are the detector dimensions and
        m is the number of (motorless) images. You may view the implemented darling readers as example templates
        for implementing your own reader.

        Args:
            scan_id (:obj:`str`):scan id to load from, e.g 1.1, 2.1 etc...
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded

        Returns:
            data, motors : data of shape=(a,b,m) and an empty motor array.

        """

        self.scan_params = self.config(scan_id)

        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:

            motors = np.array([], dtype=np.float32)

            if roi:
                r1, r2, c1, c2 = roi
                data = h5f[scan_id][self.scan_params["data_name"]][:, r1:r2, c1:c2]
            else:
                data = h5f[scan_id][self.scan_params["data_name"]][:, :, :]

            data = data.reshape(
                (*self.scan_params["scan_shape"], data.shape[-2], data.shape[-1])
            )
            data = data.swapaxes(0, -2)
            data = data.swapaxes(1, -1)

        return data, motors
        

class RockingScan(MosaScan):
    """Load a 1D rocking scan. This is a id03 specific implementation matphing aspecific beamline mosa scan macro.

    A rocking scan is simply a set of 2D detector images collected at different rocking angles of the goniometer.

    NOTE: This reader was specifically written for data collection at id03. For general purpose reading of data you
    must implement your own reader class. The exact reding of data is strongly dependent on data aqusition scheme and
    data structure implementation.

    Args:
        abs_path_to_h5_file str (:obj:`str`): absolute path to the h5 file with the diffraction images.
    """

    def __call__(self, scan_id, roi=None):
        """Load a scan

        this loads the rocking scan data array with shape a,b,m where a,b are the detector dimensions and
        m is the motor dimensions. You may view the implemented darling readers as example templates
        for implementing your own reader.

        Args:
            scan_id (:obj:`str`):scan id to load from, e.g 1.1, 2.1 etc...
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded

        Returns:
            data, motors : data of shape=(a,b,m) and the
                motor arrays as an array of shape=(k, n)
                where k is the number of motors used in the
                scan (i.e k=1 for a rocking scan).

        """
        return super().__call__(scan_id, roi)


class EnergyScan(Reader):
    """Load a 2D energy scan. This is a id03 specific implementation matphing a specific beamline energy scan macro.

    NOTE: This reader was specifically written for data collection at id03. For general purpose reading of data you
    must implement your own reader class. The exact reding of data is strongly dependent on data aqusition scheme and
    data structure implementation. You may view the implemented darling readers as example templates for implementing
    your own reader.

    Args:
        abs_path_to_h5_file str (:obj:`str`): absolute path to one of the h5 file with the diffraction images. FOr an energy
            scan there is one file per layer in z. This may be any one of these paths. Provided the file name ends with
            layer_0.h5 the file path will be rebuilt upon read call.
    """

    def __init__(self, abs_path_to_h5_file):
        self.abs_path_to_h5_file = abs_path_to_h5_file
        self.config = darling.metadata.ID03(abs_path_to_h5_file)
        self.scan_params = None

    def _get_layer_path(self, scan_id):
        layer_number = str(int(scan_id[0]) - 1)
        layer_tag = r"layer_" + layer_number + ".h5"
        return re.sub(r"layer_\d+\.h5", layer_tag, self.abs_path_to_h5_file)

    def __call__(self, scan_id, roi=None):
        """Load a scan

        this loads the mosa data array with shape N,N,m,n where N is the detector dimension and
        m,n are the motor dimensions as ordered in the self.motor_names.

        Args:
            scan_id (:obj:`str`):scan id to load from, e.g 1.1, 2.1 etc...
            roi (:obj:`tuple` of :obj:`int`): row_min row_max and column_min and column_max,
                defaults to None, in which case all data is loaded

        Returns:
            data, motors : data of shape (a,b,m,n) and motors tuple of len=m and len=n

        """
        abs_path_to_h5_file = self._get_layer_path(scan_id)

        with h5py.File(abs_path_to_h5_file, "r") as h5f:
            key0 = list(h5f.keys())[0]
            self.config.abs_path_to_h5_file = abs_path_to_h5_file
            self.scan_params = self.config(key0)

            _, det_rows, det_cols = h5f[key0][self.scan_params["data_name"]].shape
            n_energy = len(h5f.keys())
            n_phis = self.scan_params["scan_shape"][0]

            if roi is None:
                data = np.zeros((det_rows, det_cols, n_phis, n_energy), dtype=np.uint16)
            else:
                r1, r2, c1, c2 = roi
                data = np.zeros((r2 - r1, c2 - c1, n_phis, n_energy), dtype=np.uint16)

            energy = np.zeros((n_phis, n_energy))
            phi = np.zeros((n_phis, n_energy))

            ccmth_motor = self.config.motor_map["ccmth"]
            phi_motor = self.scan_params["motor_names"][0]

            # second motor is assumed to be integrated in 1 step
            if self.scan_params["scan_shape"][1] != 1:
                raise NotImplementedError(
                    "The scan apears to be 3D with 2 angular motors + energy"
                )

            for i, key in enumerate(h5f.keys()):  # iterates over energies.
                phi_stack = h5f[key][self.scan_params["data_name"]][:, :, :]
                phi_stack = np.swapaxes(phi_stack, 0, 1)
                phi_stack = np.swapaxes(phi_stack, 1, 2)
                if roi is None:
                    data[:, :, :, i] = phi_stack
                else:
                    data[:, :, :, i] = phi_stack[r1:r2, c1:c2, :]

                energy[:, i] = h5f[key][ccmth_motor][()]
                phi[:, i] = h5f[key][phi_motor][:]

        motors = np.array([phi, energy]).astype(np.float32)

        return data, motors


if __name__ == "__main__":
    pass
