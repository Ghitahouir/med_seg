import os
from pathlib import Path

import numpy as np
import pydicom
import torch
from monai.transforms import (
    InvertibleTransform,
    MapTransform,
    TraceableTransform,
    Transform,
)
from monai.utils import ensure_tuple, ensure_tuple_rep
from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset
from pydicom.multival import MultiValue
from pydicom.uid import ImplicitVRLittleEndian
from pydicom.valuerep import PersonName


class Renamed(MapTransform, InvertibleTransform, TraceableTransform):
    """This transform changes the name of the key in a dictionary dataset.
    Careful: rename each key separately.
    """

    def __init__(self, keys, src_key, dst_key, allow_missing_keys):
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.dst_key = dst_key
        self.src_key = src_key

    def __call__(self, data):
        d = dict(data)
        keys = d.keys()
        if self.src_key in keys:
            element = d[self.src_key]
            d.pop(self.src_key)
            d[self.dst_key] = element
        return d

class PopKeyd(MapTransform, InvertibleTransform, TraceableTransform):
    """This transform deletes the name of a key in a dictionary dataset.
    Careful: rename each key separately.
    """

    def __init__(self, keys, src_key, allow_missing_keys):
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.src_key = src_key

    def __call__(self, data):
        d = dict(data)
        keys = d.keys()
        if self.src_key in keys:
            d.pop(self.src_key)
        return d


class Write_dicom(Transform):
    """
    This transforms writes and saves arrays to DICOM format.
    """

    def __init__(self, save: bool, save_location: str = None):
        self.save = save

    def __call__(self, array, as_dicom, save_location):
        meta = pydicom.dcmread(as_dicom)
        array = (array * 255**2).astype(np.uint16)
        ## Creating the header information
        file_meta = Dataset()
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = meta.file_meta.MediaStorageSOPClassUID
        file_meta.MediaStorageSOPInstanceUID = meta.file_meta.MediaStorageSOPInstanceUID
        file_meta.ImplementationClassUID = meta.file_meta.ImplementationClassUID

        ## Creating the UID information in the metadata
        ds = FileDataset(save_location, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = meta.Modality
        ds.ContentDate = meta.ContentDate
        ds.ContentTime = meta.ContentTime
        ds.StudyInstanceUID = meta.StudyInstanceUID
        ds.SeriesInstanceUID = meta.SeriesInstanceUID
        ds.SOPInstanceUID = meta.SOPInstanceUID
        ds.SOPClassUID = meta.SOPClassUID

        ## These are the necessary imaging components of the FileDataset object.
        ds.SamplesPerPixel = meta.SamplesPerPixel
        ds.PhotometricInterpretation = meta.PhotometricInterpretation
        ds.PixelRepresentation = meta.PixelRepresentation
        ds.BitsStored = array[0][0].nbytes * 8
        ds.BitsAllocated = array[0][0].nbytes * 8
        ds.SmallestImagePixelValue = array.min().tobytes()
        ds.LargestImagePixelValue = array.max().tobytes()
        ds.Columns = array.shape[0]
        ds.Rows = array.shape[1]
        ds.PixelSpacing = meta.PixelSpacing

        ## Creating some necessary metadata keys for future transforms such as Cropd
        ds.RescaleSlope = meta.RescaleSlope
        ds.ImageOrientationPatient = meta.ImageOrientationPatient
        ds.ImagePositionPatient = meta.ImagePositionPatient
        ds.RescaleIntercept = meta.RescaleIntercept

        ## Saving the pixel data
        ds.PixelData = array.tobytes()

        if self.save:
            pydicom.dcmwrite(save_location, dataset=ds, write_like_original=False)
            return ds
        return ds


class LoadDicom(Transform):
    """
    Load dicom file or files from provided path.

    """

    def __init__(self, image_only: bool = False, dtype="float32") -> None:
        """
        Args:

            image_only: if True return only the image volume, otherwise return image data array and header dict.
            dtype: if not None convert the loaded image to this data type.

        Note:

            - The transform returns an image data array if `image_only` is True,
              or a tuple of two elements containing the data array, and the meta data in a dictionary format otherwise.

        """

        self.image_only = image_only
        self.dtype = dtype

    def read(self, files):
        "open data and load it, careful can be a list of dicoms (directory)"
        if isinstance(files, list):
            files = [filepath for filepath in files if filepath.is_file()]

        elif files.is_dir():
            files = [
                sub_filepath
                for sub_filepath in Path(files).glob("**/*")
                if sub_filepath.is_file()
            ]

        if not isinstance(files, list):
            files = [files]
        meta = []
        data = []
        for filepath in files:
            img = pydicom.dcmread(filepath)
            data.append(img)
            meta_data = pydicom.dcmread(filepath, stop_before_pixels=True)
            meta.append(meta_data)
        return data, meta

    def dictify(self, meta, remove_overlay_data=True):

        """
        Turns a pydicom Dataset into a dict with keys derived from the Element tags.

        Parameters
        ----------
        meta : pydicom.dataset.Dataset
            The Dataset to dictify
        remove_overlay_data : boolean
            Whether to remove the Overlay Data

        Returns
        -------
        output : dict
        """
        output = dict()
        for elem in meta:
            if not isinstance(elem.value, MultiValue):
                if not isinstance(elem.value, PersonName):
                    if elem.VR != "SQ":
                        output[elem.name] = elem.value
                    else:
                        output[elem.name] = [self.dictify(item) for item in elem]
        if remove_overlay_data:
            if "Overlay Data" in output:
                del output["Overlay Data"]
        return output

    def affine2d(self, data):
        """
        Returns the affine matrix of a given Dicom. For now it only works for 2D slices.
        The formula for 3D inputs can be found here: https://nipy.org/nibabel/dicom/dicom_orientation.html .
        """
        F11, F21, F31 = data[0].ImageOrientationPatient[3:]
        F12, F22, F32 = data[0].ImageOrientationPatient[:3]

        dr, dc = data[0].PixelSpacing
        Sx, Sy, Sz = data[0].ImagePositionPatient

        return np.array(
            [
                [F11 * dr, F12 * dc, 0, Sx],
                [F21 * dr, F22 * dc, 0, Sy],
                [F31 * dr, F32 * dc, 0, Sz],
                [0, 0, 0, 1],
            ]
        )

    def get_data(self, data, meta):
        """
        Gets img array and meta_data.
        For nows, we can't use self.dictify because when we create batches we need to have the exact same keys in the meta_dict.
        For my data, it is not the case if we use dictify because of the differences within the train dataset.
        For very regular data that has the exact same header/meta_data, you can uncomment the corresponding line and comment the
        lines between the stars.
        """

        pix = [0.5, 0.5]
        shape = data[0].pixel_array.shape
        affine = self.affine2d(meta)

        # --------uncomment for dictify--------
        # meta_data = self.dictify(meta[0], remove_overlay_data=True)
        # -------------------------------------

        # adding the pixel spacing, array shape and affine

        # *******
        meta_data = dict()
        meta_data["Rescale Slope"] = float(meta[0]["RescaleSlope"].value)
        meta_data["Image Orientation Patient"] = data[0].ImageOrientationPatient
        meta_data["Image Position Patient"] = data[0].ImagePositionPatient
        meta_data["Rescale Intercept"] = float(meta[0]["RescaleIntercept"].value)
        # *******

        meta_data["Pixel Spacing"] = pix
        meta_data["Data Shape"] = shape
        meta_data["Affine"] = affine

        scaled = []

        for k in range(len(data)):

            slope = float(meta[k]["RescaleSlope"].value)
            intercept = float(meta[k]["RescaleIntercept"].value)
            scaled_data = data[k].pixel_array * slope + intercept
            scaled.append(scaled_data)

        if len(data) > 1:
            output = np.stack(scaled)
        else:
            output = scaled_data

        return output, meta_data

    def __call__(self, filename):
        """
        Load Dicom file and meta data from the given filename(s).

        Args:
            filename: path file or file-like object or a list of files.
                will save the filename to meta_data with key `filename_or_obj`.
                if provided a list of files, use the filename of first file to save,
                and will stack them together as multi-channels data.
                if provided directory path instead of file path, will treat it as
                DICOM images series and read.
        """
        img, meta = self.read(filename)
        img_array, meta_data = self.get_data(img, meta)
        img_array = img_array.astype(self.dtype, copy=False)

        if self.image_only:
            return img_array
        meta_data[
            "filename_or_obj"
        ] = f"{ensure_tuple(filename)[0]}"  # Path obj should be strings for data loader

        return img_array, meta_data


class LoadDicomd(MapTransform):
    """
    Dictionary-based wrapper of LoadDicom,
    It can load both image data and metadata. When loading a list of files in one key,
    the arrays will be stacked and a new dimension will be added as the first dimension
    In this case, the meta data of the first image will be used to represent the stacked result.
    The affine transform of all the stacked images should be same.
    The output metadata field will be created as ``meta_keys`` or ``key_{meta_key_postfix}``.

    """

    def __init__(
        self,
        keys,
        dtype=np.float32,
        meta_keys=None,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
        image_only: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: if not None convert the loaded image data to this data type.
            meta_keys: explicitly indicate the key to store the corresponding meta data dictionary.
                the meta data is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The meta data is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting: whether allow to overwrite existing meta data of same key.
                default is False, which will raise exception if encountering existing key.
            image_only: if True return dictionary containing just only the image volumes, otherwise return
                dictionary containing image data array and header dict per input key.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadDicom(image_only, dtype)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(
                f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}."
            )
        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.meta_keys, self.meta_key_postfix
        ):
            data = self._loader(d[key])
            if self._loader.image_only:
                if not isinstance(data, np.ndarray):
                    raise ValueError(
                        "loader must return a numpy array (because image_only=True was used)."
                    )
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError(
                        "loader must return a tuple or list (because image_only=False was used)."
                    )
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(
                        f"Meta data with key {meta_key} already exists and overwriting=False."
                    )
                d[meta_key] = data[1]
        return d
