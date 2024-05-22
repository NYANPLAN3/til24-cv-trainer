"""Refer to https://docs.voxel51.com/recipes/custom_importer.html."""

import json
import os

import fiftyone as fo
import fiftyone.utils.data as foud

__all__ = ["JSONLImporter"]


class JSONLImporter(foud.LabeledImageDatasetImporter):
    """Importer."""

    LABEL_FILE = "vlm.jsonl"
    IMAGE_DIR = "images"

    def __init__(
        self,
        dataset_dir,
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        """Init."""
        super(JSONLImporter, self).__init__(
            dataset_dir=dataset_dir, shuffle=shuffle, seed=seed, max_samples=max_samples
        )
        self._labels_file = os.path.join(self.dataset_dir, self.LABEL_FILE)
        self._labels = None
        self._iter_labels = None

    def __iter__(self):
        """Iter."""
        self._iter_labels = iter(self._labels)
        return self

    def __next__(self):
        """Returns information about the next sample in the dataset.

        Returns:
            an  ``(image_path, image_metadata, label)`` tuple, where

            -   ``image_path``: the path to the image on disk
            -   ``image_metadata``: an
                :class:`fiftyone.core.metadata.ImageMetadata` instances for the
                image, or ``None`` if :meth:`has_image_metadata` is ``False``
            -   ``label``: an instance of :meth:`label_cls`, or a dictionary
                mapping field names to :class:`fiftyone.core.labels.Label`
                instances, or ``None`` if the sample is unlabeled

        Raises:
            StopIteration: if there are no more samples to import
        """
        impath, annos = next(self._iter_labels)

        immeta = fo.ImageMetadata.build_for(impath)
        iw, ih = immeta.width, immeta.height
        dets = []
        for anno in annos:
            caption = anno["caption"]
            l, t, w, h = anno["bbox"]
            bbox = (l / iw, t / ih, w / iw, h / ih)
            det = fo.Detection(label=caption, bounding_box=bbox)
            dets.append(det)
        lbl = fo.Detections(detections=dets)
        return impath, immeta, lbl

    def __len__(self):
        """The total number of samples that will be imported.

        Raises:
            TypeError: if the total number is not known
        """
        return len(self._labels)

    @property
    def has_dataset_info(self):
        """Whether this importer produces a dataset info dictionary."""
        return False

    @property
    def has_image_metadata(self):
        """Whether this importer produces :class:`fiftyone.core.metadata.ImageMetadata` instances for each image."""
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) returned by this importer."""
        return fo.Detections

    def setup(self):
        """Setup."""
        labels = []
        with open(self._labels_file, "r") as f:
            for line in f:
                l = line.strip()
                if l == "":
                    continue
                lbl = json.loads(l)
                impath = os.path.join(self.dataset_dir, self.IMAGE_DIR, lbl["image"])
                annos = lbl["annotations"]
                labels.append((impath, annos))

        self._labels = self._preprocess_list(labels)

    def close(self, *args):
        """Performs any necessary actions after the last sample has been imported."""
        pass
