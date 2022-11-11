import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import torch
from PIL import Image
from PIL.Image import LANCZOS
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.crop import Settings, crop_image, download_and_cache_models

from .sd_vae import AutoencoderKL


@dataclass
class InferrenceResult:
    original: np.ndarray
    z: np.ndarray
    rec: np.ndarray
    loss: np.ndarray

    @property
    def df(self) -> pd.DataFrame:
        """Loss dataframe sorted with descending order."""
        return pd.DataFrame(
            data={
                "idx": np.arange(len(self.loss)),
                "loss": self.loss,
            }
        ).sort_values("loss", ascending=False)

    def plot_most_and_least_lossy_images(self, n: int = 5) -> None:
        """Plot most and least lossy images."""

        # most lossy images
        fig, axes = plt.subplots(1, n, figsize=(30, 5))
        fig.suptitle("Most lossy images", fontsize=20)
        _batched_imshow(axes, np.array([self.rec[i] for i in self.df.index[:n]]))
        for i, ax in enumerate(axes):
            ax.set_title(f"loss: {self.df.iloc[i, 1]}", fontsize=10)
        plt.show()

        # least lossy images
        fig, axes = plt.subplots(1, n, figsize=(30, 5))
        fig.suptitle("Least lossy images", fontsize=20)
        _batched_imshow(axes, np.array([self.rec[i] for i in self.df.index[-n:]]))
        for i, ax in zip(reversed(range(1, n + 1)), axes):
            ax.set_title(f"loss: {self.df.iloc[-i, 1]}", fontsize=10)
        plt.show()


class VAEHandler:
    """Handler for VAE model."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir

        conf = omegaconf.OmegaConf.load(self.model_dir / "config.yaml")
        self.vae = AutoencoderKL(
            ddconfig=conf.model.params.ddconfig,
            lossconfig=conf.model.params.lossconfig,
            embed_dim=conf.model.params.embed_dim,
            ckpt_path=self.model_dir / "kl-f8-anime2.ckpt",
        )

    def calc_encoded_tensor(self, img_batch: torch.Tensor) -> torch.Tensor:
        """Calc latent tensor from image."""
        return self.vae.encode(img_batch).mode()

    def calc_decoded_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from latent tensor."""
        return self.vae.decode(z)

    def get_loss_results(self, dataset: DataLoader) -> InferrenceResult:
        """Calc loss from image."""

        original = []
        z = []
        rec = []
        losses = []

        idx = 0

        self.vae.eval()

        for _ in tqdm(range(len(dataset))):
            images = next(dataset)
            for i in range(len(images)):
                inputs = images[i:i+1]
                posteriors = self.vae.encode(inputs)
                z_ = posteriors.sample()
                reconstructions = self.vae.decode(z_)

                inputs = inputs.cpu().detach().numpy()
                z_ = z_.cpu().detach().numpy()
                reconstructions = reconstructions.cpu().detach().numpy()

                loss = self.calc_max_recon_loss_per_chunk(inputs, reconstructions)

                original.append(inputs)
                z.append(z_)
                rec.append(reconstructions)
                losses.append(loss)

                idx += 1

        res = InferrenceResult(
            original=np.concatenate(original, axis=0),
            z=np.concatenate(z, axis=0),
            rec=np.concatenate(rec, axis=0),
            loss=np.array(losses),
        )

        return res

    def calc_max_recon_loss_per_chunk(
        self,
        inputs: np.ndarray,
        reconstructions: np.ndarray,
        rows: int = 8,
        cols: int = 8,
    ) -> np.ndarray:
        """Calculate reconstruction losses per a chunk,
        then return maximum of them.
        """
        orig, rec = denormalize(np.concatenate([inputs, reconstructions]))

        loss = []
        for row_orig, row_rec in zip(np.array_split(orig, rows, axis=0), np.array_split(rec, rows, axis=0)):
            for chunk_orig, chunk_rec in zip(np.array_split(row_orig, cols, axis=1), np.array_split(row_rec, cols, axis=1)):
                loss.append(np.mean((chunk_orig - chunk_rec) ** 2))

        return np.max(np.array(loss))

class ImageFolder(Dataset):
    """Dataset class for image."""

    # image extensions
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(
        self,
        img_dir: str,
        transform: list[transforms.ToTensor | transforms.Normalize] | None = None
    ) -> None:
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index: int):
        path = self.img_paths[index]
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_paths)

    def make_iterator(self, batch_size: int = 5, shuffle: bool = True) -> DataLoader:
        """Make iterator for dataset."""
        return iter(DataLoader(self, batch_size=batch_size, shuffle=shuffle))

    def _get_img_paths(self, img_dir: str) -> list[str]:
        """Get image paths from image directory."""
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in ImageFolder.IMG_EXTENSIONS
        ]

        return img_paths


def preprocess_images(
    src: Path,
    dst: Path,
    width: int = 512,
    height: int = 512,
    focal_model_dir: Path | None = None,
) -> None:
    """Preprocess images in a directory.
    If focal_model_dir is specified, images are cropped to the focal area.
    """
    os.makedirs(dst, exist_ok=True)
    files = os.listdir(src)

    def resize_image(image: Image, width: int, height: int) -> Image:
        ratio = width / height
        src_ratio = image.width / image.height

        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width

        resized = image.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        return res

    def save_pic(image: Image, filename: str, index: int) -> None:
        filename_part = os.path.splitext(filename)[0]
        filename_part = os.path.basename(filename_part)

        basename = f"{index:05}-{filename_part}"
        image.save(os.path.join(dst, f"{basename}.png"))

    for index, image_file in enumerate(tqdm(files)):
        subindex = [0]
        filename = os.path.join(src, image_file)
        try:
            image = Image.open(filename).convert("RGB")
        except Exception:
            continue

        if image.width != image.height:
            dnn_model_path = None
            try:
                dnn_model_path = download_and_cache_models(focal_model_dir)
            except Exception as e:
                print("Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.", e)

            autocrop_settings = Settings(
                crop_width = width,
                crop_height = height,
                face_points_weight = 0.9,
                entropy_points_weight = 0.15,
                corner_points_weight = 0.5,
                annotate_image = False,
                dnn_model_path = dnn_model_path,
            )
            for focal in crop_image(image, autocrop_settings):
                save_pic(focal, filename, index)
        else:
            image = resize_image(image, width, height)
            save_pic(image, filename, index)


def denormalize(img_batch: np.ndarray) -> list[np.ndarray]:
    """Denormalize image tensors to plot."""

    if len(img_batch.shape) != 4:
        imgs = np.array([img_batch])
    else:
        imgs = img_batch

    # denormalize if the images are not latent vectors
    if not img_batch.shape[1] == 4:
        imgs = imgs + 0.5
    # min-max normalization for latent vectors
    else:
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

    # transpose to (H, W, C)
    imgs = [np.transpose(img, (1, 2, 0)) for img in imgs]

    return imgs


def _batched_imshow(axes: plt.Axes, img_batch: torch.Tensor) -> plt.Axes:
    """Plot images per batch."""
    imgs = denormalize(img_batch)

    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.axis("off")

    return axes
