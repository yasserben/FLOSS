import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import torch
from PIL import Image
import torch
import torch.nn.functional as F

CLASSNAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic aicign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def colorize_mask(mask, add=None):
    palette = [
        128,
        64,
        128,
        244,
        35,
        232,
        70,
        70,
        70,
        102,
        102,
        156,
        190,
        153,
        153,
        153,
        153,
        153,
        250,
        170,
        30,
        220,
        220,
        0,
        107,
        142,
        35,
        152,
        251,
        152,
        70,
        130,
        180,
        220,
        20,
        60,
        255,
        0,
        0,
        0,
        0,
        142,
        0,
        0,
        70,
        0,
        60,
        100,
        0,
        80,
        100,
        0,
        0,
        230,
        119,
        11,
        32,
        0,
        0,
        0,
    ]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
        # mask: numpy array of the mask
    if add is not None:
        add = add.cpu()
        add = torch.squeeze(add)
        mask[add == 0] = 19
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
    new_mask.putpalette(palette)
    return new_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.2f} seconds".format(self.avg)


def generate_indices(max_index, percentage, seed=0):
    """
    Generate indices for training and validation set and save them to a file
    Args:
        max_index: the maximum index of the dataset
        percentage: the percentage of the dataset to be used for training/validation
        seed: the seed for the random number generator

    Returns: the indices for the training/validation set

    """
    np.random.seed(seed)
    indices = np.array([str(x).zfill(5) + ".png" for x in range(max_index)])
    np.random.shuffle(indices)
    np.savetxt(
        f"./dataset_creation/gta5_style_{int(percentage*100)}.txt",
        indices[: int(max_index * percentage)],
        fmt="%s",
    )
    return indices[: int(max_index * percentage)]


def create_source_dataset(
    num_samples, max_index=24966, root="/home/ids/benigmim/dataset", seed=0
):
    """
    Create a dataset of the source images with num_images images
    """
    np.random.seed(seed)
    indices = np.array([str(x).zfill(5) + ".png" for x in range(max_index)])
    np.random.shuffle(indices)
    create_directory(os.path.join(root, f"gta5_{num_samples}_samples"))
    for i in indices[:num_samples]:
        shutil.copy(
            os.path.join(root, "gta5/images", i),
            os.path.join(root, f"gta5_{num_samples}_samples", i),
        )


def create_directory(dir):
    """
    Create a directory if it does not exist
    Args:
        dir: the directory to be created

    Returns: None

    """
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created")


def generate_mosaic(num_images, filename, filepath, size):
    """
    Generate a mosaic of the images in the filepath using torchvision
    """
    import torchvision
    import glob
    import os
    from PIL import Image
    from torchvision import transforms

    # convert_tensor = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    convert_tensor = transforms.Compose(
        [transforms.Resize((640, 1088)), transforms.Resize(size), transforms.ToTensor()]
    )

    # Get all images
    # images = list(filename.read().splitlines())
    with open(filename) as f:
        images = f.read().splitlines()
    # images = glob.glob(filename)

    # Create a list of images
    imgs = []
    for i in images[:num_images]:
        imgs.append(convert_tensor(Image.open(os.path.join(filepath, i))))
        # imgs.append(Image.open(os.path.join(filepath,i)))

    # imgs[-2] =imgs[-2][:, :, :-3]

    # Create a new image of the same size
    mosaic = torchvision.utils.make_grid(imgs, nrow=1, padding=1)

    # Save the mosaic
    torchvision.utils.save_image(
        mosaic, os.path.join(filepath, f"mosaic_{num_images}.png")
    )


def generate_list_ckpt(training_steps, checkpoint_steps):
    """
    Generate a list of checkpoints using the maximum number of training_steps
    Args:
        training_steps: the maximum number of training steps
        checkpoint_steps: the number of steps between each checkpoint

    Returns:

    """
    return [x for x in range(checkpoint_steps, training_steps + 1, checkpoint_steps)]


def save_semantic_map(pred, name="seg_map", mask=None):
    pred = pred.unsqueeze(dim=0)
    pred = torch.argmax(pred, dim=1)
    pred_1 = torch.squeeze(pred)
    pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, mask)
    pred_1_map.save(f"./zgarbage/{name}.png")
    return pred_1_map


def visualize_semantic_map(pred, mask=None):
    pred = pred.unsqueeze(dim=0)
    pred = torch.argmax(pred, dim=1)
    pred_1 = torch.squeeze(pred)
    pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, mask)
    plt.imshow(pred_1_map)
    plt.show()
    return pred_1_map


def save_semantic_map_maxed(pred, name, output_dir="./zgarbage"):
    pred_1 = torch.squeeze(pred)
    pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, None)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the image
    output_path = os.path.join(output_dir, f"{name}.png")
    pred_1_map.save(output_path)

    return pred_1_map


def visualize_semantic_map_maxed(pred, mask=None):
    pred_1 = torch.squeeze(pred)
    pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, mask)
    plt.imshow(pred_1_map)
    plt.show()
    return pred_1_map


def save_uncertainty(
    logits, method="softmax", output_path="./zgarbage/uncertainty.png", temperature=1
):
    """
    Visualize uncertainty in semantic segmentation predictions as a black and white image.

    Args:
    logits (torch.Tensor): Tensor of shape (C, H, W) where C is the number of classes.
    method (str): Method to calculate uncertainty. Options: 'softmax', 'difference', 'entropy'.
    output_path (str): Path to save the output image.

    Returns:
    None: Saves the uncertainty visualization as a grayscale image.
    """
    if logits.dim() != 3:
        raise ValueError("Input logits should be a 3D tensor (C, H, W)")

    scaled_logits = logits / temperature

    if method == "softmax":
        uncertainty = 1 - torch.max(F.softmax(scaled_logits, dim=0), dim=0)[0]
    elif method == "difference":
        sorted_logits, _ = torch.sort(scaled_logits, dim=0, descending=True)
        uncertainty = sorted_logits[0] - sorted_logits[1]
        uncertainty = 1 - F.normalize(uncertainty, dim=0)
    elif method == "entropy":
        probabilities = F.softmax(scaled_logits, dim=0)
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=0)
        uncertainty = entropy / np.log2(
            scaled_logits.shape[0]
        )  # Normalize by max possible entropy
    else:
        raise ValueError("Invalid method. Choose 'softmax', 'difference', or 'entropy'")

    # Convert to numpy and scale to 0-255
    uncertainty_np = (uncertainty.cpu().numpy() * 255).astype(np.uint8)

    # Create grayscale image (white = high uncertainty, black = low uncertainty)
    grayscale_image = Image.fromarray(uncertainty_np, mode="L")

    # Save as image
    grayscale_image.save(output_path)
    print(f"Uncertainty visualization saved to {output_path}")


def get_rgb_from_semantic_map_maxed(pred, mask=None):
    pred_1 = np.asarray(pred, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, mask)
    return pred_1_map


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def visualize_seg(image, means, stds, i):
    image = torch.clamp(denorm(image, means, stds), 0, 1)
    image = image[i]
    image = image.cpu().detach().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.show()


def save_rgb(image, index=0, file_path="./zgarbage/rgb.png"):
    """
    Saves an RGB image from a batch of images in a PyTorch tensor.

    Parameters:
    - image (Tensor): A PyTorch tensor containing a batch of images with shape (N, 3, H, W).
    - index (int): Index of the image in the batch to save.
    - file_path (str): Path to save the image file.
    """
    # Select the image at the specified index and move it to CPU
    image = image[index].cpu().detach().numpy()

    # Transpose the image from (3, H, W) to (H, W, 3) for PIL compatibility
    if image.shape[0] == 3:  # Check if it's a 3-channel image
        image = image.transpose((1, 2, 0))

    # Normalize and convert to uint8 if not already
    if image.dtype != np.uint8:
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to 0-1
        image = (255 * image).astype(np.uint8)  # Scale to 0-255

    # Create a PIL image and save it
    im = Image.fromarray(image, "RGB")
    im.save(file_path)


def visualize_mask(mask, i=0):
    # Check shape of mask
    if len(mask.shape) == 3:
        # If mask is a torch tensor
        if isinstance(mask, torch.Tensor):
            mask = mask[i]
            mask = mask.cpu().detach().numpy()
            plt.imshow(mask, cmap="gray")
            plt.show()
        # If mask is a numpy array
        else:
            plt.imshow(mask[i], cmap="gray")
            plt.show()
    elif len(mask.shape) == 2:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().detach().numpy()
            plt.imshow(mask, cmap="gray")
            plt.show()
        else:
            plt.imshow(mask, cmap="gray")
            plt.show()


def save_mask(mask, filename="mask.png", folder="zgarbage"):
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Prepare the mask for saving
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()

    if len(mask.shape) == 3:
        mask = mask[0]  # Take the first channel if it's a 3D array

    # Normalize the mask to 0-255 range
    mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
    mask = mask.astype(np.uint8)

    # Create the full path
    full_path = os.path.join(folder, filename)

    # Save the mask as an image
    plt.imsave(full_path, mask, cmap="gray")
    print(f"Mask saved as {full_path}")


def save_pred_uncertainty(bb_results, index=0):
    import torch.nn.functional as F
    import torchvision.utils as vutils

    a = bb_results[index].seg_logits.data
    a = F.softmax(a, dim=0)
    b = torch.max(a, dim=0)[0].unsqueeze(0)
    vutils.save_image(b, "./zgarbage/softmaxed_logit.jpg")


def save_pred_uncertainty_v2(bb_results, index=0):
    import torch
    import torch.nn.functional as F
    import torchvision.utils as vutils

    # Get the logits
    logits = bb_results[index].seg_logits.data

    # Get the values and indices of the two highest logits for each pixel
    top2_values, _ = torch.topk(logits, k=2, dim=0)

    # Compute the difference between the highest and second highest logits
    diff = top2_values[0] - top2_values[1]

    # Normalize the difference to [0, 1]
    normalized_diff = (diff - diff.min()) / (diff.max() - diff.min())

    # Add a channel dimension
    normalized_diff = normalized_diff.unsqueeze(0)

    # Save the image
    vutils.save_image(normalized_diff, "./zgarbage/logit_difference.jpg")


def save_feature_map(feature_map, output_path=None, alpha=0.6, cmap="jet"):
    """
    Save a given feature map as an image.

    Args:
        feature_map (torch.Tensor): The feature map to visualize.
        output_path (str, optional): The path to save the output image.
                                     If None, saves to './zgarbage/attention.png'.
        alpha (float, optional): The alpha value for the overlay. Defaults to 0.6.
        cmap (str, optional): The colormap to use. Defaults to "jet".
    """
    # Set default output path if not specified
    if output_path is None:
        output_path = "./zgarbage/attention.png"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert the feature map to a numpy array
    feature_map_np = feature_map.squeeze().detach().cpu().numpy()

    # Create a new figure
    plt.figure()

    # Plot the feature map
    plt.imshow(feature_map_np, alpha=alpha, cmap=cmap)

    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    # Close the figure to free up memory
    plt.close()


def save_compact_tensor(tensor, data_samples):
    """
    Save a given tensor in a compact, compressed format.

    Args:
        tensor (torch.Tensor): The tensor to save.
        data_samples (list): List of data samples containing image information.
    """

    root_folder = "/home/nvme/benigmim/dataset/cityscapes_tube"

    output_folder = f"{data_samples[0].pad_shape[0]}_{data_samples[0].pad_shape[1]}"
    output_folder = os.path.join(root_folder, output_folder)
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the original filename and resolution
    original_filename = os.path.basename(data_samples[0].img_path)
    # Remove the "leftImg8bit.png" part from the filename
    original_filename = original_filename.replace("_leftImg8bit.png", "")
    original_resolution = (
        f"{data_samples[0].pad_shape[1]}x{data_samples[0].pad_shape[0]}"
    )

    # Create the new filename
    new_filename = f"tensor_{original_filename}_{original_resolution}.npz"
    output_path = os.path.join(output_folder, new_filename)

    # Convert tensor to numpy array
    numpy_array = tensor.detach().cpu().numpy()

    # Save the numpy array in compressed format
    np.savez_compressed(output_path, tensor=numpy_array)

    print(f"Saved compressed tensor to: {output_path}")


def save_mask_mosaic(masks, mask_cls=None, n=9, filename="mask_mosaic.jpg"):
    """
    Save the first n masks from a batch of masks in a square mosaic layout with class predictions as titles.

    Args:
        masks (torch.Tensor): Tensor of masks with shape [B, N, H, W]
        mask_cls (torch.Tensor, optional): Tensor of class predictions with shape [B, N, C]
        n (int): Number of masks to display (should be a perfect square)
        filename (str): Name of the output file
    """
    import math
    import matplotlib.pyplot as plt

    # Ensure n is a perfect square
    grid_size = int(math.sqrt(n))
    n = grid_size * grid_size

    # Create the subplot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # Convert masks to numpy if they're torch tensors
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    # Get class predictions if provided
    if mask_cls is not None:
        if isinstance(mask_cls, torch.Tensor):
            # Get the index of the highest predicted class for each mask
            pred_classes = torch.argmax(mask_cls, dim=-1)[0].cpu().numpy()  # [N]
        else:
            pred_classes = np.argmax(mask_cls[0], axis=-1)  # [N]

    # Ensure output directory exists
    os.makedirs("./zgarbage", exist_ok=True)

    # Plot each mask
    for idx in range(n):
        row = idx // grid_size
        col = idx % grid_size

        # Get the mask
        mask = masks[0, idx]  # Taking from first batch

        # Get the subplot
        if grid_size > 1:
            ax = axes[row, col]
        else:
            ax = axes

        # Plot the mask
        ax.imshow(mask, cmap="gray")
        ax.axis("off")

        # Add class prediction as title if available
        if mask_cls is not None:
            class_idx = pred_classes[idx]
            if class_idx < len(CLASSNAMES):
                title = CLASSNAMES[class_idx]
                ax.set_title(title, fontsize=8, pad=1)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join("./zgarbage", filename))
    plt.close()


def save_semantic_map_rank(pred, rank=1, name="seg_map_rank", mask=None):
    """
    Save semantic map using the nth highest predicted class for each pixel.

    Args:
        pred (Tensor): Prediction tensor of shape [C, H, W]
        rank (int): Which rank to use (1 = highest, 2 = second highest, etc.)
        name (str): Name of the output file
        mask (Tensor, optional): Optional mask to apply

    Returns:
        PIL.Image: Colorized semantic map
    """
    pred = pred.unsqueeze(dim=0)  # Add batch dimension

    # Get the indices of the top-k predictions, where k is our desired rank
    # values will be [B, rank, H, W]
    # indices will be [B, rank, H, W]
    values, indices = torch.topk(pred, k=rank, dim=1)

    # Take the desired rank (rank-1 because indices are 0-based)
    pred_ranked = indices[:, rank - 1, :, :]  # Shape: [B, H, W]

    pred_1 = torch.squeeze(pred_ranked)  # Remove batch dimension
    pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, mask)
    pred_1_map.save(f"./zgarbage/{name}.png")
    return pred_1_map


def save_training_mosaic(
    inputs, preds, template_preds, output_dir="./zgarbage", name="training_mosaic"
):
    """
    Create and save a mosaic of training images, student predictions, and teacher predictions.

    Args:
        inputs (torch.Tensor): RGB images [B, 3, H, W]
        preds (torch.Tensor): Student model predictions [B, C, H, W]
        template_preds (torch.Tensor): Teacher model predictions [B, H, W]
        output_dir (str): Directory to save the mosaic
        name (str): Name of the output file
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get batch size
    batch_size = inputs.shape[0]

    # Create figure with a grid of subplots
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))

    # If batch_size is 1, wrap axes in a list to make it 2D
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    # For each sample in the batch
    for i in range(batch_size):
        # 1. Plot RGB image
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        # Normalize if needed
        if img.max() > 1:
            img = img / 255.0
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        # 2. Plot student prediction
        student_pred = torch.argmax(preds[i], dim=0).cpu().numpy()
        student_colored = colorize_mask(student_pred)
        axes[i, 1].imshow(student_colored)
        axes[i, 1].set_title("Student Prediction")
        axes[i, 1].axis("off")

        # 3. Plot teacher prediction (ground truth)
        teacher_colored = colorize_mask(template_preds[i].cpu().numpy())
        axes[i, 2].imshow(teacher_colored)
        axes[i, 2].set_title("Teacher Prediction")
        axes[i, 2].axis("off")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches="tight", dpi=150)
    plt.close()
