import argparse
import os
from contextlib import nullcontext
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transparent_background import Remover
import struct

from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, get_device, remove_background


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def export_to_vertices_bin(mesh,output_dir, file_name = "vertices.bin"):
    vertices = np.asanyarray(mesh.vertices,dtype=np.float32)
    file_path = os.path.join(output_dir, file_name)

    with open(file_path,'wb') as f:
        f.write(struct.pack('I', vertices.ndim))
        for d in vertices.shape:
            f.write(struct.pack('I', d))
        dtype_str = str(vertices.dtype)
        f.write(struct.pack('I', len(dtype_str)))
        f.write(dtype_str.encode('utf-8'))
        f.write(vertices.tobytes())

def export_to_faces_bin(mesh,output_dir, file_name = "faces.bin"):
    faces = np.asanyarray(mesh.faces,dtype=np.int32)
    file_path = os.path.join(output_dir, file_name)

    with open(file_path,'wb') as f:
        f.write(struct.pack('I', faces.ndim))
        for d in faces.shape:
            f.write(struct.pack('I', d))
        dtype_str = str(faces.dtype)
        f.write(struct.pack('I', len(dtype_str)))
        f.write(dtype_str.encode('utf-8'))
        f.write(faces.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", type=str, nargs="+", help="Path to input image(s) or folder."
    )
    parser.add_argument(
        "--device",
        default=get_device(),
        type=str,
        help=f"Device to use. If no CUDA/MPS-compatible device is found, the baking will fail. Default: '{get_device()}'",
    )


    parser.add_argument(
        "--reduction_count_type",
        choices=["keep", "vertex", "faces"],
        default="keep",
        help="Vertex count type",
    )

    
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/stable-point-aware-3d",
        type=str,
        help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/stable-point-aware-3d'",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=1.3,
        type=float,
        help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="Output directory to save the results. Default: 'output/'",
    )
    parser.add_argument(
        "--texture-resolution",
        default=1024,
        type=int,
        help="Texture atlas resolution. Default: 1024",
    )
    parser.add_argument(
        "--low-vram-mode",
        action="store_true",
        help=(
            "Use low VRAM mode. SPAR3D consumes 10.5GB of VRAM by default. "
            "This mode will reduce the VRAM consumption to roughly 7GB but in exchange "
            "the model will be slower. Default: False"
        ),
    )
    parser.add_argument(
        "--bbox-width",
        type=check_positive,
        help="Bounding box width to scale the mesh. If not specified, the mesh will not be scaled.",
    )

    parser.add_argument(
        "--bbox-height",
        type=check_positive,
        help="Bounding box height to scale the mesh. If not specified, the mesh will not be scaled.",
    )
    parser.add_argument(
        "--bbox-depth",
        type=check_positive,
        help="Bounding box depth to scale the mesh. If not specified, the mesh will not be scaled.",
    )

    remesh_choices = ["none"]
    if TRIANGLE_REMESH_AVAILABLE:
        remesh_choices.append("triangle")
    if QUAD_REMESH_AVAILABLE:
        remesh_choices.append("quad")
    parser.add_argument(
        "--remesh_option",
        choices=remesh_choices,
        default="none",
        help="Remeshing option",
    )
    if TRIANGLE_REMESH_AVAILABLE or QUAD_REMESH_AVAILABLE:
        parser.add_argument(
            "--reduction_count_type",
            choices=["keep", "vertex", "faces"],
            default="keep",
            help="Vertex count type",
        )
        parser.add_argument(
            "--target_count",
            type=check_positive,
            help="Selected target count.",
            default=2000,
        )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for inference"
    )
    args = parser.parse_args()

    # Ensure args.device contains cuda
    devices = ["cuda", "mps", "cpu"]
    if not any(args.device in device for device in devices):
        raise ValueError("Invalid device. Use cuda, mps or cpu")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        device = "cpu"

    print("Device used: ", device)

    model = SPAR3D.from_pretrained(
        args.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
        low_vram_mode=args.low_vram_mode,
    )
    model.to(device)
    model.eval()

    bg_remover = Remover(device=device)
    images = []
    idx = 0
    for image_path in args.image:

        def handle_image(image_path, idx):
            image = remove_background(
                Image.open(image_path).convert("RGBA"), bg_remover
            )
            image = foreground_crop(image, args.foreground_ratio)
            os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)
            image.save(os.path.join(output_dir, str(idx), "input.png"))
            images.append(image)

        if os.path.isdir(image_path):
            image_paths = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
            for image_path in image_paths:
                handle_image(image_path, idx)
                idx += 1
        else:
            handle_image(image_path, idx)
            idx += 1

    vertex_count = (
        -1
        if args.reduction_count_type == "keep"
        else (
            args.target_count
            if args.reduction_count_type == "vertex"
            else args.target_count // 2
        )
    )

    for i in tqdm(range(0, len(images), args.batch_size)):
        image = images[i : i + args.batch_size]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with (
                torch.autocast(device_type=device, dtype=torch.bfloat16)
                if "cuda" in device
                else nullcontext()
            ):
                mesh, glob_dict = model.run_image(
                    image,
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                    vertex_count=vertex_count,
                    return_points=True,
                )
            # print(type(mesh))
            # print(mesh.extents)
            # print(mesh.units)

            if args.bbox_width is None:
                args.bbox_width = 1
            if args.bbox_height is None:
                args.bbox_height = 1
            if args.bbox_depth is None:
                args.bbox_depth = 1

            x,y,z = mesh.extents

            scale_x = args.bbox_width/x
            scale_y = args.bbox_height / y
            scale_z = args.bbox_depth / z

            scale_factor = [scale_x,scale_y,scale_z]

            print(f"Original mesh extents: x={x}, y={y}, z={z}")
            print(f"Target bbox: width={args.bbox_width}, height={args.bbox_height}, depth={args.bbox_depth}")
            print(f"Scale factors: x={scale_x}, y={scale_y}, z={scale_z}")

            mesh.apply_scale(scale_factor)

            new_x, new_y, new_z = mesh.extents
            print(f"New mesh extents after scaling: x={new_x}, y={new_y}, z={new_z}")

        if torch.cuda.is_available():
            print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
        elif torch.backends.mps.is_available():
            print(
                "Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB"
            )

        if len(image) == 1:
            out_mesh_path = os.path.join(output_dir, str(i), "mesh.glb")
            mesh.export(out_mesh_path, include_normals=True)
            out_points_path = os.path.join(output_dir, str(i), "points.ply")
            glob_dict["point_clouds"][0].export(out_points_path)
            
            export_to_faces_bin(mesh=mesh,output_dir=output_dir)
            export_to_vertices_bin(mesh=mesh,output_dir=output_dir)


        else:
            for j in range(len(mesh)):
                out_mesh_path = os.path.join(output_dir, str(i + j), "mesh.glb")
                mesh[j].export(out_mesh_path, include_normals=True)
                out_points_path = os.path.join(output_dir, str(i + j), "points.ply")
                glob_dict["point_clouds"][j].export(out_points_path)
                export_to_faces_bin(mesh=mesh,output_dir=output_dir)
                export_to_vertices_bin(mesh=mesh,output_dir=output_dir)