import os
import io
from datetime import datetime
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from resizeimage import resizeimage
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class InspyrenetRembg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",), "torchscript_jit": (["default", "on"],)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit):
        if torchscript_jit == "default":
            remover = Remover()
        else:
            remover = Remover(jit=True)
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type="rgba")
            out = pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        return (img_stack, mask)


class InspyrenetRembgAdvanced:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "torchscript_jit": (["default", "on"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit, threshold):
        if torchscript_jit == "default":
            remover = Remover()
        else:
            remover = Remover(jit=True)
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type="rgba", threshold=threshold)
            out = pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        return (img_stack, mask)


# model = InspyrenetRembg()
model = InspyrenetRembgAdvanced()


def remove_background_inspyrenet(input_image_data):
    image = pil2tensor(input_image_data)
    # img_stack, mask = model.remove_background(image, torchscript_jit="default")
    img_stack, mask = model.remove_background(image, torchscript_jit="default", threshold="0.7")
    output_image = tensor2pil(
        img_stack.squeeze()
    )
    return output_image


input_folder = "./input/"
output_folder_base = "./output/"

current_date = datetime.now().strftime("%d-%m-%y")
output_folder = os.path.join(output_folder_base, current_date)
os.makedirs(output_folder, exist_ok=True)

def main():
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Leer la imagen original
            with open(input_path, "rb") as input_file:
                input_image_data = input_file.read()
            input_image = Image.open(io.BytesIO(input_image_data))

            # Remover el fondo
            output_image = remove_background_inspyrenet(input_image)

            # Convertir a RGB si es necesario
            if output_image.mode != "RGBA":
                output_image = output_image.convert("RGBA")

            # Recortar la parte de la imagen que tiene contenido (sin transparencia)
            bbox = output_image.getbbox()
            if bbox:
                output_image = output_image.crop(bbox)

            # Crear un fondo blanco de 900x900
            canvas_size = 900
            margin = 35
            
            background = Image.new(
                "RGBA", (canvas_size, canvas_size), (255, 255, 255, 255)
            )


            # Calcular el tamaño máximo permitido para la imagen procesada
            max_size = canvas_size - 2 * margin
            output_image = resizeimage.resize_contain(
                output_image, [max_size, max_size]
            )

            # Centrar la imagen en el fondo blanco
            offset = (
                (canvas_size - output_image.width) // 2,
                (canvas_size - output_image.height) // 2,
            )
            background.paste(output_image, offset, output_image)

            # Eliminar el canal alfa para asegurar el fondo blanco
            background = background.convert("RGB")

            # Guardar la imagen procesada con fondo blanco
            output_path = os.path.splitext(output_path)[0] + ".png"
            background.save(output_path, format="PNG")

    print(
        f"Procesamiento completado. Imágenes procesadas guardadas en: {output_folder}"
    )


if __name__ == "__main__":
    main()
