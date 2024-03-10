import gradio as gr
import cv2
import torch
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv
import numpy as np
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from utils import print_colored
from PIL import Image

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_masks(image):
    # 1.1 Load the model and predict the mask
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    # 1.2 Use the automatic mask generator to generate the mask
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_bgr = np.array(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    return masks

def get_mask_image(masks):
    # 1.3 Annotate the image with the mask
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    image = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
    image[:,:,3] = 1
    for mask in sorted_masks:
        seg = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.1]])
        image[seg] = color_mask

    return image

def get_mask_image(image, masks, x, y):
    # Convert (x, y) from percentage coordinates to pixel coordinates
    width, height = image.size
    x_pixel = int((x / 100.0) * width)
    y_pixel = int((y / 100.0) * height)
    
    image_np = np.array(image)
    
    # Initialize a blank mask of the same size as the image
    result_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    # Iterate over masks to find the one that contains the (x, y) point
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    for mask in sorted_masks:
        seg = mask['segmentation']
        # print(seg)
        if seg[y_pixel, x_pixel] == 1:
            result_mask = seg
            break
    
    # Create an output image where mask region is white and everything else is black
    output_image_np = np.zeros(image_np.shape, dtype=np.uint8)
    output_image_np[result_mask == 1] = 255  # Set mask region to white

    output_image = Image.fromarray(output_image_np)
    
    return output_image

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    
    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask>0.5] = 1.0 # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0,3,1,2)
    image = torch.from_numpy(image)
    # print_colored('The dimension of control image is ' + str(image.ndim), "blue")
    return image

def inpaint_image(init_image, mask_image, control_image, prompt):
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float32, use_safetensors=True)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=True
    ).to(device=DEVICE)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    output = pipe(
        prompt=prompt,
        num_inference_steps=20,
        eta=1.0,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
    ).images[0]
    
    return output

def process_image(image, x, y, prompt):
    # 1. Use the Segment Anything model to find the object at the click location
    masks = get_masks(image)
    mask_image = get_mask_image(image, masks, x, y)
    # print_colored('image type is ' + str(type(image)), 'green')

    # return mask_image
    
    # 2. Use the ControlNet Inpainting model to inpaint the selected area
    # resized_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint.jpg").resize((512,512))
    # resized_mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-mask.jpg").resize((512, 512))
    resized_image = image.resize((512,512))
    resized_mask_image = mask_image.resize((512, 512))
   
    control_image = make_inpaint_condition(resized_image, resized_mask_image)

    output = inpaint_image(resized_image, resized_mask_image, control_image, prompt)
    
    # Resize the output back to the original dimensions of the input image
    original_dimensions = image.size  
    output_resized = output.resize(original_dimensions)
    
    return output_resized

# Adjusting the Gradio Interface initialization with correct parameters
iface = gr.Interface(fn=process_image,
                     inputs=[
                         gr.Image(type='pil'),  
                         gr.Number(label="X Click"),
                         gr.Number(label="Y Click"),
                         gr.Textbox(label="Prompt")
                     ],
                     outputs=gr.Image(),
                     description="Upload an image to inpaint")

iface.launch()