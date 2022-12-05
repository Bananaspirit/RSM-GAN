<h1 align="center">Generation of field edge images by Generative adversarial networks(GAN) method 

## Introduction

This repository contains two files with the .ipynb extension that need to be imported into Google Collaboration for stable code operation. Google Coollaboratory is a cloud service with virtual machines on which you can do Machine Learning. The figure shows the scheme of the notebooks in the Coollaboratory. To start working with the data, they need to be processed, the DeOldify project is used for this.
![Снимок экрана от 2022-06-27 20-49-04](https://user-images.githubusercontent.com/106806088/176846971-40400cc4-3282-4020-a7d0-f9020b7c31e3.png)

## Image preprocessing
Before proceeding to the generation of images, they need to be preprocessed and their natural colors restored.
### source_url or source_path
In the file visualize.py , project [DeOldify](https://github.com/jantic/DeOldify/tree/master/deoldify ) two functions are implemented plot_transformed_image_from_url and plot_transformed_image, where the first is passed the url as access to the image, and the second path to the image. The second function will be used for batch processing of photos, since the original images are stored on Google drive and the url can only be obtained on the storage of this image.
### render_factor
The rendering coefficient was empirically defined as "12", but as it turned out later, almost every third image differs in contrast with the original ones, so ideally it is necessary to divide the original images into groups according to the degree of contrast and select the rendering value individually.
### watermarked
False or True opposite watermarks activate or deactivate it. Here it is presented as the author's idea that your photos were processed using the DeOldify solution.
```python
render_factor = 12 
watermarked = False
images_names = os.listdir('/your_images_path/') 

for name in images_names:
   try:
      path = f'/your_images_path/{name}'
      image_path = colorizer.plot_transformed_image(path, render_factor=render_factor, watermarked=watermarked)
   except Exception:
       print('error')
```
A more detailed description of preprocessing is in the file [DeOldify_preprocessing_images.icons](https://github.com/Bananaspirit/RSM-GAN/blob/main/DeOldify_preprocesing_images.ipynb).
After that, the images need to be resized from 1280x760 to 1024x1024, since this is the largest possible version of the generated image. I use libraries such as os and opencv, which simplifies working with the file system and image processing.
Sample code for batch image processing:
```python
import os
import cv2

images_names = os.listdir('/your_deoldify_images_path/')
for name in images_names:
   try:
       image = cv2.imread(f'/your_deoldify_images_path/{name}')
       resized = cv2.resize(image, (1024,1024))
       cv2.imwrite(f'/your_deoldify_images1024_path/{name}', resized)
   except Exception:
       pass
```
After all the transformation of the images, they need to be prepared and create a training датасет([Training__&_postprocesing_images_using_pytorch.ipynb](https://github.com/Bananaspirit/RSM-GAN/blob/main/Training__%26_postprocesing_images_using_pytorch.ipynb)). Creating a zip archive with images and their meta information in json format. Here you need to specify the path to the processed images, as well as specify the path to the archive in which all training data will be stored (the archive is created automatically).
Code example:
```python
!python dataset_tool.py \
--source=/your_deoldify_images1024_path/ \
--dest=/your_path/pics.zip
```
Before starting the training of the model, it is necessary to understand the available resources of the virtual machine and the arguments passed in the code. The table shows a comparative characteristic for different image sizes and the number of video cards, as well as other parameters.
<p align="center">
<img src="https://user-images.githubusercontent.com/106806088/176851086-c607007d-3952-4163-a714-b0cb550cd4d7.png" />
</p>

The parameter “kimg” is responsible for the number of iterations, or rather for the number of images on which the discriminator is trained, 1 kimg = 1000 images.
## Network training
```python
!python train.py \
--outdir=/your_out_dir/ \
--data=/your_path/pics.zip \
--snap=10 \
--mirror=1 \
--gpus=1 \
--aug=ada \
--target=0.7
```

Learn more about the arguments passed [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch ).
Google Collaboration provides 8 hours of access per day to a GPU-based accelerator, which makes training extremely difficult and the network does not have time to learn until the next metric display to fix the result and continue from that moment. The figure shows an example of the code for starting training from a saved moment.
```python
!python train.py \
--outdir=/your_out_dir/ \
--data=/your_path/pics.zip \
--remove=/your_network_path/ \
--snap=10 \
--mirror=1 \
--gpus=1 \
--aug=ada \
--target=0.7
```
After the model is trained, you can start generating images directly.
```python
!python generate.py \
--outdir=/your_out_dir/ \
--trunc=0.5 \
--seeds=50-100 \
--network=/your_network_path/
```
## Post-processing of images
Since square images are needed to train the network for better convergence and the resolution should be a power of 2, first we changed the resolution to 1024 x 1024, in this resolution the original image is compressed in width, it needs to be restored to its original dimensions of 1280x960.
```python
import os
import cv2

images_names = os.listdir('/your_generated_images_path/') 
for name in images_names:
    try:
        image = cv2.imread(f'/your_generated_images_path/{name}')
        resized = cv2.resize(image, (1280,960)) 
        cv2.imwrite(f'/your_final_images_path/{name}', resized) 
    except Exception:
        pass
```

## Conclusion

### In conclusion, I have fixed for myself that the GAN image generation method is a very complex multi-level and complex system that still needs to be studied and studied, so in order to understand this topic, a more detailed study of other projects and studies, as well as technical documentation, is necessary.








