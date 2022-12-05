<h1 align="center">Генерация изображений кромки поля методом Generative adversarial networks(GAN) 

## Введение

В данном репозитории представлены два файла с расширением .ipynb, которые нужно импортировать в Google Collaboratory для стабильной работы кода. Google Collaboratory, это облачный сервис с виртуальными машинами, на которых можно заниматься Machine Learning. На рисунке представлена схема работы блокнотов в Collaboratory. Для начала работы с данными их надо обработать, для этого использован проект DeOldify. 
![Снимок экрана от 2022-06-27 20-49-04](https://user-images.githubusercontent.com/106806088/176846971-40400cc4-3282-4020-a7d0-f9020b7c31e3.png)

## Предобработка изображений
Прежде чем перейти к генерации изображений, их необходимо предобработать и восстановить их естественные цвета.
### source_url or source_path
В файле visualize.py, проекта [DeOldify](https://github.com/jantic/DeOldify/tree/master/deoldify) реализованы две функции plot_transformed_image_from_url и plot_transformed_image, где первой передается url в качестве доступа к изображению, а второй path к изображению. Для пакетной обработки фото будет использована именно вторая функция, так как исходные изображения хранятся на google drive и url можно получить лишь на хранилище этого изображения.
### render_factor
Коэффициент рендеринга был эмпирически определен как "12", но как выяснилось в дальнейшем, почти каждое третье изображение различно по контрастности с отсальными, поэтому в идеале необходимо разбить исходные изображения в группы по степени контрастности и подбирать значение рендеринга индивидуально.
### watermarked
False или True напротив вотермарки активируют или деактивируют ее. Здесь она представлена как авторская задумка, что ваши фото были обработаны с помощью решения DeOldify.

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
Более детальное описание preprocesing находится в файле [DeOldify_preprocesing_images.ipynb](https://github.com/Bananaspirit/RSM-GAN/blob/main/DeOldify_preprocesing_images.ipynb).
После чего изображениям необходимо изменить размер с 1280x960 до 1024x1024, так как это наибольший возможный вариант генерируемого изображения. Мною используются такие библиотеки как os и opencv, что упрощает работу с файловой системой и обработкой изображений.
Пример кода для пакетной обработки изображений: 
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
После всех трансформации изображений, их необходимо подготовить и создать тренировочный датасет([Training__&_postprocesing_images_using_pytorch.ipynb](https://github.com/Bananaspirit/RSM-GAN/blob/main/Training__%26_postprocesing_images_using_pytorch.ipynb)). Создаем zip архив с изображениями и их метаинформацией в формате json. Здесь необходимо указать путь к обработанным картинкам, так же указать путь к архиву, в котором будут хранится все тренировочные данные(архив создается автоматически).
Пример кода:
```python
!python dataset_tool.py \
--source=/your_deoldify_images1024_path/ \
--dest=/your_path/pics.zip
```
Прежде чем запустить тренировку модели необходимо понимать располагаемыми ресурсами виртуальной машины и передаваемыми аргументами в коде. В таблице представлена сравнительная характеристика для разных размеров изображений и количества видеокарт, а также других параметров.
<p align="center">
<img src="https://user-images.githubusercontent.com/106806088/176851086-c607007d-3952-4163-a714-b0cb550cd4d7.png" />
</p>

Параметр “kimg” отвечает за количество итераций, а точнее за количество изображений на которых обучается дискриминатор, 1 kimg = 1000 изображений.
## Тренировка сети
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
Подробнее о передаваемых аргументах [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).
Google Collaboratory предоставляет 8 часов доступа в день к ускорителю на базе GPU, что крайне усложняет обучение и сеть не успевает обучиться до следующего отображения метрики чтобы зафиксировать результат и продолжить с этого момента. На рисунке пример кода для запуска обучения с сохраненного момента.
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
После того как модель натренирована, можно приступить непосредственно к генерации изображений.
```python
!python generate.py \
--outdir=/your_out_dir/ \
--trunc=0.5 \
--seeds=50-100 \
--network=/your_network_path/
```
## Постобработка изображений
Так как для тренировки сети необходимы квадратные изображения для лучшей сходимости и разрешение должно быть степенью 2, вначале мы изменили разрешение на 1024х1024, в таком разрешении оригинальное изображение сжато по ширине, его необходимо восстановить до исходных размеров 1280х960.
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


