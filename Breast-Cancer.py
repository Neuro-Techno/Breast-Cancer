import os
import cv2
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from PIL import Image, ImageEnhance
import sklearn.model_selection as ms
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img

#تعین مسیر هایه image,mask

TRAIN_IMAGE_PATH = "/Dataset/Inputs_Train"
TRAIN_MASK_PATH = "/Dataset/Masks_Train"


#بارگزاری عکس ها در لیست
Train_Mask_List = sorted(next(os.walk(TRAIN_MASK_PATH))[2])
Train_Image_List = sorted(next(os.walk(TRAIN_IMAGE_PATH))[2])

RotPicNUM = 55  #تعداد تصاویری که قصد چرخاندنشان را داریم
brightPicNUM = 22   #تعداد تصاویری که قصد افزایش و یا کاهش نورشان را داریم


#data Augmentation
    #1- 15% افزایش نور عکس ها


# خواندن نام فایل‌ها در پوشه و افزودن آن‌ها به یک لیست
image_filenames = os.listdir(TRAIN_IMAGE_PATH)


random_images=[]
mask_names=[]
for i in np.random.choice(np.arange(0, 56), size=brightPicNUM, replace=False):
    random_images.append(Train_Image_List[i])
    mask_names.append(Train_Mask_List[i])


# ساخت لیست جدید برای نگهداری تصاویر با افزایش 15 درصدی نور
brightened_images = []
brightened_masks = []

for image_filename in random_images:

    img = Image.open(os.path.join(TRAIN_IMAGE_PATH, image_filename))

    # افزایش 15 درصدی نور تصویر
    enhanced_img = ImageEnhance.Brightness(img).enhance(1.15)

    # اضافه کردن تصویر به لیست جدید
    brightened_images.append(enhanced_img)

for mask_filename in mask_names:

    mask = Image.open(os.path.join(TRAIN_MASK_PATH, mask_filename))

    enhanced_img = ImageEnhance.Brightness(mask).enhance(1.15)

    brightened_masks.append(enhanced_img)

# ذخیره تصاویر با افزایش نور در پوشه جدید
output_img_path = "/Dataset/highBrightnessImg"
output_mask_path = "/Dataset/highBrightnessMask"

if not os.path.exists(output_img_path):
    os.makedirs(output_img_path)
if not os.path.exists(output_mask_path):
    os.makedirs(output_mask_path)

for i, image in enumerate(brightened_images):
    output_filename = f"image_{i}.jpg"
    output_filepath = os.path.join(output_img_path, output_filename)
    image.save(output_filepath)

for i, image in enumerate(brightened_masks):
    output_filename = f"image_{i}.jpg"
    output_filepath = os.path.join(output_mask_path, output_filename)
    image.save(output_filepath)


    #2- 15 کاهش نور

# خواندن نام فایل‌ها در پوشه و افزودن آن‌ها به یک لیست
image_filenames = os.listdir(TRAIN_IMAGE_PATH)

# انتخاب ۹ تصویر تصادفی از لیست
# random_images = random.sample(image_filenames, k=9)

random_images=[]
mask_names=[]
for i in np.random.choice(np.arange(0, 56), size=brightPicNUM, replace=False):
    random_images.append(Train_Image_List[i])
    mask_names.append(Train_Mask_List[i])

# ساخت لیست جدید برای نگهداری تصاویر با کاهش 15 درصدی نور
brightened_images = []
brightened_masks = []

for image_filename in random_images:
    # باز کردن تصویر
    img = Image.open(os.path.join(TRAIN_IMAGE_PATH, image_filename))

    # کاهش 15 درصدی نور تصویر
    enhanced_img = ImageEnhance.Brightness(img).enhance(0.85)

    # اضافه کردن تصویر به لیست جدید
    brightened_images.append(enhanced_img)

for mask_filename in mask_names:

    mask = Image.open(os.path.join(TRAIN_MASK_PATH, mask_filename))

    enhanced_img = ImageEnhance.Brightness(mask).enhance(0.85)

    brightened_masks.append(enhanced_img)

# ذخیره تصاویر با کاهش نور در پوشه جدید
output_img_path = "/Dataset/lowBrightnessImg"
output_mask_path = "/Dataset/lowBrightnessMask"

if not os.path.exists(output_img_path):
    os.makedirs(output_img_path)
if not os.path.exists(output_mask_path):
    os.makedirs(output_mask_path)

for i, image in enumerate(brightened_images):
    output_filename = f"image_{i}.jpg"
    output_filepath = os.path.join(output_img_path, output_filename)
    image.save(output_filepath)

for i, image in enumerate(brightened_masks):
    output_filename = f"image_{i}.jpg"
    output_filepath = os.path.join(output_mask_path, output_filename)
    image.save(output_filepath)

    #3- چرخاندن سی درجه یه تصاویر ترین

# تعریف مسیر پوشه حاوی تصاویر
dir_path = TRAIN_IMAGE_PATH

# تعریف یک ImageDataGenerator برای چرخش تصاویر
datagen = ImageDataGenerator(rotation_range=30)

# خواندن نام فایل‌ها در پوشه انتخاب شده
file_names = os.listdir(dir_path)

# انتخاب 5 تصویر غیر تکراری
selected_files = np.random.choice(file_names, size=RotPicNUM, replace=False)

# برای هر تصویر انتخاب شده، تغییرات لازم را اعمال کرده و در پوشه خروجی ذخیره می‌کنیم
output_dir = '/Dataset/rotatedImages'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for i in np.random.choice(np.arange(0, 56), size=RotPicNUM, replace=False):
    image_name=Train_Image_List[i]
    dir_path = TRAIN_IMAGE_PATH

# for file_name in selected_files:
    # خواندن تصویر و تبدیل آن به آرایه numpy
    img = load_img(os.path.join(dir_path, image_name))
    x = img_to_array(img)

    # تغییرات روی تصویر با استفاده از ImageDataGenerator
    x = datagen.random_transform(x)

    # تبدیل آرایه numpy به تصویر و ذخیره در پوشه خروجی
    img = array_to_img(x)
    img.save(os.path.join(output_dir, image_name))


    mask_name=Train_Mask_List[i]
    dir_path = TRAIN_MASK_PATH

    output_mask = '/Dataset/rotatedMasks'

    if not os.path.exists(output_mask):
        os.makedirs(output_mask)

    # خواندن تصویر و تبدیل آن به آرایه numpy
    mask = load_img(os.path.join(dir_path, mask_name))
    x = img_to_array(mask)

    # تغییرات روی تصویر با استفاده از ImageDataGenerator
    x = datagen.random_transform(x)

    # تبدیل آرایه numpy به تصویر و ذخیره در پوشه خروجی
    mask = array_to_img(x)
    mask.save(os.path.join(output_mask, mask_name))

print("Data Augmentation finished successfully")


# ایجاد یک لیست با شیپ (55,32,32,3)
inputs = np.zeros((len(Train_Image_List)+RotPicNUM+2 *
                  brightPicNUM, 768, 896, 3), dtype=np.uint8)


# تابع برای خواندن تصاویر و ذخیره آن ها در لیست

def read_images_from_dir(dir_path):
    images = []
    for filename in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, filename))
        if img is not None:
            images.append(img)
    return images


# خواندن تصاویر از سه پوشه و اضافه کردن آن ها به لیست
dir_path1 = TRAIN_IMAGE_PATH
images1 = read_images_from_dir(dir_path1)
inputs[:len(images1)] = images1

dir_path2 = '/Dataset/rotatedImages'
images2 = read_images_from_dir(dir_path2)
inputs[len(images1):(len(images1)+len(images2))] = images2

dir_path3 = '/Dataset/highBrightnessImg'
images3 = read_images_from_dir(dir_path3)
inputs[(len(images1)+len(images2)):(len(images1)+len(images2)+len(images3))] = images3

dir_path4 = '/Dataset/lowBrightnessImg'
images4 = read_images_from_dir(dir_path4)
inputs[(len(images1)+len(images2)+len(images3)):(len(images1)+len(images2)+len(images3)+len(images4))] = images4

inputs = inputs/255

print('inputs.shape: ', inputs.shape)


ground_truth = np.zeros(
    (len(Train_Mask_List)+RotPicNUM+2*brightPicNUM, 768, 896, 3), dtype=np.bool)



def read_images_from_dir(dir_path):
    images = []
    for filename in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, filename))
        if img is not None:
            images.append(img)
    return images


dir_path1 = TRAIN_MASK_PATH
images1 = read_images_from_dir(dir_path1)
ground_truth[:len(images1)] = images1

dir_path2 = '/Dataset/rotatedMasks'
images2 = read_images_from_dir(dir_path2)
ground_truth[len(images1):(len(images1)+len(images2))] = images2

dir_path3 = '/Dataset/highBrightnessMask'
images3 = read_images_from_dir(dir_path3)
inputs[(len(images1)+len(images2)):(len(images1)+len(images2)+len(images3))] = images3

dir_path4 = '/Dataset/lowBrightnessMask'
images4 = read_images_from_dir(dir_path4)
inputs[(len(images1)+len(images2)+len(images3)):(len(images1)+len(images2)+len(images3)+len(images4))] = images4

ground_truth = ground_truth/255

print('ground_truth: ', ground_truth.shape)

xtrain, xtest, ytrain, ytest = ms.train_test_split(inputs, ground_truth, train_size=0.8)
print("ytest.shape: ", ytest.shape)
print("ytrain.shape: ", ytrain.shape)


#طراحی شبکه U-net

inputs = Input((768, 896, 3))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
conv4 = BatchNormalization()(conv4)

up5 = UpSampling2D(size=(2, 2))(conv4)
up5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
up5 = BatchNormalization()(up5)
merge5 = concatenate([conv3, up5], axis=3)
conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge5)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
conv5 = BatchNormalization()(conv5)

up6 = UpSampling2D(size=(2, 2))(conv5)
up6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
up6 = BatchNormalization()(up6)
merge6 = concatenate([conv2, up6], axis=3)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
conv6 = BatchNormalization()(conv6)

up7 = UpSampling2D(size=(2, 2))(conv6)
up7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
up7 = BatchNormalization()(up7)
merge7 = concatenate([conv1, up7], axis=3)
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
conv7 = BatchNormalization()(conv7)

outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv7)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(lr=1e-4),
              loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=2,
                    epochs=30)

pd.DataFrame(history.history).plot()
