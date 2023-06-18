# Breast-Cancer
Breast cancer diagnosis using U-NET neural network: This network uses deep learning and convolutional networks to detect specific regions of interest from
breast pathological images.

In summary, for training the U-Net network in this project, 56 images were selected for training. Additionally, 55 non-repetitive images from the training set were chosen and rotated. Furthermore, another 22 images were selected that improved by decreasing the brightness by 15%, and 22 other images that improved by increasing the brightness by 15%. Finally, these 155 images were divided into train and test sets in an 80 to 20 ratio and trained with 35 epochs. The performance trend of the network can be determined by observing the accuracy, val_accuracy, loss, and val_loss graphs.

"Input-Train" and "Mask-Train" are just samples of my training images, and you can provide your own relevant images. Please provide your images as input 
and their corresponding masks as the training labels. With this approach, the U-NET neural network will be able to learn and detect the desired patterns 
in your images."
