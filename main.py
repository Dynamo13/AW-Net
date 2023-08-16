from model import*
from dataloader import*
import os
import random

def main(input_dir, mask_dir, image_height, image_width, image_channel, num_classes, batch_size, epochs, val_samples):

    img_size = (image_height,image_width)
    input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
    ]
    )
    mask_img_paths = sorted(
    [
        os.path.join(mask_dir, fname)
        for fname in os.listdir(mask_dir)
    ]
    )
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(mask_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_mask_img_paths = mask_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_mask_img_paths = mask_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = SpinePTXT(
        batch_size, img_size, train_input_img_paths, train_mask_img_paths,num_classes
    )
    val_gen = SpinePTXT(batch_size, img_size, val_input_img_paths, val_mask_img_paths,num_classes)
    
    model=AW_Net((image_height,image_width,image_channel),num_classes, dropout_rate=0.0, batch_norm=True)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    history = model.fit(train_gen,validation_data=val_gen,epochs=epochs,)


if __name__ == "__main__":
    input_dir = "E:\\BraTS_data\\Image"
    mask_dir = "E:\\BraTS_data\\Mask"
    image_height=128
    image_width=128
    image_channel=3
    img_size = (image_height,image_width)
    num_classes = 4
    batch_size = 8
    epochs=200
    val_samples = 250
    main(input_dir, mask_dir, image_height, image_width, image_channel, num_classes, batch_size, epochs, val_samples)
	
