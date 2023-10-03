from model import*
from dataloader import*
import os
import random
import sys

def main(input_dir, mask_dir,weight_dir,
    image_height=128,
    image_width=128,
    image_channel=3,
    img_size = (128,128),
    num_classes = 4,
    batch_size = 8,
    epochs=200,
    val_samples = 250,):

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
    joined_list=list(zip(input_img_paths, mask_img_paths))
    random.Random(1337).shuffle(joined_list)
    input_img_paths, mask_img_paths = zip(*joined_list)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_mask_img_paths = mask_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_mask_img_paths = mask_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = Loader(
        batch_size, img_size, train_input_img_paths, train_mask_img_paths,num_classes
    )
    val_gen = Loader(batch_size, img_size, val_input_img_paths, val_mask_img_paths,num_classes)
    
    model=AW_Net((image_height,image_width,image_channel),num_classes, dropout_rate=0.0, batch_norm=True)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=weight_dir+"\\weights.h5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
    )
    es=tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
    history = model.fit(train_gen,validation_data=val_gen,epochs=epochs,callbacks=[checkpoint_callback,es])


if __name__ == "__main__":
    img_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    weight_dir = sys.argv[3]
    main(img_dir,mask_dir,weight_dir)
	
