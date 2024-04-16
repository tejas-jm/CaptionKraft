# Import necessary functions from each file
from library_setup import *
from data_preprocessing import load_captions_data, train_val_split, IMAGES_PATH, CAPTIONS_PATH,EMBED_DIM,FF_DIM,EPOCHS
from vectorizer_and_augmentation import vectorization, image_augmentation
from visualisations import visualisation, captions_length, word_occurrences
from data_setup import make_dataset
from model_definition import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
from training import cross_entropy, early_stopping, lr_schedule, train_dataset, validation_dataset

# Load and preprocess the data
text_data, train_data, validation_data, test_data = load_captions_data(IMAGES_PATH, CAPTIONS_PATH)
train_data, validation_data = train_val_split(train_data)

# Define and adapt the vectorizer
vectorization.adapt(text_data)

# Visualize data
visualisation(train_data, 7)
captions_length(text_data)
word_occurrences(text_data)

# Prepare datasets
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
validation_dataset = make_dataset(list(validation_data.keys()), list(validation_data.values()))

# Define and compile the model
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=2)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=3)
caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation)

caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

# Train the model
history = caption_model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, callbacks=[early_stopping])

# Save the trained model
model_save_path = 'captioning_efficientnetB0.keras'
caption_model.save(model_save_path)
print("Model saved successfully at:", model_save_path)

# Plot training and validation loss
plt.figure(figsize=(15, 7), dpi=200)
plt.plot([x+1 for x in range(len(history.history['loss']))], history.history['loss'], color='#004EFF', marker='o')
plt.plot([x+1 for x in range(len(history.history['loss']))], history.history['val_loss'], color='#00008B', marker='h')
plt.title('Train VS Validation', fontsize=15, fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.legend(['Train Loss', 'Validation Loss'], loc='best')
plt.show()
