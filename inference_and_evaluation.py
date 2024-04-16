from vectorizer_and_augmentation import vectorization
from data_preprocessing import test_data, SEQ_LENGTH
from library_setup import *
from model_definition import caption_model
from data_setup import decode_and_resize

vocab = vectorization.get_vocabulary()
INDEX_LOOKUP = dict(zip(range(len(vocab)), vocab))
MAX_DECODED_SENTENCE_LENGTH = SEQ_LENGTH - 1
test_images = list(test_data.keys())

def generate_caption(image):
    # Read the image from the disk
    image = decode_and_resize(image)

    # Pass the image to the CNN
    image = tf.expand_dims(image, 0)
    image = caption_model.cnn_model(image)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(image, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(MAX_DECODED_SENTENCE_LENGTH):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = INDEX_LOOKUP[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    
    return decoded_caption

def BLEU_score(actual, predicted):
    # Standardizing the actual captions
    processed_actual = []
    for i in actual:
        cap = [INDEX_LOOKUP[x] for x in vectorization(i).numpy() if INDEX_LOOKUP[x] != '']
        cap = ' '.join(cap)
        processed_actual.append(cap)
    
    # Calculating the BLEU score by comparing the predicted caption with five actual captions.
    b1 = corpus_bleu(processed_actual, predicted, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(processed_actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(processed_actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(processed_actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    
    return [
        (f'{b1 * 100}'),
        (f'{b2 * 100}'),
        (f'{b3 * 100}'),
        (f'{b4 * 100}')
    ]


# Initialize lists to store BLEU scores
l1 = []
l2 = []
l3 = []
l4 = []
for i in range(1000):
    # Select one sample image
    sample_image = np.random.choice(test_images)

    # Generate caption for the selected image
    predicted_caption = generate_caption(sample_image)

    # Get the actual captions for the selected image
    actual_captions = test_data[sample_image]

    # Calculate BLEU scores
    bleu_scores = BLEU_score(actual_captions, [predicted_caption] * len(actual_captions))

    # Append BLEU scores to respective lists
    l1.append(float(bleu_scores[3]))  
    l2.append(float(bleu_scores[2])) 
    l3.append(float(bleu_scores[1]))  
    l4.append(float(bleu_scores[0])) 

# Calculate the average BLEU scores
avg_bleu1 = np.mean(l1)
avg_bleu2 = np.mean(l2)
avg_bleu3 = np.mean(l3)
avg_bleu4 = np.mean(l4)

# Print the average BLEU scores
print("Average BLEU-1 score:", avg_bleu1)
print("Average BLEU-2 score:", avg_bleu2)
print("Average BLEU-3 score:", avg_bleu3)
print("Average BLEU-4 score:", avg_bleu4)
