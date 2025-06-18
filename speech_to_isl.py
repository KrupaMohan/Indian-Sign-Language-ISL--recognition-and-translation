import os
import cv2
import numpy as np
import string
import speech_recognition as sr

# Configuration
data_path = "F:\ISL\data"
LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
TARGET_SIZE = 100
MAX_CHARS_PER_LINE = 10  # wrap after 10 letters per line

def clean_input(text):
    allowed_chars = set(string.ascii_letters + " ")
    cleaned = ''.join(ch for ch in text if ch in allowed_chars)
    return cleaned

def text_to_isl_images(text):
    text = text.upper()
    images = []

    for char in text:
        if char == " ":
            # Add a white space block
            images.append(np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255)
            continue
        if char not in LABELS:
            print(f"Skipping unsupported character: {char}")
            continue

        folder = os.path.join(data_path, char)
        if not os.path.exists(folder):
            print(f"Folder not found for: {char}")
            continue

        # Get the first image in the folder
        first_image_file = next(
            (f for f in os.listdir(folder) if not f.startswith('.') and os.path.isfile(os.path.join(folder, f))),
            None
        )
        if not first_image_file:
            print(f"No images found in folder: {char}")
            continue

        img_path = os.path.join(folder, first_image_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

        if len(img.shape) == 2:  # grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        images.append(img)

    return images

def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def pad_image_width(img, target_width):
    h, w = img.shape[:2]
    if w == target_width:
        return img
    pad_width = target_width - w
    if len(img.shape) == 3:
        pad_shape = (h, pad_width, img.shape[2])
    else:
        pad_shape = (h, pad_width)
    padding = np.ones(pad_shape, dtype=np.uint8) * 255
    padded_img = np.concatenate((img, padding), axis=1)
    return padded_img

def display_isl_word(text):
    isl_images = text_to_isl_images(text)
    if not isl_images:
        print("No valid ISL images to display.")
        return

    lines = chunk_list(isl_images, MAX_CHARS_PER_LINE)

    h_concat_lines = []
    max_width = 0
    # Horizontally concatenate images per line and find max width
    for line_imgs in lines:
        h_concat = cv2.hconcat(line_imgs)
        h_concat_lines.append(h_concat)
        if h_concat.shape[1] > max_width:
            max_width = h_concat.shape[1]

    # Pad all lines to max width
    for i in range(len(h_concat_lines)):
        h_concat_lines[i] = pad_image_width(h_concat_lines[i], max_width)

    combined_image = cv2.vconcat(h_concat_lines)

    cv2.imshow("ISL Translation", combined_image)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def main():
    raw_text = recognize_speech()
    if not raw_text:
        return

    cleaned_text = clean_input(raw_text)
    if not any(ch.isalpha() for ch in cleaned_text):
        print("No valid letters found or no images available.")
        return

    display_isl_word(cleaned_text)

if __name__ == "__main__":
    main()