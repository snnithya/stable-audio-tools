import pandas as pd
import random
def get_custom_metadata(info, audio):
    
    # genres = info['genres']
    # # random permute genres
    # random.shuffle(genres)

    # caption = f"Genres: {', '.join(genres)}"
    try:
        prompt = info['Qwen2_5-caption']
        # randomly shuffle the sentences in the prompt, and drop each with 0.1 probability
        sentences = prompt.split('. ')
        random.shuffle(sentences)
        sentences = [s for s in sentences if random.random() > 0.1]
        caption = '. '.join(sentences)

    except KeyError:
        print(f"KeyError: 'Qwen2_5-caption' not found in {info['relpath']}, falling back to genres.")
        # default back to genres
        genres = info['genres']
        # random permute genres
        random.shuffle(genres)
        caption = f"Genres: {', '.join(genres)}"

    return {"prompt": caption}