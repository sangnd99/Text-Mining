import pickle
import argparse
import re
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

t = time.time()

# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words=set(stopwords.words('english')) 

emojis={':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
              ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Predict sentiment of text string'
    )
    parser.add_argument(
        '-t',
        '--input_text',
        type=str,
        metavar='',
        required=True,
        help='Text input for prediction'
    )
    parser.add_argument(
        '-f',
        '--model_file',
        type=str,
        metavar='',
        required=True,
        help='Name of model'
    )
    return parser.parse_args()

def preprocess(textdata):
    processedText=[]
    
    wordLemm=WordNetLemmatizer()
    
    for text in textdata:
        text.lower()
        
        text=re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", 'URL', text)
        text=re.sub("@[^\s]+", 'USER', text)
        text=text.replace('&', ' ')
        
        for emoji in emojis.keys():
            text=text.replace(emoji, 'EMO' + emojis[emoji] + ' ')
            
        text=re.sub("[^a-zA-Z0-9]", " ", text)
        text=re.sub(r"(.)\1\1+", r"\1", text)
        
        words=''
        for word in text.split():
            if word not in stop_words:
                if len(word) > 1:
                    word=wordLemm.lemmatize(word)
                    words+=(word+' ')
        
        processedText.append(words)
        
    return processedText
    
def main():
    args = parse_arguments()
    cleaned_text = preprocess([args.input_text])
    tfidf_path = "./models/vectorizer.pickle"
    model_path = f"./models/{args.model_file}"
    out = ""

    with open(tfidf_path, 'rb') as file:
        vectorizer = pickle.load(file)
    
    X = vectorizer.transform(cleaned_text).toarray()
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    res = model.predict(X)
    if res[0] == 0:
        out = "Negative"
    else:
        out = "Posivive"       
    print(f"This sentence is {out}")
    print(f"Finished at: {round(time.time() - t)} sec")

if __name__ == "__main__":
    main()