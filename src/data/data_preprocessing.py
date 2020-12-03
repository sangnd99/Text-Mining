import argparse
import pandas as pd
import re
import time
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

t = time.time()

nltk.download('stopwords')
nltk.download('wordnet')

stop_words=set(stopwords.words('english')) 

emojis={':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
              ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

def parse_arguments():
    parser=argparse.ArgumentParser(
        description='Data preprocessing'
    )

    parser.add_argument(
        '-i',
        '--input_file',
        type=str,
        metavar='',
        required=True,
        help='Name of input data file'
    )

    parser.add_argument(
        '-o',
        '--output_file',
        type=str,
        metavar='',
        required=True,
        help='Name of output data file'
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
    input_path = f"./data/raw/{args.input_file}"
    output_path = f"./data/processed/{args.output_file}"
    df = pd.read_csv(input_path, encoding='ISO-8859-1')
    df = df[['target','text']]
    df['target'].replace(4,1)

    text, labels = list(df['text']), list(df['target'])

    processedText = preprocess(text)
    print(f"Finished time: {round(time.time() - t)} sec")

    res = {
        'text':processedText,
        'target':labels
    }
    out = pd.DataFrame(res)
    out.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()