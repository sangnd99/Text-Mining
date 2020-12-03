from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import pickle
import time
import argparse

t = time.time()

def parse_argument():
    parser = argparse.ArgumentParser(
        description='Convert text string to vector'
    )
    parser.add_argument(
        '-i',
        '--input_file',
        type=str,
        metavar='',
        required=True,
        help='Name of preprocessed data'
    )

    parser.add_argument(
        '-o',
        '--output_file',
        type=str,
        metavar='',
        required=True,
        help='Name of output model (should name .pkl or .pickle)'
    )

    return parser.parse_args()

def read_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    text = df['text'].values.astype('U')
    target = df['target'].values
    return text, target

def main():
    args = parse_argument()
    input_path = f"./data/processed/{args.input_file}"
    output_path = f"./models/{args.output_file}"
    text, label = read_data(input_path)

    # Tf-idf transform
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    vectorizer.fit(text)
    with open('./models/vectorizer.pickle', 'wb') as file:
        pickle.dump(vectorizer, file)

    # Train, text split
    X_train, X_test, y_train, y_test = train_test_split(
        text,
        label,
        test_size= 0.2,
        random_state= 42,
        stratify= label
    )

    X_train_vec = vectorizer.transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    # Create and training model
    model = RandomForestClassifier(
        n_estimators= 1000, 
        min_samples_split= 8, 
        min_samples_leaf= 3, 
        max_features= 3, 
        max_depth= 110,
    )
    model.fit(X_train_vec, y_train)

    # # Output result
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)} \n")
    print(f"Confusion matrix: \n {confusion_matrix(y_test, y_pred)} \n")

    # Store model for later use
    with open(output_path, 'wb') as file:
        pickle.dump(model, file)

    print(f"Finished in: {round(time.time() - t)} sec")

if __name__ == "__main__":
    main()