import numpy as np
from itertools import permutations
import docx
from tika import parser
import re
import pickle
import nltk
nltk.download('punkt')
import keras
import os


def naivemodel(fichier):
    '''
    Simulation des résultats données par l'outil SAM "Synthesis Analysis Motor" de MyTeam qui reçoit en entrée un document et renvoie sa catégorie.  
    '''

    innovationp = "Très probablement innovation "
    innovation =  "Innovation mais pourrait passer en R&D "
    rd = "Probablement R&D "
    rdp = "Très probablement R&D "

    with open(fichier,'r') as f:
        content = f.readline()
    
    if content[0] == 'a':
        return rdp
    elif content[0] == 'b':
        return rd  
    elif content[0] == 'c':
        return innovation
    else:
        return innovationp


### LOADING MODELS ###

dir_path = '../'
models_list = []

def loss(y_true, y_pred):
                return keras.losses.binary_crossentropy(y_true[:, 0], y_pred[:, 0]) * 10000 + keras.losses.mean_absolute_error(y_true[:, 1], y_pred[:, 1])


# Loading the models
for i in range(10):
    model = keras.models.load_model(
        dir_path + 'ML/Modeles_et_fichiers_intermediaires/cnn_nouvelle_variable_50_' + str(
            i) + '.h5', custom_objects={'loss': loss})
    models_list.append(model)


def find_technical_part_in_txt(contents, key_words):
                """"Function to find the technical parts in a text file."""

                # Find matching pattern for the titles we are interested in:
                match = []
                for item in list(permutations(key_words, 2)):
                    for m in re.finditer(rf"[0-9][.][0-9]\s.*{item[0]}.*{item[1]}", contents.lower()):
                        match.append((m.start(), m))

                # If we have some matchs and a even number of them (because of the summary), we proceed:
                if (len(match) > 0) & (len(match) % 2 == 0):
                    idx_list = []
                    for item in match:
                        idx_list.append(item[0])

                    set_idx = sorted(list(set(idx_list)))
                    final_idx = set_idx[int(len(set_idx) / 2):]

                    text = []
                    for item in final_idx:
                        try:
                            txt = contents[item:]
                            try:
                                idx = txt.lower().index('annexes\n')
                                text.append(txt[:idx])
                            except ValueError:
                                text.append(txt)
                        except ValueError:
                            continue
                    new_text = []
                    if len(text) > 1:
                        for item in text[:-1]:
                            int_idx = int(item[0])
                            final_idx = item.lower().index('\n' + str(int_idx + 1))
                            new_text.append(item[:final_idx])
                        new_text.append(text[-1])
                    else:
                        new_text = text
                else:
                    new_text = []
                return new_text


def read_file(file):
    """A function to read an inputed file, transform it into text and extract the parts of interest."""

    key_words = ['description', 'travaux', 'démarche', 'réalisés']
    texts_list = []

    # .txt:
    if file.content_type == "text/plain":
        file.open('r')

        full_text = str(file.read(), "latin-1")
        texts_list = find_technical_part_in_txt(full_text, key_words)

    # .pdf:
    elif file.content_type == "application/pdf":
        from datetime import datetime

        print("#######************##########")
        print(file.name)
        print("#######************##########")

        now = datetime.now()
        date = now.strftime("%Y/%m/%d")
        nom = file.name.replace(' ','_')
        chemin = "media/%s/%s/%s" %(date, "benhachy", nom)

        full_text = parser.from_file(chemin)['content']
        texts_list = find_technical_part_in_txt(full_text, key_words)

    # .docx:
    elif file.content_type== "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Read the docx file and initialize some variables:
        doc = docx.Document(file)
        full_text = ""
        current_text = ""
        is_technical_part = False
        # Go through every paragraph:
        for para in doc.paragraphs:
            # If it's a title..
            if para.style.name.split(" ")[0] == "Heading":
                # ..and we are in the technical part..
                if is_technical_part:
                    # ..and its title level is higher (1>2 for titles): it's the end of the technical part.
                    if int(para.style.name.split(" ")[1]) < heading_level:
                        is_technical_part = False
                        texts_list.append(current_text)
                        current_text = ""
                # If we are not in the technical part: count the number of matches with our key words.
                else:
                    count_matchs = 0
                    for word in key_words:
                        if re.search(word, para.text.lower()):
                            count_matchs += 1
                    # If we have more than 2 matches, it is the beginning of the technical part:
                    if count_matchs > 1:
                        heading_level = int(para.style.name.split(" ")[1])
                        is_technical_part = True
            # If it is not a title, but we are in a technical part: add it to the text.
            if is_technical_part:
                current_text += para.text + "\n"
            full_text += para.text + "\n"

        # After the end of a technical part, add it to the list and proceed:
        if current_text != "":
            texts_list.append(current_text)

    technical_part_detected = len(texts_list)
    # If we haven't find the technical part, output the full inputed text:
    if technical_part_detected == 0:
        texts_list.append(full_text)
    return texts_list, technical_part_detected


def tokenize_text(text):
                # Load the entity classifier:

                open_file = open(dir_path + "ML/Modeles_et_fichiers_intermediaires/vocabulary_light", "rb")
                vocab = pickle.load(open_file)
                open_file.close()

                max_size = 15509
                X = np.zeros((1, max_size))
                for i, word in enumerate(text[:max_size]):
                    if word in vocab:
                        X[0, i] = vocab[word]
                    else:
                        X[0, i] = vocab['<UNK>']  # If the word isn't in the vocabulary, take the '<UNK>' token
                return X


def import_and_predict(raw_text):
                """"Function wrapping up the prediction pipeline of the model."""
                # Text pre-processing:
                text_prediction = re.sub(r'[^\w\s]', ' ', raw_text)
                text_prediction = re.sub('R D', '', text_prediction)
                text_prediction = nltk.word_tokenize(text_prediction, language='french')
                X_prediction = tokenize_text(text_prediction)

                predictions = []
                for model in models_list:
                    predictions.append(model.predict(X_prediction))
                return np.mean(predictions, axis=0)

def process_prob_cir(prob_cir):
            """Function to rework and display the first output of the model."""
            if prob_cir < 0.3:
                return("Très probablement innovation")
            elif prob_cir < 0.5:
                return("Innovation mais pourrait passer en R&D")
            elif prob_cir < 0.9:
                return("Probablement R&D ")
            else:
                return("Très probablement R&D")


def process_montant_pred(montant_pred, montant):
            """Function to rework and display the second output of the model."""
            if (montant > montant_pred / 3) & (montant < montant_pred * 3):
                return True
            else:
                return False
            

def light_model(file):

    # Récupération de fichier
    
    #with open(chemin ,'r') as doc:
    texts_list = read_file(file)
    results_list = []

    for text in texts_list[0]:

        print("###############################")
        print(texts_list)
        print("###############################")

        # application des models NLP
        prediction = import_and_predict(text[0])

        prob_cir = prediction[0][0]
        montant_pred = prediction[0][1]

        print("###############################")
        print(prob_cir)
        print(montant_pred)
        print("###############################")

        # proba to résultat
        result = process_prob_cir(prob_cir)
        
        print("###############################")
        print(result)
        print("###############################")

        results_list.append([result, montant_pred])

        print("###############################")
        print(results_list)
        print("###############################")


      

    return results_list