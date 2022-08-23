import numpy as np
import pandas as pd
from itertools import permutations
import docx
from tika import parser
import re
import pickle
import nltk
import keras
import spacy

import ipdb; ipdb.set_trace()


from . import knowledge_graph_functions as kg
from ML.Full.Modules_spektral_custom.loaders import DisjointLoader
from ML.Full.cnn_gnn_functions import loss, Graphs_Dataset, tokenize_text
from spektral.layers import ECCConv, GlobalAvgPool

### LOADING MODELS ###
dir_path = 'ML/Full/'

# Loading spacy's model

spacy_model = spacy.load(dir_path + "Modeles_et_fichiers_intermediaires/Fine_tuned_spacy_model/")

# Loading prediction models
models_list = []
for i in range(10):
    model = keras.models.load_model(
        dir_path + 'Modeles_et_fichiers_intermediaires/cnn+gnn_nouvelle_variable_50_' + str(i) + '.h5'
        , custom_objects = {'ECCConv': ECCConv, 'GlobalAvgPool': GlobalAvgPool, 'loss': loss})
    models_list.append(model)


def find_technical_part_in_txt(contents, key_words):
            """"Function to find the technical parts in a text file."""

            # Find matching pattern for the titles we are interested in:
            match = []
            for item in list(permutations(key_words, 2)):
                for m in re.finditer(rf"[0-9][.][0-9]\s.*{item[0]}.*{item[1]}", contents.lower()):
                    match.append((m.start(), m))

            # If we have some matches and an even number of them (because of the summary), we proceed:
            if (len(match) > 0) & (len(match) % 2 == 0):
                # idx_list will contain the starting index of each match
                idx_list = []
                for item in match:
                    idx_list.append(item[0])
                # sorting the idx_list
                set_idx = sorted(list(set(idx_list)))
                # final_idx is the second half of set_idx
                final_idx = set_idx[int(len(set_idx) / 2):]

                text = []
                for item in final_idx:
                    try:
                        txt = contents[item:]
                        try:
                            # What if they forget to put annexes?
                            idx = txt.lower().index('annexes\n')
                            text.append(txt[:idx])
                        except ValueError:
                            text.append(txt)
                    except ValueError:
                        continue
                
                # new_text est la valeur de retour, its length will determoine the number of detected projects 
                new_text = []
                
                # if multiple projects have been detected, it is like processing the multiple projects
                if len(text) > 1:
                    # Why the last?
                    for item in text[:-1]:
                        # Converting a letter to an int?
                        int_idx = int(item[0])
                        final_idx = item.lower().index('\n' + str(int_idx + 1))
                        new_text.append(item[:final_idx])
                    new_text.append(text[-1])
                else:
                    new_text = text
            else:
                new_text = []
            return new_text


def extract_username(file_name):
    username = ""
    for char in file_name:
        if char == "_":
            break
        else:
            username += char
    return username

def delete_username(file_name, username):
    return file_name[len(username)+1:]

def read_file(file):
    """A function to read an inputed file, transform it into text and extract the parts of interest."""

    key_words = ['description', 'travaux', 'démarche', 'réalisés']
    texts_list = []
    
    # .txt:
    if file.content_type == "text/plain":
        file.open('r')
        full_text = str(file.read(), "utf-8")
        texts_list = find_technical_part_in_txt(full_text, key_words)

    # .pdf:
    elif file.content_type == "application/pdf":
        from datetime import datetime
        now = datetime.now()
        date = now.strftime("%Y/%m/%d")
        nom = file.name.replace(' ','_')
        chemin = "media/%s/%s/%s" %(date, extract_username(file.name), nom)

        full_text = parser.from_file(chemin)['content']
        texts_list = find_technical_part_in_txt(full_text, key_words)

    # .docx:
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
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

def import_and_predict(raw_text):
            """"Function wrapping up the prediction pipeline of the model."""

            # Preprocess our text:
            text_KG = kg.texts_preprocessing(raw_text)
            # Pass our text through Spacy's fine-tuned module:
            doc = spacy_model(text_KG)
            # Get Camembert's representation of the text :
            token_sentence_cam, output_cam = kg.camembert_representation(text_KG)
            # Extract tokens
            tokens_text, tokens_tags, tokens_ids = kg.tokens_extraction(doc)
            # Classify entities
            entities_labels = kg.entity_labelling(doc, tokens_text, tokens_tags, tokens_ids, token_sentence_cam,
                                                  output_cam)
            # Preprocess tokens
            tokens_text, tokens_tags, tokens_ids, entities_labels = kg.tokens_preprocessing(tokens_text, tokens_tags,
                                                                                            tokens_ids, entities_labels)
            # Extraction relations
            all_relations, all_relations_clusterised, tokens_text_sent, tokens_tags_sent, tokens_ids_sent, entities_labels_sent = kg.relation_extraction(
                doc, tokens_text, tokens_tags, tokens_ids, entities_labels)
            # Clean our relations
            all_relations_clusterised, negation = kg.relations_cleaning(all_relations_clusterised)
            # Transform our relation edges using our clusters :
            all_relations_clusterised = kg.clusterize_relations(all_relations_clusterised, negation)
            relations = pd.DataFrame(sum(all_relations_clusterised, []), columns=["sbj", "edges", "obj"])
            relations["edges"] = relations["edges"].apply(lambda x: [x])
            relations = relations.groupby(['sbj', 'obj'], as_index=False).agg(lambda x: sum(x, []))

            ###############          Representation CNN & GNN           ###############

            # Text pre-processing:
            text_prediction = re.sub(r'[^\w\s]', ' ', raw_text)
            text_prediction = re.sub('R D', '', text_prediction)
            text_prediction = nltk.word_tokenize(text_prediction, language='french')
            X_prediction = tokenize_text(text_prediction)
            graph = Graphs_Dataset(relations, X_prediction)
            graph_loader = DisjointLoader(graph)
            predictions = []
            for model in models_list:
                predictions.append(model.predict(graph_loader.load(), steps=graph_loader.steps_per_epoch))
            
            print("#################")
            print(predictions)
            print("#################")

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
            

def model(file):

    # Récupération de fichier
    texts_list = read_file(file)

    results_list = []
    

    for text in texts_list[0]:
        
        # Making predictions
        prediction = import_and_predict(text)

        prob_cir = prediction[0][0]
        montant_pred = prediction[0][1]

        # Displaying the first output of the model
        result = process_prob_cir(prob_cir)

        results_list.append([result, montant_pred])

    return results_list