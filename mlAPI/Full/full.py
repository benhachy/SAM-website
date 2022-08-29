# importing the model
from ML.Full.ml import import_and_predict, process_prob_cir, find_technical_part_in_txt

class FullModel:
    def __init__(self):        
        pass

    '''
    def preprocessing(self, input_data):
        file = input_data.get("filename")
        return file
    '''

    def predict(self, input_data):
        return import_and_predict(input_data)

    def postprocessing(self, prediction):
        prob_cir = prediction[0][0]
        montant_pred = prediction[0][1]

        # Displaying the first output of the model
        nature = process_prob_cir(prob_cir)
        
        return {"nature":nature, "montant":montant_pred}


    def compute_prediction(self, input_data):
        
        if len(input_data)< 1500:
            return {"resultat": "La synthèse soumise par le fichier est trop courte pour être évaluée."}
        
        key_words = ['description', 'travaux', 'démarche', 'réalisés']
        nbr_proj = find_technical_part_in_txt(input_data, key_words)
        
        if len(nbr_proj) == 0:
            return {"resultat": "Attention ! Nous ne sommes pas parvenus à detecter la description technique dans votre rapport. "
                                             "Assurez-vous que votre fichier ne contient que cette dernière, ou procédez à ce "
                                             " découpage manuellement avant de continuer !"}

        #input_data = self.preprocessing(input_data)
        prediction = self.predict(input_data)
        prediction = self.postprocessing(prediction)
        
        return prediction