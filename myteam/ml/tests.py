from django.test import TestCase
from ml.Full.full import FullModel



class MLTests(TestCase):
    def test_nlp_algorithm(self):
        
        # path to the test files
        path = r"ml/ROUX.docx"
        
        # Input Data
        with open(path,'r') as test_file:
            input_data = test_file

        # Making predictions
        alg = FullModel()
        response = alg.compute_prediction(input_data)
        
        # Making Assertions
        self.assertEqual("Error", response["status"])
        #self.assertEqual("Très probablement R&D", response["nature"])

        #self.assertEqual(40277.3671875, response["montant"])


