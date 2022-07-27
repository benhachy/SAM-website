from django.test import TestCase
from ml.Full.full import FullModel



class MLTests(TestCase):
    def test_sam_cii(self):
        
        # path to the test file
        path = r"ml/TestFiles/HOLYDIS.txt"
        
        # Input Data
        with open(path,'r') as test_file:
            input_data = test_file.read()

        # Making predictions
        alg = FullModel()
        
        response = alg.compute_prediction(input_data)
    
        # Making Assertions
        self.assertEqual("Innovation mais pourrait passer en R&D", response["nature"])
        
    def test_sam_cir(self):
        
        # path to the test file
        path = r"ml/TestFiles/ROUX.txt"
        
        # Input Data
        with open(path,'r', encoding='utf-8') as test_file:
            input_data = test_file.read()

        # Making predictions
        alg = FullModel()
        
        response = alg.compute_prediction(input_data)
    
        # Making Assertions
        self.assertEqual("Très probablement R&D", response["nature"])

    def test_sam_full(self):
        # path to the test files
        path = r"ml/TestFiles/Rempli.txt"
        
        # Input Data
        with open(path,'r') as test_file:
            input_data = test_file.read()

        # Making predictions
        alg = FullModel()
        
        response = alg.compute_prediction(input_data)

        expected = 'Attention ! Nous ne sommes pas parvenus à detecter la description technique dans votre rapport. Assurez-vous que votre fichier ne contient que cette dernière, ou procédez à ce  découpage manuellement avant de continuer !'

        # Making Assertions
        self.assertEqual(expected, response["resultat"])


    def test_sam_empty(self):
            # path to the test files
            path = r"ml/TestFiles/Vide.txt"
            
            # Input Data
            with open(path,'r') as test_file:
                input_data = test_file.read()

            # Making predictions
            alg = FullModel()
            
            response = alg.compute_prediction(input_data)

            expected = "La synthèse soumise par le fichier est trop courte pour être évaluée."

            # Making Assertions
            self.assertEqual(expected, response["resultat"])