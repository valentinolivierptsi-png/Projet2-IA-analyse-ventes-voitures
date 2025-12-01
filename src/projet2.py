import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import MEstimateEncoder
# Configuration Graphique
sns.set_style("whitegrid")
sns.set_context("talk")


class DataCleaner:
    def __init__(self, df):
        self.df = df.copy() # Copie de sécurité pour ne pas modifier l'original

    def clean_prix(self):
        print("Nettoyage colonne Prix")
        extract = self.df["Prix"].str.extract(r'([\d\.,\s]+)\s*([kK]?)', expand=True) # on crée une vairable extract qui a deux colonnes une avec nos nombre et une avec k ou K expand True on ne prend pas le reste genre $ euro...

        extract[0] = extract[0].str.replace(r'\s', '', regex=True).str.replace(',', '.')#on en leve les espace et on remplace les virgule par des point comprehensible en python pour flaot 

        self.df['Prix'] = pd.to_numeric(extract[0], errors='coerce') # on numerise en float notre premiere colonne avec les chiffre  , si erreur == Na 

        filtre_k = extract[1].str.lower() == 'k' # on crée un filtre si deuxieme colonnes == k ou K, car lower donc k devient forcement minuscule
        
        self.df.loc[filtre_k, 'Prix'] = self.df.loc[filtre_k, 'Prix'] * 1000 # prend les lignes et colonnes avec k ou  K et les multiplie par 1000

        return self.df

    def clean_km(self):
        print("Nettoyage colonne Kilometrage")
        # On utilise une regex qui capture plusieurs chiffres, potentiellement séparés par des espaces.
        extract_km = self.df["Kilometrage"].str.extract(r'(\d[\d\s]*)', expand=False)  # On utilise regex qui capture plusieurs chiffres potentiellement separes par des espaces et expand false nous renvoie une series
        
        km_clean = extract_km.str.replace(r'\s', '', regex=True)# On enlève les potentiels espace avec regex True sinon croit que c'est une chaine str

        self.df["Kilometrage"] = pd.to_numeric(km_clean, errors="coerce")
        return self.df
        
    
    def clean_puissance(self):
        print("Nettoyage colonne Puissance")
        
        extract_ch = self.df["Puissance"].str.extract(r"(\d+)",expand= False)# on extrait le nombre de cheveaux 
        ch_clean= extract_ch.str.replace(r'\s',"",regex =True) # pas besoin mais on sait jamais 
        self.df["Puissance"] = pd.to_numeric(ch_clean,errors="coerce")
        
        return self.df
    
    def create_age(self):
        print("Création de la variable 'Age'")
        self.df['Age'] = 2025 - self.df['Annee']
        return self.df

    def drop_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Doublons supprimés : {before - len(self.df)}") # on affiche cb de ligne on a enlevé 
        return self.df

    def get_clean_df(self):
        self.drop_duplicates()
        self.clean_prix()
        self.clean_km()
        self.clean_puissance()
        self.create_age()
        return self.df


class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def check_missing(self):
        print("\nValeurs Manquantes :")
        print(self.df.isna().sum())

    def check_outliers(self):
        print(self.df.describe())


class DataVisualizer:
    def __init__(self, df):
        self.df = df
    def plot_distribution(self, col):
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[col], kde=True)
        plt.title(f"Distribution : {col}")
        plt.show()

    def plot_boxplot(self, col):
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=self.df[col])
        plt.title(f"Boxplot : {col}")
        plt.show()


class MLPipeline:
    def __init__(self, path):
        self.path = path
        self.df = None

    def run(self):
        print("Démarrage du Pipeline Projet 2")

            
        self.df = pd.read_csv(self.path)
        print(f"Données chargées : {self.df.shape}")

        print("\nINSPECTION INITIALE :")
        print("Colonnes :", self.df.columns.tolist()) 
        print(self.df.head()) 
        
        # Nettoyage (Appel Cleaner)
        print("\nPHASE 1 : NETTOYAGE")
        cleaner = DataCleaner(self.df)
        self.df = cleaner.get_clean_df()

        # Analyse (Appel Analyzer)
        print("\nPHASE 2 : ANALYSE RAPIDE")
        analyzer = DataAnalyzer(self.df)
        
        analyzer.check_missing()
        analyzer.check_outliers()

        #  voir si les regex ont marché
        print("\nTypes des colonnes actuels :")
        print(self.df.dtypes)

        # Visualisation (Appel Visualizer)
        print("\n-PHASE 3 : VISUALISATION ")
        viz = DataVisualizer(self.df)
        
        viz.plot_distribution('Prix')
        viz.plot_distribution('Kilometrage')

        # Preprocessing ML 
        print("\nPHASE 4 : PREPROCESSING ML")
        print("Split Train/Test")
        self.df = self.df.dropna(subset=['Prix']) # on ne peut pas apprendre si on n'a pas la réponse (Prix)
        X = self.df.drop(columns = ["Prix","Annee"]) # on enleve la reponse et l'annee car on a age mtn 
        y = self.df["Prix"] 
        self.X_train , self.X_test , self.y_train ,self.y_test = train_test_split(X,y,test_size=(0.2), random_state = 42) # on fait cela pour eviter le data leakage avannt notre encoding.
        print("train set",self.X_train.shape)
        print("test set",self.X_test.shape)
        
        print("Encodage & Scaling")
        imputer = SimpleImputer(strategy = "median")# pour les chiffres -> Médiane
        imputer2 = SimpleImputer(strategy="most_frequent")# pour le texte -> Le plus fréquent
        num_col = self.X_train.select_dtypes(include="number").columns# on identifie les colonnes par type
        obj_cols = self.X_train.select_dtypes(include=["object"]).columns# on identifie les colonnes par type
        self.X_train[num_col] = imputer.fit_transform(self.X_train[num_col])# On bouche les trous des colonnes numériques attention au fit sur test !!! 
        self.X_test[num_col] = imputer.transform(self.X_test[num_col])
        self.X_train[obj_cols] = imputer2.fit_transform(self.X_train[obj_cols])
        self.X_test [obj_cols] = imputer2.transform(self.X_test [obj_cols])
        
        encoder = MEstimateEncoder(cols=obj_cols,m =10.0)# Target Encoding : Remplace la marque par le prix moyen de cette marque. 
        self.X_train = encoder.fit_transform(self.X_train,self.y_train)# on apprend les prix moyens sur le train avec y_train
        self.X_test=encoder.transform(self.X_test)# on applique les prix moyens appris sur le TEST
        scaler = RobustScaler() # RobustScaler sur mediane car permet d'ignorer  les outliers 
        self.X_train=pd.DataFrame(scaler.fit_transform(self.X_train),columns=self.X_train.columns,index=self.X_train.index) # on fait un data frame car sinon cela renvoie un array moche
        self.X_test=pd.DataFrame(scaler.transform(self.X_test),columns=self.X_test.columns,index=self.X_test.index) #same 
        print(self.X_train.head())
        print("\nFichiers prêts.")
        


if __name__ == "__main__":
    input_file = "data/voitures_sales.csv"
    pipeline = MLPipeline(input_file)
    pipeline.run()