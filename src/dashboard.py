import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from projet2 import DataCleaner, MLPipeline

#pour lancer taper cette commande :
#streamlit run src/dashboard.py

# Titre simple
st.title("Projet 2 : Analyse Voitures")

# On charge le fichier CSV directement
# On suppose que le fichier est bien là
path = "data/voitures_sales.csv" 

# Lecture du fichier
df = pd.read_csv(path)

# Menu sur le côté
menu = st.sidebar.radio("Menu", ["Nettoyage", "Graphiques", "Machine Learning"])

# --- PAGE 1 : NETTOYAGE ---
if menu == "Nettoyage":
    st.header("1. Nettoyage des données")
    
    st.write("Données brutes :")
    st.write(df.head()) # Affiche le début du tableau

    # Le bouton pour lancer le nettoyage
    if st.button("Nettoyer les données"):
        # appelle la classe que j'ai codée dans projet2.py
        cleaner = DataCleaner(df)
        df_clean = cleaner.get_clean_df()
        
        #  sauvegarde le résultat pour ne pas le perdre quand je change de page
        st.session_state['df_clean'] = df_clean
        
        st.success("Nettoyage terminé !")
        st.write("Données propres :")
        st.write(df_clean.head())

# PAGE 2 : GRAPHIQUES
elif menu == "Graphiques":
    st.header("2. Visualisation")
    
    # On vérifie si on a bien nettoyé avant
    if 'df_clean' in st.session_state:
        df_viz = st.session_state['df_clean']
        
        st.subheader("Histogramme des Prix")
        fig = plt.figure()
        sns.histplot(df_viz['Prix'], kde=True)
        st.pyplot(fig)
        
        st.subheader("Prix selon le Km")
        fig2 = plt.figure()
        sns.scatterplot(data=df_viz, x='Kilometrage', y='Prix', hue='Carburant')
        st.pyplot(fig2)
    else:
        st.warning("Il faut lancer le nettoyage dans l'onglet 1 d'abord !")

# PAGE 3 : ML
elif menu == "Machine Learning":
    st.header("3. Entraînement du Modèle")
    
    if st.button("Lancer l'entraînement"):
        st.write("Lancement du pipeline...")
        
        # utilise la classe MLPipeline
        pipeline = MLPipeline(path)
        pipeline.run()
        
        st.success("Terminé !")
        st.write("Voici à quoi ressemblent les données prêtes pour l'IA (X_train) :")
        st.dataframe(pipeline.X_train.head())