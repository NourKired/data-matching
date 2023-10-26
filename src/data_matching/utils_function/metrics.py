from gensim.models import Word2Vec
import os
import multiprocessing as mp
from sklearn.decomposition import PCA
from EmbDI.utils import *
from EmbDI.graph import graph_generation
from EmbDI.sentence_generation_strategies import random_walks_generation
from EmbDI.embeddings import learn_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import torch


def walk(walk_strategy, snt_lgt, file):
    """
    Génère les marches aléatoires pour un fichier d'entrée donné selon la stratégie de marche spécifiée.

    Args:
        walk_strategy (str): La stratégie de marche à utiliser.
        snt_lgt (int): La longueur des phrases de marche.
        file (str): Le fichier d'entrée contenant les informations du graphe.

    Returns:
        str: Le nom du fichier généré contenant les marches aléatoires.
    """
    print("path",os.path.join("./1st_edgelist",file.replace("test",""))+".txt")
    configuration = {
        'walk_strategy': walk_strategy,
        'flatten': 'all',
        'input_file': os.path.join("./1st_edgelist",file.replace("test",""))+".txt",
        'n_sentences': 'default',
        'sentence_length': snt_lgt,
        'write_walks': True,
        'intersection': False,
        'backtrack': True,
        'output_file': file+'_'+str(snt_lgt)+"_"+walk_strategy,
        'repl_numbers': False,
        'repl_strings': False,
        'follow_replacement': False,
        'mlflow': False
    }
    prefixes, edgelist = read_edgelist(configuration['input_file'])
    print("edgelist,",edgelist)
    graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
    print("graph",graph)
    if configuration['n_sentences'] == 'default':
        #  Compute the number of sentences according to the rule of thumb.
        configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))
    walks = random_walks_generation(configuration, graph)
    print("walks",walks)
    return file+'_'+str(snt_lgt)+"_"+walk_strategy


def embedding(n_dim, window_size, training_algorithm, learning_method, walk_file):
    """
    Utilise l'algorithme EMBDI pour apprendre les embeddings des données à partir des marches aléatoires générées.
    Args:
        n_dim (int): La dimension des embeddings à apprendre.
        window_size (int): La taille de la fenêtre pour le contexte des mots dans EMBDI.
        training_algorithm (str): L'algorithme d'apprentissage utilisé dans EMBDI.
        learning_method (str): La méthode d'apprentissage utilisée dans EMBDI.
        walk_file (str): Le fichier contenant les marches aléatoires.

    Returns:
        None
    """
    input_name_dataset_file=os.path.basename(walk_file)[:-6]
    file_input_name_dataset = input_name_dataset_file+'.emb' 
    with open(file_input_name_dataset, 'w') as file:
        file.write('')
    output_embeddings_file=os.path.join("./embeddings",file_input_name_dataset)
    n_walk=os.path.join("./pipeline/n_walk",walk_file)
    write_n_walk=True
    learn_embeddings(
        output_embeddings_file,
        walk_file,
        write_n_walk,
        n_dim,
        window_size,
        training_algorithm=training_algorithm,
        learning_method=learning_method,
        workers=mp.cpu_count(),
        sampling_factor=0.001,
    )


def Matrice_de_passage(A1, A2):
    """
    Calcule la matrice de passage entre deux ensembles de données dans un espace vectoriel.

    Args:
        A1 (np.array): Le premier ensemble de données.
        A2 (np.array): Le deuxième ensemble de données.

    Returns:
        np.array: La matrice de passage.
        np.array: Les vecteurs propres de A1.
        np.array: Les vecteurs propres de A2.
        np.array: Les valeurs singulières de A1.
        np.array: Les valeurs singulières de A2.
    """
    U1, s1, V1 = np.linalg.svd(A1)
    U2, s2, V2 = np.linalg.svd(A2)
    matrice_passageu1_u2 = np.dot(U1.T,U2)
    return matrice_passageu1_u2,U1,U2,V1,V2,s1,s2



def get_embeddings_Embdi(df_KG,filepath):
    """
    Récupère les embeddings des données à partir des résultats de l'algorithme EMBDI.

    Args:
        df_KG (pd.DataFrame): Le dataframe contenant les données.
        input_name_dataset_dataset (str): Le nom du dataset.
        window_size (int): La taille de la fenêtre pour le contexte des mots dans EMBDI.
        nbwalk (int): Le nombre de marches aléatoires générées.
        n_dim (int): La dimension des embeddings appris.
        training_algorithm (str): L'algorithme d'apprentissage utilisé dans EMBDI.
        learning_method (str): La méthode d'apprentissage utilisée dans EMBDI.
        walk_strategy (str): La stratégie de marche utilisée.
        filepath (str): Le chemin du fichier contenant les embeddings.

    Returns:
        list: Les labels des attributs.
        np.array: Les embeddings des attributs.
    """
    embeddings = []
    labels=[]
    try:
        with open("./embeddings/"+filepath, "r") as file:
            for i,line in enumerate(file):
                if i!=0:
                    values = line.strip().split()
                    node = values[0]
                    if node.startswith("cid") :
                        labels.append(node[5:])
                        embeddings.append(values[1:])       
                    # if node.startswith("tt") and node[5:] in df_KG.values :
                    #     labels.append(node[4:])
                    #     embeddings.append(values[1:])                    
        embeddings=np.asarray(embeddings,dtype=float)
        embeddings.astype(float)
        return labels,embeddings
    except:
        return [],[]

def get_emb(my_list, n, k):
    """
    Retourne une liste d'éléments aléatoires parmi une liste donnée.

    Args:
        my_list (list): La liste d'éléments d'origine.
        n (int): Le nombre total d'éléments de la liste d'origine.
        k (int): Le nombre d'éléments à retirer.

    Returns:
        list: La liste d'éléments aléatoires résultante.
    """
    y_true=np.ones(n)
    indexes_to_remove = list(np.random.randint(0, n,size=k))
    indexes_to_remove.sort(reverse=True)  
    for index in indexes_to_remove:
        y_true[index]=0
        np.delete(my_list, index)
    return my_list,y_true,indexes_to_remove


def get_emb_model(model, tokenizer, text):
    """
    Récupère les embeddings d'un texte à partir d'un modèle pré-entraîné.

    Args:
        model (torch model): Le modèle pré-entraîné.
        tokenizer (tokenizer): Le tokenizer associé au modèle.
        text (str): Le texte à vectoriser.

    Returns:
        list: Les embeddings du texte.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0) 
    embeddings = model(input_ids)[0]
    embedding_1d = torch.mean(embeddings, dim=1) 
    return list(embedding_1d.detach().numpy()[0])

def get_embeddings(df1, n_dim, filepath,model,tokenizer):
    """
    Récupère les embeddings des attributs de l'ontologie et du dataset à partir des résultats de l'algorithme EMBDI.

    Args:
        df1 (pd.DataFrame): Le dataframe contenant les données de l'ontologie.
        input_name_dataset (str): Le nom du dataset.
        window_size (int): La taille de la fenêtre pour le contexte des mots dans EMBDI.
        n_walk (int): Le nombre de marches aléatoires générées.
        n_dim (int): La dimension des embeddings appris.
        training_algorithm (str): L'algorithme d'apprentissage utilisé dans EMBDI.
        learning_method (str): La méthode d'apprentissage utilisée dans EMBDI.
        walk_strategy (str): La stratégie de marche utilisée.
        filepath (str): Le chemin du fichier contenant les embeddings.

    Returns:
        tuple: Un tuple contenant les embeddings, les vrais labels, les labels prédits et la matrice d'embedding.
    """
    labels,emb_data_input=get_embeddings_Embdi(df1,filepath)  
    print(labels,"labels")
    print(emb_data_input.shape,"emb_data_input")
    if len(emb_data_input)!=0:
        # update_and_save_word2vec_model(i,combinliste(list(labels),3))
        bert_data_input=[]
        for lab in labels:
            l=get_emb_model(model,tokenizer,lab)
            # l=get_word2vec(lab)    
            bert_data_input.append(list(l))
        emb_data_input=np.array(emb_data_input).astype(float)
        bert_data_input=np.array(bert_data_input).astype(float)
        try:
            pca = PCA(n_components=n_dim)
            pca.fit(bert_data_input)
            bert_data_input = np.array(pca.transform(bert_data_input)).astype(float)
        except :
            l=np.array(bert_data_input).shape[0]
            pca1 = PCA(n_components=l)
            pca2 = PCA(n_components=l)
            pca1.fit(bert_data_input)
            bert_data_input = np.array(pca1.transform(bert_data_input)).astype(float)
            pca=pca1
            pca2.fit(emb_data_input)
            emb_data_input = np.array(pca2.transform(emb_data_input)).astype(float)
        print(emb_data_input.shape,bert_data_input.shape)
        return emb_data_input,bert_data_input,labels,pca
    else:
        return None    


def detect_similar_attributes(df1,df2, n_dim, filepath,model,tokenizer):
    """
    Détecte les attributs similaires entre un ensemble d'attributs et les attributs du dataset D.

    Args:
        df1 (pd.DataFrame): Le dataframe contenant les données de l'ontologie.
        dataset_input_name_dataset (str): Le nom du dataset.
        window_size (int): La taille de la fenêtre pour le contexte des mots dans EMBDI.
        n_walk (int): Le nombre de marches aléatoires générées.
        n_dim (int): La dimension des embeddings appris.
        training_algorithm (str): L'algorithme d'apprentissage utilisé dans EMBDI.
        learning_method (str): La méthode d'apprentissage utilisée dans EMBDI.
        walk_strategy (str): La stratégie de marche utilisée.
        filepath (str): Le chemin du fichier contenant les embeddings.

    Returns:
        tuple: Un tuple contenant les embeddings, les vrais labels, les labels prédits et la matrice d'embedding.
    """
    l=get_embeddings(df1,n_dim,filepath,model,tokenizer)
    all_detected=[]
    y_pred=[]
    y_true=[]
    if l is not None :
        print(l,"l")
        emb_data_input,bert_data_input,labels,pca=l # emb_data_input = vecteurs embdi des Données input tabulaire   
        # / bert_data_input = vecteurs du modele des Données input tabulaire 
        indexes_to_remove = np.random.randint(0, len(labels),size=0)
        
        selected_labels = [labels[i] for i in range(len(labels)) if i not in indexes_to_remove]
        selected_emb_data_input= [emb_data_input[i] for i in range(len(emb_data_input)) if i not in indexes_to_remove]
        selected_bert_data_input= [bert_data_input[i] for i in range(len(bert_data_input)) if i not in indexes_to_remove]
        
        P,U1,U2,V1,V2,s1,s2=Matrice_de_passage(A1=selected_bert_data_input,A2=selected_emb_data_input)
        k=0
        # cols=[col for col in df2.columns if col  in selected_labels]
        cols_=df2.columns 
        y_true=[1 if lab in selected_labels else 0 for lab in cols_]
        k=len(cols_) # ENSEMBLE D'ATTRIBUTS
        p=len(selected_labels) # ATTRIBUS DATA INPUT
        diff=k-p
        cols_=list(cols_)
        if diff>0:
            to_add=p-int(k%p)
            for i in range(to_add):
                cols_.append("none")
        M_labKG_A2_tot=[]
        for it in range((len(cols_)//p)):
            M_labKG : list =[]
            cols_it=cols_[it*p:it*p+p]
            print("cols_it",cols_it)
            for labKG_td in cols_it:
                if labKG_td not in selected_labels:
                    pass
                else:
                    v_labKG_td=np.array(get_emb_model(model,tokenizer,labKG_td)).reshape(1, -1)
                    M_labKG.append(v_labKG_td)    
            M_labKG=np.array(M_labKG)
            M_labKG=M_labKG.reshape(M_labKG.shape[0], M_labKG.shape[-1])
            M_labKG_A1=pca.transform(M_labKG)
            # M_labKG_A1=M_labKG
            M_labKG_U1=np.dot(M_labKG_A1,U1)
            M_labKG_U2=np.dot(M_labKG_U1,P)
            M_labKG_A2=np.dot(M_labKG_U2, V2)
            if diff>0:
                if it ==len(cols_)//p-1:
                    M_labKG_A2=M_labKG_A2[:-diff]
            for v_attr_ea,label_ea in zip(M_labKG_A2,cols_it):
                cos_sim_values=cosine_similarity(v_attr_ea.reshape(1, -1), selected_emb_data_input)[0]
                indices = list(np.where(cos_sim_values >= 0.3)[0])
                detected_labels = [selected_labels[i] for i in indices]
                all_detected.append(detected_labels)
                if label_ea not in selected_labels: # je ne veux pas le detecter (FP / FN)
                    if len(detected_labels)>0:
                        y_pred.append(0)   
                    else:
                        y_pred.append(1)   
                else:# je veux le detecter (TP / TN)
                    if label_ea in detected_labels:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
            M_labKG_A2_tot.append(M_labKG_A2)
        return l,y_true,y_pred,np.array(M_labKG_A2),indexes_to_remove,M_labKG_A2
    else:
        return None


def get_matches(df1,df2, walk_strategy, n_walk, input_name_dataset, n_dim, window_size, training_algorithm, learning_method,model,tokenizer):
    """
    Fonction principale qui coordonne l'ensemble du processus de calcul des embeddings et de détection des attributs similaires.

    Args:
        df1 (pd.DataFrame): Le dataframe contenant les données de l'ontologie.
        walk_strategy (str): La stratégie de marche utilisée.
        n_walk (int): Le nombre de marches aléatoires générées.
        input_name_dataset (str): Le nom du dataset.
        n_dim (int): La dimension des embeddings appris.
        window_size (int): La taille de la fenêtre pour le contexte des mots dans EMBDI.
        training_algorithm (str): L'algorithme d'apprentissage utilisé dans EMBDI.
        learning_method (str): La méthode d'apprentissage utilisée dans EMBDI.

    Returns:
        tuple: Un tuple contenant les résultats de l'expérimentation.
    """
    # walk_path=walk(walk_strategy,n_walk,input_name_dataset)
    walk_path=input_name_dataset+'_'+str(n_walk)+"_"+walk_strategy
    walk_file=walk_path+".n_walk"
    embedding(n_dim,window_size,training_algorithm,learning_method,walk_file)
    filepath=os.path.basename(walk_file)[:-6]+".emb"
    b,y_true,y_pred,M_E_Emb,indexes_to_remove,M_E_MOD=detect_similar_attributes(df1,df2,n_dim,filepath,model,tokenizer)
    precision,recall,f1_Score = precision_score(y_true, y_pred),recall_score(y_true, y_pred),f1_score(y_true, y_pred)
    new_row = {'dataset': input_name_dataset,'walk_strategy':walk_strategy, 'n_walk':n_walk,'dimension':n_dim,'window':window_size,'learning_methd':learning_method,'training_algorithms':training_algorithm,'F1_score':f1_Score,'recall':np.round(recall,2),'precision':np.round(precision,2),"indexes_to_remove": str(indexes_to_remove),"model_name": model.config.name_or_path}
    return  b,new_row,M_E_Emb,M_E_MOD






def combinliste(seq, k): 
    '''
    Doc: cette fonction renvoie en sortie les differents combinaison de k elem parmis seq 
    utile pour generer les diffentes combinaision des dimensions pour les plots
    '''
    p = []
    i, imax = 0, 2**len(seq)-1
    while i<=imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j<=jmax:
            if (i>>j)&1==1:
                s.append(seq[j])
            j += 1
        if len(s)==k:
            p.append(s)
        i += 1 
    return p




def update_and_save_word2vec_model(i, listes):
    if i==0:
        model = Word2Vec.load("modele.bin")
    else:
        model = Word2Vec.load("modele2.bin")
    model.build_vocab(listes, update=True)
    model.train(listes, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("modele2.bin")

def get_word2vec(text):
    model = Word2Vec.load("modele2.bin")
    embedding = model.wv[text]
    return embedding

