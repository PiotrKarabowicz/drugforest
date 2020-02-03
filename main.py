from flask import Flask, request, render_template
from pymed import PubMed
import pandas as pd
import sklearn
import pickle

import nltk
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('url.html')

@app.route('/page1')
def page1():
    return render_template('req1.html')

@app.route('/page1', methods=['POST'])
def my_form_post():
    from pymed import PubMed
    text = request.form['text']
    max = 1000
    #count_vect.fit(text)
    text = [text]
    pubmed = PubMed(tool="MyTool", email="p.karabowicz@gmail.com")
    results1 = pubmed.query(text, max_results=max)

    lista_abstract_3=[]

    for i in results1:
        lista_abstract_3.append(i.abstract)
    import pandas as pd
    df_abstract = pd.DataFrame(lista_abstract_3, columns = ['abstracts'])
    df_abstract['abstracts_lower'] = df_abstract['abstracts'].str.lower()

    df_abstract_1 = df_abstract.dropna()
    import pickle
    rnd = pickle.load(open('C:\\Users\\UMB\\Desktop\\flaskapp\\static\\finalized_model_32.sav', 'rb'))
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100)
    result1=count_vect.fit_transform(df_abstract_1['abstracts_lower'])
    result2 = rnd.predict(result1)

    df_abstract_1['class'] = result2

    len_df = len(result2)
#unsupervised learning
    from gensim.models import Word2Vec
    from nltk.corpus import stopwords

    stop = stopwords.words('english')

    df_abstract_1['abstracts_stop'] = df_abstract_1['abstracts_lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df_abstract_1['tokenized'] = df_abstract_1.apply(lambda row: nltk.word_tokenize(row['abstracts_lower']), axis=1)
    model_ted1 = Word2Vec(sentences=df_abstract_1['tokenized'], size=200, window=10, min_count=1, workers=4, sg=0)

    embedding_clusters = []
    word_clusters = []

    embeddings = []
    words = []
    for similar_word, _ in model_ted1.wv.most_similar(positive = ['protein', 'target'], topn=30):
        words.append(similar_word)
        embeddings.append(model_ted1[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    from sklearn.manifold import TSNE
    import numpy as np

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


    #plt.show()

    #list list_protein
    keys = ['protein', 'target']
    most_sim = model_ted1.wv.most_similar(positive = keys, topn=1000)

    most_sim_key = []
    for w, n in most_sim:
        most_sim_key.append(w)

    post_tag_list = nltk.pos_tag(most_sim_key)

    listNN = []
    for w , k in post_tag_list:
        if k == "NN":
            listNN.append(w)

    listJJ = []
    for w , k in post_tag_list:
        if k == "JJ":
            listJJ.append(w)

    lista = [listNN, listJJ]

    flat_list = []
    for sublist in lista:
        for item in sublist:
            flat_list.append(item)

    file = open('C:\\Users\\UMB\\Desktop\\flaskapp\\static\\lista_bialek_bez1.txt', "r")
    wprowadzony_tekst = file.read()

    wt = wprowadzony_tekst.split(',')

    wt2 =[]
    for w in wt:
        w = w.replace("[", "")
        w = w.replace("]", "")
        w= w.replace("'", "")
        wt2.append(w)

    wt2 = [x.lower() for x in wt2 ]

    tablica_in =[]
    for w in flat_list:
        if w in wt2:
            tablica_in.append(w)

    # percent true
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    from pymed import PubMed
    import time

    #count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100)
    def query(list_target):
        pubmed = PubMed(tool="MyTool", email="p.karabowicz@gmail.com")
        lista=[]
        for w in list_target:
            time.sleep(1)
            lista.append(pubmed.query(w, max_results=20))
        return lista

    def lista_bastract_pred1(lista):
        lista_abstract_pred=[]
        for n in lista:
        #for k in n:
                lista_abstract_pred.append(n.abstract)
        return lista_abstract_pred

    def percent_true(ynew2):
        percent_true = round((list(ynew2).count(1))/len(list(ynew2))*100,1)
        return percent_true

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    import pickle
    rnd = pickle.load(open('C:\\Users\\UMB\\Desktop\\flaskapp\\static\\finalized_model_32.sav', 'rb'))

    a = query(tablica_in)
    d = []
    for w in a:
        b = lista_bastract_pred1(w) ## ziterować
        d.append(b)

    d2=[]
    for g in d:
        d1 = list(filter(None.__ne__, g))
        d2.append(d1)

    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100)
    lista_y=[]
    for w in d2:
        result1=count_vect.fit_transform(w)
        result2 = rnd.predict(result1)
        lista_y.append(result2)

    yy = []
    for y in lista_y:
        yy.append(percent_true(y))

    import pandas as pd
    data_tuples = list(zip(tablica_in,yy))
    df_list = pd.DataFrame(data_tuples, columns=['Protein','Probability'])

    df_list = df_list.sort_values('Probability', ascending = False)
    df_list = df_list.reset_index()

    df_abstract_1 = df_abstract_1[['abstracts', 'class']]
    #result3 = str(result2)
    embedding_clusters = []
    word_clusters = []

    embeddings = []
    words = []
    for similar_word, _ in model_ted1.wv.most_similar(positive = [tablica_in], topn=30):
        words.append(similar_word)
        embeddings.append(model_ted1[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    from sklearn.manifold import TSNE
    import numpy as np

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


    import matplotlib.pyplot as plt

    import matplotlib.cm as cm
    keys = tablica_in
    title = "Most similar for you request"
    labels = keys
    a=0.7
    filename = 'C:\\Users\\UMB\\Desktop\\flaskapp\\static\\output1.png'

    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    return render_template('index22.html', len_df = len_df,
                            lista_abstract=[df_abstract_1.to_html(classes='data')],
                            text=text, titles=df_abstract_1.columns.values,
                            df_list = [df_list.to_html(classes='data')],
                            titles1=df_list.columns.values)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
