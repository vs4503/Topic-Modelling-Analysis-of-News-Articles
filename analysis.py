"""
Authors: Alejandrina Jimenez
        Vaibhav Santurkar

Analysis section of project 1
"""
import json
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

"""
Functions used for tasks 6-9
"""


def sentence_to_words(sentences):
    """
    Tokenizes the body of the articles
    :param sentences: Body of article
    :return: tokenized words
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=False))  # deacc=True removes punctuations


def compute_coherence_values(mallet_path, dictionary, corpus, texts, limit, start=2, step=3):
    """
    Computecoherence for various number of topics
    :param dictionary: Gensim dictionary
    :param corpus: Gensim corpus
    :param texts: List of input texts
    :param limit: Max num of topics
    :return: model_list : List of LDA topic models
             coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def dominant_topic_articles(ldamodel, corpus, titles, dates):
    """
    Returns a dataframe with the prediction of topics according to the articles body.
    :param ldamodel: Optimal model
    :param corpus: corpus
    :param titles: titles of the articles in a list
    :param dates: dates of the articles in a list
    :return: dataframe with the predictions
    """
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num)]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Topic']

    # Add Titles and dates of the articles to the end of the output
    sent_topics_df['Title'] = titles
    sent_topics_df['Date'] = dates

    return(sent_topics_df)


# Task 6:

with open('articles.json') as f:
   data = json.load(f)

df = pd.DataFrame(data['article'])

# Convert to list
data_rev = df.preprocessed.values.tolist()
data_titles = df.title.values.tolist()
data_date = df.date.values.tolist()
# tokenize preprocessed body
data_words = list(sentence_to_words(data_rev))

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# We use Mallet for this part
mallet_path_donwload = '/content/mallet-2.0.8/bin/mallet' # update this path according to the download
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path_donwload, corpus=corpus, num_topics=20, id2word=id2word)
model_list, coherence_values = compute_coherence_values(mallet_path= mallet_path_donwload, dictionary=id2word, corpus=corpus, texts=data_words,
                                                        start=2, limit=30, step=6)

# Show graph
limit=30; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Task 7: Choose the optimal model from graph
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
"""
this printed the following:
Num Topics = 2  has Coherence Value of 0.4601
Num Topics = 8  has Coherence Value of 0.4763
Num Topics = 14  has Coherence Value of 0.4591
Num Topics = 20  has Coherence Value of 0.4946
Num Topics = 26  has Coherence Value of 0.4931
"""
# From the previous output we selected the optimal model, in this case it was 20 topics,
# which is the 4th model of the list and is shown in the graph:
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)

# Task 8:
df_topic_words = pd.DataFrame()
list_topics = optimal_model.print_topics(num_words=20)
topic_id = list(range(0, 20))
topic_name = ['Public Health and Safety',
'Upcoming Fixtures',
'Football Statistics',
'Politics and Activisim',
'Predictive Analysis',
'Medical Treatment',
'Fantasy Football Statistics',
'Business and Travel',
'NFL Match',
'Journalism and Literature',
'Public Health',
'Lifestyle',
'Jobs and Employment',
'Law Enforcement and Violence',
'Player Trading Market',
'Food Prep',
'Literature and Publishing',
'Public Education',
'Lifestyle',
'Environment']

# get 20 most common words and its weights
for x in range(0, 20):
    df_topic_words[str(topic_id[x]) + ':' + topic_name[x]] = list_topics[x][1].split('+')
# Generate table for the LaTex Report
print(df_topic_words.to_latex(index=False))

# task 9
df_topic_sents_keywords = dominant_topic_articles(ldamodel=optimal_model, corpus=corpus,
                                                  titles= data_titles,dates =data_date)
# Generate table for the LaTex Report
print(df_topic_sents_keywords.to_latex(index=False))


