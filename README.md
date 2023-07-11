#project-1-csci-720_project1

Topic Modelling Analysis of News Articles
=========================================

Name of Participants
--------------------
* Alejandrina Jimenez Guzman (aj7354)
* Vaibhav Santurkar (vs4503)

Overview
--------
In this project we scrap data from 2 websites, namely The Washington Post and NFL News. We then dump the article data, specifically the title, author, date and body content, 
into a json file. The body content data is then lemmatized, stemmed and stripped of all stop words and dumped into a seperate field in the json file. Then using the json file 
and gensim we performed topic modeling. Additionally for each topic we found the 20 most likely words that occurred in that topic.

Data description
-----------------
We obtained the data from the following sites, and scrapped a total of approximately 100-105 articles.
1) The Washington Post: https://www.washingtonpost.com/
2) NFL News: https://www.nfl.com/news/

Python Modules Used
--------------------
1) Requests
2) BeautifulSoup4
3) JSON
4) NLTK
5) Pandas
6) Gensim

Setup
------
Both programs do not require arguements to run. The **articles.json** file needs to kept in the same directory as the programs to be correctly loaded and read. 

We used Mallet to get a better quality of topics. Gensim provides a wrapper to implement Mallet’s LDA from within Gensim itself. We only need to download the zipfile provided in this repo, unzip it and provide the path to mallet in the unzipped directory to gensim.models.wrappers.LdaMallet(). 


