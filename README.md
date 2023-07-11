Overview
--------
In this project we scrap data from 2 websites, namely The Washington Post and NFL News. We then dump the article data, specifically the title, author, date and body content, 
into a json file. The body content data is then lemmatized, stemmed and stripped of all stop words and dumped into a seperate field in the json file. Then using the json file 
and gensim we performed topic modeling. Additionally for each topic we found the 20 most likely words that occurred in that topic.

Data description
-----------------
Data obtained from the following sites.
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



