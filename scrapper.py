# -*- coding: utf-8 -*-
"""
@author: Alejandrina Jimenez Guzman (aj7354@rit.edu), Vaibhav Santurkar (vs4503@rit.edu)
"""
"""
This program scraps data from 2 websites and dumps the data into json. Later the data
is stemmed, lemmatized and stripped of all stop words for further processing. 
"""

# Importing required libraries
import requests
from bs4 import BeautifulSoup
import re
import json
import nltk
import string
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# Creating an Exception class name ServerNotFound
class ServerNotFound(Exception):
    """ Raises when website server did not respond """
    pass

# FOR WASHINGTON POST---
def get_link(url,header):
    
    # Making HTTP request and downloading the response in "response" variable
    response = requests.request("GET",url, headers=header)
    if not response:
        print("\nServerNotFound")
    
    else:
        
        # Parse the links from the downloaded response
        
        article_links = []
        
        link_data = BeautifulSoup(response.text,'html.parser')
        
        for link in link_data.findAll('h2', {'class' : ["font--headline font-size-lg font-bold left relative", 
                                                        "font--headline font-size-sm font-bold left relative"]}):
            for link_tag in link.findAll('a', href = re.compile("https://www.washingtonpost.com/")):
                article_links.append(link_tag['href'])
    
        return article_links
    return None

# FOR NFL NEWS---
def get_link_nfl(url, header):
    
    #Repeat the same process as before with the NFL site structure
    response = requests.request("GET",url, headers=header)
    if not response:
        print("\nServerNotFound")
    
    else:
        nfl_article_links = []
        
        nfl_link_data = BeautifulSoup(response.text, 'html.parser')
        
        for link in nfl_link_data.findAll('div', class_ = "d3-o-media-object d3-o-media-object--vertical d3-o-content-tray__card"):
            for link_tag in link.findAll('a', href = re.compile("/news/")):
                nfl_article_links.append(link_tag['href'])
        
        return nfl_article_links
    return None

def get_data(download_url, header):
    #Extract the url of the article itself and parse the html data from the page
    page_response = requests.get(download_url, headers=header)
    if not page_response:
        print("\nPageNotFound")
    
    else:
        page_data = BeautifulSoup(page_response.text, 'html.parser')
        return page_data
    return None

# Scraping the title of the page

# FOR WASHINGTON POST---
def scrap_title(page_data):
    
    #Title scrapped using the article HTML tags
    scraped_title = page_data.head.title.string
    if scraped_title == None:
        scrapped_title = "None"
    return scraped_title

# FOR NFL NEWS---
def nfl_scrap_title(page_data):
    
    #Title scrapped using the article's HTML tree structure to find
    # the class within which the title is stored.
    nfl_scrapped_title = page_data.find('h1', class_ = 'nfl-c-article__title')
    if nfl_scrapped_title == None:
        nfl_scrapped_title = "None"
    title_text = str(nfl_scrapped_title.text).strip()
    return title_text


# Scraping the Author name

# For WASHINGTON POST---
def scrap_auth_name(page_data):
    
    #Author name scrapped using the article's HTML tree structure to find
    # the class within which the author name is stored for both sites.
    auth_name = page_data.find('a', attrs = {'class','gray-darkest b bb bc-gray bt-hover'})
    if auth_name == None:
        auth_name_string = "None"
        return auth_name_string
    return auth_name.text

# For NFL---
def nfl_scrap_auth_name(page_data):
    nfl_auth_name = page_data.find('a', class_ = 'nfl-o-cta--link nfl-o-author__name')
    if nfl_auth_name == None:
        author_name = "None"
    else:
        author_name = str(nfl_auth_name.text).strip()
    return author_name


# Scraping the date on which it last time updated

# For WASHINGTON POST---
def scrap_date(page_data):
    
    #Washington Post articles only list the time the article was posted.
    #Older articles are removed periodically from the site, so here the 
    #time of the article is scrapped.
    date = page_data.find('span', attrs={'data-qa','display-date'})
    if date == None:
        date_string = "None"
        return date_string
    
    return date.text

# For NFL
def nfl_scrap_date(page_data):
    
    #Date scrapped from NFL site
    nfl_date = page_data.find('div', class_ = 'nfl-c-article__dates')
    if nfl_date == None:
        nfl_date_string = "None"
        return nfl_date_string
    date_text = str(nfl_date.text).replace("Published:", " ").strip()
    return date_text

# Scraping the Article content

def scrap_cont(page_data):
    
    #Extract the data from the article using the 'p' tag
    content = page_data.find_all('p')
    
    body_content = []
    for para in content:
        cont_txt = para.text
        cont_txt.strip()
        body_content.append(cont_txt)
    content = ' '.join(map(str, body_content)) 
    return content

#NLTK downloads for the tokenizer, lemmatizer and stemmer

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
    
# Function to remove stop words

def remove_stop_words(article_content):
    
    stop_words = set(stopwords.words('english'))
    
    #Tokenize all words in the paragraph
    tokens = word_tokenize(article_content)
    
    filtered_paragraph = [word for word in tokens if not word.lower() in stop_words]
    
    filtered_paragraph = []
    
    #Filter the tokens for stop words and add the correct words
    for word_token in tokens:
        if word_token not in stop_words:
            filtered_paragraph.append(word_token)
    
    return filtered_paragraph

# The following functions are from the Medium Post:
# https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258
# function to convert nltk tag to wordnet tag

def nltk_to_wordnet_tags(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
def lemmatize_and_stem_content(filtered_content):
    
    lemmatizer = WordNetLemmatizer()
    
    stemmer = PorterStemmer()
    
    nltk_tagged = nltk.pos_tag(filtered_content)
    
    wordnet_tagged = map(lambda x:(x[0], nltk_to_wordnet_tags(x[1])), nltk_tagged)
    
    lemmatized_content = []
    
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_content.append(word)
        else:
            lemmatized_content.append(lemmatizer.lemmatize(word, tag))
    
    lemmatized_text = ' '.join(map(str, lemmatized_content))
    
    words = word_tokenize(lemmatized_text)
    
    final_content = []
    
    #Tokenize and stem words
    for w in words:
        final_content.append(stemmer.stem(w))
    
    final_text = ' '.join(map(str, final_content))
    
    #Strip text of punctuation and special characters
    exclusion_characters = string.punctuation
    
    table_ = str.maketrans('', '', exclusion_characters)
    
    final_text = final_text.translate(table_)
    
    encoded_final = final_text.encode("ascii", "ignore")
    
    final_content = encoded_final.decode()
    
    return final_content
    
    

if __name__=="__main__":
    
    #------ URL to Scrap -------
   
    url = "https://www.washingtonpost.com/"
    
    nfl_url = "https://www.nfl.com"
   
    # Header of Your Browser is required
   
    header = {
        'User-Agent':  "Chrome/91.0.4472.124"
        }
    
    #------ Dictionary to store data -------
    
    data = {}
    
    data['article'] = []

    # First Making request and getting the links
    
    page_links = []
    
    page_links = get_link(url, header)
    
    nfl_page_links = []
    
    nfl_page_links = get_link_nfl(nfl_url, header)
        
    #Loop through links and scrap the required data.
    
    for download_url in page_links:
        
        scrapped_data = get_data(download_url, header)
        
        article_title = scrap_title(scrapped_data)
        
        article_author = scrap_auth_name(scrapped_data)
        
        article_date = scrap_date(scrapped_data)
        
        article_content = scrap_cont(scrapped_data)
        
        article_filtered_content = remove_stop_words(article_content)
        
        article_lemmatized_content = lemmatize_and_stem_content(article_filtered_content)
        
        #Dump scrapped data into dictionary
        
        data['article'].append({
            'title': article_title,
            'author': article_author,
            'date': article_date,
            'body': article_content,
            'preprocessed': article_lemmatized_content})
    
    for nfl_download_url in nfl_page_links:
        
        final_url = nfl_url + nfl_download_url
        
        nfl_scrapped_data = get_data(final_url, header)
        
        nfl_article_title = nfl_scrap_title(nfl_scrapped_data)
        
        nfl_author_name = nfl_scrap_auth_name(nfl_scrapped_data)
    
        nfl_article_date = nfl_scrap_date(nfl_scrapped_data)
        
        nfl_article_content = scrap_cont(nfl_scrapped_data)
        
        nfl_filtered_content = remove_stop_words(nfl_article_content)
        
        nfl_lemmatized_content = lemmatize_and_stem_content(nfl_filtered_content)
        
        data['article'].append({
            'title': nfl_article_title,
            'author': nfl_author_name,
            'date': nfl_article_date,
            'body': nfl_article_content,
            'preprocessed': nfl_lemmatized_content})
    
    #Dump dictionary data into .json file
    
    with open('articles.json', 'w', encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False)
        
    
    