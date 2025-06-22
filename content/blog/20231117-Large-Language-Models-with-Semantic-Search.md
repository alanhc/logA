---
title: 20231117-Large Language Models with Semantic Search
date: 2023-11-17
tags:
  - llm
  - ai
---

![](https://i.imgur.com/06Un2NL.png)
## keyword-search

![](https://i.imgur.com/JujinkN.png)

![](https://i.imgur.com/iBetWva.png)
![](https://i.imgur.com/Buh9p6p.png)

![](https://i.imgur.com/D5JHXo4.png)

![](https://i.imgur.com/6hfnXi8.png)

### Setup
Load needed API keys and relevant Python libaries.
```python
# !pip install cohere
# !pip install weaviate-client
```

```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```
Let's start by imporing Weaviate to access the Wikipedia database.
```python
import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])
```

```python
client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)
```

```python
client.is_ready() 
```

    True
### Keyword Search

```python
def keyword_search(query,
                   results_lang='en',
                   properties = ["title","url","text"],
                   num_results=3):

    where_filter = {
    "path": ["lang"],
    "operator": "Equal",
    "valueString": results_lang
    }
    
    response = (
        client.query.get("Articles", properties)
        .with_bm25(
            query=query
        )
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
        )

    result = response['data']['Get']['Articles']
    return result
```


```python
query = "What is the most viewed televised event?"
keyword_search_results = keyword_search(query)
print(keyword_search_results)
```

### Try modifying the search options
- Other languages to try: `en, de, fr, es, it, ja, ar, zh, ko, hi`


```python
properties = ["text", "title", "url", 
             "views", "lang"]
```


```python
def print_result(result):
    """ Print results with colorful formatting """
    for i,item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()
```


```python
print_result(keyword_search_results)
```


```python
query = "What is the most viewed televised event?"
keyword_search_results = keyword_search(query, results_lang='de')
print_result(keyword_search_results)
```

### How to get your own API key

For this course, an API key is provided for you.  If you would like to develop projects with Cohere's API outside of this classroom, you can register for an API key [here](https://dashboard.cohere.ai/welcome/register?utm_source=partner&utm_medium=website&utm_campaign=DeeplearningAI).

## Lesson 2: Embeddings

### Setup
Load needed API keys and relevant Python libaries.


```python
# !pip install cohere umap-learn altair datasets
```


```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```


```python
import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])
```


```python
import pandas as pd
```

### Word Embeddings

Consider a very small dataset of three words.


```python
three_words = pd.DataFrame({'text':
  [
      'joy',
      'happiness',
      'potato'
  ]})

three_words
```



Let's create the embeddings for the three words:

```python
three_words_emb = co.embed(texts=list(three_words['text']),
                           model='embed-english-v2.0').embeddings
```


```python
word_1 = three_words_emb[0]
word_2 = three_words_emb[1]
word_3 = three_words_emb[2]
```


```python
word_1[:10]
```




    [2.3203125,
     -0.18334961,
     -0.578125,
     -0.7314453,
     -2.2050781,
     -2.59375,
     0.35205078,
     -1.6220703,
     0.27954102,
     0.3083496]
### Sentence Embeddings

Consider a very small dataset of three sentences.


```python
sentences = pd.DataFrame({'text':
  [
   'Where is the world cup?',
   'The world cup is in Qatar',
   'What color is the sky?',
   'The sky is blue',
   'Where does the bear live?',
   'The bear lives in the the woods',
   'What is an apple?',
   'An apple is a fruit',
  ]})

sentences
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Where is the world cup?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The world cup is in Qatar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What color is the sky?</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The sky is blue</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Where does the bear live?</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The bear lives in the the woods</td>
    </tr>
    <tr>
      <th>6</th>
      <td>What is an apple?</td>
    </tr>
    <tr>
      <th>7</th>
      <td>An apple is a fruit</td>
    </tr>
  </tbody>
</table>
</div>



Let's create the embeddings for the three sentences:


```python
emb = co.embed(texts=list(sentences['text']),
               model='embed-english-v2.0').embeddings

# Explore the 10 first entries of the embeddings of the 3 sentences:
for e in emb:
    print(e[:3])
```

    [0.27319336, -0.37768555, -1.0273438]
    [0.49804688, 1.2236328, 0.4074707]
    [-0.23571777, -0.9375, 0.9614258]
    [0.08300781, -0.32080078, 0.9272461]
    [0.49780273, -0.35058594, -1.6171875]
    [1.2294922, -1.3779297, -1.8378906]
    [0.15686035, -0.92041016, 1.5996094]
    [1.0761719, -0.7211914, 0.9296875]



```python
len(emb[0])
```




    4096




```python
#import umap
#import altair as alt
```


```python
from utils import umap_plot
```

    /usr/local/lib/python3.9/site-packages/umap/distances.py:1063: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
      @numba.jit()
    /usr/local/lib/python3.9/site-packages/umap/distances.py:1071: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
      @numba.jit()
    /usr/local/lib/python3.9/site-packages/umap/distances.py:1086: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
      @numba.jit()



```python
chart = umap_plot(sentences, emb)
```


```python
chart.interactive()
```

### Articles Embeddings
```python
import pandas as pd
wiki_articles = pd.read_pickle('wikipedia.pkl')
wiki_articles
```

```python
import numpy as np
from utils import umap_plot_big
```


```python
articles = wiki_articles[['title', 'text']]
embeds = np.array([d for d in wiki_articles['emb']])

chart = umap_plot_big(articles, embeds)
chart.interactive()
```

## dense-retrieval


![](https://i.imgur.com/t4T8EWa.png)

![](https://i.imgur.com/hQgsxUj.png)

![](https://i.imgur.com/ww2LTK7.png)

![](https://i.imgur.com/1RZeHzR.png)


![](https://i.imgur.com/4A6KPyV.png)

![](https://i.imgur.com/ymbfYrt.png)

### Dense Retrieval
### Setup

Load needed API keys and relevant Python libaries.


```python
# !pip install cohere 
# !pip install weaviate-client Annoy
```


```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```


```python
import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])
```


```python
import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])
```


```python
client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)
client.is_ready() #check if True
```

### Part 1: Vector Database for semantic Search


```python
def dense_retrieval(query, 
                    results_lang='en', 
                    properties = ["text", "title", "url", "views", "lang", "_additional {distance}"],
                    num_results=5):

    nearText = {"concepts": [query]}
    
    # To filter by language
    where_filter = {
    "path": ["lang"],
    "operator": "Equal",
    "valueString": results_lang
    }
    response = (
        client.query
        .get("Articles", properties)
        .with_near_text(nearText)
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
    )

    result = response['data']['Get']['Articles']

    return result
```


```python
from utils import print_result
```

#### Bacic Query


```python
query = "Who wrote Hamlet?"
dense_retrieval_results = dense_retrieval(query)
print_result(dense_retrieval_results)
```

#### Medium Query


```python
query = "What is the capital of Canada?"
dense_retrieval_results = dense_retrieval(query)
print_result(dense_retrieval_results)
```


```python
from utils import keyword_search

query = "What is the capital of Canada?"
keyword_search_results = keyword_search(query, client)
print_result(keyword_search_results)
```

#### Complicated Query


```python
from utils import keyword_search

query = "Tallest person in history?"
keyword_search_results = keyword_search(query, client)
print_result(keyword_search_results)
```


```python
query = "Tallest person in history"
dense_retrieval_results = dense_retrieval(query)
print_result(dense_retrieval_results)
```


```python
query = "أطول رجل في التاريخ"
dense_retrieval_results = dense_retrieval(query)
print_result(dense_retrieval_results)
```


```python
query = "film about a time travel paradox"
dense_retrieval_results = dense_retrieval(query)
print_result(dense_retrieval_results)
```

### Part 2: Building Semantic Search from Scratch

#### Get the text archive:


```python
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import re
```


```python
text = """
Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.

Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles.
Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.

Interstellar premiered on October 26, 2014, in Los Angeles.
In the United States, it was first released on film stock, expanding to venues using digital projectors.
The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014.
It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight.
It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""
```

#### Chunking: 


```python
# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts])
```


```python
texts
```


```python
# Split into a list of paragraphs
texts = text.split('\n\n')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts])
```


```python
texts
```


```python
# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts])
```


```python
title = 'Interstellar (film)'

texts = np.array([f"{title} {t}" for t in texts])
```


```python
texts
```

#### Get the embeddings:


```python
response = co.embed(
    texts=texts.tolist()
).embeddings
```


```python
embeds = np.array(response)
embeds.shape
```

#### Create the search index:


```python
search_index = AnnoyIndex(embeds.shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10) # 10 trees
search_index.save('test.ann')
```


```python
pd.set_option('display.max_colwidth', None)

def search(query):

  # Get the query's embedding
  query_embed = co.embed(texts=[query]).embeddings

  # Retrieve the nearest neighbors
  similar_item_ids = search_index.get_nns_by_vector(query_embed[0],
                                                    3,
                                                  include_distances=True)
  # Format the results
  results = pd.DataFrame(data={'texts': texts[similar_item_ids[0]],
                              'distance': similar_item_ids[1]})

  print(texts[similar_item_ids[0]])
    
  return results
```


```python
query = "How much did the film make?"
search(query)
```

## rerank

![](https://i.imgur.com/tc8rgLQ.png)



![](https://i.imgur.com/a9CDcDx.png)

![](https://i.imgur.com/Ilpdqy0.png)

![](https://i.imgur.com/fAaqkvN.png)
## ReRank

### Setup

Load needed API keys and relevant Python libaries.


```python
# !pip install cohere 
# !pip install weaviate-client
```


```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```


```python
import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])
```


```python
import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])
```


```python
client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)
```

### Dense Retrieval


```python
from utils import dense_retrieval
```


```python
query = "What is the capital of Canada?"
```


```python
dense_retrieval_results = dense_retrieval(query, client)
```


```python
from utils import print_result
```


```python
print_result(dense_retrieval_results)
```

    item 0
    _additional:{'distance': -150.8129}
    
    lang:en
    
    text:The governor general of the province had designated Kingston as the capital in 1841. However, the major population centres of Toronto and Montreal, as well as the former capital of Lower Canada, Quebec City, all had legislators dissatisfied with Kingston. Anglophone merchants in Quebec were the main group supportive of the Kingston arrangement. In 1842, a vote rejected Kingston as the capital, and study of potential candidates included the then-named Bytown, but that option proved less popular than Toronto or Montreal. In 1843, a report of the Executive Council recommended Montreal as the capital as a more fortifiable location and commercial centre, however, the Governor General refused to execute a move without a parliamentary vote. In 1844, the Queen's acceptance of a parliamentary vote moved the capital to Montreal.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 1
    _additional:{'distance': -150.29314}
    
    lang:en
    
    text:For brief periods, Toronto was twice the capital of the united Province of Canada: first from 1849 to 1852, following unrest in Montreal, and later 1856–1858. After this date, Quebec was designated as the capital until 1866 (one year before Canadian Confederation). Since then, the capital of Canada has remained Ottawa, Ontario.
    
    title:Toronto
    
    url:https://en.wikipedia.org/wiki?curid=64646
    
    views:3000
    
    
    item 2
    _additional:{'distance': -150.03601}
    
    lang:en
    
    text:Selection of Ottawa as the capital of Canada predates the Confederation of Canada. The selection was contentious and not straightforward, with the parliament of the United Province of Canada holding more than 200 votes over several decades to attempt to settle on a legislative solution to the location of the capital.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 3
    _additional:{'distance': -149.92947}
    
    lang:en
    
    text:Until the late 18th century Québec was the most populous city in present-day Canada. As of the census of 1790, Montreal surpassed it with 18,000 inhabitants, but Quebec (pop. 14,000) remained the administrative capital of New France. It was then made the capital of Lower Canada by the Constitutional Act of 1791. From 1841 to 1867, the capital of the Province of Canada rotated between Kingston, Montreal, Toronto, Ottawa and Quebec City (from 1852 to 1856 and from 1859 to 1866).
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 4
    _additional:{'distance': -149.7189}
    
    lang:en
    
    text:The Quebec Conference on Canadian Confederation was held in the city in 1864. In 1867, Queen Victoria chose Ottawa as the definite capital of the Dominion of Canada, while Quebec City was confirmed as the capital of the newly created province of Quebec.
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 5
    _additional:{'distance': -149.35098}
    
    lang:en
    
    text:Montreal was the capital of the Province of Canada from 1844 to 1849, but lost its status when a Tory mob burnt down the Parliament building to protest the passage of the Rebellion Losses Bill. Thereafter, the capital rotated between Quebec City and Toronto until in 1857, Queen Victoria herself established Ottawa as the capital due to strategic reasons. The reasons were twofold. First, because it was located more in the interior of the Province of Canada, it was less susceptible to attack from the United States. Second, and perhaps more importantly, because it lay on the border between French and English Canada, Ottawa was seen as a compromise between Montreal, Toronto, Kingston and Quebec City, which were all vying to become the young nation's official capital. Ottawa retained the status as capital of Canada when the Province of Canada joined with Nova Scotia and New Brunswick to form the Dominion of Canada in 1867.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 6
    _additional:{'distance': -148.69257}
    
    lang:en
    
    text:Ottawa was chosen as the capital for two primary reasons. First, Ottawa's isolated location, surrounded by dense forest far from the Canada–US border and situated on a cliff face, would make it more defensible from attack. Second, Ottawa was approximately midway between Toronto and Kingston (in Canada West) and Montreal and Quebec City (in Canada East) making the selection an important political compromise.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 7
    _additional:{'distance': -148.68246}
    
    lang:en
    
    text:A number of buildings across Canada are reserved by the Crown for the use of the monarch and his viceroys. Each is called "Government House", but may be customarily known by some specific name. The sovereign's and governor general's official residences are Rideau Hall in Ottawa and the Citadelle in Quebec City. Each of these royal seats holds pieces from the Crown Collection. Further, though neither was ever used for their intended purpose, Hatley Castle in British Columbia was purchased in 1940 by King George VI in Right of Canada to use as his home during the course of the Second World War and the Emergency Government Headquarters, built in 1959 at CFS Carp and decommissioned in 1994, included a residential apartment for the sovereign or governor general in the case of a nuclear attack on Ottawa.
    
    title:Monarchy of Canada
    
    url:https://en.wikipedia.org/wiki?curid=56504
    
    views:2000
    
    
    item 8
    _additional:{'distance': -148.63625}
    
    lang:en
    
    text:Ottawa (, ; ) is the capital city of Canada. It is located at the confluence of the Ottawa River and the Rideau River in the southern portion of the province of Ontario. Ottawa borders Gatineau, Quebec, and forms the core of the Ottawa–Gatineau census metropolitan area (CMA) and the National Capital Region (NCR). Ottawa had a city population of 1,017,449 and a metropolitan population of 1,488,307, making it the fourth-largest city and fourth-largest metropolitan area in Canada.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 9
    _additional:{'distance': -148.31482}
    
    lang:en
    
    text:Toronto, the capital of Ontario, is the centre of Canada's financial services and banking industry. Neighbouring cities are home to product distribution, IT centres, and manufacturing industries. Canada's Federal Government is the largest single employer in the National Capital Region, which centres on the border cities of Ontario's Ottawa and Quebec's Gatineau.
    
    title:Ontario
    
    url:https://en.wikipedia.org/wiki?curid=22218
    
    views:3000
    
    
    item 10
    _additional:{'distance': -147.95712}
    
    lang:en
    
    text:Ottawa is the political centre of Canada and headquarters to the federal government. The city houses numerous foreign embassies, key buildings, organizations, and institutions of Canada's government, including the Parliament of Canada, the Supreme Court, the residence of Canada's viceroy, and Office of the Prime Minister.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 11
    _additional:{'distance': -147.67686}
    
    lang:en
    
    text:Canada is a country in North America. Its ten provinces and three territories extend from the Atlantic Ocean to the Pacific Ocean and northward into the Arctic Ocean, covering over , making it the world's second-largest country by total area. Its southern and western border with the United States, stretching , is the world's longest binational land border. Canada's capital is Ottawa, and its three largest metropolitan areas are Toronto, Montreal, and Vancouver.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 12
    _additional:{'distance': -147.3634}
    
    lang:en
    
    text:In 1849, after violence in Montreal a series of votes was held, with Kingston and Bytown both again considered as capitals. However, the successful proposal was for two cities to share capital status, and the legislature to alternate sitting in each: Quebec City and Toronto, in a policy known as perambulation. Logistical difficulties made this an unpopular arrangement, and although an 1856 vote passed for the lower house of parliament to relocate permanently to Quebec City, the upper house refused to approve funding.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 13
    _additional:{'distance': -146.92775}
    
    lang:en
    
    text:Montreal has the second-largest economy of Canadian cities based on GDP and the largest in Quebec. In 2014, Metropolitan Montreal was responsible for of Quebec's GDP. The city is today an important centre of commerce, finance, industry, technology, culture, world affairs and is the headquarters of the Montreal Exchange. In recent decades, the city was widely seen as weaker than that of Toronto and other major Canadian cities, but it has recently experienced a revival.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 14
    _additional:{'distance': -146.91176}
    
    lang:en
    
    text:Quebec City ( or ; ), officially Québec (), is the capital city of the Canadian province of Quebec. As of July 2021, the city had a population of 549,459, and the metropolitan area had a population of 839,311. It is the eleventh-largest city and the seventh-largest metropolitan area in Canada. It is also the second-largest city in the province after Montreal. It has a humid continental climate with warm summers coupled with cold and snowy winters.
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 15
    _additional:{'distance': -146.8559}
    
    lang:en
    
    text:The prime minister of Canada () is the head of government of Canada. Under the Westminster system, the prime minister governs with the confidence of a majority the elected House of Commons; as such, the prime minister typically sits as a member of Parliament (MP) and leads the largest party or a coalition of parties. As first minister, the prime minister selects ministers to form the Cabinet, and serves as its chair. Constitutionally, the Crown exercises executive power on the advice of the Cabinet, which is collectively responsible to the House of Commons.
    
    title:Prime Minister of Canada
    
    url:https://en.wikipedia.org/wiki?curid=24135
    
    views:2000
    
    
    item 16
    _additional:{'distance': -146.75198}
    
    lang:en
    
    text:St. John's, the capital and largest city of Newfoundland and Labrador, is Canada's 22nd-largest census metropolitan area and it is home to about 40% of the province's population. St. John's is the seat of the House of Assembly of Newfoundland and Labrador as well as the jurisdiction's highest court, the Newfoundland and Labrador Court of Appeal.
    
    title:Newfoundland and Labrador
    
    url:https://en.wikipedia.org/wiki?curid=21980
    
    views:2000
    
    
    item 17
    _additional:{'distance': -146.7267}
    
    lang:en
    
    text:London was named for the British capital of London by John Graves Simcoe, who also named the local river the Thames, in 1793. Simcoe had intended London to be the capital of Upper Canada. Guy Carleton (Governor Dorchester) rejected that plan after the War of 1812, but accepted Simcoe's second choice, the present site of Toronto, to become the capital city of what would become the Province of Ontario, at Confederation, on 1 July 1867.
    
    title:London, Ontario
    
    url:https://en.wikipedia.org/wiki?curid=40353
    
    views:2000
    
    
    item 18
    _additional:{'distance': -146.70326}
    
    lang:en
    
    text:The funding impasse led to the ending of the legislature's role in determining the seat of government. The legislature requested the Queen make the determination of the seat of government. The Queen then acted on the advice of her governor general Edmund Head, who, after reviewing proposals from various cities, selected the recently renamed Ottawa. The Queen sent a letter to colonial authorities selecting Ottawa as the capital, effective December 31, 1857. George Brown, briefly a co-premier of the Province of Canada, attempted to reverse this decision, but was unsuccessful. The Queen's choice was ratified by the Parliament in 1859, with Quebec serving as interim capital from 1859 to 1865. The relocation process began in 1865, with the first session of Parliament held in the new buildings in 1866, and the buildings were generally well received by legislators.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 19
    _additional:{'distance': -146.62202}
    
    lang:en
    
    text:By federal law (Air Canada Public Participation Act), Air Canada has been obligated to keep its head office in Montreal. Its corporate headquarters is Air Canada Centre (French: "Centre Air Canada"), also known as La Rondelle ("The Puck" in French), a 7-storey building located on the grounds of Montréal–Trudeau International Airport in Saint-Laurent.
    
    title:Air Canada
    
    url:https://en.wikipedia.org/wiki?curid=145623
    
    views:2000
    
    
    item 20
    _additional:{'distance': -146.50558}
    
    lang:en
    
    text:Montreal ( ; officially Montréal, ) is the second-most populous city in Canada and most populous city in the Canadian province of Quebec. Founded in 1642 as "Ville-Marie", or "City of Mary", it is named after Mount Royal, the triple-peaked hill around which the early city of Ville-Marie is built. The city is centred on the Island of Montreal, which obtained its name from the same origin as the city, and a few much smaller peripheral islands, the largest of which is Île Bizard. The city is east of the national capital Ottawa, and southwest of the provincial capital, Quebec City.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 21
    _additional:{'distance': -146.47559}
    
    lang:en
    
    text:The administrative region in which it is situated is officially referred to as Capitale-Nationale, and the term "national capital" is used to refer to Quebec City itself at the provincial level.
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 22
    _additional:{'distance': -146.45479}
    
    lang:en
    
    text:Air Canada is the flag carrier and the largest airline of Canada by size and passengers carried. Air Canada maintains its headquarters in the borough of Saint-Laurent, Montreal, Quebec. The airline, founded in 1937, provides scheduled and charter air transport for passengers and cargo to 222 destinations worldwide. It is a founding member of the Star Alliance. Air Canada's major hubs are at Montréal–Trudeau International Airport (YUL), Toronto Pearson International Airport (YYZ), Calgary International Airport (YYC), and Vancouver International Airport (YVR). The airline's regional service is Air Canada Express.
    
    title:Air Canada
    
    url:https://en.wikipedia.org/wiki?curid=145623
    
    views:2000
    
    
    item 23
    _additional:{'distance': -146.34396}
    
    lang:en
    
    text:Across the Ottawa River, which forms the border between Ontario and Quebec, lies the city of Gatineau, itself the result of amalgamation of the former Quebec cities of Hull and Aylmer. Although formally and administratively separate cities in two separate provinces, Ottawa and Gatineau (along with a number of nearby municipalities) collectively constitute the National Capital Region, which is considered a single metropolitan area. One federal Crown corporation, the National Capital Commission, or NCC, has significant land holdings in both cities, including sites of historical and touristic importance. The NCC, through its responsibility for planning and development of these lands, has a key role in shaping the development of the city. Around the main urban area is an extensive greenbelt, administered by the NCC for conservation and leisure, and comprising mostly forest, farmland and marshland.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 24
    _additional:{'distance': -146.22684}
    
    lang:en
    
    text:As the national capital of Canada, tourism is an important part of Ottawa's economy, particularly after the 150th anniversary of Canada which was centred in Ottawa. The lead-up to the festivities saw much investment in civic infrastructure, upgrades to tourist infrastructure and increases in national cultural attractions. The National Capital Region annually attracts an estimated 22 million tourists, who spend about 2.2 billion dollars and support 30,600 jobs directly.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 25
    _additional:{'distance': -146.18494}
    
    lang:en
    
    text:In Canada, Calgary has the second-highest concentration of head offices in Canada (behind Toronto), the most head offices per capita, and the highest head office revenue per capita. Some large employers with Calgary head offices include Canada Safeway Limited, Westfair Foods Ltd., Suncor Energy, Agrium, Flint Energy Services Ltd., Shaw Communications, and Canadian Pacific Railway. CPR moved its head office from Montreal in 1996 and Imperial Oil moved from Toronto in 2005. Encana's new 58-floor corporate headquarters, the Bow, became the tallest building in Canada outside of Toronto. In 2001, the city became the corporate headquarters of the TSX Venture Exchange.
    
    title:Calgary
    
    url:https://en.wikipedia.org/wiki?curid=15895358
    
    views:2000
    
    
    item 26
    _additional:{'distance': -146.10544}
    
    lang:en
    
    text:The Crown is the pinnacle of the Canadian Forces, with the constitution placing the monarch in the position of commander-in-chief of the entire force, though the governor general carries out the duties attached to the position and also bears the title of "Commander-in-Chief in and over Canada". Further, included in Canada's constitution are the various treaties between the Crown and Canadian First Nations, Inuit, and Métis peoples, who view these documents as agreements directly and only between themselves and the reigning monarch, illustrating the relationship between sovereign and aboriginals.
    
    title:Monarchy of Canada
    
    url:https://en.wikipedia.org/wiki?curid=56504
    
    views:2000
    
    
    item 27
    _additional:{'distance': -146.09323}
    
    lang:en
    
    text:The city currently has one professional team, the baseball team Capitales de Québec, which plays in the Frontier League in downtown's Stade Canac. The team was established in 1999 and originally played in the Northern League. It has seven league titles, won in 2006, 2009, 2010, 2011, 2012, 2013 and 2017. A professional basketball team, the Quebec Kebs, played in National Basketball League of Canada in 2011 but folded before the 2012 season, and a semi-professional soccer team, the Dynamo de Québec, played in the Première ligue de soccer du Québec, until 2019.
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 28
    _additional:{'distance': -146.02489}
    
    lang:en
    
    text:Quebec ( ; ) is one of the thirteen provinces and territories of Canada. It is the largest province by area and the second-largest by population. Much of the population lives in urban areas along the St. Lawrence River, between the most populous city, Montreal, and the provincial capital, Quebec City. Quebec is the home of the Québécois nation. Located in Central Canada, the province shares land borders with Ontario to the west, Newfoundland and Labrador to the northeast, New Brunswick to the southeast, and a coastal border with Nunavut; in the south it borders Maine, New Hampshire, Vermont, and New York in the United States.
    
    title:Quebec
    
    url:https://en.wikipedia.org/wiki?curid=7954867
    
    views:3000
    
    
    item 29
    _additional:{'distance': -146.00629}
    
    lang:en
    
    text:The monarchy of Canada is Canada's form of government embodied by the Canadian sovereign and head of state. It is at the core of Canada's constitutional federal structure and Westminster-style parliamentary democracy. The monarchy is the foundation of the executive (King-in-Council), legislative (King-in-Parliament), and judicial (King-on-the-Bench) branches of both federal and provincial jurisdictions. The king of Canada since 8 September 2022 has been Charles III.
    
    title:Monarchy of Canada
    
    url:https://en.wikipedia.org/wiki?curid=56504
    
    views:2000
    
    
    item 30
    _additional:{'distance': -145.96207}
    
    lang:en
    
    text:Because the prime minister is in practice the most politically powerful member of the Canadian government, they are sometimes erroneously referred to as Canada's head of state, when, in fact, that role belongs to the Canadian monarch, represented by the governor general. The prime minister is, instead, the head of government and is responsible for advising the Crown on how to exercise much of the royal prerogative and its executive powers, which are governed by the constitution and its conventions. However, the function of the prime minister has evolved with increasing power. Today, per the doctrines of constitutional monarchy, the advice given by the prime minister is ordinarily binding, meaning the prime minister effectively carries out those duties ascribed to the sovereign or governor general, leaving the latter to act in predominantly ceremonial fashions. As such, the prime minister, supported by the Office of the Prime Minister (PMO), controls the appointments of many key figures in Canada's system of governance, including the governor general, the Cabinet, justices of the Supreme Court, senators, heads of Crown corporations, ambassadors and high commissioners, the provincial lieutenant governors, and approximately 3,100 other positions. Further, the prime minister plays a prominent role in the legislative process—with the majority of bills put before Parliament originating in the Cabinet—and the leadership of the Canadian Armed Forces.
    
    title:Prime Minister of Canada
    
    url:https://en.wikipedia.org/wiki?curid=24135
    
    views:2000
    
    
    item 31
    _additional:{'distance': -145.96161}
    
    lang:en
    
    text:Canada is a parliamentary democracy and a constitutional monarchy in the Westminster tradition. The country's head of government is the prime minister, who holds office by virtue of their ability to command the confidence of the elected House of Commons, and is appointed by the governor general, representing the monarch of Canada, the head of state. The country is a Commonwealth realm and is officially bilingual (English and French) at the federal level. It ranks among the highest in international measurements of government transparency, civil liberties, quality of life, economic freedom, education, gender equality and environmental sustainability. It is one of the world's most ethnically diverse and multicultural nations, the product of large-scale immigration. Canada's long and complex relationship with the United States has had a significant impact on its economy and culture.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 32
    _additional:{'distance': -145.95244}
    
    lang:en
    
    text:As access to new lands remained problematic because they were still monopolized by the Clique du Château, an exodus of Canadiens towards New England began and went on for the next one hundred years. This phenomenon is known as the Grande Hémorragie and greatly threatened the survival of the Canadien nation. The massive British immigration ordered from London that soon followed the failed rebellion compounded this problem. In order to combat this, the Church adopted the revenge of the cradle policy. In 1844, the capital of the Province of Canada was moved from Kingston to Montreal.
    
    title:Quebec
    
    url:https://en.wikipedia.org/wiki?curid=7954867
    
    views:3000
    
    
    item 33
    _additional:{'distance': -145.82161}
    
    lang:en
    
    text:Ottawa is known as the most educated city in Canada, with over half the population having graduated from college and/or university. Ottawa has the highest per capita concentration of engineers, scientists, and residents with PhDs in Canada. The city has two main public universities, and two main public colleges.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 34
    _additional:{'distance': -145.80045}
    
    lang:en
    
    text:Ottawa's primary employers are the Public Service of Canada and the high-tech industry, although tourism and healthcare also represent increasingly sizeable economic activities. The federal government is the city's largest employer, employing over 116,000 individuals from the National Capital Region. The national headquarters for many federal departments are in Ottawa, particularly throughout Centretown and in the Terrasses de la Chaudière and Place du Portage complexes in Hull. The National Defence Headquarters in Ottawa is the main command centre for the Canadian Armed Forces and hosts the Department of National Defence. During the summer, the city hosts the Ceremonial Guard, which performs functions such as the Changing the Guard.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 35
    _additional:{'distance': -145.77827}
    
    lang:en
    
    text:Historically the commercial capital of Canada, Montreal was surpassed in population and in economic strength by Toronto in the 1970s. It remains an important centre of commerce, aerospace, transport, finance, pharmaceuticals, technology, design, education, art, culture, tourism, food, fashion, video game development, film, and world affairs. Montreal is the location of the headquarters of the International Civil Aviation Organization, and was named a UNESCO City of Design in 2006. In 2017, Montreal was ranked the 12th-most liveable city in the world by the Economist Intelligence Unit in its annual Global Liveability Ranking, although it slipped to rank 40 in the 2021 index, primarily due to stress on the healthcare system from the COVID-19 pandemic. It is regularly ranked as a top ten city in the world to be a university student in the QS World University Rankings.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 36
    _additional:{'distance': -145.69058}
    
    lang:en
    
    text:Alberta's capital city, Edmonton, is located at about the geographic centre of the province. It is the most northerly major city in Canada and serves as a gateway and hub for resource development in northern Canada. With its proximity to Canada's largest oil fields, the region has most of western Canada's oil refinery capacity. Calgary is about south of Edmonton and north of Montana, surrounded by extensive ranching country. Almost 75% of the province's population lives in the Calgary–Edmonton Corridor. The land grant policy to the railways served as a means to populate the province in its early years.
    
    title:Alberta
    
    url:https://en.wikipedia.org/wiki?curid=717
    
    views:2000
    
    
    item 37
    _additional:{'distance': -145.42075}
    
    lang:en
    
    text:Most jobs in Quebec City are concentrated in public administration, defence, services, commerce, transport and tourism. As the provincial capital, the city benefits from being a regional administrative and services centre: apropos, the provincial government is the largest employer in the city, employing 27,900 people as of 2007. CHUQ (the local hospital network) is the city's largest institutional employer, with more than 10,000 employees in 2007. The unemployment rate in June 2018 was 3.8%, below the national average (6.0%) and the second-lowest of Canada's 34 largest cities, behind Peterborough (2.7%).
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 38
    _additional:{'distance': -145.40527}
    
    lang:en
    
    text:Toronto is an international centre for business and finance. Generally considered the financial and industrial capital of Canada, Toronto has a high concentration of banks and brokerage firms on Bay Street in the Financial District. The Toronto Stock Exchange is the world's seventh-largest stock exchange by market capitalization. The five largest financial institutions of Canada, collectively known as the Big Five, have national offices in Toronto.
    
    title:Toronto
    
    url:https://en.wikipedia.org/wiki?curid=64646
    
    views:3000
    
    
    item 39
    _additional:{'distance': -145.37503}
    
    lang:en
    
    text:Along with concrete high-rises such as Édifice Marie-Guyart and Le Concorde on parliament hill (see List of tallest buildings in Quebec City), the city's skyline is dominated by the massive Château Frontenac hotel, perched on top of Cap-Diamant. It was designed by architect Bruce Price, as one of a series of "château" style hotels built for the Canadian Pacific Railway company. The railway company sought to encourage luxury tourism and bring wealthy travellers to its trains.
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 40
    _additional:{'distance': -145.36432}
    
    lang:en
    
    text:The name "Canada" refers to this settlement. Although the Acadian settlement at Port-Royal was established three years earlier, Quebec came to be known as the cradle of North America's Francophone population. The place seemed favourable to the establishment of a permanent colony.
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 41
    _additional:{'distance': -145.33707}
    
    lang:en
    
    text:British Columbia's capital is Victoria, located at the southeastern tip of Vancouver Island. Only a narrow strip of Vancouver Island, from Campbell River to Victoria, is significantly populated. Much of the western part of Vancouver Island and the rest of the coast is covered by temperate rainforest.
    
    title:British Columbia
    
    url:https://en.wikipedia.org/wiki?curid=3392
    
    views:3000
    
    
    item 42
    _additional:{'distance': -145.33646}
    
    lang:en
    
    text:The King of Canada has delegated his prerogative to grant armorial bearings to the Governor General of Canada. Canada has its own Chief Herald and Herald Chancellor. The Canadian Heraldic Authority, the governmental agency which is responsible for creating arms and promoting Canadian heraldry, is situated at Rideau Hall.
    
    title:Coat of arms
    
    url:https://en.wikipedia.org/wiki?curid=55284
    
    views:2000
    
    
    item 43
    _additional:{'distance': -145.27737}
    
    lang:en
    
    text:Generally, Canadian provinces have steadily grown in population along with Canada. However, some provinces such as Saskatchewan, Prince Edward Island and Newfoundland and Labrador have experienced long periods of stagnation or population decline. Ontario and Quebec have always been the two biggest provinces in Canada, with together over 60% of the population at any given time. The population of the West relative to Canada as a whole has steadily grown over time, while that of Atlantic Canada has declined.
    
    title:Provinces and territories of Canada
    
    url:https://en.wikipedia.org/wiki?curid=75763
    
    views:3000
    
    
    item 44
    _additional:{'distance': -145.24377}
    
    lang:en
    
    text:As Canada's capital, Ottawa has played host to a number of significant cultural events in Canadian history, including the first visit of the reigning Canadian sovereign—King George VI, with his consort, Queen Elizabeth—to his parliament, on 19 May 1939. VE Day was marked with a large celebration on 8 May 1945, the first raising of the country's new national flag took place on 15 February 1965, and the centennial of Confederation was celebrated on 1 July 1967. Queen Elizabeth II was in Ottawa on 17 April 1982, to issue a royal proclamation of the enactment of the Constitution Act. In 1983, Prince Charles and Diana Princess of Wales came to Ottawa for a state dinner hosted by then Prime Minister Pierre Trudeau. In 2011, Ottawa was selected as the first city to receive Prince William, Duke of Cambridge, and Catherine, Duchess of Cambridge during their tour of Canada.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 45
    _additional:{'distance': -145.18558}
    
    lang:en
    
    text:By 1951, Montreal's population had surpassed one million. However, Toronto's growth had begun challenging Montreal's status as the economic capital of Canada. Indeed, the volume of stocks traded at the Toronto Stock Exchange had already surpassed that traded at the Montreal Stock Exchange in the 1940s. The Saint Lawrence Seaway opened in 1959, allowing vessels to bypass Montreal. In time, this development led to the end of the city's economic dominance as businesses moved to other areas. During the 1960s, there was continued growth as Canada's tallest skyscrapers, new expressways and the subway system known as the Montreal Metro were finished during this time. Montreal also held the World's Fair of 1967, better known as Expo67.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 46
    _additional:{'distance': -145.11502}
    
    lang:en
    
    text:Ontario ( ; ) is one of the thirteen provinces and territories of Canada. Located in Central Canada, it is Canada's most populous province, with 38.3 percent of the country's population, and is the second-largest province by total area (after Quebec). Ontario is Canada's fourth-largest jurisdiction in total area when the territories of the Northwest Territories and Nunavut are included. It is home to the nation's capital city, Ottawa, and the nation's most populous city, Toronto, which is Ontario's provincial capital.
    
    title:Ontario
    
    url:https://en.wikipedia.org/wiki?curid=22218
    
    views:3000
    
    
    item 47
    _additional:{'distance': -145.09877}
    
    lang:en
    
    text:In the heart of downtown are the British Columbia Parliament Buildings, The Empress Hotel, Victoria Police Department Station Museum, the gothic Christ Church Cathedral, and the Royal British Columbia Museum/IMAX National Geographic Theatre, with large exhibits on local Aboriginal peoples, natural history, and modern history, along with travelling international exhibits. In addition, the heart of downtown also has the Maritime Museum of British Columbia, Emily Carr House, Victoria Bug Zoo, and Market Square. The oldest (and most intact) Chinatown in Canada is within downtown. The Art Gallery of Greater Victoria is close to downtown in the Rockland neighbourhood several city blocks from Craigdarroch Castle built by industrialist Robert Dunsmuir and Government House, the official residence of the Lieutenant-Governor of British Columbia.
    
    title:Victoria, British Columbia
    
    url:https://en.wikipedia.org/wiki?curid=32388
    
    views:2000
    
    
    item 48
    _additional:{'distance': -145.04091}
    
    lang:en
    
    text:Toronto ( ; or ) is the capital city of the Canadian province of Ontario. With a recorded population of 2,794,356 in 2021, it is the most populous city in Canada and the fourth most populous city in North America. The city is the anchor of the Golden Horseshoe, an urban agglomeration of 9,765,188 people (as of 2021) surrounding the western end of Lake Ontario, while the Greater Toronto Area proper had a 2021 population of 6,712,341. Toronto is an international centre of business, finance, arts, sports and culture, and is recognized as one of the most multicultural and cosmopolitan cities in the world.
    
    title:Toronto
    
    url:https://en.wikipedia.org/wiki?curid=64646
    
    views:3000
    
    
    item 49
    _additional:{'distance': -145.01453}
    
    lang:en
    
    text:The prime minister is supported by the Prime Minister's Office and heads the Privy Council Office. The prime minister also effectively appoints individuals to the Senate of Canada and to the Supreme Court of Canada and other federal courts, along with choosing the leaders and boards, as required under law, of various Crown corporations. Under the "Constitution Act, 1867", government power is vested in the monarch (who is the head of state), but in practice the role of the monarch—or their representative, the governor general (or the administrator)—is largely ceremonial and only exercised on the advice of a Cabinet minister. The prime minister also provides advice to the monarch of Canada for the selection of the governor general.
    
    title:Prime Minister of Canada
    
    url:https://en.wikipedia.org/wiki?curid=24135
    
    views:2000
    
    
    item 50
    _additional:{'distance': -144.99521}
    
    lang:en
    
    text:The national flag of Canada (), often simply referred to as the Canadian flag or, unofficially, as the Maple Leaf or "" (; ), consists of a red field with a white square at its centre in the ratio of , in which is featured a stylized, red, 11-pointed maple leaf charged in the centre. It is the first flag to have been adopted by both houses of Parliament and officially proclaimed by the Canadian monarch as the country's official national flag. The flag has become the predominant and most recognizable national symbol of Canada.
    
    title:Flag of Canada
    
    url:https://en.wikipedia.org/wiki?curid=97066
    
    views:2000
    
    
    item 51
    _additional:{'distance': -144.94948}
    
    lang:en
    
    text:Since the days of King Louis XIV, the monarch is the fount of all honours in Canada and the orders, decorations, and medals form "an integral element of the Crown." Hence, the insignia and medallions for these awards bear a crown, cypher, and/or portrait of the monarch. Similarly, the country's heraldic authority was created by the Queen and, operating under the authority of the governor general, grants new coats of arms, flags, and badges in Canada. Use of the royal crown in such symbols is a gift from the monarch showing royal support and/or association, and requires her approval before being added.
    
    title:Monarchy of Canada
    
    url:https://en.wikipedia.org/wiki?curid=56504
    
    views:2000
    
    
    item 52
    _additional:{'distance': -144.89908}
    
    lang:en
    
    text:The Canadiens have developed strong rivalries with two fellow Original Six franchises, with whom they frequently shared divisions and competed in post-season play. The oldest is with the Toronto Maple Leafs, who first faced the Canadiens as the Toronto Arenas in 1917. The teams met 16 times in the playoffs, including five Stanley Cup Finals. Featuring the two largest cities in Canada and two of the largest fanbases in the league, the rivalry is sometimes dramatized as being emblematic of Canada's English and French linguistic divide. From 1938 to 1970, they were the only two Canadian teams in the league.
    
    title:Montreal Canadiens
    
    url:https://en.wikipedia.org/wiki?curid=42966
    
    views:2000
    
    
    item 53
    _additional:{'distance': -144.89122}
    
    lang:en
    
    text:Canadian Pacific Railway (CPR), headquartered in Calgary, Alberta, was founded here in 1881. Its corporate headquarters occupied Windsor Station at 910 Peel Street until 1995. With the Port of Montreal kept open year-round by icebreakers, lines to Eastern Canada became surplus, and now Montreal is the railway's eastern and intermodal freight terminus. CPR connects at Montreal with the Port of Montreal, the Delaware and Hudson Railway to New York, the Quebec Gatineau Railway to Quebec City and Buckingham, the Central Maine and Quebec Railway to Halifax, and Canadian National Railway (CN). The CPR's flagship train, "The Canadian", ran daily from Windsor Station to Vancouver, but in 1978 all passenger services were transferred to Via. Since 1990, "The Canadian" has terminated in Toronto instead of in Montreal.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 54
    _additional:{'distance': -144.88953}
    
    lang:en
    
    text:Calgary was designated as one of the cultural capitals of Canada in 2012. While many Calgarians continue to live in the city's suburbs, more central neighbourhoods such as Kensington, Inglewood, Forest Lawn, Bridgeland, Marda Loop, the Mission District, and especially the Beltline, have become more popular and density in those areas has increased.
    
    title:Calgary
    
    url:https://en.wikipedia.org/wiki?curid=15895358
    
    views:2000
    
    
    item 55
    _additional:{'distance': -144.88878}
    
    lang:en
    
    text:Nova Scotia's capital and largest municipality is Halifax, which is home to over 45% of the province's population as of the 2021 census. Halifax is the thirteenth-largest census metropolitan area in Canada, the largest municipality in Atlantic Canada, and Canada's second-largest coastal municipality after Vancouver.
    
    title:Nova Scotia
    
    url:https://en.wikipedia.org/wiki?curid=21184
    
    views:2000
    
    
    item 56
    _additional:{'distance': -144.88113}
    
    lang:en
    
    text:6.313 million Toronto ; 4.277 million Montreal ; 2.632 million Vancouver ; 1.611 million Calgary ; 1.519 million Edmonton ; 1.423 million OTTAWA (capital) (2022)
    
    title:Urban area
    
    url:https://en.wikipedia.org/wiki?curid=764593
    
    views:3000
    
    
    item 57
    _additional:{'distance': -144.85863}
    
    lang:en
    
    text:Centretown is next to downtown, which includes a substantial economic and architectural government presence across multiple branches of government. The legislature's work takes place in the parliamentary precinct, which includes buildings on Parliament Hill and others downtown, such as the Senate of Canada Building. Important buildings in the executive branch include the Office of the Prime Minister and Privy Council as well as many civil service buildings. The Supreme Court of Canada building can also be found in this area.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 58
    _additional:{'distance': -144.79507}
    
    lang:en
    
    text:With an area of , Newfoundland is the world's 16th-largest island, Canada's fourth-largest island, and the largest Canadian island outside the North. The provincial capital, St. John's, is located on the southeastern coast of the island; Cape Spear, just south of the capital, is the easternmost point of North America, excluding Greenland. It is common to consider all directly neighbouring islands such as New World, Twillingate, Fogo and Bell Island to be 'part of Newfoundland' (i.e., distinct from Labrador). By that classification, Newfoundland and its associated small islands have a total area of .
    
    title:Newfoundland (island)
    
    url:https://en.wikipedia.org/wiki?curid=26304966
    
    views:2000
    
    
    item 59
    _additional:{'distance': -144.78793}
    
    lang:en
    
    text:Canada is one of the oldest continuing monarchies in the world. Initially established in the 16th century, monarchy in Canada has evolved through a continuous succession of initially French and later British sovereigns into the independent Canadian sovereigns of today. The institution that is Canada's system of constitutional monarchy is sometimes colloquially referred to as the "Maple Crown".
    
    title:Monarchy of Canada
    
    url:https://en.wikipedia.org/wiki?curid=56504
    
    views:2000
    
    
    item 60
    _additional:{'distance': -144.77293}
    
    lang:en
    
    text:Edmonton ( ) is the capital city of the Canadian province of Alberta. Edmonton is situated on the North Saskatchewan River and is the centre of the Edmonton Metropolitan Region, which is surrounded by Alberta's central region. The city anchors the north end of what Statistics Canada defines as the "Calgary–Edmonton Corridor".
    
    title:Edmonton
    
    url:https://en.wikipedia.org/wiki?curid=95405
    
    views:2000
    
    
    item 61
    _additional:{'distance': -144.767}
    
    lang:en
    
    text:Montreal-based CN was formed in 1919 by the Canadian government following a series of country-wide rail bankruptcies. It was formed from the Grand Trunk, Midland and Canadian Northern Railways, and has risen to become CPR's chief rival in freight carriage in Canada. Like the CPR, CN divested itself of passenger services in favour of Via. CN's flagship train, the "Super Continental", ran daily from Central Station to Vancouver and subsequently became a Via train in 1978. It was eliminated in 1990 in favour of rerouting "The Canadian".
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 62
    _additional:{'distance': -144.69998}
    
    lang:en
    
    text:In Canada, these head of state powers belong to the monarch as part of the royal prerogative, but the Governor General has been permitted to exercise them since 1947 and has done so since the 1970s.
    
    title:Head of state
    
    url:https://en.wikipedia.org/wiki?curid=13456
    
    views:2000
    
    
    item 63
    _additional:{'distance': -144.66977}
    
    lang:en
    
    text:Nunavut comprises a major portion of Northern Canada and most of the Arctic Archipelago. Its vast territory makes it the fifth-largest country subdivision in the world, as well as North America's second-largest (after Greenland). The capital Iqaluit (formerly Frobisher Bay), on Baffin Island in the east, was chosen by a capital plebiscite in 1995. Other major communities include the regional centres of Rankin Inlet and Cambridge Bay.
    
    title:Nunavut
    
    url:https://en.wikipedia.org/wiki?curid=7129693
    
    views:2000
    
    
    item 64
    _additional:{'distance': -144.66476}
    
    lang:en
    
    text:The vast majority of Canada's population is concentrated in areas close to the Canada–US border. Its four largest provinces by area (Quebec, Ontario, British Columbia and Alberta) are also (with Quebec and Ontario switched in order) its most populous; together they account for 86% of the country's population. The territories (the Northwest Territories, Nunavut and Yukon) account for over a third of Canada's area but are only home to 0.3% of its population, which skews the national population density value.
    
    title:Provinces and territories of Canada
    
    url:https://en.wikipedia.org/wiki?curid=75763
    
    views:3000
    
    
    item 65
    _additional:{'distance': -144.66019}
    
    lang:en
    
    text:As King of Canada, Charles III is the head of state for the Government of Alberta. His duties in Alberta are carried out by Lieutenant Governor Salma Lakhani. The King and lieutenant governor are figureheads whose actions are highly restricted by custom and constitutional convention. The lieutenant governor handles numerous honorific duties in the name of the King. The government is headed by the premier. The premier is normally a member of the Legislative Assembly, and draws all the members of the Cabinet from among the members of the Legislative Assembly. The City of Edmonton is the seat of the provincial government—the capital of Alberta. The current premier is Danielle Smith, who was sworn in on October 11th, 2022.
    
    title:Alberta
    
    url:https://en.wikipedia.org/wiki?curid=717
    
    views:2000
    
    
    item 66
    _additional:{'distance': -144.62424}
    
    lang:en
    
    text:Toronto is the capital of Ontario with the Ontario Legislative Building, often metonymically known as Queen's Park after the street and park surrounding it, being located in downtown Toronto. Most of the provincial government offices are also located in downtown Toronto.
    
    title:Greater Toronto Area
    
    url:https://en.wikipedia.org/wiki?curid=266720
    
    views:2000
    
    
    item 67
    _additional:{'distance': -144.60934}
    
    lang:en
    
    text:Montreal was referred to as "Canada's Cultural Capital" by "Monocle" magazine. The city is Canada's centre for French-language television productions, radio, theatre, film, multimedia, and print publishing. Montreal's many cultural communities have given it a distinct local culture. Montreal was designated as the World Book Capital for the year 2005 by UNESCO.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 68
    _additional:{'distance': -144.6033}
    
    lang:en
    
    text:From 2007 to 2011, Winnipeg was the "murder capital" of Canada, with the highest per-capita rate of homicides; as of 2019 it is in second place, behind Thunder Bay. Winnipeg had the 13-highest violent crime index in Canada, and the highest robbery rate. Winnipeg was the "violent crime capital" of Canada in 2020 according to the Statistics Canada police-reported violent crime severity index. Despite high overall violent crime rates, crime in Winnipeg is mostly concentrated in the inner city, which makes up only 19% of the population but was the site of 86.4% of the city's shootings, 66.5% of the robberies, 63.3% of the homicides and 59.5% of the sexual assaults in 2012.
    
    title:Winnipeg
    
    url:https://en.wikipedia.org/wiki?curid=100730
    
    views:2000
    
    
    item 69
    _additional:{'distance': -144.5503}
    
    lang:en
    
    text:The second largest concentration of British Columbia population is at the southern tip of Vancouver Island, which is made up of the 13 municipalities of Greater Victoria, Victoria, Saanich, Esquimalt, Oak Bay, View Royal, Highlands, Colwood, Langford, Central Saanich/Saanichton, North Saanich, Sidney, Metchosin, Sooke, which are part of the Capital Regional District. The metropolitan area also includes several Indian reserves (the governments of which are not part of the regional district). Almost half of the Vancouver Island population is in Greater Victoria.
    
    title:British Columbia
    
    url:https://en.wikipedia.org/wiki?curid=3392
    
    views:3000
    
    
    item 70
    _additional:{'distance': -144.51027}
    
    lang:en
    
    text:Although it has been argued that the term "head of state" is a republican one inapplicable in a constitutional monarchy such as Canada, where the monarch is the embodiment of the state and thus cannot be head of it, the sovereign is regarded by official government sources, judges, constitutional scholars, and pollsters as the head of state, while the governor general and lieutenant governors are all only representatives of, and thus equally subordinate to, that figure. Some governors general, their staff, government publications, and constitutional scholars like Ted McWhinney and C. E. S. Franks have, however, referred to the position of governor general as that of Canada's head of state, though sometimes qualifying the assertion with "de facto" or "effective"; Franks has hence recommended that the governor general be named officially as the head of state. Still others view the role of head of state as being shared by both the sovereign and her viceroys. Since 1927, governors general have been received on state visits abroad as though they were heads of state.
    
    title:Monarchy of Canada
    
    url:https://en.wikipedia.org/wiki?curid=56504
    
    views:2000
    
    
    item 71
    _additional:{'distance': -144.49982}
    
    lang:en
    
    text:The city is home to the Toronto Stock Exchange, the headquarters of Canada's five largest banks, and the headquarters of many large Canadian and multinational corporations. Its economy is highly diversified with strengths in technology, design, financial services, life sciences, education, arts, fashion, aerospace, environmental innovation, food services, and tourism. Toronto is the third-largest tech hub in North America after Silicon Valley and New York City, and the fastest growing.
    
    title:Toronto
    
    url:https://en.wikipedia.org/wiki?curid=64646
    
    views:3000
    
    
    item 72
    _additional:{'distance': -144.48755}
    
    lang:en
    
    text:The Port of Montreal is one of the largest inland ports in the world handling 26 million tonnes of cargo annually. As one of the most important ports in Canada, it remains a transshipment point for grain, sugar, petroleum products, machinery, and consumer goods. For this reason, Montreal is the railway hub of Canada and has always been an extremely important rail city; it is home to the headquarters of the Canadian National Railway, and was home to the headquarters of the Canadian Pacific Railway until 1995.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 73
    _additional:{'distance': -144.47365}
    
    lang:en
    
    text:Potential CFL expansion markets are the Maritimes, Quebec City, Saskatoon, London, and Windsor, all of which have been lobbying for Canadian Football League franchises in recent years. During the 1970s and 1980s, Harold Ballard attempted multiple times to secure a second CFL team for Toronto (either by way of expansion or by relocating the Hamilton Tiger-Cats), under the premise that Canada's largest city could support two teams.
    
    title:Canadian Football League
    
    url:https://en.wikipedia.org/wiki?curid=56802
    
    views:2000
    
    
    item 74
    _additional:{'distance': -144.4472}
    
    lang:en
    
    text:The city's landmarks include the Château Frontenac hotel that dominates the skyline and the Citadelle of Quebec, an intact fortress that forms the centrepiece of the ramparts surrounding the old city and includes a secondary royal residence. The National Assembly of Quebec (provincial legislature), the Musée national des beaux-arts du Québec ("National Museum of Fine Arts of Quebec"), and the Musée de la civilisation ("Museum of Civilization") are found within or near Vieux-Québec.
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 75
    _additional:{'distance': -144.42685}
    
    lang:en
    
    text:The city itself was not attacked during the War of 1812, when the United States again attempted to annex Canadian lands. Amid fears of another American attack on Quebec City, construction of the Citadelle of Quebec began in 1820. The Americans did not attack Canada after the War of 1812, but the Citadelle continued to house a large British garrison until 1871. It is still in use by the military and is also a tourist attraction.
    
    title:Quebec City
    
    url:https://en.wikipedia.org/wiki?curid=100727
    
    views:2000
    
    
    item 76
    _additional:{'distance': -144.36205}
    
    lang:en
    
    text:Ottawa has the most educated population among Canadian cities and is home to a number of colleges and universities, research and cultural institutions, including the University of Ottawa, Carleton University, Algonquin College, the National Arts Centre, the National Gallery of Canada; and numerous national museums, monuments, and historic sites. It is one of the most visited cities in Canada, with over 11 million visitors in 2018.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 77
    _additional:{'distance': -144.33684}
    
    lang:en
    
    text:Montreal was incorporated as a city in 1832. The opening of the Lachine Canal permitted ships to bypass the unnavigable Lachine Rapids, while the construction of the Victoria Bridge established Montreal as a major railway hub. The leaders of Montreal's business community had started to build their homes in the Golden Square Mile from about 1850. By 1860, it was the largest municipality in British North America and the undisputed economic and cultural centre of Canada.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 78
    _additional:{'distance': -144.23914}
    
    lang:en
    
    text:For over a century and a half, Montreal was the industrial and financial centre of Canada. This legacy has left a variety of buildings including factories, elevators, warehouses, mills, and refineries, that today provide an invaluable insight into the city's history, especially in the downtown area and the Old Port area. There are 50 National Historic Sites of Canada, more than any other city.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    
    item 79
    _additional:{'distance': -144.23871}
    
    lang:en
    
    text:Incorporated as a town in 1892 with a population of 700 and then as a city in 1904 with a population of 8,350, Edmonton became the capital of Alberta when the province was formed a year later, on September 1, 1905. In November 1905, the Canadian Northern Railway (CNR) arrived in Edmonton, accelerating growth.
    
    title:Edmonton
    
    url:https://en.wikipedia.org/wiki?curid=95405
    
    views:2000
    
    
    item 80
    _additional:{'distance': -144.18848}
    
    lang:en
    
    text:The nationalist movement in Quebec, particularly after the election of the "Parti Québécois" in 1976, contributed to driving many businesses and English-speaking people out of Quebec to Ontario, and as a result, Toronto surpassed Montreal as the largest city and economic centre of Canada. Depressed economic conditions in the Maritime Provinces have also resulted in de-population of those provinces in the 20th century, with heavy migration into Ontario.
    
    title:Ontario
    
    url:https://en.wikipedia.org/wiki?curid=22218
    
    views:3000
    
    
    item 81
    _additional:{'distance': -144.17209}
    
    lang:en
    
    text:Shortly after Canadian Confederation in 1867, the need for distinctive Canadian flags emerged. The first Canadian flag was that then used as the flag of the Governor General of Canada, a Union Flag with a shield in the centre bearing the quartered arms of Ontario, Quebec, Nova Scotia and New Brunswick, surrounded by a wreath of maple leaves. In 1870, the Red Ensign, with the addition of the Canadian composite shield in the fly, began to be used unofficially on land and sea and was known as the "Canadian Red Ensign". As new provinces joined the Confederation, their arms were added to the shield. In 1892, the British admiralty approved the use of the Red Ensign for Canadian use at sea.
    
    title:Flag of Canada
    
    url:https://en.wikipedia.org/wiki?curid=97066
    
    views:2000
    
    
    item 82
    _additional:{'distance': -144.15552}
    
    lang:en
    
    text:Established in 1975, the Great Canadian Theatre Company specializes in the production of Canadian plays at a local level. The cities museum landscape is notable for containing six of Canada's nine national museums, the Canada Agriculture and Food Museum, the Canada Aviation and Space Museum, the Canada Science and Technology Museum, Canadian Museum of Nature, Canadian War Museum and National Gallery of Canada. The National Gallery of Canada; designed by famous architect Moshe Safdie, it is a permanent home to the "Maman" sculpture. The Canadian War Museum houses over 3.75 million artifacts and was moved to an expanded facility in 2005. The Canadian Museum of Nature was built in 1905, and underwent a major renovation between 2004 and 2010, leading to a centrepiece Blue Whale skeleton, and the creation of a monthly nightclub experience, "Nature Nocturne".
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 83
    _additional:{'distance': -144.15211}
    
    lang:en
    
    text:It is also notable that three cities (Montreal, Toronto, and Vancouver) are from Canada and three other cities (Beijing, Hong Kong, and Shanghai) are from the People's Republic of China. No other countries are represented by more than one city.
    
    title:Monopoly (game)
    
    url:https://en.wikipedia.org/wiki?curid=19692
    
    views:2000
    
    
    item 84
    _additional:{'distance': -144.14737}
    
    lang:en
    
    text:Themes of nature, pioneers, trappers, and traders played an important part in the early development of Canadian symbolism. Modern symbols emphasize the country's geography, cold climate, lifestyles and the Canadianization of traditional European and Indigenous symbols. The use of the maple leaf as a Canadian symbol dates to the early 18th century. The maple leaf is depicted on Canada's current and previous flags, and on the Arms of Canada. Canada's official tartan, known as the "maple leaf tartan", has four colours that reflect the colours of the maple leaf as it changes through the seasons—green in the spring, gold in the early autumn, red at the first frost, and brown after falling. The Arms of Canada are closely modelled after the royal coat of arms of the United Kingdom with French and distinctive Canadian elements replacing or added to those derived from the British version.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 85
    _additional:{'distance': -144.126}
    
    lang:en
    
    text:The 2021 Canadian census enumerated a total population of 36,991,981, an increase of around 5.2 percent over the 2016 figure. The main drivers of population growth are immigration and, to a lesser extent, natural growth. Canada has one of the highest per-capita immigration rates in the world, driven mainly by economic policy and also family reunification. A record number of 405,000 immigrants were admitted to Canada in 2021. New immigrants settle mostly in major urban areas in the country, such as Toronto, Montreal and Vancouver. Canada also accepts large numbers of refugees, accounting for over 10 percent of annual global refugee resettlements; it resettled more than 28,000 in 2018.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 86
    _additional:{'distance': -144.10197}
    
    lang:en
    
    text:A highly developed country, Canada has the 24th highest nominal per capita income globally and the sixteenth-highest ranking on the Human Development Index. Its advanced economy is the eighth-largest in the world, relying chiefly upon its abundant natural resources and well-developed international trade networks. Canada is part of several major international and intergovernmental institutions or groupings including the United Nations, NATO, the G7, the Group of Ten, the G20, the Organisation for Economic Co-operation and Development (OECD), the World Trade Organization (WTO), the Commonwealth of Nations, the Arctic Council, the , the Asia-Pacific Economic Cooperation forum, and the Organization of American States.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 87
    _additional:{'distance': -144.0854}
    
    lang:en
    
    text:Toronto became the capital of the province of Ontario after its official creation in 1867. The seat of government of the Ontario Legislature is at Queen's Park. Because of its provincial capital status, the city was also the location of Government House, the residence of the viceregal representative of the Crown in right of Ontario.
    
    title:Toronto
    
    url:https://en.wikipedia.org/wiki?curid=64646
    
    views:3000
    
    
    item 88
    _additional:{'distance': -144.06784}
    
    lang:en
    
    text:The Canadian Heraldic Authority (CHA) grants former prime ministers an augmentation of honour on the coat of arms of those who apply for them. The heraldic badge, referred to by the CHA as the "mark of the Prime Ministership of Canada", consists of four red maple leaves joined at the stem on a white field ("Argent four maple leaves conjoined in cross at the stem Gules"); the augmentation is usually a canton or centred in the chief. Joe Clark, Pierre Trudeau, John Turner, Brian Mulroney, Kim Campbell, Jean Chrétien and Paul Martin were granted arms with the augmentation.
    
    title:Prime Minister of Canada
    
    url:https://en.wikipedia.org/wiki?curid=24135
    
    views:2000
    
    
    item 89
    _additional:{'distance': -144.02658}
    
    lang:en
    
    text:In 2011, Canadian forces participated in the NATO-led intervention into the Libyan Civil War, and also became involved in battling the Islamic State insurgency in Iraq in the mid-2010s. The COVID-19 pandemic in Canada began on January 27, 2020, with wide social and economic disruption. In 2021, the remains of hundreds of Indigenous people were discovered near the former sites of Canadian Indian residential schools. Administered by the Canadian Catholic Church and funded by the Canadian government from 1828 to 1997, these boarding schools attempted to assimilate Indigenous children into Euro-Canadian culture.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 90
    _additional:{'distance': -144.01222}
    
    lang:en
    
    text:Ottawa is headquarters to numerous major medical organizations and institutions such as Canadian Red Cross, Canadian Blood Services, Health Canada, Canadian Medical Association, Royal College of Physicians and Surgeons of Canada, Canadian Nurses Association, and the Medical Council of Canada.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 91
    _additional:{'distance': -144.00433}
    
    lang:en
    
    text:The skyline has been controlled by building height restrictions originally implemented to keep Parliament Hill and the Peace Tower at visible from most parts of the city. Today, several buildings are slightly taller than the Peace Tower, with the tallest being the Claridge Icon at 143 metres. Many federal buildings in the National Capital Region are managed by Public Works Canada, which leads to heritage conservation in its renovations and management of buildings, such as the renovation of the Senate Building. Most of the federal land in the region is managed by the National Capital Commission; its control of much undeveloped land and appropriations powers gives the NCC a great deal of influence over the city's development.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 92
    _additional:{'distance': -143.96895}
    
    lang:en
    
    text:Canada's constitution is based on the Westminster parliamentary model, wherein the role of the King is both legal and practical, but not political. The sovereign is vested with all the powers of state, collectively known as the royal prerogative, leading the populace to be considered subjects of the Crown. However, as the sovereign's power stems from the people and the monarch is a constitutional one, he or she does not rule alone, as in an absolute monarchy. Instead, the Crown is regarded as a corporation sole, with the monarch being the centre of a construct in which the power of the whole is shared by multiple institutions of government—the executive, legislative, and judicial—acting under the sovereign's authority, which is entrusted for exercise by the politicians (the elected and appointed parliamentarians and the ministers of the Crown generally drawn from among them) and the judges and justices of the peace. The monarchy has thus been described as the underlying principle of Canada's institutional unity and the monarch as a "guardian of constitutional freedoms" whose "job is to ensure that the political process remains intact and is allowed to function."
    
    title:Monarchy of Canada
    
    url:https://en.wikipedia.org/wiki?curid=56504
    
    views:2000
    
    
    item 93
    _additional:{'distance': -143.92328}
    
    lang:en
    
    text:While the monarchy is the source of authority in Canada, in practice its position is mainly symbolic. The use of the executive powers is directed by the Cabinet, a committee of ministers of the Crown responsible to the elected House of Commons and chosen and headed by the prime minister (at present Justin Trudeau), the head of government. The governor general or monarch may, though, in certain crisis situations exercise their power without ministerial advice. To ensure the stability of government, the governor general will usually appoint as prime minister the individual who is the current leader of the political party that can obtain the confidence of a plurality in the House of Commons. The Prime Minister's Office (PMO) is thus one of the most powerful institutions in government, initiating most legislation for parliamentary approval and selecting for appointment by the Crown, besides the aforementioned, the governor general, lieutenant governors, senators, federal court judges, and heads of Crown corporations and government agencies. The leader of the party with the second-most seats usually becomes the leader of the Official Opposition and is part of an adversarial parliamentary system intended to keep the government in check.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 94
    _additional:{'distance': -143.91855}
    
    lang:en
    
    text:The Constitution of Canada is the supreme law of the country, and consists of written text and unwritten conventions. The "Constitution Act, 1867" (known as the British North America Act, 1867 prior to 1982), affirmed governance based on parliamentary precedent and divided powers between the federal and provincial governments. The Statute of Westminster, 1931 granted full autonomy, and the "Constitution Act, 1982" ended all legislative ties to Britain, as well as adding a constitutional amending formula and the "Canadian Charter of Rights and Freedoms". The "Charter" guarantees basic rights and freedoms that usually cannot be over-ridden by any government—though a notwithstanding clause allows Parliament and the provincial legislatures to override certain sections of the "Charter" for a period of five years.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 95
    _additional:{'distance': -143.89182}
    
    lang:en
    
    text:The main symbol of the monarchy is the sovereign himself, described as "the personal expression of the Crown in Canada," and his image is thus used to signify Canadian sovereignty and government authority—his image, for instance, appearing on currency, and his portrait in government buildings. The sovereign is further both mentioned in and the subject of songs, loyal toasts, and salutes. A royal cypher, appearing on buildings and official seals, or a crown, seen on provincial and national coats of arms, as well as police force and Canadian Forces regimental and maritime badges and rank insignia, is also used to illustrate the monarchy as the locus of authority, the latter without referring to any specific monarch.
    
    title:Monarchy of Canada
    
    url:https://en.wikipedia.org/wiki?curid=56504
    
    views:2000
    
    
    item 96
    _additional:{'distance': -143.87836}
    
    lang:en
    
    text:In addition to the market's local media services, Ottawa is home to several national media operations, including CPAC (Canada's national legislature broadcaster) and the parliamentary bureau staff of virtually all of Canada's major newsgathering organizations in television, radio and print. The city is also home to the head office of the Canadian Broadcasting Corporation.
    
    title:Ottawa
    
    url:https://en.wikipedia.org/wiki?curid=22219
    
    views:2000
    
    
    item 97
    _additional:{'distance': -143.83647}
    
    lang:en
    
    text:Other prominent symbols include the national motto "" ("From Sea to Sea"), the sports of ice hockey and lacrosse, the beaver, Canada goose, common loon, Canadian horse, the Royal Canadian Mounted Police, the Canadian Rockies, and more recently the totem pole and Inuksuk. Material items such as Canadian beer, maple syrup, tuques, canoes, nanaimo bars, butter tarts and the Quebec dish of poutine are defined as uniquely Canadian. Canadian coins feature many of these symbols: the loon on the $1 coin, the Arms of Canada on the 50¢ piece, the beaver on the nickel. The penny, removed from circulation in 2013, featured the maple leaf. An image of the previous monarch, Queen Elizabeth II, appears on $20 bank notes, and on the obverse of all current Canadian coins.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 98
    _additional:{'distance': -143.83636}
    
    lang:en
    
    text:Canada has a parliamentary system within the context of a constitutional monarchy—the monarchy of Canada being the foundation of the executive, legislative, and judicial branches. The reigning monarch is , who is also monarch of 14 other Commonwealth countries and each of Canada's 10 provinces. The person who is the Canadian monarch is the same as the British monarch, although the two institutions are separate. The monarch appoints a representative, the governor general, with the advice of the prime minister, to carry out most of their federal royal duties in Canada.
    
    title:Canada
    
    url:https://en.wikipedia.org/wiki?curid=5042916
    
    views:4000
    
    
    item 99
    _additional:{'distance': -143.83032}
    
    lang:en
    
    text:Several companies are headquartered in Greater Montreal Area including Rio Tinto Alcan, Bombardier Inc., Canadian National Railway, CGI Group, Air Canada, Air Transat, CAE, Saputo, Cirque du Soleil, Stingray Group, Quebecor, Ultramar, Kruger Inc., Jean Coutu Group, Uniprix, Proxim, Domtar, Le Château, Power Corporation, Cellcom Communications, Bell Canada. Standard Life, Hydro-Québec, AbitibiBowater, Pratt and Whitney Canada, Molson, Tembec, Canada Steamship Lines, Fednav, Alimentation Couche-Tard, SNC-Lavalin, MEGA Brands, Aeroplan, Agropur, Metro Inc., Laurentian Bank of Canada, National Bank of Canada, Transat A.T., Via Rail, GardaWorld, Novacam Technologies, SOLABS, Dollarama, Rona and the Caisse de dépôt et placement du Québec.
    
    title:Montreal
    
    url:https://en.wikipedia.org/wiki?curid=7954681
    
    views:3000
    
    


### Improving Keyword Search with ReRank


```python
from utils import keyword_search
```


```python
query_1 = "What is the capital of Canada?"
```


```python
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                         num_results=3
                        )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))
```

    i:0
    Monarchy of Canada
    i:1
    Early modern period
    i:2
    Flag of Canada
    i:3
    Flag of Canada
    i:4
    Prime Minister of Canada
    i:5
    Hamilton, Ontario
    i:6
    Liberal Party of Canada
    i:7
    Stephen Harper
    i:8
    Monarchy of Canada
    i:9
    Flag of Canada
    i:10
    Order of Canada
    i:11
    University of Toronto
    i:12
    Newfoundland (island)
    i:13
    Liberal Party of Canada
    i:14
    Newfoundland (island)
    i:15
    Flag of Canada
    i:16
    North American Free Trade Agreement
    i:17
    Pea
    i:18
    Monarchy of Canada
    i:19
    Prime Minister of Canada
    i:20
    Hamilton, Ontario
    i:21
    Aesop's Fables
    i:22
    Revolutions of 1989
    i:23
    R.S.C. Anderlecht
    i:24
    Hudson's Bay Company
    i:25
    Liberal Party of Canada
    i:26
    2020–21 NBA season
    i:27
    Filibuster
    i:28
    Hardcore punk
    i:29
    Early modern period
    i:30
    Skopje
    i:31
    Venture capital
    i:32
    Wakanda
    i:33
    Arjuna
    i:34
    Luhansk
    i:35
    Arlington National Cemetery
    i:36
    North American Free Trade Agreement
    i:37
    Global North and Global South
    i:38
    Shia–Sunni relations
    i:39
    Jacob Zuma
    i:40
    Early modern period
    i:41
    Maui
    i:42
    Gerhard Schröder
    i:43
    Revolutions of 1989
    i:44
    Earl Warren
    i:45
    Mary Celeste
    i:46
    Exodus: Gods and Kings
    i:47
    Phnom Penh
    i:48
    Quebec
    i:49
    Air Canada
    i:50
    Americas
    i:51
    Canada
    i:52
    Indigenous peoples
    i:53
    Toronto Blue Jays
    i:54
    Canada men's national soccer team
    i:55
    Métis
    i:56
    Monarchy of Canada
    i:57
    Capitol Records
    i:58
    Air Canada
    i:59
    Canadian Broadcasting Corporation
    i:60
    Canada
    i:61
    Air Canada
    i:62
    Air Canada
    i:63
    Order of Canada
    i:64
    Canada
    i:65
    Order of Canada
    i:66
    Toronto
    i:67
    Air Canada
    i:68
    Ontario
    i:69
    Canada
    i:70
    United States men's national soccer team
    i:71
    Canada
    i:72
    George VI
    i:73
    Canada men's national soccer team
    i:74
    Canada
    i:75
    Steve Nash
    i:76
    Monarchy of Canada
    i:77
    Canada men's national soccer team
    i:78
    Pierre Trudeau
    i:79
    Pierre Trudeau
    i:80
    Air Canada
    i:81
    Air Canada
    i:82
    Monarchy of Canada
    i:83
    Air Canada
    i:84
    Air Canada
    i:85
    Air Canada
    i:86
    Canada
    i:87
    Walmart
    i:88
    Canadian Broadcasting Corporation
    i:89
    Canada
    i:90
    Air Canada
    i:91
    Wayne Gretzky
    i:92
    Canada
    i:93
    Underground Railroad
    i:94
    Ottawa
    i:95
    Montreal
    i:96
    Joni Mitchell
    i:97
    Air Canada
    i:98
    Conservative Party of Canada
    i:99
    Canada
    i:100
    Montreal
    i:101
    Millennials
    i:102
    Charles de Gaulle
    i:103
    Canada
    i:104
    Monarchy of Canada
    i:105
    Beaver
    i:106
    Canada men's national soccer team
    i:107
    Canada men's national soccer team
    i:108
    Winnipeg
    i:109
    Reindeer
    i:110
    Monarchy of Canada
    i:111
    Order of Canada
    i:112
    Country music
    i:113
    Platinum Jubilee of Elizabeth II
    i:114
    Quebec
    i:115
    Monarchy of Canada
    i:116
    Nestlé
    i:117
    Air Canada
    i:118
    Canada
    i:119
    Air Canada
    i:120
    Air Canada
    i:121
    Canada
    i:122
    McGill University
    i:123
    Ice hockey
    i:124
    Air Canada
    i:125
    Canada
    i:126
    Canada
    i:127
    Monarchy of Canada
    i:128
    Supertramp
    i:129
    Canada
    i:130
    Sildenafil
    i:131
    Justin Trudeau
    i:132
    Presbyterianism
    i:133
    Order of Canada
    i:134
    Provinces and territories of Canada
    i:135
    Canada
    i:136
    Canada men's national soccer team
    i:137
    Air Canada
    i:138
    North America
    i:139
    Canada
    i:140
    Flag of Canada
    i:141
    Constitution
    i:142
    Quebec
    i:143
    Degrassi: The Next Generation
    i:144
    Celine Dion
    i:145
    Quebec
    i:146
    Air Canada
    i:147
    Canadian Broadcasting Corporation
    i:148
    Mohammed bin Salman
    i:149
    Monarch butterfly
    i:150
    Canada
    i:151
    University of Toronto
    i:152
    Beaufort scale
    i:153
    Canada
    i:154
    Conservative Party of Canada
    i:155
    Assisted suicide
    i:156
    Liberal Party of Canada
    i:157
    Fleur-de-lis
    i:158
    Bobcat
    i:159
    Air Canada
    i:160
    Embraer E-Jet family
    i:161
    Canada
    i:162
    Canada men's national soccer team
    i:163
    Santa Claus
    i:164
    Canada
    i:165
    Monarchy of Canada
    i:166
    Canada
    i:167
    Canada
    i:168
    Great Reset
    i:169
    Stephen Harper
    i:170
    Methodism
    i:171
    De Havilland Canada Dash 8
    i:172
    Montreal
    i:173
    Martin Van Buren
    i:174
    Monarchy of Canada
    i:175
    Air Canada
    i:176
    Sidney Crosby
    i:177
    Rudyard Kipling
    i:178
    Canada
    i:179
    Kyoto Protocol
    i:180
    Quebec
    i:181
    Liberal Party of Canada
    i:182
    North America
    i:183
    Alphonso Davies
    i:184
    Liberal Party of Canada
    i:185
    Boeing F/A-18E/F Super Hornet
    i:186
    Underground Railroad
    i:187
    Canada men's national soccer team
    i:188
    Canada
    i:189
    The Marshall Mathers LP
    i:190
    2022 FIFA World Cup
    i:191
    History of slavery
    i:192
    War of 1812
    i:193
    Stephen Harper
    i:194
    Monarchy of Canada
    i:195
    Air Canada
    i:196
    History of Ukraine
    i:197
    Conservative Party of Canada
    i:198
    Commonwealth realm
    i:199
    Monarchy of Canada
    i:200
    Bryan Adams
    i:201
    Air Canada
    i:202
    Ottawa
    i:203
    Lynx
    i:204
    Air Canada
    i:205
    Air Canada
    i:206
    Air Canada
    i:207
    Methodism
    i:208
    Canada men's national soccer team
    i:209
    Calgary
    i:210
    Monarchy of Canada
    i:211
    Canada
    i:212
    Canada men's national soccer team
    i:213
    Canada
    i:214
    The Home Depot
    i:215
    Conservatism
    i:216
    Air Canada
    i:217
    Air Canada
    i:218
    Monarchy of Canada
    i:219
    Montreal
    i:220
    ICICI Bank
    i:221
    Doctor of Medicine
    i:222
    Great Plains
    i:223
    Mennonites
    i:224
    Union Jack
    i:225
    Canada
    i:226
    Air Canada
    i:227
    Capitol Records
    i:228
    Ontario
    i:229
    Regiment
    i:230
    Air Canada
    i:231
    Eugene Levy
    i:232
    Air Canada
    i:233
    North America
    i:234
    Reindeer
    i:235
    Liberal Party of Canada
    i:236
    Scotland
    i:237
    Tim Hortons
    i:238
    Chrystia Freeland
    i:239
    Air Canada
    i:240
    Canada men's national soccer team
    i:241
    Alphonso Davies
    i:242
    Air Canada
    i:243
    Common European Framework of Reference for Languages
    i:244
    Canada men's national soccer team
    i:245
    Provinces and territories of Canada
    i:246
    Canada
    i:247
    Ice hockey
    i:248
    Flag of Canada
    i:249
    Order of Canada
    i:250
    Quebec
    i:251
    Thanksgiving
    i:252
    Provinces and territories of Canada
    i:253
    Canada men's national soccer team
    i:254
    North American Free Trade Agreement
    i:255
    Provinces and territories of Canada
    i:256
    Alberta
    i:257
    Canada
    i:258
    Conservative Party of Canada
    i:259
    Calgary
    i:260
    Steve Nash
    i:261
    Canada
    i:262
    Al Capone
    i:263
    Monarchy of Canada
    i:264
    Conservative Party of Canada
    i:265
    Prime Minister of Canada
    i:266
    Underground Railroad
    i:267
    Air Canada
    i:268
    Heavy water
    i:269
    Boeing P-8 Poseidon
    i:270
    Alphonso Davies
    i:271
    Winnipeg
    i:272
    Flag of Canada
    i:273
    12 Rules for Life
    i:274
    Provinces and territories of Canada
    i:275
    Air Canada
    i:276
    Sidney Crosby
    i:277
    Canada
    i:278
    Calgary
    i:279
    Prime Minister of Canada
    i:280
    Christopher Plummer
    i:281
    Lynx
    i:282
    Wolf
    i:283
    Warner Music Group
    i:284
    Canada
    i:285
    Americas
    i:286
    Prince Edward, Duke of Kent and Strathearn
    i:287
    Canada
    i:288
    Provinces and territories of Canada
    i:289
    Provinces and territories of Canada
    i:290
    Canada
    i:291
    Liberal Party of Canada
    i:292
    Ontario
    i:293
    Pierre Trudeau
    i:294
    Three Days Grace
    i:295
    Canada
    i:296
    Reindeer
    i:297
    Newfoundland and Labrador
    i:298
    Millennials
    i:299
    London, Ontario
    i:300
    De Havilland Canada Dash 8
    i:301
    North America
    i:302
    Canada
    i:303
    COVID-19 pandemic
    i:304
    Canada men's national soccer team
    i:305
    Hyundai Motor Company
    i:306
    Monarchy of Canada
    i:307
    Volkswagen
    i:308
    Canada
    i:309
    War of 1812
    i:310
    Roaring Twenties
    i:311
    Countries of the United Kingdom
    i:312
    Gujarati people
    i:313
    McGill University
    i:314
    Canada
    i:315
    McGill University
    i:316
    Greater Toronto Area
    i:317
    Great Replacement
    i:318
    Monarchy of Canada
    i:319
    Canada
    i:320
    British Empire
    i:321
    Juris Doctor
    i:322
    Liberal Party of Canada
    i:323
    2022 Atlantic hurricane season
    i:324
    Canada
    i:325
    North American Free Trade Agreement
    i:326
    Monarchy of Canada
    i:327
    Sidney Crosby
    i:328
    Canada men's national soccer team
    i:329
    Beluga whale
    i:330
    Baby boomers
    i:331
    Canada men's national soccer team
    i:332
    'Ndrangheta
    i:333
    Commonwealth realm
    i:334
    Rugby union
    i:335
    Newfoundland and Labrador
    i:336
    Canada
    i:337
    Alberta
    i:338
    Iroquois
    i:339
    British Empire
    i:340
    Monarchy of Canada
    i:341
    Air Canada
    i:342
    Canada men's national soccer team
    i:343
    Avril Lavigne
    i:344
    Methodism
    i:345
    William Howard Taft
    i:346
    Monarchy of Canada
    i:347
    Acronym
    i:348
    Pat Benatar
    i:349
    Ontario
    i:350
    Monarchy of Canada
    i:351
    Flag of Canada
    i:352
    Connor McDavid
    i:353
    Foreigner (band)
    i:354
    Canada men's national soccer team
    i:355
    Schitt's Creek
    i:356
    2001: A Space Odyssey (film)
    i:357
    Petroleum
    i:358
    Calgary
    i:359
    Mikhail Baryshnikov
    i:360
    Nova Scotia
    i:361
    Chelsea Manning
    i:362
    Monarchy of Canada
    i:363
    Monarchy of Canada
    i:364
    Chiropractic
    i:365
    Liberal Party of Canada
    i:366
    Pierre Trudeau
    i:367
    The Batman (film)
    i:368
    Folk music
    i:369
    Canada men's national soccer team
    i:370
    Def Leppard
    i:371
    War of 1812
    i:372
    Def Leppard
    i:373
    Prime Minister of Canada
    i:374
    Canada
    i:375
    Dunkin' Donuts
    i:376
    ABBA
    i:377
    Order of Canada
    i:378
    Pierre Trudeau
    i:379
    Commonwealth realm
    i:380
    Nope (film)
    i:381
    Habeas corpus
    i:382
    Flag of Canada
    i:383
    Coronation Street
    i:384
    Jim Carrey
    i:385
    Celine Dion
    i:386
    Electronic cigarette
    i:387
    Venture capital
    i:388
    Air Canada
    i:389
    Ethnic group
    i:390
    Border Collie
    i:391
    Order of Canada
    i:392
    Air Canada
    i:393
    Nickelback
    i:394
    Ottawa
    i:395
    Air Canada
    i:396
    Trade union
    i:397
    Prince Edward Island
    i:398
    Air Canada
    i:399
    Nova Scotia
    i:400
    Donald Sutherland
    i:401
    School shooting
    i:402
    Ice hockey
    i:403
    Kyoto Protocol
    i:404
    Canada
    i:405
    Ontario
    i:406
    Lady-in-waiting
    i:407
    Ottawa
    i:408
    Conservative Party of Canada
    i:409
    Heartland (Canadian TV series)
    i:410
    Air Canada
    i:411
    IKEA
    i:412
    Tristan Thompson
    i:413
    Fringe (TV series)
    i:414
    Suzuki
    i:415
    Air Canada
    i:416
    Provinces and territories of Canada
    i:417
    National Hockey League
    i:418
    Russell Williams (criminal)
    i:419
    Coat of arms
    i:420
    ICICI Bank
    i:421
    Canadian Broadcasting Corporation
    i:422
    Kevin O'Leary
    i:423
    University of Toronto
    i:424
    Dodge
    i:425
    Proud Boys
    i:426
    Air Canada
    i:427
    Family Guy
    i:428
    Canada
    i:429
    Union Jack
    i:430
    Canada men's national soccer team
    i:431
    Flag of Canada
    i:432
    Air Canada
    i:433
    Credit card
    i:434
    Canada
    i:435
    Justin Trudeau
    i:436
    Acre
    i:437
    McGill University
    i:438
    Conservation status
    i:439
    YMCA
    i:440
    Canadian dollar
    i:441
    ITER
    i:442
    The Mist (film)
    i:443
    The Big Bang Theory
    i:444
    Air Canada
    i:445
    Fair use
    i:446
    Provinces and territories of Canada
    i:447
    Osteopathy
    i:448
    Midsomer Murders
    i:449
    North American Free Trade Agreement
    i:450
    Rolling Stone's 100 Greatest Artists of All Time
    i:451
    Union Jack
    i:452
    Commonwealth Games
    i:453
    Canada
    i:454
    Canada
    i:455
    London, Ontario
    i:456
    Super Bowl LV
    i:457
    Mackenzie Phillips
    i:458
    Toyota
    i:459
    Alberta
    i:460
    Toronto Pearson International Airport
    i:461
    Mennonites
    i:462
    Adobe Inc.
    i:463
    Vancouver
    i:464
    Canada men's national soccer team
    i:465
    Canada
    i:466
    Columbia Records
    i:467
    Newfoundland (island)
    i:468
    Winnipeg
    i:469
    Justin Trudeau
    i:470
    Victoria, British Columbia
    i:471
    The Lord of the Rings: The Fellowship of the Ring
    i:472
    Air Canada
    i:473
    Toronto Blue Jays
    i:474
    Toronto Pearson International Airport
    i:475
    Calgary
    i:476
    Bell UH-1 Iroquois
    i:477
    Nicolas Sarkozy
    i:478
    Fake news
    i:479
    Ottawa
    i:480
    Marshall Plan
    i:481
    Canada
    i:482
    Winnipeg
    i:483
    Marshmello
    i:484
    Conservative Party of Canada
    i:485
    Canadian Broadcasting Corporation
    i:486
    Yellowstone (American TV series)
    i:487
    Edmonton
    i:488
    Sitting Bull
    i:489
    North American Free Trade Agreement
    i:490
    Canada men's national soccer team
    i:491
    Order of Canada
    i:492
    Iroquois
    i:493
    Prince Edward Island
    i:494
    Homosexuality
    i:495
    Canada
    i:496
    Quebec City
    i:497
    Prime Minister of Canada
    i:498
    Martin Short
    i:499
    Church of England



```python
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                         num_results=500
                        )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    #print(result.get('text'))
```


```python
def rerank_responses(query, responses, num_responses=10):
    reranked_responses = co.rerank(
        model = 'rerank-english-v2.0',
        query = query,
        documents = responses,
        top_n = num_responses,
        )
    return reranked_responses
```


```python
texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_1, texts)
```


```python
for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()
```

    i:0
    RerankResult<document['text']: Selection of Ottawa as the capital of Canada predates the Confederation of Canada. The selection was contentious and not straightforward, with the parliament of the United Province of Canada holding more than 200 votes over several decades to attempt to settle on a legislative solution to the location of the capital., index: 407, relevance_score: 0.9875684>
    
    i:1
    RerankResult<document['text']: Montreal was the capital of the Province of Canada from 1844 to 1849, but lost its status when a Tory mob burnt down the Parliament building to protest the passage of the Rebellion Losses Bill. Thereafter, the capital rotated between Quebec City and Toronto until in 1857, Queen Victoria herself established Ottawa as the capital due to strategic reasons. The reasons were twofold. First, because it was located more in the interior of the Province of Canada, it was less susceptible to attack from the United States. Second, and perhaps more importantly, because it lay on the border between French and English Canada, Ottawa was seen as a compromise between Montreal, Toronto, Kingston and Quebec City, which were all vying to become the young nation's official capital. Ottawa retained the status as capital of Canada when the Province of Canada joined with Nova Scotia and New Brunswick to form the Dominion of Canada in 1867., index: 100, relevance_score: 0.9795897>
    
    i:2
    RerankResult<document['text']: Ottawa is the political centre of Canada and headquarters to the federal government. The city houses numerous foreign embassies, key buildings, organizations, and institutions of Canada's government, including the Parliament of Canada, the Supreme Court, the residence of Canada's viceroy, and Office of the Prime Minister., index: 202, relevance_score: 0.9753901>
    
    i:3
    RerankResult<document['text']: Until the late 18th century Québec was the most populous city in present-day Canada. As of the census of 1790, Montreal surpassed it with 18,000 inhabitants, but Quebec (pop. 14,000) remained the administrative capital of New France. It was then made the capital of Lower Canada by the Constitutional Act of 1791. From 1841 to 1867, the capital of the Province of Canada rotated between Kingston, Montreal, Toronto, Ottawa and Quebec City (from 1852 to 1856 and from 1859 to 1866)., index: 496, relevance_score: 0.9711838>
    
    i:4
    RerankResult<document['text']: Ottawa was chosen as the capital for two primary reasons. First, Ottawa's isolated location, surrounded by dense forest far from the Canada–US border and situated on a cliff face, would make it more defensible from attack. Second, Ottawa was approximately midway between Toronto and Kingston (in Canada West) and Montreal and Quebec City (in Canada East) making the selection an important political compromise., index: 479, relevance_score: 0.96653706>
    
    i:5
    RerankResult<document['text']: Canada is a country in North America. Its ten provinces and three territories extend from the Atlantic Ocean to the Pacific Ocean and northward into the Arctic Ocean, covering over , making it the world's second-largest country by total area. Its southern and western border with the United States, stretching , is the world's longest binational land border. Canada's capital is Ottawa, and its three largest metropolitan areas are Toronto, Montreal, and Vancouver., index: 481, relevance_score: 0.9421884>
    
    i:6
    RerankResult<document['text']: Although both rebellions were put down in short order, the British government sent Lord Durham to investigate the causes. He recommended self-government be granted and Lower and Upper Canada be re-joined in an attempt to assimilate the French Canadians. Accordingly, the two colonies were merged into the Province of Canada by the "Act of Union 1840", with the capital at Kingston, and Upper Canada becoming known as Canada West. Parliamentary self-government was granted in 1848. There were heavy waves of immigration in the 1840s, and the population of Canada West more than doubled by 1851 over the previous decade. As a result, for the first time, the English-speaking population of Canada West surpassed the French-speaking population of Canada East, tilting the representative balance of power., index: 68, relevance_score: 0.86567897>
    
    i:7
    RerankResult<document['text']: Ottawa is headquarters to numerous major medical organizations and institutions such as Canadian Red Cross, Canadian Blood Services, Health Canada, Canadian Medical Association, Royal College of Physicians and Surgeons of Canada, Canadian Nurses Association, and the Medical Council of Canada., index: 394, relevance_score: 0.86153823>
    
    i:8
    RerankResult<document['text']: Ontario ( ; ) is one of the thirteen provinces and territories of Canada. Located in Central Canada, it is Canada's most populous province, with 38.3 percent of the country's population, and is the second-largest province by total area (after Quebec). Ontario is Canada's fourth-largest jurisdiction in total area when the territories of the Northwest Territories and Nunavut are included. It is home to the nation's capital city, Ottawa, and the nation's most populous city, Toronto, which is Ontario's provincial capital., index: 228, relevance_score: 0.4989891>
    
    i:9
    RerankResult<document['text']: With sixty percent of Canada's steel produced in Hamilton by Stelco and Dofasco, the city has become known as the Steel Capital of Canada. After nearly declaring bankruptcy, Stelco returned to profitability in 2004. On August 26, 2007 United States Steel Corporation acquired Stelco for C$38.50 in cash per share, owning more than 76 percent of Stelco's outstanding shares. On September 17, 2014, US Steel Canada announced it was applying for bankruptcy protection and it would close its Hamilton operations., index: 5, relevance_score: 0.49455282>
    


### Improving Dense Retrieval with ReRank


```python
from utils import dense_retrieval
```


```python
query_2 = "Who is the tallest person in history?"
```


```python
results = dense_retrieval(query_2,client)
```


```python
for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))
    print()
```

    i:0
    Robert Wadlow
    
    i:1
    Manute Bol
    
    i:2
    Sultan Kösen
    
    i:3
    Sultan Kösen
    
    i:4
    Netherlands
    
    i:5
    Robert Wadlow
    
    i:6
    Randy Johnson
    
    i:7
    Manute Bol
    
    i:8
    Harald Hardrada
    
    i:9
    Manute Bol
    



```python
texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_2, texts)
```


```python
for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()
```

    i:0
    RerankResult<document['text']: Robert Pershing Wadlow (February 22, 1918 July 15, 1940), also known as the Alton Giant and the Giant of Illinois, was a man who was the tallest person in recorded history for whom there is irrefutable evidence. He was born and raised in Alton, Illinois, a small city near St. Louis, Missouri., index: 0, relevance_score: 0.9734939>
    
    i:1
    RerankResult<document['text']: Sultan Kösen (born 10 December 1982) is a Turkish farmer who holds the Guinness World Record for tallest living male at . Of Kurdish ethnicity, he is the seventh tallest man in history., index: 2, relevance_score: 0.8664718>
    
    i:2
    RerankResult<document['text']: The Dutch are the tallest people in the world, by nationality, with an average height of for adult males and for adult females in 2009. The average height of young males in the Netherlands increased from 5 feet, 4 inches to approximately 6 feet between the 1850s until the early 2000s. People in the south are on average about shorter than those in the north., index: 4, relevance_score: 0.80162543>
    
    i:3
    RerankResult<document['text']: Kösen turned 40 years old on 10 December 2022. He celebrated his birthday a few days early by visiting the Ripley's Believe It or Not! museum in Orlando, Florida, USA and posing next to a life-sized statue of Robert Wadlow, the tallest man ever at 272 cm (8 ft 11.1 in)., index: 3, relevance_score: 0.6874202>
    
    i:4
    RerankResult<document['text']: Bol came from a family of extraordinarily tall men and women. He said: "My mother was , my father , and my sister is . And my great-grandfather was even taller—." His ethnic group, the Dinka, and the Nilotic people of which they are a part, are among the tallest populations in the world. Bol's hometown, Turalei, is the origin of other exceptionally tall people, including basketball player Ring Ayuel. "I was born in a village, where you cannot measure yourself," Bol reflected. "I learned I was 7 foot 7 in 1979, when I was grown. I was about 18 or 19.", index: 1, relevance_score: 0.6396235>
    
## generating-answers

![](https://i.imgur.com/VN9mivQ.png)

![](https://i.imgur.com/LJ7qIh0.png)

![](https://i.imgur.com/k9OYY63.png)

![](https://i.imgur.com/7zgUZuE.png)

lama index 和langchain import PDF text
## Generating Answers


```python
question = "Are side projects important when you are starting to learn about AI?"
```


```python
text = """
The rapid rise of AI has led to a rapid rise in AI jobs, and many people are building exciting careers in this field. A career is a decades-long journey, and the path is not always straightforward. Over many years, I’ve been privileged to see thousands of students as well as engineers in companies large and small navigate careers in AI. In this and the next few letters, I’d like to share a few thoughts that might be useful in charting your own course.

Three key steps of career growth are learning (to gain technical and other skills), working on projects (to deepen skills, build a portfolio, and create impact) and searching for a job. These steps stack on top of each other:

Initially, you focus on gaining foundational technical skills.
After having gained foundational skills, you lean into project work. During this period, you’ll probably keep learning.
Later, you might occasionally carry out a job search. Throughout this process, you’ll probably continue to learn and work on meaningful projects.
These phases apply in a wide range of professions, but AI involves unique elements. For example:

AI is nascent, and many technologies are still evolving. While the foundations of machine learning and deep learning are maturing — and coursework is an efficient way to master them — beyond these foundations, keeping up-to-date with changing technology is more important in AI than fields that are more mature.
Project work often means working with stakeholders who lack expertise in AI. This can make it challenging to find a suitable project, estimate the project’s timeline and return on investment, and set expectations. In addition, the highly iterative nature of AI projects leads to special challenges in project management: How can you come up with a plan for building a system when you don’t know in advance how long it will take to achieve the target accuracy? Even after the system has hit the target, further iteration may be necessary to address post-deployment drift.
While searching for a job in AI can be similar to searching for a job in other sectors, there are some differences. Many companies are still trying to figure out which AI skills they need and how to hire people who have them. Things you’ve worked on may be significantly different than anything your interviewer has seen, and you’re more likely to have to educate potential employers about some elements of your work.
Throughout these steps, a supportive community is a big help. Having a group of friends and allies who can help you — and whom you strive to help — makes the path easier. This is true whether you’re taking your first steps or you’ve been on the journey for years.

I’m excited to work with all of you to grow the global AI community, and that includes helping everyone in our community develop their careers. I’ll dive more deeply into these topics in the next few weeks.

Last week, I wrote about key steps for building a career in AI: learning technical skills, doing project work, and searching for a job, all of which is supported by being part of a community. In this letter, I’d like to dive more deeply into the first step.

More papers have been published on AI than any person can read in a lifetime. So, in your efforts to learn, it’s critical to prioritize topic selection. I believe the most important topics for a technical career in machine learning are:

Foundational machine learning skills. For example, it’s important to understand models such as linear regression, logistic regression, neural networks, decision trees, clustering, and anomaly detection. Beyond specific models, it’s even more important to understand the core concepts behind how and why machine learning works, such as bias/variance, cost functions, regularization, optimization algorithms, and error analysis.
Deep learning. This has become such a large fraction of machine learning that it’s hard to excel in the field without some understanding of it! It’s valuable to know the basics of neural networks, practical skills for making them work (such as hyperparameter tuning), convolutional networks, sequence models, and transformers.
Math relevant to machine learning. Key areas include linear algebra (vectors, matrices, and various manipulations of them) as well as probability and statistics (including discrete and continuous probability, standard probability distributions, basic rules such as independence and Bayes rule, and hypothesis testing). In addition, exploratory data analysis (EDA) — using visualizations and other methods to systematically explore a dataset — is an underrated skill. I’ve found EDA particularly useful in data-centric AI development, where analyzing errors and gaining insights can really help drive progress! Finally, a basic intuitive understanding of calculus will also help. In a previous letter, I described how the math needed to do machine learning well has been changing. For instance, although some tasks require calculus, improved automatic differentiation software makes it possible to invent and implement new neural network architectures without doing any calculus. This was almost impossible a decade ago.
Software development. While you can get a job and make huge contributions with only machine learning modeling skills, your job opportunities will increase if you can also write good software to implement complex AI systems. These skills include programming fundamentals, data structures (especially those that relate to machine learning, such as data frames), algorithms (including those related to databases and data manipulation), software design, familiarity with Python, and familiarity with key libraries such as TensorFlow or PyTorch, and scikit-learn.
This is a lot to learn! Even after you master everything in this list, I hope you’ll keep learning and continue to deepen your technical knowledge. I’ve known many machine learning engineers who benefitted from deeper skills in an application area such as natural language processing or computer vision, or in a technology area such as probabilistic graphical models or building scalable software systems.

How do you gain these skills? There’s a lot of good content on the internet, and in theory reading dozens of web pages could work. But when the goal is deep understanding, reading disjointed web pages is inefficient because they tend to repeat each other, use inconsistent terminology (which slows you down), vary in quality, and leave gaps. That’s why a good course — in which a body of material has been organized into a coherent and logical form — is often the most time-efficient way to master a meaningful body of knowledge. When you’ve absorbed the knowledge available in courses, you can switch over to research papers and other resources.

Finally, keep in mind that no one can cram everything they need to know over a weekend or even a month. Everyone I know who’s great at machine learning is a lifelong learner. In fact, given how quickly our field is changing, there’s little choice but to keep learning if you want to keep up. How can you maintain a steady pace of learning for years? I’ve written about the value of habits. If you cultivate the habit of learning a little bit every week, you can make significant progress with what feels like less effort.

In the last two letters, I wrote about developing a career in AI and shared tips for gaining technical skills. This time, I’d like to discuss an important step in building a career: project work.

It goes without saying that we should only work on projects that are responsible and ethical, and that benefit people. But those limits leave a large variety to choose from. I wrote previously about how to identify and scope AI projects. This and next week’s letter have a different emphasis: picking and executing projects with an eye toward career development.

A fruitful career will include many projects, hopefully growing in scope, complexity, and impact over time. Thus, it is fine to start small. Use early projects to learn and gradually step up to bigger projects as your skills grow.

When you’re starting out, don’t expect others to hand great ideas or resources to you on a platter. Many people start by working on small projects in their spare time. With initial successes — even small ones — under your belt, your growing skills increase your ability to come up with better ideas, and it becomes easier to persuade others to help you step up to bigger projects.

What if you don’t have any project ideas? Here are a few ways to generate them:

Join existing projects. If you find someone else with an idea, ask to join their project.
Keep reading and talking to people. I come up with new ideas whenever I spend a lot of time reading, taking courses, or talking with domain experts. I’m confident that you will, too.
Focus on an application area. Many researchers are trying to advance basic AI technology — say, by inventing the next generation of transformers or further scaling up language models — so, while this is an exciting direction, it is hard. But the variety of applications to which machine learning has not yet been applied is vast! I’m fortunate to have been able to apply neural networks to everything from autonomous helicopter flight to online advertising, partly because I jumped in when relatively few people were working on those applications. If your company or school cares about a particular application, explore the possibilities for machine learning. That can give you a first look at a potentially creative application — one where you can do unique work — that no one else has done yet.
Develop a side hustle. Even if you have a full-time job, a fun project that may or may not develop into something bigger can stir the creative juices and strengthen bonds with collaborators. When I was a full-time professor, working on online education wasn’t part of my “job” (which was doing research and teaching classes). It was a fun hobby that I often worked on out of passion for education. My early experiences recording videos at home helped me later in working on online education in a more substantive way. Silicon Valley abounds with stories of startups that started as side projects. So long as it doesn’t create a conflict with your employer, these projects can be a stepping stone to something significant.
Given a few project ideas, which one should you jump into? Here’s a quick checklist of factors to consider:

Will the project help you grow technically? Ideally, it should be challenging enough to stretch your skills but not so hard that you have little chance of success. This will put you on a path toward mastering ever-greater technical complexity.
Do you have good teammates to work with? If not, are there people you can discuss things with? We learn a lot from the people around us, and good collaborators will have a huge impact on your growth.
Can it be a stepping stone? If the project is successful, will its technical complexity and/or business impact make it a meaningful stepping stone to larger projects? (If the project is bigger than those you’ve worked on before, there’s a good chance it could be such a stepping stone.)
Finally, avoid analysis paralysis. It doesn’t make sense to spend a month deciding whether to work on a project that would take a week to complete. You'll work on multiple projects over the course of your career, so you’ll have ample opportunity to refine your thinking on what’s worthwhile. Given the huge number of possible AI projects, rather than the conventional “ready, aim, fire” approach, you can accelerate your progress with “ready, fire, aim.”

"""
```

### Setup

Load needed API keys and relevant Python libaries.


```python
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```


```python
import cohere

import numpy as np
import warnings
warnings.filterwarnings('ignore')
```

### Chunking


```python
# Split into a list of paragraphs
texts = text.split('\n\n')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts if t])
```


```python
texts[:3]
```




    array(['The rapid rise of AI has led to a rapid rise in AI jobs, and many people are building exciting careers in this field. A career is a decades-long journey, and the path is not always straightforward. Over many years, I’ve been privileged to see thousands of students as well as engineers in companies large and small navigate careers in AI. In this and the next few letters, I’d like to share a few thoughts that might be useful in charting your own course.',
           'Three key steps of career growth are learning (to gain technical and other skills), working on projects (to deepen skills, build a portfolio, and create impact) and searching for a job. These steps stack on top of each other:',
           'Initially, you focus on gaining foundational technical skills.\nAfter having gained foundational skills, you lean into project work. During this period, you’ll probably keep learning.\nLater, you might occasionally carry out a job search. Throughout this process, you’ll probably continue to learn and work on meaningful projects.\nThese phases apply in a wide range of professions, but AI involves unique elements. For example:'],
          dtype='<U2738')



### Embeddings


```python
co = cohere.Client(os.environ['COHERE_API_KEY'])

# Get the embeddings
response = co.embed(
    texts=texts.tolist(),
).embeddings

```

### Build a search index


```python
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
```


```python
# Check the dimensions of the embeddings
embeds = np.array(response)

# Create the search index, pass the size of embedding
search_index = AnnoyIndex(embeds.shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10) # 10 trees
search_index.save('test.ann')
```




    True



### Searching Articles


```python
def search_andrews_article(query):
    # Get the query's embedding
    query_embed = co.embed(texts=[query]).embeddings
    
    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0],
                                                    10,
                                                  include_distances=True)

    search_results = texts[similar_item_ids[0]]
    
    return search_results
```


```python
results = search_andrews_article(
    "Are side projects a good idea when trying to build a career in AI?"
)

print(results[0])
```

    Join existing projects. If you find someone else with an idea, ask to join their project.
    Keep reading and talking to people. I come up with new ideas whenever I spend a lot of time reading, taking courses, or talking with domain experts. I’m confident that you will, too.
    Focus on an application area. Many researchers are trying to advance basic AI technology — say, by inventing the next generation of transformers or further scaling up language models — so, while this is an exciting direction, it is hard. But the variety of applications to which machine learning has not yet been applied is vast! I’m fortunate to have been able to apply neural networks to everything from autonomous helicopter flight to online advertising, partly because I jumped in when relatively few people were working on those applications. If your company or school cares about a particular application, explore the possibilities for machine learning. That can give you a first look at a potentially creative application — one where you can do unique work — that no one else has done yet.
    Develop a side hustle. Even if you have a full-time job, a fun project that may or may not develop into something bigger can stir the creative juices and strengthen bonds with collaborators. When I was a full-time professor, working on online education wasn’t part of my “job” (which was doing research and teaching classes). It was a fun hobby that I often worked on out of passion for education. My early experiences recording videos at home helped me later in working on online education in a more substantive way. Silicon Valley abounds with stories of startups that started as side projects. So long as it doesn’t create a conflict with your employer, these projects can be a stepping stone to something significant.
    Given a few project ideas, which one should you jump into? Here’s a quick checklist of factors to consider:


### Generating Answers


```python
def ask_andrews_article(question, num_generations=1):
    
    # Search the text archive
    results = search_andrews_article(question)

    # Get the top result
    context = results[0]

    # Prepare the prompt
    prompt = f"""
    Excerpt from the article titled "How to Build a Career in AI" 
    by Andrew Ng: 
    {context}
    Question: {question}
    
    Extract the answer of the question from the text provided. 
    If the text doesn't contain the answer, 
    reply that the answer is not available."""

    prediction = co.generate(
        prompt=prompt,
        max_tokens=70,
        model="command-nightly",
        temperature=0.5,
        num_generations=num_generations
    )

    return prediction.generations
```


```python
results = ask_andrews_article(
    "Are side projects a good idea when trying to build a career in AI?",

)

print(results[0])
```


```python
results = ask_andrews_article(
    "Are side projects a good idea when trying to build a career in AI?",
    num_generations=3
)

for gen in results:
    print(gen)
    print('--')
```

     Yes, side projects are a good idea when trying to build a career in AI. They can help you develop your skills and knowledge, and can also be a stepping stone to a more substantive project. However, it is important to ensure that your side project does not create a conflict with your employer.
    --
     Yes. Side projects are a good idea when trying to build a career in AI. They can help you to develop new ideas and to strengthen bonds with collaborators. They can also be a stepping stone to something more significant. However, it is important to note that you should not create a conflict with your employer.
    --
     Yes. A side hustle can stir the creative juices and strengthen bonds with collaborators.
    --



```python
results = ask_andrews_article(
    "What is the most viewed televised event?",
    num_generations=5
)

```


```python
for gen in results:
    print(gen)
    print('--')
```

     The most viewed televised event is the Super Bowl.
    --
     The most viewed televised event is the Super Bowl.
    --
     The most viewed televised event is the Super Bowl.
    --
     The most viewed televised event is the Super Bowl.
    --
     The most viewed televised event is the Super Bowl.
    --



```python

```


```python

```


```python

```


```python

```

### Congratulations on finishing the course!

To start building with the Cohere LLMs, get your API key by registering [here](https://dashboard.cohere.ai/welcome/register?utm_source=partner&utm_medium=website&utm_campaign=DeeplearningAI). 

Learn more about LLMs at Cohere’s [LLM.University](https://LLM.University).

## Ref
- https://learn.deeplearning.ai/large-language-models-semantic-search
- 