# MCP-server-for-restaurant-review-data-that-is-added-to-vector-db.
MCP server for restaurant review data that is added to vector db. Having functionality of web search and persistent memory
MCP server will store restaurent review data. Once MCP is created, this will be used by any other resources.

Claude desktop will access MCP server as result and will have details of restaurant review data. Now anyone using claude desktop can also query about restaurent review data also

run ollama locally

1)
uptill now Old program created as below.  Programs are shared in main.py, memory.py and vector.py
Created program RAG pipeline created to load restaurant review data using chroma db (code already shared in vector.py)
Created persistent memory to store prompts and responses --  memory.py
logic is that we will ask for restaurant review queries which will be first looked into RAG pipeline where doc is provided -- vector.py
Then looked into google search    - main.py
Then provide response
Webui is created using streamlit - main.py
run ollama locally

2)
Now new requirement is to create MCP server for all above. MCP server will be access by cloud desktop and can see context data for restaurant review data form vector db sqllite, and google search from  tavily web search
MCP server should expose these functions. 
realistic_restaurant_reviews data from xls 
sqllite contain data from realistic_restaurant_reviews.csv which contain columns as Title,Date,Rating,Review. so all can be view using read access only from RAG. expose these functions patterns to know reviews of various restaurant, tehre rating, title and date.


