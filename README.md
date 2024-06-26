# chiedo-bot

Project for cloud engineering course

## Project Description

This is a project on creating a chatbot to find information on unibz website, regarding university details. 

There will be a simple web application where current students, prospective students and professors can ask questions and expect a reply of the answers. The web application as well as the backend infrastructure will be hosted on AWS. 

Hosting this web application on cloud allows for flexibility and scalability during high demand period, ie. when the university application period opens. 

This project aims to explore the following questions:
1. What are the possible approaches to implement the backend architecture?
2. How does the application scale and perform?
3. What are the most costly components in terms of time and price?
4. How does the application ensure reliability?

## Notes

To run the demo script in local may need to set the following variable:
```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
poetry run python3 demo-script.py 
```

To run docker compose, first increase virtual memory limit
```
colima ssh
sudo sysctl -w vm.max_map_count=262144
exit
```

## Sample Questions
1. What is the tuition fees fuori corso?
2. What is the fine for delayed payment?
3. How much is the revenue stamp?
