from haystack.pipelines import Pipeline
from haystack.nodes import PreProcessor, Crawler, BM25Retriever, FARMReader

from haystack.document_stores import OpenSearchDocumentStore

import os
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.WARN)

document_store = OpenSearchDocumentStore(
    username = 'admin',
    password = os.getenv('OPENSEARCH_INITIAL_ADMIN_PASSWORD'),
    host = 'localhost',
    port = '9200'
)

def crawl_websites():
    crawler = Crawler(
        urls=["https://guide.unibz.it/en/study-career/engineering/fees/", "https://guide.unibz.it/en/languages/language-courses/faq/"],   # Websites to crawl
        crawler_depth=0,    # How many links to follow
        output_dir="crawled_files_1",
    )
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=500,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=crawler, name="crawler", inputs=['File'])
    indexing_pipeline.add_node(component=preprocessor, name="preprocessor", inputs=['crawler'])
    indexing_pipeline.add_node(component=document_store, name="document_store", inputs=['preprocessor'])

    indexing_pipeline.run()


retriever = BM25Retriever(document_store=document_store)
reader =  FARMReader(model_name_or_path="deepset/roberta-base-squad2-distilled")

def query_results(question):

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    query_pipeline.add_node(component=reader, name="reader", inputs=["retriever"])


    results = query_pipeline.run(query=question)

    print("\nQuestion: ", results["query"])
    print("\nAnswers:")

    for index in range(0,2):
        if results["answers"][index].score >= 0.5:
            print("- answer:", results["answers"][index].answer, ", score: ", results["answers"][index].score)    


def main() -> None:
    # crawl_websites()
    while True:
        question = input("Enter a question here: ")
        query_results(question)

    
if __name__ == '__main__':
    main()