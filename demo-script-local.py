from haystack.pipelines import Pipeline
from haystack.nodes import PreProcessor, Crawler, BM25Retriever, FARMReader

from haystack.document_stores import InMemoryDocumentStore

import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

crawler = Crawler(
    urls=["https://guide.unibz.it/en/study-career/engineering/fees/"],   # Websites to crawl
    crawler_depth=0,    # How many links to follow
    output_dir="crawled_files_0",  # The directory to store the crawled files, not very important, we don't use the files in this example
)
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)
document_store = InMemoryDocumentStore(use_bm25=True)
indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=crawler, name="crawler", inputs=['File'])
indexing_pipeline.add_node(component=preprocessor, name="preprocessor", inputs=['crawler'])
indexing_pipeline.add_node(component=document_store, name="document_store", inputs=['preprocessor'])

indexing_pipeline.run()

retriever = BM25Retriever(document_store=document_store)
reader =  FARMReader(model_name_or_path="deepset/roberta-base-squad2-distilled")

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
query_pipeline.add_node(component=reader, name="reader", inputs=["retriever"])


results = query_pipeline.run(query="How much is the revenue stamp?")

print("\nQuestion: ", results["query"])
print("\nAnswers:")
# for answer in results["answers"]:
#     print("- ", answer.answer)
for index in range(0,2):
    print("- answer:", results["answers"][index].answer, ", score: ", results["answers"][index].score)