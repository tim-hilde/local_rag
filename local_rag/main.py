import chromadb
from llama_index.core import (
	Settings,
	SimpleDirectoryReader,
	StorageContext,
	VectorStoreIndex,
)
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm = Ollama("llama3.2")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# create the chroma client and add our data
remote_db = chromadb.HttpClient()
chroma_collection = remote_db.get_or_create_collection("local_rag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

input_dir = "~/Zettelkasten/"

reader = SimpleDirectoryReader(input_dir=input_dir, required_exts=[".md"])
documents = reader.load_data(num_workers=4)

markdown_parser = MarkdownNodeParser(include_metadata=True)

title_extractor = TitleExtractor()

ingestion_pipeline = IngestionPipeline(
	transformations=[markdown_parser, title_extractor]
)

nodes = ingestion_pipeline.run(documents=documents)
index = VectorStoreIndex.build_index_from_nodes(nodes, storage_context=storage_context)
