import os
import chromadb
from llama_index.core import (
	Settings,
	SimpleDirectoryReader,
	StorageContext,
	VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

llm = Ollama("llama3.2")
Settings.llm = llm
Settings.embed_model = OllamaEmbedding("snowflake-arctic-embed:33m")

# create the chroma client and add our data
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("local_rag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

db_path = "./chroma_db"
directory_exists = os.path.exists(db_path) and os.path.isdir(db_path)
if directory_exists:
	index = VectorStoreIndex.from_vector_store(vector_store)
else:
	# ingestion
	input_dir = "~/Zettelkasten/"

	reader = SimpleDirectoryReader(input_dir=input_dir, required_exts=[".md"])
	documents = reader.load_data(num_workers=1)

	markdown_parser = MarkdownNodeParser(include_metadata=True)

	ingestion_pipeline = IngestionPipeline(transformations=[markdown_parser])

	nodes = ingestion_pipeline.run(documents=documents)
	# Index/embedding
	index = VectorStoreIndex(nodes, storage_context=storage_context)


response_modes = [
	"refine",
	"compact",
	"tree_summarize",
	"simple_summarize",
	"no_text",
	"accumulate",
	"compact_accumulate",
]

query = "Was ist technische Intentionalität?"

for mode in response_modes:
	query_engine = index.as_query_engine(response_mode=mode)
	response = query_engine.query(query)
	print(f"\nResponse mode: {mode}\n")
	print(response)

retriever = index.as_retriever()
retrieved_noed = retriever.query(query)
