#!/usr/bin/env python3
"""
GraphRAG Entity Extraction for Knowledge Graph Generation

"""

# Import graphrag package which sets up our module structure
import asyncio
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer

from src.db import Neo4jGraphDB

# from superb_agent_tool_graphrag.handlers.custom_graph_transformer import LocationAwareLLMGraphTransformer
from src.llm import ChatOpenAITool

# from src.retrieval_server import update_node_types, update_relation_types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("G-RAG-EXTRACT")


# Add the parent directory to the module search path to import required modules
parent_dir = str(Path(__file__).absolute().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# Load environment variables from .env file
load_dotenv(override=True)

openai_model = os.getenv("OPENAI_MODEL", "o4-mini")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_URL")  # For custom OpenAI-compatible APIs

# Store Neo4j connection globally
neo4j_retriever = None
neo4j_uri = os.getenv("NEO4J_URL", "bolt://localhost:7687")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "password")


class GraphRAGProcessor:
    """Processor class for building GraphRAG from text chunks"""

    # Define constants at class level
    DROP_INDEX_QUERY = "DROP INDEX entities IF EXISTS;"
    HYBRID_SEARCH_INDEX_DROP_QUERY = "DROP INDEX keyword IF EXISTS;"
    LABELS_QUERY = "CALL db.labels()"
    FILTER_LABELS = ["Chunk", "Document"]

    def __init__(self):
        """Initialize the processor with required components"""
        # Connect to Neo4j
        self.graph_db = Neo4jGraphDB(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            embedding_model_name="text-embedding-3-small",
        )
        logger.info("Connected to Neo4j database")

        # Prepare LLM configuration
        llm_config = {
            "model": openai_model,
        }
        if openai_api_key:
            llm_config["api_key"] = openai_api_key
        if openai_base_url:
            llm_config["base_url"] = openai_base_url

        # Initialize LLM for graph transformation
        self.llm = ChatOpenAITool(**llm_config)
        logger.info(
            f"Initialized LLM with model: {openai_model} and base_url: {openai_base_url or 'default'}"
        )

        # Initialize Graph Transformer
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[],
            node_properties=False,
            relationship_properties=False,
            ignore_tool_usage=True,  # Use prompt-driven approach
        )
        logger.info("Initialized LLMGraphTransformer")

        # Track processed documents
        self.cleaned_graph_documents_list = []

    def handle_backticks_nodes_relationship_id_type(
        self, graph_document_list: List[GraphDocument]
    ):
        """Clean backticks and empty values from nodes and relationships"""
        logger.debug("Cleaning nodes and relationships")
        for graph_document in graph_document_list:
            # Clean node id and types
            cleaned_nodes = []
            for node in graph_document.nodes:
                if node.type.strip() and node.id.strip():
                    node.type = node.type.replace("`", "")
                    cleaned_nodes.append(node)

            # Clean relationship id types and source/target node id and types
            cleaned_relationships = []
            for rel in graph_document.relationships:
                if (
                    rel.type.strip()
                    and rel.source.id.strip()
                    and rel.source.type.strip()
                    and rel.target.id.strip()
                    and rel.target.type.strip()
                ):
                    rel.type = rel.type.replace("`", "")
                    rel.source.type = rel.source.type.replace("`", "")
                    rel.target.type = rel.target.type.replace("`", "")
                    cleaned_relationships.append(rel)

            graph_document.relationships = cleaned_relationships
            graph_document.nodes = cleaned_nodes

        return graph_document_list

    def create_relation_between_chunks(self, file_name: str) -> list:
        """Create relationships between chunks and build a connected graph structure"""
        logger.info(
            f"Creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks for file: {file_name}"
        )

        # Sort chunks by index to ensure proper sequential processing
        self.cleaned_graph_documents_list = sorted(
            self.cleaned_graph_documents_list,
            key=lambda doc: doc.source.metadata.get("chunkIdx", 0),
        )

        current_chunk_id = 0
        lst_chunks_including_hash = []
        batch_data = []
        relationships = []
        offset = 0

        for i, chunk in enumerate(self.cleaned_graph_documents_list):
            # Generate unique ID for each chunk
            page_content_sha1 = hashlib.sha1(chunk.source.page_content.encode())
            previous_chunk_id = current_chunk_id
            current_chunk_id = page_content_sha1.hexdigest()
            position = i + 1

            if i > 0:
                offset += len(
                    self.cleaned_graph_documents_list[i - 1].source.page_content
                )

            firstChunk = True if i == 0 else False

            # Update metadata
            metadata = {
                "position": position,
                "length": len(chunk.source.page_content),
                "content_offset": offset,
                "hash": current_chunk_id,
                "fileName": file_name,  # Add file name to chunk metadata
            }
            self.cleaned_graph_documents_list[i].source.metadata.update(metadata)

            # Create document with updated metadata
            chunk_document = Document(
                page_content=chunk.source.page_content, metadata=metadata
            )

            # Prepare chunk data for Neo4j
            chunk_data = {
                "id": current_chunk_id,
                "pg_content": chunk_document.page_content,
                "position": position,
                "length": chunk_document.metadata["length"],
                "f_name": file_name,  # Use the provided file name
                "previous_id": previous_chunk_id,
                "content_offset": offset,
            }

            # ===== Add timestamp metadata if available ===== #
            if (
                "start_pts" in chunk.source.metadata
                and "end_pts" in chunk.source.metadata
            ):
                chunk_data["start_time"] = (
                    chunk.source.metadata["start_pts"] / 1_000_000_000
                )
                chunk_data["end_time"] = (
                    chunk.source.metadata["end_pts"] / 1_000_000_000
                )
            elif (
                "start_ntp_float" in chunk.source.metadata
                and "end_ntp_float" in chunk.source.metadata
            ):
                chunk_data["start_time"] = chunk.source.metadata["start_ntp_float"]
                chunk_data["end_time"] = chunk.source.metadata["end_ntp_float"]
            # ===== End of adding timestamp metadata ===== #

            batch_data.append(chunk_data)
            lst_chunks_including_hash.append(
                {"chunk_id": current_chunk_id, "chunk_doc": chunk}
            )

            # Create relationships between chunks
            if firstChunk:
                relationships.append(
                    {"type": "FIRST_CHUNK", "chunk_id": current_chunk_id}
                )
            else:
                relationships.append(
                    {
                        "type": "NEXT_CHUNK",
                        "previous_chunk_id": previous_chunk_id,
                        "current_chunk_id": current_chunk_id,
                    }
                )

        # Create Chunk nodes with all properties
        query_to_create_chunk_and_PART_OF_relation = """
            UNWIND $batch_data AS data
            MERGE (c:Chunk {id: data.id})
            SET c.text = data.pg_content, 
                c.position = data.position, 
                c.length = data.length, 
                c.fileName = data.f_name, 
                c.content_offset = data.content_offset
            WITH data, c
            SET c.start_time = CASE WHEN data.start_time IS NOT NULL THEN data.start_time END,
                c.end_time = CASE WHEN data.end_time IS NOT NULL THEN data.end_time END
            WITH data, c
            MERGE (d:Document {fileName: data.f_name})
            MERGE (c)-[:PART_OF]->(d)
        """
        self.graph_db.graph_db.query(
            query_to_create_chunk_and_PART_OF_relation,
            params={"batch_data": batch_data},
        )

        # Create FIRST_CHUNK relationship
        query_to_create_FIRST_relation = """
            UNWIND $relationships AS relationship
            MATCH (d:Document {fileName: $f_name})
            MATCH (c:Chunk {id: relationship.chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                    MERGE (d)-[:FIRST_CHUNK]->(c))
        """
        self.graph_db.graph_db.query(
            query_to_create_FIRST_relation,
            params={"f_name": file_name, "relationships": relationships},
        )

        # Create NEXT_CHUNK relationships connecting chunks in sequence
        query_to_create_NEXT_CHUNK_relation = """
            UNWIND $relationships AS relationship
            MATCH (c:Chunk {id: relationship.current_chunk_id})
            WITH c, relationship
            MATCH (pc:Chunk {id: relationship.previous_chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                    MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
        self.graph_db.graph_db.query(
            query_to_create_NEXT_CHUNK_relation,
            params={"relationships": relationships},
        )

        return lst_chunks_including_hash

    def merge_relationship_between_chunk_and_entities(self, file_name):
        """Create HAS_ENTITY relationships between chunks and extracted entities"""
        logger.debug("Creating HAS_ENTITY relationships between chunks and entities")

        batch_data = []

        # Prepare data for creating relationships
        for graph_doc in self.cleaned_graph_documents_list:
            for node in graph_doc.nodes:
                query_data = {
                    "hash": graph_doc.source.metadata["hash"],
                    "node_type": node.type,
                    "node_id": node.id,
                    "file_name": file_name,  # Include file name to maintain document boundaries
                }
                batch_data.append(query_data)

        if batch_data:
            # Create relationships in Neo4j - now with file name constraint
            unwind_query = """
                UNWIND $batch_data AS data
                MATCH (c:Chunk {id: data.hash})
                MATCH (d:Document {fileName: data.file_name})
                WHERE (c)-[:PART_OF]->(d)
                CALL apoc.merge.node([data.node_type], {id: data.node_id, fileName: data.file_name}) YIELD node AS n
                MERGE (c)-[:HAS_ENTITY]->(n)
            """
            self.graph_db.graph_db.query(
                unwind_query, params={"batch_data": batch_data}
            )

    def create_fulltext(self):
        logger.info("Starting the process of creating full-text indexes.")
        try:
            drop_query = self.DROP_INDEX_QUERY

            self.graph_db.graph_db.query(drop_query)
            logger.info("Dropped existing index (if any).")

            result = self.graph_db.graph_db.query(self.LABELS_QUERY)
            labels = [record["label"] for record in result]

            for label in self.FILTER_LABELS:
                if label in labels:
                    labels.remove(label)
            if labels:
                labels_str = ":" + "|".join([f"`{label}`" for label in labels])
                logger.info("Fetched labels.")
            else:
                logger.info("Full text index is not created as labels are empty")
                return

            fulltext_query = f"CREATE FULLTEXT INDEX entities FOR (n{labels_str}) ON EACH [n.id, n.description, n.fileName]"
            self.graph_db.graph_db.query(fulltext_query)
            logger.info("Created full-text index.")
        except Exception as e:
            logger.error(f"An error occurred during the session: {e}")
        logger.info("Full-text index creation process completed.")

    def process_chunks(self, chunks):
        """Process chunks to extract entities and relationships and build graph"""
        logger.info("Starting to process chunks")

        # Get filename from metadata or generate one
        file_name = None
        for chunk in chunks:
            if "file" in chunk.get("metadata", {}):
                file_name = chunk["metadata"]["file"]
                break
            elif "fileName" in chunk.get("metadata", {}):
                file_name = chunk["metadata"]["fileName"]
                break
            elif "video_path" in chunk.get("metadata", {}):
                file_name = chunk["metadata"]["video_path"]
                break

        # If no filename found in metadata, create one based on timestamp
        if not file_name:
            file_name = f"document_{int(time.time())}"

        logger.info(f"Processing chunks for file: {file_name}")

        # No longer clearing existing data in Neo4j
        # Instead, we'll just add to the existing graph

        # Convert chunks to documents
        docs = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in chunks
        ]

        # Use the LLM to convert the text to graph structure
        logger.info("Converting documents to graph structure")
        graph_documents = self.transformer.convert_to_graph_documents(docs)
        time.sleep(1)  # Use time.sleep for safe interval between requests

        # Clean the graph documents
        cleaned_graph_documents = self.handle_backticks_nodes_relationship_id_type(
            graph_documents
        )
        # Store the cleaned documents
        self.cleaned_graph_documents_list = cleaned_graph_documents

        # Add graph documents to Neo4j
        logger.info("Adding graph documents to Neo4j")
        self.graph_db.add_graph_documents_with_filename(
            cleaned_graph_documents, file_name, baseEntityLabel=True
        )

        # Create document node with real filename
        params = {"fileName": file_name}
        query = "MERGE(d:Document {fileName: $props.fileName}) SET d += $props"
        param = {"props": params}
        self.graph_db.graph_db.query(query, param)

        # Build relationships between chunks
        chunk_list = self.create_relation_between_chunks(file_name)

        # Connect chunks to entities
        self.merge_relationship_between_chunk_and_entities(file_name)

        self.create_fulltext()  # create_vector_fulltext_indexes)

        logger.info("Finished processing chunks")

        return True

    def initialize_knowledge_graph(self):
        """
        Reset and initialize the knowledge graph by clearing all existing data.
        Use this method when you want to start with a fresh graph instead of incremental building.

        Returns:
            Status information about the initialization
        """
        logger.info("Initializing knowledge graph by clearing all existing data")

        try:
            # Clear all data from Neo4j
            self.graph_db.graph_db.query("MATCH (n) DETACH DELETE n")

            # Recreate full-text indexes
            self.create_fulltext()  # create_vector_fulltext_indexes()

            logger.info(
                "Knowledge graph initialized successfully. All existing data has been cleared."
            )

            return True
        except Exception as e:
            logger.error(f"Error initializing knowledge graph: {str(e)}", exc_info=True)
            return False


def process_sample_chunks() -> Dict[str, Any]:
    """
    Process the predefined sample chunks to extract entities and build a knowledge graph.

    Returns:
        A dictionary with summary information about the extracted entities and relationships
    """
    logger.info("(GraphRAGExtract) Processing sample chunks")

    # Sample content with timestamps to mimic what would come from VLM
    SAMPLE_CHUNKS_1 = [
        {
            "text": "<0.0> <10.0> A person wearing a red jacket is walking through the warehouse carrying a box. They place it on a shelf next to several other boxes.",
            "id": 0,
            "metadata": {
                "start_pts": 0,
                "end_pts": 10000000000,  # 10 seconds in nanoseconds
                "file": "sample_warehouse.mp4",
                "is_first": True,
                "is_last": False,
                "request_id": "test-request-id",
                "chunkIdx": 0,
            },
        },
        {
            "text": "<10.0> <20.0> A forklift is moving pallets from one area to another. The operator is wearing a yellow helmet and seems to be carefully navigating around other workers.",
            "id": 1,
            "metadata": {
                "start_pts": 10000000000,
                "end_pts": 20000000000,
                "file": "sample_warehouse.mp4",
                "is_first": False,
                "is_last": False,
                "request_id": "test-request-id",
                "chunkIdx": 1,
            },
        },
        {
            "text": "<20.0> <30.0> One worker is inspecting items on a conveyor belt. They appear to be checking for defects and occasionally removing items from the belt.",
            "id": 2,
            "metadata": {
                "start_pts": 20000000000,
                "end_pts": 30000000000,
                "file": "sample_warehouse.mp4",
                "is_first": False,
                "is_last": True,
                "request_id": "test-request-id",
                "chunkIdx": 2,
            },
        },
        {
            "text": "<30.0> <40.0> Suddenly, the forklift hit a stack of boxes, causing them to topple over onto the person in the red jacket. The worker quickly moves out of the way to avoid injury.",
            "id": 3,
            "metadata": {
                "start_pts": 30000000000,
                "end_pts": 40000000000,
                "file": "sample_warehouse.mp4",
                "is_first": True,
                "is_last": False,
                "request_id": "incremental-update",
                "chunkIdx": 0,
            },
        },
        {
            "text": "<40.0> <50.0> The forklift operator quickly gets out of the vehicle to check on the person in the red jacket. They both appear shaken but unharmed.",
            "id": 4,
            "metadata": {
                "start_pts": 40000000000,
                "end_pts": 50000000000,
                "file": "sample_warehouse.mp4",
                "is_first": False,
                "is_last": False,
                "request_id": "incremental-update",
                "chunkIdx": 1,
            },
        },
    ]

    SAMPLE_CHUNKS_3 = [
        {
            "text": "<0.0> <10.0> A person wearing a red jacket is walking through the second warehouse carrying a box. They place it on a shelf next to several other boxes.",
            "id": 0,
            "metadata": {
                "start_pts": 0,
                "end_pts": 10000000000,  # 10 seconds in nanoseconds
                "file": "sample_warehouse2.mp4",
                "is_first": True,
                "is_last": False,
                "request_id": "test-request-id",
                "chunkIdx": 0,
            },
        },
        {
            "text": "<10.0> <20.0> A forklift  is moving pallets from one area to another in the second warehouse. The operator is wearing a yellow helmet and seems to be carefully navigating around other workers.",
            "id": 1,
            "metadata": {
                "start_pts": 10000000000,
                "end_pts": 20000000000,
                "file": "sample_warehouse2.mp4",
                "is_first": False,
                "is_last": False,
                "request_id": "test-request-id",
                "chunkIdx": 1,
            },
        },
        {
            "text": "<20.0> <30.0> One worker is inspecting items on a conveyor belt in the second warehouse. They appear to be checking for defects and occasionally removing items from the belt.",
            "id": 2,
            "metadata": {
                "start_pts": 20000000000,
                "end_pts": 30000000000,
                "file": "sample_warehouse2.mp4",
                "is_first": False,
                "is_last": True,
                "request_id": "test-request-id",
                "chunkIdx": 2,
            },
        },
    ]

    processor = GraphRAGProcessor()

    result = processor.process_chunks(SAMPLE_CHUNKS_1)
    if not result:
        logger.error("Failed to process initial sample chunks 1")
        return {}

    result = processor.process_chunks(SAMPLE_CHUNKS_3)
    if not result:
        logger.error("Failed to process incremental sample chunks 3")
        return {}
    return result


async def initialize_graph_db() -> Dict[str, Any]:
    """
    Initializes the knowledge graph by clearing all existing data from the Neo4j database.
    Use this when you want to start with a fresh graph instead of incremental building.
    For example, when user asks to initialize, create, reset, or prepare the knowledge graph explicitly, run this tool.

    Returns:
        A dictionary with information about the initialization process
    """
    logger.info("(GraphRAGExtract) Initializing knowledge graph")
    processor = GraphRAGProcessor()
    result = processor.initialize_knowledge_graph()
    return result


def main():
    ret = asyncio.run(initialize_graph_db())
    if not ret:
        logger.error("Failed to initialize knowledge graph")
        return
    print("Initialized knowledge graph")
    ret = process_sample_chunks()
    if not ret:
        logger.error("Failed to process sample chunks")
        return
    print("Processed sample chunks")


if __name__ == "__main__":
    main()
