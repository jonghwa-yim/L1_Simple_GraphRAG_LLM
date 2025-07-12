#!/usr/bin/env python3
"""
GraphRAG Retrieval Server - Knowledge Graph-based Retrieval Augmented Generation

"""

import logging
import os

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

from src.llm import ChatOpenAITool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("G-RAG-RETRIEVAL")

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

# Store chat history and node types globally with thread safety
# detected_node_types: List[str] = []
# detected_rel_types: List[str] = []
# detected_nr_updated = True
# history_lock = Lock()


# def update_node_types():
#     """Update the global node types by querying the database"""
#     global detected_node_types, detected_nr_updated
#     try:
#         retriever = create_retriever()
#         # Query to get all unique node labels except Chunk and Document
#         result = retriever.query("""
#             CALL db.labels() YIELD label
#             WHERE label <> 'Chunk' AND label <> 'Document' AND label <> '__Entity__'
#             RETURN collect(label) as labels
#         """)
#         if result and len(result) > 0:
#             detected_node_types = result[0].get('labels', [])
#     except Exception as e:
#         print(f"Error updating node types: {str(e)}")
#     finally:
#         detected_nr_updated = True
#         logger.info(f"Updated Node: {detected_node_types} ... nr_update: {detected_nr_updated}")
#         print(f"Updated Node: {detected_node_types} ... nr_update: {detected_nr_updated}")


# def update_relation_types():
#     """Update the global relationship types by querying the database"""
#     global detected_rel_types, detected_nr_updated
#     try:
#         retriever = create_retriever()
#         # Query to get all unique node labels except Chunk and Document
#         result = retriever.query("""
#             MATCH ()-[r]-()
#             WHERE type(r) <> 'PART_OF'
#             AND type(r) <> 'FIRST_CHUNK'
#             AND type(r) <> 'NEXT_CHUNK'
#             AND type(r) <> 'HAS_ENTITY'
#             RETURN DISTINCT type(r) AS relationshipType
#         """)
#         if result and len(result) > 0:
#             detected_rel_types = [res.get('relationshipType', []) for res in result]
#     except Exception as e:
#         print(f"Error updating relationship types: {str(e)}")
#     finally:
#         detected_nr_updated = True
#         logger.info(f"Updated Rel: {detected_rel_types} ... nr_update: {detected_nr_updated}")
#         print(f"Updated Rel: {detected_rel_types} ... nr_update: {detected_nr_updated}")


class GraphRAGRetriever:
    def __init__(self):
        # Connector to Neo4j
        self.graph_db_conn = Neo4jGraph(
            url=neo4j_uri, username=neo4j_username, password=neo4j_password
        )

        # Prepare LLM configuration
        llm_config = {
            "model": openai_model,
        }
        if openai_api_key:
            llm_config["api_key"] = openai_api_key
        if openai_base_url:
            llm_config["base_url"] = openai_base_url
        self.llm = ChatOpenAITool(**llm_config)
        logger.info("Initialized GraphRAGRetriever")

    def update_relation_types(self):
        """Update the global relationship types by querying the database"""
        try:
            # Query to get all unique node labels except Chunk and Document
            result = self.graph_db_conn.query("""
                MATCH ()-[r]-()
                WHERE type(r) <> 'PART_OF' 
                AND type(r) <> 'FIRST_CHUNK' 
                AND type(r) <> 'NEXT_CHUNK' 
                AND type(r) <> 'HAS_ENTITY'
                RETURN DISTINCT type(r) AS relationshipType
            """)
            if result and len(result) > 0:
                detected_rel_types = [res.get("relationshipType", []) for res in result]
        except Exception as e:
            logger.info(f"Error updating relationship types: {str(e)}")
        finally:
            self.rel_types = detected_rel_types
            logger.info(f"Updated Rel: {detected_rel_types}")

    def update_node_types(self):
        """Update the global node types by querying the database"""
        try:
            # Query to get all unique node labels except Chunk and Document
            result = self.graph_db_conn.query("""
                CALL db.labels() YIELD label
                WHERE label <> 'Chunk' AND label <> 'Document' AND label <> '__Entity__'
                RETURN collect(label) as labels
            """)
            if result and len(result) > 0:
                detected_node_types = result[0].get("labels", [])
        except Exception as e:
            logger.info(f"Error updating node types: {str(e)}")
        finally:
            self.node_types = detected_node_types
            logger.info(f"Updated Node: {detected_node_types}")

    def natural_language_to_cypher(self, nl_query: str) -> str:
        """
        Convert a natural language query to a Cypher query using LLM
        """

        prompt = f"""You are an expert to generate an answer using the knowledge that can be retrieved from neo4j DB for user's question. In neo4j, graph rag and vector store are formed. Try to retrieve proper information by interpreting user's question into proper cypher query. Then, generate natural language answer from the knowledges. If result is empty three times of calling query_neo4j, retrieve all chunk text in neo4j.

Current Graph Schema: Node types: {", ".join(self.node_types)}. Relationship types: {", ".join(self.rel_types)}. Careful about adding -S or -ES or using capital or small letter for node and relationship types.

In neo4j database, nouns are stored in 'id' of the entity. Nodes in neo4j have labels containing the string label. Note that '<id>' (identity in database) is the integer id
=== Entity nodes of Neo4j database and its data format: 
{{"identity": int, "labels": [str, "__Entity__" ], "properties": {{"id": str, "embedding": float array, "fileName": str}}, "elementID": str}}

labels(entity) = [str, "__Entity__"] and entity.id = str.

RELATIONSHIPS:
- (Chunk)-[:PART_OF]->(Document): Chunks belong to documents
- (Chunk)-[:HAS_ENTITY]->(Entity): Chunks contain entities
- (Entity)-[r]->(Entity): type(r) is various relationships between entities
- (Chunk)-[:NEXT_CHUNK]->(Chunk): Sequential ordering of chunks
- (Document)-[:FIRST_CHUNK]->(Chunk): Points to the first chunk in document
- (Chunk)-[:SIMILAR]-(Chunk): Connects semantically similar chunks

SEARCH CAPABILITIES:
- Fulltext search on entities: Use db.index.fulltext.queryNodes('entities', search_term)
- Fulltext search on chunks: Use db.index.fulltext.queryNodes('keyword', search_term)
- When performing fulltext search, use infinitive form of the verb

TIPS:
- Step-by-step think: From the natural language query, extract verbs and find matching relationship in the Relationship types
- Always contain fileName in the return value
- When searching for entities, use Node labels and return the id of the entities
- For time-based queries, use chunk.start_time and chunk.end_time
- All nodes have embeddings for similarity search
- Unless a specific file is mentioned, search across all files
- When looking for activities, always search Relationship.
- When looking for activities or relationships, also retrieve connected entities.

Translate the following natural language query into a Cypher query:

Query: {nl_query}

Respond with ONLY the Cypher query without explanation or additional text."""

        # Get the response from the LLM
        response = self.llm.invoke(prompt)  # o4-mini does not support temperature

        # Extract just the Cypher query from the response
        cypher_query = response.content

        return cypher_query

    def execute_query(self, cypher_query) -> str:
        print("Executing query...")
        try:
            # Execute Cypher query to retrieve documents
            result = self.graph_db_conn.query(cypher_query)
        except Exception as e:
            error_message = f"Error executing Cypher query: {str(e)}"
            return error_message

        # Format and return the result as a string
        if isinstance(result, dict) and "data" in result:
            # Handle standard Neo4j response format
            return str(result["data"])
        else:
            # Return the raw result as a string
            return str(result)


def main():
    rag_retriever = GraphRAGRetriever()

    # Example Cypher queries to demonstrate different types of retrieval
    EXAMPLE_QUERIES = [
        # 1. Basic query to retrieve all entities of a specific type
        {
            "description": "Retrieve all Person entities",
            "query": "MATCH (p:Person) RETURN p.id, p.description",
        },
        # 2. Find relationships between entities
        {
            "description": "Find all relationships between workers and equipment",
            "query": "MATCH (p:Person)-[r]->(e) WHERE e:Equipment OR e:Tool RETURN p.id, type(r), e.id",
        },
        # 3. Time-based query for video chunks
        {
            "description": "Find events that happened in the first 20 seconds",
            "query": "MATCH (c:Chunk) WHERE c.start_time <= 20.0 RETURN c.text ORDER BY c.start_time",
        },
        # 4. Query for specific actions or events
        {
            "description": "Find all chunks where someone is carrying something",
            "query": "MATCH (c:Chunk)-[:HAS_ENTITY]->(a:Action)-[:PERFORMED_BY]->(p:Person) WHERE a.id CONTAINS 'carry' RETURN c.text",
        },
        # 5. Query using vector similarity (if implemented)
        {
            "description": "Find chunks semantically similar to 'safety procedures'",
            "query": """
            WITH "safety procedures" AS query
            CALL db.index.vector.queryNodes('vector', 3, apoc.text.encode(query), {})
            YIELD node, score
            WHERE score > 0.7
            RETURN node.text, score
            ORDER BY score DESC
            """,
        },
        # 6. Complex query combining multiple entity types and relationships
        {
            "description": "Find interactions between workers and equipment in the warehouse",
            "query": """
            MATCH (p:Person)-[r1]->(a:Action)-[r2]->(e)
            WHERE (e:Equipment OR e:Location) AND e.id CONTAINS 'warehouse'
            RETURN p.id, a.id, e.id, r1, r2
            """,
        },
    ]

    rag_retriever.update_node_types()
    rag_retriever.update_relation_types()

    # Example natural language queries and their expected Cypher translations
    EXAMPLE_NL_QUERIES = [
        "Who are all the people in the video?",
        "What equipment is being used?",
        "What happened in the video?",
        "Show me all instances where someone is carrying something",
        "Was there any accident such as topping over, crash, hit, fall in the video?",
    ]
    for i, example in enumerate(EXAMPLE_NL_QUERIES):
        cypher_query = rag_retriever.natural_language_to_cypher(example)
        result = rag_retriever.execute_query(cypher_query)
        print(f"===== Test {i + 1} =====\nNatural Language Query: {example}")
        print(f"Cypher Query: {cypher_query}")
        print(f"Result: {result}\n")

    return


if __name__ == "__main__":
    main()
