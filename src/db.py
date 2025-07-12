"""
Minimal implementation of the neo4j_db module to support GraphRAG
"""

from hashlib import md5
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core._api.deprecation import deprecated
from langchain_openai import OpenAIEmbeddings

BASE_ENTITY_LABEL = "__Entity__"

include_docs_query = (
    "MERGE (d:Document {id:$document.metadata.id}) "
    "SET d.text = $document.page_content "
    "SET d += $document.metadata "
    "WITH d "
)


def _get_node_import_query_with_filename(
    baseEntityLabel: bool, include_source: bool
) -> str:
    if baseEntityLabel:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.id, fileName: $fileName}}) "
            "SET source += row.properties "
            f"{'MERGE (d)-[:MENTIONS]->(source) ' if include_source else ''}"
            "WITH source, row "
            "CALL apoc.create.addLabels( source, [row.type] ) YIELD node "
            "RETURN distinct 'done' AS result"
        )
    else:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.type], {id: row.id, fileName: $fileName}, "
            "row.properties, {}) YIELD node "
            f"{'MERGE (d)-[:MENTIONS]->(node) ' if include_source else ''}"
            "RETURN distinct 'done' AS result"
        )


def _get_rel_import_query_with_filename(baseEntityLabel: bool) -> str:
    if baseEntityLabel:
        return (
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.source, fileName: $fileName}}) "
            f"MERGE (target:`{BASE_ENTITY_LABEL}` {{id: row.target, fileName: $fileName}}) "
            "WITH source, target, row "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )
    else:
        return (
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.source_label], {id: row.source, fileName: $fileName},"
            "{}, {}) YIELD node as source "
            "CALL apoc.merge.node([row.target_label], {id: row.target, fileName: $fileName},"
            "{}, {}) YIELD node as target "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )


@deprecated(
    since="0.3.8",
    removal="1.0",
    alternative_import="langchain_neo4j.graphs.neo4j_graph._remove_backticks",
)
def _remove_backticks(text: str) -> str:
    return text.replace("`", "")


class Neo4jGraphDB:
    """Neo4j graph database wrapper for GraphRAG"""

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        name="neo4j_db",
        embedding_model_name="text-embedding-3-small",
    ) -> None:
        """Initialize connection to Neo4j and set up embeddings"""

        # Initialize Neo4j connection
        self.graph_db = Neo4jGraph(
            url=url,
            username=username,
            password=password,
            sanitize=True,
            refresh_schema=False,
        )

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
        )

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "\n-", ".", ";", ",", " ", ""],
        )

    def add_graph_documents_with_filename(
        self,
        graph_documents: List[GraphDocument],
        file_name: str,
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        """
        Add graph documents with filename-based node separation.

        Args:
            graph_documents: List of GraphDocument objects
            file_name: The filename to associate with all nodes/relationships
            include_source: If True, stores the source document
            baseEntityLabel: If True, adds __Entity__ label to nodes
        """
        if baseEntityLabel:  # Check if constraint already exists
            constraint_exists = any(
                [
                    el["labelsOrTypes"] == [BASE_ENTITY_LABEL]
                    and set(el["properties"])
                    == {"id", "fileName"}  # NEW constraint: id + fileName
                    for el in self.graph_db.structured_schema.get("metadata", {}).get(
                        "constraint", []
                    )
                ]
            )

            if not constraint_exists:
                # Create constraint - need to include fileName for separation
                self.graph_db.query(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (b:{BASE_ENTITY_LABEL}) "
                    "REQUIRE (b.id, b.fileName) IS UNIQUE;"
                )
                self.graph_db.refresh_schema()

        node_import_query = _get_node_import_query_with_filename(
            baseEntityLabel, include_source
        )
        rel_import_query = _get_rel_import_query_with_filename(baseEntityLabel)

        for document in graph_documents:
            if not document.source.metadata.get("id"):
                document.source.metadata["id"] = md5(
                    document.source.page_content.encode("utf-8")
                ).hexdigest()

            # Remove backticks from node types
            for node in document.nodes:
                node.type = _remove_backticks(node.type)
                # Add fileName to node properties
                if not node.properties:
                    node.properties = {}
                node.properties["fileName"] = file_name

            # Import nodes
            self.graph_db.query(
                node_import_query,
                {
                    "data": [el.__dict__ for el in document.nodes],
                    "document": document.source.__dict__,
                    "fileName": file_name,
                },
            )

            # Import relationships
            self.graph_db.query(
                rel_import_query,
                {
                    "data": [
                        {
                            "source": el.source.id,
                            "source_label": _remove_backticks(el.source.type),
                            "target": el.target.id,
                            "target_label": _remove_backticks(el.target.type),
                            "type": _remove_backticks(
                                el.type.replace(" ", "_").upper()
                            ),
                            "properties": el.properties,
                        }
                        for el in document.relationships
                    ],
                    "fileName": file_name,
                },
            )
