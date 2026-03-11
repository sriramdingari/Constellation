"""Cypher query templates for the Constellation code knowledge graph.

All queries use plain MERGE — no APOC dependency.
"""


def upsert_entities_query(label: str) -> str:
    """Generate a Cypher query to upsert entities of a given node label.

    One query is generated per entity type so that labels can be statically
    embedded in the Cypher text (Neo4j does not support dynamic labels in
    MERGE without APOC).
    """
    return f"""
    UNWIND $entities AS entity
    MERGE (n:{label} {{id: entity.id}})
    SET n += entity.properties
    RETURN count(n) AS count
    """


def create_relationships_query(rel_type: str) -> str:
    """Generate a Cypher query to create/merge relationships of a given type.

    One query is generated per relationship type so that the type can be
    statically embedded in the Cypher text.
    """
    return f"""
    UNWIND $relationships AS rel
    MATCH (source {{id: rel.source_id}})
    MATCH (target {{id: rel.target_id}})
    MERGE (source)-[r:{rel_type}]->(target)
    SET r += rel.properties
    RETURN count(r) AS count
    """


UPSERT_REPOSITORY = """
MERGE (r:Repository {name: $name})
SET r.source = $source,
    r.last_indexed_at = datetime(),
    r.last_commit_sha = $commit_sha,
    r.entity_count = $entity_count
RETURN r
"""

GET_REPOSITORY = """
MATCH (r:Repository {name: $name})
RETURN r
"""

LIST_REPOSITORIES = """
MATCH (r:Repository)
RETURN r
ORDER BY r.name
"""

DELETE_REPOSITORY = """
MATCH (r:Repository {name: $name})
OPTIONAL MATCH (r)-[*]->(n)
DETACH DELETE n, r
"""

GET_FILE_HASHES = """
MATCH (f:File {repository: $repository})
RETURN f.file_path AS file_path, f.content_hash AS content_hash
"""

DELETE_STALE_FILES = """
UNWIND $file_paths AS path
MATCH (f:File {repository: $repository, file_path: path})
OPTIONAL MATCH (f)-[*]->(n)
DETACH DELETE n, f
"""

CREATE_VECTOR_INDEX = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS
FOR (n:{label})
ON (n.embedding)
OPTIONS {{indexConfig: {{
    `vector.dimensions`: $dimensions,
    `vector.similarity_function`: 'cosine'
}}}}
"""
