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
    ON CREATE SET n.__created__ = true
    WITH n, entity, coalesce(n.__created__, false) AS created
    SET n = entity.properties
    RETURN sum(CASE WHEN created THEN 1 ELSE 0 END) AS count
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
    ON CREATE SET r.__created__ = true
    WITH r, rel, coalesce(r.__created__, false) AS created
    SET r = rel.properties
    RETURN sum(CASE WHEN created THEN 1 ELSE 0 END) AS count
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
OPTIONAL MATCH (n {repository: $name})
WITH r, collect(n) AS nodes
FOREACH (node IN nodes | DETACH DELETE node)
DETACH DELETE r
"""

GET_FILE_HASHES = """
MATCH (f:File {repository: $repository})
RETURN f.file_path AS file_path, f.content_hash AS content_hash
"""

GET_FILE_ENTITY_SNAPSHOT = """
MATCH (n {repository: $repository, file_path: $file_path})
RETURN n.id AS id, labels(n) AS labels
"""

COUNT_REPOSITORY_ENTITIES = """
MATCH (n {repository: $repository})
RETURN count(n) AS count
"""

DELETE_FILE_OUTBOUND_RELATIONSHIPS = """
MATCH (n {repository: $repository, file_path: $file_path})
WHERE NOT n:Package
OPTIONAL MATCH (n)-[r]->()
WITH collect(r) AS relationships
FOREACH (relationship IN relationships | DELETE relationship)
"""

DELETE_ENTITIES_BY_IDS = """
UNWIND $entity_ids AS entity_id
MATCH (n {repository: $repository, id: entity_id})
WHERE NOT n:Package
DETACH DELETE n
"""

DELETE_STALE_FILES = """
UNWIND $file_paths AS path
MATCH (n {repository: $repository, file_path: path})
WHERE NOT n:Package
DETACH DELETE n
"""

DELETE_ORPHAN_PACKAGES = """
MATCH (p:Package {repository: $repository})
WHERE NOT ()-->(p)
AND NOT EXISTS {
    MATCH (child:Package {repository: $repository})
    WHERE child.id STARTS WITH p.id + '.'
}
WITH collect(p) AS packages
FOREACH (package IN packages | DETACH DELETE package)
RETURN size(packages) AS count
"""

DELETE_ORPHAN_REFERENCES = """
MATCH (r:Reference {repository: $repository})
WHERE NOT ()-->(r)
WITH collect(r) AS references
FOREACH (reference IN references | DETACH DELETE reference)
RETURN size(references) AS count
"""

GET_VECTOR_INDEX = """
SHOW INDEXES YIELD name, type, options
WHERE type = 'VECTOR' AND name = $index_name
RETURN options
"""


def drop_index_query(index_name: str) -> str:
    """Generate a Cypher query to drop an index by name if it exists."""
    return f"DROP INDEX {index_name} IF EXISTS"


CREATE_VECTOR_INDEX = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS
FOR (n:{label})
ON (n.embedding)
OPTIONS {{indexConfig: {{
    `vector.dimensions`: $dimensions,
    `vector.similarity_function`: 'cosine'
}}}}
"""
