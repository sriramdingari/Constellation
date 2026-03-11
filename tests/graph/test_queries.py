"""Tests for constellation.graph.queries — Cypher query templates."""

from constellation.graph.queries import (
    upsert_entities_query,
    create_relationships_query,
    UPSERT_REPOSITORY,
    GET_REPOSITORY,
    LIST_REPOSITORIES,
    DELETE_REPOSITORY,
    GET_FILE_HASHES,
    GET_FILE_ENTITY_SNAPSHOT,
    COUNT_REPOSITORY_ENTITIES,
    DELETE_FILE_OUTBOUND_RELATIONSHIPS,
    DELETE_ENTITIES_BY_IDS,
    DELETE_STALE_FILES,
    DELETE_ORPHAN_PACKAGES,
    DELETE_ORPHAN_REFERENCES,
    GET_VECTOR_INDEX,
    CREATE_VECTOR_INDEX,
    drop_index_query,
)


class TestUpsertEntitiesQuery:
    """Validate the per-label entity upsert query generator."""

    def test_returns_string(self):
        result = upsert_entities_query("Class")
        assert isinstance(result, str)

    def test_contains_unwind(self):
        result = upsert_entities_query("Method")
        assert "UNWIND" in result

    def test_contains_merge_with_label(self):
        result = upsert_entities_query("Class")
        assert "MERGE" in result
        assert ":Class" in result

    def test_merges_on_id(self):
        result = upsert_entities_query("Interface")
        assert "id:" in result or "id :" in result

    def test_sets_properties(self):
        result = upsert_entities_query("Field")
        assert "SET" in result

    def test_counts_only_new_nodes(self):
        result = upsert_entities_query("Constructor")
        assert "ON CREATE SET" in result
        assert "__created__" in result
        assert "WITH n, entity, coalesce(n.__created__, false) AS created" in result
        assert "SET n = entity.properties" in result

    def test_returns_count(self):
        result = upsert_entities_query("Constructor")
        assert "count" in result.lower()

    def test_different_labels_produce_different_queries(self):
        q1 = upsert_entities_query("Class")
        q2 = upsert_entities_query("Method")
        assert q1 != q2
        assert ":Class" in q1
        assert ":Method" in q2

    def test_no_apoc(self):
        result = upsert_entities_query("Class")
        assert "apoc" not in result.lower()


class TestCreateRelationshipsQuery:
    """Validate the per-type relationship creation query generator."""

    def test_returns_string(self):
        result = create_relationships_query("CALLS")
        assert isinstance(result, str)

    def test_contains_correct_relationship_type(self):
        result = create_relationships_query("EXTENDS")
        assert ":EXTENDS" in result

    def test_contains_unwind(self):
        result = create_relationships_query("IMPLEMENTS")
        assert "UNWIND" in result

    def test_matches_source_and_target(self):
        result = create_relationships_query("CALLS")
        assert "source" in result.lower()
        assert "target" in result.lower()

    def test_uses_merge_not_create(self):
        result = create_relationships_query("CALLS")
        assert "MERGE" in result

    def test_sets_properties(self):
        result = create_relationships_query("USES_TYPE")
        assert "SET" in result

    def test_counts_only_new_relationships(self):
        result = create_relationships_query("IMPORTS")
        assert "ON CREATE SET" in result
        assert "__created__" in result
        assert "WITH r, rel, coalesce(r.__created__, false) AS created" in result
        assert "SET r = rel.properties" in result

    def test_returns_count(self):
        result = create_relationships_query("IMPORTS")
        assert "count" in result.lower()

    def test_different_types_produce_different_queries(self):
        q1 = create_relationships_query("CALLS")
        q2 = create_relationships_query("EXTENDS")
        assert q1 != q2
        assert ":CALLS" in q1
        assert ":EXTENDS" in q2

    def test_no_apoc(self):
        result = create_relationships_query("CALLS")
        assert "apoc" not in result.lower()


class TestUpsertRepository:
    """Validate the UPSERT_REPOSITORY query."""

    def test_contains_merge(self):
        assert "MERGE" in UPSERT_REPOSITORY

    def test_targets_repository_label(self):
        assert ":Repository" in UPSERT_REPOSITORY

    def test_uses_name_parameter(self):
        assert "$name" in UPSERT_REPOSITORY

    def test_sets_last_indexed_at(self):
        assert "last_indexed_at" in UPSERT_REPOSITORY

    def test_returns_result(self):
        assert "RETURN" in UPSERT_REPOSITORY


class TestGetRepository:
    """Validate the GET_REPOSITORY query."""

    def test_contains_match(self):
        assert "MATCH" in GET_REPOSITORY

    def test_targets_repository_label(self):
        assert ":Repository" in GET_REPOSITORY

    def test_uses_name_parameter(self):
        assert "$name" in GET_REPOSITORY

    def test_returns_result(self):
        assert "RETURN" in GET_REPOSITORY


class TestListRepositories:
    """Validate the LIST_REPOSITORIES query."""

    def test_contains_match(self):
        assert "MATCH" in LIST_REPOSITORIES

    def test_has_order_by(self):
        assert "ORDER BY" in LIST_REPOSITORIES

    def test_orders_by_name(self):
        assert "r.name" in LIST_REPOSITORIES


class TestDeleteRepository:
    """Validate the DELETE_REPOSITORY query."""

    def test_uses_detach_delete(self):
        assert "DETACH DELETE" in DELETE_REPOSITORY

    def test_targets_repository_label(self):
        assert ":Repository" in DELETE_REPOSITORY

    def test_uses_name_parameter(self):
        assert "$name" in DELETE_REPOSITORY

    def test_deletes_nodes_by_repository_property(self):
        assert "repository: $name" in DELETE_REPOSITORY


class TestGetFileHashes:
    """Validate the GET_FILE_HASHES query."""

    def test_returns_file_path(self):
        assert "file_path" in GET_FILE_HASHES

    def test_returns_content_hash(self):
        assert "content_hash" in GET_FILE_HASHES

    def test_filters_by_repository(self):
        assert "$repository" in GET_FILE_HASHES

    def test_targets_file_label(self):
        assert ":File" in GET_FILE_HASHES


class TestGetFileEntitySnapshot:
    """Validate the GET_FILE_ENTITY_SNAPSHOT query."""

    def test_returns_ids_and_labels(self):
        assert "RETURN n.id AS id" in GET_FILE_ENTITY_SNAPSHOT
        assert "labels(n) AS labels" in GET_FILE_ENTITY_SNAPSHOT

    def test_filters_by_repository_and_file_path(self):
        assert "$repository" in GET_FILE_ENTITY_SNAPSHOT
        assert "$file_path" in GET_FILE_ENTITY_SNAPSHOT


class TestCountRepositoryEntities:
    """Validate the COUNT_REPOSITORY_ENTITIES query."""

    def test_counts_repository_owned_nodes(self):
        assert "count" in COUNT_REPOSITORY_ENTITIES.lower()
        assert "repository" in COUNT_REPOSITORY_ENTITIES

    def test_filters_by_repository_parameter(self):
        assert "$repository" in COUNT_REPOSITORY_ENTITIES


class TestDeleteStaleFiles:
    """Validate the DELETE_STALE_FILES query."""

    def test_uses_unwind(self):
        assert "UNWIND" in DELETE_STALE_FILES

    def test_uses_detach_delete(self):
        assert "DETACH DELETE" in DELETE_STALE_FILES

    def test_filters_by_repository(self):
        assert "$repository" in DELETE_STALE_FILES


class TestDeleteOrphanReferences:
    """Validate the DELETE_ORPHAN_REFERENCES query."""

    def test_targets_reference_label(self):
        assert ":Reference" in DELETE_ORPHAN_REFERENCES

    def test_filters_by_repository(self):
        assert "$repository" in DELETE_ORPHAN_REFERENCES

    def test_deletes_unreferenced_nodes(self):
        assert "DETACH DELETE" in DELETE_ORPHAN_REFERENCES

    def test_does_not_traverse_arbitrary_descendants(self):
        assert "[*]" not in DELETE_STALE_FILES

    def test_deletes_by_file_path_without_directly_removing_packages(self):
        assert "file_path: path" in DELETE_STALE_FILES
        assert "NOT n:Package" in DELETE_STALE_FILES


class TestDeleteFileOutboundRelationships:
    """Validate the DELETE_FILE_OUTBOUND_RELATIONSHIPS query."""

    def test_deletes_outgoing_relationships_for_one_file(self):
        assert "MATCH (n {repository: $repository, file_path: $file_path})" in (
            DELETE_FILE_OUTBOUND_RELATIONSHIPS
        )
        assert "MATCH (n)-[r]->()" in DELETE_FILE_OUTBOUND_RELATIONSHIPS
        assert "DELETE r" in DELETE_FILE_OUTBOUND_RELATIONSHIPS

    def test_does_not_clear_package_relationships(self):
        assert "NOT n:Package" in DELETE_FILE_OUTBOUND_RELATIONSHIPS


class TestDeleteEntitiesByIds:
    """Validate the DELETE_ENTITIES_BY_IDS query."""

    def test_uses_unwind(self):
        assert "UNWIND $entity_ids AS entity_id" in DELETE_ENTITIES_BY_IDS

    def test_filters_by_repository(self):
        assert "$repository" in DELETE_ENTITIES_BY_IDS

    def test_does_not_delete_packages(self):
        assert "NOT n:Package" in DELETE_ENTITIES_BY_IDS


class TestDeleteOrphanPackages:
    """Validate the DELETE_ORPHAN_PACKAGES query."""

    def test_targets_package_label(self):
        assert ":Package" in DELETE_ORPHAN_PACKAGES

    def test_preserves_parent_namespaces_while_child_packages_exist(self):
        assert "STARTS WITH p.id + '.'" in DELETE_ORPHAN_PACKAGES

    def test_returns_deleted_count_for_iterative_cleanup(self):
        assert "RETURN size(packages) AS count" in DELETE_ORPHAN_PACKAGES

    def test_only_removes_nodes_without_incoming_relationships(self):
        assert "WHERE NOT ()-->(p)" in DELETE_ORPHAN_PACKAGES


class TestGetVectorIndex:
    """Validate the GET_VECTOR_INDEX query."""

    def test_filters_by_vector_indexes(self):
        assert "type = 'VECTOR'" in GET_VECTOR_INDEX

    def test_filters_by_index_name(self):
        assert "$index_name" in GET_VECTOR_INDEX


class TestDropIndexQuery:
    """Validate the drop_index_query helper."""

    def test_generates_drop_index_statement(self):
        query = drop_index_query("vector_class_embedding")
        assert query == "DROP INDEX vector_class_embedding IF EXISTS"


class TestCreateVectorIndex:
    """Validate the CREATE_VECTOR_INDEX template."""

    def test_contains_vector_index(self):
        assert "VECTOR INDEX" in CREATE_VECTOR_INDEX

    def test_contains_cosine(self):
        assert "cosine" in CREATE_VECTOR_INDEX

    def test_has_label_placeholder(self):
        assert "{label}" in CREATE_VECTOR_INDEX

    def test_has_index_name_placeholder(self):
        assert "{index_name}" in CREATE_VECTOR_INDEX

    def test_references_embedding_property(self):
        assert "embedding" in CREATE_VECTOR_INDEX

    def test_references_dimensions(self):
        assert "dimensions" in CREATE_VECTOR_INDEX
