"""Tests for constellation.graph.schema — Neo4j schema constants."""

from constellation.graph.schema import (
    NODE_LABELS,
    EMBEDDABLE_LABELS,
    CONSTRAINTS,
    COMPOSITE_INDEXES,
)


class TestNodeLabels:
    """Validate the NODE_LABELS list."""

    EXPECTED_LABELS = [
        "Repository",
        "File",
        "Package",
        "Class",
        "Interface",
        "Method",
        "Constructor",
        "Field",
    ]

    def test_all_eight_labels_present(self):
        assert len(NODE_LABELS) == 8

    def test_contains_expected_labels(self):
        for label in self.EXPECTED_LABELS:
            assert label in NODE_LABELS, f"Missing label: {label}"

    def test_labels_are_strings(self):
        for label in NODE_LABELS:
            assert isinstance(label, str)


class TestEmbeddableLabels:
    """Validate the EMBEDDABLE_LABELS list."""

    EXPECTED = ["Method", "Class", "Interface", "Constructor"]

    def test_embeddable_labels_count(self):
        assert len(EMBEDDABLE_LABELS) == 4

    def test_embeddable_labels_content(self):
        assert set(EMBEDDABLE_LABELS) == set(self.EXPECTED)

    def test_embeddable_labels_are_subset_of_node_labels(self):
        for label in EMBEDDABLE_LABELS:
            assert label in NODE_LABELS


class TestConstraints:
    """Validate uniqueness constraints."""

    def test_eight_constraints(self):
        assert len(CONSTRAINTS) == 8

    def test_each_constraint_creates_uniqueness(self):
        for constraint in CONSTRAINTS:
            assert "CREATE CONSTRAINT" in constraint
            assert "IF NOT EXISTS" in constraint
            assert "REQUIRE" in constraint
            assert "IS UNIQUE" in constraint

    def test_one_constraint_per_node_label(self):
        for label in NODE_LABELS:
            matches = [c for c in CONSTRAINTS if f"(n:{label})" in c]
            assert len(matches) == 1, f"Expected exactly 1 constraint for {label}, got {len(matches)}"

    def test_repository_constraint_uses_name(self):
        repo_constraints = [c for c in CONSTRAINTS if "Repository" in c]
        assert len(repo_constraints) == 1
        assert "n.name" in repo_constraints[0]

    def test_non_repository_constraints_use_id(self):
        non_repo = [c for c in CONSTRAINTS if "Repository" not in c]
        for constraint in non_repo:
            assert "n.id" in constraint


class TestCompositeIndexes:
    """Validate composite indexes."""

    def test_at_least_three_indexes(self):
        assert len(COMPOSITE_INDEXES) >= 3

    def test_each_index_creates_index(self):
        for index in COMPOSITE_INDEXES:
            assert "CREATE INDEX" in index
            assert "IF NOT EXISTS" in index

    def test_file_index_exists(self):
        file_indexes = [i for i in COMPOSITE_INDEXES if "File" in i]
        assert len(file_indexes) >= 1

    def test_indexes_include_repository_column(self):
        for index in COMPOSITE_INDEXES:
            assert "n.repository" in index
