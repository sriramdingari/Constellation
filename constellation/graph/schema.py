"""Neo4j schema constants — node labels, constraints, and indexes."""

NODE_LABELS = [
    "Repository",
    "File",
    "Package",
    "Class",
    "Interface",
    "Method",
    "Constructor",
    "Field",
]

EMBEDDABLE_LABELS = ["Method", "Class", "Interface", "Constructor"]

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Repository) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:File) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Package) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Class) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Interface) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Method) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Constructor) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Field) REQUIRE n.id IS UNIQUE",
]

COMPOSITE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS FOR (n:File) ON (n.repository, n.file_path)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Class) ON (n.repository, n.file_path)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Method) ON (n.repository, n.file_path)",
]
