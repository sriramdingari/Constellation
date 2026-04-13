# C# Call/Reference Resolution Design

## Context

Constellation's C# parser (`constellation/parsers/dotnet.py`) extracts entities — classes, methods, constructors, fields, interfaces, enums, namespaces — but does not extract call sites. This means Telescope's `get_callers`, `get_callees`, `get_function_context`, and `get_impact` return empty results for C# code.

The Java and JS/TS parsers already follow a dual-channel call/reference model established in the earlier call/reference resolution foundation project. Every real call site produces either:

1. A resolved `CALLS` edge to a concrete declaration entity
2. A `CALLS` edge to a site-specific `Reference` entity

This project extends that same model to C#, bringing the fourth advertised language to call-graph parity.

The primary target codebase is cleartax-dev — a 4,000-file C# monolith with 52 controllers, heavy domain models across `AT.Business`, `AT.DBModels`, `AT.Web2`, and a validation engine. The codebase uses legacy MVC patterns with limited DI adoption.

## Goals

- Extract call sites from C# method and constructor bodies.
- Resolve calls to concrete same-repo targets when static evidence supports it.
- Represent unresolved calls as first-class `Reference` entities with call-site identity.
- Detect test and endpoint stereotypes so Telescope's `get_impact` can categorize callers correctly.
- Follow the same contract, resolution philosophy, and identity model as Java and JS/TS.

## Non-Goals

- No interface/abstract dispatch resolution (OVERRIDES relationships). Deferred to v2.
- No constructor injection / DEPENDS_ON relationships. Deferred to v2.
- No struct or record type support beyond what the current parser already handles. Deferred to v2.
- No local function support. Deferred to v2.
- No cross-file type resolution beyond what using directives provide.
- No runtime, reflection, or dynamic dispatch modeling.
- No HTTP route path extraction from endpoint attributes. Telescope does not use route metadata today.

## Approach

Extend Constellation's existing `dotnet.py` incrementally. The current parser correctly extracts all entity types and structural relationships. Layer call resolution on top using the same architecture as the JS/TS parser. Cherry-pick using-directive tracking and type resolution patterns from the code_graph_service reference implementation where they improve resolution quality.

## Reference Implementation

The C# parser at `personal/whats-the-update/code_graph_service/services/parsers/dotnet_parser.py` (2,046 lines) provides a reference for:

- Using-directive extraction (lines 412-483)
- Call extraction via `invocation_expression` (lines 1676-1726)
- Type resolution context (lines 268-304)
- Framework detection patterns (lines 28-163)

This reference informs the design but is not ported wholesale. Constellation's data model, resolution contract, and identity scheme take precedence.

## Using-Directive Tracking

A new first pass before entity extraction scans the root for `using_directive` AST nodes and populates the parsing context.

### What gets tracked

- **Regular usings** (`using AT.Business;`) — namespace strings for type resolution
- **Static usings** (`using static AT.Common.Constants;`) — enables resolving bare calls to static class members
- **Alias usings** (`using Svc = AT.Business.TaxService;`) — maps alias name to fully qualified type
- **Global usings** (`global using AT.Shared;`) — treated identically to regular usings since parsing is per-file

### Data structure

Added to `_ParsingContext`:

```
usings: list[str]              # ["AT.Business", "AT.Common"]
using_statics: list[str]       # ["AT.Common.Constants"]
using_aliases: dict[str, str]  # {"Svc": "AT.Business.TaxService"}
```

### AST detection

Tree-sitter C# represents `using_directive` nodes with:

- Regular using (`using System;`): a `qualified_name` child, no `name` field.
- Alias using (`using Svc = MyNs.MyService;`): a `name` field (`node.child_by_field_name("name")`) containing the alias identifier, plus a `qualified_name` child for the target.
- Static using (`using static MyNs.Constants;`): a `static` modifier child plus a `qualified_name`.
- Global using (`global using System;`): a `global` modifier child plus any of the above forms.

Alias detection must use `node.child_by_field_name("name")` — if it returns non-None, the directive is an alias. The `=` token is a bare separator, not a named node. The reference implementation (code_graph_service) uses a `name_equals` child type that does not exist in tree-sitter C# — do not follow that pattern.

### How it is used downstream

When resolving a type name like `TaxService` at a call site:

1. Check if `TaxService` is a class in the current file.
2. Check if `TaxService` is an alias in `using_aliases`.
3. For each namespace in `usings`, check if `{namespace}.TaxService` matches a known class.
4. Check the file's own namespace — `{current_namespace}.TaxService`.
5. If none match, the type is external and calls through it stay unresolved.

## Call Extraction

### AST nodes

Tree-sitter C# represents calls as two node types:

- **`invocation_expression`** — method calls like `Validate()`, `service.Calculate()`, `TaxService.StaticMethod()`
- **`object_creation_expression`** — constructor calls like `new TaxService()`

The function part of an `invocation_expression` is one of:

- `identifier_name` — bare call (same-class or static-using member)
- `member_access_expression` — qualified call with receiver (`service.Calculate()`)
- `generic_name` — generic call like `Convert<int>()`
- `conditional_access_expression` — null-conditional call like `handler?.Invoke()`. Contains a `member_binding_expression` child with the method name. The receiver is the expression before `?.`. This pattern is idiomatic in C# for delegate/event invocations and must be unwrapped to extract receiver and method name.

### Traversal strategy

Call extraction performs a recursive walk of the method/constructor body, visiting all descendants. This ensures calls nested inside `await_expression`, `assignment_expression`, `return_statement`, `if_statement` conditions, LINQ chains, and other compound expressions are found.

Lambda bodies (`lambda_expression`) and anonymous delegate bodies (`anonymous_method_expression`) are walked and calls within them are attributed to the enclosing method, similar to how the JS/TS parser handles arrow functions inside method bodies.

Constructor bodies (`constructor_declaration > block`) are walked with the same logic as method bodies.

### Compile-time operators to skip

The following look like `invocation_expression` in the AST but are compile-time operators, not real call sites. They must be filtered out:

- `nameof` — `nameof(DoWork)` is a compile-time string literal
- `typeof` — `typeof(TaxService)` returns a `Type` object
- `sizeof` — `sizeof(int)` returns a size constant
- `default` — `default(T)` returns a default value

These are identified by checking whether the function part of the `invocation_expression` is an `identifier_name` matching one of these keywords.

### Resolution order

Mirrors the JS/TS resolution order from the foundation spec:

1. **Same-class methods** — `Validate()` resolves when the current class has a method named `Validate`.
2. **this/base calls** — `this.Validate()` resolves via same-class lookup. `base.Validate()` stays unresolved (inheritance chain not tracked in v1).
3. **Local receiver typing** — `var svc = new TaxService(); svc.Calculate()` resolves when the `new` expression target is a known same-file class.
4. **Static member calls via class name** — `TaxService.Calculate()` resolves when `TaxService` is in the same file or resolvable via using directives and the target class has a static method named `Calculate`.
5. **Static using members** — bare `DoSomething()` resolves when a `using static` imports a class that has a static method with that name, and the call did not resolve at tier 1.
6. **Using-alias resolution** — `Svc.Calculate()` resolves when `Svc` is a using alias mapping to a known class.
7. **Otherwise unresolved** — a `Reference` entity is created with call-site identity.

### Scope tracking

C# method bodies have one scope level relevant for receiver typing: local variable declarations with `new` expressions.

```
local_instance_types: dict[str, str]  # {"svc": "AT.Business.TaxService"}
```

Populated by scanning the method body for `local_declaration_statement` nodes, then navigating `variable_declaration > variable_declarator` children. When a `variable_declarator` has an `object_creation_expression` as its initializer (after the `=` token), extract the type name from the `object_creation_expression` and the variable name from the `variable_declarator`'s `identifier` child. The full AST path is:

```
local_declaration_statement
  variable_declaration
    implicit_type (var) | type_name
    variable_declarator
      identifier ("svc")
      =
      object_creation_expression
        new
        identifier ("TaxService")
        argument_list
```

This is the same concept as the JS/TS parser's `_collect_scope_instance_ids`.

For same-file class resolution, the existing two-pass architecture collects class names and their method members before call extraction begins:

```
module_class_ids: dict[str, str]           # {"TaxService": "repo::NS.TaxService"}
class_method_ids: dict[str, dict[str, str]]  # {class_id: {"Calculate": method_id}}
class_static_method_ids: dict[str, dict[str, str]]
```

### Deduplication

Same `seen_reference_targets` set as JS/TS. One Reference entity per unique call site. CALLS edges emitted for every occurrence.

### Call-site identity

Same format as JS/TS:

```
{source_method_id}::ref:{file_path}:{line}:{col}:{called_symbol}
```

Note: Java uses a slightly different format (`{source_method_id}::ref:{called_full}:{line}:{col}`), but both are handled by the same pipeline regex `_REF_SITE_RE`.

This ensures unresolved references from C# participate in the same pipeline normalization and do not merge across call sites.

### Reference entity properties

Each unresolved Reference entity should carry these properties for consistency with JS/TS:

```python
properties={
    "symbol": called_symbol,          # "service.Calculate"
    "receiver": receiver_text,         # "service" (when applicable)
    "enclosing_declaration_id": source_id,
    "enclosing_declaration_name": enclosing_name,
}
```

## Framework Detection

### Test detection

The current parser checks 6 attributes. Expand to cover:

**Attribute-based** (extend `TEST_ATTRIBUTES` frozenset):
- NUnit: `Test`, `TestCase`, `TestCaseSource`, `Theory`, `SetUp`, `TearDown`, `OneTimeSetUp`, `OneTimeTearDown`, `TestFixture`
- xUnit: `Fact`, `Theory`, `InlineData`, `MemberData`, `ClassData`
- MSTest: `TestMethod`, `DataTestMethod`, `DataRow`, `TestInitialize`, `TestCleanup`, `ClassInitialize`, `ClassCleanup`, `TestClass`

**Filename-based** (new `TEST_FILE_PATTERNS`):
- Files ending in `Test.cs`, `Tests.cs`, `Spec.cs`, `Specs.cs`
- All methods in matching files get `stereotypes=["test"]`

**Path-based** (new `TEST_PATH_PATTERNS`):
- Files under directories matching `/test/`, `/tests/`, `.Tests/`, `.Test/`
- All methods in matching files get `stereotypes=["test"]`

### Endpoint detection

New capability for impact analysis categorization.

**Class-level detection** (`CONTROLLER_BASE_CLASSES` frozenset):
- Base classes: `Controller`, `ControllerBase`, `ApiController`
- Attributes: `[ApiController]`, `[Controller]`

**Method-level detection** (`ENDPOINT_ATTRIBUTES` frozenset):
- `HttpGet`, `HttpPost`, `HttpPut`, `HttpDelete`, `HttpPatch`, `HttpHead`, `HttpOptions`
- `Route`, `RoutePrefix`

When a method is in a controller class and has an HTTP attribute, add `stereotypes=["endpoint"]`.

### Extensibility

All detection sets are module-level frozensets:

```python
TEST_ATTRIBUTES = frozenset({...})
TEST_FILE_PATTERNS = frozenset({...})
TEST_PATH_PATTERNS = frozenset({...})
CONTROLLER_BASE_CLASSES = frozenset({...})
ENDPOINT_ATTRIBUTES = frozenset({...})
```

Adding support for a new test framework or web framework means extending the relevant set. No structural changes required.

Framework detection is isolated into dedicated methods:

- `_detect_test_stereotypes(node, ctx) -> list[str]`
- `_detect_endpoint_stereotypes(node, ctx) -> list[str]`

These return stereotype strings and can be overridden or extended without touching call extraction logic.

## Pipeline Integration

Add `"csharp"` to the language gate in `_normalize_parse_result`:

```python
if language in {"python", "javascript", "csharp"}:
```

The C# parser reports `language = "csharp"`. This enables canonical `{file_entity_id}#{local_path}` normalization for all C# entities.

**Why C# joins the gate (but Java does not):** C# entity IDs use `{repository}::{namespace}.Class.Method` format. When two files in different directories define classes in the same namespace (common in partial-class scenarios and test fixtures), their entity IDs can collide. File-scoped normalization (`{file_entity_id}#local_path`) prevents this. Java entity IDs already include the fully-qualified package path which provides natural uniqueness, so Java does not need this normalization.

This is a deliberate change to C# entity identity. Any existing indexed C# graphs must be re-indexed after this change.

The existing `_REF_SITE_RE` regex handles the C# reference ID format without changes.

## Expected Unresolved Patterns

Certain common C# patterns will produce unresolved Reference entities by design:

- **Chained method calls / LINQ** — `list.Where(...).Select(...).ToList()` produces three `invocation_expression` nodes. Only the first (`list.Where(...)`) has a potentially resolvable receiver. The subsequent calls have `invocation_expression` receivers (the return value of the previous call), which are not statically typeable. Each becomes an unresolved Reference. This is by design — recall-first.
- **External library calls** — `logger.LogInformation(...)`, `context.SaveChangesAsync()`, `HttpClient.GetAsync(...)` — always unresolved since the library code is not in the repo.
- **base calls** — `base.OnInit()` — unresolved in v1 since we do not track inheritance chains.
- **Dynamic / reflection** — `MethodInfo.Invoke(...)` — inherently unresolvable statically.

For cleartax-dev, expect 40-50% resolution rate. The remaining unresolved references are still valuable — they appear in `get_callees` results and `get_impact` traversal.

## Architecture Boundaries

### Parser owns

- Identifying call sites in method and constructor bodies
- Attempting static resolution using local context, class members, and using directives
- Deciding whether a call site is resolved or unresolved
- Emitting either a resolved `CALLS` edge or a `Reference` entity plus `CALLS` edge
- Detecting test and endpoint stereotypes

### Pipeline owns

- Canonical ID normalization
- File-scoped entity identity
- Ensuring unresolved reference IDs remain stable after normalization

### Telescope owns

- Querying the graph for callers, callees, impact
- Categorizing callers by stereotype (test, endpoint, other)
- No Telescope changes required — it already handles Reference entities and stereotypes

## Identity Rules

### Declaration identity

Resolved calls target canonical declaration IDs in the format `{repository}::{namespace}.Class.Method`.

### Unresolved reference identity

Same call-site identity model as Java and JS/TS:

```
{source_method_id}::ref:{file_path}:{line}:{col}:{called_symbol}
```

Components: repository, file path, enclosing declaration ID, line, column, raw callee text.

This prevents:
- Two `Calculate()` calls in different methods from merging
- Same-name calls in different files from merging

## Success Criteria

1. Every real call site in a C# method or constructor body produces a CALLS edge — either resolved or to a Reference entity.
2. Same-class method calls resolve to concrete targets.
3. Cross-namespace calls resolve when using directives provide enough evidence.
4. `new Foo()` constructor calls emit CALLS edges.
5. Unresolved references use call-site identity and do not merge across sites.
6. Test methods are correctly stereotyped (attribute, filename, and path detection).
7. Endpoint methods are correctly stereotyped.
8. Telescope's `get_callers`, `get_callees`, `get_function_context`, and `get_impact` return meaningful results for C# code after re-indexing.
9. Pipeline normalization handles C# Reference entities correctly.
10. All existing C# entity extraction tests continue to pass.

## Recommended Follow-On Sequence

After this project:

1. **Interface/abstract dispatch** — OVERRIDES relationships, so `get_callers(IService.Method)` finds callers via all implementations
2. **Constructor injection** — DEPENDS_ON relationships for DI wiring visibility
3. **`is` pattern matching receiver typing** — `if (obj is TaxService svc) svc.Calculate()` — typed variables from pattern matching provide additional local receiver information
4. **Struct/record support** — first-class entity types for value types
5. **Local function support** — nested method resolution within method bodies
6. **Base class resolution** — `base.Method()` calls resolve when the base class is in the same repo
