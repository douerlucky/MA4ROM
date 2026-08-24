from itertools import combinations

from rdflib import Graph, RDF, RDFS, OWL, BNode, URIRef
from rdflib.collection import Collection
import json
from utils.name_similarity import name_overlap


def _transitive_closure(direct_map: dict, universe: set[str]) -> dict:
    """
    direct_map: node -> [neighbors]
    返回每个节点的传递闭包（不含自身）。
    """
    closure = {}

    def dfs(node: str, seen: set[str]):
        for nxt in direct_map.get(node, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            dfs(nxt, seen)

    for node in universe:
        visited = set()
        dfs(node, visited)
        closure[node] = sorted(visited)
    return closure


def _build_union_members(g: Graph) -> dict:
    """
    提取所有 union class expression 的成员。

    之前只扫描 ``rdfs:domain``/``rdfs:range`` 中出现的
    blank node。但 OWL restriction 的 ``owl:onClass`` 也可以是
    union expression（例如 ``cityIn`` 的 Country/Province 两条
    restriction），这会让下游 endpoint matcher 无法判断其为合法
    端点。扫描整张图中的 unionOf 声明，并递归展开嵌套
    union，不依赖数据集名或目标分数。

    返回:
      {
        "expression_id": ["ClassA", "ClassB", ...]
      }
    """
    union_members: dict[str, list[str]] = {}

    # A union member may itself be an anonymous union expression.  Flattening
    # it is safe because union is associative; intersection/complement nodes
    # are intentionally not traversed and therefore cannot be mistaken for a
    # disjunctive endpoint.
    def flatten(term, seen: set) -> list[str]:
        if isinstance(term, BNode):
            if term in seen:
                return []
            seen = {*seen, term}
            nested_lists = list(g.objects(term, OWL.unionOf))
            if not nested_lists:
                return []
            out: list[str] = []
            for list_node in nested_lists:
                try:
                    members = list(Collection(g, list_node))
                except Exception:
                    continue
                for member in members:
                    out.extend(flatten(member, seen))
            return out
        return [str(term)] if term is not None else []

    # Include named classes as well as blank class expressions.  OWL permits
    # ``:Parent owl:unionOf (...)`` and downstream property declarations may
    # reference that named parent directly.
    for expr, _, _list_node in g.triples((None, OWL.unionOf, None)):
        members: list[str] = []
        for list_node in g.objects(expr, OWL.unionOf):
            try:
                raw_members = list(Collection(g, list_node))
            except Exception:
                continue
            for member in raw_members:
                members.extend(flatten(member, {expr}))
        if members:
            union_members[str(expr)] = sorted(set(members))

    return union_members


def _expand_class_expression(g: Graph, term, seen: set | None = None) -> list[str]:
    """Expand a graph-local class expression into named endpoint classes.

    RDF blank-node identifiers are local to one parsed graph and cannot be
    safely joined with identifiers produced by a second parse.  Consumers
    therefore need the concrete members copied into the ontology payload at
    parse time.  Anonymous intersections/complements remain opaque (their
    blank-node marker is retained) instead of being treated as a union.
    """
    seen = set(seen or set())
    if term in seen:
        return []
    seen.add(term)
    union_lists = list(g.objects(term, OWL.unionOf))
    if not union_lists:
        return [str(term)]

    members: list[str] = []
    for list_node in union_lists:
        try:
            items = list(Collection(g, list_node))
        except Exception:
            continue
        for item in items:
            members.extend(_expand_class_expression(g, item, seen))

    # A named class which additionally declares owl:unionOf remains a usable
    # named endpoint in its own right; anonymous expressions do not have that
    # identity and are represented only by their concrete members.
    if isinstance(term, URIRef):
        members.insert(0, str(term))
    return list(dict.fromkeys(members))


def _expanded_class_terms(g: Graph, terms) -> list[str]:
    values: list[str] = []
    for term in terms:
        values.extend(_expand_class_expression(g, term))
    return list(dict.fromkeys(values))


def _read_disjoint_union_subclass_edges(g: Graph) -> list[tuple[str, str]]:
    """Expand OWL ``disjointUnionOf`` into sound subclass edges.

    ``owl:disjointUnionOf`` states that the named parent class is exactly the
    union of its members.  In particular, every member is a subclass of that
    parent.  The old parser retained the disjointness information but dropped
    this positive entailment, which made perfectly valid domain/range pairs
    (for example a concrete political class and ``AdministrativeArea``) look
    unrelated to the endpoint matcher.

    Only named URI classes are returned.  Anonymous class expressions are
    still handled by ``union_members`` and are intentionally not promoted to
    named hierarchy nodes here.
    """
    edges: list[tuple[str, str]] = []
    for parent_term, members_list in g.subject_objects(OWL.disjointUnionOf):
        parent = _named_class_uri(parent_term)
        if not parent:
            continue
        try:
            members = Collection(g, members_list)
        except Exception:
            continue
        for member_term in members:
            member = _named_class_uri(member_term)
            if member and member != parent:
                edges.append((member, parent))
    return edges


def _named_class_uri(term) -> str | None:
    """Return a URI for a named OWL class term, excluding blank expressions."""
    if isinstance(term, URIRef):
        return str(term)
    return None


def _read_disjointness(g: Graph) -> tuple[set[str], set[tuple[str, str]], list[tuple[str, ...]]]:
    """
    Read the named-class portion of OWL disjointness axioms.

    Both ``owl:disjointWith`` and OWL 2's ``owl:AllDisjointClasses`` use
    symmetric pairwise semantics.  Anonymous class expressions are deliberately
    skipped here: this module exposes class-URI constraints to downstream
    mapping code, so treating a blank expression as a named class would create
    an unusable and potentially unsound constraint.
    """
    named_classes: set[str] = set()
    direct_pairs: set[tuple[str, str]] = set()
    all_disjoint_sets: set[tuple[str, ...]] = set()

    def add_pair(left: str, right: str) -> None:
        if left == right:
            # A self-disjoint axiom makes an ontology inconsistent.  It should
            # not make every use of that class fail candidate compatibility.
            return
        direct_pairs.add(tuple(sorted((left, right))))

    # OWL 1 / RDF style: :A owl:disjointWith :B .
    for left_term, right_term in g.subject_objects(OWL.disjointWith):
        left = _named_class_uri(left_term)
        right = _named_class_uri(right_term)
        if not left or not right:
            continue
        named_classes.update((left, right))
        add_pair(left, right)

    # OWL 2 style:
    # _:axiom a owl:AllDisjointClasses ; owl:members ( :A :B :C ) .
    for axiom in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for members_list in g.objects(axiom, OWL.members):
            try:
                members = sorted({
                    uri
                    for term in Collection(g, members_list)
                    if (uri := _named_class_uri(term)) is not None
                })
            except Exception:
                # A malformed RDF list should not prevent the rest of an
                # ontology from being usable.
                continue

            if len(members) < 2:
                continue
            member_tuple = tuple(members)
            all_disjoint_sets.add(member_tuple)
            named_classes.update(member_tuple)
            for left, right in combinations(member_tuple, 2):
                add_pair(left, right)

    return named_classes, direct_pairs, sorted(all_disjoint_sets)


def _build_incompatibility_closure(
    direct_pairs: set[tuple[str, str]],
    descendants_of: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Expand direct OWL disjointness through subclass inheritance.

    If A is disjoint with B, every named subclass of A is incompatible with B
    and with every named subclass of B.  The result is symmetric and excludes
    self-pairs so it is safe to use as a candidate filter.
    """
    incompatible: dict[str, set[str]] = {}

    for left, right in direct_pairs:
        left_family = {left, *descendants_of.get(left, [])}
        right_family = {right, *descendants_of.get(right, [])}
        for left_member in left_family:
            for right_member in right_family:
                if left_member == right_member:
                    continue
                incompatible.setdefault(left_member, set()).add(right_member)
                incompatible.setdefault(right_member, set()).add(left_member)

    return {
        class_uri: sorted(class_incompatibilities)
        for class_uri, class_incompatibilities in sorted(incompatible.items())
    }


def are_classes_disjoint(
    left_class_uri: str | None,
    right_class_uri: str | None,
    ontology: dict | None,
) -> bool:
    """
    Return whether two named classes are incompatible under OWL disjointness.

    ``read_ontology`` precomputes the inheritance-aware closure in
    ``ontology['incompatible_classes']``.  The small direct-axiom fallback
    makes this helper useful with manually assembled legacy ontology dicts as
    well, but callers should normally pass the result of ``read_ontology``.
    """
    if not left_class_uri or not right_class_uri or left_class_uri == right_class_uri:
        return False
    if not ontology:
        return False

    incompatible = ontology.get("incompatible_classes", {})
    if right_class_uri in incompatible.get(left_class_uri, []):
        return True

    # Backward-compatible fallback for a dict which contains direct edges but
    # was not produced by the current read_ontology implementation.
    direct = ontology.get("direct_disjoint_classes", {})
    return right_class_uri in direct.get(left_class_uri, [])


def read_ontology(path: str) -> dict:
    """
    解析本体文件（Turtle 格式），返回 classes / object_properties / datatype_properties。

    参数:
        path: 本体文件路径（从 config.ONTOLOGY_PATH 传入，不在此处硬编码）
    """
    g = Graph()
    g.parse(path, format="turtle")

    classes = set()
    object_properties = {}
    datatype_properties = {}
    subclass_of = {}
    children_of = {}
    disjoint_classes, direct_disjoint_pairs, all_disjoint_class_sets = _read_disjointness(g)

    # 读取 Classes（显式声明）
    for s in g.subjects(RDF.type, OWL.Class):
        classes.add(str(s))

    # 兼容很多本体的“隐式类声明”写法：
    # 仅通过 rdfs:subClassOf 出现，但没有显式 rdf:type owl:Class
    for child, parent in g.subject_objects(RDFS.subClassOf):
        if not isinstance(child, BNode):
            classes.add(str(child))
        if not isinstance(parent, BNode):
            classes.add(str(parent))

    # 兼容通过 domain/range 间接出现的命名类
    # （跳过 blank node class expression）
    for _, _, d in g.triples((None, RDFS.domain, None)):
        if not isinstance(d, BNode):
            classes.add(str(d))
    for _, _, r in g.triples((None, RDFS.range, None)):
        if not isinstance(r, BNode):
            classes.add(str(r))

    # A class may be declared only inside a disjointness axiom.  Preserve it
    # as a usable candidate class even when it has no explicit owl:Class type.
    classes.update(disjoint_classes)

    # 读取 class hierarchy (child -> parent)
    # 仅保留命名 Class（跳过 blank node / 限制表达式）
    for child, parent in g.subject_objects(RDFS.subClassOf):
        if isinstance(child, BNode) or isinstance(parent, BNode):
            continue
        child_uri = str(child)
        parent_uri = str(parent)
        if child_uri not in subclass_of:
            subclass_of[child_uri] = []
        if parent_uri not in subclass_of[child_uri]:
            subclass_of[child_uri].append(parent_uri)

        if parent_uri not in children_of:
            children_of[parent_uri] = []
        if child_uri not in children_of[parent_uri]:
            children_of[parent_uri].append(child_uri)

    # OWL's disjointUnionOf is both a partition and a union axiom.  Retain
    # the positive subclass entailment as well as the disjointness pairs read
    # above; endpoint/domain matching needs both halves of the axiom.
    for child_uri, parent_uri in _read_disjoint_union_subclass_edges(g):
        classes.update((child_uri, parent_uri))
        if parent_uri not in subclass_of.setdefault(child_uri, []):
            subclass_of[child_uri].append(parent_uri)
        if child_uri not in children_of.setdefault(parent_uri, []):
            children_of[parent_uri].append(child_uri)

    # 读取 Object Properties + domain + range
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        raw_domain = list(g.objects(prop, RDFS.domain))
        raw_range = list(g.objects(prop, RDFS.range))
        # Materialize graph-local union members now; callers should not have
        # to compare blank-node IDs produced by a separate RDF parse.
        domain = _expanded_class_terms(g, raw_domain)
        range_ = _expanded_class_terms(g, raw_range)

        object_properties[str(prop)] = {
            "domain": domain,
            "range": range_,
            "domain_expressions": [str(d) for d in raw_domain],
            "range_expressions": [str(r) for r in raw_range],
        }

    # 读取 Datatype Properties + domain + range
    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        raw_domain = list(g.objects(prop, RDFS.domain))
        raw_range = list(g.objects(prop, RDFS.range))
        domain = _expanded_class_terms(g, raw_domain)
        range_ = _expanded_class_terms(g, raw_range)

        datatype_properties[str(prop)] = {
            "domain": domain,
            "range": range_,
            "domain_expressions": [str(d) for d in raw_domain],
            "range_expressions": [str(r) for r in raw_range],
        }

    # 传递闭包：child -> all ancestors / parent -> all descendants
    ancestors_of = _transitive_closure(subclass_of, classes)
    descendants_of = _transitive_closure(children_of, classes)
    incompatible_classes = _build_incompatibility_closure(
        direct_disjoint_pairs,
        descendants_of,
    )
    # Include classes with no declared incompatibility as empty lists.  This
    # makes both maps total over ontology["classes"], which is convenient for
    # downstream candidate filtering and JSON serialisation.
    incompatible_classes = {
        class_uri: incompatible_classes.get(class_uri, [])
        for class_uri in sorted(classes)
    }

    direct_disjoint_classes = {}
    for left, right in direct_disjoint_pairs:
        direct_disjoint_classes.setdefault(left, set()).add(right)
        direct_disjoint_classes.setdefault(right, set()).add(left)
    direct_disjoint_classes = {
        class_uri: sorted(direct_disjoint_classes.get(class_uri, set()))
        for class_uri in sorted(classes)
    }

    # Preserve the human-readable ontology evidence used by context-enhanced
    # Class selection.  Returning only a URI/local name discards exactly the
    # label/comment semantics an LLM needs to distinguish sibling Classes with
    # identical structural position.  This is generic RDF/OWL metadata; no
    # dataset-specific vocabulary or score feedback is involved.
    class_annotations = {}
    for class_uri in sorted(classes):
        class_term = URIRef(class_uri)
        labels = sorted({
            str(value).strip()
            for value in g.objects(class_term, RDFS.label)
            if str(value).strip()
        })
        comments = sorted({
            str(value).strip()
            for value in g.objects(class_term, RDFS.comment)
            if str(value).strip()
        })
        class_annotations[class_uri] = {
            "labels": labels,
            "comments": comments,
        }

    union_members = _build_union_members(g)

    return {
        "classes": sorted(classes),
        "class_annotations": class_annotations,
        "object_properties": object_properties,
        "datatype_properties": datatype_properties,
        "subclass_of": subclass_of,
        "children_of": children_of,
        "ancestors_of": ancestors_of,
        "descendants_of": descendants_of,
        "union_members": union_members,
        # Direct OWL declarations, useful for diagnostics/auditing.
        "direct_disjoint_classes": direct_disjoint_classes,
        "all_disjoint_class_sets": [list(members) for members in all_disjoint_class_sets],
        # Symmetric inheritance-aware closure.  For example, if A and B are
        # disjoint, every subclass of A is incompatible with every subclass of
        # B.  Use are_classes_disjoint(...) instead of reimplementing this.
        "incompatible_classes": incompatible_classes,
    }


def local_name(uri: str) -> str:
    """
    从 URI 中提取本地名称。

    "http://conference#Paper"    → "Paper"
    "http://conference#hasTitle" → "hasTitle"
    "http://example.org/Person"  → "Person"
    """
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def _semantic_class_match_score(hint: str, candidate: str, ontology: dict | None = None) -> float:
    """
    单个 class 间匹配分：
      - 精确匹配: 1.0
      - 子类/父类可达: 0.85
      - 本地名相似: 0.5
      - 不匹配: 0.0
    """
    if not hint or not candidate:
        return 0.0

    if candidate == hint:
        return 1.0

    hint_local = local_name(hint)
    cand_local = local_name(candidate)
    if cand_local.lower() == hint_local.lower():
        return 1.0

    if ontology:
        ancestors_of = ontology.get("ancestors_of", {})
        if candidate in ancestors_of.get(hint, []):
            return 0.85
        if hint in ancestors_of.get(candidate, []):
            return 0.85

    if name_overlap(hint_local, cand_local) > 0.6:
        return 0.5
    return 0.0


def hint_match(hint: str, prop_values: list, ontology: dict | None = None) -> float:
    """
    判断 hint（Class URI）是否出现在属性的 domain/range 声明列表中。

    打分规则：
      精确匹配（URI 完全相同）    → 1.0
      本地名完全匹配（忽略大小写）→ 1.0
      名称部分相似（>0.6）        → 0.5
      属性无 domain/range 声明    → 0.3（不惩罚，本体可能不完整）
      完全不匹配                  → 0.0
    """
    if not hint:
        return 0.3
    if not prop_values:
        return 0.3

    union_members = (ontology or {}).get("union_members", {})
    best = 0.0

    for val in prop_values:
        # 若声明是 union class expression，展开其成员再匹配
        members = union_members.get(val, [])
        if members:
            member_best = 0.0
            for m in members:
                member_best = max(member_best, _semantic_class_match_score(hint, m, ontology))
            best = max(best, member_best)
            continue

        best = max(best, _semantic_class_match_score(hint, val, ontology))

    return best


if __name__ == "__main__":
    from config import ONTOLOGY_PATH   # ← 路径从 config 读取
    ontology = read_ontology(ONTOLOGY_PATH)
    print(json.dumps(ontology, indent=4))
