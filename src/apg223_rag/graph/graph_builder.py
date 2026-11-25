import re
import networkx as nx
from typing import List, Dict, Any, Tuple

PRUDENTIAL_STD_RE = re.compile(r"\b(?:APS|CPS|CPG)\s+\d{3}\b")

DOMAIN_CONCEPTS = [
    "LVR", "loan-to-valuation ratio", "LMI", "lenders mortgage insurance",
    "serviceability", "interest rate buffer", "HEM", "HPI", "HELP",
    "override", "overrides", "interest-only", "reverse mortgage",
    "HELOC", "SMSF", "hardship", "stress testing", "guarantor", "broker", "NIS"
]


def build_regulatory_graph(parents: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Builds a directed Knowledge Graph linking:
      - Section -> Subsection (Parent ID)
      - Subsection -> Prudential Standards (APS 220, CPS 220, CPS 510, etc.)
      - Subsection -> Core Mortgage Lending Concepts (LVR, HELP, Overrides, etc.)
    """
    G = nx.DiGraph()

    for p in parents:
        p_id = p["parent_id"]
        sec = p["section"]
        subsec = p["subsection"]
        text = p["content"]

        G.add_node(sec, node_type="Section")
        G.add_node(p_id, node_type="Subsection", label=subsec, section=sec)
        G.add_edge(sec, p_id, relation="CONTAINS_SUBSECTION")

        for std in set(PRUDENTIAL_STD_RE.findall(text)):
            G.add_node(std, node_type="PrudentialStandard")
            G.add_edge(p_id, std, relation="REFERENCES_STANDARD")

        text_lower = text.lower()
        for concept in DOMAIN_CONCEPTS:
            if concept.lower() in text_lower:
                norm_name = concept.upper() if len(concept) <= 5 else concept.title()
                G.add_node(norm_name, node_type="RegulatoryConcept")
                G.add_edge(p_id, norm_name, relation="ADDRESSES_CONCEPT")

    return G


def traverse_graph_for_query(
    G: nx.DiGraph,
    query: str,
    top_k_parents: int = 2
) -> Tuple[List[str], List[str]]:
    """Finds matched entities in the query and traverses edges to linked Parent subsections."""
    q_lower = query.lower()
    matched_entities = [
        node for node, attrs in G.nodes(data=True)
        if attrs.get("node_type") in {"PrudentialStandard", "RegulatoryConcept"}
        and str(node).lower() in q_lower
    ]

    parent_scores: Dict[str, int] = {}
    triples: List[str] = []

    for entity in matched_entities:
        for pred in G.predecessors(entity):
            if G.nodes[pred].get("node_type") == "Subsection":
                parent_scores[pred] = parent_scores.get(pred, 0) + 1
                rel = G.get_edge_data(pred, entity)["relation"]
                label = G.nodes[pred].get("label", pred)
                triples.append(f"({label} [{pred}]) --[{rel}]--> ({entity})")

    ranked_parent_ids = sorted(parent_scores, key=lambda pid: parent_scores[pid], reverse=True)
    return ranked_parent_ids[:top_k_parents], list(set(triples))