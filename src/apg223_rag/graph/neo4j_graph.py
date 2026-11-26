import ssl
import re
import truststore
from typing import List, Dict, Any, Tuple
import certifi
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from apg223_rag.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE

truststore.inject_into_ssl()

PRUDENTIAL_STD_RE = re.compile(r"\b(?:APS|CPS|CPG)\s+\d{3}\b")

CONCEPT_PATTERNS: Dict[str, re.Pattern] = {
    "LVR": re.compile(r"\b(?:LVR|LVRs|loan-to-valuation ratios?)\b", re.IGNORECASE),
    "LMI": re.compile(r"\b(?:LMI|lenders mortgage insurance)\b", re.IGNORECASE),
    "Serviceability": re.compile(r"\bserviceability\b", re.IGNORECASE),
    "Interest Rate Buffer": re.compile(r"\binterest rate buffers?\b", re.IGNORECASE),
    "HEM": re.compile(r"\b(?:HEM|Household Expenditure Measure)\b"),
    "HPI": re.compile(r"\b(?:HPI|Henderson Poverty Index)\b"),
    "HELP": re.compile(r"\b(?:HELP|Higher Education Loan Program)\b"),
    "NIS": re.compile(r"\b(?:NIS|net income surplus)\b"),
    "Override": re.compile(r"\boverrid(?:e|es|ing|den)\b", re.IGNORECASE),
    "Interest-Only": re.compile(r"\binterest-only\b", re.IGNORECASE),
    "Reverse Mortgage": re.compile(r"\breverse mortgages?\b", re.IGNORECASE),
    "HELOC": re.compile(r"\b(?:HELOC|HELOCs|home equity lines?-of-credit)\b", re.IGNORECASE),
    "SMSF": re.compile(r"\b(?:SMSF|SMSFs|self-managed superannuation funds?)\b", re.IGNORECASE),
    "Hardship": re.compile(r"\bhardship\b", re.IGNORECASE),
    "Stress Testing": re.compile(r"\bstress test(?:ing|s)?\b", re.IGNORECASE),
    "Guarantor": re.compile(r"\bguarantors?\b", re.IGNORECASE),
    "Broker": re.compile(r"\bbrokers?\b", re.IGNORECASE),
}


class Neo4jGraphStore:
    def __init__(self):
        if not NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD is not set in .env.")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.database = NEO4J_DATABASE

    def _connect_with_ssl_fallback(self):
        # 1. Standard verified TLS using macOS Keychain / certifi
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            driver.verify_connectivity()
            return driver
        except ServiceUnavailable as e:
            if "SSLCertVerificationError" not in str(e) and "self-signed" not in str(e):
                raise
            # 2. Fallback for local SSL-inspecting proxies/firewalls (still encrypted TLS via neo4j+ssc)
            ssc_uri = NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://")
            driver = GraphDatabase.driver(ssc_uri, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            driver.verify_connectivity()
            return driver

    def close(self):
        self.driver.close()

    def build_graph(self, parents: List[Dict[str, Any]], glossary: List[Dict[str, Any]]) -> Tuple[int, int]:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            session.run("CREATE CONSTRAINT sub_id IF NOT EXISTS FOR (s:Subsection) REQUIRE s.parent_id IS UNIQUE")
            session.run("CREATE CONSTRAINT std_code IF NOT EXISTS FOR (p:PrudentialStandard) REQUIRE p.code IS UNIQUE")
            session.run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:RegulatoryConcept) REQUIRE c.name IS UNIQUE")

            for p in parents:
                p_id = p["parent_id"]
                sec = p["section"]
                subsec = p["subsection"]
                text = p["content"]

                session.run(
                    """
                    MERGE (sec:Section {name: $sec})
                    MERGE (sub:Subsection {parent_id: $p_id})
                    SET sub.title = $subsec, sub.page_start = $p_start, sub.page_end = $p_end
                    MERGE (sec)-[:CONTAINS_SUBSECTION]->(sub)
                    """,
                    sec=sec, p_id=p_id, subsec=subsec, p_start=p["page_start"], p_end=p["page_end"],
                )

                for std in set(PRUDENTIAL_STD_RE.findall(text)):
                    session.run(
                        """
                        MATCH (sub:Subsection {parent_id: $p_id})
                        MERGE (std:PrudentialStandard {code: $std})
                        MERGE (sub)-[:REFERENCES_STANDARD]->(std)
                        """,
                        p_id=p_id, std=std,
                    )

                for canonical_name, pattern in CONCEPT_PATTERNS.items():
                    if pattern.search(text):
                        session.run(
                            """
                            MATCH (sub:Subsection {parent_id: $p_id})
                            MERGE (c:RegulatoryConcept {name: $cname})
                            MERGE (sub)-[:ADDRESSES_CONCEPT]->(c)
                            """,
                            p_id=p_id, cname=canonical_name,
                        )

            for g in glossary:
                session.run(
                    """
                    MERGE (gt:GlossaryTerm {term: $term})
                    SET gt.definition = $definition, gt.page = $page
                    """,
                    term=g["term"], definition=g["definition"], page=g["page"],
                )

            counts = session.run(
                "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS rels"
            ).single()
            return (counts["nodes"], counts["rels"]) if counts else (0, 0)

    def query_graph(self, query: str, top_k_parents: int = 2) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        matched_standards = list(set(PRUDENTIAL_STD_RE.findall(query)))
        matched_concepts = [cname for cname, pat in CONCEPT_PATTERNS.items() if pat.search(query)]

        parent_scores: Dict[str, int] = {}
        all_triples: List[str] = []

        with self.driver.session(database=self.database) as session:
            records = session.run(
                """
                MATCH (sub:Subsection)-[r]->(target)
                WHERE (target:PrudentialStandard AND target.code IN $standards)
                   OR (target:RegulatoryConcept AND target.name IN $concepts)
                RETURN sub.parent_id AS pid,
                       sub.title AS title,
                       type(r) AS rel,
                       coalesce(target.code, target.name) AS target_name
                """,
                standards=matched_standards,
                concepts=matched_concepts,
            )

            for rec in records:
                pid = rec["pid"]
                weight = 3 if rec["rel"] == "REFERENCES_STANDARD" else 1
                parent_scores[pid] = parent_scores.get(pid, 0) + weight
                all_triples.append(f"({rec['title']} [{pid}]) --[{rec['rel']}]--> ({rec['target_name']})")

            glossary_hits = []
            for greq in session.run("MATCH (gt:GlossaryTerm) RETURN gt.term AS term, gt.definition AS def"):
                term = greq["term"]
                if len(term) >= 3 and re.search(rf"\b{re.escape(term)}\b", query, re.IGNORECASE):
                    glossary_hits.append({"term": term, "definition": greq["def"]})

        ranked_pids = sorted(parent_scores, key=lambda k: parent_scores[k], reverse=True)
        return ranked_pids[:top_k_parents], list(dict.fromkeys(all_triples)), glossary_hits