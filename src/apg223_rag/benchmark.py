import json
import time
from typing import List, Dict, Any
from fpdf import FPDF
from apg223_rag.config import PROCESSED_DIR, PARENTS_JSON_PATH, CHILDREN_JSON_PATH
from apg223_rag.retrieval.vector_store import PineconeVectorStore
from apg223_rag.retrieval.keyword_search import BM25KeywordStore
from apg223_rag.graph.neo4j_graph import Neo4jGraphStore
from apg223_rag.retrieval.hybrid_retriever import HybridGraphRetriever
from apg223_rag.generation.generator import generate_answer
from apg223_rag.generation.evaluator import evaluate_response

# 15 Ground-Truth Benchmark Questions across APG 223
BENCHMARK_DATASET: List[Dict[str, Any]] = [
    {
        "id": "Q01",
        "category": "Governance & Board",
        "question": "How often must an ADI's residential mortgage risk management framework be independently reviewed under CPS 220?",
        "expected_subsection": "Oversight and review",
        "expected_paragraph": "Paragraph 12 (at least every three years)",
    },
    {
        "id": "Q02",
        "category": "Remuneration & Brokers",
        "question": "What controls should an ADI apply to mortgage broker commission structures under CPS 510?",
        "expected_subsection": "Remuneration",
        "expected_paragraph": "Paragraphs 17-20 (clawback provisions, monitoring broker defaults)",
    },
    {
        "id": "Q03",
        "category": "Serviceability (Multi-Hop)",
        "question": "What minimum interest rate buffer does Attachment C of APS 220 require, and how are honeymoon rates treated?",
        "expected_subsection": "Serviceability assessments",
        "expected_paragraph": "Paragraph 32 (at least 3.0 per cent, ignoring honeymoon rates)",
    },
    {
        "id": "Q04",
        "category": "Serviceability & Revolving Debt",
        "question": "What monthly repayment rate does APG 223 give as an example for assessing credit card or revolving debt limits?",
        "expected_subsection": "Serviceability assessments",
        "expected_paragraph": "Paragraph 35 (3 per cent per month on total committed limit)",
    },
    {
        "id": "Q05",
        "category": "Income & Living Expenses",
        "question": "What minimum haircut should be applied to non-salary and expected rental income, and how should HEM or HPI be used?",
        "expected_subsection": "Assessment and verification of income, living expenses and other debt commitments",
        "expected_paragraph": "Paragraphs 39, 43-44 (at least 20% discount; greater of declared expenses or scaled HEM/HPI)",
    },
    {
        "id": "Q06",
        "category": "Overrides & Capital",
        "question": "When is a loan classified as an override, and which prudential standard governs its capital treatment?",
        "expected_subsection": "Overrides",
        "expected_paragraph": "Paragraphs 47-52 (NIS < $0 or waivers; APS 112)",
    },
    {
        "id": "Q07",
        "category": "HELP Debt (Multi-Hop)",
        "question": "Under what condition can an ADI exclude HELP debt repayments from a serviceability assessment?",
        "expected_subsection": "Treatment of Higher Education Loan Program (HELP) debt obligations",
        "expected_paragraph": "Paragraph 55 (expected to pay off HELP debt within 12 months)",
    },
    {
        "id": "Q08",
        "category": "Third-Party Origination",
        "question": "What obligations apply under CPS 231 when income verification or lending authority is delegated to third parties?",
        "expected_subsection": "Other expectations with respect to loan origination by third parties",
        "expected_paragraph": "Paragraphs 56-59 (ADI retains risk/responsibility, periodic testing, timely cessation)",
    },
    {
        "id": "Q09",
        "category": "Scorecards & Models",
        "question": "Why should an ADI avoid relying solely on a scorecard vendor to validate its credit scoring model?",
        "expected_subsection": "Scorecard approvals",
        "expected_paragraph": "Paragraph 62 (material conflict of interest; use independent third-party assessor)",
    },
    {
        "id": "Q10",
        "category": "Specific Loan Types",
        "question": "What specific risk controls apply to reverse mortgages, including LVR caps and third-party marketing?",
        "expected_subsection": "Reverse mortgages",
        "expected_paragraph": "Paragraph 68 (actuarial advice, age-related LVR caps, cautioning against waiving legal advice)",
    },
    {
        "id": "Q11",
        "category": "SMSF Lending",
        "question": "What unique risks arise from residential mortgage loans to self-managed superannuation funds (SMSFs)?",
        "expected_subsection": "Loans to self-managed superannuation funds",
        "expected_paragraph": "Paragraph 70 (limited legal recourse on default, governance forum approval)",
    },
    {
        "id": "Q12",
        "category": "Valuation & LVR",
        "question": "What LVR threshold does APRA consider high LVR lending, and how should off-the-plan developer discounts be treated?",
        "expected_subsection": "Loan-to-valuation ratios",
        "expected_paragraph": "Paragraphs 81, 86 (LVR above 90%; developer discounts excluded from borrower deposit)",
    },
    {
        "id": "Q13",
        "category": "Guarantors",
        "question": "What checks should a prudent ADI perform when a loan relies on a parental guarantor to cover deposit shortfalls?",
        "expected_subsection": "Guarantors",
        "expected_paragraph": "Paragraph 88 (guarantor income, creditworthiness, enforceability, pledged collateral value)",
    },
    {
        "id": "Q14",
        "category": "Hardship & Collections",
        "question": "How must arrears and non-commercial hardship concessions be reported for prudential purposes?",
        "expected_subsection": "Hardship loans and collections",
        "expected_paragraph": "Paragraph 98 (do not freeze/re-age arrears; report non-commercial terms as restructured items)",
    },
    {
        "id": "Q15",
        "category": "Stress Testing & LMI",
        "question": "Can Lenders Mortgage Insurance (LMI) replace loan origination due diligence, and how is stress testing governed under APS 110?",
        "expected_subsection": "Lenders mortgage insurance",
        "expected_paragraph": "Paragraphs 99, 106-108 (LMI does not replace due diligence; APS 110 ICAAP stress testing)",
    },
]


def _clean_latin1(text: str) -> str:
    """Sanitizes Unicode punctuation so standard FPDF fonts render cleanly."""
    replacements = {
        "\u2013": "-", "\u2014": "--", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "*", "\u2265": ">=", "\u2264": "<=",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def export_benchmark_pdf(results: List[Dict[str, Any]], summary: Dict[str, Any], output_path: str):
    pdf = FPDF()
    pdf.set_left_margin(12)
    pdf.set_right_margin(12)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title Banner
    pdf.set_fill_color(24, 52, 94)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(12, 9)
    pdf.cell(pdf.epw, 8, "APG 223 Hybrid + Neo4j Graph RAG: Evaluation Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(12)
    pdf.cell(
        pdf.epw,
        6,
        f"15-Question Regulatory Retrieval & Faithfulness Benchmark | Hit@3 Recall: {summary['hit_at_3_pct']}%",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    # Summary Metrics
    pdf.set_text_color(20, 20, 20)
    pdf.set_xy(pdf.l_margin, 38)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(pdf.epw, 8, "1. Aggregate Performance Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        pdf.epw,
        6,
        (
            f"Total Questions: {summary['total_questions']}  |  "
            f"Retrieval Hit@3: {summary['hits']}/{summary['total_questions']} ({summary['hit_at_3_pct']}%)  |  "
            f"Avg Faithfulness: {summary['avg_faithfulness']}/5  |  "
            f"Avg Relevance: {summary['avg_relevance']}/5  |  "
            f"Avg Precision: {summary['avg_precision']}/5  |  "
            f"Avg Latency: {summary['avg_latency_s']}s"
        ),
    )
    pdf.ln(4)

    # Detailed Question-by-Question Breakdown
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(pdf.epw, 8, "2. Question-by-Question Retrieval & LLM Judge Audit", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for r in results:
        status = "PASS (HIT)" if r["retrieval_hit"] else "MISS"
        pdf.set_x(pdf.l_margin)
        pdf.set_fill_color(235, 243, 252) if r["retrieval_hit"] else pdf.set_fill_color(253, 236, 234)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(
            pdf.epw,
            7,
            _clean_latin1(
                f"{r['id']} [{r['category']}] - {status} | "
                f"F:{r['faithfulness']}/5 R:{r['answer_relevance']}/5 P:{r['context_precision']}/5 ({r['latency_s']}s)"
            ),
            new_x="LMARGIN",
            new_y="NEXT",
            fill=True,
        )

        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(pdf.epw, 5, _clean_latin1(f"Q: {r['question']}"), new_x="LMARGIN", new_y="NEXT")

        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.multi_cell(
            pdf.epw,
            4.5,
            _clean_latin1(f"Expected: {r['expected_subsection']} ({r['expected_paragraph']})"),
            new_x="LMARGIN",
            new_y="NEXT",
        )

        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            pdf.epw,
            4.5,
            _clean_latin1(f"Retrieved Parents: {', '.join(r['retrieved_parents'])}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )

        if r["graph_triples"]:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                pdf.epw,
                4.5,
                _clean_latin1(f"Active Graph Triples: {' ; '.join(r['graph_triples'][:3])}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )

        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 4.5, _clean_latin1(f"Answer: {r['answer']}"), new_x="LMARGIN", new_y="NEXT")

        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(
            pdf.epw,
            4.5,
            _clean_latin1(f"Judge Note: {r['judge_reasoning']}"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(3)

    pdf.output(output_path)

def run_benchmark():
    parents = json.loads(PARENTS_JSON_PATH.read_text())
    children = json.loads(CHILDREN_JSON_PATH.read_text())

    print("Initializing BGE model, Pinecone, BM25 & Neo4j once for 15-question benchmark...")
    bm25_store = BM25KeywordStore(children)
    vector_store = PineconeVectorStore()
    neo4j_store = Neo4jGraphStore()
    retriever = HybridGraphRetriever(parents, children, vector_store, bm25_store, neo4j_store)

    results = []
    try:
        for item in BENCHMARK_DATASET:
            t0 = time.perf_counter()
            bundle = retriever.retrieve(item["question"])
            answer = generate_answer(item["question"], bundle)
            scores = evaluate_response(item["question"], answer, bundle)
            elapsed = round(time.perf_counter() - t0, 2)

            retrieved_labels = [f"{p['parent_id']} ({p['subsection']})" for p in bundle["parents"]]
            expected_sub = item["expected_subsection"].lower()
            hit = any(expected_sub in p["subsection"].lower() for p in bundle["parents"])

            print(
                f"[{item['id']}] Hit@3={hit} | F={scores['faithfulness']} R={scores['answer_relevance']} "
                f"P={scores['context_precision']} | {elapsed}s"
            )

            results.append({
                **item,
                "retrieval_hit": hit,
                "retrieved_parents": retrieved_labels,
                "graph_triples": bundle["graph_triples"],
                "answer": answer.strip(),
                "faithfulness": scores["faithfulness"],
                "answer_relevance": scores["answer_relevance"],
                "context_precision": scores["context_precision"],
                "judge_reasoning": scores["reasoning"],
                "latency_s": elapsed,
            })
    finally:
        neo4j_store.close()

    total = len(results)
    hits = sum(1 for r in results if r["retrieval_hit"])
    summary = {
        "total_questions": total,
        "hits": hits,
        "hit_at_3_pct": round((hits / total) * 100, 1),
        "avg_faithfulness": round(sum(r["faithfulness"] for r in results) / total, 2),
        "avg_relevance": round(sum(r["answer_relevance"] for r in results) / total, 2),
        "avg_precision": round(sum(r["context_precision"] for r in results) / total, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / total, 2),
    }

    json_out = PROCESSED_DIR / "benchmark_results.json"
    pdf_out = PROCESSED_DIR / "apg223_benchmark_report.pdf"
    json_out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    export_benchmark_pdf(results, summary, str(pdf_out))

    print("\n=== BENCHMARK COMPLETE ===")
    print(json.dumps(summary, indent=2))
    print(f"PDF Report saved to: {pdf_out}")


if __name__ == "__main__":
    run_benchmark()