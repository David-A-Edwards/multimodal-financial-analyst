import streamlit as st
import os
import json
import shutil
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.embeddings.clip import ClipEmbedding
import qdrant_client

load_dotenv()

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def clear_chat_history():
    st.session_state.messages = []

# --- CONFIGURATION ---
st.set_page_config(page_title="Multimodal Financial Analyst", layout="wide")
st.title("📊 AI Financial Analyst (Dual-Store RAG)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        st.error("⚠️ LlamaCloud Key missing")
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Google API Key missing")
        
    model_name = st.selectbox(
        "Vision Model", 
        ["gemini-3-pro-preview", "gemini-2.5-flash", "gemini-2.5-pro"]
    )

# --- CORE LOGIC ---
@st.cache_resource
def get_multimodal_index(file_path):
    client = qdrant_client.QdrantClient(location=":memory:")
    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")
    storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

    parser = LlamaParse(
        result_type="markdown",
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        premium_mode=True,
        extract_image_block_types=["image", "chart", "diagram", "table"] 
    )

    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor=file_extractor
    ).load_data()

    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        image_embed_model=ClipEmbedding()
    )
    
    return index

# --- UI LOGIC ---
uploaded_file = st.file_uploader("Upload Report (PDF)", type=["pdf"])

if uploaded_file:
    with open("temp_report.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    with st.status("🧠 Building Dual-Store Index...", expanded=True):
        st.write("Extracting charts and tables...")
        index = get_multimodal_index("temp_report.pdf")
        st.write("✅ Indexing Complete.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Try to parse JSON outputs for better formatting
            if message["role"] == "assistant":
                try:
                    # Only attempt JSON parse if it looks like JSON (starts with { after cleanup)
                    content = message["content"].strip()
                    if content.startswith("```json"):
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif content.startswith("```"):
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    if content.startswith("{"):
                        data = json.loads(content)
                        st.json(data)
                    else:
                        st.markdown(message["content"])
                except (json.JSONDecodeError, IndexError):
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])
        
    query = st.chat_input("Ask about the visual data...", on_submit=clear_chat_history)
    
    if query:
        st.chat_message("user").write(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        mm_llm = GeminiMultiModal(model_name=model_name, api_key=os.getenv("GOOGLE_API_KEY"))
        Settings.multi_modal_llm = mm_llm
        
        query_engine = index.as_query_engine(
            similarity_top_k=10, 
            image_similarity_top_k=5
        )
        
        # --- UNIVERSAL ANALYST PROTOCOL (SOTA: Graph-Based + Multi-Path + Ranked Voting) ---
        universal_query = f"""
You are a Senior Data Analyst. Extract precise information by combining multiple SOTA techniques:
structural understanding, semantic matching, confidence voting, and multi-hop reasoning.

**TASK:** Answer: "{query}"

---

**PHASE 0: SEMANTIC DECOMPOSITION**
[Previous Phase 0: Query Intent Parsing, Semantic Field Mapping, Semantic Queries]

---

**PHASE 1: MENTAL MODEL + SPATIAL GRAPH CONSTRUCTION [SOTA: Graph-Based Reasoning]**

Building on spatial layout recognition, construct a semantic graph of the document:

1. **SPATIAL LAYOUT RECOGNITION:**
   [Previous spatial layout content]

2. **DOCUMENT STRUCTURE GRAPH (SOTA: Graph-Based Document Structure Analysis):**
   - Represent document as a graph: nodes = document elements (sections, tables, charts, text blocks)
   - Edges = logical/spatial relationships between elements
   - Types of edges:
     * Spatial: "left of", "above", "inside", "adjacent to"
     * Semantic: "references", "elaborates", "contradicts", "supports"
     * Temporal: "precedes", "follows"
   - Example:
     * Node: "POS CAGR bubble"
     * Edges: [right_of: "E-com CAGR bubble"], [located_in: "Saudi Arabia section"], [temporal_anchor: "2024-30 forecast"]
   - This creates a searchable structure for multi-hop reasoning

3. **SEMANTIC CONTENT MAPPING:**
   [Previous mapping + enhanced with graph context]

4. **VISUAL ENCODING PATTERNS:**
   [Previous patterns]

5. **CONSISTENCY CHECK:**
   [Previous consistency]

---

**PHASE 2: SEMANTIC MATCHING WITH MULTI-PATH RETRIEVAL**

6. **SEMANTIC FIELD EXPANSION (SOTA: Knowledge Graph-Aware Retrieval):**
   - Don't just list synonyms; construct semantic relationships:
     * "CAGR" → related_to: "growth rate", "annual growth", "trend", "forecast change"
     * "POS" → related_to: "point of sale", "retail", "physical", "in-store", "brick-and-mortar"
     * "market size" → related_to: "market value", "TAM", "addressable market", "revenue", "scale"
   - Create hierarchical semantic paths for multi-hop retrieval:
     * Path 1: CAGR → compound annual growth rate → annual change → metric
     * Path 2: CAGR → growth rate → trend → forecast
   - This enables fallback retrieval if primary path fails

7. **MULTI-PATH SEMANTIC MATCHING (SOTA: Knowledge Graph-Driven RAG):**
   - Try multiple semantic paths simultaneously:
     * Path A: Query "CAGR" → Search for exact label "CAGR"
     * Path B: Query "CAGR" → Search for synonym "growth rate"
     * Path C: Query "CAGR" → Search via graph edges (nodes related to "growth")
     * Path D: Query "CAGR" → Search in graph context (metrics near "market size" nodes)
   - Track which path succeeds and its confidence
   - Example output:
     * "Path A (exact match): FOUND - 'CAGR' label, confidence: HIGH"
     * "Path B (synonym): FOUND - 'growth rate' label, confidence: MEDIUM"
     * "Path C (graph relation): FOUND - via 'metric' relationship, confidence: MEDIUM"

8. **SEMANTIC CONFLICT RESOLUTION WITH GRAPH TRAVERSAL:**
   - When multiple candidates exist (e.g., "POS CAGR" and "E-com CAGR"):
     * Construct sub-graphs for each candidate
     * Sub-graph A: POS → related metrics, regions, temporal anchors
     * Sub-graph B: E-com → related metrics, regions, temporal anchors
     * Query asks for: POS → Sub-graph A is target → Select POS CAGR
   - Rate confidence based on how strongly candidate's sub-graph matches query context

---

**PHASE 2.5: CONFIDENCE ESTIMATION WITH MULTIPLE RANKING METHODS [SOTA: Ranked Voting Self-Consistency]**

**Before extraction, perform confidence assessment:**

9. **MULTI-DIMENSIONAL CONFIDENCE SCORING:**
   - Exact Match Confidence: Does label match query exactly? (HIGH=100%, MEDIUM=70%, LOW=30%)
   - Semantic Match Confidence: Does semantic field match? (synonyms, related terms)
   - Graph Proximity Confidence: How many edges separate query concept from found data? (close=HIGH, far=LOW)
   - Spatial Consistency: Does mental model prediction match found location?
   - Cross-Reference Validation: Do related data points support this value?
   
10. **RANKED VOTING FOR CONFLICTING CANDIDATES [SOTA: Ranked Voting Based Self-Consistency]:**
    - If multiple values are candidates for extraction:
      * Candidate A: "POS CAGR = 5%" | Score: exact_match(0.95) + semantic(0.9) + proximity(0.85) + spatial(0.9) + cross_ref(0.88) = 4.48/5
      * Candidate B: "E-com CAGR = 11%" | Score: exact_match(0.3) + semantic(0.4) + proximity(0.5) + spatial(0.4) + cross_ref(0.5) = 2.1/5
    - Ranked voting selects Candidate A (highest score)
    - Report ranking and why lower-ranked candidates were rejected

11. **SELF-CONSISTENCY SAMPLING (SOTA: Multiple Reasoning Paths):**
    - Generate 3 independent extraction attempts using different entry points:
      * Attempt 1: Start from query keywords
      * Attempt 2: Start from spatial layout prediction
      * Attempt 3: Start from semantic graph relationships
    - If all 3 attempts converge on same answer → confidence VERY HIGH
    - If attempts diverge → confidence MEDIUM, investigate conflicts

---

**PHASE 3: VALIDATE MENTAL MODEL + GRAPH AGAINST QUERY**
[Previous Phase 3]

---

**PHASE 4: UNIVERSAL VALIDATION PROTOCOLS + MULTI-HOP REASONING**

12. **SCOPE BOUNDARY:**
    [Previous]

13. **HIERARCHY VALIDATION:**
    [Previous]

14. **INTENT MATCHING:**
    [Previous]

15. **LABEL VERIFICATION WITH GRAPH NAVIGATION:**
    - Retrieve labels not just from direct match but via graph edges
    - Example: Query asks "What is the growth rate?" 
      * Direct label search: "growth rate" → Found
      * Graph navigation: "growth rate" → related_to "CAGR" → Found
    - Both paths valid; exact match has higher confidence

16. **VALUE-METRIC ALIGNMENT:**
    [Previous]

17. **CROSS-REFERENCE CONSISTENCY WITH MULTI-HOP REASONING [SOTA: Knowledge Graph Multi-Hop]:**
    - Check if value is consistent with related values across the document:
      * Direct: Is "POS CAGR = 5%" supported by neighboring data?
      * 1-hop: Does "POS market size" align with CAGR trend?
      * 2-hop: Does "total market" growth rate align with "POS market" growth rate?
    - Traverse graph edges to collect supporting evidence
    - Example: "POS CAGR = 5% is supported by (1) explicit label, (2) market size trend trend, (3) forecast period alignment"

18. **TEMPORAL ANCHORING WITH GRAPH CONTEXT:**
    [Previous + enhanced with graph temporal relationships]

---

**PHASE 5: STRUCTURED REASONING AND EXPLAINABILITY [SOTA: Graph-of-Thought Reasoning]**

19. **GENERATE REASONING PATH (Graph-of-Thought):**
    - Instead of linear chain-of-thought, generate a graph of reasoning steps:
      * Node 1: Query decomposed as [concept: CAGR, entity: POS, location: Saudi Arabia, period: 2024-30]
      * Node 2: Mental model invoked → predicts [region: left column, visual: bubble, label: POS CAGR]
      * Node 3: Semantic matching → finds [exact match: "POS CAGR" label]
      * Node 4: Graph traversal → confirms [POS node related to left column, temporal anchor 2024-30]
      * Node 5: Confidence voting → scores [5 dimensions, ranked voting = Candidate A wins]
      * Node 6: Cross-reference → validates [related metrics align with CAGR = 5%]
      * Edge A→B: Mental model used to guide search
      * Edge B→C: Prediction confirmed by semantic match
      * Edge C→D: Label leads to graph traversal
      * Edge D→E: Graph evidence used in voting
      * Edge E→F: Final validation via cross-references
    - This creates an explainable, verifiable reasoning path

---

**OUTPUT FORMAT:**

{{
  "query_semantics": {{
    "intent": "[...]",
    "key_entities": "[...]",
    "semantic_field": "[...]",
    "semantic_graph": "[Hierarchical relationships and multi-hop paths]"
  }},
  "document_structure": {{
    "spatial_layout": "[...]",
    "semantic_content_mapping": "[...]",
    "document_graph": "{{nodes: [POS_CAGR_bubble, E-com_CAGR_bubble, Saudi_Arabia_section, ...], edges: [left_of, temporal_anchor, ...]}}"
  }},
  "semantic_matching": {{
    "concept_matches": {{
      "CAGR": "{{exact: 'CAGR', paths: [Path A: exact label, Path B: synonym match, Path C: graph edge], confidence: 'HIGH'}}"
    }},
    "multi_path_results": {{
      "Path_A_exact": "FOUND - label 'CAGR', confidence: HIGH",
      "Path_B_synonym": "FOUND - label 'growth rate', confidence: MEDIUM",
      "Path_C_graph": "FOUND - via graph relationships, confidence: MEDIUM"
    }},
    "conflicts_detected": "[...]",
    "resolution_method": "[Ranked voting selected: ...]"
  }},
  "confidence_assessment": {{
    "multi_dimensional_scores": {{
      "exact_match": 0.95,
      "semantic_match": 0.90,
      "graph_proximity": 0.85,
      "spatial_consistency": 0.90,
      "cross_reference_validation": 0.88
    }},
    "overall_confidence_score": 0.896,
    "confidence_tier": "HIGH",
    "ranked_candidates": [
      "{{rank: 1, candidate: 'POS CAGR = 5%', score: 4.48/5}}",
      "{{rank: 2, candidate: 'E-com CAGR = 11%', score: 2.1/5}}"
    ],
    "self_consistency_results": "{{Attempt 1: 5%, Attempt 2: 5%, Attempt 3: 5% → Convergence: YES, confidence boost}}"
  }},
  "reasoning_path": {{
    "graph_of_thought": [
      "{{node: Query_Decomposition, output: [CAGR, POS, Saudi Arabia, 2024-30]}}",
      "{{node: Mental_Model_Prediction, output: [left column, bubble visual, POS CAGR label]}}",
      "{{node: Semantic_Matching, output: exact match found}}",
      "{{node: Graph_Traversal, output: confirmed via document structure graph}}",
      "{{node: Confidence_Voting, output: POS CAGR ranked highest}}",
      "{{node: Cross_Reference_Validation, output: related metrics align}}"
    ],
    "edges": ["Query→Model", "Model→Semantic", "Semantic→Graph", "Graph→Voting", "Voting→Validation"]
  }},
  "answer": "5%",
  "confidence": "VERY HIGH (0.896 multi-dimensional score, self-consistency confirmed)",
  "protocol_checklist": {{
    "protocol_12_scope_boundary": "PASS",
    "protocol_13_hierarchy_validation": "PASS",
    "protocol_14_intent_matching": "PASS",
    "protocol_15_label_verification_with_graph": "PASS - Found via Path A (exact match)",
    "protocol_16_value_metric_alignment": "PASS",
    "protocol_17_cross_reference_multi_hop": "PASS - 2-hop validation confirms",
    "protocol_18_temporal_anchoring": "PASS"
  }},
  "multi_hop_validation": {{
    "direct": "POS CAGR bubble labeled 5%",
    "1_hop": "POS market size trend aligns with 5% growth",
    "2_hop": "Total market forecasts align with POS segment trend",
    "supporting_evidence": 3
  }},
  "labels_found": "[...]",
  "value_source": "[...]",
  "ambiguities_or_caveats": "[...]"
}}

**CRITICAL ANTI-HALLUCINATION RULES:**
- **Rule A:** Use ranked voting for conflicts. Never arbitrarily choose.
- **Rule B:** Self-consistency check: If 3 extraction attempts diverge, confidence = MEDIUM/LOW, investigate.
- **Rule C:** Multi-hop validation required for high-stakes extractions. State 1-hop and 2-hop supporting evidence.
- **Rule D:** Always report all candidate scores, not just the winner. Transparency enables downstream verification.
- **Rule E:** If graph traversal finds multiple paths to answer, report all paths and confidence for each.

**DEBUGGING CHECKLIST:**
1. What does the query ask for semantically?
2. What semantic paths exist to this answer?
3. Which path has strongest confidence (exact match > synonym > inference)?
4. Are there competing candidates? Use ranked voting.
5. Does answer make sense via 1-hop and 2-hop validation?
6. Did 3 self-consistency attempts converge?
7. Final confidence: Multi-dimensional score + convergence check.
"""

        with st.spinner("Analyzing charts and tables..."):
            response = query_engine.query(universal_query)
        
        with st.expander("👁️ Visual Evidence Used"):
            if hasattr(response, 'metadata') and 'image_nodes' in response.metadata:
                for img_node in response.metadata['image_nodes']:
                    st.write("Found relevant chart:")
                    if hasattr(img_node, 'image_path'):
                         st.image(img_node.image_path)
        
        # Parse and display JSON response nicely
        try:
            # Clean up the response text to ensure it's valid JSON
            json_str = str(response).strip()
            if json_str.startswith("```json"):
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif json_str.startswith("```"):
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            # Only attempt JSON parse if it looks like JSON
            if json_str.startswith("{"):
                data = json.loads(json_str)
                st.chat_message("assistant").json(data)
                st.session_state.messages.append({"role": "assistant", "content": json.dumps(data)})
            else:
                # Response is plain text, not JSON
                st.chat_message("assistant").write(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
        except (json.JSONDecodeError, IndexError):
            # Fallback if the LLM didn't output perfect JSON
            st.chat_message("assistant").write(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})