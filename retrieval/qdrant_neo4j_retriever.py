from qdrant_client import QdrantClient
from extract_entity_relation.models.llama import test_ollama_llama
import json
from create_neo4j_graph.utils.qdrant import llama_embeddings

from rapidfuzz import fuzz

def get_user_keywords(
    user_query,
    prompt_template,
    model_name,
    output_path=None,
    qdrant_client: QdrantClient=None,
    collection_name="Parents",
    threshold=50
):
    # 1. Build prompt
    complete_prompt = prompt_template.format(query=user_query)
    print("\n=== Preparing prompt for LLM ===")
    print(f"User Query: {user_query}")
    print(f"Complete Prompt: {complete_prompt}")

    # 2. Get LLM response
    response = test_ollama_llama(
        complete_prompt=complete_prompt,
        model_name=model_name,
        output_path=output_path
    )
    if not response:
        print("LLM response was empty.")
        return None

    # 3. Clean and parse the response
    response = response.strip().strip("```").strip()
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "keywords" in parsed:
            keywords = [kw.strip().lower() for kw in parsed["keywords"]]
        elif isinstance(parsed, list):
            keywords = [kw.strip().lower() for kw in parsed]
        else:
            raise ValueError("Unexpected JSON format.")
    except Exception as e:
        print(f"Response JSON parse error: {e}")
        return None

    print(f"\nExtracted Keywords: {keywords}")

    if not qdrant_client or not collection_name:
        print("No Qdrant client provided, returning only keywords.")
        return keywords

    # 4. Fuzzy search inside payload
    print("\n=== Fuzzy searching in Qdrant payloads ===")
    matched_results = []
    try:
        scroll_result, _ = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100
        )

        print(f"Checking {len(scroll_result)} parent records for fuzzy matches...")
        for point in scroll_result:
            payload = point.payload
            payload_keywords = [kw.lower() for kw in payload.get("keywords", [])]
            
            # Debug: Show what we're comparing
            if len(matched_results) < 3:  # Show details for first few records
                print(f"Record {point.id}: payload_keywords = {payload_keywords}")
                print(f"  vs extracted keywords = {keywords}")
            
            # Fuzzy match check
            best_match_score = 0
            best_match_keyword = ""
            for kw in keywords:
                for pk in payload_keywords:
                    score = fuzz.ratio(kw, pk)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_keyword = f"{kw} vs {pk}"
            
            if len(matched_results) < 3:  # Show scores for first few records
                print(f"  Best match score: {best_match_score} ({best_match_keyword})")
            
            if best_match_score >= threshold:
                matched_results.append(point)
                print(f"  ✅ MATCH! Score: {best_match_score}")

    except Exception as e:
        print(f"Qdrant payload search error: {e}")

    print(f"\nFound {len(matched_results)} matching records.")
    for res in matched_results[:5]:
        print(f"- Title: {res.payload.get('title')}")
        print(f"  DOI: {res.payload.get('doi')}")
        print(f"  Keywords: {res.payload.get('keywords')}\n")

    return matched_results


def retriever_search_with_parent(
    neo4j_driver: None,
    qdrant_client: QdrantClient,
    parent_collection,
    children_collection,
    query,
    prompt_template,
    model_name,
    evidence_mapping_path=None
):
    print(f"Searching for: {query}")

    # Load evidence mapping if provided
    evidence_mapping = {}
    uuid_to_mapping_number = {}
    if evidence_mapping_path:
        try:
            with open(evidence_mapping_path, 'r', encoding='utf-8') as f:
                evidence_mapping = json.load(f)
            # Create reverse mapping: UUID -> mapping_number
            # evidence_mapping format: {"uuid": "evidence_text"} but we need UUID -> mapping_number
            # The mapping file actually has UUID as key and evidence text as value
            # We need to find the mapping number for each UUID
            mapping_number = 1
            for uuid_key, evidence_text in evidence_mapping.items():
                uuid_to_mapping_number[uuid_key] = str(mapping_number)
                mapping_number += 1
            print(f"Loaded evidence mapping with {len(uuid_to_mapping_number)} UUID mappings")
        except Exception as e:
            print(f"Warning: Could not load evidence mapping: {e}")

    # Step 1: User query embedding
    # 'llama_embeddings' fonksiyonunun bu bağlamda mevcut olduğunu varsayıyoruz.
    query_embedding = llama_embeddings(query)
    if query_embedding is None:
        print("Failed to generate embedding")
        return []

    # Step 2: Parent search (inside payload)
    parent_results = get_user_keywords(
        user_query=query,
        prompt_template=prompt_template,
        model_name=model_name,
        qdrant_client=qdrant_client,
        collection_name=parent_collection
    )

    if not parent_results:
        print("No parent records found with fuzzy search, trying direct vector search...")
        # Fallback: Direct vector search without keyword filtering
        try:
            parent_search_result = qdrant_client.search(
                collection_name=parent_collection,
                query_vector=query_embedding,
                limit=5,
                with_payload=True,
                score_threshold=0.4
            )
            parent_results = parent_search_result if parent_search_result else []
            print(f"Fallback search found {len(parent_results)} parent records")
        except Exception as e:
            print(f"Fallback search failed: {e}")
            return []

    parent_ids = [p.payload.get("parent_id") for p in parent_results if p.payload.get("parent_id") is not None]
    parent_ids = list(set(parent_ids))  # Remove duplicates
    print(f"Found {len(parent_ids)} unique parent_ids from search: {parent_ids}")

    # Step 3: Children search (parent_id filter + vector search)
    children_results = []
    for pid in parent_ids:
        # Qdrant filter
        filter_payload = {
            "must": [{"key": "parent_id", "match": {"value": pid}}]
        }

        # Qdrant search
        search_result = qdrant_client.search(
            collection_name=children_collection,
            query_vector=query_embedding,
            query_filter=filter_payload,
            limit=20,  # 20 results per parent
            with_payload=True,
            score_threshold=0.3  # lower threshold
        )

        # Qdrant 1.7.0+ usually returns a list directly; keep this check for backward compatibility
        if isinstance(search_result, tuple):
            results, _ = search_result
        else:
            results = search_result
            
        children_results.extend(results)

    if not children_results:
        print("No children records found")
        return []

    # Step 4: Extract evidence text from Qdrant results and find matching UUIDs in evidence mapping
    print(f"Found {len(children_results)} children Qdrant results.")
    
    # Extract evidence texts from Qdrant results
    qdrant_evidence_texts = []
    for child in children_results:
        evidence_text = child.payload.get("evidence", "").strip()
        if evidence_text:
            qdrant_evidence_texts.append(evidence_text)
    
    print(f"Extracted {len(qdrant_evidence_texts)} evidence texts from Qdrant results.")
    print(f"Sample evidence texts: {qdrant_evidence_texts[:2]}")
    
    # Find matching UUIDs in evidence mapping for these evidence texts
    neo4j_search_uuids = []
    for evidence_text in qdrant_evidence_texts:
        # Look for this evidence text in the evidence mapping
        for uuid_key, mapped_text in evidence_mapping.items():
            if evidence_text.strip() == mapped_text.strip():
                neo4j_search_uuids.append(uuid_key)
                break
    
    print(f"Found {len(neo4j_search_uuids)} matching UUIDs in evidence mapping.")
    print(f"Sample matching UUIDs: {neo4j_search_uuids[:3]}")

    # Step 5: Neo4j search using evidence_ids (UUIDs from evidence mapping)
    with neo4j_driver.session() as session:
        if not neo4j_search_uuids:
            print("No matching UUIDs found in evidence mapping, skipping Neo4j search")
            graph_nodes = []
        else:
            # Neo4j query: any item in $point_ids must match any element in n.evidence_sentences.
            neo4j_query = """
            MATCH (n)
            WHERE n.evidence_sentences IS NOT NULL
            AND ANY(eid IN $point_ids WHERE eid IN n.evidence_sentences)
            RETURN n, n.name as name, n.evidence_sentences as evidence_sentences
            """
            neo4j_result = session.run(neo4j_query, point_ids=neo4j_search_uuids)
            
            graph_nodes = []
            for record in neo4j_result:
                graph_nodes.append({
                    "node": dict(record["n"]), 
                    "name": record["name"],
                    "evidence_sentences": record["evidence_sentences"]
                })

    print(f"Found {len(graph_nodes)} related graph nodes in Neo4j.")

    # Step 6: Combine Children results with Neo4j nodes using evidence text matching
    combined_results = []
    
    # Create a mapping from evidence text to Neo4j nodes
    # {evidence_text: [graph_node1, graph_node2, ...]}
    evidence_to_graph_nodes = {}
    for graph_node in graph_nodes:
        # For each evidence sentence UUID in this node, find the corresponding text
        for evidence_uuid in graph_node["evidence_sentences"]:
            evidence_uuid_str = str(evidence_uuid)
            if evidence_uuid_str in evidence_mapping:
                evidence_text = evidence_mapping[evidence_uuid_str]
                if evidence_text not in evidence_to_graph_nodes:
                    evidence_to_graph_nodes[evidence_text] = []
                evidence_to_graph_nodes[evidence_text].append(graph_node)

    # Get parent information for all parent_ids
    parent_info = {}
    for pid in parent_ids:
        try:
            parent_search = qdrant_client.search(
                collection_name=parent_collection,
                query_vector=[0.0] * 4096,  # Dummy vector since we're filtering
                query_filter={"must": [{"key": "parent_id", "match": {"value": pid}}]},
                limit=1,
                with_payload=True
            )
            if parent_search:
                parent_info[pid] = parent_search[0].payload
        except Exception as e:
            print(f"Warning: Could not fetch parent info for parent_id {pid}: {e}")

    for child in children_results:
        # Get parent information
        parent_id = child.payload.get("parent_id")
        parent_payload = parent_info.get(parent_id, {})
        
        # Get child evidence sentence
        child_evidence_text = child.payload.get("evidence", "").strip()
        
        # Find matching graph nodes and their evidence sentences
        related_graph_nodes_with_evidence = []
        if child_evidence_text in evidence_to_graph_nodes:
            for graph_node in evidence_to_graph_nodes[child_evidence_text]:
                # Get evidence sentences for this graph node
                graph_evidence_sentences = []
                for evidence_uuid in graph_node["evidence_sentences"]:
                    evidence_uuid_str = str(evidence_uuid)
                    if evidence_uuid_str in evidence_mapping:
                        evidence_text = evidence_mapping[evidence_uuid_str]
                        graph_evidence_sentences.append({
                            "uuid": evidence_uuid_str,
                            "text": evidence_text
                        })
                
                # Enhanced graph node with evidence sentences
                enhanced_graph_node = {
                    "name": graph_node["name"],
                    "node": graph_node["node"],
                    "evidence_sentences": graph_node["evidence_sentences"],
                    "evidence_sentences_text": graph_evidence_sentences
                }
                related_graph_nodes_with_evidence.append(enhanced_graph_node)

        result_item = {
            "qdrant_score": child.score,
            "qdrant_payload": child.payload,
            "qdrant_id": str(child.id),
            "parent_payload": parent_payload,
            "child_evidence_sentence": child_evidence_text,
            "related_graph_nodes": related_graph_nodes_with_evidence
        }

        combined_results.append(result_item)
        
    print(f"Combined {len(combined_results)} Qdrant results with {sum(len(item['related_graph_nodes']) for item in combined_results)} graph links.")


    return combined_results