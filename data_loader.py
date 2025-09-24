import json
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional

ES_CLOUD_ID = "sih:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ0NDViZTM3OGE2OTE0ZDJlOTkwM2IzNjcwNzVmMmRlNCQzOTlhNjgxNDcwM2Q0M2Q1YjFiOTVlNjVkMDc5ODE5Zg=="
ES_USERNAME = "elastic"
ES_PASSWORD = "WK44MbgScextyOUYsLa7GbWV"

class ElasticGroundwaterSearch:
    def __init__(self, json_file: str, index_name: str = "groundwater", embedding_model: str = "all-MiniLM-L6-v2"):
        self.json_file = json_file
        self.index_name = index_name
        self.es = Elasticsearch(
            cloud_id=ES_CLOUD_ID,
            basic_auth=(ES_USERNAME, ES_PASSWORD)
        )
        self.model = SentenceTransformer(embedding_model)
        # Define invalid block names that should be excluded
        self.invalid_blocks = {
            "total", "district_total", "state_total", "",
            "TOTAL", "Total", "District Total", "STATE_TOTAL",
            "district total", "state total", "grand total",
            "GRAND_TOTAL", "DISTRICT_TOTAL"
        }
        self._create_index()
        self._load_data()

    def _create_index(self):
        if self.es.indices.exists(index=self.index_name):
            return

        index_config = {
            "mappings": {
                "properties": {
                    "state_name": {"type": "keyword"},
                    "district_name": {"type": "keyword"},
                    "block_name": {"type": "keyword"},
                    "total_gw_availability_ham": {"type": "float"},
                    "net_gw_availability_ham": {"type": "float"},
                    "stage_of_development_percent": {"type": "float"},
                    "rainfall_total_mm": {"type": "float"},
                    "total_category": {"type": "keyword"},
                    "combined_text": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
                    "unique_id": {"type": "keyword"},
                    "is_individual_block": {"type": "boolean"}  # Flag to identify individual blocks
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=index_config)
        print(f"✅ Created index: {self.index_name}")

    def _is_valid_block(self, record: Dict[str, Any]) -> bool:
        """
        Check if a record represents an individual block (not a summary/total)
        """
        block_name = str(record.get('block_name', '')).strip()
        district_name = str(record.get('district_name', '')).strip()
        state_name = str(record.get('state_name', '')).strip()

        # Skip if block name is in invalid list
        if block_name.lower() in [name.lower() for name in self.invalid_blocks]:
            return False

        # Skip if block name is empty or None
        if not block_name or block_name.lower() == 'none' or block_name.lower() == 'null':
            return False

        # Skip if block name is same as district name (likely a district total)
        if block_name.lower() == district_name.lower():
            return False

        # Skip if it's a state-level record (state = district)
        if state_name.lower() == district_name.lower():
            return False

        return True

    def _load_data(self):
        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        actions = []
        seen_records = set()
        valid_blocks = 0
        skipped_blocks = 0

        for rec in data:
            # Check if this is a valid individual block
            if not self._is_valid_block(rec):
                skipped_blocks += 1
                block_name = rec.get('block_name', 'EMPTY')
                district_name = rec.get('district_name', 'UNKNOWN')
                print(f"⚠️ Skipping summary/invalid record: {district_name} > {block_name}")
                continue

            # Create unique identifier
            unique_key = f"{rec.get('state_name', '')}_{rec.get('district_name', '')}_{rec.get('block_name', '')}"

            # Skip if we've already seen this combination
            if unique_key in seen_records:
                print(f"⚠️ Skipping duplicate: {unique_key}")
                skipped_blocks += 1
                continue
            seen_records.add(unique_key)

            # Add flag to identify this as an individual block
            rec["is_individual_block"] = True

            # Build combined text with better formatting
            parts = [
                f"State: {rec.get('state_name', 'N/A')}",
                f"District: {rec.get('district_name', 'N/A')}",
                f"Block: {rec.get('block_name', 'N/A')}",
                f"Total GW Availability: {rec.get('total_gw_availability_ham', 'N/A')} HAM",
                f"Net GW Availability: {rec.get('net_gw_availability_ham', 'N/A')} HAM",
                f"Development Stage: {rec.get('stage_of_development_percent', 'N/A')}%",
                f"Rainfall: {rec.get('rainfall_total_mm', 'N/A')} mm",
                f"Category: {rec.get('total_category', 'N/A')}"
            ]
            rec["combined_text"] = " | ".join(parts)
            rec["unique_id"] = unique_key

            # Generate embedding
            emb = self.model.encode(rec["combined_text"], convert_to_numpy=True, normalize_embeddings=True)
            rec["embedding"] = emb.tolist()

            actions.append({
                "_index": self.index_name,
                "_id": unique_key,
                "_source": rec
            })
            valid_blocks += 1

        helpers.bulk(self.es, actions)
        print(f"✅ Indexed {valid_blocks} valid individual blocks")
        print(f"⚠️ Skipped {skipped_blocks} summary/invalid records")

    def _get_base_individual_blocks_filter(self):
        """
        Base filter to ensure we only get individual blocks in searches
        """
        return {
            "bool": {
                "must": [
                    {"term": {"is_individual_block": True}}
                ],
                "must_not": [
                    {"terms": {"block_name": list(self.invalid_blocks)}},
                    {"term": {"block_name": ""}},
                ]
            }
        }

    def search(self, query: str, top_k: int = 10, min_score: float = 0.1, district_name: str = None):
        """
        Improved hybrid search with better scoring and deduplication - ONLY individual blocks

        Args:
            query: Search query string
            top_k: Number of results to return
            min_score: Minimum relevance score threshold
            district_name: Optional district filter for targeted search
        """
        query_vector = self.model.encode([query], convert_to_numpy=True)[0].tolist()

        # Build must clauses
        must_clauses = [self._get_base_individual_blocks_filter()]

        # Add district filter if provided
        if district_name:
            must_clauses.append({"term": {"district_name.keyword": district_name.upper()}})

        body = {
            "size": top_k * 2,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "state_name^3",
                                    "district_name^3",
                                    "block_name^2",
                                    "total_category^2",
                                    "combined_text^1"
                                ],
                                "type": "best_fields",
                                "boost": 2.5  # Emphasize text matching more
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    # Improved vector similarity scoring with better normalization
                                    "source": """
                                        double similarity = cosineSimilarity(params.query_vector, 'embedding');
                                        return Math.max(0, (similarity + 1.0) / 2.0 * 5.0);
                                    """,
                                    "params": {"query_vector": query_vector}
                                },
                                "boost": 1.5  # Less emphasis on semantic similarity for now
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "min_score": min_score,
            "_source": {
                "excludes": ["embedding"]
            }
        }

        res = self.es.search(index=self.index_name, body=body)
        hits = res["hits"]["hits"]

        # Post-process results
        unique_results = []
        seen_locations = set()

        for hit in hits:
            source = hit["_source"]
            location_key = f"{source.get('state_name', '')}_{source.get('district_name', '')}_{source.get('block_name', '')}"

            if location_key in seen_locations:
                continue
            seen_locations.add(location_key)

            source["_relevance_score"] = hit["_score"]
            unique_results.append(source)

            if len(unique_results) >= top_k:
                break

        return unique_results

    def detect_district_in_query(self, query: str) -> str:
        """
        Extract district name from query using comprehensive district list

        Returns:
            District name in uppercase if found, None otherwise
        """
        query_lower = query.lower()

        # Comprehensive list of Rajasthan districts (including common variations)
        rajasthan_districts = {
            'ajmer': 'AJMER',
            'jaipur': 'JAIPUR',
            'jodhpur': 'JODHPUR',
            'udaipur': 'UDAIPUR',
            'kota': 'KOTA',
            'bikaner': 'BIKANER',
            'alwar': 'ALWAR',
            'bharatpur': 'BHARATPUR',
            'pali': 'PALI',
            'sikar': 'SIKAR',
            'churu': 'CHURU',
            'ganganagar': 'GANGANAGAR',
            'sri ganganagar': 'GANGANAGAR',
            'sriganganagar': 'GANGANAGAR',
            'hanumangarh': 'HANUMANGARH',
            'jhunjhunu': 'JHUNJHUNU',
            'nagaur': 'NAGAUR',
            'barmer': 'BARMER',
            'jaisalmer': 'JAISALMER',
            'jalore': 'JALORE',
            'sirohi': 'SIROHI',
            'dungarpur': 'DUNGARPUR',
            'banswara': 'BANSWARA',
            'chittaurgarh': 'CHITTAURGARH',
            'chittorgarh': 'CHITTAURGARH',
            'rajsamand': 'RAJSAMAND',
            'bhilwara': 'BHILWARA',
            'tonk': 'TONK',
            'sawai madhopur': 'SAWAI MADHOPUR',
            'sawaimadhopur': 'SAWAI MADHOPUR',
            'karauli': 'KARAULI',
            'dholpur': 'DHOLPUR',
            'dausa': 'DAUSA',
            'pratapgarh': 'PRATAPGARH',
            'bundi': 'BUNDI',
            'jhalawar': 'JHALAWAR',
            'baran': 'BARAN'
        }

        # Check for district mentions in query
        for district_key, district_name in rajasthan_districts.items():
            if f" {district_key} " in f" {query_lower} " or query_lower.startswith(district_key) or query_lower.endswith(district_key):
                return district_name

        return None

    def search_parameter_range(self, parameter: str, min_value: float = None, max_value: float = None,
                             top_k: int = None, sort_order: str = "desc", location_filters: Dict[str, str] = None):
        """
        Generic search for blocks within a specific parameter range with optional location filtering

        Args:
            parameter: Field name (rainfall_total_mm, total_gw_availability_ham, etc.)
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            top_k: Number of results (None = all matching results)
            sort_order: "desc" for highest first, "asc" for lowest first
            location_filters: Dict with optional district_name/state_name filters
        """
        # Build must clauses
        must_clauses = [self._get_base_individual_blocks_filter()]

        # Add parameter range filter
        range_query = {}
        if min_value is not None:
            range_query["gte"] = min_value
        if max_value is not None:
            range_query["lte"] = max_value

        if range_query:  # Only add if there are range conditions
            must_clauses.append({"range": {parameter: range_query}})

        # Add location filters if provided
        if location_filters:
            for field, value in location_filters.items():
                if value:
                    # Use exact term matching for location fields
                    must_clauses.append({"term": {f"{field}.keyword": value}})

        body = {
            "size": top_k if top_k else 10000,  # Large number if no limit specified
            "query": {"bool": {"must": must_clauses}},
            "sort": [{parameter: {"order": sort_order}}],
            "_source": {"excludes": ["embedding"]}
        }

        res = self.es.search(index=self.index_name, body=body)
        hits = res["hits"]["hits"]

        results = []
        seen_blocks = set()  # Prevent duplicates

        for hit in hits:
            source = hit["_source"]
            # Create unique identifier for deduplication
            block_id = f"{source.get('district_name', '')}_{source.get('block_name', '')}"

            if block_id not in seen_blocks:
                seen_blocks.add(block_id)
                # Verify the parameter value meets our criteria (safety check)
                param_value = source.get(parameter, 0)
                if min_value is not None and param_value < min_value:
                    continue
                if max_value is not None and param_value > max_value:
                    continue

                results.append(source)

        return results

    def search_top_blocks_by_parameter(self, parameter: str, top_k: int, highest: bool = True,
                                     location_filters: Dict[str, str] = None):
        """
        Get top N blocks by any parameter (highest or lowest values) with optional location filtering

        Args:
            parameter: Field name to sort by
            top_k: Number of results to return
            highest: True for highest values, False for lowest values
            location_filters: Dict with optional district_name/state_name filters
        """
        sort_order = "desc" if highest else "asc"
        return self.search_parameter_range(parameter, top_k=top_k, sort_order=sort_order,
                                         location_filters=location_filters)

    def semantic_search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.7):
        """
        Pure semantic search using only vector similarity - ONLY individual blocks
        """
        query_vector = self.model.encode([query], convert_to_numpy=True)[0].tolist()

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        self._get_base_individual_blocks_filter(),
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding')",
                                    "params": {"query_vector": query_vector}
                                }
                            }
                        }
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]}
        }

        res = self.es.search(index=self.index_name, body=body)
        hits = res["hits"]["hits"]

        filtered_results = []
        for hit in hits:
            similarity = hit["_score"]
            if similarity >= similarity_threshold:
                source = hit["_source"]
                source["_similarity_score"] = similarity
                filtered_results.append(source)

        return filtered_results

    def keyword_search(self, query: str, top_k: int = 10):
        """
        Pure keyword search using BM25 - ONLY individual blocks
        """
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        self._get_base_individual_blocks_filter(),
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "state_name^3",
                                    "district_name^3",
                                    "block_name^2",
                                    "total_category^2",
                                    "combined_text"
                                ],
                                "type": "best_fields"
                            }
                        }
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]}
        }

        res = self.es.search(index=self.index_name, body=body)
        hits = res["hits"]["hits"]
        return [hit["_source"] for hit in hits]

    def advanced_search(self, filters: Dict[str, Any], top_k: int = 10):
        """
        Advanced search with multiple filters - ONLY individual blocks
        """
        must_clauses = [self._get_base_individual_blocks_filter()]

        # Text-based filters
        for field in ["state_name", "district_name", "block_name", "total_category"]:
            if field in filters and filters[field]:
                must_clauses.append({
                    "match": {field: filters[field]}
                })

        # Numeric range filters
        numeric_fields = [
            "total_gw_availability_ham",
            "net_gw_availability_ham",
            "stage_of_development_percent",
            "rainfall_total_mm"
        ]

        for field in numeric_fields:
            if f"{field}_min" in filters or f"{field}_max" in filters:
                range_query = {}
                if f"{field}_min" in filters:
                    range_query["gte"] = filters[f"{field}_min"]
                if f"{field}_max" in filters:
                    range_query["lte"] = filters[f"{field}_max"]

                must_clauses.append({
                    "range": {field: range_query}
                })

        body = {
            "size": top_k,
            "query": {"bool": {"must": must_clauses}},
            "_source": {"excludes": ["embedding"]}
        }

        res = self.es.search(index=self.index_name, body=body)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def get_district_blocks(self, district_name: str) -> List[Dict[str, Any]]:
        body = {
            "size": 100,
            "query": {
                "bool": {
                    "must": [
                        self._get_base_individual_blocks_filter(),
                        {"match": {"district_name": district_name}}  # Use match instead of term
                    ]
                }
            },
            "sort": [{"block_name": {"order": "asc"}}],
            "_source": {"excludes": ["embedding"]}
        }

        res = self.es.search(index=self.index_name, body=body)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def rainfall_stats(self, district: str = None, state: str = None, individual_blocks_only: bool = True):
        """
        Enhanced rainfall statistics - can include or exclude summary records
        """
        must_clauses = []

        if individual_blocks_only:
            must_clauses.append(self._get_base_individual_blocks_filter())

        if district:
            must_clauses.append({"term": {"district_name.keyword": district}})
        if state:
            must_clauses.append({"term": {"state_name.keyword": state}})

        body = {
            "size": 0,
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}]
                }
            },
            "aggs": {
                "rainfall_stats": {
                    "stats": {"field": "rainfall_total_mm"}
                },
                "rainfall_histogram": {
                    "histogram": {
                        "field": "rainfall_total_mm",
                        "interval": 100
                    }
                }
            }
        }

        res = self.es.search(index=self.index_name, body=body)
        stats = res["aggregations"]["rainfall_stats"]

        return {
            "count": stats["count"],
            "highest": stats["max"],
            "lowest": stats["min"],
            "average": stats["avg"],
            "sum": stats["sum"],
            "histogram": res["aggregations"]["rainfall_histogram"]["buckets"],
            "individual_blocks_only": individual_blocks_only
        }

    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Check for data quality issues
        """
        # Count individual blocks
        individual_body = {
            "size": 0,
            "query": self._get_base_individual_blocks_filter(),
            "aggs": {
                "individual_blocks": {"value_count": {"field": "unique_id"}}
            }
        }
        individual_res = self.es.search(index=self.index_name, body=individual_body)

        # Count all documents
        total_body = {
            "size": 0,
            "query": {"match_all": {}},
            "aggs": {
                "total_docs": {"value_count": {"field": "unique_id"}},
                "missing_state": {"missing": {"field": "state_name"}},
                "missing_district": {"missing": {"field": "district_name"}},
                "missing_block": {"missing": {"field": "block_name"}},
            }
        }
        total_res = self.es.search(index=self.index_name, body=total_body)
        total_aggs = total_res["aggregations"]

        individual_count = individual_res["aggregations"]["individual_blocks"]["value"]
        total_count = total_aggs["total_docs"]["value"]

        return {
            "total_documents": total_count,
            "individual_blocks": individual_count,
            "summary_records": total_count - individual_count,
            "missing_state_count": total_aggs["missing_state"]["doc_count"],
            "missing_district_count": total_aggs["missing_district"]["doc_count"],
            "missing_block_count": total_aggs["missing_block"]["doc_count"],
        }

# Example usage and testing functions
def test_searches(searcher: ElasticGroundwaterSearch):
    """Test different search methods with individual blocks only"""
    print("\n=== Testing High Rainfall Blocks (>800mm) ===")
    high_rainfall = searcher.search_high_rainfall_blocks(min_rainfall=800, top_k=10)
    print(f"Found {len(high_rainfall)} blocks with >800mm rainfall:")
    for i, result in enumerate(high_rainfall[:5], 1):
        print(f"{i}. {result['district_name']} > {result['block_name']}: {result.get('rainfall_total_mm', 'N/A')} mm")

    print("\n=== Testing Hybrid Search ===")
    results = searcher.search("high rainfall", top_k=5)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['district_name']} > {result['block_name']}")
        print(f"   Rainfall: {result.get('rainfall_total_mm', 'N/A')} mm")

    print("\n=== Data Quality Report ===")
    quality = searcher.validate_data_quality()
    print(json.dumps(quality, indent=2))

if __name__ == "__main__":
    searcher = ElasticGroundwaterSearch("rajasthan_groundwater_blocks.json")
    test_searches(searcher)
