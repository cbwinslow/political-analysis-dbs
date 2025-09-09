from neo4j import GraphDatabase
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class GraphService:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "political123")
            )
        )
    
    def close(self):
        self.driver.close()
    
    async def create_legislator_node(self, legislator_data: Dict[str, Any]):
        """Create a legislator node in Neo4j"""
        with self.driver.session() as session:
            query = """
            MERGE (l:Legislator {id: $id})
            SET l.name = $name,
                l.party = $party,
                l.state = $state,
                l.district = $district,
                l.chamber = $chamber,
                l.created_at = datetime()
            RETURN l
            """
            result = session.run(query, legislator_data)
            return result.single()
    
    async def create_bill_node(self, bill_data: Dict[str, Any]):
        """Create a bill node in Neo4j"""
        with self.driver.session() as session:
            query = """
            MERGE (b:Bill {id: $id})
            SET b.title = $title,
                b.bill_number = $bill_number,
                b.status = $status,
                b.summary = $summary,
                b.introduced_date = date($introduced_date),
                b.created_at = datetime()
            RETURN b
            """
            result = session.run(query, bill_data)
            return result.single()
    
    async def create_vote_relationship(self, vote_data: Dict[str, Any]):
        """Create a vote relationship between legislator and bill"""
        with self.driver.session() as session:
            query = """
            MATCH (l:Legislator {id: $legislator_id})
            MATCH (b:Bill {id: $bill_id})
            MERGE (l)-[v:VOTED {vote: $vote, date: datetime($vote_date)}]->(b)
            RETURN v
            """
            result = session.run(query, vote_data)
            return result.single()
    
    async def create_committee_relationship(self, legislator_id: str, committee_data: Dict[str, Any]):
        """Create committee membership relationship"""
        with self.driver.session() as session:
            query = """
            MATCH (l:Legislator {id: $legislator_id})
            MERGE (c:Committee {id: $committee_id})
            SET c.name = $committee_name,
                c.chamber = $chamber
            MERGE (l)-[m:MEMBER_OF {role: $role, start_date: date($start_date)}]->(c)
            RETURN m
            """
            params = {
                "legislator_id": legislator_id,
                **committee_data
            }
            result = session.run(query, params)
            return result.single()
    
    async def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity"""
        with self.driver.session() as session:
            query = """
            MATCH (n {id: $entity_id})-[r]-(m)
            RETURN n, r, m, type(r) as relationship_type
            LIMIT 50
            """
            result = session.run(query, {"entity_id": entity_id})
            
            relationships = []
            for record in result:
                relationships.append({
                    "source": dict(record["n"]),
                    "target": dict(record["m"]),
                    "relationship": {
                        "type": record["relationship_type"],
                        "properties": dict(record["r"])
                    }
                })
            
            return relationships
    
    async def get_influence_network(self, limit: int = 100) -> Dict[str, Any]:
        """Get the political influence network"""
        with self.driver.session() as session:
            # Get legislators and their voting patterns
            query = """
            MATCH (l:Legislator)-[v:VOTED]->(b:Bill)
            WITH l, COUNT(v) as vote_count, 
                 SUM(CASE WHEN v.vote = 'yes' THEN 1 ELSE 0 END) as yes_votes
            WHERE vote_count > 5
            WITH l, vote_count, yes_votes, 
                 ROUND(toFloat(yes_votes) / vote_count, 2) as support_rate
            
            MATCH (l)-[v1:VOTED]->(b:Bill)<-[v2:VOTED]-(other:Legislator)
            WHERE l <> other AND v1.vote = v2.vote
            WITH l, other, support_rate, COUNT(*) as agreement_count
            WHERE agreement_count > 3
            
            RETURN l.id as legislator1_id, l.name as legislator1_name, l.party as party1,
                   other.id as legislator2_id, other.name as legislator2_name, other.party as party2,
                   agreement_count, support_rate
            ORDER BY agreement_count DESC
            LIMIT $limit
            """
            
            result = session.run(query, {"limit": limit})
            
            nodes = {}
            edges = []
            
            for record in result:
                # Add nodes
                leg1_id = record["legislator1_id"]
                leg2_id = record["legislator2_id"]
                
                if leg1_id not in nodes:
                    nodes[leg1_id] = {
                        "id": leg1_id,
                        "name": record["legislator1_name"],
                        "party": record["party1"],
                        "support_rate": record["support_rate"]
                    }
                
                if leg2_id not in nodes:
                    nodes[leg2_id] = {
                        "id": leg2_id,
                        "name": record["legislator2_name"],
                        "party": record["party2"]
                    }
                
                # Add edge
                edges.append({
                    "source": leg1_id,
                    "target": leg2_id,
                    "weight": record["agreement_count"],
                    "type": "voting_agreement"
                })
            
            return {
                "nodes": list(nodes.values()),
                "edges": edges
            }
    
    async def detect_communities(self) -> List[Dict[str, Any]]:
        """Detect communities in the political network using Louvain algorithm"""
        with self.driver.session() as session:
            # First, create a projection if it doesn't exist
            projection_query = """
            CALL gds.graph.project.cypher(
                'politicalNetwork',
                'MATCH (l:Legislator) RETURN id(l) AS id, l.party AS party',
                'MATCH (l1:Legislator)-[:VOTED {vote: "yes"}]->(b:Bill)<-[:VOTED {vote: "yes"}]-(l2:Legislator)
                 WHERE l1 <> l2
                 RETURN id(l1) AS source, id(l2) AS target, COUNT(*) AS weight'
            )
            """
            
            try:
                session.run(projection_query)
            except Exception:
                # Projection might already exist
                pass
            
            # Run community detection
            community_query = """
            CALL gds.louvain.stream('politicalNetwork')
            YIELD nodeId, communityId
            MATCH (l:Legislator) WHERE id(l) = nodeId
            RETURN l.id as legislator_id, l.name as name, l.party as party, 
                   communityId
            ORDER BY communityId, l.name
            """
            
            result = session.run(community_query)
            
            communities = {}
            for record in result:
                community_id = record["communityId"]
                if community_id not in communities:
                    communities[community_id] = {
                        "id": community_id,
                        "members": [],
                        "parties": {}
                    }
                
                member = {
                    "id": record["legislator_id"],
                    "name": record["name"],
                    "party": record["party"]
                }
                
                communities[community_id]["members"].append(member)
                
                # Count party distribution
                party = record["party"] or "Unknown"
                if party not in communities[community_id]["parties"]:
                    communities[community_id]["parties"][party] = 0
                communities[community_id]["parties"][party] += 1
            
            return list(communities.values())
    
    async def get_bill_support_network(self, bill_id: str) -> Dict[str, Any]:
        """Get network of legislators who supported/opposed a specific bill"""
        with self.driver.session() as session:
            query = """
            MATCH (b:Bill {id: $bill_id})<-[v:VOTED]-(l:Legislator)
            RETURN l.id as legislator_id, l.name as name, l.party as party, 
                   l.state as state, v.vote as vote
            """
            
            result = session.run(query, {"bill_id": bill_id})
            
            supporters = []
            opponents = []
            abstainers = []
            
            for record in result:
                legislator = {
                    "id": record["legislator_id"],
                    "name": record["name"],
                    "party": record["party"],
                    "state": record["state"]
                }
                
                vote = record["vote"]
                if vote == "yes":
                    supporters.append(legislator)
                elif vote == "no":
                    opponents.append(legislator)
                else:
                    abstainers.append(legislator)
            
            return {
                "bill_id": bill_id,
                "supporters": supporters,
                "opponents": opponents,
                "abstainers": abstainers,
                "support_count": len(supporters),
                "opposition_count": len(opponents),
                "abstain_count": len(abstainers)
            }
    
    async def get_party_voting_alignment(self) -> Dict[str, Any]:
        """Analyze voting alignment between parties"""
        with self.driver.session() as session:
            query = """
            MATCH (l1:Legislator)-[v1:VOTED]->(b:Bill)<-[v2:VOTED]-(l2:Legislator)
            WHERE l1.party IS NOT NULL AND l2.party IS NOT NULL 
            AND l1.party <> l2.party
            WITH l1.party as party1, l2.party as party2, 
                 COUNT(CASE WHEN v1.vote = v2.vote THEN 1 END) as agreements,
                 COUNT(*) as total_comparisons
            WHERE total_comparisons > 10
            RETURN party1, party2, agreements, total_comparisons,
                   ROUND(toFloat(agreements) / total_comparisons, 3) as alignment_rate
            ORDER BY alignment_rate DESC
            """
            
            result = session.run(query)
            
            alignments = []
            for record in result:
                alignments.append({
                    "party1": record["party1"],
                    "party2": record["party2"],
                    "agreements": record["agreements"],
                    "total_comparisons": record["total_comparisons"],
                    "alignment_rate": record["alignment_rate"]
                })
            
            return {
                "party_alignments": alignments,
                "analysis_timestamp": "2024-01-01T00:00:00Z"  # You'd use actual timestamp
            }