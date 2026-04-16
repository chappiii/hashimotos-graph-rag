from neo4j import GraphDatabase
from qdrant_client import QdrantClient


def connect_neo4j(uri, auth):
    driver = GraphDatabase.driver(uri, auth=auth)
    driver.verify_connectivity()
    print("Neo4j connection established.")
    return driver


def connect_qdrant(url):
    client = QdrantClient(url=url)
    print("Qdrant connection established.")
    return client
