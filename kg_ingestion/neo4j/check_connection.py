"""
Verify that Neo4j is reachable and the database is accessible.
All connection values come from kg_config (which reads .env).

Run: uv run -m kg_ingestion.neo4j.check_connection
"""

import sys
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from kg_ingestion.config.kg_config import (
    NEO4J_URI,
    NEO4J_AUTH,
    NEO4J_DATABASE,
)


def check_connection() -> bool:
    print(f"URI:      {NEO4J_URI}")
    print(f"User:     {NEO4J_AUTH[0]}")
    print(f"Database: {NEO4J_DATABASE}")
    print()

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
    except ServiceUnavailable:
        print("FAIL - Neo4j is not reachable at the URI above.")
        print("       Is Neo4j running? Check Desktop app or `neo4j start`.")
        return False
    except AuthError:
        print("FAIL - Authentication rejected.")
        print("       Check NEO4J_USER and NEO4J_PASSWORD in your .env.")
        return False

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("RETURN 1 AS ok")
        row = result.single()
        if row["ok"] != 1:
            print("FAIL - Query sanity check returned unexpected result.")
            driver.close()
            return False

        rows = session.run(
            "CALL dbms.components() YIELD name, versions RETURN name, versions"
        ).data()
        version = rows[0]["versions"][0] if rows else "unknown"

    driver.close()
    print(f"OK - Connected to Neo4j {version}, database '{NEO4J_DATABASE}'")
    return True


if __name__ == "__main__":
    ok = check_connection()
    sys.exit(0 if ok else 1)
