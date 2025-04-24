import json
from neo4j import GraphDatabase
import csv


def db_connection(db_name):
    uri = "bolt://localhost:7687"
    username = "nyambuu"
    password = "12345678"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session(database=db_name)
    result = session.run("RETURN 'Hello, Neo4j!' AS message")
    for record in result:
        print(record["message"])

    return driver, session


def verify_database(session):
    result = session.run("CALL db.info() YIELD name")
    for record in result:
        print(f"Current database: {record['name']}")


def close_connection(driver, session):
    session.close()
    driver.close()
    print("Connection closed.")