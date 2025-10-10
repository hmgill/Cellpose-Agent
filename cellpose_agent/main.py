"""

"""

import sys
import argparse
from langfuse import get_client
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

from config import settings
from agents.agent import CellposeAgent
from stores import neo4j_store



def setup_observability():
    """Initializes Langfuse and Smolagents instrumentation."""
    get_client()
    SmolagentsInstrumentor().instrument()
    print("✓ Observability and instrumentation initialized.")


    
def run_agent_tasks():
    """Initializes and runs the agent against a predefined list of tasks."""

    agent = CellposeAgent()
    
    tasks = [
        "What does cellpose flow threshold control?",
    ]

    for task in tasks:
        try:
            agent.run(task)
        except Exception as e:
            print(f"An error occurred while processing the task: '{task}'. Error: {e}")

    get_client().flush()
    print("\n--- All traces sent to Langfuse ---")

    

def main():
    """
    Main entry point for the application.
    Use '--build-kg' to populate the knowledge graph.
    Run without arguments to start the agent.
    """
    parser = argparse.ArgumentParser(
        description="A RAG agent for cellpose-sam segmentation. Use --build-kg for first-time setup."
    )
    parser.add_argument(
        '--build-kg',
        action='store_true',
        help="Build and populate the Neo4j knowledge graph from source documents. This is a one-time setup step."
    )
    args = parser.parse_args()

    # --- Mode 1: Build the Knowledge Graph ---
    if args.build_kg:
        print("\n--- Knowledge Graph Build Mode ---")
        settings.configure_llama_index()
        try:
            neo4j_store.initialize_knowledge_graph()
            print("\n✅ Knowledge graph has been successfully built and populated in Neo4j.")
        except Exception as e:
            print(f"\n❌ An error occurred during knowledge graph creation: {e}")
        sys.exit(0)

    # --- Mode 2: Run the Agent (Default) ---
    print("\n--- Agent Execution Mode ---")
    setup_observability()
    settings.configure_llama_index()

    # Prerequisite Check: Ensure the knowledge graph is ready
    try:
        node_count, _ = neo4j_store.check_graph_status()
        if node_count == 0:
            print("\n❌ FATAL ERROR: The knowledge graph is empty.")
            print("Please run the one-time setup command before starting the agent:")
            print("\n    python main.py --build-kg\n")
            sys.exit(1)
        print(f"✓ Knowledge graph is ready with {node_count} nodes.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not connect to or verify Neo4j: {e}")
        sys.exit(1)

    # If prerequisites are met, run the agent
    run_agent_tasks()


if __name__ == "__main__":
    main()
