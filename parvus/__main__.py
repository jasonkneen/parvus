"""
Parvus CLI and main entry point.
Run with: python -m parvus
"""

import argparse
import json
import sys
from pathlib import Path

from .parvus import ParvusCompressor

def main():
    parser = argparse.ArgumentParser(
        description="Parvus - Quantum-inspired data compression system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file (JSON or CSV)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for compressed data",
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query string to search in compressed data",
    )
    
    parser.add_argument(
        "--load", "-l",
        type=str,
        help="Load previously compressed data",
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Streamlit GUI",
    )
    
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the API server",
    )
    
    args = parser.parse_args()

    # Initialize compressor
    compressor = ParvusCompressor()

    # Launch GUI
    if args.gui:
        import streamlit.web.bootstrap
        import os
        gui_path = Path(__file__).parent.parent / "gui.py"
        sys.argv = ["streamlit", "run", str(gui_path)]
        streamlit.web.bootstrap.run()
        return

    # Start server
    if args.server:
        from flask import Flask
        import os
        server_path = Path(__file__).parent.parent / "server.py"
        os.system(f"python {server_path}")
        return

    # Load existing compressed data
    if args.load:
        try:
            load_dir = Path(args.load)
            compressor.load_state(
                str(load_dir / "compressed_data.pkl"),
                str(load_dir / "faiss_index.bin")
            )
            print(f"Successfully loaded compressed data from {args.load}")
        except Exception as e:
            print(f"Error loading compressed data: {e}")
            return

    # Process new input file
    if args.input:
        try:
            input_path = Path(args.input)
            if input_path.suffix.lower() == '.json':
                embeddings = compressor.load_data_from_json(str(input_path))
            elif input_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.read_csv(str(input_path))
                embeddings = compressor.load_data_from_dataframe(df)
            else:
                print(f"Unsupported file format: {input_path.suffix}")
                return

            compressor.compress(embeddings)
            print("Data compressed successfully")

            # Save compressed data if output directory specified
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                compressor.save_state(
                    str(output_dir / "compressed_data.pkl"),
                    str(output_dir / "faiss_index.bin")
                )
                print(f"Compressed data saved to {args.output}")

        except Exception as e:
            print(f"Error processing input file: {e}")
            return

    # Perform query if specified
    if args.query:
        try:
            results, distances = compressor.query(args.query)
            print("\nQuery Results:")
            for idx, (result, distance) in enumerate(zip(results, distances), 1):
                print(f"\n{idx}. Distance: {distance:.4f}")
                print(f"   Content: {result}")
        except Exception as e:
            print(f"Error performing query: {e}")
            return

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()

if __name__ == "__main__":
    main()
