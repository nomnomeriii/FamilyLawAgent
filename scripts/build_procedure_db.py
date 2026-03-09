from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from family_law_agent.procedure import ingest_procedure_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Chroma index for NY family law forms.")
    parser.add_argument("--forms-dir", default="./NY Family Law Forms", help="Directory containing forms PDFs/TXT")
    parser.add_argument("--db-dir", default="./db_procedure", help="Output Chroma persistence directory")
    args = parser.parse_args()

    result = ingest_procedure_documents(forms_directory=args.forms_dir, persist_directory=args.db_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
