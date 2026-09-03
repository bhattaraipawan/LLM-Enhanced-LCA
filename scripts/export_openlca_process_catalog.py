"""Export a reproducible process catalog from the active openLCA database.

The script connects to the openLCA IPC server running on the local computer,
retrieves descriptors for every Process in the active database, and writes one
Excel workbook with two worksheets:

    Processes  - process descriptors used for LLM candidate retrieval/matching
    Metadata   - database label, export time, software environment, and counts

The study configuration defaults to ELCD 3.2 and IPC port 8080.

Important
---------
The database label is provenance metadata; it does not switch databases inside
openLCA. Before running this script, open ELCD 3.2 in openLCA and start the IPC
server on port 8080.

Usage
-----
    LLM/scripts/export_openlca_process_catalog.py

Requirements
------------
    pip install -r requirements_ELCD.txt
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import olca_ipc as ipc
import olca_schema as o
import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.table import Table, TableStyleInfo


EXPORTER_VERSION = "1.0.0"
CATALOG_SCHEMA_VERSION = "1.0"
DEFAULT_DATABASE_LABEL = "ELCD 3.2"
DEFAULT_IPC_PORT = 8080
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT/"ELCD_Check"
DEFAULT_OUTPUT_FILE = "ELCD_Process_Catalog.xlsx"

PROCESS_COLUMNS = [
    "process_uuid",
    "process_name",
    "category",
    "location",
    "library",
    "process_type",
]


def _text(value: Any) -> str:
    """Convert an openLCA value safely to text."""
    if value is None:
        return ""
    return getattr(value, "value", str(value))


def _category_path(value: Any) -> str:
    """Convert an openLCA category path to a readable hierarchy."""
    if not value:
        return ""
    return " > ".join(str(part) for part in value)


def _package_version(package_name: str) -> str:
    """Return an installed package version without failing the export."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-detected"


def _autosize_columns(worksheet, max_width: int = 80) -> None:
    """Set readable Excel column widths with a conservative maximum width."""
    for column_cells in worksheet.columns:
        letter = column_cells[0].column_letter
        longest = max(
            (len(str(cell.value)) for cell in column_cells if cell.value is not None),
            default=0,
        )
        worksheet.column_dimensions[letter].width = min(longest + 2, max_width)


def _format_header(worksheet) -> None:
    """Apply simple publication-neutral formatting to the header row."""
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="center")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export process descriptors from the active openLCA database to "
            "a reproducible Excel catalog."
        )
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_IPC_PORT,
        help=f"openLCA IPC port (default: {DEFAULT_IPC_PORT})",
    )
    parser.add_argument(
        "--database-label",
        default=DEFAULT_DATABASE_LABEL,
        help=(
            "Database name/version recorded in metadata; this does not switch "
            f'databases in openLCA (default: "{DEFAULT_DATABASE_LABEL}")'
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Excel filename (default: {DEFAULT_OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    print(f"Connecting to openLCA IPC server at localhost:{args.port} ...")
    client = ipc.Client(args.port)

    try:
        descriptors = list(client.get_descriptors(o.Process))
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve processes from openLCA. Confirm that openLCA is "
            f"running, ELCD 3.2 is the active database, and the IPC server is "
            f"started on port {args.port}."
        ) from exc

    if not descriptors:
        raise RuntimeError(
            "The IPC request returned zero process descriptors. Confirm that the "
            "intended ELCD 3.2 database is active and contains process datasets."
        )

    rows: list[dict[str, str]] = []
    for process in descriptors:
        rows.append(
            {
                "process_uuid": _text(getattr(process, "id", "")),
                "process_name": _text(getattr(process, "name", "")),
                "category": _category_path(getattr(process, "category_path", None)),
                "location": _text(getattr(process, "location", "")),
                "library": _text(getattr(process, "library", "")),
                "process_type": _text(getattr(process, "process_type", "")),
            }
        )

    rows.sort(
        key=lambda row: (
            row["process_name"].casefold(),
            row["location"].casefold(),
            row["process_uuid"],
        )
    )

    processes_df = pd.DataFrame(rows, columns=PROCESS_COLUMNS)
    exported_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    metadata_rows = [
        ("catalog_schema_version", CATALOG_SCHEMA_VERSION),
        ("exporter_version", EXPORTER_VERSION),
        ("exported_at_utc", exported_at),
        ("database_label", args.database_label),
        ("ipc_host", "localhost"),
        ("ipc_port", args.port),
        ("process_count", len(processes_df)),
        ("output_file", args.output_file),
        ("python_version", platform.python_version()),
        ("platform", platform.platform()),
        ("olca_ipc_version", _package_version("olca-ipc")),
        ("olca_schema_version", _package_version("olca-schema")),
        ("pandas_version", _package_version("pandas")),
        ("openpyxl_version", _package_version("openpyxl")),
        (
            "provenance_note",
            "The catalog reflects the database active in openLCA at export time. "
            "The study configuration uses ELCD 3.2. The database label stored here "
            "is metadata and does not independently verify or switch the active "
            "openLCA database.",
        ),
    ]
    metadata_df = pd.DataFrame(metadata_rows, columns=["field", "value"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        processes_df.to_excel(writer, sheet_name="Processes", index=False)
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

        workbook = writer.book
        process_sheet = workbook["Processes"]
        metadata_sheet = workbook["Metadata"]

        process_sheet.freeze_panes = "A2"
        metadata_sheet.freeze_panes = "A2"
        _format_header(process_sheet)
        _format_header(metadata_sheet)

        # Excel table gives readers native filtering/sorting while preserving the
        # data exactly as exported.
        table = Table(displayName="OpenLCAProcesses", ref=process_sheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        process_sheet.add_table(table)

        metadata_sheet.auto_filter.ref = metadata_sheet.dimensions
        _autosize_columns(process_sheet, max_width=80)
        _autosize_columns(metadata_sheet, max_width=110)

    print("\nExport complete.")
    print(f"Database label:     {args.database_label}")
    print(f"Processes exported: {len(processes_df):,}")
    print(f"Excel workbook:     {output_path.resolve()}")
    print("Worksheets:         Processes, Metadata")


if __name__ == "__main__":
    main()
