from __future__ import annotations

import io
import struct
import unittest
import zipfile
from unittest.mock import patch

from app.utils.chart import pie_summary_to_png
from app.utils.spreadsheet import (
    bom_download_zip,
    bom_items_from_xlsx,
    bom_rows_to_excel_with_pie,
    pie_summary_rows,
    read_xlsx_rows,
    rows_to_excel,
    safe_sheet_name,
)


def _sparse_shared_string_workbook() -> bytes:
    """Build a worksheet where row two omits its intermediate B cell."""

    shared_strings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="5" uniqueCount="5">
  <si><t>description</t></si>
  <si><t>quantity</t></si>
  <si><t>unit</t></si>
  <si><t>Cement</t></si>
  <si><t>kg</t></si>
</sst>"""
    worksheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c>
      <c r="B1" t="s"><v>1</v></c>
      <c r="C1" t="s"><v>2</v></c>
    </row>
    <row r="2">
      <c r="A2" t="s"><v>3</v></c>
      <c r="C2" t="s"><v>4</v></c>
    </row>
  </sheetData>
</worksheet>"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("xl/sharedStrings.xml", shared_strings)
        workbook.writestr("xl/worksheets/sheet1.xml", worksheet)
    return output.getvalue()


class SpreadsheetTests(unittest.TestCase):
    def test_rows_to_excel_round_trip(self) -> None:
        rows = [
            ["description", "quantity", "unit", "active"],
            ["Cement & aggregate <blend>", 12.5, "kg", True],
            ["Sand", None, "m3", False],
        ]

        content = rows_to_excel(rows, sheet_name="BOM / Results")

        self.assertTrue(content.startswith(b"PK"))
        expected = [rows[0], rows[1], ["Sand", "", "m3", False]]
        self.assertEqual(read_xlsx_rows(content), expected)
        with zipfile.ZipFile(io.BytesIO(content)) as workbook:
            self.assertIn("[Content_Types].xml", workbook.namelist())
            self.assertIn("xl/workbook.xml", workbook.namelist())
            workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
            self.assertIn('name="BOM   Results"', workbook_xml)

    def test_sparse_xlsx_cells_keep_their_real_column_positions(self) -> None:
        content = _sparse_shared_string_workbook()

        rows = read_xlsx_rows(content)
        items = bom_items_from_xlsx(content)

        self.assertEqual(
            rows,
            [
                ["description", "quantity", "unit"],
                ["Cement", "", "kg"],
            ],
        )
        self.assertEqual(
            items,
            [
                {
                    "row_number": 2,
                    "description": "Cement",
                    "quantity": "",
                    "unit": "kg",
                }
            ],
        )

    def test_invalid_xlsx_has_a_useful_error(self) -> None:
        with self.assertRaisesRegex(ValueError, r"valid \.xlsx"):
            read_xlsx_rows(b"not a workbook")

    def test_unit_price_is_not_mistaken_for_unit(self) -> None:
        content = rows_to_excel(
            [
                ["description", "quantity", "unit price"],
                ["cement", 2, 12.5],
            ]
        )
        with self.assertRaisesRegex(ValueError, "description, quantity, and unit"):
            bom_items_from_xlsx(content)

    def test_header_only_workbook_is_rejected(self) -> None:
        content = rows_to_excel([["description", "quantity", "unit"]])
        with self.assertRaisesRegex(ValueError, "usable BOM item"):
            bom_items_from_xlsx(content)

    def test_column_beyond_xfd_is_rejected_before_allocation(self) -> None:
        worksheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData><row r="1"><c r="ZZZZZZ1" t="inlineStr"><is><t>x</t></is></c></row></sheetData>
</worksheet>"""
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as workbook:
            workbook.writestr("xl/worksheets/sheet1.xml", worksheet)
        with self.assertRaisesRegex(ValueError, "XFD"):
            read_xlsx_rows(output.getvalue())

    def test_pie_summary_aggregates_and_groups_small_rows(self) -> None:
        bom_rows = [
            ["material name", "total_kg_co2e"],
            ["Cement", 80],
            ["Sand", 4],
            ["Brick", 6],
            ["Cement", 10],
            ["Missing", "MISSING"],
            ["Negative", -2],
        ]

        summary = pie_summary_rows(bom_rows)

        self.assertEqual(summary[0], ["material name", "total_kg_co2e", "share"])
        self.assertEqual([row[0] for row in summary[1:]], ["Cement", "Brick", "Other"])
        self.assertEqual(summary[1][1], 90)
        self.assertAlmostEqual(summary[1][2], 0.9)
        self.assertEqual(summary[3][1], 4)
        self.assertAlmostEqual(sum(float(row[2]) for row in summary[1:]), 1.0)

    def test_dependency_free_chart_is_a_valid_rgb_png(self) -> None:
        summary = [
            ["material name", "total_kg_co2e", "share"],
            ["Cement", 75, 0.75],
            ["Other", 25, 0.25],
        ]

        content = pie_summary_to_png(summary, prefer_matplotlib=False)

        self.assertEqual(content[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(content[12:16], b"IHDR")
        self.assertEqual(struct.unpack(">II", content[16:24]), (1000, 700))
        self.assertEqual(content[24], 8)
        self.assertEqual(content[25], 2)

    def test_download_zip_contains_readable_workbook_and_chart(self) -> None:
        bom_rows = [
            ["material name", "total_kg_co2e"],
            ["Cement", 100],
        ]
        fallback_png = pie_summary_to_png(
            pie_summary_rows(bom_rows),
            prefer_matplotlib=False,
        )

        with patch(
            "app.utils.chart.pie_summary_to_png",
            return_value=fallback_png,
        ):
            content = bom_download_zip(bom_rows, "../../My BOM.xlsx")

        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "My_BOM_gwp_results.xlsx",
                    "My_BOM_pie_chart.png",
                },
            )
            workbook = archive.read("My_BOM_gwp_results.xlsx")
            chart = archive.read("My_BOM_pie_chart.png")

        self.assertEqual(read_xlsx_rows(workbook), bom_rows)
        self.assertTrue(chart.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_native_chart_workbook_contains_summary_and_drawing_parts(self) -> None:
        bom_rows = [
            ["material name", "total_kg_co2e"],
            ["Cement", 70],
            ["Brick", 30],
        ]

        content = bom_rows_to_excel_with_pie(bom_rows)

        with zipfile.ZipFile(io.BytesIO(content)) as workbook:
            names = set(workbook.namelist())
            self.assertIn("xl/worksheets/sheet2.xml", names)
            self.assertIn("xl/drawings/drawing1.xml", names)
            self.assertIn("xl/charts/chart1.xml", names)
        self.assertEqual(read_xlsx_rows(content), bom_rows)

    def test_sheet_name_is_excel_safe(self) -> None:
        self.assertEqual(safe_sheet_name("  bad[]:*?/\\ name  "), "bad        name")
        self.assertLessEqual(len(safe_sheet_name("x" * 100)), 31)
        self.assertEqual(safe_sheet_name("''"), "Sheet1")


if __name__ == "__main__":
    unittest.main()
