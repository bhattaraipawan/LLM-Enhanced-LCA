"""Dependency-free XLSX, BOM, and export helpers.

The application only needs a small, well-defined subset of the XLSX format.
Keeping that implementation here avoids making workbook upload and download
depend on pandas or openpyxl.  Cell references, rather than the sequence in
which cells appear in XML, determine column positions; this is important
because Excel omits blank cells from otherwise populated rows.
"""

from __future__ import annotations

import csv
import io
import math
import posixpath
import re
import zipfile
from datetime import datetime, timezone
from numbers import Real
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from .text import find_column, material_name_from_query, normalized_header

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REFERENCE_RE = re.compile(r"^\$?([A-Za-z]+)")
_INVALID_XML_CHARS_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]"
)
_MAX_XLSX_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_ENTRIES = 10_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_WORKSHEET_ROWS = 100_000
_MAX_WORKSHEET_CELLS = 1_000_000
_MAX_EXCEL_COLUMN_INDEX = 16_383  # XFD, zero based
_MAX_EXCEL_ROW = 1_048_576


def calculate_result_table(result: dict[str, Any]) -> list[list[Any]]:
    """Return the compact, user-facing table for a material calculation."""

    return [
        ["material name", "source", "kg_co2e_per_kg"],
        [
            material_name_from_query(str(result.get("input") or "")),
            result.get("source", ""),
            result.get("kg_co2e_per_kg"),
        ],
    ]


def calculate_result_to_csv(result: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(calculate_result_table(result))
    return output.getvalue()


def calculate_result_to_excel(result: dict[str, Any]) -> bytes:
    return rows_to_excel(calculate_result_table(result), sheet_name="GWP Result")


def excel_cell(column_index: int, row_index: int) -> str:
    """Convert zero-based ``column_index`` and one-based row to an A1 reference."""

    if column_index < 0:
        raise ValueError("column_index must be zero or greater")
    if column_index > _MAX_EXCEL_COLUMN_INDEX:
        raise ValueError("column_index exceeds Excel's XFD column limit")
    if row_index < 1:
        raise ValueError("row_index must be one or greater")
    if row_index > _MAX_EXCEL_ROW:
        raise ValueError("row_index exceeds Excel's row limit")

    column_name = ""
    number = column_index + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        column_name = chr(65 + remainder) + column_name
    return f"{column_name}{row_index}"


def _column_index(column_name: str) -> int:
    """Convert an Excel column name to a zero-based index."""

    if not column_name or not column_name.isalpha():
        raise ValueError(f"Invalid Excel column name: {column_name!r}")

    result = 0
    for character in column_name.upper():
        result = result * 26 + ord(character) - 64
    index = result - 1
    if index > _MAX_EXCEL_COLUMN_INDEX:
        raise ValueError(
            f"Excel column {column_name!r} exceeds the XFD column limit."
        )
    return index


def cell_column_name(cell_ref: str) -> str:
    match = _CELL_REFERENCE_RE.match(str(cell_ref or ""))
    return match.group(1).upper() if match else ""


def _clean_xml_text(value: object) -> str:
    return _INVALID_XML_CHARS_RE.sub("", str(value))


def _xml_text(value: object) -> str:
    return escape(_clean_xml_text(value))


def _xml_attribute(value: object) -> str:
    return escape(
        _clean_xml_text(value),
        {
            '"': "&quot;",
            "'": "&apos;",
        },
    )


def _cell_xml(cell_ref: str, value: object) -> str:
    if value is None:
        return f'<c r="{cell_ref}"/>'

    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{int(value)}</v></c>'

    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric):
            rendered = str(int(numeric)) if numeric.is_integer() else repr(numeric)
            return f'<c r="{cell_ref}"><v>{rendered}</v></c>'

    text = _clean_xml_text(value)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<c r="{cell_ref}" t="inlineStr"><is><t{preserve}>'
        f"{escape(text)}</t></is></c>"
    )


def excel_sheet_xml(
    rows: Iterable[Sequence[object]],
    drawing_rel_id: str | None = None,
) -> str:
    """Serialize rows into an XLSX worksheet XML document."""

    sheet_rows: list[str] = []
    max_columns = 0
    for row_index, row in enumerate(rows, start=1):
        materialized = list(row)
        max_columns = max(max_columns, len(materialized))
        cells = [
            _cell_xml(excel_cell(column_index, row_index), value)
            for column_index, value in enumerate(materialized)
        ]
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    drawing_xml = (
        f'<drawing r:id="{_xml_attribute(drawing_rel_id)}"/>'
        if drawing_rel_id
        else ""
    )
    rel_namespace = f' xmlns:r="{_OFFICE_REL_NS}"' if drawing_rel_id else ""
    last_cell = excel_cell(max(max_columns - 1, 0), max(len(sheet_rows), 1))
    dimension = f"A1:{last_cell}"
    formatted_column_count = max(max_columns, 3)

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{_MAIN_NS}"{rel_namespace}>
  <dimension ref="{dimension}"/>
  <cols>
    <col min="1" max="1" width="20" customWidth="1"/>
    <col min="2" max="2" width="18" customWidth="1"/>
    <col min="3" max="{formatted_column_count}" width="20" customWidth="1"/>
  </cols>
  <sheetData>{"".join(sheet_rows)}</sheetData>{drawing_xml}
</worksheet>"""


def safe_sheet_name(name: object) -> str:
    """Return a valid Excel worksheet name (maximum 31 characters)."""

    cleaned = _INVALID_XML_CHARS_RE.sub("", str(name or "Sheet1"))
    cleaned = re.sub(r"[\[\]:*?/\\]", " ", cleaned).strip().strip("'")
    return (cleaned or "Sheet1")[:31]


def _core_properties_xml(now: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>LLM-Enhanced WBLCA</dc:creator>
  <cp:lastModifiedBy>LLM-Enhanced WBLCA</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


_ROOT_RELS_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="{_OFFICE_REL_NS}/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="{_PACKAGE_REL_NS}/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="{_OFFICE_REL_NS}/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

_APP_PROPERTIES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>LLM-Enhanced WBLCA</Application>
</Properties>"""


def _zip_xml_files(files: dict[str, str | bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as workbook:
        for path, content in files.items():
            workbook.writestr(path, content)
    return output.getvalue()


def rows_to_excel(
    rows: Iterable[Sequence[object]],
    sheet_name: str = "GWP Result",
) -> bytes:
    """Create a small standards-compliant XLSX workbook from row values."""

    materialized_rows = [list(row) for row in rows]
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    escaped_sheet_name = _xml_attribute(safe_sheet_name(sheet_name))

    files: dict[str, str] = {
        "[Content_Types].xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        "_rels/.rels": _ROOT_RELS_XML,
        "docProps/core.xml": _core_properties_xml(now),
        "docProps/app.xml": _APP_PROPERTIES_XML,
        "xl/workbook.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{_MAIN_NS}" xmlns:r="{_OFFICE_REL_NS}">
  <sheets>
    <sheet name="{escaped_sheet_name}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""",
        "xl/_rels/workbook.xml.rels": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="{_OFFICE_REL_NS}/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        "xl/worksheets/sheet1.xml": excel_sheet_xml(materialized_rows),
    }
    return _zip_xml_files(files)


def numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _header_index(header: Sequence[object], name: str) -> int:
    target = normalized_header(name)
    for index, value in enumerate(header):
        if normalized_header(value) == target:
            return index
    raise ValueError(f"BOM rows must contain a '{name}' column.")


def pie_summary_rows(
    bom_rows: Sequence[Sequence[object]],
    other_threshold: float = 0.05,
) -> list[list[object]]:
    """Aggregate positive GWP totals for a pie chart.

    Rows contributing less than ``other_threshold`` of the grand total are
    grouped into a single ``Other`` row.
    """

    output_header: list[object] = ["material name", "total_kg_co2e", "share"]
    if len(bom_rows) < 2:
        return [output_header]

    threshold = float(other_threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("other_threshold must be between 0 and 1")

    header = list(bom_rows[0])
    material_col = _header_index(header, "material name")
    total_col = _header_index(header, "total_kg_co2e")
    totals: dict[str, float] = {}

    for row in bom_rows[1:]:
        total = numeric_value(row[total_col] if total_col < len(row) else None)
        if total is None or total <= 0:
            continue
        material = str(
            row[material_col] if material_col < len(row) else "Unknown"
        ).strip()
        material = material or "Unknown"
        totals[material] = totals.get(material, 0.0) + total

    grand_total = sum(totals.values())
    if grand_total <= 0:
        return [output_header]

    major_rows: list[list[object]] = []
    other_total = 0.0
    for material, total in sorted(
        totals.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        share = total / grand_total
        if share < threshold:
            other_total += total
        else:
            major_rows.append([material, total, share])

    if other_total > 0:
        major_rows.append(["Other", other_total, other_total / grand_total])

    return [output_header, *major_rows]


def xlsx_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        xml = workbook.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError("The Excel shared-string table is invalid.") from exc

    namespace = {"x": _MAIN_NS}
    strings: list[str] = []
    for item in root.findall("x:si", namespace):
        parts = [node.text or "" for node in item.findall(".//x:t", namespace)]
        strings.append("".join(parts))
    return strings


def xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> object:
    namespace = {"x": _MAIN_NS}
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        parts = [node.text or "" for node in cell.findall(".//x:t", namespace)]
        return "".join(parts).strip()

    value_node = cell.find("x:v", namespace)
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(value)].strip()
        except (ValueError, IndexError):
            return value
    if cell_type == "b":
        return value == "1"
    if cell_type in {"str", "e", "d"}:
        return value

    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return value
    return int(number) if number.is_integer() else number


def _worksheet_from_relationships(workbook: zipfile.ZipFile) -> str | None:
    try:
        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships_root = ET.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
    except (KeyError, ET.ParseError):
        return None

    sheets = workbook_root.find(f"{{{_MAIN_NS}}}sheets")
    first_sheet = sheets.find(f"{{{_MAIN_NS}}}sheet") if sheets is not None else None
    if first_sheet is None:
        return None

    relationship_id = first_sheet.attrib.get(f"{{{_OFFICE_REL_NS}}}id")
    if not relationship_id:
        return None

    for relationship in relationships_root.findall(
        f"{{{_PACKAGE_REL_NS}}}Relationship"
    ):
        if relationship.attrib.get("Id") != relationship_id:
            continue
        target = relationship.attrib.get("Target", "").replace("\\", "/")
        if not target:
            return None
        if target.startswith("/"):
            path = posixpath.normpath(target.lstrip("/"))
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        if path.startswith("../") or path == "..":
            return None
        return path
    return None


def first_worksheet_path(workbook: zipfile.ZipFile) -> str:
    names = set(workbook.namelist())
    related_path = _worksheet_from_relationships(workbook)
    if related_path in names:
        return related_path
    if "xl/worksheets/sheet1.xml" in names:
        return "xl/worksheets/sheet1.xml"

    worksheet_paths = sorted(
        name
        for name in names
        if name.startswith("xl/worksheets/")
        and name.endswith(".xml")
        and "/_rels/" not in name
    )
    if not worksheet_paths:
        raise ValueError("No worksheet found in the Excel file.")
    return worksheet_paths[0]


def _validate_workbook_archive(workbook: zipfile.ZipFile) -> None:
    entries = workbook.infolist()
    if len(entries) > _MAX_ARCHIVE_ENTRIES:
        raise ValueError("The Excel workbook contains too many archive entries.")

    total_size = 0
    for entry in entries:
        if entry.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError(
                f"The Excel workbook member {entry.filename!r} is too large."
            )
        total_size += entry.file_size
        if total_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("The expanded Excel workbook is too large.")


def read_xlsx_rows(content: bytes | bytearray | memoryview) -> list[list[object]]:
    """Read values from the first worksheet in an XLSX byte string.

    Missing cells are represented as empty strings.  For example, cells A2 and
    C2 produce ``["value from A2", "", "value from C2"]`` rather than shifting
    C2 into the second list position.
    """

    raw_content = bytes(content)
    if len(raw_content) > _MAX_XLSX_BYTES:
        raise ValueError(
            f"Upload must not exceed {_MAX_XLSX_BYTES} bytes."
        )

    try:
        workbook = zipfile.ZipFile(io.BytesIO(raw_content))
    except (TypeError, zipfile.BadZipFile) as exc:
        raise ValueError("Upload must be a valid .xlsx Excel file.") from exc

    try:
        with workbook:
            _validate_workbook_archive(workbook)
            shared_strings = xlsx_shared_strings(workbook)
            sheet_xml = workbook.read(first_worksheet_path(workbook))
    except KeyError as exc:
        raise ValueError("The Excel workbook is missing its worksheet data.") from exc

    try:
        root = ET.fromstring(sheet_xml)
    except ET.ParseError as exc:
        raise ValueError("The Excel worksheet XML is invalid.") from exc

    namespace = {"x": _MAIN_NS}
    rows: list[list[object]] = []
    cell_count = 0

    worksheet_rows = root.findall(".//x:sheetData/x:row", namespace)
    if len(worksheet_rows) > _MAX_WORKSHEET_ROWS:
        raise ValueError("The Excel worksheet contains too many rows.")

    for row in worksheet_rows:
        values_by_column: dict[int, object] = {}
        next_implicit_column = 0
        for cell in row.findall("x:c", namespace):
            cell_count += 1
            if cell_count > _MAX_WORKSHEET_CELLS:
                raise ValueError("The Excel worksheet contains too many cells.")
            column_name = cell_column_name(cell.attrib.get("r", ""))
            if column_name:
                column_index = _column_index(column_name)
            else:
                column_index = next_implicit_column
            values_by_column[column_index] = xlsx_cell_value(cell, shared_strings)
            next_implicit_column = column_index + 1

        if not values_by_column:
            continue

        values = [""] * (max(values_by_column) + 1)
        for column_index, value in values_by_column.items():
            values[column_index] = value
        rows.append(values)

    return rows


def bom_items_from_xlsx(content: bytes | bytearray | memoryview) -> list[dict[str, Any]]:
    """Parse description, quantity, and unit columns from an XLSX upload."""

    rows = read_xlsx_rows(content)
    header_index: int | None = None
    description_col = quantity_col = unit_col = None

    for index, row in enumerate(rows):
        description_col = find_column(
            list(row),
            [
                "description",
                "material",
                "material description",
                "item description",
                "product",
            ],
        )
        quantity_col = find_column(list(row), ["quantity", "qty", "amount"])
        unit_col = find_column(list(row), ["unit", "units", "uom"])
        if (
            description_col is not None
            and quantity_col is not None
            and unit_col is not None
            and len({description_col, quantity_col, unit_col}) == 3
        ):
            header_index = index
            break

    if (
        header_index is None
        or description_col is None
        or quantity_col is None
        or unit_col is None
    ):
        raise ValueError(
            "Excel file must contain description, quantity, and unit columns."
        )

    items: list[dict[str, Any]] = []
    max_required_col = max(description_col, quantity_col, unit_col)
    for row_number, row in enumerate(
        rows[header_index + 1 :],
        start=header_index + 2,
    ):
        padded = list(row) + [""] * (max_required_col + 1 - len(row))
        description = str(padded[description_col] or "").strip()
        quantity = padded[quantity_col]
        unit = str(padded[unit_col] or "").strip()

        if not description and quantity in ("", None) and not unit:
            continue
        if not description:
            continue

        items.append(
            {
                "row_number": row_number,
                "description": description,
                "quantity": quantity,
                "unit": unit,
            }
        )

    if not items:
        raise ValueError("Excel file does not contain any usable BOM item rows.")
    return items


def pie_chart_xml(
    summary_row_count: int,
    sheet_name: str = "Pie Summary",
) -> str:
    """Return DrawingML for the optional native Excel pie chart."""

    last_row = max(summary_row_count, 2)
    sheet_ref = safe_sheet_name(sheet_name).replace("'", "''")
    cat_ref = f"'{sheet_ref}'!$A$2:$A${last_row}"
    val_ref = f"'{sheet_ref}'!$B$2:$B${last_row}"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="{_OFFICE_REL_NS}">
  <c:chart>
    <c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>GWP Contribution</a:t></a:r></a:p></c:rich></c:tx><c:overlay val="0"/></c:title>
    <c:plotArea>
      <c:layout/>
      <c:pieChart>
        <c:varyColors val="1"/>
        <c:ser>
          <c:idx val="0"/><c:order val="0"/>
          <c:cat><c:strRef><c:f>{_xml_text(cat_ref)}</c:f></c:strRef></c:cat>
          <c:val><c:numRef><c:f>{_xml_text(val_ref)}</c:f></c:numRef></c:val>
          <c:dLbls><c:showLegendKey val="0"/><c:showVal val="0"/><c:showCatName val="0"/><c:showSerName val="0"/><c:showPercent val="1"/><c:showLeaderLines val="1"/></c:dLbls>
        </c:ser>
        <c:firstSliceAng val="0"/>
      </c:pieChart>
    </c:plotArea>
    <c:legend><c:legendPos val="r"/><c:layout/><c:overlay val="0"/></c:legend>
    <c:plotVisOnly val="1"/>
  </c:chart>
</c:chartSpace>"""


def pie_drawing_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <xdr:twoCellAnchor>
    <xdr:from><xdr:col>4</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>1</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>
    <xdr:to><xdr:col>13</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>22</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>
    <xdr:graphicFrame macro="">
      <xdr:nvGraphicFramePr><xdr:cNvPr id="2" name="GWP Contribution Pie Chart"/><xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr>
      <xdr:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>
      <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId1"/></a:graphicData></a:graphic>
    </xdr:graphicFrame>
    <xdr:clientData/>
  </xdr:twoCellAnchor>
</xdr:wsDr>"""


def bom_rows_to_excel_with_pie(bom_rows: Sequence[Sequence[object]]) -> bytes:
    """Create an XLSX with result data, a summary sheet, and a native pie chart."""

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result_sheet = _xml_attribute(safe_sheet_name("BOM Results"))
    summary_sheet = _xml_attribute(safe_sheet_name("Pie Summary"))
    summary_rows = pie_summary_rows(bom_rows)

    files: dict[str, str] = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/drawings/drawing1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>
  <Override PartName="/xl/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>
</Types>""",
        "_rels/.rels": _ROOT_RELS_XML,
        "docProps/core.xml": _core_properties_xml(now),
        "docProps/app.xml": _APP_PROPERTIES_XML,
        "xl/workbook.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{_MAIN_NS}" xmlns:r="{_OFFICE_REL_NS}">
  <sheets>
    <sheet name="{result_sheet}" sheetId="1" r:id="rId1"/>
    <sheet name="{summary_sheet}" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>""",
        "xl/_rels/workbook.xml.rels": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="{_OFFICE_REL_NS}/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="{_OFFICE_REL_NS}/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>""",
        "xl/worksheets/sheet1.xml": excel_sheet_xml(bom_rows),
        "xl/worksheets/sheet2.xml": excel_sheet_xml(
            summary_rows,
            drawing_rel_id="rId1",
        ),
        "xl/worksheets/_rels/sheet2.xml.rels": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="{_OFFICE_REL_NS}/drawing" Target="../drawings/drawing1.xml"/>
</Relationships>""",
        "xl/drawings/drawing1.xml": pie_drawing_xml(),
        "xl/drawings/_rels/drawing1.xml.rels": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="{_OFFICE_REL_NS}/chart" Target="../charts/chart1.xml"/>
</Relationships>""",
        "xl/charts/chart1.xml": pie_chart_xml(
            len(summary_rows),
            sheet_name="Pie Summary",
        ),
    }
    return _zip_xml_files(files)


def _safe_archive_base(base_name: object) -> str:
    name = re.sub(r"\.xlsx$", "", str(base_name or ""), flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return name or "bom"


def bom_download_zip(
    bom_rows: Sequence[Sequence[object]],
    base_name: str,
) -> bytes:
    """Package the BOM result workbook and a PNG contribution chart."""

    # Import lazily so workbook-only operations never initialize plotting
    # libraries or require them to be installed.
    from .chart import pie_summary_to_png

    safe_base = _safe_archive_base(base_name)
    summary_rows = pie_summary_rows(bom_rows)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{safe_base}_gwp_results.xlsx",
            rows_to_excel(bom_rows, sheet_name="BOM Results"),
        )
        archive.writestr(
            f"{safe_base}_pie_chart.png",
            pie_summary_to_png(summary_rows),
        )
    return output.getvalue()


__all__ = [
    "bom_download_zip",
    "bom_items_from_xlsx",
    "bom_rows_to_excel_with_pie",
    "calculate_result_table",
    "calculate_result_to_csv",
    "calculate_result_to_excel",
    "cell_column_name",
    "excel_cell",
    "excel_sheet_xml",
    "find_column",
    "first_worksheet_path",
    "numeric_value",
    "pie_chart_xml",
    "pie_drawing_xml",
    "pie_summary_rows",
    "read_xlsx_rows",
    "rows_to_excel",
    "safe_sheet_name",
    "xlsx_cell_value",
    "xlsx_shared_strings",
]
