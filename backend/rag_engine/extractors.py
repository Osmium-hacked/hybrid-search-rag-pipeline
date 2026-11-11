"""
Turns an uploaded file into text sections, whatever its format.

Text formats are parsed directly. Images and scanned PDF pages are read by
a vision model, and audio or video is transcribed, so the text the judge
checks answers against is what those models extracted.
"""

import base64
import csv
import email
import email.policy
import io
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Optional

from defusedxml import ElementTree

from .config import GROQ_API_KEY, MAX_OCR_PAGES, TRANSCRIPTION_MODEL, VISION_MODEL


class UnsupportedFormat(ValueError):
    pass


class ExtractionError(ValueError):
    pass


@dataclass
class Section:
    text: str
    page: Optional[int] = None
    location: Optional[str] = None


@dataclass
class Extracted:
    sections: list[Section]
    # How the text was obtained, when a model produced it.
    method: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


# Office and e-book files are zip archives. These bound how much they can
# expand to when unpacked.
MAX_UNZIPPED_BYTES = 60 * 1024 * 1024
MAX_SPREADSHEET_ROWS = 5000


# --- helpers -----------------------------------------------------------------

def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _looks_like_text(data: bytes) -> bool:
    head = data[:8192]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        # A multi-byte character cut off at the end of the sample is fine.
        if e.start >= len(head) - 3:
            return True
    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in head)
    return printable / max(len(head), 1) > 0.95


def _open_zip(data: bytes) -> zipfile.ZipFile:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ExtractionError("The file is damaged or isn't the format its name says.")
    if sum(info.file_size for info in archive.infolist()) > MAX_UNZIPPED_BYTES:
        raise ExtractionError("The file expands to more data than HyRAG can read.")
    return archive


def _xml(archive: zipfile.ZipFile, name: str):
    return ElementTree.fromstring(archive.read(name))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _numbered(names: Iterable[str], pattern: str) -> list[str]:
    """Sorts names like slide2.xml, slide10.xml by their number."""
    matches = [(int(m.group(1)), n) for n in names if (m := re.fullmatch(pattern, n))]
    return [n for _, n in sorted(matches)]


class _HTMLText(HTMLParser):
    BLOCK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "header", "footer", "table", "pre", "blockquote",
    }
    SKIP = {"script", "style", "noscript", "template", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skipping = max(0, self.skipping - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)


def _html_to_text(markup: str) -> str:
    parser = _HTMLText()
    parser.feed(markup)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _rows_to_text(rows: list[list[str]]) -> str:
    """Writes table rows as 'Header: value' lines, which search matches better than bare cells."""
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return ""
    header = rows[0]
    has_header = len(rows) > 1 and all(h.strip() and not _is_number(h) for h in header)
    if not has_header:
        return "\n".join(" | ".join(c.strip() for c in row) for row in rows)
    lines = [" | ".join(h.strip() for h in header)]
    for row in rows[1:]:
        pairs = [f"{h.strip()}: {c.strip()}" for h, c in zip(header, row) if c.strip()]
        pairs += [c.strip() for c in row[len(header):] if c.strip()]
        lines.append("; ".join(pairs))
    return "\n".join(lines)


def _is_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02}:{secs:02}" if hours else f"{minutes}:{secs:02}"


# --- model-based readers -----------------------------------------------------

@lru_cache(maxsize=1)
def _groq():
    from groq import Groq

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    # The free tier allows only a few images a minute, so back off and retry
    # (the SDK honours the provider's retry-after header).
    return Groq(api_key=GROQ_API_KEY, max_retries=5)


# Tested against text-only and wide images: wordings that let the model skip
# the description made it repeat fragments of the text instead.
VISION_PROMPT = (
    "You are an OCR engine. Return the image's text exactly once, in reading order. "
    "Use Markdown for headings, lists and tables. "
    "If the image is a chart, diagram or photo, then after the text add one paragraph "
    "beginning 'Visual description:' that states only what is visibly shown, including "
    "chart values. Never repeat text and never add information that isn't visible."
)


def _jpeg_for_model(image) -> bytes:
    """Downscales an image and encodes it as JPEG to stay under the model's size limit."""
    from PIL import Image

    image = image.convert("RGB")
    image.thumbnail((2048, 2048), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


def _read_image(image) -> str:
    encoded = base64.b64encode(_jpeg_for_model(image)).decode()
    response = _groq().chat.completions.create(
        model=VISION_MODEL,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }],
    )
    text = response.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _model_name(model: str) -> str:
    return model.split("/")[-1]


# --- format readers ----------------------------------------------------------

def _plain_text(data: bytes) -> Extracted:
    return Extracted([Section(_decode_text(data))])


def _markup(data: bytes) -> Extracted:
    return Extracted([Section(_html_to_text(_decode_text(data)))])


def _json(data: bytes) -> Extracted:
    text = _decode_text(data)
    try:
        text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        pass
    return Extracted([Section(text)])


def _notebook(data: bytes) -> Extracted:
    try:
        cells = json.loads(_decode_text(data)).get("cells", [])
    except (ValueError, AttributeError):
        raise ExtractionError("The notebook isn't valid JSON.")
    sections = []
    for number, cell in enumerate(cells, start=1):
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)
        if text.strip():
            kind = "Code" if cell.get("cell_type") == "code" else "Text"
            sections.append(Section(text, location=f"Cell {number} ({kind.lower()})"))
    return Extracted(sections)


def _csv(data: bytes, delimiter: Optional[str] = None) -> Extracted:
    text = _decode_text(data)
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    warnings = []
    if len(rows) > MAX_SPREADSHEET_ROWS:
        warnings.append(f"Only the first {MAX_SPREADSHEET_ROWS:,} rows were read.")
        rows = rows[:MAX_SPREADSHEET_ROWS]
    return Extracted([Section(_rows_to_text(rows))], warnings=warnings)


def _pdf(data: bytes) -> Extracted:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("The PDF is password protected.")
        pages = list(reader.pages)
    except PdfReadError:
        raise ExtractionError("The PDF is damaged and couldn't be read.")

    sections, scanned = [], []
    for number, page in enumerate(pages, start=1):
        text = page.extract_text() or ""
        if len(text.strip()) >= 20:
            sections.append(Section(text, page=number, location=f"Page {number}"))
        else:
            scanned.append((number, page))

    if not scanned:
        return Extracted(sections)

    # Pages with no text layer are usually scans. Read their images instead.
    warnings = []
    if len(scanned) > MAX_OCR_PAGES:
        warnings.append(
            f"{len(scanned)} pages had no text layer. Only the first {MAX_OCR_PAGES} were read as images."
        )
        scanned = scanned[:MAX_OCR_PAGES]

    def read_page(item):
        number, page = item
        try:
            images = [image.image for image in page.images]
        except Exception:
            return None
        images = [image for image in images if image is not None]
        if not images:
            return None
        largest = max(images, key=lambda image: image.width * image.height)
        if largest.width * largest.height < 200 * 200:
            return None
        text = _read_image(largest)
        return Section(text, page=number, location=f"Page {number}") if text else None

    # One page at a time: parallel calls only run into the rate limit sooner.
    read = [s for s in map(read_page, scanned) if s]
    if not read:
        return Extracted(sections, warnings=warnings)
    sections = sorted(sections + read, key=lambda s: s.page)
    return Extracted(sections, method=f"Scanned pages read by {_model_name(VISION_MODEL)}", warnings=warnings)


def _image(data: bytes) -> Extracted:
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise ExtractionError("The image is damaged or in a format HyRAG can't open.")
    frames = getattr(image, "n_frames", 1)
    if frames > 1:
        image.seek(0)
    text = _read_image(image)
    warnings = ["Only the first frame was read."] if frames > 1 else []
    return Extracted(
        [Section(text, location="Image")],
        method=f"Read by {_model_name(VISION_MODEL)}",
        warnings=warnings,
    )


def _media(name: str, data: bytes) -> Extracted:
    response = _groq().audio.transcriptions.create(
        file=(name, data),
        model=TRANSCRIPTION_MODEL,
        response_format="verbose_json",
        temperature=0,
    )
    segments = getattr(response, "segments", None)
    if segments is None:
        segments = (getattr(response, "model_extra", None) or {}).get("segments")
    method = f"Transcribed by {_model_name(TRANSCRIPTION_MODEL)}"
    if not segments:
        return Extracted([Section(getattr(response, "text", "") or "")], method=method)

    # Group segments into passages of about a minute, labelled with their time range.
    sections, buffer, start = [], [], None
    for segment in segments:
        get = segment.get if isinstance(segment, dict) else lambda key: getattr(segment, key)
        if start is None:
            start = get("start")
        buffer.append(get("text").strip())
        end = get("end")
        if end - start >= 60 or sum(map(len, buffer)) >= 900:
            sections.append(Section(" ".join(buffer), location=f"{_timestamp(start)}–{_timestamp(end)}"))
            buffer, start = [], None
    if buffer:
        sections.append(Section(" ".join(buffer), location=f"{_timestamp(start)}–{_timestamp(end)}"))
    return Extracted(sections, method=method)


def _word_paragraphs(root) -> list[str]:
    paragraphs = []
    for element in root.iter():
        tag = _local(element.tag)
        if tag == "p":
            parts = []
            for node in element.iter():
                node_tag = _local(node.tag)
                if node_tag == "t" and node.text:
                    parts.append(node.text)
                elif node_tag == "tab":
                    parts.append("\t")
                elif node_tag in ("br", "cr"):
                    parts.append("\n")
            paragraphs.append("".join(parts))
    return paragraphs


def _docx(data: bytes) -> Extracted:
    archive = _open_zip(data)
    names = set(archive.namelist())
    if "word/document.xml" not in names:
        raise ExtractionError("The file isn't a valid Word document.")
    root = _xml(archive, "word/document.xml")

    # Read the body in order, writing table rows with cells separated by pipes.
    body = next((el for el in root if _local(el.tag) == "body"), root)
    blocks = []
    for element in body:
        tag = _local(element.tag)
        if tag == "p":
            blocks.extend(_word_paragraphs(element))
        elif tag == "tbl":
            for row in (el for el in element.iter() if _local(el.tag) == "tr"):
                cells = [
                    " ".join(_word_paragraphs(cell)).strip()
                    for cell in row
                    if _local(cell.tag) == "tc"
                ]
                blocks.append(" | ".join(cells))
            blocks.append("")
        elif tag == "sdt":
            blocks.extend(_word_paragraphs(element))

    sections = [Section("\n".join(blocks))]
    for extra, label in (("word/footnotes.xml", "Footnotes"), ("word/endnotes.xml", "Endnotes")):
        if extra in names:
            text = "\n".join(p for p in _word_paragraphs(_xml(archive, extra)) if p.strip())
            if text:
                sections.append(Section(text, location=label))
    return Extracted(sections)


def _slide_notes(archive: zipfile.ZipFile, slide: str) -> Optional[str]:
    """Finds a slide's speaker notes through its relationships file."""
    rels = posixpath.join(posixpath.dirname(slide), "_rels", posixpath.basename(slide) + ".rels")
    try:
        root = _xml(archive, rels)
    except KeyError:
        return None
    for rel in root:
        if rel.get("Type", "").endswith("/notesSlide"):
            return posixpath.normpath(posixpath.join(posixpath.dirname(slide), rel.get("Target", "")))
    return None


def _pptx(data: bytes) -> Extracted:
    archive = _open_zip(data)
    names = archive.namelist()
    slides = _numbered(names, r"ppt/slides/slide(\d+)\.xml")
    if not slides:
        raise ExtractionError("The file isn't a valid PowerPoint presentation.")

    sections = []
    for number, slide in enumerate(slides, start=1):
        text = "\n".join(p for p in _word_paragraphs(_xml(archive, slide)) if p.strip())
        notes_name = _slide_notes(archive, slide)
        if notes_name in names:
            notes = "\n".join(
                p for p in _word_paragraphs(_xml(archive, notes_name))
                if p.strip() and not p.strip().isdigit()
            )
            if notes:
                text += f"\n\nSpeaker notes:\n{notes}"
        if text.strip():
            sections.append(Section(text, page=number, location=f"Slide {number}"))
    return Extracted(sections)


def _xlsx(data: bytes) -> Extracted:
    from openpyxl import load_workbook

    _open_zip(data)  # size check before openpyxl unpacks it
    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception:
        raise ExtractionError("The file isn't a valid Excel workbook.")

    sections, warnings, total = [], [], 0
    for sheet in workbook.worksheets:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            if total >= MAX_SPREADSHEET_ROWS:
                break
            rows.append([_cell(value) for value in row])
            total += 1
        text = _rows_to_text(rows)
        if text:
            sections.append(Section(f"Sheet: {sheet.title}\n{text}", location=f"Sheet {sheet.title}"))
    if total >= MAX_SPREADSHEET_ROWS:
        warnings.append(f"Only the first {MAX_SPREADSHEET_ROWS:,} rows were read.")
    workbook.close()
    return Extracted(sections, warnings=warnings)


def _opendocument(data: bytes) -> Extracted:
    archive = _open_zip(data)
    if "content.xml" not in archive.namelist():
        raise ExtractionError("The file isn't a valid OpenDocument file.")
    root = _xml(archive, "content.xml")
    lines = []

    def walk(element):
        tag = _local(element.tag)
        if tag == "table-row":
            cells = [" ".join(cell.itertext()).strip() for cell in element if _local(cell.tag) == "table-cell"]
            lines.append(" | ".join(cells).rstrip(" |"))
        elif tag in ("p", "h"):
            lines.append("".join(element.itertext()))
        else:
            for child in element:
                walk(child)

    walk(root)
    return Extracted([Section("\n".join(line for line in lines if line.strip()))])


def _epub(data: bytes) -> Extracted:
    archive = _open_zip(data)
    names = archive.namelist()
    order = []
    try:
        container = _xml(archive, "META-INF/container.xml")
        opf_path = next(el.get("full-path") for el in container.iter() if _local(el.tag) == "rootfile")
        opf = _xml(archive, opf_path)
        base = posixpath.dirname(opf_path)
        manifest = {
            el.get("id"): posixpath.normpath(posixpath.join(base, el.get("href", "")))
            for el in opf.iter() if _local(el.tag) == "item"
        }
        order = [manifest[el.get("idref")] for el in opf.iter() if _local(el.tag) == "itemref" and el.get("idref") in manifest]
    except (KeyError, StopIteration, ElementTree.ParseError):
        pass
    if not order:
        order = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))

    sections = []
    for number, name in enumerate((n for n in order if n in names), start=1):
        text = _html_to_text(_decode_text(archive.read(name)))
        if text.strip():
            sections.append(Section(text, location=f"Section {number}"))
    return Extracted(sections)


def _rtf(data: bytes) -> Extracted:
    from striprtf.striprtf import rtf_to_text

    return Extracted([Section(rtf_to_text(_decode_text(data), errors="ignore"))])


def _eml(data: bytes) -> Extracted:
    message = email.message_from_bytes(data, policy=email.policy.default)
    header = "\n".join(
        f"{key}: {message[key]}" for key in ("Subject", "From", "To", "Cc", "Date") if message[key]
    )
    body = message.get_body(preferencelist=("plain", "html"))
    text = ""
    if body is not None:
        content = body.get_content()
        text = _html_to_text(content) if body.get_content_type() == "text/html" else content
    attachments = [part.get_filename() for part in message.iter_attachments() if part.get_filename()]
    warnings = [f"Attachments weren't read: {', '.join(attachments)}."] if attachments else []
    return Extracted([Section(f"{header}\n\n{text}")], warnings=warnings)


# --- dispatch ----------------------------------------------------------------

READERS: dict[str, Callable[[str, bytes], Extracted]] = {}


def _register(extensions: str, reader: Callable[[str, bytes], Extracted]) -> None:
    for extension in extensions.split():
        READERS[extension] = reader


_register(".pdf", lambda name, data: _pdf(data))
_register(".docx .docm .dotx", lambda name, data: _docx(data))
_register(".pptx .pptm .ppsx", lambda name, data: _pptx(data))
_register(".xlsx .xlsm", lambda name, data: _xlsx(data))
_register(".odt .odp .ods", lambda name, data: _opendocument(data))
_register(".epub", lambda name, data: _epub(data))
_register(".rtf", lambda name, data: _rtf(data))
_register(".eml", lambda name, data: _eml(data))
_register(".html .htm .xhtml", lambda name, data: _markup(data))
_register(".json", lambda name, data: _json(data))
_register(".ipynb", lambda name, data: _notebook(data))
_register(".csv", lambda name, data: _csv(data))
_register(".tsv", lambda name, data: _csv(data, "\t"))
_register(".png .jpg .jpeg .webp .gif .bmp .tif .tiff", lambda name, data: _image(data))
_register(".mp3 .wav .m4a .ogg .opus .flac .webm .mp4 .mpeg .mpga", _media)

# Old binary Office formats need converting first.
LEGACY = {".doc": ".docx", ".ppt": ".pptx", ".xls": ".xlsx"}

# Shown to users. Anything else that turns out to be text is read as plain text.
FORMAT_GROUPS = {
    "Documents": "PDF, Word, PowerPoint, Excel, OpenDocument, EPUB, RTF",
    "Text": "TXT, Markdown, CSV, JSON, HTML, email, code and other plain text",
    "Images": "PNG, JPG, WebP, GIF, TIFF, BMP",
    "Audio and video": "MP3, WAV, M4A, OGG, FLAC, MP4, WebM",
}


def extract(name: str, data: bytes) -> Extracted:
    extension = Path(name).suffix.lower()
    if extension in LEGACY:
        raise UnsupportedFormat(
            f"{extension} is an older Office format. Save it as {LEGACY[extension]} and upload it again."
        )
    reader = READERS.get(extension)
    if reader is None:
        if not _looks_like_text(data):
            raise UnsupportedFormat(f"HyRAG can't read {extension or 'files without an extension'} files.")
        reader = lambda name, data: _plain_text(data)

    extracted = reader(name, data)
    extracted.sections = [s for s in extracted.sections if s.text.strip()]
    return extracted
