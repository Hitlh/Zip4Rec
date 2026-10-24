"""Trusted single-column renderer. Generated code supplies rows of original-text spans."""
import ast
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont
from workflow import digest

ROOT = Path(__file__).resolve().parents[1]
FONT = Path(__file__).parent / 'fonts/DejaVuSans.ttf'
ENGINE_FILES = ['document.py', 'layout_engine.py', 'layout_worker.py']


@lru_cache(maxsize=32)
def load_font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=8192)
def supported_glyph(path, size, char):
    fnt = load_font(path, size)
    return char.isspace() or bytes(fnt.getmask(char)) != bytes(fnt.getmask('\U0010ffff'))


@dataclass(frozen=True)
class Constraints:
    font_path: str = str(FONT)
    font_size: int = 14
    heading_size: int = 15
    min_width: int = 256
    max_width: int = 1536
    max_height: int = 16384
    max_pixels: int = 8_000_000

    def __post_init__(self):
        if not Path(self.font_path).is_file():
            raise FileNotFoundError(self.font_path)
        if any(type(getattr(self, name)) is not int for name in
               ['font_size', 'heading_size', 'min_width', 'max_width', 'max_height', 'max_pixels']):
            raise ValueError('fixed typography and canvas limits must be integers')
        if not (6 <= self.font_size <= 96 and 6 <= self.heading_size <= 96
                and 32 <= self.min_width <= self.max_width <= 4096
                and 32 <= self.max_height <= 32768 and 1 <= self.max_pixels <= 32_000_000):
            raise ValueError('invalid fixed typography or canvas limits')


def engine_hashes():
    return {name: digest(Path(__file__).parent / name) for name in ENGINE_FILES}


def validate_source(source):
    """A restricted Python subset: layout arithmetic/control flow and font metrics only.

    This restriction plus process limits contains accidental generated-code failures;
    it is not an OS security sandbox for adversarial Python.
    """
    if not isinstance(source, str) or len(source) > 30_000:
        raise ValueError('layout source must be at most 30000 characters')
    tree = ast.parse(source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != 'layout' or tree.body[0].decorator_list):
        raise ValueError('source must contain exactly def layout(document, metrics, constraints)')
    fn = tree.body[0]
    if ([a.arg for a in fn.args.args] != ['document', 'metrics', 'constraints']
            or fn.args.defaults or fn.args.kw_defaults or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs or fn.returns
            or any(a.annotation for a in fn.args.args)):
        raise ValueError('layout must have three plain arguments without defaults or annotations')
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda,
                 ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
                 ast.Delete, ast.Yield, ast.YieldFrom, ast.Await)
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or (isinstance(node, ast.FunctionDef) and node is not fn):
            raise ValueError('unsupported layout Python statement: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('private names are not allowed')
        if isinstance(node, ast.Attribute):
            if (not isinstance(node.value, ast.Name) or node.value.id != 'metrics'
                    or node.attr not in {'flow', 'width'} or not isinstance(node.ctx, ast.Load)):
                raise ValueError('only metrics.flow and metrics.width attributes are available')
    return tree


def row_text(row, fields):
    parts, previous = [], None
    for field_id, start, end in row:
        if previous is not None and previous != field_id:
            parts.append(' ')
        parts.append(fields[field_id][start:end])
        previous = field_id
    return ''.join(parts)


class FontMetrics:
    def __init__(self, constraints):
        self.font = load_font(constraints.font_path, constraints.font_size)

    def width(self, text):
        box = self.font.getbbox(text, anchor='lt')
        return max(self.font.getlength(text), box[2]) - min(0, box[0])

    def flow(self, fields, width, pack=False):
        """Return exact [field_id,start,end] spans; long words also wrap without loss."""
        if not isinstance(width, (int, float)) or not math.isfinite(width) or width < 32:
            raise ValueError('flow width must be finite and >=32')
        if not fields:
            return []
        texts = {f['id']: f['text'] for f in fields}
        rows, current = [], []
        for field in fields:
            if not pack and current:
                rows.append(current)
                current = []
            text, start = field['text'], 0
            while start < len(text):
                low, high = start, len(text)
                while low < high:
                    mid = (low + high + 1) // 2
                    proposal = current + [[field['id'], start, mid]]
                    if self.width(row_text(proposal, texts)) <= width:
                        low = mid
                    else:
                        high = mid - 1
                end = low
                if end == start:
                    if not current:
                        raise ValueError('a glyph is wider than the available width')
                    rows.append(current)
                    current = []
                    continue
                if end < len(text):
                    boundary = text.rfind(' ', start, end + 1)
                    if boundary > start:
                        end = boundary + 1
                        # A trailing space can itself exceed the line width.
                        if self.width(row_text(current + [[field['id'], start, end]], texts)) > width:
                            end = boundary
                current.append([field['id'], start, end])
                start = end
                if start < len(text):
                    rows.append(current)
                    current = []
        if current:
            rows.append(current)
        return rows


def execute_layout(source, documents, constraints, timeout=20):
    """Execute candidates in a separate CPU-limited process without inherited API keys."""
    validate_source(source)
    payload = dict(source=source, documents=documents, constraints=asdict(constraints))
    env = {'PYTHONPATH': str(ROOT), 'PYTHONIOENCODING': 'utf-8'}
    with tempfile.TemporaryFile(mode='w+') as output, tempfile.TemporaryFile(mode='w+') as error:
        result = subprocess.run([sys.executable, '-m', 'optical_compression.layout_worker'],
            input=json.dumps(payload), text=True, stdout=output, stderr=error,
            cwd=ROOT, env=env, timeout=timeout)
        output.seek(0)
        error.seek(0)
        content = output.read(8_000_001)
        if result.returncode or len(content) > 8_000_000:
            raise ValueError('layout worker failed: ' + error.read(2000))
    plans = json.loads(content)
    if not isinstance(plans, list) or len(plans) != len(documents):
        raise ValueError('layout worker returned the wrong number of plans')
    return plans


def validate_rows(rows, fields):
    texts = {f['id']: f['text'] for f in fields}
    coverage = {key: 0 for key in texts}
    completed, active = set(), None
    if not isinstance(rows, list) or not rows or len(rows) > 20_000:
        raise ValueError('expect a nonempty bounded row list')
    for row in rows:
        if not isinstance(row, list) or not row or len(row) > len(fields) + 1:
            raise ValueError('invalid row')
        for span in row:
            if not isinstance(span, list) or len(span) != 3:
                raise ValueError('each text span is [field_id,start,end]')
            key, start, end = span
            if (key not in texts or type(start) is not int or type(end) is not int
                    or start != coverage[key] or not start < end <= len(texts[key])):
                raise ValueError('text lost, changed, duplicated or reordered within a field')
            if key != active:
                if active is not None:
                    completed.add(active)
                if key in completed:
                    raise ValueError('fields may be reordered but not interleaved')
                active = key
            coverage[key] = end
    if coverage != {key: len(value) for key, value in texts.items()}:
        raise ValueError('all original text characters must appear exactly once')
    return [row_text(row, texts) for row in rows]


def render_plan(doc, plan, constraints, path):
    if not isinstance(plan, dict) or set(plan) != {'width', 'padding', 'line_gap', 'section_gap', 'user_rows', 'item_rows'}:
        raise ValueError('unexpected layout plan keys')
    for key in ['width', 'padding', 'line_gap', 'section_gap']:
        if type(plan[key]) is not int:
            raise ValueError('layout dimensions must be integers')
    width, pad = plan['width'], plan['padding']
    if not (constraints.min_width <= width <= constraints.max_width and 1 <= pad <= width // 4
            and 0 <= plan['line_gap'] <= 64 and 0 <= plan['section_gap'] <= 256):
        raise ValueError('layout dimensions violate constraints')
    texts = {section: validate_rows(plan[section + '_rows'], doc[section]) for section in ['user', 'item']}
    body = load_font(constraints.font_path, constraints.font_size)
    heading = load_font(constraints.font_path, constraints.heading_size)
    for char in set(''.join(f['text'] for fields in doc.values() for f in fields)):
        if not supported_glyph(constraints.font_path, constraints.font_size, char):
            raise ValueError('fixed font has no glyph for ' + repr(char))
    if not all(supported_glyph(constraints.font_path, constraints.heading_size, c)
               for c in 'USER PROFILECANDIDATE ITEM'):
        raise ValueError('fixed font cannot render the section headings')
    commands, y = [], pad
    for section, title in [('user', 'USER PROFILE'), ('item', 'CANDIDATE ITEM')]:
        if section == 'item':
            y += plan['section_gap']
        for text, fnt in [(title, heading)] + [(t, body) for t in texts[section]]:
            box = fnt.getbbox(text, anchor='lt')
            ink_width = max(fnt.getlength(text), box[2]) - min(0, box[0])
            if ink_width > width - 2 * pad:
                raise ValueError('text exceeds page width')
            commands.append((pad - min(0, box[0]), y, text, fnt))
            y += max(1, box[3] - box[1]) + plan['line_gap']
    height = y + pad
    if height > constraints.max_height or height * width > constraints.max_pixels:
        raise ValueError('page exceeds canvas limits')
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    for x, y, text, fnt in commands:
        draw.text((x, y), text, font=fnt, fill='black', anchor='lt')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    return dict(width=width, height=height, image_sha256=digest(path),
                content_preserved=True, geometry_valid=True, sections=texts)


class FrozenLayout:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        metadata_path = self.directory / 'layout.json'
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata['format'] != 'zip4rec-top-bottom-v1' or not self.metadata['qualified']:
            raise ValueError('layout is not a qualified frozen strategy')
        if self.metadata['engine_sha256'] != engine_hashes():
            raise ValueError('layout engine changed; revalidate and freeze the strategy again')
        if not {'layout.py', 'fonts/body.ttf'} <= set(self.metadata['files_sha256']):
            raise ValueError('frozen layout must fingerprint both source and font')
        for name, expected in self.metadata['files_sha256'].items():
            target = (self.directory / name).resolve()
            if not target.is_relative_to(self.directory) or digest(target) != expected:
                raise ValueError('frozen layout asset changed: ' + name)
        cfg = dict(self.metadata['constraints'])
        if cfg['font_path'] != 'fonts/body.ttf':
            raise ValueError('frozen typography must reference its bundled font')
        cfg['font_path'] = str(self.directory / cfg['font_path'])
        self.constraints = Constraints(**cfg)
        self.source = (self.directory / 'layout.py').read_text()
        validate_source(self.source)
        self.signature = digest(metadata_path)

    def render_tasks(self, tasks):
        from .document import document
        output = []
        for start in range(0, len(tasks), 32):
            batch = tasks[start:start + 32]
            docs = [document(t['profile'], t['item']) for t in batch]
            plans = execute_layout(self.source, docs, self.constraints)
            for task, doc, plan in zip(batch, docs, plans):
                result = render_plan(doc, plan, self.constraints, task['path'])
                result.pop('sections')
                output.append(dict(sample_id=task['sample_id'], **result))
        return output
