import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _frame_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid[grid != bg].reshape(-1)))
    assert counts, "grid must contain foreground"
    return int(counts.most_common(1)[0][0])


def _line_positions(full_line_mask: np.ndarray) -> list[int]:
    idx = [int(i) for i, ok in enumerate(full_line_mask) if ok]
    assert len(idx) >= 2
    return idx


def _spans(lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    assert spans
    return spans


def _parse_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int) -> dict:
    r0, r1, c0, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    rows = _line_positions(np.all(patch == frame_color, axis=1))
    cols = _line_positions(np.all(patch == frame_color, axis=0))
    assert rows[0] == 0 and rows[-1] == patch.shape[0] - 1
    assert cols[0] == 0 and cols[-1] == patch.shape[1] - 1
    row_spans = _spans(rows)
    col_spans = _spans(cols)
    return {
        "bbox": bbox,
        "row_spans": row_spans,
        "col_spans": col_spans,
    }


def _frame_objects(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    objs: list[dict] = []
    for comp in _components(grid == frame_color):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        bbox = (min(rs), max(rs), min(cs), max(cs))
        try:
            objs.append(_parse_frame(grid, bbox, frame_color))
        except AssertionError:
            continue
    assert len(objs) >= 2, "need at least two frames"
    return objs


def _cell_color(patch: np.ndarray, bg: int, frame_color: int) -> int:
    values = set(map(int, np.unique(patch)))
    values.discard(bg)
    values.discard(frame_color)
    assert len(values) <= 1
    return bg if not values else int(next(iter(values)))


def _read_template(grid: np.ndarray, frame: dict, bg: int, frame_color: int) -> np.ndarray:
    r0, _, c0, _ = frame["bbox"]
    rows = frame["row_spans"]
    cols = frame["col_spans"]
    template = np.full((len(rows), len(cols)), bg, dtype=int)
    for i, (rr0, rr1) in enumerate(rows):
        for j, (cc0, cc1) in enumerate(cols):
            cell = grid[r0 + rr0 : r0 + rr1 + 1, c0 + cc0 : c0 + cc1 + 1]
            template[i, j] = _cell_color(cell, bg, frame_color)
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the logical slot-color template from the one marked frame into all other framed grids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    frame_color = _frame_color(g, bg)
    frames = _frame_objects(g, bg, frame_color)

    shape_set = {
        (len(frame["row_spans"]), len(frame["col_spans"]))
        for frame in frames
    }
    assert len(shape_set) == 1, "all frames must have same logical shape"

    templates = [_read_template(g, frame, bg, frame_color) for frame in frames]
    marked = [idx for idx, tpl in enumerate(templates) if np.any(tpl != bg)]
    assert len(marked) == 1, "expected exactly one marked frame"
    template = templates[marked[0]]

    out = g.copy()
    for idx, frame in enumerate(frames):
        if idx == marked[0]:
            continue
        r0, _, c0, _ = frame["bbox"]
        for i, (rr0, rr1) in enumerate(frame["row_spans"]):
            for j, (cc0, cc1) in enumerate(frame["col_spans"]):
                color = int(template[i, j])
                out[r0 + rr0 : r0 + rr1 + 1, c0 + cc0 : c0 + cc1 + 1] = color
    return out


MAX_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (18, 34)
GRID_WIDTH_RANGE = (18, 34)
FRAME_COUNT_RANGE = (2, 5)
LOGICAL_ROWS = (2, 3)
LOGICAL_COLS = (2, 3)
CELL_SIZE_RANGE = (2, 5)
MARGIN = 1
GAP_RANGE = (2, 4)


def _build_frame(
    rows: int,
    cols: int,
    cell_heights: list[int],
    cell_widths: list[int],
    bg: int,
    frame_color: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h = sum(cell_heights) + rows + 1
    w = sum(cell_widths) + cols + 1
    patch = np.full((h, w), bg, dtype=int)
    row_lines = [0]
    pos = 0
    row_spans: list[tuple[int, int]] = []
    for ch in cell_heights:
        row_spans.append((pos + 1, pos + ch))
        pos += ch + 1
        row_lines.append(pos)
    col_lines = [0]
    pos = 0
    col_spans: list[tuple[int, int]] = []
    for cw in cell_widths:
        col_spans.append((pos + 1, pos + cw))
        pos += cw + 1
        col_lines.append(pos)
    patch[row_lines, :] = frame_color
    patch[:, col_lines] = frame_color
    return patch, row_spans, col_spans


def _sample_sizes(count: int) -> list[int]:
    return [random.randint(*CELL_SIZE_RANGE) for _ in range(count)]


def _frame_dims(rows: int, cols: int) -> tuple[int, int, list[int], list[int]]:
    hs = _sample_sizes(rows)
    ws = _sample_sizes(cols)
    return sum(hs) + rows + 1, sum(ws) + cols + 1, hs, ws


def _place_bboxes(
    grid_h: int,
    grid_w: int,
    dims: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    positions: list[tuple[int, int]] = []
    order = list(range(len(dims)))
    random.shuffle(order)
    placed: dict[int, tuple[int, int]] = {}
    for idx in order:
        h, w = dims[idx]
        candidates = []
        for r0 in range(MARGIN, grid_h - h - MARGIN + 1):
            for c0 in range(MARGIN, grid_w - w - MARGIN + 1):
                r1, c1 = r0 + h, c0 + w
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(grid_h, r1 + 1)
                cc1 = min(grid_w, c1 + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placed[idx] = pos
        r0, c0 = pos
        occupied[r0 : r0 + h, c0 : c0 + w] = True
    for idx in range(len(dims)):
        positions.append(placed[idx])
    return positions


def _paint_frame(
    grid: np.ndarray,
    frame_patch: np.ndarray,
    top: int,
    left: int,
) -> None:
    h, w = frame_patch.shape
    grid[top : top + h, left : left + w] = frame_patch


def _template_matrix(rows: int, cols: int, bg: int, palette: list[int]) -> np.ndarray:
    total = rows * cols
    fill_count = random.randint(1, total - (1 if total > 1 else 0))
    chosen = random.sample(range(total), fill_count)
    tpl = np.full((rows, cols), bg, dtype=int)
    for k in chosen:
        tpl[k // cols, k % cols] = random.choice(palette)
    if np.all(tpl != bg):
        rr = random.randrange(rows)
        cc = random.randrange(cols)
        tpl[rr, cc] = bg
    return tpl


def _render_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    frame_color = random.choice([c for c in range(1, 10)])
    other_colors = [c for c in range(1, 10) if c != frame_color]
    palette = random.sample(other_colors, random.randint(1, min(4, len(other_colors))))

    logical_rows = random.randint(*LOGICAL_ROWS)
    logical_cols = random.randint(*LOGICAL_COLS)
    frame_count = random.randint(*FRAME_COUNT_RANGE)

    frame_specs = []
    dims = []
    for _ in range(frame_count):
        fh, fw, hs, ws = _frame_dims(logical_rows, logical_cols)
        frame_specs.append((hs, ws))
        dims.append((fh, fw))

    grid_h = max(sum(h for h, _ in dims) + (frame_count + 1) * MARGIN, random.randint(*GRID_HEIGHT_RANGE))
    grid_w = max(max(w for _, w in dims) + 2 * MARGIN, random.randint(*GRID_WIDTH_RANGE))

    extra_h = sum(h for h, _ in dims) + (frame_count - 1) * random.randint(*GAP_RANGE) + 2 * MARGIN
    extra_w = sum(w for _, w in dims) + (frame_count - 1) * random.randint(*GAP_RANGE) + 2 * MARGIN
    grid_h = max(grid_h, min(max(GRID_HEIGHT_RANGE[1], extra_h), 40))
    grid_w = max(grid_w, min(max(GRID_WIDTH_RANGE[1], extra_w), 40))

    if grid_h < max(h for h, _ in dims) + 2 * MARGIN or grid_w < max(w for _, w in dims) + 2 * MARGIN:
        return None

    positions = _place_bboxes(grid_h, grid_w, dims)
    if positions is None:
        return None

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    template_idx = random.randrange(frame_count)
    template = _template_matrix(logical_rows, logical_cols, bg, palette)

    for idx, ((hs, ws), (top, left)) in enumerate(zip(frame_specs, positions)):
        patch, row_spans, col_spans = _build_frame(logical_rows, logical_cols, hs, ws, bg, frame_color)
        if idx == template_idx:
            for i, (r0, r1) in enumerate(row_spans):
                for j, (c0, c1) in enumerate(col_spans):
                    color = int(template[i, j])
                    if color != bg:
                        patch[r0 : r1 + 1, c0 : c1 + 1] = color
        _paint_frame(inp, patch, top, left)

    try:
        out = _render_output(inp)
    except Exception:
        return None

    if np.array_equal(out, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
