from __future__ import annotations

import random

import numpy as np


def _mode_color(values: np.ndarray) -> int:
    vals, cnts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _run_groups(flags: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(flags.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            groups.append((start, idx - 1))
            start = None
    if start is not None:
        groups.append((start, len(flags) - 1))
    return groups


def _complement_spans(groups: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for start, end in groups:
        if start > prev + 1:
            spans.append((prev + 1, start - 1))
        prev = end
    if prev < limit - 1:
        spans.append((prev + 1, limit - 1))
    if any(lo > hi for lo, hi in spans):
        return []
    return spans


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "mask must be non-empty"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _is_rotation_of(src: np.ndarray, dst: np.ndarray) -> bool:
    for k in range(4):
        rot = np.rot90(src, k)
        if rot.shape == dst.shape and np.array_equal(rot, dst):
            return True
    return False


def _extract_cell(cell: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, int]:
    mask = cell != bg
    assert mask.any(), "every cell must contain a foreground pattern"
    assert _component_count(mask) == 1, "cell pattern must be one connected component"
    colors = [int(v) for v in np.unique(cell[mask])]
    assert len(colors) == 1, "cell pattern must use exactly one foreground color"
    return mask, _crop_mask(mask), colors[0]


def _find_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g))
    best: tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None = None
    best_score = -1

    for sep in colors:
        row_groups = _run_groups(np.all(g == sep, axis=1))
        col_groups = _run_groups(np.all(g == sep, axis=0))
        if not row_groups or not col_groups:
            continue

        row_spans = _complement_spans(row_groups, g.shape[0])
        col_spans = _complement_spans(col_groups, g.shape[1])
        if len(row_spans) < 2 or len(col_spans) < 3:
            continue

        row_sizes = {hi - lo + 1 for lo, hi in row_spans}
        col_sizes = {hi - lo + 1 for lo, hi in col_spans}
        row_thickness = {hi - lo + 1 for lo, hi in row_groups}
        col_thickness = {hi - lo + 1 for lo, hi in col_groups}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if len(row_thickness) != 1 or len(col_thickness) != 1:
            continue

        non_sep = g[g != sep]
        if non_sep.size == 0:
            continue
        bg = _mode_color(non_sep)
        if bg == sep:
            continue

        ok = True
        for r0, r1 in row_spans:
            ref_mask, ref_crop, _ref_color = _extract_cell(g[r0 : r1 + 1, col_spans[0][0] : col_spans[0][1] + 1], bg)
            if not ref_mask.any():
                ok = False
                break
            for c0, c1 in col_spans[1:]:
                _mask, crop, _color = _extract_cell(g[r0 : r1 + 1, c0 : c1 + 1], bg)
                if not _is_rotation_of(crop, ref_crop):
                    ok = False
                    break
            if not ok:
                break

        if not ok:
            continue

        score = len(row_spans) * len(col_spans)
        if score > best_score:
            best_score = score
            best = (sep, bg, row_spans, col_spans)

    assert best is not None, "failed to find a regular separator grid"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every non-reference sprite in each partition row to match the leftmost sprite orientation.

    Input structure:
    - A single separator color forms full horizontal and vertical divider lines that split the canvas into equal-sized cells.
    - Each cell contains background plus one connected monochrome asymmetric sprite.
    - Within any one row of cells, all sprites share the same base shape up to 0/90/180/270-degree rotation.
    - The leftmost cell of each row is the reference cell for that row's target orientation.

    Rule:
    - Detect the divider grid and preserve the separator lines and background.
    - For each row of cells, read the leftmost sprite as the reference orientation.
    - In every other cell of that row, keep the cell's own foreground color but redraw the sprite so its occupied cells exactly match the reference cell's occupied cells.
    - Leave the reference cells unchanged.

    Key insight:
    - The leftmost cell is not just another example: it defines the orientation template that every other same-row sprite must rotate into.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _sep, bg, row_spans, col_spans = _find_structure(g)
    out = g.copy()

    for r0, r1 in row_spans:
        ref_cell = g[r0 : r1 + 1, col_spans[0][0] : col_spans[0][1] + 1]
        ref_mask, _ref_crop, _ref_color = _extract_cell(ref_cell, bg)

        for c0, c1 in col_spans[1:]:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            _mask, _crop, color = _extract_cell(cell, bg)
            patch = np.full(cell.shape, bg, dtype=int)
            patch[ref_mask] = color
            out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _rotation_signature(mask: np.ndarray) -> tuple[tuple[int, int, bytes], ...]:
    norm = _crop_mask(mask)
    rots = []
    for k in range(4):
        rot = np.rot90(norm, k)
        rots.append((int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes()))
    return tuple(sorted(set(rots)))


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 0], [1, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
        [[1, 0, 1], [1, 1, 1]],
        [[1, 1, 1, 0], [0, 1, 0, 0]],
        [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1]],
    ]

    seen: set[tuple[tuple[int, int, bytes], ...]] = set()
    out: list[np.ndarray] = []
    for rows in raw:
        mask = np.array(rows, dtype=bool)
        if _component_count(mask) != 1:
            continue
        sig = _rotation_signature(mask)
        if len(sig) != 4 or sig in seen:
            continue
        seen.add(sig)
        out.append(mask)
    return out


_SHAPES = _build_shape_library()
_GEN_ATTEMPTS = 400


def _sample_layout() -> tuple[int, int, int, int, bool] | None:
    rows = random.randint(2, 4)
    cols = random.randint(3, 5)
    cell_h = random.randint(5, 7)
    cell_w = random.randint(5, 7)
    border = random.choice([False, True])
    h = rows * cell_h + (rows + 1 if border else rows - 1)
    w = cols * cell_w + (cols + 1 if border else cols - 1)
    if h > 30 or w > 30:
        return None
    if min(cell_h, cell_w) < 5:
        return None
    return rows, cols, cell_h, cell_w, border


def _make_spans(count: int, size: int, border: bool) -> tuple[list[int], list[tuple[int, int]], int]:
    seps: list[int] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    if border:
        seps.append(0)
        pos = 1
    for idx in range(count):
        spans.append((pos, pos + size - 1))
        pos += size
        if border or idx < count - 1:
            seps.append(pos)
            pos += 1
    return seps, spans, pos


def _embed_shape(shape: np.ndarray, cell_h: int, cell_w: int, rotation: int) -> np.ndarray:
    inner = min(cell_h, cell_w) - 2
    square = np.zeros((inner, inner), dtype=bool)
    h, w = shape.shape
    r0 = (inner - h) // 2
    c0 = (inner - w) // 2
    square[r0 : r0 + h, c0 : c0 + w] = shape
    square = np.rot90(square, rotation)

    canvas = np.zeros((cell_h, cell_w), dtype=bool)
    rr = (cell_h - inner) // 2
    cc = (cell_w - inner) // 2
    canvas[rr : rr + inner, cc : cc + inner] = square
    return canvas


def _choose_row_colors(cols: int, palette: list[int]) -> list[int]:
    mode = random.choice(["per_cell", "per_row", "two_tone"])
    if mode == "per_row":
        color = random.choice(palette)
        return [color] * cols
    if mode == "two_tone":
        a, b = random.sample(palette, 2)
        return [random.choice([a, b]) for _ in range(cols)]
    return [random.choice(palette) for _ in range(cols)]


def _sample_row_shape(inner_limit: int, used: set[tuple[tuple[int, int, bytes], ...]]) -> np.ndarray | None:
    choices = [shape for shape in _SHAPES if max(shape.shape) <= inner_limit and _rotation_signature(shape) not in used]
    if not choices:
        return None
    shape = random.choice(choices)
    used.add(_rotation_signature(shape))
    return shape


def _paint_patch(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, bg: int, color: int) -> None:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] = patch


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    layout = _sample_layout()
    if layout is None:
        return None
    rows, cols, cell_h, cell_w, border = layout
    inner_limit = min(cell_h, cell_w) - 2
    if inner_limit < 3:
        return None

    palette = list(range(10))
    bg, sep = random.sample(palette, 2)
    fg_palette = [color for color in palette if color not in {bg, sep}]

    row_sep_idxs, row_spans, height = _make_spans(rows, cell_h, border)
    col_sep_idxs, col_spans, width = _make_spans(cols, cell_w, border)
    if height > 30 or width > 30:
        return None

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    inp[row_sep_idxs, :] = sep
    inp[:, col_sep_idxs] = sep
    out[row_sep_idxs, :] = sep
    out[:, col_sep_idxs] = sep

    used_shapes: set[tuple[tuple[int, int, bytes], ...]] = set()
    for row_idx, (r0, r1) in enumerate(row_spans):
        del row_idx  # row-specific randomization is handled below.
        shape = _sample_row_shape(inner_limit, used_shapes)
        if shape is None:
            return None

        ref_rot = random.randrange(4)
        rotations = [ref_rot]
        while True:
            rotations = [ref_rot] + [random.randrange(4) for _ in range(cols - 1)]
            if any(rot != ref_rot for rot in rotations[1:]):
                break

        row_colors = _choose_row_colors(cols, fg_palette)
        ref_mask = _embed_shape(shape, cell_h, cell_w, ref_rot)
        for col_idx, (c0, c1) in enumerate(col_spans):
            del c1
            color = row_colors[col_idx]
            cur_mask = _embed_shape(shape, cell_h, cell_w, rotations[col_idx])
            _paint_patch(inp, r0, c0, cur_mask, bg, color)
            _paint_patch(out, r0, c0, ref_mask, bg, color)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
