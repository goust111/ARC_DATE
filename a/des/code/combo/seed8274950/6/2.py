from __future__ import annotations

from collections import Counter, deque

import numpy as np


RED = 2
_GEN_RETRIES = 300
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_COLOR_POOL = tuple(c for c in range(10) if c != RED)
_RNG = np.random.default_rng()


def _mode_color(values) -> int:
    counts = Counter(int(v) for v in values)
    return int(counts.most_common(1)[0][0])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _spans_from_lines(lines: list[int], size: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(set(int(x) for x in lines)) + [int(size)]
    spans: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            spans.append((int(a + 1), int(b - 1)))
    return spans


def _find_cell_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    h, w = grid.shape
    row_lines: dict[int, list[int]] = {}
    col_lines: dict[int, list[int]] = {}

    for r in range(h):
        color = int(grid[r, 0])
        if np.all(grid[r, :] == color):
            row_lines.setdefault(color, []).append(int(r))
    for c in range(w):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color):
            col_lines.setdefault(color, []).append(int(c))

    best = None
    for color in set(row_lines) & set(col_lines):
        row_spans = _spans_from_lines(row_lines[color], h)
        col_spans = _spans_from_lines(col_lines[color], w)
        if not row_spans or not col_spans:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        cell_h = next(iter(row_sizes))
        cell_w = next(iter(col_sizes))
        if cell_h != cell_w:
            continue
        score = (
            len(row_spans) * len(col_spans),
            len(row_lines[color]) + len(col_lines[color]),
        )
        candidate = (score, int(color), row_spans, col_spans, int(cell_h))
        if best is None or score > best[0]:
            best = candidate

    assert best is not None, "failed to detect the board grid"
    _, line_color, row_spans, col_spans, cell_size = best
    return int(line_color), row_spans, col_spans, int(cell_size)


def _infer_background(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> int:
    perimeter = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            perimeter.extend(int(grid[r0, c]) for c in range(c0, c1 + 1))
            if r1 != r0:
                perimeter.extend(int(grid[r1, c]) for c in range(c0, c1 + 1))
            for r in range(r0 + 1, r1):
                perimeter.append(int(grid[r, c0]))
                if c1 != c0:
                    perimeter.append(int(grid[r, c1]))
    assert perimeter, "no cell perimeter pixels found"
    return _mode_color(perimeter)


def _component_has_hole(cell: np.ndarray, bg: int, comp: list[tuple[int, int]]) -> bool:
    h, w = cell.shape
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        blocked[r, c] = True

    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    for r in range(h):
        for c in range(w):
            if int(cell[r, c]) == bg and not blocked[r, c] and not seen[r, c]:
                return True
    return False


def _cell_has_ring(cell: np.ndarray, bg: int) -> bool:
    fg_mask = np.asarray(cell != bg, dtype=bool)
    if not fg_mask.any():
        return False
    for comp in _components4(fg_mask):
        if _component_has_hole(cell, int(bg), comp):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the board cells whose interior pattern contains a closed hole and encode
    their positions in a reduced output grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _line_color, row_spans, col_spans, _cell_size = _find_cell_layout(g)
    bg = _infer_background(g, row_spans, col_spans)

    out = np.full((len(row_spans), len(col_spans)), bg, dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            if _cell_has_ring(cell, bg):
                out[rr, cc] = RED
    return out


def _frame_template(h: int, w: int) -> np.ndarray:
    tpl = np.zeros((h, w), dtype=bool)
    tpl[0, :] = True
    tpl[-1, :] = True
    tpl[:, 0] = True
    tpl[:, -1] = True
    return tpl


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _build_ring_frame(avail: int, rng: np.random.Generator) -> np.ndarray:
    h = _randint(rng, 3, avail)
    w = _randint(rng, 3, avail)
    return _frame_template(h, w)


def _build_double_hole(avail: int, rng: np.random.Generator) -> np.ndarray:
    h = _randint(rng, 5, avail)
    w = _randint(rng, 5, avail)
    tpl = _frame_template(h, w)
    if rng.random() < 0.5:
        tpl[1:-1, w // 2] = True
    else:
        tpl[h // 2, 1:-1] = True
    return tpl


def _build_ring_with_tail(avail: int, rng: np.random.Generator) -> np.ndarray:
    core_h = _randint(rng, 3, avail - 1)
    core_w = _randint(rng, 3, avail)
    tpl = np.zeros((core_h + 1, core_w), dtype=bool)
    tpl[1, :] = True
    tpl[-1, :] = True
    tpl[1:, 0] = True
    tpl[1:, -1] = True
    tpl[0, core_w // 2] = True
    return tpl


def _build_solid(avail: int, rng: np.random.Generator) -> np.ndarray:
    h = _randint(rng, 2, min(avail, 4))
    w = _randint(rng, 2, min(avail, 4))
    return np.ones((h, w), dtype=bool)


def _build_l_shape(avail: int, rng: np.random.Generator) -> np.ndarray:
    h = _randint(rng, 3, avail)
    w = _randint(rng, 3, avail)
    tpl = np.zeros((h, w), dtype=bool)
    tpl[:, 0] = True
    tpl[-1, :] = True
    if h >= 4 and w >= 4 and rng.random() < 0.5:
        tpl[:, 1] = True
    return tpl


def _build_plus(avail: int, rng: np.random.Generator) -> np.ndarray:
    odds = [x for x in range(3, avail + 1) if x % 2 == 1]
    h = int(rng.choice(odds))
    w = int(rng.choice(odds))
    tpl = np.zeros((h, w), dtype=bool)
    tpl[h // 2, :] = True
    tpl[:, w // 2] = True
    return tpl


def _build_open_frame(avail: int, rng: np.random.Generator) -> np.ndarray:
    lo = 4 if avail >= 4 else 3
    h = _randint(rng, lo, avail)
    w = _randint(rng, lo, avail)
    tpl = _frame_template(h, w)
    gap = 1 if w <= 4 else _randint(rng, 1, min(2, w - 2))
    start = (w - gap) // 2
    tpl[0, start : start + gap] = False
    return tpl


def _build_t_shape(avail: int, rng: np.random.Generator) -> np.ndarray:
    h = _randint(rng, 3, avail)
    w = _randint(rng, 3, avail)
    tpl = np.zeros((h, w), dtype=bool)
    tpl[0, :] = True
    tpl[:, w // 2] = True
    return tpl


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(mask, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, k=int(rng.integers(0, 4)))
    return np.array(out, copy=True)


def _embed_template(template: np.ndarray, cell_size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = template.shape
    assert h <= cell_size - 2 and w <= cell_size - 2
    top = 1 if cell_size - h == 1 else int(rng.integers(1, cell_size - h))
    left = 1 if cell_size - w == 1 else int(rng.integers(1, cell_size - w))
    mask = np.zeros((cell_size, cell_size), dtype=bool)
    mask[top : top + h, left : left + w] = template
    return mask


def _make_cell_mask(cell_size: int, want_ring: bool, rng: np.random.Generator) -> np.ndarray:
    avail = cell_size - 2
    builders = []
    if want_ring:
        builders.append(_build_ring_frame)
        if avail >= 4:
            builders.append(_build_ring_with_tail)
        if avail >= 5:
            builders.append(_build_double_hole)
    else:
        builders.extend((_build_solid, _build_l_shape, _build_plus, _build_open_frame, _build_t_shape))

    for _ in range(80):
        builder = builders[int(rng.integers(0, len(builders)))]
        template = _random_transform(builder(avail, rng), rng)
        if template.shape[0] > avail or template.shape[1] > avail:
            continue
        mask = _embed_template(template, cell_size, rng)
        if len(_components4(mask)) != 1:
            continue
        probe = np.zeros((cell_size, cell_size), dtype=int)
        probe[mask] = 1
        if _cell_has_ring(probe, 0) != want_ring:
            continue
        return mask
    raise RuntimeError("failed to build a valid cell pattern")


def _layout_from_params(count: int, cell_size: int, bordered: bool) -> tuple[int, list[tuple[int, int]], list[int]]:
    if bordered:
        spans = [(1 + i * (cell_size + 1), 1 + i * (cell_size + 1) + cell_size - 1) for i in range(count)]
        lines = [i * (cell_size + 1) for i in range(count + 1)]
        total = count * cell_size + count + 1
    else:
        spans = [(i * (cell_size + 1), i * (cell_size + 1) + cell_size - 1) for i in range(count)]
        lines = [cell_size + i * (cell_size + 1) for i in range(count - 1)]
        total = count * cell_size + count - 1
    return int(total), spans, lines


def _build_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    rows = _randint(rng, 2, 5)
    cols = _randint(rng, 2, 5)
    cell_size = _randint(rng, 5, 8)
    bordered = bool(rng.integers(0, 2))

    bg = int(rng.choice(_COLOR_POOL))
    line_choices = [c for c in _COLOR_POOL if c != bg]
    line_color = int(rng.choice(line_choices))
    fg_choices = [c for c in _COLOR_POOL if c not in {bg, line_color}]

    total_h, row_spans, row_lines = _layout_from_params(rows, cell_size, bordered)
    total_w, col_spans, col_lines = _layout_from_params(cols, cell_size, bordered)

    grid = np.full((total_h, total_w), bg, dtype=int)
    if bordered:
        grid[:, :] = line_color
    for r in row_lines:
        grid[r, :] = line_color
    for c in col_lines:
        grid[:, c] = line_color
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            grid[r0 : r1 + 1, c0 : c1 + 1] = bg

    flags = rng.random((rows, cols)) < 0.5
    if flags.all():
        flags[int(rng.integers(0, rows)), int(rng.integers(0, cols))] = False
    if not flags.any():
        flags[int(rng.integers(0, rows)), int(rng.integers(0, cols))] = True

    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            want_ring = bool(flags[rr, cc])
            mask = _make_cell_mask(cell_size, want_ring, rng)
            color = int(rng.choice(fg_choices))
            cell = np.full((cell_size, cell_size), bg, dtype=int)
            cell[mask] = color
            if _cell_has_ring(cell, bg) != want_ring:
                return None
            grid[r0 : r1 + 1, c0 : c1 + 1] = cell

    expected = np.full((rows, cols), bg, dtype=int)
    expected[flags] = RED

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _build_example(_RNG)
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid puzzle after {_GEN_RETRIES} attempts")
