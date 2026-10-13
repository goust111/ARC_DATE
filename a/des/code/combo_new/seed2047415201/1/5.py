from collections import Counter
import random

import numpy as np


BLACK = 0
YELLOW = 4
FOREGROUND_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 500


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    vals = [int(v) for v in indices]
    if not vals:
        return []
    out: list[tuple[int, int]] = []
    start = prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = prev = value
    out.append((start, prev))
    return out


def _spans(size: int, separator_runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for lo, hi in separator_runs:
        assert start < lo, "separator line must leave a nonempty panel"
        spans.append((start, lo))
        start = hi + 1
    assert start < size, "separator line must leave a nonempty panel"
    spans.append((start, size))
    return spans


def _panel_spans(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = np.where(np.all(g == YELLOW, axis=1))[0]
    sep_cols = np.where(np.all(g == YELLOW, axis=0))[0]
    assert sep_rows.size > 0 and sep_cols.size > 0, "expected yellow row and column separators"
    row_spans = _spans(g.shape[0], _runs(sep_rows))
    col_spans = _spans(g.shape[1], _runs(sep_cols))
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected a multi-panel grid"
    return row_spans, col_spans


def _components(panel: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, panel.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for sr in range(h):
        for sc in range(w):
            color = int(panel[sr, sc])
            if seen[sr, sc] or color in (BLACK, YELLOW):
                continue

            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(panel[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
            norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
            crop = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
            for r, c in coords:
                crop[r - r0, c - c0] = int(panel[r, c])
            comps.append({"shape": norm, "crop": crop, "bbox": (r0, c0, r1, c1)})

    return comps


def _outlier_crop(panel: np.ndarray) -> np.ndarray:
    comps = _components(panel)
    assert 3 <= len(comps) <= 5, "each panel must contain three to five objects"
    counts = Counter(comp["shape"] for comp in comps)
    assert len(counts) == 2, "expected exactly one common shape and one outlier shape"
    outlier_shapes = [shape for shape, count in counts.items() if count == 1]
    assert len(outlier_shapes) == 1, "the outlier shape must be unique"
    assert max(counts.values()) == len(comps) - 1, "all non-outlier objects must share one shape"
    outlier_shape = outlier_shapes[0]
    matches = [comp for comp in comps if comp["shape"] == outlier_shape]
    assert len(matches) == 1
    return matches[0]["crop"].copy()  # type: ignore[index]


def _join_crops(crops: list[np.ndarray]) -> np.ndarray:
    assert crops, "expected at least one outlier crop"
    out_h = max(int(crop.shape[0]) for crop in crops)
    out_w = sum(int(crop.shape[1]) for crop in crops) + len(crops) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)
    c = 0
    for crop in crops:
        h, w = map(int, crop.shape)
        out[:h, c : c + w] = crop
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each panel's unique odd-shaped object and concatenate those crops in panel order.

    Input structure:
    - The grid has a Black background and complete Yellow rows and columns that
      divide it into a regular array of rectangular panels.
    - Each panel contains three to five monochrome 4-connected foreground objects.
    - Within each panel, all but one object share the same translated binary shape.

    Rule:
    - Identify the full Yellow separator rows and columns, then read panels from
      left to right and top to bottom.
    - In each panel, normalize every foreground object's binary mask by translating
      its bounding box to the origin, and select the single shape that occurs once.
    - Crop that odd object to its compact bounding box.
    - Place the panel-order crops horizontally with one Black separator column,
      using a Black canvas tall enough for the tallest crop.

    Key insight:
    - The Yellow lattice defines independent local "find the different shape"
      contexts; only after solving every panel does the row-major panel order
      explain how the extracted objects become the output strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_spans, col_spans = _panel_spans(g)

    crops: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            crops.append(_outlier_crop(g[r0:r1, c0:c1]))
    return _join_crops(crops)


# --------------- generator ---------------


def _normalize(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in cells))


def _rotate(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize({(c, -r) for r, c in shape})


def _reflect(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize({(r, -c) for r, c in shape})


def _variants(base: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[tuple[tuple[int, int], ...]] = []
    for flipped in (False, True):
        shape = _reflect(base) if flipped else _normalize(base)
        for _ in range(4):
            if shape not in seen:
                seen.add(shape)
                shapes.append(shape)
            shape = _rotate(shape)
    return shapes


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


BASE_SHAPES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
)

ALL_SHAPES = sorted(
    {shape for base in BASE_SHAPES for shape in _variants(base)},
    key=lambda s: (len(s), _shape_dims(s), s),
)
COMPACT_SHAPES = [shape for shape in ALL_SHAPES if _shape_dims(shape)[0] <= 2 and _shape_dims(shape)[1] <= 2]


def _choose_shapes(panel_rows: int, panel_cols: int) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    if panel_rows == 3 and panel_cols == 3:
        pool = COMPACT_SHAPES
        common = random.choice(pool)
        candidates = [s for s in pool if s != common and abs(len(s) - len(common)) <= 1]
    else:
        pool = ALL_SHAPES
        common = random.choice(pool)
        candidates = [s for s in pool if s != common and abs(len(s) - len(common)) <= 2]

    assert candidates, "shape pool must contain a compatible odd shape"
    return common, random.choice(candidates)


def _candidate_positions(panel_h: int, panel_w: int, shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    h, w = _shape_dims(shape)
    positions = [(r, c) for r in range(panel_h - h + 1) for c in range(panel_w - w + 1)]
    random.shuffle(positions)
    return positions


def _can_place(blocked: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> bool:
    for rr, cc in shape:
        if bool(blocked[r0 + rr, c0 + cc]):
            return False
    return True


def _mark_blocked(blocked: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> None:
    h, w = map(int, blocked.shape)
    for rr, cc in shape:
        r, c = r0 + rr, c0 + cc
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                blocked[nr, nc] = True


def _object_colors(n: int, outlier_index: int) -> list[int]:
    palette = random.sample(FOREGROUND_COLORS, random.randint(2, min(4, len(FOREGROUND_COLORS))))
    colors = [random.choice(palette) for _ in range(n)]
    if random.choice([True, False]) and n > 1:
        common_indices = [i for i in range(n) if i != outlier_index]
        colors[outlier_index] = colors[random.choice(common_indices)]
    return colors


def _make_panel(panel_h: int, panel_w: int, panel_rows: int, panel_cols: int) -> np.ndarray | None:
    n_objects = random.randint(3, 5)
    common_shape, odd_shape = _choose_shapes(panel_rows, panel_cols)
    odd_index = random.randrange(n_objects)
    items = [common_shape for _ in range(n_objects)]
    items[odd_index] = odd_shape
    colors = _object_colors(n_objects, odd_index)

    order = list(range(n_objects))
    random.shuffle(order)
    order.sort(key=lambda i: (_shape_dims(items[i])[0] * _shape_dims(items[i])[1]), reverse=True)

    panel = np.full((panel_h, panel_w), BLACK, dtype=int)
    blocked = np.zeros((panel_h, panel_w), dtype=bool)

    for i in order:
        shape = items[i]
        placed = False
        for r0, c0 in _candidate_positions(panel_h, panel_w, shape):
            if not _can_place(blocked, shape, r0, c0):
                continue
            for rr, cc in shape:
                panel[r0 + rr, c0 + cc] = int(colors[i])
            _mark_blocked(blocked, shape, r0, c0)
            placed = True
            break
        if not placed:
            return None

    return panel


def _choose_layout() -> tuple[int, int, int, int]:
    panel_rows, panel_cols = random.choice(((2, 2), (3, 3)))
    if panel_rows == 3:
        return panel_rows, panel_cols, random.randint(8, 9), random.randint(8, 9)
    return panel_rows, panel_cols, random.randint(10, 14), random.randint(10, 14)


def _assemble(panels: list[list[np.ndarray]], panel_h: int, panel_w: int) -> np.ndarray:
    panel_rows = len(panels)
    panel_cols = len(panels[0])
    h = panel_rows * panel_h + panel_rows - 1
    w = panel_cols * panel_w + panel_cols - 1
    grid = np.full((h, w), BLACK, dtype=int)

    for pr, row in enumerate(panels):
        for pc, panel in enumerate(row):
            r0 = pr * (panel_h + 1)
            c0 = pc * (panel_w + 1)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    for pr in range(1, panel_rows):
        grid[pr * panel_h + pr - 1, :] = YELLOW
    for pc in range(1, panel_cols):
        grid[:, pc * panel_w + pc - 1] = YELLOW
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols, panel_h, panel_w = _choose_layout()
    panels: list[list[np.ndarray]] = []
    for _pr in range(panel_rows):
        row: list[np.ndarray] = []
        for _pc in range(panel_cols):
            panel = _make_panel(panel_h, panel_w, panel_rows, panel_cols)
            if panel is None:
                return None
            row.append(panel)
        panels.append(row)

    grid = _assemble(panels, panel_h, panel_w)
    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
