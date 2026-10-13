from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg  # noqa: E402


BG = Color.BLACK
MARKER = Color.BLUE
LINE = Color.TEAL
OBJECT_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_RANGE = (18, 24)
LINE_COUNT = 2
LINE_GAP_MIN = 6
OBJECT_COUNT_CHOICES = (5, 6)
MARKER_MARGIN = 1
OBJECT_MARGIN = 2
OBJECT_PAD = 1
OBJECT_DIMS = [(3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4), (4, 5), (5, 4)]
MAX_GENERATION_ATTEMPTS = 400
MAX_SHAPE_ATTEMPTS = 250


def _marker_runs(values: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    rows, cols = np.where(values == MARKER)
    for r, c in zip(rows, cols):
        out.setdefault(int(r), []).append(int(c))
    return out


def _draw_lines(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    bg = int(get_bg(grid))
    out = np.asarray(grid, dtype=int).copy()
    hmask = np.zeros_like(out, dtype=bool)
    vmask = np.zeros_like(out, dtype=bool)

    row_markers = _marker_runs(out)
    active_rows: list[int] = []
    for r, cols in row_markers.items():
        cols = sorted(set(cols))
        if len(cols) < 2:
            continue
        active_rows.append(int(r))
        for c0, c1 in zip(cols, cols[1:]):
            for c in range(c0 + 1, c1):
                if int(grid[r, c]) == bg:
                    out[r, c] = LINE
                    hmask[r, c] = True

    col_markers: dict[int, list[int]] = {}
    rows, cols = np.where(grid == MARKER)
    for r, c in zip(rows, cols):
        col_markers.setdefault(int(c), []).append(int(r))
    active_cols: list[int] = []
    for c, rows_in_col in col_markers.items():
        rows_in_col = sorted(set(rows_in_col))
        if len(rows_in_col) < 2:
            continue
        active_cols.append(int(c))
        for r0, r1 in zip(rows_in_col, rows_in_col[1:]):
            for r in range(r0 + 1, r1):
                if int(grid[r, c]) == bg:
                    out[r, c] = LINE
                    vmask[r, c] = True

    return out, hmask, vmask, sorted(active_rows), sorted(active_cols)


def _object_components(grid: np.ndarray, bg: int) -> list[dict]:
    colors = [
        int(color)
        for color in np.unique(grid)
        if int(color) not in {int(bg), MARKER, LINE}
    ]
    return find_components(
        grid,
        bg=bg,
        colors=colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )


def _erase_component(
    out: np.ndarray,
    component: dict,
    bg: int,
    horizontal: bool,
    vertical: bool,
) -> None:
    coords = component["coords"]
    r0, c0, _r1, _c1 = map(int, component["bbox"])
    remove = np.zeros(len(coords), dtype=bool)

    if horizontal:
        cut = 2 if np.any(coords[:, 1] >= c0 + 2) else 1
        remove |= coords[:, 1] < c0 + cut
    if vertical:
        cut = 2 if np.any(coords[:, 0] >= r0 + 2) else 1
        remove |= coords[:, 0] < r0 + cut

    erased = coords[remove]
    if erased.size:
        out[erased[:, 0], erased[:, 1]] = int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """Draw marker-defined cyan connectors, then erase each crossed object from the side encoded by connector direction.

    Input structure:
    - The grid has a dominant black background, several small monochrome 4-connected foreground objects, and blue marker pixels.
    - Some blue markers share rows or columns; foreground object colors never use blue or cyan.
    - Foreground objects are interpreted independently in their own local bounding boxes.

    Rule:
    - For every row with at least two blue markers, connect each adjacent marker pair with a cyan horizontal segment on the background cells between them; do the analogous operation vertically for columns.
    - For each foreground object, inspect whether cyan pixels appear inside its bounding box because of a horizontal connector, a vertical connector, or both.
    - Horizontal crossing erases that object's foreground pixels in its own leftmost two bounding-box columns; vertical crossing erases its foreground pixels in its own topmost two bounding-box rows; if a two-column or two-row object would vanish completely, erase only one local column or row instead.
    - Keep all blue markers and all cyan connector pixels in the output.

    Key insight:
    - The cyan connectors are not decorative output: their orientation inside each object's local bbox is the control signal that decides which side of that object gets trimmed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    out, hmask, vmask, _active_rows, _active_cols = _draw_lines(g)
    components = _object_components(g, bg)

    for component in components:
        r0, c0, r1, c1 = map(int, component["bbox"])
        horizontal = bool(hmask[r0 : r1 + 1, c0 : c1 + 1].any())
        vertical = bool(vmask[r0 : r1 + 1, c0 : c1 + 1].any())
        if horizontal or vertical:
            _erase_component(out, component, bg, horizontal, vertical)

    return out


def _random_line_positions(size: int) -> list[int]:
    choices = list(range(4, size - 4))
    for _ in range(200):
        sample = sorted(random.sample(choices, LINE_COUNT))
        if min(b - a for a, b in zip(sample, sample[1:])) >= LINE_GAP_MIN:
            return sample
    raise RuntimeError("failed to sample separated connector lines")


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    out = [(int(r), int(c))]
    while (r, c) != (tr, tc):
        choices = []
        if r < tr:
            choices.append((r + 1, c))
        elif r > tr:
            choices.append((r - 1, c))
        if c < tc:
            choices.append((r, c + 1))
        elif c > tc:
            choices.append((r, c - 1))
        r, c = random.choice(choices)
        out.append((int(r), int(c)))
    return out


def _shape_survives(mask: np.ndarray, need_h: bool, need_v: bool) -> bool:
    remain = mask.copy()
    if need_h:
        remain[:, :2] = False
    if need_v:
        remain[:2, :] = False
    return bool(remain.any())


def _generate_shape(
    h: int,
    w: int,
    cross_row: int | None,
    cross_col: int | None,
    need_h: bool,
    need_v: bool,
) -> np.ndarray | None:
    min_area = 4
    max_area = min(10, h * w)
    for _ in range(MAX_SHAPE_ATTEMPTS):
        hub_r = cross_row if cross_row is not None and random.random() < 0.7 else random.randrange(h)
        hub_c = cross_col if cross_col is not None and random.random() < 0.7 else random.randrange(w)
        hub = (int(hub_r), int(hub_c))

        terminals = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        cells = {hub}
        for terminal in terminals:
            cells.update(_path_between(hub, terminal))

        if not (min_area <= len(cells) <= max_area):
            continue

        target = random.randint(len(cells), max_area)
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if cross_row is not None and bool(mask[cross_row].all()):
            continue
        if cross_col is not None and bool(mask[:, cross_col].all()):
            continue
        if need_h and not np.any(mask[:, 2:]):
            continue
        if need_v and not np.any(mask[2:, :]):
            continue
        if not _shape_survives(mask, need_h, need_v):
            continue
        return mask
    return None


def _bbox_ok(
    top: int,
    left: int,
    h: int,
    w: int,
    row_idx: int | None,
    col_idx: int | None,
    h_rows: list[int],
    v_cols: list[int],
    existing: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> bool:
    rows_hit = [i for i, r in enumerate(h_rows) if top <= r < top + h]
    cols_hit = [i for i, c in enumerate(v_cols) if left <= c < left + w]
    if row_idx is None:
        if rows_hit:
            return False
    elif rows_hit != [row_idx]:
        return False

    if col_idx is None:
        if cols_hit:
            return False
    elif cols_hit != [col_idx]:
        return False

    r0 = max(0, top - OBJECT_PAD)
    c0 = max(0, left - OBJECT_PAD)
    r1 = min(shape[0] - 1, top + h - 1 + OBJECT_PAD)
    c1 = min(shape[1] - 1, left + w - 1 + OBJECT_PAD)
    for er0, ec0, er1, ec1 in existing:
        if not (r1 < er0 or er1 < r0 or c1 < ec0 or ec1 < c0):
            return False
    return True


def _category_plan(n_objects: int, h_rows: list[int], v_cols: list[int]) -> list[dict]:
    base = [
        {"kind": "h_only", "need_h": True, "need_v": False},
        {"kind": "v_only", "need_h": False, "need_v": True},
        {"kind": "both", "need_h": True, "need_v": True},
        {"kind": "none", "need_h": False, "need_v": False},
    ]
    extras = []
    crossed_kinds = [
        {"kind": "h_only", "need_h": True, "need_v": False},
        {"kind": "v_only", "need_h": False, "need_v": True},
        {"kind": "both", "need_h": True, "need_v": True},
    ]
    while len(base) + len(extras) < n_objects:
        extras.append(dict(random.choice(crossed_kinds)))
    specs = base + extras
    random.shuffle(specs)

    for spec in specs:
        spec["row_idx"] = random.randrange(len(h_rows)) if spec["need_h"] else None
        spec["col_idx"] = random.randrange(len(v_cols)) if spec["need_v"] else None
    specs.sort(key=lambda item: (item["need_h"] + item["need_v"]), reverse=True)
    return specs


def _render_markers(grid: np.ndarray, h_rows: list[int], v_cols: list[int]) -> None:
    horizontal_pairs = [
        (MARKER_MARGIN, grid.shape[1] - 1 - MARKER_MARGIN),
        (MARKER_MARGIN + 1, grid.shape[1] - 2 - MARKER_MARGIN),
    ]
    vertical_pairs = [
        (MARKER_MARGIN, grid.shape[0] - 1 - MARKER_MARGIN),
        (MARKER_MARGIN + 1, grid.shape[0] - 2 - MARKER_MARGIN),
    ]

    for r, (c0, c1) in zip(h_rows, horizontal_pairs):
        grid[r, c0] = MARKER
        grid[r, c1] = MARKER
    for c, (r0, r1) in zip(v_cols, vertical_pairs):
        grid[r0, c] = MARKER
        grid[r1, c] = MARKER


def _place_objects(grid: np.ndarray, h_rows: list[int], v_cols: list[int], colors: list[int]) -> bool:
    specs = _category_plan(len(colors), h_rows, v_cols)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for color, spec in zip(colors, specs):
        dim_choices = random.sample(OBJECT_DIMS, len(OBJECT_DIMS))
        placed = False
        for h, w in dim_choices:
            positions = []
            for top in range(OBJECT_MARGIN, grid.shape[0] - h - OBJECT_MARGIN):
                for left in range(OBJECT_MARGIN, grid.shape[1] - w - OBJECT_MARGIN):
                    if _bbox_ok(top, left, h, w, spec["row_idx"], spec["col_idx"], h_rows, v_cols, placed_boxes, grid.shape):
                        positions.append((top, left))
            random.shuffle(positions)
            for top, left in positions:
                cross_row = h_rows[spec["row_idx"]] - top if spec["row_idx"] is not None else None
                cross_col = v_cols[spec["col_idx"]] - left if spec["col_idx"] is not None else None
                mask = _generate_shape(h, w, cross_row, cross_col, spec["need_h"], spec["need_v"])
                if mask is None:
                    continue

                coords = np.argwhere(mask)
                abs_rows = coords[:, 0] + top
                abs_cols = coords[:, 1] + left
                if np.any(grid[abs_rows, abs_cols] != BG):
                    continue

                grid[abs_rows, abs_cols] = int(color)
                pad_box = (
                    max(0, top - OBJECT_PAD),
                    max(0, left - OBJECT_PAD),
                    min(grid.shape[0] - 1, top + h - 1 + OBJECT_PAD),
                    min(grid.shape[1] - 1, left + w - 1 + OBJECT_PAD),
                )
                placed_boxes.append(pad_box)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _semantic_summary(grid: np.ndarray) -> dict | None:
    bg = int(get_bg(grid))
    marked = int(np.sum(grid == MARKER))
    if marked < 6 or marked > 10:
        return None

    line_grid, hmask, vmask, active_rows, active_cols = _draw_lines(grid)
    if len(active_rows) != LINE_COUNT or len(active_cols) != LINE_COUNT:
        return None

    components = _object_components(grid, bg)
    if not (5 <= len(components) <= 6):
        return None

    categories = {"h_only": 0, "v_only": 0, "both": 0, "none": 0}
    touched = 0
    unchanged = 0

    for component in components:
        r0, c0, r1, c1 = map(int, component["bbox"])
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            return None
        if not (4 <= int(component["area"]) <= 10):
            return None

        rows_with_h = np.any(hmask[r0 : r1 + 1, c0 : c1 + 1], axis=1).sum()
        cols_with_v = np.any(vmask[r0 : r1 + 1, c0 : c1 + 1], axis=0).sum()
        if rows_with_h > 1 or cols_with_v > 1:
            return None

        horizontal = bool(hmask[r0 : r1 + 1, c0 : c1 + 1].any())
        vertical = bool(vmask[r0 : r1 + 1, c0 : c1 + 1].any())
        if horizontal and vertical:
            categories["both"] += 1
        elif horizontal:
            categories["h_only"] += 1
        elif vertical:
            categories["v_only"] += 1
        else:
            categories["none"] += 1

        if horizontal or vertical:
            touched += 1
        else:
            unchanged += 1

    if min(categories.values()) < 1:
        return None
    if touched < 3 or unchanged < 1:
        return None

    out = main(grid)
    if np.array_equal(out, line_grid):
        return None

    return {
        "components": len(components),
        "markers": marked,
        "categories": categories,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(*GRID_RANGE)
        w = random.randint(*GRID_RANGE)
        grid = np.full((h, w), BG, dtype=int)

        h_rows = _random_line_positions(h)
        v_cols = _random_line_positions(w)
        _render_markers(grid, h_rows, v_cols)

        n_objects = random.choice(OBJECT_COUNT_CHOICES)
        colors = random.sample(OBJECT_COLORS, n_objects)
        if not _place_objects(grid, h_rows, v_cols, colors):
            continue

        if _semantic_summary(grid) is not None:
            return grid

    raise RuntimeError("failed to generate a valid gated-erasure ARC input")
