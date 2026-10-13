from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
MARKER = 9
BASE_COLORS = (1, 2, 3, 4, 6, 7, 8)
RECT_COUNT_OPTIONS = (3, 4)
RECT_SIDE_OPTIONS = (4, 5)
GRID_HEIGHT_OPTIONS = tuple(range(16, 21))
GRID_WIDTH_OPTIONS = tuple(range(15, 21))
MAX_GENERATION_ATTEMPTS = 500
_RECENT_OUTPUT_SIGS: set[bytes] = set()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    comps = _components(grid[:-1] != BG)
    rects: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        if np.any(patch == BG):
            raise AssertionError("foreground object must be a filled rectangle")
        colors = set(map(int, np.unique(patch)))
        base_colors = colors - {MARKER}
        if len(base_colors) != 1:
            raise AssertionError("each rectangle must have exactly one base color")
        base = next(iter(base_colors))
        marker_coords = np.argwhere(patch == MARKER)
        if len(marker_coords) < 1:
            raise AssertionError("each rectangle needs at least one marker")
        if np.any((marker_coords[:, 0] == 0) | (marker_coords[:, 0] == patch.shape[0] - 1)):
            raise AssertionError("markers must stay off the border")
        if np.any((marker_coords[:, 1] == 0) | (marker_coords[:, 1] == patch.shape[1] - 1)):
            raise AssertionError("markers must stay off the border")
        rects.append(
            {
                "base": int(base),
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "markers": {(int(r), int(c)) for r, c in marker_coords},
            }
        )
    if len(rects) < 2:
        raise AssertionError("expected multiple rectangles")
    return rects


def _composite_from_rects(rects: list[dict], control_order: list[int]) -> np.ndarray:
    by_color = {int(rect["base"]): rect for rect in rects}
    if set(control_order) != set(by_color):
        raise AssertionError("control colors must match rectangle base colors")
    shape_set = {tuple(rect["patch"].shape) for rect in rects}
    if len(shape_set) != 1:
        raise AssertionError("all rectangles must share one shape")
    h, w = next(iter(shape_set))
    out = np.full((h, w), int(control_order[-1]), dtype=int)
    for color in control_order:
        patch = by_color[int(color)]["patch"]
        out[patch == MARKER] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bottom control sequence as layer Z-order, then merge the scattered rectangles into one centered composite.

    Input structure:
    - A black background contains 3 or 4 separated filled rectangles of the same size in the upper area.
    - Each rectangle has a unique base color and one or two maroon marker pixels inside it.
    - The bottom row is a control row whose non-black pixels list those base colors from left to right.

    Rule:
    - Read the control row left to right; the first color is the bottom layer and the last color is the top layer.
    - Match each control color to the rectangle with that base color and treat each rectangle as one layer.
    - Build one rectangle of the shared size: ordinary base cells are overwritten by later layers, while any marker pixel keeps color 9 regardless of layer order.
    - Erase the original rectangles and the control row, then place the composite rectangle at the grid center.

    Key insight:
    - The bottom control row does not describe a path here; it encodes stacking priority, and marker pixels pierce through every base layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 4

    control_order = [int(v) for v in g[-1] if int(v) != BG]
    assert len(control_order) >= 2
    rects = _extract_rectangles(g)
    composite = _composite_from_rects(rects, control_order)

    out = np.zeros_like(g)
    h, w = composite.shape
    r0 = (g.shape[0] - h) // 2
    c0 = (g.shape[1] - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = composite
    return out


def _boxes_clear(boxes: list[tuple[int, int, int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in boxes:
        if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
            return False
    return True


def _place_rectangles(h: int, w: int, side: int, count: int) -> list[tuple[int, int]] | None:
    max_r = h - side - 3
    max_c = w - side - 1
    if max_r < 1 or max_c < 1:
        return None

    candidates = [(r, c) for r in range(1, max_r + 1) for c in range(1, max_c + 1)]
    random.shuffle(candidates)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_topleft: list[tuple[int, int]] = []
    for r, c in candidates:
        box = (r, c, r + side - 1, c + side - 1)
        if not _boxes_clear(placed_boxes, box):
            continue
        placed_boxes.append(box)
        placed_topleft.append((r, c))
        if len(placed_topleft) == count:
            break

    if len(placed_topleft) != count:
        return None

    rows = {r for r, _ in placed_topleft}
    cols = {c for _, c in placed_topleft}
    if len(rows) == 1 or len(cols) == 1:
        return None
    return placed_topleft


def _marker_layouts(side: int, count: int) -> list[set[tuple[int, int]]] | None:
    interior = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
    total_cells = len(interior)
    if total_cells < count:
        return None

    marker_counts = [1] * count
    extra_capacity = min(count, total_cells - count)
    if extra_capacity > 0 and random.random() < 0.7:
        extra = random.randint(1, extra_capacity)
        for idx in random.sample(range(count), extra):
            marker_counts[idx] += 1

    cells = random.sample(interior, sum(marker_counts))
    layouts: list[set[tuple[int, int]]] = []
    at = 0
    for need in marker_counts:
        layout = {cells[at + i] for i in range(need)}
        at += need
        layouts.append(layout)
    if len(set().union(*layouts)) != sum(marker_counts):
        return None
    return layouts


def _row_major_order(rect_meta: list[dict]) -> list[int]:
    return [int(item["base"]) for item in sorted(rect_meta, key=lambda item: (item["bbox"][0], item["bbox"][1]))]


def _column_major_order(rect_meta: list[dict]) -> list[int]:
    return [int(item["base"]) for item in sorted(rect_meta, key=lambda item: (item["bbox"][1], item["bbox"][0]))]


def _signature(grid: np.ndarray) -> bytes:
    return grid.shape[0].to_bytes(1, "little") + grid.shape[1].to_bytes(1, "little") + grid.tobytes()


def _analyze_generated(inp: np.ndarray, rect_meta: list[dict], control_order: list[int], expected: np.ndarray) -> bool:
    rects = _extract_rectangles(inp)
    if len(rects) != len(rect_meta):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    union_markers = set().union(*(rect["markers"] for rect in rects))
    if len(union_markers) < len(rects):
        return False

    shape = rects[0]["patch"].shape
    if len(union_markers) >= shape[0] * shape[1]:
        return False

    if len(control_order) < 2:
        return False
    alt_order = list(control_order)
    alt_order[0], alt_order[-1] = alt_order[-1], alt_order[0]
    if np.array_equal(_composite_from_rects(rects, control_order), _composite_from_rects(rects, alt_order)):
        return False

    if set(control_order) != {int(rect["base"]) for rect in rects}:
        return False
    if control_order == _row_major_order(rect_meta):
        return False
    if control_order == _column_major_order(rect_meta):
        return False
    if np.count_nonzero(expected[-1]) != 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_HEIGHT_OPTIONS)
    w = random.choice(GRID_WIDTH_OPTIONS)
    count = random.choice(RECT_COUNT_OPTIONS)
    side = random.choice(RECT_SIDE_OPTIONS)
    if count == 4 and side == 5 and (h < 18 or w < 17):
        return None

    positions = _place_rectangles(h, w, side, count)
    if positions is None:
        return None

    marker_layouts = _marker_layouts(side, count)
    if marker_layouts is None:
        return None

    bases = random.sample(BASE_COLORS, count)
    grid = np.zeros((h, w), dtype=int)
    rect_meta: list[dict] = []
    for (r0, c0), base, markers in zip(positions, bases, marker_layouts):
        patch = np.full((side, side), int(base), dtype=int)
        for mr, mc in markers:
            patch[mr, mc] = MARKER
        grid[r0 : r0 + side, c0 : c0 + side] = patch
        rect_meta.append(
            {
                "base": int(base),
                "bbox": (r0, c0, r0 + side - 1, c0 + side - 1),
                "markers": set(markers),
            }
        )

    spatial_orders = {tuple(_row_major_order(rect_meta)), tuple(_column_major_order(rect_meta))}
    perms = [list(p) for p in itertools.permutations(bases) if tuple(p) not in spatial_orders]
    if not perms:
        return None
    control_order = random.choice(perms)

    control_cols = sorted(random.sample(range(1, w - 1), count))
    for col, color in zip(control_cols, control_order):
        grid[-1, col] = int(color)

    expected = np.zeros_like(grid)
    composite = np.full((side, side), int(control_order[-1]), dtype=int)
    for markers in marker_layouts:
        for mr, mc in markers:
            composite[mr, mc] = MARKER
    out_r = (h - side) // 2
    out_c = (w - side) // 2
    expected[out_r : out_r + side, out_c : out_c + side] = composite

    if not _analyze_generated(grid, rect_meta, control_order, expected):
        return None

    out_sig = _signature(expected)
    if out_sig in _RECENT_OUTPUT_SIGS:
        return None
    _RECENT_OUTPUT_SIGS.add(out_sig)
    if len(_RECENT_OUTPUT_SIGS) > 512:
        _RECENT_OUTPUT_SIGS.pop()
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
