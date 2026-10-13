from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
MARKER = 4
LAYER_COLOR_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
COUNT_PATTERNS = ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
GEN_RETRIES = 400

Rect = tuple[int, int, int, int]


def _bbox_of_color(grid: np.ndarray, color: int) -> Rect:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"missing color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains_rect(outer: Rect, inner: Rect) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _contains_cell(rect: Rect, r: int, c: int) -> bool:
    return rect[0] <= r <= rect[2] and rect[1] <= c <= rect[3]


def _is_on_edge(rect: Rect, r: int, c: int) -> bool:
    return _contains_cell(rect, r, c) and (r == rect[0] or r == rect[2] or c == rect[1] or c == rect[3])


def _corners(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _boundary_cells(rect: Rect) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _side_cells(rect: Rect, side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _infer_layers(grid: np.ndarray) -> list[dict[str, int | Rect]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BACKGROUND, MARKER))
    assert len(colors) == 3, "expected exactly three rectangle-layer colors"

    layers = []
    for color in colors:
        rect = _bbox_of_color(grid, color)
        layers.append({"color": color, "rect": rect, "area": _rect_area(rect)})

    layers.sort(key=lambda item: int(item["area"]), reverse=True)
    areas = [int(layer["area"]) for layer in layers]
    assert len(set(areas)) == 3, "expected three different layer areas"
    assert _contains_rect(layers[0]["rect"], layers[1]["rect"])
    assert _contains_rect(layers[1]["rect"], layers[2]["rect"])
    assert layers[0]["rect"] != layers[1]["rect"] != layers[2]["rect"]
    return layers


def _layer_color_at(layers: list[dict[str, int | Rect]], r: int, c: int) -> int:
    for layer in reversed(layers):
        rect = layer["rect"]
        assert isinstance(rect, tuple)
        if _contains_cell(rect, r, c):
            return int(layer["color"])
    raise AssertionError("marker must lie inside at least one completed layer")


def _winner_color(grid: np.ndarray) -> int:
    layers = _infer_layers(grid)
    marker_positions = np.argwhere(grid == MARKER)
    assert 5 <= len(marker_positions) <= 9, "expected 5-9 marker pixels"

    votes: Counter[int] = Counter({int(layer["color"]): 0 for layer in layers})
    for r0, c0 in marker_positions:
        r, c = int(r0), int(c0)
        assert not any(_is_on_edge(layer["rect"], r, c) for layer in layers)
        votes[_layer_color_at(layers, r, c)] += 1

    ranked = votes.most_common()
    assert ranked[0][1] > ranked[1][1], "marker winner must be unique"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the completed nested rectangle layer that receives the most yellow markers.

    Input structure:
    - The grid has black background, exactly three distinct sparse rectangle-layer colors,
      and yellow marker pixels.
    - Each non-yellow foreground color is a partial sampling of one axis-aligned rectangle
      boundary; the three rectangles are strictly nested with distinct areas.
    - Yellow markers are never on rectangle edges and must be interpreted against the
      rectangles after they are completed and overpainted from outermost to innermost.

    Rule:
    - Take the bbox of each non-yellow foreground color as that layer's full rectangle.
    - Order the rectangles by area from large to small and treat the final owner of a cell
      as the smallest completed rectangle that contains it.
    - Count yellow markers by that completed-layer ownership, choose the unique largest count,
      and output only the winning layer as a filled rectangle on black background.

    Key insight:
    - Marker ownership comes from the nested completion/overwrite result, not from proximity to
      visible sparse samples; a marker near one layer's samples can still belong to another layer
      after the rectangles are completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layers = _infer_layers(g)
    winner = _winner_color(g)

    out = np.full_like(g, BACKGROUND)
    for layer in layers:
        color = int(layer["color"])
        if color != winner:
            continue
        r0, c0, r1, c1 = layer["rect"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        return out
    raise AssertionError("winner color not found among inferred layers")


def _assignment_grid(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect
        owner[r0 : r1 + 1, c0 : c1 + 1] = idx
    return owner


def _edge_mask(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rects:
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def _candidate_cells(shape: tuple[int, int], rects: list[Rect]) -> list[list[tuple[int, int]]]:
    owner = _assignment_grid(shape, rects)
    edges = _edge_mask(shape, rects)
    out: list[list[tuple[int, int]]] = []
    for idx in range(len(rects)):
        coords = np.argwhere((owner == idx) & ~edges)
        out.append([(int(r), int(c)) for r, c in coords])
    return out


def _focus_marker(rects: list[Rect]) -> tuple[tuple[int, int], str, int]:
    inner = rects[2]
    side = random.choice(SIDES)
    if side == "top":
        coord = (inner[1] + inner[3]) // 2
        return (inner[0] - 1, coord), side, coord
    if side == "bottom":
        coord = (inner[1] + inner[3]) // 2
        return (inner[2] + 1, coord), side, coord
    if side == "left":
        coord = (inner[0] + inner[2]) // 2
        return (coord, inner[1] - 1), side, coord
    coord = (inner[0] + inner[2]) // 2
    return (coord, inner[3] + 1), side, coord


def _sample_without_touching(
    pool: list[tuple[int, int]], count: int, occupied: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []

    for _ in range(80):
        random.shuffle(pool)
        chosen: list[tuple[int, int]] = []
        blocked = set(occupied)
        for cell in pool:
            if cell in blocked:
                continue
            r, c = cell
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in blocked):
                continue
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in chosen):
                continue
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
        if len(chosen) == count:
            return chosen
    return None


def _orth_value(cell: tuple[int, int], side: str) -> int:
    return cell[1] if side in ("top", "bottom") else cell[0]


def _sample_boundary(
    rect: Rect,
    *,
    forced_side: str | None = None,
    forced_orth: int | None = None,
) -> set[tuple[int, int]]:
    chosen = set(_corners(rect))
    if forced_side is not None and forced_orth is not None:
        side_pts = _side_cells(rect, forced_side)
        for cell in side_pts:
            if _orth_value(cell, forced_side) == forced_orth:
                chosen.add(cell)
                break

    keep_prob = random.uniform(0.18, 0.45)
    for cell in _boundary_cells(rect):
        if cell in chosen:
            continue
        if random.random() < keep_prob:
            chosen.add(cell)
    return chosen


def _render_sparse_input(
    shape: tuple[int, int],
    rects: list[Rect],
    colors: list[int],
    markers: list[tuple[int, int]],
    focus_side: str,
    focus_orth: int,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for idx, (rect, color) in enumerate(zip(rects, colors)):
        forced_side = focus_side if idx == 2 else None
        forced_orth = focus_orth if idx == 2 else None
        for r, c in _sample_boundary(rect, forced_side=forced_side, forced_orth=forced_orth):
            grid[r, c] = color
    for r, c in markers:
        grid[r, c] = MARKER
    return grid


def _nearest_visible_sample_color(
    grid: np.ndarray, r: int, c: int, layer_colors: list[int]
) -> int | None:
    best = None
    winners: set[int] = set()
    for color in layer_colors:
        coords = np.argwhere(grid == color)
        dist = min(abs(r - int(rr)) + abs(c - int(cc)) for rr, cc in coords)
        if best is None or dist < best:
            best = dist
            winners = {int(color)}
        elif dist == best:
            winners.add(int(color))
    if len(winners) == 1:
        return next(iter(winners))
    return None


def _validate_counterexample(grid: np.ndarray) -> bool:
    layers = _infer_layers(grid)
    layer_colors = [int(layer["color"]) for layer in layers]
    for r0, c0 in np.argwhere(grid == MARKER):
        r, c = int(r0), int(c0)
        true_color = _layer_color_at(layers, r, c)
        nearest = _nearest_visible_sample_color(grid, r, c, layer_colors)
        if nearest is not None and nearest != true_color:
            return True
    return False


def _choose_rects() -> tuple[tuple[int, int], list[Rect]]:
    inner_h = random.randint(5, 7)
    inner_w = random.randint(5, 7)
    outer_gap_r = random.randint(2, 3)
    outer_gap_c = random.randint(2, 3)
    inner_gap_r = 3
    inner_gap_c = 3

    middle_h = inner_h + 2 * inner_gap_r
    middle_w = inner_w + 2 * inner_gap_c
    outer_h = middle_h + 2 * outer_gap_r
    outer_w = middle_w + 2 * outer_gap_c

    top_margin = random.randint(0, 2)
    left_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)

    height = outer_h + top_margin + bottom_margin
    width = outer_w + left_margin + right_margin

    outer = (top_margin, left_margin, top_margin + outer_h - 1, left_margin + outer_w - 1)
    middle = (
        outer[0] + outer_gap_r,
        outer[1] + outer_gap_c,
        outer[2] - outer_gap_r,
        outer[3] - outer_gap_c,
    )
    inner = (
        middle[0] + inner_gap_r,
        middle[1] + inner_gap_c,
        middle[2] - inner_gap_r,
        middle[3] - inner_gap_c,
    )
    return (height, width), [outer, middle, inner]


def _choose_counts() -> tuple[int, int, int]:
    counts = list(random.choice(COUNT_PATTERNS))
    random.shuffle(counts)
    return int(counts[0]), int(counts[1]), int(counts[2])


def _build_markers(
    shape: tuple[int, int],
    rects: list[Rect],
    counts: tuple[int, int, int],
    focus_cell: tuple[int, int],
) -> list[tuple[int, int]] | None:
    per_layer = _candidate_cells(shape, rects)
    if focus_cell not in per_layer[1]:
        return None

    markers = [focus_cell]
    occupied = {focus_cell}

    for layer_idx, need in enumerate(counts):
        remaining = need - (1 if layer_idx == 1 else 0)
        chosen = _sample_without_touching(per_layer[layer_idx][:], remaining, occupied)
        if chosen is None:
            return None
        markers.extend(chosen)
        occupied.update(chosen)

    if len(markers) != sum(counts):
        return None
    return markers


def _attempt_generate() -> np.ndarray | None:
    shape, rects = _choose_rects()
    colors = random.sample(list(LAYER_COLOR_POOL), 3)
    counts = _choose_counts()

    focus_cell, focus_side, focus_orth = _focus_marker(rects)
    markers = _build_markers(shape, rects, counts, focus_cell)
    if markers is None or focus_cell not in markers:
        return None

    grid = _render_sparse_input(shape, rects, colors, markers, focus_side, focus_orth)

    expected = np.full(shape, BACKGROUND, dtype=int)
    winner_idx = max(range(3), key=lambda idx: counts[idx])
    wr0, wc0, wr1, wc1 = rects[winner_idx]
    expected[wr0 : wr1 + 1, wc0 : wc1 + 1] = colors[winner_idx]

    try:
        if not np.array_equal(main(grid), expected):
            return None
        if not _validate_counterexample(grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-layer voting input")
