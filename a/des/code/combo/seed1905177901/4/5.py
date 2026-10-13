from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

PRIMARY_COLORS = (RED, BLUE, YELLOW)
VISIBLE_FOR_PRIMARY = {
    RED: frozenset((RED, MAGENTA, ORANGE)),
    BLUE: frozenset((BLUE, MAGENTA, GREEN)),
    YELLOW: frozenset((YELLOW, ORANGE, GREEN)),
}
MIXED_BY_PAIR = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}
VALID_COLORS = frozenset((BG, BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE))


GENERATION_ATTEMPTS = 400
POSITION_ATTEMPTS = 5000
DIMENSION_ATTEMPTS = 200

GRID_MIN = 14
GRID_MAX = 24
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 8
MIN_OVERLAP_AREA = 2
MIN_BG_CELLS = 20

PATTERN_SPECS = (
    (2, "disjoint"),
    (2, "overlap"),
    (3, "disjoint"),
    (3, "one_pair"),
    (3, "chain"),
)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _rect_height(rect: tuple[int, int, int, int]) -> int:
    return int(rect[2] - rect[0] + 1)


def _rect_width(rect: tuple[int, int, int, int]) -> int:
    return int(rect[3] - rect[1] + 1)


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    return _rect_height(rect) * _rect_width(rect)


def _pair_overlap_spans(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int]:
    row_overlap = max(0, min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0])) + 1)
    col_overlap = max(0, min(int(a[3]), int(b[3])) - max(int(a[1]), int(b[1])) + 1)
    return row_overlap, col_overlap


def _pair_overlap_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    row_overlap, col_overlap = _pair_overlap_spans(a, b)
    return int(row_overlap * col_overlap)


def _pair_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int]:
    row_gap = max(int(b[0]) - int(a[2]) - 1, int(a[0]) - int(b[2]) - 1, 0)
    col_gap = max(int(b[1]) - int(a[3]) - 1, int(a[1]) - int(b[3]) - 1, 0)
    return row_gap, col_gap


def _triple_overlap_area(rects: list[tuple[int, int, int, int]]) -> int:
    if len(rects) < 3:
        return 0
    r0 = max(int(rect[0]) for rect in rects)
    c0 = max(int(rect[1]) for rect in rects)
    r1 = min(int(rect[2]) for rect in rects)
    c1 = min(int(rect[3]) for rect in rects)
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _render_input_from_layers(
    layers: list[dict[str, int | tuple[int, int, int, int]]],
    shape: tuple[int, int],
) -> np.ndarray:
    red_mask = np.zeros(shape, dtype=bool)
    blue_mask = np.zeros(shape, dtype=bool)
    yellow_mask = np.zeros(shape, dtype=bool)

    for layer in layers:
        color = int(layer["color"])
        r0, c0, r1, c1 = layer["bbox"]
        if color == RED:
            red_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        elif color == BLUE:
            blue_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        elif color == YELLOW:
            yellow_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        else:
            raise AssertionError(f"unexpected primary color {color}")

    assert not np.any(red_mask & blue_mask & yellow_mask), "generator forbids triple overlaps"

    out = np.full(shape, BG, dtype=int)
    out[red_mask & ~blue_mask & ~yellow_mask] = RED
    out[~red_mask & blue_mask & ~yellow_mask] = BLUE
    out[~red_mask & ~blue_mask & yellow_mask] = YELLOW
    out[red_mask & blue_mask & ~yellow_mask] = MAGENTA
    out[red_mask & ~blue_mask & yellow_mask] = ORANGE
    out[~red_mask & blue_mask & yellow_mask] = GREEN
    return out


def _pack_layers(
    layers: list[dict[str, int | tuple[int, int, int, int]]],
) -> np.ndarray:
    ordered = sorted(
        layers,
        key=lambda layer: (
            -_rect_area(layer["bbox"]),
            int(layer["bbox"][0]),
            int(layer["bbox"][1]),
        ),
    )
    out_h = max(_rect_height(layer["bbox"]) for layer in ordered)
    out_w = sum(_rect_width(layer["bbox"]) for layer in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for layer in ordered:
        color = int(layer["color"])
        rect = layer["bbox"]
        h = _rect_height(rect)
        w = _rect_width(rect)
        out[out_h - h : out_h, c : c + w] = color
        c += w + 1

    return out


def _infer_layers(
    grid: np.ndarray,
) -> list[dict[str, int | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = {int(v) for v in np.unique(g)}
    assert colors <= VALID_COLORS, f"unexpected colors: {sorted(colors - VALID_COLORS)}"

    layers: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in PRIMARY_COLORS:
        mask = np.isin(g, tuple(VISIBLE_FOR_PRIMARY[color]))
        if not bool(mask.any()):
            continue
        bbox = _bbox_from_mask(mask)
        assert bbox is not None
        r0, c0, r1, c1 = bbox
        assert bool(mask[r0 : r1 + 1, c0 : c1 + 1].all()), "each primary must form one solid rectangle"
        layers.append({"color": int(color), "bbox": bbox})

    assert layers, "expected at least one primary rectangle"
    assert np.array_equal(_render_input_from_layers(layers, g.shape), g), "input must be explainable by the inferred layers"
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose the mixed translucent rectangles into their original primary-color layers.

    Input structure:
    - Black background.
    - Each original layer is one axis-aligned solid rectangle in red, blue, or yellow.
    - Overlaps are additive and only create magenta, orange, or green.

    Rule:
    - Treat each mixed color as evidence that both corresponding primaries are present.
    - Reconstruct every primary rectangle as the filled bbox of all cells that contain that primary.
    - Sort the recovered rectangles by area descending and draw them bottom-aligned
      on a new canvas with one black column between neighbors.
    """
    layers = _infer_layers(np.asarray(grid, dtype=int))
    return _pack_layers(layers)


def _sample_dimensions(n_rects: int) -> list[tuple[int, int]] | None:
    dims: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for _ in range(n_rects):
        for _attempt in range(DIMENSION_ATTEMPTS):
            h = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
            w = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
            area = h * w
            if area in used_areas:
                continue
            used_areas.add(area)
            dims.append((h, w))
            break
        else:
            return None
    return dims


def _sample_canvas_shape(
    dims: list[tuple[int, int]],
    pattern: str,
) -> tuple[int, int]:
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)
    min_h = max(GRID_MIN, max_h + 4)
    min_w = max(GRID_MIN, max_w + 4)

    if len(dims) == 3 and pattern == "disjoint":
        min_h = max(min_h, 18)
        min_w = max(min_w, 18)
    elif len(dims) == 3:
        min_h = max(min_h, 16)
        min_w = max(min_w, 16)

    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)
    return h, w


def _relation_spec(n_rects: int, pattern: str) -> dict[tuple[int, int], bool]:
    relations = {(i, j): False for i in range(n_rects) for j in range(i + 1, n_rects)}
    if pattern == "disjoint":
        return relations
    if pattern == "overlap":
        assert n_rects == 2
        relations[(0, 1)] = True
        return relations
    if pattern == "one_pair":
        pair = random.choice(list(relations))
        relations[pair] = True
        return relations
    if pattern == "chain":
        assert n_rects == 3
        center = random.randrange(3)
        for idx in range(3):
            if idx == center:
                continue
            pair = tuple(sorted((center, idx)))
            relations[pair] = True
        return relations
    raise AssertionError(f"unknown pattern {pattern}")


def _positions_match_relations(
    rects: list[tuple[int, int, int, int]],
    relations: dict[tuple[int, int], bool],
) -> bool:
    if _triple_overlap_area(rects) > 0:
        return False

    for pair, should_overlap in relations.items():
        a = rects[pair[0]]
        b = rects[pair[1]]
        overlap_area = _pair_overlap_area(a, b)
        if should_overlap:
            if overlap_area < MIN_OVERLAP_AREA:
                return False
        else:
            if overlap_area != 0:
                return False
            row_gap, col_gap = _pair_gap(a, b)
            if max(row_gap, col_gap) < 1:
                return False
    return True


def _sample_positions(
    dims: list[tuple[int, int]],
    shape: tuple[int, int],
    relations: dict[tuple[int, int], bool],
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    for _ in range(POSITION_ATTEMPTS):
        rects = []
        for rect_h, rect_w in dims:
            r0 = random.randint(0, h - rect_h)
            c0 = random.randint(0, w - rect_w)
            rects.append((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
        if _positions_match_relations(rects, relations):
            return rects
    return None


def _build_ground_truth(
    layers: list[dict[str, int | tuple[int, int, int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    row_max = max(int(layer["bbox"][2]) for layer in layers) + 1
    col_max = max(int(layer["bbox"][3]) for layer in layers) + 1
    shape = (row_max, col_max)
    inp = _render_input_from_layers(layers, shape)
    out = _pack_layers(layers)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_rects, pattern = random.choice(PATTERN_SPECS)
    dims = _sample_dimensions(n_rects)
    if dims is None:
        return None

    canvas_shape = _sample_canvas_shape(dims, pattern)
    relations = _relation_spec(n_rects, pattern)
    rects = _sample_positions(dims, canvas_shape, relations)
    if rects is None:
        return None

    colors = random.sample(list(PRIMARY_COLORS), n_rects)
    layers = [{"color": int(color), "bbox": rect} for color, rect in zip(colors, rects)]

    inp, expected = _build_ground_truth(layers)
    if int((inp == BG).sum()) < MIN_BG_CELLS:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
