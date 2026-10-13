from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, draw_line, find_components, get_bg, in_bounds


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_RNG = np.random.default_rng()

GRID_H_RANGE = (13, 20)
GRID_W_RANGE = (13, 20)
PAIR_COUNT_RANGE = (3, 6)
ROTATION_DISTANCE_RANGE = (2, 5)
GEN_ATTEMPTS = 500


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    """Inclusive cells on an axis-aligned segment."""
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("expected an axis-aligned segment")


def _clear_segment(grid: np.ndarray, a: Coord, b: Coord, bg: int) -> bool:
    for r, c in _segment_cells(a, b)[1:-1]:
        if int(grid[r, c]) != int(bg):
            return False
    return True


def _rotate_ccw_about_pivot(source: Coord, pivot: Coord) -> Coord:
    sr, sc = source
    pr, pc = pivot
    dr, dc = sr - pr, sc - pc
    return pr - dc, pc + dr


def _inverse_source_from_target(target: Coord, pivot: Coord) -> Coord:
    tr, tc = target
    pr, pc = pivot
    dr, dc = tr - pr, tc - pc
    return pr + dc, pc - dr


def _anchor_rows_cols(anchor_bbox: BBox) -> tuple[set[int], set[int]]:
    r0, c0, r1, c1 = anchor_bbox
    return {int(r0), int(r1)}, {int(c0), int(c1)}


def _ray_endpoint(target: Coord, anchor_bbox: BBox) -> Coord | None:
    r, c = target
    r0, c0, r1, c1 = anchor_bbox
    if r == r0 or r == r1:
        if c < c0:
            return r, c0 - 1
        if c > c1:
            return r, c1 + 1
        return None
    if c == c0 or c == c1:
        if r < r0:
            return r0 - 1, c
        if r > r1:
            return r1 + 1, c
        return None
    return None


def _paint_ray(out: np.ndarray, target: Coord, anchor_bbox: BBox) -> None:
    endpoint = _ray_endpoint(target, anchor_bbox)
    if endpoint is None:
        out[target] = Color.ORANGE
        return
    draw_line(out, target[0], target[1], end_r=endpoint[0], end_c=endpoint[1], color=Color.ORANGE)


def _find_anchor_bbox(grid: np.ndarray, bg: int) -> BBox:
    green_components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREEN],
        return_format="dict",
    )
    anchors = [
        comp["bbox"]
        for comp in green_components
        if int(comp["area"]) == 4 and bbox_h(comp["bbox"]) == 2 and bbox_w(comp["bbox"]) == 2
    ]
    assert len(anchors) == 1, "expected exactly one 2x2 green anchor"
    return anchors[0]


def _singleton_color_coords(grid: np.ndarray, bg: int, color: int) -> list[Coord]:
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )
    coords: list[Coord] = []
    for comp in components:
        assert int(comp["area"]) == 1, f"expected color {color} to appear only as singleton pixels"
        coords.append(tuple(map(int, comp["coords"][0])))
    return coords


def _paired_pivot(source: Coord, pivots: list[Coord], grid: np.ndarray, bg: int) -> Coord:
    matches = [
        pivot
        for pivot in pivots
        if (pivot[0] == source[0] or pivot[1] == source[1]) and _clear_segment(grid, source, pivot, bg)
    ]
    assert len(matches) == 1, "each source must pair with exactly one pivot"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each magenta source 90 degrees counterclockwise around its blue pivot, then extend only the anchor-aligned rotated pixels toward the green 2x2 anchor.

    Input structure:
    - The background is black.
    - There is exactly one green 2x2 anchor square.
    - Blue pixels are singleton pivots and magenta pixels are singleton sources; each source shares exactly one row or one column with exactly one pivot.

    Rule:
    - Pair every magenta source with its unique blue pivot on the same row or column.
    - Rotate the source 90 degrees counterclockwise around that pivot and erase the original magenta pixel.
    - Paint the rotated position orange.
    - If the rotated orange pixel lies on one of the anchor's two rows or two columns, extend an orange ray from that pixel toward the anchor until the cell just outside the anchor; otherwise keep it as a single orange pixel.
    - Keep the blue pivots and the green anchor unchanged.

    Key insight:
    - The rotation is not the final goal; it creates new orange positions, and only those new positions that become row- or column-aligned with the anchor emit rays.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    anchor_bbox = _find_anchor_bbox(g, bg)
    pivots = _singleton_color_coords(g, bg, Color.BLUE)
    sources = _singleton_color_coords(g, bg, Color.PINK)

    assert pivots, "expected at least one pivot"
    assert sources, "expected at least one source"

    pivot_set = set(pivots)
    anchor_cells = {
        (r, c)
        for r in range(anchor_bbox[0], anchor_bbox[2] + 1)
        for c in range(anchor_bbox[1], anchor_bbox[3] + 1)
    }

    source_to_pivot: list[tuple[Coord, Coord]] = []
    targets: list[Coord] = []
    for source in sources:
        pivot = _paired_pivot(source, pivots, g, bg)
        target = _rotate_ccw_about_pivot(source, pivot)
        assert in_bounds(g.shape, target[0], target[1]), "rotated target must remain inside the grid"
        source_to_pivot.append((source, pivot))
        targets.append(target)

    assert len(set(targets)) == len(targets), "rotated targets must be distinct"
    assert not any(target in pivot_set for target in targets), "rotated targets cannot land on pivots"
    assert not any(target in anchor_cells for target in targets), "rotated targets cannot land on the anchor"

    out = g.copy()
    for source in sources:
        out[source] = bg

    for target in targets:
        out[target] = Color.ORANGE
    for target in targets:
        _paint_ray(out, target, anchor_bbox)

    return out


def _sample_anchor(shape: tuple[int, int]) -> BBox:
    h, w = shape
    r0 = int(_RNG.integers(3, h - 4))
    c0 = int(_RNG.integers(3, w - 4))
    return r0, c0, r0 + 1, c0 + 1


def _sample_modes(pair_count: int) -> list[str]:
    aligned_count = int(_RNG.integers(1, min(3, pair_count - 1) + 1))
    aligned_modes: list[str] = []
    if aligned_count >= 2 and float(_RNG.random()) < 0.65:
        aligned_modes.extend(["row", "col"])
    while len(aligned_modes) < aligned_count:
        aligned_modes.append(str(_RNG.choice(["row", "col"])))
    modes = aligned_modes + ["none"] * (pair_count - aligned_count)
    _RNG.shuffle(modes)
    return modes


def _target_candidates(shape: tuple[int, int], anchor_bbox: BBox, mode: str) -> list[Coord]:
    h, w = shape
    r0, c0, r1, c1 = anchor_bbox
    candidates: list[Coord] = []
    if mode == "row":
        for r in (r0, r1):
            for c in range(0, c0 - 1):
                candidates.append((r, c))
            for c in range(c1 + 2, w):
                candidates.append((r, c))
    elif mode == "col":
        for c in (c0, c1):
            for r in range(0, r0 - 1):
                candidates.append((r, c))
            for r in range(r1 + 2, h):
                candidates.append((r, c))
    else:
        anchor_rows, anchor_cols = _anchor_rows_cols(anchor_bbox)
        for r in range(h):
            if r in anchor_rows:
                continue
            for c in range(w):
                if c in anchor_cols:
                    continue
                candidates.append((r, c))
    return candidates


def _pivot_source_options(target: Coord, shape: tuple[int, int]) -> list[tuple[Coord, Coord]]:
    h, w = shape
    options: list[tuple[Coord, Coord]] = []
    distances = list(range(ROTATION_DISTANCE_RANGE[0], ROTATION_DISTANCE_RANGE[1] + 1))
    _RNG.shuffle(distances)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _RNG.shuffle(directions)
    for d in distances:
        for ur, uc in directions:
            dr, dc = ur * d, uc * d
            pivot = (target[0] - dr, target[1] - dc)
            source = _inverse_source_from_target(target, pivot)
            if in_bounds((h, w), pivot[0], pivot[1]) and in_bounds((h, w), source[0], source[1]):
                options.append((pivot, source))
    return options


def _orth_adjacent(a: Coord, b: Coord) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _expected_output(shape: tuple[int, int], anchor_bbox: BBox, pairs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = anchor_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = Color.GREEN
    for pair in pairs:
        pr, pc = pair["pivot"]
        out[pr, pc] = Color.BLUE
    for pair in pairs:
        _paint_ray(out, pair["target"], anchor_bbox)
    return out


def _build_input(shape: tuple[int, int], anchor_bbox: BBox, pairs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = anchor_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = Color.GREEN
    for pair in pairs:
        pr, pc = pair["pivot"]
        sr, sc = pair["source"]
        grid[pr, pc] = Color.BLUE
        grid[sr, sc] = Color.PINK
    return grid


def _layout_ok(shape: tuple[int, int], anchor_bbox: BBox, pairs: list[dict]) -> bool:
    if not pairs:
        return False
    aligned = sum(1 for pair in pairs if pair["aligned"])
    if aligned == 0 or aligned == len(pairs):
        return False

    inp = _build_input(shape, anchor_bbox, pairs)
    out = _expected_output(shape, anchor_bbox, pairs)

    bg = int(get_bg(inp))
    try:
        if not np.array_equal(main(inp), out):
            return False
    except AssertionError:
        return False

    blue_components = find_components(inp, bg=bg, connectivity=4, monochromatic=True, colors=[Color.BLUE])
    pink_components = find_components(inp, bg=bg, connectivity=4, monochromatic=True, colors=[Color.PINK])
    if len(blue_components) != len(pairs) or len(pink_components) != len(pairs):
        return False
    if any(int(comp["area"]) != 1 for comp in blue_components + pink_components):
        return False

    pivots = [tuple(map(int, comp["coords"][0])) for comp in blue_components]
    for comp in pink_components:
        source = tuple(map(int, comp["coords"][0]))
        matches = [pivot for pivot in pivots if (pivot[0] == source[0] or pivot[1] == source[1]) and _clear_segment(inp, source, pivot, bg)]
        if len(matches) != 1:
            return False

    return True


def _try_place_pairs(shape: tuple[int, int], anchor_bbox: BBox, modes: list[str]) -> list[dict] | None:
    anchor_cells = {
        (r, c)
        for r in range(anchor_bbox[0], anchor_bbox[2] + 1)
        for c in range(anchor_bbox[1], anchor_bbox[3] + 1)
    }
    input_fg = set(anchor_cells)
    output_fg = set(anchor_cells)
    reserved_clear: set[Coord] = set()
    pivots: list[Coord] = []
    sources: list[Coord] = []
    pairs: list[dict] = []

    for mode in modes:
        candidates = _target_candidates(shape, anchor_bbox, mode)
        if not candidates:
            return None
        order = _RNG.permutation(len(candidates))
        placed = False
        for idx in order:
            target = candidates[int(idx)]
            endpoint = _ray_endpoint(target, anchor_bbox)
            ray_cells = [target] if endpoint is None else _segment_cells(target, endpoint)

            if any(cell in output_fg for cell in ray_cells):
                continue
            if any(cell in input_fg for cell in ray_cells):
                continue

            for pivot, source in _pivot_source_options(target, shape):
                if pivot in anchor_cells or source in anchor_cells:
                    continue
                if pivot in input_fg or source in input_fg:
                    continue
                if pivot in reserved_clear or source in reserved_clear:
                    continue
                if pivot in output_fg:
                    continue

                corridor = _segment_cells(source, pivot)[1:-1]
                if any(cell in input_fg for cell in corridor):
                    continue

                if any(source[0] == other[0] or source[1] == other[1] for other in pivots):
                    continue
                if any(pivot[0] == other[0] or pivot[1] == other[1] for other in sources):
                    continue

                if any(_orth_adjacent(pivot, other) for other in pivots):
                    continue
                if any(_orth_adjacent(source, other) for other in sources):
                    continue

                pair = {
                    "pivot": pivot,
                    "source": source,
                    "target": target,
                    "aligned": endpoint is not None,
                }
                pairs.append(pair)
                pivots.append(pivot)
                sources.append(source)
                input_fg.add(pivot)
                input_fg.add(source)
                output_fg.add(pivot)
                output_fg.update(ray_cells)
                reserved_clear.update(ray_cells)
                reserved_clear.update(corridor)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return pairs


def _gen_attempt() -> np.ndarray | None:
    shape = (
        int(_RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1)),
        int(_RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1)),
    )
    anchor_bbox = _sample_anchor(shape)
    pair_count = int(_RNG.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    modes = _sample_modes(pair_count)
    pairs = _try_place_pairs(shape, anchor_bbox, modes)
    if pairs is None:
        return None
    if not _layout_ok(shape, anchor_bbox, pairs):
        return None
    return _build_input(shape, anchor_bbox, pairs)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the pivot-rotation ray task")
