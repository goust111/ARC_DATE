from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


RING_COLORS = (Color.BLUE, Color.GREEN)
SPRITE_COLOR = Color.GREY
RECOLOR_MAP = {
    0: Color.PINK,
    1: Color.RED,
    2: Color.YELLOW,
    3: Color.ORANGE,
}
MAX_BAND_ID = max(RECOLOR_MAP)
GEN_ATTEMPTS = 400
SPRITE_SIZES = (3, 4, 5, 6)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _is_connected4(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    stack = [next(iter(cell_set))]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cell_set)


def _build_shape_library() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    base = [(r, c) for r in range(4) for c in range(4)]
    by_size = {size: [] for size in SPRITE_SIZES}
    for size in SPRITE_SIZES:
        seen: set[tuple[tuple[int, int], ...]] = set()
        for combo in itertools.combinations(base, size):
            norm = _normalize_shape(combo)
            if norm in seen or not _is_connected4(norm):
                continue
            seen.add(norm)
            by_size[size].append(norm)
    return by_size


_SHAPES_BY_SIZE = _build_shape_library()
_ALL_SHAPES = [shape for size in SPRITE_SIZES for shape in _SHAPES_BY_SIZE[size]]


def _other_ring_color(color: int) -> int:
    return Color.GREEN if int(color) == Color.BLUE else Color.BLUE


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _ring_local_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_rectangular_ring(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return False

    mask = _ring_local_mask(comp)
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return False
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    return not bool(mask[1:-1, 1:-1].any())


def _sorted_rings(components: list[dict]) -> list[dict]:
    rings = [comp for comp in components if int(comp["color"]) in RING_COLORS and _is_rectangular_ring(comp)]
    rings.sort(key=lambda comp: -_bbox_area(comp["bbox"]))
    assert rings, "expected at least one rectangular ring"
    for outer, inner in zip(rings, rings[1:], strict=False):
        or0, oc0, or1, oc1 = map(int, outer["bbox"])
        ir0, ic0, ir1, ic1 = map(int, inner["bbox"])
        assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "rings must be strictly nested"
    return rings


def _ring_bands(rings: list[dict]) -> list[int]:
    bands: list[int] = []
    current_band = 0
    previous_color: int | None = None
    for ring in rings:
        color = int(ring["color"])
        if previous_color is None or color != previous_color:
            current_band += 1
        bands.append(current_band)
        previous_color = color
    assert current_band <= MAX_BAND_ID, "ring-band mapping only covers bands 1..3"
    return bands


def _depth_map(shape: tuple[int, int], ring_bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    depth = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in ring_bboxes:
        depth[r0 + 1 : r1, c0 + 1 : c1] += 1
    return depth


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "background must be black"
    colors = set(map(int, np.unique(g)))
    assert colors <= {Color.BLACK, Color.BLUE, Color.GREEN, Color.GREY}, "unexpected colors in input"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    rings = _sorted_rings(components)
    ring_bboxes = [tuple(map(int, ring["bbox"])) for ring in rings]
    bands = _ring_bands(rings)
    depth = _depth_map(g.shape, ring_bboxes)

    sprite_components = [comp for comp in components if int(comp["color"]) == SPRITE_COLOR]
    assert sprite_components, "expected grey sprites"

    sprite_records = []
    for comp in sprite_components:
        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        depths = {int(depth[r, c]) for r, c in coords}
        assert len(depths) == 1, "each sprite must stay inside one ring region"
        region_depth = depths.pop()
        band = 0 if region_depth == 0 else bands[region_depth - 1]
        assert band in RECOLOR_MAP, "output color mapping is undefined for this band id"
        sprite_records.append(
            {
                "component": comp,
                "region_depth": int(region_depth),
                "band": int(band),
                "recolor": int(RECOLOR_MAP[band]),
            }
        )

    return {
        "bg": bg,
        "rings": rings,
        "ring_bboxes": ring_bboxes,
        "ring_bands": bands,
        "sprites": sprite_records,
        "depth_map": depth,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each gray sprite by the band depth of the concentric ring region containing it.

    Input structure:
    - The background is black, with several 1-pixel-thick nested rectangular rings in blue and green.
    - Rings are ordered from outermost to innermost, and adjacent rings may share the same color.
    - Small gray 4-connected sprites lie entirely inside one region: outside all rings, between two rings, or inside the innermost ring.

    Rule:
    - Sort the rings by nesting depth from outside to inside.
    - Merge consecutive same-colored rings into one ring-band; every region inherits the band id of its outer adjacent ring, while the exterior of the outermost ring has band 0.
    - Keep the rings unchanged and recolor each gray sprite by its region's band id: band 0 -> pink, band 1 -> red, band 2 -> yellow, band 3 -> orange.

    Key insight:
    - The output color depends on ring-band grouping, not raw ring index: regions separated only by same-colored adjacent rings still belong to the same band.
    """
    analysis = _analyze_grid(grid)
    out = np.asarray(grid, dtype=int).copy()
    for sprite in analysis["sprites"]:
        color = int(sprite["recolor"])
        for r, c in sprite["component"]["coords"]:
            out[int(r), int(c)] = color
    return out


def _ring_sequence() -> list[int]:
    start = random.choice(RING_COLORS)
    repeat_index = random.randrange(3)
    seq = [int(start)]
    for idx in range(3):
        prev = seq[-1]
        seq.append(prev if idx == repeat_index else _other_ring_color(prev))
    assert len(seq) == 4
    assert any(seq[i] == seq[i + 1] for i in range(3))
    bands = []
    cur = 0
    prev = None
    for color in seq:
        if prev is None or color != prev:
            cur += 1
        bands.append(cur)
        prev = color
    assert max(bands) == 3
    return seq


def _sample_layout() -> tuple[tuple[int, int], list[tuple[int, int, int, int]], list[int]]:
    for _ in range(200):
        inner_h = random.randint(5, 7)
        inner_w = random.randint(5, 8)
        steps = [random.choice((3, 3, 4)) for _ in range(3)]
        grow = 2 * sum(steps)
        outer_h = inner_h + grow
        outer_w = inner_w + grow
        max_margin_h = (30 - outer_h) // 2
        max_margin_w = (30 - outer_w) // 2
        if max_margin_h < 2 or max_margin_w < 2:
            continue
        margin_h = random.randint(2, min(3, max_margin_h))
        margin_w = random.randint(2, min(3, max_margin_w))
        grid_h = outer_h + 2 * margin_h
        grid_w = outer_w + 2 * margin_w
        if max(grid_h, grid_w) > 30:
            continue

        r0 = margin_h
        c0 = margin_w
        bboxes = []
        cur_h = outer_h
        cur_w = outer_w
        for idx in range(4):
            bboxes.append((r0, c0, r0 + cur_h - 1, c0 + cur_w - 1))
            if idx < 3:
                shrink = steps[idx]
                r0 += shrink
                c0 += shrink
                cur_h -= 2 * shrink
                cur_w -= 2 * shrink

        inner_box = bboxes[-1]
        if inner_box[2] - inner_box[0] - 1 < 3 or inner_box[3] - inner_box[1] - 1 < 3:
            continue
        return (grid_h, grid_w), bboxes, steps
    raise RuntimeError("failed to sample a valid ring layout")


def _candidate_positions(
    region_mask: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    sh = max(r for r, _c in shape) + 1
    sw = max(c for _r, c in shape) + 1
    h, w = region_mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            ok = True
            for dr, dc in shape:
                rr = r0 + dr
                cc = c0 + dc
                if not region_mask[rr, cc] or blocked[rr, cc]:
                    ok = False
                    break
            if ok:
                positions.append((r0, c0))
    return positions


def _place_shape(
    grid: np.ndarray,
    region_mask: np.ndarray,
    blocked: np.ndarray,
) -> bool:
    sizes = list(SPRITE_SIZES)
    random.shuffle(sizes)
    sizes.sort()
    for size in sizes:
        shapes = _SHAPES_BY_SIZE[size][:]
        random.shuffle(shapes)
        for shape in shapes:
            positions = _candidate_positions(region_mask, blocked, shape)
            if not positions:
                continue
            r0, c0 = random.choice(positions)
            for dr, dc in shape:
                rr = r0 + dr
                cc = c0 + dc
                grid[rr, cc] = SPRITE_COLOR
                for nr in range(rr - 1, rr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if in_bounds(grid.shape, nr, nc):
                            blocked[nr, nc] = True
            return True
    return False


def _place_sprites(grid: np.ndarray, ring_bboxes: list[tuple[int, int, int, int]]) -> bool:
    depth = _depth_map(grid.shape, ring_bboxes)
    blocked = np.zeros(grid.shape, dtype=bool)
    region_ids = list(range(len(ring_bboxes) + 1))
    random.shuffle(region_ids)
    for region_depth in region_ids:
        region_mask = (depth == region_depth) & (grid == Color.BLACK)
        count = 1 + int(random.random() < 0.55)
        for _ in range(count):
            if not _place_shape(grid, region_mask, blocked):
                return False
    return True


def _validate_instance(grid: np.ndarray) -> None:
    analysis = _analyze_grid(grid)
    rings = analysis["rings"]
    ring_bands = analysis["ring_bands"]
    sprites = analysis["sprites"]

    assert len(rings) == 4, "generator should build four nested rings"
    ring_colors = [int(ring["color"]) for ring in rings]
    assert any(ring_colors[i] == ring_colors[i + 1] for i in range(3)), "must include a same-color adjacency"
    assert len(set(ring_bands)) == 3, "generator should expose all three positive band ids"
    assert any(ring_bands[i] == ring_bands[i + 1] for i in range(3)), "must create two distinct regions sharing one band"

    depths_seen = {sprite["region_depth"] for sprite in sprites}
    assert depths_seen == set(range(5)), "every region must contain at least one sprite"
    recolors_seen = {sprite["recolor"] for sprite in sprites}
    assert recolors_seen == set(RECOLOR_MAP.values()), "all output recolor classes must appear"

    for sprite in sprites:
        area = int(sprite["component"]["area"])
        assert 3 <= area <= 6, "sprite size must stay within the stated range"

    out = main(grid)
    for ring in rings:
        color = int(ring["color"])
        for r, c in ring["coords"]:
            assert int(out[int(r), int(c)]) == color, "rings must remain unchanged"
    for sprite in sprites:
        expected = int(sprite["recolor"])
        for r, c in sprite["component"]["coords"]:
            assert int(out[int(r), int(c)]) == expected, "sprite recolor mismatch"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        (h, w), ring_bboxes, _steps = _sample_layout()
        grid = np.full((h, w), Color.BLACK, dtype=int)
        colors = _ring_sequence()
        for bbox, color in zip(ring_bboxes, colors, strict=False):
            _draw_ring(grid, bbox, color)
        if not _place_sprites(grid, ring_bboxes):
            continue
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
