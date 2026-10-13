from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]
Point = tuple[int, int]
Direction = tuple[int, int]


_DIRS4: tuple[Direction, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PATTERN_ANCHORS: dict[Direction, Point] = {
    (-1, 0): (2, 1),  # bottom-middle cell of the 3x3 motif
    (1, 0): (0, 1),   # top-middle
    (0, -1): (1, 2),  # right-middle
    (0, 1): (1, 0),   # left-middle
}


_PATTERN_POOL: tuple[np.ndarray, ...] = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Point]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Point]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[Point] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in _DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[Point]) -> BBox:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _legend_box(grid: np.ndarray, bg: int) -> tuple[int, BBox]:
    candidates: list[tuple[int, int, BBox]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 5 or width < 5:
                continue
            if len(comp) != 2 * (height + width) - 4:
                continue

            local = np.zeros((height, width), dtype=bool)
            for r, c in comp:
                local[r - r0, c - c0] = True
            if not np.array_equal(local, _border_mask(height, width)):
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or not np.any(interior != bg):
                continue
            candidates.append((len(comp), color, (r0, c0, r1, c1)))

    assert candidates, "expected a complete legend frame"
    max_area = max(area for area, _color, _bbox_value in candidates)
    best = [item for item in candidates if item[0] == max_area]
    assert len(best) == 1, "legend frame must be unique"
    _area, color, bbox_value = best[0]
    return color, bbox_value


def _legend_mapping(grid: np.ndarray, bg: int, legend_bbox: BBox, legend_color: int) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = legend_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mapping: dict[int, np.ndarray] = {}

    for color in np.unique(inner):
        color = int(color)
        if color in {bg, legend_color}:
            continue
        comps = _components(inner == color)
        assert len(comps) == 2, "each legend color should form a key pixel and one 3x3 motif"
        comps.sort(key=len)
        assert len(comps[0]) == 1, "legend key must be a singleton pixel"
        pr0, pc0, pr1, pc1 = _bbox(comps[1])
        assert pr1 - pr0 + 1 == 3 and pc1 - pc0 + 1 == 3, "legend motif must occupy a 3x3 bbox"
        motif = inner[pr0 : pr1 + 1, pc0 : pc1 + 1] == color
        mapping[color] = motif

    assert mapping, "expected at least one legend mapping"
    return mapping


def _gap_from_ring_bbox(bbox_value: BBox, direction: Direction) -> Point:
    r0, c0, r1, c1 = bbox_value
    if direction == (-1, 0):
        return r0, c0 + (c1 - c0) // 2
    if direction == (1, 0):
        return r1, c0 + (c1 - c0) // 2
    if direction == (0, -1):
        return r0 + (r1 - r0) // 2, c0
    if direction == (0, 1):
        return r0 + (r1 - r0) // 2, c1
    raise ValueError(direction)


def _pattern_origin(gap: Point, direction: Direction) -> Point:
    dr, dc = direction
    anchor_r, anchor_c = _PATTERN_ANCHORS[direction]
    landing_r = gap[0] + 2 * dr
    landing_c = gap[1] + 2 * dc
    return landing_r - anchor_r, landing_c - anchor_c


def _ring_objects(grid: np.ndarray, bg: int, legend_bbox: BBox) -> list[dict[str, object]]:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    rings: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components((grid == color) & (~legend_mask)):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 5 or width < 5 or height % 2 == 0 or width % 2 == 0:
                continue
            if len(comp) != 2 * (height + width) - 5:
                continue

            local = np.zeros((height, width), dtype=bool)
            for r, c in comp:
                local[r - r0, c - c0] = True
            border = _border_mask(height, width)
            if np.any(local & (~border)):
                continue
            missing = np.argwhere(border & (~local))
            if missing.shape[0] != 1:
                continue

            mr, mc = map(int, missing[0])
            direction: Direction | None = None
            if mr == 0 and mc == width // 2:
                direction = (-1, 0)
            elif mr == height - 1 and mc == width // 2:
                direction = (1, 0)
            elif mc == 0 and mr == height // 2:
                direction = (0, -1)
            elif mc == width - 1 and mr == height // 2:
                direction = (0, 1)
            if direction is None:
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_pixels = np.argwhere((interior != bg) & (interior != color))
            assert inner_pixels.shape[0] == 1, "each ring must contain exactly one inner control pixel"
            ir, ic = map(int, inner_pixels[0])
            dot = (r0 + 1 + ir, c0 + 1 + ic)

            rings.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "direction": direction,
                    "gap": (r0 + mr, c0 + mc),
                    "dot": dot,
                    "dot_color": int(grid[dot]),
                }
            )

    assert rings, "expected at least one gapped rectangular ring"
    rings.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each ring's inner color dot through the ring gap and render the legend-linked 3x3 symbol outside the ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    legend_color, legend_bbox = _legend_box(g, bg)
    mapping = _legend_mapping(g, bg, legend_bbox, legend_color)
    rings = _ring_objects(g, bg, legend_bbox)

    out = g.copy()
    h, w = out.shape
    for ring in rings:
        dot_r, dot_c = ring["dot"]  # type: ignore[misc]
        dot_color = int(ring["dot_color"])
        assert dot_color in mapping, "inner dot color must exist in the legend"
        out[dot_r, dot_c] = bg

        pr0, pc0 = _pattern_origin(ring["gap"], ring["direction"])  # type: ignore[arg-type]
        assert 0 <= pr0 and pr0 + 3 <= h and 0 <= pc0 and pc0 + 3 <= w, "ejected motif must fit the grid"
        patch = out[pr0 : pr0 + 3, pc0 : pc0 + 3]
        patch[mapping[dot_color]] = dot_color

    return out


# --------------- generator ---------------

_GEN_BG = 0
_GEN_LEGEND_COLOR = 5
_GEN_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_GRID_H_RANGE = (28, 36)
_GEN_GRID_W_RANGE = (28, 36)
_GEN_LEGEND_COLOR_COUNT = (3, 4)
_GEN_RING_SIZE_CHOICES = (5, 7, 9)
_GEN_PADDING = 1
_GEN_ATTEMPTS = 300


def _legend_patch(colors: list[int], motifs: list[np.ndarray]) -> np.ndarray:
    count = len(colors)
    height = 4 * count + 1
    width = 7
    patch = np.full((height, width), _GEN_BG, dtype=int)
    patch[0, :] = _GEN_LEGEND_COLOR
    patch[-1, :] = _GEN_LEGEND_COLOR
    patch[:, 0] = _GEN_LEGEND_COLOR
    patch[:, -1] = _GEN_LEGEND_COLOR

    for idx, (color, motif) in enumerate(zip(colors, motifs, strict=True)):
        top = 1 + 4 * idx
        patch[top + 1, 1] = int(color)
        patch[top : top + 3, 3:6][motif] = int(color)
    return patch


def _corner_origin(corner: str, grid_shape: tuple[int, int], patch_shape: tuple[int, int]) -> Point:
    h, w = grid_shape
    ph, pw = patch_shape
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _in_bounds(bbox_value: BBox, grid_shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox_value
    h, w = grid_shape
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _expand_bbox(bbox_value: BBox, pad: int, grid_shape: tuple[int, int]) -> BBox:
    r0, c0, r1, c1 = bbox_value
    h, w = grid_shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _bbox_free(blocked: np.ndarray, bbox_value: BBox, pad: int = 0) -> bool:
    er0, ec0, er1, ec1 = _expand_bbox(bbox_value, pad, blocked.shape)
    return not bool(blocked[er0 : er1 + 1, ec0 : ec1 + 1].any())


def _mark_blocked(blocked: np.ndarray, bbox_value: BBox, pad: int = 0) -> None:
    er0, ec0, er1, ec1 = _expand_bbox(bbox_value, pad, blocked.shape)
    blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _pattern_bbox_for_ring(bbox_value: BBox, direction: Direction) -> BBox:
    gap = _gap_from_ring_bbox(bbox_value, direction)
    pr0, pc0 = _pattern_origin(gap, direction)
    return pr0, pc0, pr0 + 2, pc0 + 2


def _draw_ring(grid: np.ndarray, bbox_value: BBox, color: int, direction: Direction) -> Point:
    r0, c0, r1, c1 = bbox_value
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    gap = _gap_from_ring_bbox(bbox_value, direction)
    grid[gap] = _GEN_BG
    return gap


def _candidate_ring_positions(blocked: np.ndarray, ring_h: int, ring_w: int, direction: Direction) -> list[BBox]:
    h, w = blocked.shape
    candidates: list[BBox] = []
    for r0 in range(h - ring_h + 1):
        for c0 in range(w - ring_w + 1):
            bbox_value = (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1)
            motif_bbox = _pattern_bbox_for_ring(bbox_value, direction)
            if not _in_bounds(motif_bbox, blocked.shape):
                continue
            if not _bbox_free(blocked, bbox_value, pad=_GEN_PADDING):
                continue
            if not _bbox_free(blocked, motif_bbox, pad=_GEN_PADDING):
                continue
            candidates.append(bbox_value)
    return candidates


def _sample_directions(count: int) -> list[Direction]:
    base = list(_DIRS4)
    random.shuffle(base)
    directions = base[: min(count, len(base))]
    while len(directions) < count:
        directions.append(random.choice(base))
    random.shuffle(directions)
    return directions


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*_GEN_GRID_H_RANGE)
    grid_w = random.randint(*_GEN_GRID_W_RANGE)
    color_count = random.randint(*_GEN_LEGEND_COLOR_COUNT)

    legend_colors = random.sample(list(_GEN_COLOR_POOL), color_count)
    motif_choices = random.sample(list(_PATTERN_POOL), color_count)
    legend = _legend_patch(legend_colors, motif_choices)

    remaining_ring_colors = [c for c in _GEN_COLOR_POOL if c not in legend_colors]
    min_rings = color_count
    max_rings = len(remaining_ring_colors)
    ring_count = random.randint(min_rings, max_rings)
    ring_colors = random.sample(remaining_ring_colors, ring_count)

    dot_colors = legend_colors[:]
    while len(dot_colors) < ring_count:
        dot_colors.append(random.choice(legend_colors))
    random.shuffle(dot_colors)
    directions = _sample_directions(ring_count)

    inp = np.full((grid_h, grid_w), _GEN_BG, dtype=int)
    out = inp.copy()
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    corner = random.choice(("TL", "TR", "BL", "BR"))
    lr0, lc0 = _corner_origin(corner, inp.shape, legend.shape)
    lr1 = lr0 + legend.shape[0] - 1
    lc1 = lc0 + legend.shape[1] - 1
    inp[lr0 : lr1 + 1, lc0 : lc1 + 1] = legend
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = legend
    _mark_blocked(blocked, (lr0, lc0, lr1, lc1), pad=_GEN_PADDING)

    motif_by_color = {color: motif for color, motif in zip(legend_colors, motif_choices, strict=True)}

    for ring_color, dot_color, direction in zip(ring_colors, dot_colors, directions, strict=True):
        size_options = [(h, w) for h in _GEN_RING_SIZE_CHOICES for w in _GEN_RING_SIZE_CHOICES]
        random.shuffle(size_options)
        placed = False
        for ring_h, ring_w in size_options:
            candidates = _candidate_ring_positions(blocked, ring_h, ring_w, direction)
            if not candidates:
                continue
            bbox_value = random.choice(candidates)
            gap = _draw_ring(inp, bbox_value, ring_color, direction)
            _draw_ring(out, bbox_value, ring_color, direction)

            r0, c0, r1, c1 = bbox_value
            dot_r = random.randint(r0 + 1, r1 - 1)
            dot_c = random.randint(c0 + 1, c1 - 1)
            inp[dot_r, dot_c] = dot_color

            pr0, pc0 = _pattern_origin(gap, direction)
            out[pr0 : pr0 + 3, pc0 : pc0 + 3][motif_by_color[dot_color]] = dot_color

            _mark_blocked(blocked, bbox_value, pad=_GEN_PADDING)
            _mark_blocked(blocked, (pr0, pc0, pr0 + 2, pc0 + 2), pad=_GEN_PADDING)
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(inp) != _GEN_BG:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid sample for task 6")
