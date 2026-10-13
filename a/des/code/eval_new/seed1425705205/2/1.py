from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _shape_key(coords: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), norm


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    cells = set(coords)
    if len(cells) != 2 * h + 2 * w - 4:
        return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    return True


def _is_solid_rect(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed rectangular wall using the legend color assigned to the enclosed seed's rotation.

    Input structure:
    - The background is the most common color.
    - Several same-colored 1-pixel-thick hollow rectangles are scattered on the grid.
    - Each rectangle encloses one small asymmetric seed sprite in a single shared seed color.
    - One corner contains a four-row legend made of four solid color bars, each paired with one rotated copy of the same seed sprite.

    Rule:
    - Find the hollow rectangles and extract the seed sprite inside each one.
    - Read the legend by pairing each color bar with the adjacent seed copy in the same row band.
    - Use the seed's exact rotated shape as the lookup key from rectangle interiors to legend bar colors.
    - Recolor only the rectangular border pixels; keep the seed sprites, legend, and background unchanged.

    Key insight:
    - The decisive feature is the seed's orientation, not its position or color; the corner legend teaches a rotation-to-color lookup that must be transferred to every closed wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    by_color: dict[int, list[list[tuple[int, int]]]] = {}
    for color in set(map(int, g.ravel())) - {bg}:
        by_color[int(color)] = _components(g == int(color))

    frame_candidates: dict[int, list[list[tuple[int, int]]]] = {}
    for color, comps in by_color.items():
        rects = [comp for comp in comps if _is_rect_frame(comp)]
        if rects:
            frame_candidates[int(color)] = rects
    assert frame_candidates
    frame_color = max(frame_candidates, key=lambda color: len(frame_candidates[color]))
    frames = frame_candidates[frame_color]
    assert len(frames) >= 1

    frame_boxes = [_bbox(comp) for comp in frames]
    interior_boxes = [(r0 + 1, c0 + 1, r1 - 1, c1 - 1) for r0, c0, r1, c1 in frame_boxes]

    region_seed_keys: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
    seed_color = None
    for box in interior_boxes:
        r0, c0, r1, c1 = box
        coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(g[r, c]) not in {bg, frame_color}]
        assert coords
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1
        this_seed_color = next(iter(colors))
        if seed_color is None:
            seed_color = this_seed_color
        assert seed_color == this_seed_color
        comps = _components((g == seed_color)[r0 : r1 + 1, c0 : c1 + 1])
        assert len(comps) == 1
        region_seed_keys.append(_shape_key([(r + r0, c + c0) for r, c in comps[0]]))

    assert seed_color is not None

    legend_seed_comps: list[list[tuple[int, int]]] = []
    for comp in by_color[seed_color]:
        box = _bbox(comp)
        if any(_inside(box, interior) for interior in interior_boxes):
            continue
        legend_seed_comps.append(comp)
    assert len(legend_seed_comps) == 4

    legend_bars: list[tuple[int, list[tuple[int, int]]]] = []
    for color, comps in by_color.items():
        if color in {frame_color, seed_color}:
            continue
        for comp in comps:
            if _is_solid_rect(comp):
                legend_bars.append((int(color), comp))
    assert len(legend_bars) == 4

    def _row_center(comp: list[tuple[int, int]]) -> tuple[float, float]:
        r0, c0, r1, c1 = _bbox(comp)
        return (r0 + r1) / 2.0, (c0 + c1) / 2.0

    legend_seed_comps.sort(key=lambda comp: _row_center(comp))
    legend_bars.sort(key=lambda item: _row_center(item[1]))
    rotation_to_color = {
        _shape_key(seed_comp): int(color)
        for seed_comp, (color, _bar_comp) in zip(legend_seed_comps, legend_bars, strict=True)
    }

    out = g.copy()
    for frame, seed_key in zip(frames, region_seed_keys, strict=True):
        color = rotation_to_color[seed_key]
        for r, c in frame:
            out[r, c] = int(color)
    return out


BASE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
)

GRID_MIN = 20
GRID_MAX = 30
REGION_COUNT_RANGE = (3, 6)
INTERIOR_PAD_RANGE = (1, 4)
BAR_WIDTH_RANGE = (3, 7)
MAX_SAMPLE_TRIES = 400
PLACE_TRIES = 300
GAP = 1


def _normalize_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, max(r for r, _ in norm) + 1, max(c for _, c in norm) + 1


def _rotate90(coords: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rot = tuple((c, h - 1 - r) for r, c in coords)
    return _normalize_coords(rot)


def _rotations(base: tuple[tuple[int, int], ...]) -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    coords, h, w = _normalize_coords(base)
    out = [(coords, h, w)]
    for _ in range(3):
        coords, h, w = _rotate90(coords, h, w)
        out.append((coords, h, w))
    assert len({item[0] for item in out}) == 4
    return out


def _paint_coords(grid: np.ndarray, coords: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for dr, dc in coords:
        grid[r0 + dr, c0 + dc] = int(color)


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_legend(
    grid: np.ndarray,
    blocked: np.ndarray,
    rotations: list[tuple[tuple[tuple[int, int], ...], int, int]],
    anchor: int,
    bar_colors: list[int],
    seed_color: int,
) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    band_h = max(h for _, h, _ in rotations)
    seed_box_w = max(w for _, _, w in rotations)
    bar_w = random.randrange(*BAR_WIDTH_RANGE)
    total_h = 4 * band_h + 3 * GAP
    total_w = bar_w + seed_box_w
    corner = random.choice(("tl", "tr", "bl", "br"))
    h, w = grid.shape

    if corner[0] == "t":
        base_r = 0
    else:
        base_r = h - total_h
    if corner[1] == "l":
        base_c = 0
    else:
        base_c = w - total_w

    for idx in range(4):
        band_r = base_r + idx * (band_h + GAP)
        if corner[1] == "l":
            bar_c = base_c
            seed_side_c = base_c + bar_w
        else:
            bar_c = base_c + seed_box_w
            seed_side_c = bar_c
        color = bar_colors[idx]
        grid[band_r : band_r + band_h, bar_c : bar_c + bar_w] = int(color)

        coords, shape_h, shape_w = rotations[(anchor + idx) % 4]
        seed_r = band_r + (band_h - shape_h) // 2
        if corner[1] == "l":
            seed_c = seed_side_c
        else:
            seed_c = seed_side_c - shape_w
        _paint_coords(grid, coords, seed_r, seed_c, seed_color)

    _reserve(blocked, base_r, base_c, total_h, total_w, pad=1)
    return {idx: int(bar_colors[idx]) for idx in range(4)}, (base_r, base_c, base_r + total_h - 1, base_c + total_w - 1)


def _place_regions(
    grid: np.ndarray,
    blocked: np.ndarray,
    rotations: list[tuple[tuple[tuple[int, int], ...], int, int]],
    anchor: int,
    frame_color: int,
    seed_color: int,
    orientation_to_color: dict[int, int],
) -> np.ndarray | None:
    h, w = grid.shape
    out = grid.copy()
    expected = grid.copy()
    seed_max_h = max(shape_h for _, shape_h, _ in rotations)
    seed_max_w = max(shape_w for _, _, shape_w in rotations)
    num_regions = random.randrange(*REGION_COUNT_RANGE)

    orientation_order = list(range(4))
    random.shuffle(orientation_order)
    region_orientations = orientation_order[: min(num_regions, 4)]
    while len(region_orientations) < num_regions:
        region_orientations.append(random.randrange(4))
    random.shuffle(region_orientations)

    for orientation in region_orientations:
        placed = False
        coords, shape_h, shape_w = rotations[(anchor + orientation) % 4]
        for _ in range(PLACE_TRIES):
            interior_h = seed_max_h + random.randrange(*INTERIOR_PAD_RANGE)
            interior_w = seed_max_w + random.randrange(*INTERIOR_PAD_RANGE)
            frame_h = interior_h + 2
            frame_w = interior_w + 2
            if frame_h >= h or frame_w >= w:
                continue
            r0 = random.randrange(1, h - frame_h)
            c0 = random.randrange(1, w - frame_w)
            if not _can_place(blocked, r0, c0, frame_h, frame_w):
                continue

            out[r0, c0 : c0 + frame_w] = frame_color
            out[r0 + frame_h - 1, c0 : c0 + frame_w] = frame_color
            out[r0 : r0 + frame_h, c0] = frame_color
            out[r0 : r0 + frame_h, c0 + frame_w - 1] = frame_color

            expected[r0, c0 : c0 + frame_w] = orientation_to_color[orientation]
            expected[r0 + frame_h - 1, c0 : c0 + frame_w] = orientation_to_color[orientation]
            expected[r0 : r0 + frame_h, c0] = orientation_to_color[orientation]
            expected[r0 : r0 + frame_h, c0 + frame_w - 1] = orientation_to_color[orientation]

            seed_r = r0 + 1 + random.randrange(0, interior_h - shape_h + 1)
            seed_c = c0 + 1 + random.randrange(0, interior_w - shape_w + 1)
            _paint_coords(out, coords, seed_r, seed_c, seed_color)
            _paint_coords(expected, coords, seed_r, seed_c, seed_color)

            _reserve(blocked, r0, c0, frame_h, frame_w, pad=1)
            placed = True
            break
        if not placed:
            return None
    return expected, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_SAMPLE_TRIES):
        base = random.choice(BASE_SHAPES)
        rotations = _rotations(base)
        seed_max_h = max(h for _, h, _ in rotations)
        seed_max_w = max(w for _, _, w in rotations)
        band_h = seed_max_h
        legend_h = 4 * band_h + 3 * GAP
        min_h = max(GRID_MIN, legend_h + 5)
        min_w = max(GRID_MIN, seed_max_w + 8)
        if min_h > GRID_MAX or min_w > GRID_MAX:
            continue

        h = random.randint(min_h, GRID_MAX)
        w = random.randint(min_w, GRID_MAX)
        bg, frame_color, seed_color, *bar_colors = random.sample(range(10), 7)
        grid = np.full((h, w), bg, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        anchor = random.randrange(4)

        orientation_to_color, _legend_box = _place_legend(
            grid=grid,
            blocked=blocked,
            rotations=rotations,
            anchor=anchor,
            bar_colors=bar_colors,
            seed_color=seed_color,
        )

        placed = _place_regions(
            grid=grid,
            blocked=blocked,
            rotations=rotations,
            anchor=anchor,
            frame_color=frame_color,
            seed_color=seed_color,
            orientation_to_color=orientation_to_color,
        )
        if placed is None:
            continue
        expected, inp = placed

        counts = Counter(map(int, inp.ravel()))
        if counts[bg] <= max(v for k, v in counts.items() if k != bg):
            continue

        try:
            got = main(inp)
        except Exception:
            continue
        if not np.array_equal(got, expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid sample")
