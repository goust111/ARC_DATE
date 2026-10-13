from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
BAND_COLORS = (Color.RED, Color.BLUE, Color.GREEN)
COLOR_TO_INDEX = {color: idx for idx, color in enumerate(BAND_COLORS)}
COLOR_TO_BAND = {Color.RED: 1, Color.BLUE: 2, Color.GREEN: 3}
GRID_H_RANGE = (6, 10)
GRID_W_RANGE = (8, 12)
GEN_RETRIES = 400

L_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
DISTRACTOR_SHAPES = (
    ((0, 0), (0, 1), (0, 2), (1, 1)),  # top T
    ((0, 1), (1, 0), (1, 1), (1, 2)),  # bottom T
    ((0, 0), (1, 0), (2, 0), (1, 1)),  # left T
    ((0, 1), (1, 0), (1, 1), (2, 1)),  # right T
    ((0, 0), (0, 1), (1, 1), (1, 2)),  # zig-zag
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # hook
    ((0, 0), (0, 1), (0, 2)),          # line
    ((0, 0), (1, 0), (2, 0)),          # line
)


def _collapse_bands(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 3 == 0
    band_h = g.shape[0] // 3
    bands = [g[i * band_h : (i + 1) * band_h, :] for i in range(3)]
    out = np.full((band_h, g.shape[1]), BG, dtype=int)
    for band in (bands[2], bands[1], bands[0]):
        mask = band != BG
        out[mask] = band[mask]
    return out, bands


def _component_missing_corner(comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    occupied = {tuple(map(int, rc)) for rc in comp["coords"]}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in occupied:
                return r, c
    raise AssertionError("expected one missing corner")


def _least_contributing_color(colors: list[int]) -> int:
    counts = Counter(int(color) for color in colors if int(color) != BG)
    assert counts
    min_count = min(counts.values())
    candidates = [color for color, count in counts.items() if count == min_count]
    return max(candidates, key=lambda color: COLOR_TO_BAND[int(color)])


def _l_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False)
    return [
        comp
        for comp in comps
        if int(comp["area"]) == 3
        and int(comp["bbox"][2] - comp["bbox"][0] + 1) == 2
        and int(comp["bbox"][3] - comp["bbox"][1] + 1) == 2
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse three stacked color-mask bands, then complete emergent L-shapes using the least frequent visible contributing layer color.

    Input structure:
    - The grid height is three equal horizontal bands stacked vertically on a black background.
    - The top, middle, and bottom bands are red, blue, and green monochrome masks in a shared local coordinate system.
    - Different bands may overlap at the same local positions, but only the highest-priority non-black band is visible after collapse.

    Rule:
    - Collapse the three bands to one layer with fixed priority: red band over blue band over green band.
    - In the collapsed layer, treat all non-black cells as one foreground and find every 4-connected component.
    - If a component has area 3 and exactly fills three cells of a 2x2 bounding box, it is an L-shape with one missing corner.
    - Fill that missing corner with the color whose visible contribution count inside the L is smallest; if multiple contributing colors tie, choose the lower-priority contributor among them.

    Key insight:
    - The incomplete 2x2 blocks are not stored directly in any single band; they emerge only after band collapse, and their completion color is recovered from the collapsed component's visible layer provenance.
    """
    collapsed, _bands = _collapse_bands(grid)
    out = collapsed.copy()

    for comp in _l_components(collapsed):
        missing_r, missing_c = _component_missing_corner(comp)
        colors = [int(collapsed[r, c]) for r, c in comp["coords"]]
        out[missing_r, missing_c] = _least_contributing_color(colors)

    return out


def _local_bbox(local_cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in local_cells) + 1, max(c for _, c in local_cells) + 1


def _separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_component(
    owner: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    local_cells: tuple[tuple[int, int], ...],
    colors: list[int],
) -> dict | None:
    h, w = _local_bbox(local_cells)
    H, W = owner.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not _separated(rects, r0, c0, h, w):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    coords: list[tuple[int, int]] = []
    for (dr, dc), color in zip(local_cells, colors):
        rr = r0 + dr
        cc = c0 + dc
        owner[rr, cc] = int(color)
        coords.append((rr, cc))
    rects.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return {"bbox": (r0, c0, r0 + h - 1, c0 + w - 1), "coords": coords}


def _make_l_spec(kind: str) -> tuple[tuple[tuple[int, int], ...], list[int], int]:
    missing_index = random.randrange(4)
    cells = tuple(corner for idx, corner in enumerate(L_CORNERS) if idx != missing_index)
    if kind == "two_color":
        fill_color = random.choice((Color.RED, Color.BLUE))
        major_color = random.choice([color for color in BAND_COLORS if color != fill_color])
        colors = [major_color, major_color, fill_color]
        random.shuffle(colors)
    else:
        colors = random.sample(list(BAND_COLORS), 3)
        fill_color = Color.GREEN
    return cells, colors, fill_color


def _paint_distractor(owner: np.ndarray, rects: list[tuple[int, int, int, int]]) -> bool:
    local_cells = random.choice(DISTRACTOR_SHAPES)
    colors = [random.choice(BAND_COLORS) for _ in local_cells]
    placed = _place_component(owner, rects, local_cells, colors)
    return placed is not None


def _build_bands(owner: np.ndarray, protected_bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray] | None:
    H, W = owner.shape
    bands = [np.full((H, W), BG, dtype=int) for _ in range(3)]
    for r in range(H):
        for c in range(W):
            color = int(owner[r, c])
            if color == BG:
                continue
            bands[COLOR_TO_INDEX[color]][r, c] = color

    protected = set()
    for r0, c0, r1, c1 in protected_bboxes:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                protected.add((r, c))

    overlap_candidates = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if int(owner[r, c]) in (Color.RED, Color.BLUE) and (r, c) not in protected
    ]
    if not overlap_candidates:
        return None

    overlap_count = random.randint(1, min(4, len(overlap_candidates)))
    for r, c in random.sample(overlap_candidates, overlap_count):
        top_index = COLOR_TO_INDEX[int(owner[r, c])]
        lower_indices = list(range(top_index + 1, 3))
        extra_count = random.randint(1, len(lower_indices))
        for band_index in random.sample(lower_indices, extra_count):
            bands[band_index][r, c] = BAND_COLORS[band_index]
    return bands


def _has_full_2x2(grid: np.ndarray) -> bool:
    H, W = grid.shape
    for r in range(H - 1):
        for c in range(W - 1):
            if np.all(grid[r : r + 2, c : c + 2] != BG):
                return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        collapsed, bands = _collapse_bands(grid)
    except AssertionError:
        return False

    for band, color in zip(bands, BAND_COLORS):
        if not np.all((band == BG) | (band == int(color))):
            return False

    stacked_mask = np.stack([band != BG for band in bands], axis=0)
    if int(np.count_nonzero(stacked_mask.sum(axis=0) >= 2)) < 1:
        return False

    comps = find_components(collapsed, bg=BG, connectivity=4, monochromatic=False)
    l_comps = []
    non_l_seen = False
    two_color_seen = False
    three_color_seen = False
    expected = collapsed.copy()

    for comp in comps:
        area = int(comp["area"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if area == 3 and r1 - r0 == 1 and c1 - c0 == 1:
            colors = [int(collapsed[r, c]) for r, c in comp["coords"]]
            if len(set(colors)) < 2:
                return False
            unique_colors = len(set(colors))
            two_color_seen |= unique_colors == 2
            three_color_seen |= unique_colors == 3
            for band in bands:
                if int(np.count_nonzero(band[r0 : r1 + 1, c0 : c1 + 1])) >= 3:
                    return False
            missing_r, missing_c = _component_missing_corner(comp)
            expected[missing_r, missing_c] = _least_contributing_color(colors)
            l_comps.append(comp)
        else:
            non_l_seen = True

    if len(l_comps) < 2 or not non_l_seen:
        return False
    if not two_color_seen or not three_color_seen:
        return False
    if _has_full_2x2(collapsed):
        return False

    out = main(grid)
    if np.array_equal(out, collapsed):
        return False
    if not np.array_equal(out, expected):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    owner = np.full((H, W), BG, dtype=int)
    rects: list[tuple[int, int, int, int]] = []
    target_bboxes: list[tuple[int, int, int, int]] = []

    specs = ["two_color", "three_color"]
    if random.random() < 0.55:
        specs.append(random.choice(("two_color", "three_color")))
    random.shuffle(specs)

    for kind in specs:
        local_cells, colors, _fill_color = _make_l_spec(kind)
        placed = _place_component(owner, rects, local_cells, colors)
        if placed is None:
            return None
        target_bboxes.append(placed["bbox"])

    if not _paint_distractor(owner, rects):
        return None
    extra_distractors = random.randint(0, 2)
    for _ in range(extra_distractors):
        _paint_distractor(owner, rects)

    bands = _build_bands(owner, target_bboxes)
    if bands is None:
        return None

    grid = np.vstack(bands)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
