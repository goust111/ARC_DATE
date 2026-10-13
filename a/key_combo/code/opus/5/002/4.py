from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.util
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if _ARC_UTILS_PATH.exists():
    _spec = importlib.util.spec_from_file_location("_arc_utils_shared_4", _ARC_UTILS_PATH)
    _arc_utils = importlib.util.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_arc_utils)
    components_mask = _arc_utils.components_mask
    paste = _arc_utils.paste
else:
    from scipy.ndimage import label

    _STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    _STRUCT8 = np.ones((3, 3), dtype=int)

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        struct = _STRUCT4 if int(connectivity) == 4 else _STRUCT8
        lab, n = label(mask.astype(bool), structure=struct)
        return [np.argwhere(lab == k) for k in range(1, int(n) + 1)]

    def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, **_kwargs) -> np.ndarray:
        h, w = patch.shape
        dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch
        return dst


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
NOISE_POOL = (6, 8, 9)
ASSIGNED_COLORS = (BLUE, RED, GREEN, YELLOW)
QUADRANTS = ("tl", "tr", "bl", "br")
SCAN_ORDER = ("tl", "tr", "bl", "br")
REGION_SIZE_RANGE = (10, 12)
DIAGONAL_LENGTHS = (3, 4, 5, 6, 7)
NOISE_COUNT_RANGE = (2, 4)
BLOCK_SIZE_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
GEN_ATTEMPTS = 400


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected nonempty values"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _separator_cross(grid: np.ndarray) -> tuple[int, int]:
    row_hits = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_hits = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected exactly one full grey row and one full grey column"
    row = int(row_hits[0])
    col = int(col_hits[0])
    assert 0 < row < grid.shape[0] - 1 and 0 < col < grid.shape[1] - 1, "separator must split four nonempty regions"
    return row, col


def _region_slices(grid: np.ndarray) -> dict[str, tuple[slice, slice]]:
    sep_r, sep_c = _separator_cross(grid)
    h, w = grid.shape
    return {
        "tl": (slice(0, sep_r), slice(0, sep_c)),
        "tr": (slice(0, sep_r), slice(sep_c + 1, w)),
        "bl": (slice(sep_r + 1, h), slice(0, sep_c)),
        "br": (slice(sep_r + 1, h), slice(sep_c + 1, w)),
    }


def _diagonal_length(region: np.ndarray) -> int:
    comps = components_mask(region == ORANGE, connectivity=8)
    assert len(comps) == 1, "each region must contain exactly one orange diagonal component"
    coords = comps[0]
    assert coords.shape[0] >= 3, "orange diagonal is too short"
    rows = coords[:, 0]
    cols = coords[:, 1]
    assert len(set(map(int, rows))) == coords.shape[0], "diagonal must use one cell per row"
    assert len(set(map(int, cols))) == coords.shape[0], "diagonal must use one cell per col"

    cells = {(int(r), int(c)) for r, c in coords}
    if len(set(map(int, rows - cols))) == 1:
        r0 = int(rows.min())
        c0 = int(cols.min())
        expected = {(r0 + i, c0 + i) for i in range(coords.shape[0])}
        assert cells == expected, "orange component must be a straight 45-degree diagonal"
    elif len(set(map(int, rows + cols))) == 1:
        r0 = int(rows.min())
        c1 = int(cols.max())
        expected = {(r0 + i, c1 - i) for i in range(coords.shape[0])}
        assert cells == expected, "orange component must be a straight 45-degree diagonal"
    else:
        raise AssertionError("orange component is not diagonal")
    return int(coords.shape[0])


def _dominant_region_color(region: np.ndarray) -> int:
    fg = region[(region != BLACK) & (region != GREY) & (region != ORANGE)]
    assert fg.size > 0, "region must contain a dominant foreground color"
    return _unique_mode(fg)


def _analyze_regions(grid: np.ndarray) -> dict[str, dict[str, int]]:
    info: dict[str, dict[str, int]] = {}
    for name, src in _region_slices(grid).items():
        region = grid[src]
        info[name] = {
            "dominant": _dominant_region_color(region),
            "diag_len": _diagonal_length(region),
            "area": int(region.shape[0] * region.shape[1]),
        }
    lengths = [info[name]["diag_len"] for name in QUADRANTS]
    assert len(set(lengths)) == 4, "all four diagonal lengths must be distinct"
    return info


def _assigned_corner_colors(info: dict[str, dict[str, int]]) -> dict[str, int]:
    ordered = sorted(QUADRANTS, key=lambda name: info[name]["diag_len"])
    return {name: color for name, color in zip(ordered, ASSIGNED_COLORS)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a cross-separated noisy board by ranking quadrants with their orange diagonal lengths and recoloring the corners cyclically.

    Input structure:
    - A black background grid contains exactly one full grey row and one full grey column that split the board into four rectangular quadrants.
    - Each quadrant contains one dominant non-black color with a few scattered noise pixels in other colors.
    - Each quadrant also contains exactly one orange 45-degree diagonal segment, and the four segment lengths are all different.

    Rule:
    - Find the unique full grey row and full grey column to define the four quadrants.
    - In each quadrant, denoise the foreground by identifying its unique dominant color and separately identify the orange diagonal segment.
    - Count the orange diagonal pixels in each quadrant, sort the quadrants from the smallest count to the largest count, and assign them output colors cyclically as blue, red, green, yellow in that sorted order.
    - Return a 3x3 compressed board whose center row and center column are grey and whose four corners receive the assigned color for the corresponding spatial quadrant.

    Key insight:
    - The output colors are not copied from the quadrant majorities; the decisive signal is the orange diagonal length inside each separator-defined quadrant, which determines the priority order before the cyclic recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    info = _analyze_regions(g)
    assigned = _assigned_corner_colors(info)

    out = np.full((3, 3), GREY, dtype=int)
    out[0, 0] = assigned["tl"]
    out[0, 2] = assigned["tr"]
    out[2, 0] = assigned["bl"]
    out[2, 2] = assigned["br"]
    return out


def _assigned_from_lengths(length_map: dict[str, int]) -> dict[str, int]:
    ordered = sorted(QUADRANTS, key=lambda name: length_map[name])
    return {name: color for name, color in zip(ordered, ASSIGNED_COLORS)}


def _render_expected(length_map: dict[str, int]) -> np.ndarray:
    assigned = _assigned_from_lengths(length_map)
    out = np.full((3, 3), GREY, dtype=int)
    out[0, 0] = assigned["tl"]
    out[0, 2] = assigned["tr"]
    out[2, 0] = assigned["bl"]
    out[2, 2] = assigned["br"]
    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, *, margin: int) -> bool:
    rr0 = max(0, int(r0) - int(margin))
    cc0 = max(0, int(c0) - int(margin))
    rr1 = min(occupied.shape[0], int(r0) + int(h) + int(margin))
    cc1 = min(occupied.shape[1], int(c0) + int(w) + int(margin))
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)] = True


def _place_diagonal(region: np.ndarray, occupied: np.ndarray, length: int) -> bool:
    h, w = region.shape
    down_right = random.random() < 0.5
    if down_right:
        starts = [
            (r0, c0)
            for r0 in range(1, h - length)
            for c0 in range(1, w - length)
        ]
        if not starts:
            return False
        r0, c0 = random.choice(starts)
        coords = [(r0 + i, c0 + i) for i in range(length)]
    else:
        starts = [
            (r0, c1)
            for r0 in range(1, h - length)
            for c1 in range(length, w - 1)
        ]
        if not starts:
            return False
        r0, c1 = random.choice(starts)
        coords = [(r0 + i, c1 - i) for i in range(length)]

    for r, c in coords:
        region[int(r), int(c)] = ORANGE
        occupied[int(r), int(c)] = True
    return True


def _place_main_blocks(region: np.ndarray, occupied: np.ndarray, color: int, noise_count: int) -> int | None:
    target = random.randint(max(10, int(noise_count) * 3), max(13, int(noise_count) * 4 + 2))
    area = 0
    block_count = 0
    attempts = 0
    while (area < target or block_count < 2) and attempts < 120:
        attempts += 1
        bh, bw = random.choice(BLOCK_SIZE_CHOICES)
        candidates = [
            (r0, c0)
            for r0 in range(region.shape[0] - bh + 1)
            for c0 in range(region.shape[1] - bw + 1)
            if _can_place(occupied, r0, c0, bh, bw, margin=1)
        ]
        if not candidates:
            continue
        r0, c0 = random.choice(candidates)
        region[r0 : r0 + bh, c0 : c0 + bw] = color
        _mark(occupied, r0, c0, bh, bw)
        area += int(bh * bw)
        block_count += 1
        if block_count >= 4 and area >= target:
            break
    if area < max(target, noise_count * 3) or block_count < 2:
        return None
    return area


def _place_noise(region: np.ndarray, occupied: np.ndarray, dominant: int, count: int) -> bool:
    candidates = [
        (r, c)
        for r in range(region.shape[0])
        for c in range(region.shape[1])
        if region[r, c] == BLACK and _can_place(occupied, r, c, 1, 1, margin=1)
    ]
    if len(candidates) < count:
        return False
    random.shuffle(candidates)
    for r, c in candidates[:count]:
        region[int(r), int(c)] = random.choice([color for color in NOISE_POOL if color != dominant])
        occupied[int(r), int(c)] = True
    return True


def _build_region(height: int, width: int, dominant: int, diag_len: int) -> tuple[np.ndarray, dict[str, int]] | None:
    region = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    if not _place_diagonal(region, occupied, diag_len):
        return None
    noise_count = random.randint(*NOISE_COUNT_RANGE)
    main_count = _place_main_blocks(region, occupied, dominant, noise_count)
    if main_count is None:
        return None
    if not _place_noise(region, occupied, dominant, noise_count):
        return None
    return region, {"main_count": int(main_count), "noise_count": int(noise_count)}


def _length_rank(values: dict[str, int]) -> list[str]:
    return [name for name, _value in sorted(values.items(), key=lambda item: item[1])]


def _valid_semantics(
    grid: np.ndarray,
    expected: np.ndarray,
    dominant_map: dict[str, int],
    length_map: dict[str, int],
) -> bool:
    try:
        info = _analyze_regions(grid)
    except AssertionError:
        return False

    if not np.array_equal(main(grid), expected):
        return False
    if any(info[name]["dominant"] != dominant_map[name] for name in QUADRANTS):
        return False
    if any(info[name]["diag_len"] != length_map[name] for name in QUADRANTS):
        return False

    rank = _length_rank(length_map)
    if rank == list(SCAN_ORDER):
        return False

    assigned = _assigned_from_lengths(length_map)
    if all(int(assigned[name]) == int(dominant_map[name]) for name in QUADRANTS):
        return False

    area_map = {
        "tl": int((_region_slices(grid)["tl"][0].stop - _region_slices(grid)["tl"][0].start) * (_region_slices(grid)["tl"][1].stop - _region_slices(grid)["tl"][1].start)),
        "tr": int((_region_slices(grid)["tr"][0].stop - _region_slices(grid)["tr"][0].start) * (_region_slices(grid)["tr"][1].stop - _region_slices(grid)["tr"][1].start)),
        "bl": int((_region_slices(grid)["bl"][0].stop - _region_slices(grid)["bl"][0].start) * (_region_slices(grid)["bl"][1].stop - _region_slices(grid)["bl"][1].start)),
        "br": int((_region_slices(grid)["br"][0].stop - _region_slices(grid)["br"][0].start) * (_region_slices(grid)["br"][1].stop - _region_slices(grid)["br"][1].start)),
    }
    area_values = list(area_map.values())
    if len(set(area_values)) == 4 and _length_rank(area_map) == rank:
        return False

    match_count = sum(int(assigned[name]) == int(dominant_map[name]) for name in QUADRANTS)
    if match_count >= 3:
        return False

    return True


def _attempt() -> np.ndarray | None:
    top_h = random.randint(*REGION_SIZE_RANGE)
    bottom_h = random.randint(*REGION_SIZE_RANGE)
    left_w = random.randint(*REGION_SIZE_RANGE)
    right_w = random.randint(*REGION_SIZE_RANGE)
    grid = np.zeros((top_h + 1 + bottom_h, left_w + 1 + right_w), dtype=int)
    grid[top_h, :] = GREY
    grid[:, left_w] = GREY

    dominant_colors = random.sample(list(ASSIGNED_COLORS), 4)
    dominant_map = {name: color for name, color in zip(QUADRANTS, dominant_colors)}

    chosen_lengths = random.sample(list(DIAGONAL_LENGTHS), 4)
    random.shuffle(chosen_lengths)
    length_map = {name: length for name, length in zip(QUADRANTS, chosen_lengths)}

    region_shapes = {
        "tl": (top_h, left_w),
        "tr": (top_h, right_w),
        "bl": (bottom_h, left_w),
        "br": (bottom_h, right_w),
    }
    region_origins = {
        "tl": (0, 0),
        "tr": (0, left_w + 1),
        "bl": (top_h + 1, 0),
        "br": (top_h + 1, left_w + 1),
    }

    for name in QUADRANTS:
        patch_and_meta = _build_region(
            region_shapes[name][0],
            region_shapes[name][1],
            dominant_map[name],
            length_map[name],
        )
        if patch_and_meta is None:
            return None
        patch, meta = patch_and_meta
        if meta["main_count"] < meta["noise_count"] * 3:
            return None
        paste(grid, patch, region_origins[name][0], region_origins[name][1])

    expected = _render_expected(length_map)
    if not _valid_semantics(grid, expected, dominant_map, length_map):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
