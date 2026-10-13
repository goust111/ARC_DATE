from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, paste


BG = 0
SEP = 5
BAND_H = 3
ROW_STRIDE = 4
WIDTH_CHOICES = (21, 23, 25, 27, 29)
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 500

SIDE_PATTERNS = (
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
)

MIDDLE_TEMPLATES = (
    np.array([[1, 2, 0], [1, 1, 3], [0, 3, 2]], dtype=int),
    np.array([[0, 1, 2], [1, 1, 2], [3, 0, 2]], dtype=int),
    np.array([[2, 1, 0], [1, 1, 3], [0, 2, 3]], dtype=int),
    np.array([[1, 2, 2], [0, 1, 3], [0, 3, 0]], dtype=int),
    np.array([[2, 1, 0], [2, 1, 3], [0, 1, 3]], dtype=int),
    np.array([[1, 1, 2], [0, 2, 3], [0, 0, 3]], dtype=int),
)


def _separator_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(int(grid.shape[0])) if bool((grid[int(r), :] == SEP).all())]


def _band_ranges(grid: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for r in _separator_rows(grid) + [int(grid.shape[0])]:
        if start < int(r):
            bands.append((int(start), int(r)))
        start = int(r + 1)
    return bands


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _tile_info(band: np.ndarray, comp: dict, bg: int) -> dict:
    tile = crop_bbox(band, comp["bbox"])
    colors = [int(v) for v in np.unique(tile) if int(v) != int(bg)]
    return {
        "bbox": tuple(map(int, comp["bbox"])),
        "coords": comp["coords"],
        "tile": tile,
        "mask": _mask_from_coords(comp["coords"]),
        "colors": colors,
    }


def _analyze_band(band: np.ndarray, row0: int, bg: int) -> dict:
    comps = find_components(
        band,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert len(comps) == 3, "each band must contain exactly three tiles"
    tiles = sorted((_tile_info(band, comp, bg) for comp in comps), key=lambda item: item["bbox"][1])
    left, middle, right = tiles

    assert len(left["colors"]) == 1 and len(right["colors"]) == 1, "side tiles must be monochrome"
    assert np.array_equal(left["mask"], right["mask"]), "left and right tiles must share the same shape"
    assert not np.array_equal(left["mask"], middle["tile"] != bg), "middle tile must use a different pattern"

    left_gap_edge = int(left["bbox"][3])
    right_gap_edge = int(right["bbox"][1])
    gap = int(right_gap_edge - left_gap_edge - 1)
    assert gap > 0, "left and right tiles must be separated"
    assert int(left["bbox"][1]) < int(middle["bbox"][1]) < int(right["bbox"][1]), "tile order must be left-middle-right"

    return {
        "row0": int(row0),
        "left": left,
        "middle": middle,
        "right": right,
        "gap": gap,
        "left_color": int(left["colors"][0]),
        "right_color": int(right["colors"][0]),
    }


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    bands: list[dict] = []
    for row0, row1 in _band_ranges(g):
        assert int(row1 - row0) == BAND_H, "each band must be exactly three rows tall"
        bands.append(_analyze_band(g[int(row0):int(row1), :], int(row0), bg))
    assert 4 <= len(bands) <= 5, "expected four or five bands"

    gaps = [int(band["gap"]) for band in bands]
    min_gap = min(gaps)
    teaching = [idx for idx, gap in enumerate(gaps) if int(gap) == int(min_gap)]
    assert len(teaching) == 1, "the minimum gap must be unique"
    return bands, int(teaching[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the band with the smallest left-right gap to choose one global recoloring rule for every middle tile.

    Input structure:
    - Black background rows are interrupted by full gray separator rows that split the grid into 4 or 5 horizontal bands.
    - Each band is exactly 3 rows tall and contains three disconnected 3x3-style tiles ordered left, middle, right.
    - Within a band, the left and right tiles have the same shape but different monochrome colors, while the middle tile has a different pattern and may use multiple colors.

    Rule:
    - Find the unique band whose left tile and right tile have the smallest horizontal bbox gap.
    - Read that teaching band's side colors as one mapping: left color A maps to right color B.
    - Recolor every middle tile in every band by replacing only color A with color B.
    - Keep all side tiles and gray separator rows unchanged.

    Key insight:
    - A local spatial cue, the smallest side-to-side gap, selects the one band whose color pair defines a global recoloring broadcast.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bands, teaching_idx = _analyze_grid(g)
    src = int(bands[teaching_idx]["left_color"])
    dst = int(bands[teaching_idx]["right_color"])

    for band in bands:
        row0 = int(band["row0"])
        coords = band["middle"]["coords"]
        rr = coords[:, 0] + row0
        cc = coords[:, 1]
        hit = out[rr, cc] == src
        out[rr[hit], cc[hit]] = dst
    return out


def _left_choices(width: int) -> list[int]:
    mid_c = (int(width) - 3) // 2
    return list(range(1, int(mid_c - 3)))


def _side_tile(pattern: np.ndarray, color: int) -> np.ndarray:
    return np.where(pattern.astype(bool), int(color), BG).astype(int)


def _pick_two_distinct(options: list[int]) -> tuple[int, int]:
    a, b = random.sample(options, 2)
    return int(a), int(b)


def _render_middle(template: np.ndarray, local_a: int, selected_a: int, selected_b: int, mode: str) -> np.ndarray:
    tile = np.zeros_like(template, dtype=int)
    tile[template == 1] = int(local_a)

    if mode == "change":
        tile[template == 2] = int(selected_a)
        pool = [c for c in PALETTE if c not in {int(local_a), int(selected_a)}]
        tile[template == 3] = int(random.choice(pool if pool else [selected_b]))
    elif mode == "unchanged":
        pool = [c for c in PALETTE if c not in {int(local_a), int(selected_a)}]
        if len(pool) >= 2:
            c2, c3 = _pick_two_distinct(pool)
        else:
            c2 = c3 = int(pool[0])
        tile[template == 2] = int(c2)
        tile[template == 3] = int(c3)
    elif mode == "teacher":
        pool2 = [int(selected_b)] + [c for c in PALETTE if c != int(local_a)]
        tile[template == 2] = int(random.choice(pool2))
        pool3 = [c for c in PALETTE if c != int(local_a)]
        tile[template == 3] = int(random.choice(pool3))
    else:
        include_selected = int(local_a) == int(selected_a) or random.random() < 0.45
        if include_selected and int(local_a) != int(selected_a):
            tile[template == 2] = int(selected_a)
            pool = [c for c in PALETTE if c not in {int(local_a), int(selected_a)}]
            tile[template == 3] = int(random.choice(pool if pool else [selected_b]))
        else:
            pool = [c for c in PALETTE if c not in {int(local_a), int(selected_a)}]
            if len(pool) >= 2:
                c2, c3 = _pick_two_distinct(pool)
            elif len(pool) == 1:
                c2 = c3 = int(pool[0])
            else:
                c2 = c3 = int(selected_b)
            tile[template == 2] = int(c2)
            tile[template == 3] = int(c3)
    return tile


def _expected_output(inp: np.ndarray, middle_tiles: list[np.ndarray], mid_c: int, selected_a: int, selected_b: int) -> np.ndarray:
    out = np.array(inp, copy=True)
    for band_i, tile in enumerate(middle_tiles):
        recolored = tile.copy()
        recolored[recolored == int(selected_a)] = int(selected_b)
        paste(out, recolored, int(band_i * ROW_STRIDE), int(mid_c), transparent=BG)
    return out


def _validate_case(inp: np.ndarray, expected: np.ndarray, spec: dict) -> bool:
    try:
        analyzed, teaching_idx = _analyze_grid(inp)
    except AssertionError:
        return False

    if int(teaching_idx) != int(spec["teaching_idx"]):
        return False
    if len(analyzed) != int(spec["band_count"]):
        return False

    left_colors = [int(band["left_color"]) for band in analyzed]
    right_colors = [int(band["right_color"]) for band in analyzed]
    gaps = [int(band["gap"]) for band in analyzed]
    if left_colors != list(map(int, spec["left_colors"])):
        return False
    if right_colors != list(map(int, spec["right_colors"])):
        return False
    if gaps != list(map(int, spec["gaps"])):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    diff = inp != expected
    allowed = np.zeros_like(diff, dtype=bool)
    mid_c = int(spec["mid_c"])
    for band_i, tile in enumerate(spec["middle_tiles"]):
        r0 = int(band_i * ROW_STRIDE)
        allowed[r0:r0 + BAND_H, mid_c:mid_c + 3] |= tile != BG
    if bool((diff & ~allowed).any()):
        return False

    changed_bands = []
    for band_i in range(int(spec["band_count"])):
        r0 = int(band_i * ROW_STRIDE)
        if bool(diff[r0:r0 + BAND_H, mid_c:mid_c + 3].any()):
            changed_bands.append(int(band_i))
    if int(spec["teaching_idx"]) not in changed_bands:
        return False
    if int(spec["change_band"]) not in changed_bands:
        return False
    if int(spec["unchanged_band"]) in changed_bands:
        return False
    if len(changed_bands) < 2:
        return False

    if len(set(zip(spec["left_colors"], spec["right_colors"]))) < 2:
        return False
    if len(set(spec["gaps"])) != len(spec["gaps"]):
        return False
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    band_count = random.randint(4, 5)
    width = random.choice([w for w in WIDTH_CHOICES if len(_left_choices(w)) >= band_count])
    height = int(band_count * BAND_H + (band_count - 1))
    mid_c = (width - 3) // 2

    teaching_idx = random.randrange(band_count)
    left_slots = sorted(random.sample(_left_choices(width), band_count))
    left_cols = [0] * band_count
    left_cols[teaching_idx] = int(left_slots[-1])
    remaining_slots = left_slots[:-1]
    random.shuffle(remaining_slots)
    for band_i in range(band_count):
        if band_i != teaching_idx:
            left_cols[band_i] = int(remaining_slots.pop())
    right_cols = [int(width - 3 - c) for c in left_cols]
    gaps = [int(right_cols[i] - (left_cols[i] + 3)) for i in range(band_count)]

    selected_a, selected_b = _pick_two_distinct(list(PALETTE))
    left_colors: list[int | None] = [None] * band_count
    right_colors: list[int | None] = [None] * band_count
    left_colors[teaching_idx] = int(selected_a)
    right_colors[teaching_idx] = int(selected_b)

    non_teaching = [idx for idx in range(band_count) if idx != teaching_idx]
    change_band, unchanged_band = random.sample(non_teaching, 2)
    non_selected = [c for c in PALETTE if c != int(selected_a)]
    left_colors[change_band] = int(random.choice(non_selected))
    left_colors[unchanged_band] = int(random.choice(non_selected))

    remaining = [idx for idx in non_teaching if idx not in {change_band, unchanged_band}]
    conflict_band = None
    if remaining and random.random() < 0.65:
        conflict_band = random.choice(remaining)
        left_colors[conflict_band] = int(selected_a)

    for band_i in range(band_count):
        if left_colors[band_i] is None:
            left_colors[band_i] = int(random.choice(PALETTE))

    for band_i in range(band_count):
        choices = [c for c in PALETTE if c != int(left_colors[band_i])]
        if band_i == conflict_band:
            choices = [c for c in choices if c != int(selected_b)]
        right_colors[band_i] = int(random.choice(choices))

    if len(set(zip(left_colors, right_colors))) < 2:
        return None
    if int(left_colors[change_band]) == int(selected_a) or int(left_colors[unchanged_band]) == int(selected_a):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for band_i in range(band_count - 1):
        grid[int(band_i * ROW_STRIDE + BAND_H), :] = SEP

    middle_tiles: list[np.ndarray] = []
    for band_i in range(band_count):
        side_pattern = random.choice(SIDE_PATTERNS)
        template_choices = [tmpl for tmpl in MIDDLE_TEMPLATES if not np.array_equal(tmpl > 0, side_pattern.astype(bool))]
        middle_template = random.choice(template_choices)

        left_tile = _side_tile(side_pattern, int(left_colors[band_i]))
        right_tile = _side_tile(side_pattern, int(right_colors[band_i]))

        if band_i == teaching_idx:
            mode = "teacher"
        elif band_i == change_band:
            mode = "change"
        elif band_i == unchanged_band:
            mode = "unchanged"
        else:
            mode = "random"
        middle_tile = _render_middle(
            middle_template,
            int(left_colors[band_i]),
            int(selected_a),
            int(selected_b),
            mode,
        )

        r0 = int(band_i * ROW_STRIDE)
        paste(grid, left_tile, r0, int(left_cols[band_i]), transparent=BG)
        paste(grid, middle_tile, r0, int(mid_c), transparent=BG)
        paste(grid, right_tile, r0, int(right_cols[band_i]), transparent=BG)
        middle_tiles.append(middle_tile)

    expected = _expected_output(grid, middle_tiles, mid_c, int(selected_a), int(selected_b))
    spec = {
        "band_count": int(band_count),
        "teaching_idx": int(teaching_idx),
        "change_band": int(change_band),
        "unchanged_band": int(unchanged_band),
        "left_colors": [int(v) for v in left_colors],
        "right_colors": [int(v) for v in right_colors],
        "gaps": gaps,
        "mid_c": int(mid_c),
        "middle_tiles": middle_tiles,
    }

    if not _validate_case(grid, expected, spec):
        return None
    return grid, expected, spec


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
