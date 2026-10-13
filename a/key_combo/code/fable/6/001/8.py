from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, get_bg


BG = 0
SEP = 5
PALETTE = tuple(color for color in range(10) if color not in {BG, SEP})
TRANSFORMS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)
NON_ID_TRANSFORMS = tuple(name for name in TRANSFORMS if name != "id")
GEN_RETRIES = 500


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1).copy()
    if name == "rot180":
        return np.rot90(arr, 2).copy()
    if name == "rot270":
        return np.rot90(arr, 3).copy()
    if name == "flip_lr":
        return np.fliplr(arr).copy()
    if name == "flip_ud":
        return np.flipud(arr).copy()
    if name == "transpose":
        return arr.T.copy()
    if name == "anti_transpose":
        return np.flipud(np.fliplr(arr)).T.copy()
    raise ValueError(f"unknown transform: {name}")


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    data = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, data.shape)), data.tobytes()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return crop_bbox(mask.astype(bool), bbox_coords(coords)).astype(bool)


def _sprite_to_mask(sprite: np.ndarray, bg: int = BG) -> np.ndarray:
    return _crop_mask(np.asarray(sprite) != int(bg))


def _sprite_from_mask(mask: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[np.asarray(mask, dtype=bool)] = int(color)
    return out


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    return len({_mask_signature(_apply_transform(mask, name)) for name in TRANSFORMS}) == len(TRANSFORMS)


def _canonical_asym_mask() -> np.ndarray:
    candidates = [
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]], dtype=bool),
    ]
    for cand in candidates:
        if _is_fully_asymmetric(cand):
            return cand
    raise AssertionError("failed to define a canonical asymmetric mask")


CANONICAL_ASYM_MASK = _canonical_asym_mask()


def _good_transform_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    base = CANONICAL_ASYM_MASK
    id_sig = _mask_signature(base)
    single_sigs = {name: _mask_signature(_apply_transform(base, name)) for name in NON_ID_TRANSFORMS}
    for first in NON_ID_TRANSFORMS:
        for second in NON_ID_TRANSFORMS:
            composed = _apply_transform(_apply_transform(base, first), second)
            reverse = _apply_transform(_apply_transform(base, second), first)
            composed_sig = _mask_signature(composed)
            reverse_sig = _mask_signature(reverse)
            if composed_sig == id_sig:
                continue
            if composed_sig == single_sigs[first] or composed_sig == single_sigs[second]:
                continue
            if composed_sig == reverse_sig:
                continue
            pairs.append((first, second))
    assert pairs
    return pairs


ORDER_SENSITIVE_PAIRS = _good_transform_pairs()


def _row_bands(region: np.ndarray, bg: int) -> list[tuple[int, int]]:
    occupied = np.any(region != int(bg), axis=1)
    bands: list[tuple[int, int]] = []
    r = 0
    while r < int(region.shape[0]):
        if not bool(occupied[r]):
            r += 1
            continue
        r0 = r
        while r + 1 < int(region.shape[0]) and bool(occupied[r + 1]):
            r += 1
        bands.append((int(r0), int(r)))
        r += 1
    return bands


def _extract_rows(left_region: np.ndarray, bg: int) -> list[dict]:
    rows: list[dict] = []
    for r0, r1 in _row_bands(left_region, bg):
        band = left_region[r0 : r1 + 1, :]
        comps = find_components(band, bg=bg, connectivity=4, monochromatic=True)
        assert len(comps) == 2, "each reference row must contain exactly two glyphs"
        comps = sorted(comps, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
        colors = {int(comp["color"]) for comp in comps}
        assert len(colors) == 1, "both demo glyphs in a row must share one color"
        color = int(next(iter(colors)))
        left_sprite = crop_bbox(band, comps[0]["bbox"])
        right_sprite = crop_bbox(band, comps[1]["bbox"])
        rows.append(
            {
                "color": color,
                "left": np.asarray(left_sprite, dtype=int),
                "right": np.asarray(right_sprite, dtype=int),
            }
        )
    assert rows, "expected at least one reference row"
    return rows


def _extract_query(right_region: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    comps = find_components(right_region, bg=bg, connectivity=4, monochromatic=True)
    assert len(comps) == 1, "right region must contain exactly one query glyph"
    comp = comps[0]
    return int(comp["color"]), np.asarray(crop_bbox(right_region, comp["bbox"]), dtype=int)


def _infer_transform(left_sprite: np.ndarray, right_sprite: np.ndarray, bg: int) -> str:
    left_mask = _sprite_to_mask(left_sprite, bg=bg)
    right_mask = _sprite_to_mask(right_sprite, bg=bg)
    matches = [name for name in NON_ID_TRANSFORMS if np.array_equal(_apply_transform(left_mask, name), right_mask)]
    assert len(matches) == 1, "demo row must determine a unique non-identity rigid transform"
    return matches[0]


def _separator_column(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for c in range(int(grid.shape[1])):
        column = grid[:, c]
        if int(column[0]) == int(bg):
            continue
        if np.all(column == column[0]):
            candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compose the demonstrated rigid transforms from the same-color reference rows onto the query glyph.

    Input structure:
    - The background is black, and one full-height gray separator column splits the grid into a left reference-table region and a right query region.
    - Each occupied row band in the left region contains exactly two separated same-color asymmetric glyphs; the right glyph is a rotated or mirrored copy of the left glyph.
    - The right region contains one monochrome query glyph whose color matches exactly one or two reference rows.

    Rule:
    - Find every reference row whose color matches the query glyph.
    - For each such row, infer the unique rigid transform that maps the left demo glyph to the right demo glyph.
    - Apply those inferred transforms to the query glyph in top-to-bottom row order and return the final transformed glyph as a tight crop, preserving its color.

    Key insight:
    - The table stores transformations by demonstration rather than by shape lookup, and repeated row colors mean the solver must compose multiple inferred transforms in row order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    sep_col = _separator_column(g, bg)
    left_region = g[:, :sep_col]
    right_region = g[:, sep_col + 1 :]

    rows = _extract_rows(left_region, bg)
    query_color, query_sprite = _extract_query(right_region, bg)
    matching_rows = [row for row in rows if int(row["color"]) == int(query_color)]
    assert 1 <= len(matching_rows) <= 2, "query color must match exactly one or two reference rows"

    out = np.asarray(query_sprite, dtype=int)
    for row in matching_rows:
        transform = _infer_transform(row["left"], row["right"], bg)
        out = _apply_transform(out, transform)
    return out


def _random_connected_mask() -> np.ndarray | None:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    target_area = random.randint(4, min(h * w - 1, 8))
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _crop_mask(mask)
    if min(mask.shape) < 2:
        return None
    if not _is_fully_asymmetric(mask):
        return None
    return mask


def _sample_distinct_mask(forbidden: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(200):
        mask = _random_connected_mask()
        if mask is None:
            continue
        sig = _mask_signature(mask)
        if sig in forbidden:
            continue
        return mask
    return None


def _sample_match_indices(row_count: int, match_count: int) -> set[int]:
    indices = list(range(row_count))
    if match_count == 1:
        return {random.choice(indices)}
    if row_count >= 4 and random.random() < 0.6:
        non_adjacent = [(a, b) for a in indices for b in indices if a < b and b - a >= 2]
        if non_adjacent:
            return set(random.choice(non_adjacent))
    return set(random.sample(indices, 2))


def _hidden_output(query_mask: np.ndarray, color: int, transforms: list[str]) -> np.ndarray:
    out = _sprite_from_mask(query_mask, color)
    for name in transforms:
        out = _apply_transform(out, name)
    return out


def _build_instance() -> tuple[np.ndarray, dict] | None:
    row_count = random.randint(3, 4)
    match_count = 2 if random.random() < 0.7 else 1
    match_indices = _sample_match_indices(row_count, match_count)

    query_color = random.choice(PALETTE)
    other_colors = random.sample([c for c in PALETTE if c != query_color], row_count - match_count)
    row_colors: list[int] = []
    other_iter = iter(other_colors)
    for idx in range(row_count):
        row_colors.append(query_color if idx in match_indices else next(other_iter))

    row_transforms = [random.choice(NON_ID_TRANSFORMS) for _ in range(row_count)]
    if match_count == 1:
        only_idx = next(iter(match_indices))
        row_transforms[only_idx] = random.choice(NON_ID_TRANSFORMS)
    else:
        ordered = sorted(match_indices)
        first, second = random.choice(ORDER_SENSITIVE_PAIRS)
        row_transforms[ordered[0]] = first
        row_transforms[ordered[1]] = second

    rows: list[dict] = []
    used_display_sigs: set[tuple[tuple[int, int], bytes]] = set()
    for idx in range(row_count):
        left_mask = _sample_distinct_mask(used_display_sigs)
        if left_mask is None:
            return None
        right_mask = _apply_transform(left_mask, row_transforms[idx])
        left_sig = _mask_signature(left_mask)
        right_sig = _mask_signature(right_mask)
        if right_sig in used_display_sigs or right_sig == left_sig:
            return None
        used_display_sigs.add(left_sig)
        used_display_sigs.add(right_sig)
        rows.append(
            {
                "color": row_colors[idx],
                "transform": row_transforms[idx],
                "left_mask": left_mask,
                "right_mask": right_mask,
            }
        )

    query_mask = _sample_distinct_mask(used_display_sigs)
    if query_mask is None:
        return None

    ordered_match_indices = [idx for idx, color in enumerate(row_colors) if color == query_color]
    assert len(ordered_match_indices) == match_count
    match_transforms = [row_transforms[idx] for idx in ordered_match_indices]
    expected = _hidden_output(query_mask, query_color, match_transforms)
    if _mask_signature(_sprite_to_mask(expected)) in used_display_sigs:
        return None

    row_tops: list[int] = []
    row_left_cs: list[int] = []
    row_right_cs: list[int] = []
    row_band_heights: list[int] = []
    row_leads = [random.randint(1, 2) for _ in rows]
    row_gaps = [random.randint(2, 3) for _ in rows]
    row_trails = [random.randint(1, 2) for _ in rows]

    left_zone_w = 0
    for idx, row in enumerate(rows):
        lh, lw = row["left_mask"].shape
        rh, rw = row["right_mask"].shape
        left_c = row_leads[idx]
        right_c = left_c + int(lw) + row_gaps[idx]
        width = right_c + int(rw) + row_trails[idx]
        left_zone_w = max(left_zone_w, width)
        row_left_cs.append(left_c)
        row_right_cs.append(right_c)
        row_band_heights.append(max(int(lh), int(rh)))

    current_top = 1
    for idx, band_h in enumerate(row_band_heights):
        row_tops.append(current_top)
        current_top += int(band_h)
        if idx + 1 < len(row_band_heights):
            current_top += random.randint(1, 2)
    left_needed_h = current_top + 1

    query_h, query_w = query_mask.shape
    query_left_pad = random.randint(2, 3)
    query_right_pad = random.randint(1, 3)
    right_zone_w = query_left_pad + int(query_w) + query_right_pad

    grid_h = max(left_needed_h, int(query_h) + 2)
    grid_w = left_zone_w + 1 + right_zone_w
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    sep_col = left_zone_w
    grid[:, sep_col] = SEP

    for idx, row in enumerate(rows):
        color = int(row["color"])
        left_sprite = _sprite_from_mask(row["left_mask"], color)
        right_sprite = _sprite_from_mask(row["right_mask"], color)
        top = row_tops[idx]
        grid[top : top + left_sprite.shape[0], row_left_cs[idx] : row_left_cs[idx] + left_sprite.shape[1]] = left_sprite
        grid[top : top + right_sprite.shape[0], row_right_cs[idx] : row_right_cs[idx] + right_sprite.shape[1]] = right_sprite

    query_top = random.randint(1, grid_h - int(query_h) - 1)
    query_left = sep_col + 1 + query_left_pad
    query_sprite = _sprite_from_mask(query_mask, query_color)
    grid[query_top : query_top + query_h, query_left : query_left + query_w] = query_sprite

    meta = {
        "row_colors": row_colors,
        "row_transforms": row_transforms,
        "query_color": query_color,
        "query_mask": query_mask,
        "query_sprite": query_sprite,
        "match_indices": ordered_match_indices,
        "expected": expected,
    }
    return grid, meta


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    expected = np.asarray(meta["expected"], dtype=int)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    bg = int(get_bg(grid))
    sep_col = _separator_column(grid, bg)
    rows = _extract_rows(grid[:, :sep_col], bg)
    if len(rows) != len(meta["row_colors"]):
        return False

    inferred_colors = [int(row["color"]) for row in rows]
    if inferred_colors != list(map(int, meta["row_colors"])):
        return False

    inferred_transforms = [_infer_transform(row["left"], row["right"], bg) for row in rows]
    if inferred_transforms != list(meta["row_transforms"]):
        return False

    query_color, query_sprite = _extract_query(grid[:, sep_col + 1 :], bg)
    if int(query_color) != int(meta["query_color"]):
        return False
    if _mask_signature(_sprite_to_mask(query_sprite)) != _mask_signature(meta["query_mask"]):
        return False

    display_sigs = set()
    for row in rows:
        left_sig = _mask_signature(_sprite_to_mask(row["left"], bg=bg))
        right_sig = _mask_signature(_sprite_to_mask(row["right"], bg=bg))
        if left_sig in display_sigs or right_sig in display_sigs:
            return False
        display_sigs.add(left_sig)
        display_sigs.add(right_sig)

    query_sig = _mask_signature(meta["query_mask"])
    out_sig = _mask_signature(_sprite_to_mask(expected))
    if query_sig in display_sigs or out_sig in display_sigs:
        return False

    match_indices = list(meta["match_indices"])
    if len(match_indices) not in (1, 2):
        return False
    if inferred_colors.count(int(query_color)) != len(match_indices):
        return False

    original = np.asarray(query_sprite, dtype=int)
    if len(match_indices) == 1:
        if np.array_equal(original, expected):
            return False
    else:
        first = inferred_transforms[match_indices[0]]
        second = inferred_transforms[match_indices[1]]
        first_only = _apply_transform(original, first)
        second_only = _apply_transform(original, second)
        reverse = _apply_transform(_apply_transform(original, second), first)
        if np.array_equal(expected, original):
            return False
        if np.array_equal(expected, first_only):
            return False
        if np.array_equal(expected, second_only):
            return False
        if np.array_equal(expected, reverse):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_instance()
        if built is None:
            continue
        grid, meta = built
        if _validate_instance(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid composite-lookup input")
