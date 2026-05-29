# Combo variant of 67e490f4 (index 28)
# combo_spec: D2/0+D8/1+D13/2
# Name: 严格方向+左右对称+颜色分区
#
# Design analysis for the fused variant
# - Concrete behavior:
#   The large object is still a single rectangular frame with holes, and the answer is still the
#   cropped frame bbox with each hole recolored from outside sprites. The fusion changes three
#   linked parts at once. First, matching is strict: a hole only accepts sprites with the exact same
#   bitmap, so orientation now matters. Second, the hole layout is only left-right symmetric, which
#   means a left hole and its right-side mirror should be horizontal flips of each other rather than
#   copies with the same facing. Third, outside sprites are not scattered uniformly anymore: all
#   sprites of the same color are clustered into one visible color zone in a top or bottom band.
# - Small trace:
#   Suppose the left side contains an L hole `[[1,0],[1,1]]`. In the original task, a right-side
#   copy could be treated as the same family because the solver allowed flip/rotation, and all
#   matching sprites could vote together. In this combo, the mirrored right hole is
#   `[[0,1],[1,1]]`; that is a different strict key, so the generator must provide right-facing
#   sprites explicitly. The two colors then appear in two separate outside zones, making direction
#   and color grouping simultaneously observable.
# - Complexity calibration:
#   If the generator used symmetric shapes or gave mirrored holes the same color, the D2 change
#   would be visually weak. To prevent that degeneration, the generator uses asymmetric families,
#   assigns different colors to the left and right orientations of every family, and forces the
#   old loose matcher to disagree with the strict matcher on every accepted sample.
# - Main() assumption audit:
#   Step 1 still assumes the largest monochrome non-background component is the frame; generation
#   preserves that. Step 2 already extracts exact hole masks, which is exactly what D2 needs.
#   Step 3 is the key change: the old solver expanded every sprite across 8 transforms, but that
#   assumption is now false, so each sprite contributes only to its native key. Step 4 assumes every
#   exact hole key has a sprite witness; generation guarantees this for both members of each mirrored
#   pair.
# - Placement legality:
#   Holes stay strictly inside the frame border and remain mutually 4-disconnected. A mirrored pair
#   is validated as two separate components after applying the actual flipped right-hand shape, so
#   the pair cannot merge near the vertical axis. Sprites are placed only outside the frame and only
#   inside their assigned color zones, with Chebyshev padding enforced by bbox checks.
# - Visual observability:
#   A human can see that the frame geometry is LR-only because upper and lower rows differ, yet each
#   left hole has a mirrored partner on the right. They can also see that the mirrored partner often
#   gets a different output color, while the relevant outside sprites are separated by color blocks.
#   This makes the strict-orientation rule and the color partition both inferable from a few samples.
# - Visual cleanliness and inferability:
#   The frame is kept in the middle rows, leaving clean top and bottom bands for sprite zones. The
#   number of canonical families is capped at 2-3 so the total color count stays moderate, each color
#   repeats multiple times, and the outside grouping remains readable instead of noisy.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
)


MaskKey = tuple[tuple[int, int], bytes]


def _mask_key(mask: np.ndarray) -> MaskKey:
    m = mask.astype(bool)
    return (m.shape, m.tobytes())


def _all_rot_flip(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield _mask_key(m2), m2


def _canonical_key(mask: np.ndarray) -> MaskKey:
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[MaskKey] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _extract_frame(grid: np.ndarray, bg: int) -> dict:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    return frame_candidates[0]


def _extract_holes(sub: np.ndarray, frame_col: int):
    holes = []
    hole_mask = sub != int(frame_col)
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((_mask_key(mask), (int(hr0), int(hc0)), mask, coords))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the holes of a left-right mirrored frame using strictly oriented outside sprites.

    Input structure:
    - The most common color is the background, and one large monochrome component is the frame.
    - Inside the frame bbox, hole components form a layout that is mirrored left-to-right but not
      necessarily top-to-bottom, so mirrored partners can face opposite directions.
    - Outside the frame, small sprites sit on the background and same-colored sprites are clustered
      into color-specific regions.

    Rule:
    - Locate the largest frame, crop to its bounding box, and start from a solid rectangle of the
      frame color.
    - Extract each hole by its exact bitmap and only count outside sprites with that same bitmap;
      rotated or flipped look-alikes do not match.
    - Fill the hole with the most frequent color among those exact matches.

    Key insight:
    - The frame geometry is mirrored, but the color key is stricter than the symmetry: a mirrored
      partner is usually a flipped shape, so it needs its own outside sprite population and can land
      on a different color block.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frame = _extract_frame(g, bg)
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)
    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    holes = _extract_holes(sub, frame_col)

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[MaskKey, dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        bucket = sprites_by_key.setdefault(_mask_key(mask), {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask, _coords in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        paste(out, np.full(hmask.shape, int(scol), dtype=int), int(hr0), int(hc0), mask=hmask)

    return out


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (8, 18)
FRAME_DENSITY_RANGE = (0.58, 0.90)

FAMILY_COUNT_RANGE = (2, 3)
PLACEMENT_RETRIES = 120
PLACEMENT_MULT_VALUES = [1, 2, 3]
PLACEMENT_MULT_WEIGHTS = [4, 3, 1]
MIN_TOTAL_HOLES = 6
MIN_ROW_GAP = 4
SHAPE_MAX_AREA = 8

ZONE_BAND_HEIGHT = 7
SPRITE_PADDING = 1
WINNING_SPRITE_RANGE = (2, 3)
DOMINANT_BONUS_RANGE = (1, 2)
LOSER_SPRITE_RANGE = (1, 2)

MAX_GEN_ATTEMPTS = 500


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
    ]
    seen: set[MaskKey] = set()
    out: list[np.ndarray] = []
    for arr in raw:
        mask = np.array(arr, dtype=bool)
        if int(np.sum(mask)) > SHAPE_MAX_AREA:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        out.append(mask)
    return out


SHAPE_LIBRARY = _build_shape_library()


def _layout_metrics(occ: np.ndarray) -> dict[str, int | bool]:
    hole_mask = ~occ
    row_counts = np.sum(hole_mask, axis=1).astype(int)
    mid = hole_mask.shape[0] // 2
    top_cells = int(np.sum(row_counts[:mid]))
    bottom_cells = int(np.sum(row_counts[(hole_mask.shape[0] + 1) // 2 :]))
    return {
        "lr_ok": bool(np.array_equal(hole_mask, np.fliplr(hole_mask))),
        "ud_ok": bool(np.array_equal(hole_mask, np.flipud(hole_mask))),
        "row_gap": int(np.sum(np.abs(row_counts - row_counts[::-1]))),
        "top_cells": top_cells,
        "bottom_cells": bottom_cells,
    }


def _candidate_rows(th: int, sh: int, band: str) -> list[int]:
    max_r = th - sh - 1
    if max_r < 1:
        return []
    if band == "any":
        return list(range(1, max_r + 1))
    split = max(1, max_r // 2)
    if band == "top":
        return list(range(1, split + 1))
    if band == "bottom":
        start = min(max_r, split + 1)
        return list(range(start, max_r + 1))
    raise ValueError(f"unknown band: {band}")


def _group_mask_from_cells(cells: set[tuple[int, int]], th: int, tw: int) -> np.ndarray:
    mask = np.zeros((th, tw), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    return mask


def _validate_pair(
    occ: np.ndarray,
    left_shape: np.ndarray,
    right_shape: np.ndarray,
    r: int,
    c_left: int,
    th: int,
    tw: int,
):
    sh, sw = left_shape.shape
    assert right_shape.shape == left_shape.shape
    c_right = tw - c_left - sw

    pieces = ((left_shape, r, c_left), (right_shape, r, c_right))
    new_cells: set[tuple[int, int]] = set()
    for shape, pr, pc in pieces:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if not shape[dr, dc]:
                    continue
                rr, cc = pr + dr, pc + dc
                if not occ[rr, cc]:
                    return None
                new_cells.add((rr, cc))

    group_mask = _group_mask_from_cells(new_cells, th, tw)
    if len(components_mask(group_mask, connectivity=4)) != 2:
        return None

    for rr, cc in new_cells:
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                return None

    return {
        "left": (int(r), int(c_left)),
        "right": (int(r), int(c_right)),
    }


def _carve_pair(occ: np.ndarray, left_shape: np.ndarray, right_shape: np.ndarray, pair: dict) -> None:
    for shape, side in ((left_shape, "left"), (right_shape, "right")):
        pr, pc = pair[side]
        sh, sw = shape.shape
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _try_place_lr_pair(
    occ: np.ndarray,
    left_shape: np.ndarray,
    right_shape: np.ndarray,
    th: int,
    tw: int,
    band: str,
):
    sh, sw = left_shape.shape
    rows = _candidate_rows(th, sh, band)
    max_c = (tw - 2 * sw - 1) // 2
    if not rows or max_c < 1:
        return None

    row_order = rows[:]
    random.shuffle(row_order)
    col_order = list(range(1, max_c + 1))
    random.shuffle(col_order)
    for r in row_order:
        for c_left in col_order:
            pair = _validate_pair(occ, left_shape, right_shape, r, c_left, th, tw)
            if pair is not None:
                return pair
    return None


def _fitting_orientations(shape: np.ndarray, max_sh: int, max_sw: int) -> list[np.ndarray]:
    out = []
    for oriented in _all_orientations(shape):
        if oriented.shape[0] > max_sh or oriented.shape[1] > max_sw:
            continue
        if int(np.sum(oriented)) > SHAPE_MAX_AREA:
            continue
        if np.array_equal(oriented, np.fliplr(oriented)):
            continue
        out.append(oriented)
    return out


def _select_families(th: int, tw: int):
    max_sh = th - 2
    max_sw = max(1, (tw - 3) // 2)
    candidates = []
    for base in SHAPE_LIBRARY:
        orientations = _fitting_orientations(base, max_sh, max_sw)
        if orientations:
            candidates.append((_canonical_key(base), orientations))

    if len(candidates) < FAMILY_COUNT_RANGE[0]:
        return None

    family_count = random.randint(FAMILY_COUNT_RANGE[0], min(FAMILY_COUNT_RANGE[1], len(candidates)))
    random.shuffle(candidates)
    selected = []
    for family_key, orientations in candidates[:family_count]:
        left_shape = random.choice(orientations)
        selected.append(
            {
                "family": family_key,
                "left_shape": left_shape,
                "right_shape": np.fliplr(left_shape),
            }
        )
    return selected


def _build_hole_layout(th: int, tw: int):
    families = _select_families(th, tw)
    if families is None:
        return None

    occ = np.ones((th, tw), dtype=bool)
    placed_families = []

    forced_bands = ["top", "bottom"] + ["any"] * max(0, len(families) - 2)
    random.shuffle(forced_bands[2:])

    for family, first_band in zip(families, forced_bands):
        mult = random.choices(PLACEMENT_MULT_VALUES, weights=PLACEMENT_MULT_WEIGHTS)[0]
        placements = []
        bands = [first_band] + [random.choice(["any", "top", "bottom"]) for _ in range(mult - 1)]
        for idx, band in enumerate(bands):
            pair = None
            for _ in range(PLACEMENT_RETRIES):
                pair = _try_place_lr_pair(
                    occ,
                    family["left_shape"],
                    family["right_shape"],
                    th,
                    tw,
                    band,
                )
                if pair is not None:
                    break
            if pair is None:
                if idx == 0:
                    return None
                continue
            _carve_pair(occ, family["left_shape"], family["right_shape"], pair)
            placements.append(pair)
        if not placements:
            return None
        placed_families.append(
            {
                "family": family["family"],
                "left_shape": family["left_shape"],
                "right_shape": family["right_shape"],
                "placements": placements,
            }
        )

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = 2 * sum(len(family["placements"]) for family in placed_families)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    metrics = _layout_metrics(occ)
    if not metrics["lr_ok"]:
        return None
    if metrics["ud_ok"]:
        return None
    if metrics["row_gap"] < MIN_ROW_GAP:
        return None
    if metrics["top_cells"] == 0 or metrics["bottom_cells"] == 0:
        return None

    return occ, placed_families


def _build_output(occ: np.ndarray, placed_families: list[dict], frame_col: int, bg: int):
    group_count = 2 * len(placed_families)
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < group_count:
        return None

    chosen = random.sample(avail_colors, group_count)
    output = np.full(occ.shape, frame_col, dtype=int)
    shape_info = []
    color_idx = 0

    for family in placed_families:
        left_color = int(chosen[color_idx])
        right_color = int(chosen[color_idx + 1])
        color_idx += 2
        if left_color == right_color:
            return None

        left_positions = []
        right_positions = []
        for pair in family["placements"]:
            lr, lc = pair["left"]
            rr, rc = pair["right"]
            paste(output, np.full(family["left_shape"].shape, left_color, dtype=int), lr, lc, mask=family["left_shape"])
            paste(output, np.full(family["right_shape"].shape, right_color, dtype=int), rr, rc, mask=family["right_shape"])
            left_positions.append((int(lr), int(lc)))
            right_positions.append((int(rr), int(rc)))

        shape_info.append(
            {
                "family": family["family"],
                "side": "left",
                "shape": family["left_shape"],
                "fill_color": left_color,
                "positions": left_positions,
            }
        )
        shape_info.append(
            {
                "family": family["family"],
                "side": "right",
                "shape": family["right_shape"],
                "fill_color": right_color,
                "positions": right_positions,
            }
        )

    return output, shape_info


def _split_band_into_zones(row0: int, row1: int, count: int) -> list[tuple[int, int, int, int]]:
    if count <= 0:
        return []
    gap_total = count - 1
    usable = GRID_SIZE - gap_total
    base = usable // count
    extra = usable % count
    zones = []
    c = 0
    for idx in range(count):
        width = base + (1 if idx < extra else 0)
        zones.append((row0, c, row1, c + width - 1))
        c += width + 1
    return zones


def _make_color_zones(colors: list[int]) -> dict[int, tuple[int, int, int, int]]:
    shuffled = colors[:]
    random.shuffle(shuffled)
    top_n = (len(shuffled) + 1) // 2
    bottom_n = len(shuffled) - top_n
    top_zones = _split_band_into_zones(0, ZONE_BAND_HEIGHT - 1, top_n)
    bottom_zones = _split_band_into_zones(GRID_SIZE - ZONE_BAND_HEIGHT, GRID_SIZE - 1, bottom_n)
    zones = {}
    for color, zone in zip(shuffled[:top_n], top_zones):
        zones[int(color)] = zone
    for color, zone in zip(shuffled[top_n:], bottom_zones):
        zones[int(color)] = zone
    return zones


def _place_sprite_in_zone(
    inp: np.ndarray,
    placement: np.ndarray,
    shape: np.ndarray,
    color: int,
    zone: tuple[int, int, int, int],
    bg: int,
) -> bool:
    zr0, zc0, zr1, zc1 = zone
    h, w = shape.shape
    if zr1 - zr0 + 1 < h or zc1 - zc0 + 1 < w:
        return False

    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = int(color)
    candidates = [
        (r, c)
        for r in range(zr0, zr1 - h + 2)
        for c in range(zc0, zc1 - w + 2)
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0 = max(0, r - SPRITE_PADDING)
        rr1 = min(GRID_SIZE, r + h + SPRITE_PADDING)
        cc0 = max(0, c - SPRITE_PADDING)
        cc1 = min(GRID_SIZE, c + w + SPRITE_PADDING)
        if np.any(placement[rr0:rr1, cc0:cc1] != bg):
            continue
        paste(placement, patch, r, c, transparent=bg)
        paste(inp, patch, r, c, transparent=bg)
        return True
    return False


def _scatter_sprites(inp: np.ndarray, shape_info: list[dict], bg: int) -> bool:
    placement = inp.copy()
    fill_colors = sorted({int(info["fill_color"]) for info in shape_info})
    color_zones = _make_color_zones(fill_colors)

    families: dict[MaskKey, list[dict]] = {}
    for info in shape_info:
        families.setdefault(info["family"], []).append(info)

    for infos in families.values():
        if len(infos) != 2:
            return False
        dominant_side = random.choice(["left", "right"])
        for info in infos:
            zone = color_zones[int(info["fill_color"])]
            n_win = random.randint(*WINNING_SPRITE_RANGE)
            if info["side"] == dominant_side:
                n_win += random.randint(*DOMINANT_BONUS_RANGE)
            for _ in range(n_win):
                if not _place_sprite_in_zone(inp, placement, info["shape"], int(info["fill_color"]), zone, bg):
                    return False

            loser_colors = [c for c in fill_colors if c != int(info["fill_color"])]
            if loser_colors:
                loser_color = random.choice(loser_colors)
                loser_zone = color_zones[int(loser_color)]
                n_lose = random.randint(LOSER_SPRITE_RANGE[0], min(LOSER_SPRITE_RANGE[1], max(1, n_win - 1)))
                for _ in range(n_lose):
                    if not _place_sprite_in_zone(inp, placement, info["shape"], int(loser_color), loser_zone, bg):
                        break
    return True


def _loose_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frame = _extract_frame(g, bg)
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)
    sub = crop_bbox(g, fb)
    th, tw = sub.shape
    holes = _extract_holes(sub, frame_col)

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[MaskKey, dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        for key, _oriented in _all_rot_flip(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask, _coords in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        paste(out, np.full(hmask.shape, int(scol), dtype=int), int(hr0), int(hc0), mask=hmask)
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame = _extract_frame(g, bg)
    frame_col = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])

    sub = crop_bbox(g, frame["bbox"])
    hole_mask = sub != frame_col
    row_counts = np.sum(hole_mask, axis=1).astype(int)

    holes = _extract_holes(sub, frame_col)
    coords_to_key: dict[frozenset[tuple[int, int]], MaskKey] = {}
    for key, _pos, _mask, coords in holes:
        coords_to_key[frozenset((int(rr), int(cc)) for rr, cc in coords.tolist())] = key

    tw = sub.shape[1]
    seen: set[frozenset[tuple[int, int]]] = set()
    asym_pairs = 0
    for coords_key, key in coords_to_key.items():
        if coords_key in seen:
            continue
        mirrored = frozenset((rr, tw - 1 - cc) for rr, cc in coords_key)
        if mirrored in coords_to_key and mirrored not in seen and mirrored != coords_key:
            if coords_to_key[mirrored] != key:
                asym_pairs += 1
            seen.add(coords_key)
            seen.add(mirrored)

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    colors = sorted(set(int(v) for v in np.unique(outside)) - {int(bg), int(frame_col)})
    compact_colors = 0
    for color in colors:
        coords = np.argwhere(outside == color)
        if coords.size == 0:
            continue
        cr0, _cc0 = coords.min(axis=0)
        cr1, _cc1 = coords.max(axis=0)
        in_top = int(cr1) < ZONE_BAND_HEIGHT
        in_bottom = int(cr0) >= GRID_SIZE - ZONE_BAND_HEIGHT
        if (in_top or in_bottom) and int(cr1 - cr0 + 1) <= ZONE_BAND_HEIGHT:
            compact_colors += 1

    strict_out = main(g)
    loose_out = _loose_main(g)
    return {
        "lr_ok": bool(np.array_equal(hole_mask, np.fliplr(hole_mask))),
        "ud_ok": bool(np.array_equal(hole_mask, np.flipud(hole_mask))),
        "row_gap": int(np.sum(np.abs(row_counts - row_counts[::-1]))),
        "asym_mirror_pairs": int(asym_pairs),
        "color_partition_ok": bool(compact_colors == len(colors) and len(colors) > 0),
        "strict_diff_cells": int(np.sum(strict_out != loose_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, placed_families = layout

    built = _build_output(occ, placed_families, frame_col, bg)
    if built is None:
        return None
    output, shape_info = built

    fr_min = ZONE_BAND_HEIGHT
    fr_max = GRID_SIZE - ZONE_BAND_HEIGHT - th
    if fr_min > fr_max:
        return None
    fr = random.randint(fr_min, fr_max)
    fc = random.randint(0, GRID_SIZE - tw)

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    if not _scatter_sprites(inp, shape_info, bg):
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
        metrics = _behavior_metrics(inp)
        if not metrics["lr_ok"]:
            return None
        if metrics["ud_ok"]:
            return None
        if int(metrics["asym_mirror_pairs"]) < 2:
            return None
        if not metrics["color_partition_ok"]:
            return None
        if int(metrics["strict_diff_cells"]) <= 0:
            return None
        return inp
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
