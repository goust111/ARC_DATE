# Combo variant of 67e490f4 (index 21)
# combo_spec: D1/3+D5/0
# Name: 子图案排列匹配+4连通

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Original task: each hole is a single 4-connected blob, each outside sprite is one
#    8-connected blob, and matching is direct mask equality up to rotation/flip.
#    New task: one logical "hole" is an 8-neighbor cluster made from multiple 4-connected
#    mini-blocks; one logical outside sprite is built the same way. The solver must first
#    notice that diagonal contact splits into several 4-components, then match groups by
#    the arrangement of those subcomponents instead of by one solid silhouette.
#    Example: a hole group may be three 3x3 tiles at diagonal positions inside a 9x9
#    bbox, while an outside clue uses the same tile positions but with notched 3x3 tiles.
#    Under the original logic that becomes a silhouette mismatch; under the fused logic
#    they still match because the relative sub-block layout is the same.
# b) Complexity calibration:
#    Every arrangement type has 2-4 repeated mini-blocks and appears multiple times in
#    the frame and among the outside sprites. This is visually different from direct
#    shape matching, but it keeps the number of arrangement types small enough (3-5) that
#    the color rule remains inferable from a few examples.
# c) Broken assumptions in the original main():
#    - `hole_mask` 4-components are no longer the logical destinations.
#    - source sprites are no longer usable as one `mask.tobytes()` key.
#    - diagonal contact can no longer be ignored because it controls whether a source or
#      hole is one part or several parts.
#    The new solver therefore groups by 8-neighbor clusters, then decomposes each group
#    into 4-connected subparts and canonicalizes their relative layout.
# d) Placement legality:
#    The generator places each logical group as a tight patch on a blank placement canvas
#    with `padding=1`, so different groups never become one 8-connected cluster and never
#    touch the frame border. Outside sprites use the same padding against each other and
#    against the frame bbox.
# e) Visual observability:
#    The defining signal is visible: each meaningful object looks like 2-4 separate little
#    tiles arranged diagonally / in a diamond / in a zigzag. Human solvers can see that
#    the frame holes and the outside sprites share these constellations even though the
#    source tiles have small internal notches.
# f) Visual cleanliness:
#    Hole tiles stay as clean 3x3 blocks, source tiles stay within the same 3x3 boxes, and
#    groups are spaced apart. This keeps the arrangement cue strong without ambiguous overlap.
# g) Inferrable change rule:
#    Colors are not transformed by hidden numeric properties. For each arrangement type,
#    the fill color is simply the most frequent color among outside sprites with the same
#    sub-block arrangement. Distinct arrangement types get repeated evidence.


# ---------------- Solver helpers ----------------

NUM_COLORS = 10
GRID_SIZE = 32

TILE = 3
FRAME_H_RANGE = (15, 17)
FRAME_W_RANGE = (18, 21)
ARRANGEMENT_COUNT_RANGE = (3, 3)
ARRANGEMENT_MULT_RANGE = (1, 1)
MIN_TOTAL_GROUPS = 3
FRAME_DENSITY_RANGE = (0.60, 0.86)
PLACEMENT_RETRIES = 60

WINNING_SPRITE_RANGE = (2, 2)
OTHER_COLOR_MAX = 1
SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 400

HOLE_TILE = np.ones((TILE, TILE), dtype=bool)
SOURCE_TILE_VARIANTS = [
    np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=bool),
]


def _all_rot_flip_masks(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            yield np.rot90(m1, k)


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for m in _all_rot_flip_masks(mask):
        key = (m.shape, m.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(m.copy())
    return out


def _group_signature(mask: np.ndarray) -> tuple[tuple[int, int, int, int], ...]:
    """Canonical arrangement key based on subpart layout, ignoring each subpart's interior."""
    best = None
    for oriented in _all_rot_flip_masks(mask):
        parts = []
        for coords in components_mask(oriented, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(coords)
            parts.append((int(r1 - r0 + 1), int(c1 - c0 + 1), int(r0), int(c0)))
        min_r = min(p[2] for p in parts)
        min_c = min(p[3] for p in parts)
        norm = tuple(sorted((h, w, r - min_r, c - min_c) for h, w, r, c in parts))
        if best is None or norm < best:
            best = norm
    assert best is not None
    return best


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    rel = coords - np.array([r0, c0])
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _extract_arrangement_groups(mask: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int], np.ndarray]]:
    groups = []
    for coords in components_mask(mask, connectivity=8):
        r0, c0, r1, c1 = bbox_coords(coords)
        local = _mask_from_coords(coords)
        key = _group_signature(local)
        groups.append((key, (int(r0), int(c0)), local))
    return groups


def _build_source_color_map(outside: np.ndarray, bg: int) -> dict[tuple[tuple[int, int, int, int], ...], dict[int, int]]:
    by_key: dict[tuple[tuple[int, int, int, int], ...], dict[int, int]] = {}
    for col in sorted(set(map(int, outside.flatten())) - {int(bg)}):
        color_mask = outside == int(col)
        for coords in components_mask(color_mask, connectivity=8):
            local = _mask_from_coords(coords)
            key = _group_signature(local)
            bucket = by_key.setdefault(key, {})
            bucket[int(col)] = int(bucket.get(int(col), 0)) + 1
    return by_key


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill grouped frame holes using outside sprites whose separated sub-block layouts match.

    Input structure:
    - Background is the most common color. One large monochrome frame sits in the grid.
    - Inside the frame bbox, each logical hole is an 8-neighbor cluster made from several
      smaller 4-connected pockets (typically diagonal 3x3 tiles).
    - Outside the frame, sprites use the same idea: each sprite color appears on a compact
      arrangement of multiple 4-connected mini-blocks.

    Rule:
    - Crop to the frame bbox and start from a solid frame-colored rectangle.
    - Group holes and outside sprites by their 8-neighbor cluster, then decompose each
      cluster into 4-connected subparts.
    - Match a hole-group to outside sprites by the canonical arrangement of those subparts
      under rotation/flip, and fill the whole hole-group with the most frequent matched color.

    Key insight:
    The outside clues no longer encode color with one connected silhouette. They encode color
    with a constellation of small blocks whose internal shape can vary, so the solver must
    reason about how the 4-connected pieces are arranged relative to each other.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    hole_groups = _extract_arrangement_groups(sub != int(frame_col))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    sprites_by_key = _build_source_color_map(outside, bg)

    out = np.full(sub.shape, int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in hole_groups:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)
    return out


# ---------------- Generator helpers ----------------

def _pattern_to_mask(points: list[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in points)
    max_c = max(c for _, c in points)
    out = np.zeros((TILE * (max_r + 1), TILE * (max_c + 1)), dtype=bool)
    for r, c in points:
        rr = TILE * int(r)
        cc = TILE * int(c)
        out[rr : rr + TILE, cc : cc + TILE] = HOLE_TILE
    return out


def _build_arrangement_library() -> list[np.ndarray]:
    raw_patterns = [
        [(0, 0), (1, 1)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 0), (1, 1), (2, 0)],
        [(0, 1), (1, 0), (1, 2)],
        [(0, 1), (1, 0), (1, 2), (2, 1)],
        [(0, 0), (1, 1), (2, 0), (2, 2)],
        [(0, 0), (1, 1), (2, 2), (3, 1)],
        [(0, 1), (1, 2), (2, 1), (1, 0)],
    ]
    seen: set[tuple[tuple[int, int, int, int], ...]] = set()
    lib: list[np.ndarray] = []
    for pts in raw_patterns:
        mask = _pattern_to_mask(pts)
        key = _group_signature(mask)
        if key not in seen:
            seen.add(key)
            lib.append(mask)
    return lib


ARRANGEMENT_LIBRARY = _build_arrangement_library()


def _choose_distinct_colors(n: int, bg: int, frame_col: int) -> list[int]:
    palette = [c for c in range(NUM_COLORS) if c != int(bg) and c != int(frame_col)]
    return random.sample(palette, n)


def _place_binary_patch(canvas: np.ndarray, mask: np.ndarray, *, border: int, padding: int) -> tuple[int, int] | None:
    patch = mask.astype(int)
    try:
        r, c = random_free_location(canvas, patch, transparent=0, border=border, padding=padding)
    except ValueError:
        return None
    paste(canvas, patch, int(r), int(c), transparent=0)
    return int(r), int(c)


def _build_hole_layout(th: int, tw: int):
    available = [
        mask for mask in ARRANGEMENT_LIBRARY
        if mask.shape[0] <= th - 2 and mask.shape[1] <= tw - 2 and int(np.count_nonzero(mask)) <= (th * tw) // 4
    ]
    if len(available) < ARRANGEMENT_COUNT_RANGE[0]:
        return None
    available.sort(key=lambda m: (int(m.shape[0] * m.shape[1]), int(np.count_nonzero(m))))
    kmax = min(ARRANGEMENT_COUNT_RANGE[1], len(available))
    n_types = random.randint(ARRANGEMENT_COUNT_RANGE[0], kmax)
    candidate_pool = available[: max(4, n_types + 1)]
    chosen = random.sample(candidate_pool, n_types)
    hole_canvas = np.zeros((th, tw), dtype=int)
    info = []

    for base_mask in chosen:
        inst_count = random.randint(*ARRANGEMENT_MULT_RANGE)
        placed = []
        orientations = _all_orientations(base_mask)
        for _ in range(inst_count):
            success = None
            for _attempt in range(PLACEMENT_RETRIES):
                mask = random.choice(orientations)
                pos = _place_binary_patch(hole_canvas, mask, border=1, padding=1)
                if pos is not None:
                    success = (pos[0], pos[1], mask.copy())
                    break
            if success is None:
                return None
            placed.append(success)
        info.append({"base_mask": base_mask.copy(), "instances": placed, "key": _group_signature(base_mask)})

    total_groups = sum(len(item["instances"]) for item in info)
    if total_groups < MIN_TOTAL_GROUPS:
        return None

    frame_density = 1.0 - float(np.count_nonzero(hole_canvas)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    if len(components_mask(hole_canvas.astype(bool), connectivity=8)) != total_groups:
        return None
    if sum(len(components_mask(_mask_from_coords(coords), connectivity=4)) for coords in components_mask(hole_canvas.astype(bool), connectivity=8)) <= total_groups:
        return None

    return hole_canvas.astype(bool), info


def _build_output(hole_mask: np.ndarray, info: list[dict], frame_col: int, bg: int):
    th, tw = hole_mask.shape
    output = np.full((th, tw), int(frame_col), dtype=int)
    colors = _choose_distinct_colors(len(info), bg, frame_col)
    shape_info = []
    for item, fill_color in zip(info, colors):
        instances = []
        for r, c, mask in item["instances"]:
            patch = np.full(mask.shape, int(fill_color), dtype=int)
            paste(output, patch, int(r), int(c), mask=mask)
            instances.append((int(r), int(c), mask.copy()))
        shape_info.append(
            {
                "key": item["key"],
                "base_mask": item["base_mask"].copy(),
                "fill_color": int(fill_color),
                "instances": instances,
            }
        )
    return output, shape_info


def _place_sprite_group(inp: np.ndarray, placement: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, int(r), int(c), transparent=bg)
    paste(inp, patch, int(r), int(c), transparent=bg)
    return True


def _materialize_sprite_mask(arrangement_mask: np.ndarray) -> np.ndarray:
    sprite = np.zeros_like(arrangement_mask, dtype=bool)
    for coords in components_mask(arrangement_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        assert int(r1 - r0 + 1) == TILE and int(c1 - c0 + 1) == TILE
        variant = random.choice(SOURCE_TILE_VARIANTS)
        oriented_variant = random.choice(_all_orientations(variant))
        paste(sprite, oriented_variant.astype(bool), int(r0), int(c0), transparent=False)
    return sprite


def _scatter_sprites(inp: np.ndarray, shape_info: list[dict], bg: int, frame_col: int) -> bool:
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != int(bg) and c != int(frame_col)]
    for item in shape_info:
        fill_color = int(item["fill_color"])
        arrangements = _all_orientations(item["base_mask"])
        n_win = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(0, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        loser_colors = random.sample(losers, n_other) if n_other else []

        for _ in range(n_win):
            sprite_mask = _materialize_sprite_mask(random.choice(arrangements))
            if not _place_sprite_group(inp, placement, sprite_mask, fill_color, bg):
                return False

        for loser in loser_colors:
            n_loser = random.randint(1, max(1, n_win - 1))
            for _ in range(n_loser):
                sprite_mask = _materialize_sprite_mask(random.choice(arrangements))
                if not _place_sprite_group(inp, placement, sprite_mask, int(loser), bg):
                    break
    return True


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    frame = max(comps, key=lambda comp: int(comp["area"]))
    frame_col = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])

    sub = crop_bbox(g, frame["bbox"])
    hole_groups = components_mask(sub != int(frame_col), connectivity=8)
    hole_parts = 0
    hole_split_groups = 0
    for coords in hole_groups:
        local = _mask_from_coords(coords)
        n_parts = len(components_mask(local, connectivity=4))
        hole_parts += n_parts
        if n_parts > 1:
            hole_split_groups += 1

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    src_groups = 0
    src_parts = 0
    src_split_groups = 0
    for col in sorted(set(map(int, outside.flatten())) - {int(bg)}):
        for coords in components_mask(outside == int(col), connectivity=8):
            local = _mask_from_coords(coords)
            n_parts = len(components_mask(local, connectivity=4))
            src_groups += 1
            src_parts += n_parts
            if n_parts > 1:
                src_split_groups += 1

    return {
        "hole_groups": int(len(hole_groups)),
        "hole_parts": int(hole_parts),
        "hole_split_groups": int(hole_split_groups),
        "src_groups": int(src_groups),
        "src_parts": int(src_parts),
        "src_split_groups": int(src_split_groups),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    hole_mask, info = layout

    output, shape_info = _build_output(hole_mask, info, frame_col, bg)

    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    frame_patch = np.where(hole_mask, int(bg), int(frame_col)).astype(int)
    try:
        fr = random.randint(1, GRID_SIZE - th - 1)
        fc = random.randint(1, GRID_SIZE - tw - 1)
    except ValueError:
        return None
    paste(inp, frame_patch, fr, fc)

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    metrics = _fusion_metrics(inp)
    if metrics["hole_split_groups"] != metrics["hole_groups"]:
        return None
    if metrics["src_split_groups"] < len(shape_info):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
