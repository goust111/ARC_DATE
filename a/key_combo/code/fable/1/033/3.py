from __future__ import annotations

import random
from typing import Iterable

import numpy as np


MASK = 6
ALL_COLORS = tuple(range(10))

# --------------- solver helpers ---------------


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_solid_2x2(coords: np.ndarray) -> bool:
    if coords.shape != (4, 2):
        return False
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return False
    return {(int(r), int(c)) for r, c in coords} == {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    }


def _orbit_positions(
    r: int, c: int, sr: int, sc: int, h: int, w: int
) -> tuple[tuple[int, int], ...]:
    """All in-grid quarter-turn images of (r,c) around center (sr/2, sc/2)."""
    dr2 = 2 * int(r) - int(sr)
    dc2 = 2 * int(c) - int(sc)
    rels = (
        (dr2, dc2),
        (dc2, -dr2),
        (-dr2, -dc2),
        (-dc2, dr2),
    )
    pts = []
    for rr2, cc2 in rels:
        rr_num = sr + rr2
        cc_num = sc + cc2
        assert rr_num % 2 == 0 and cc_num % 2 == 0
        rr = rr_num // 2
        cc = cc_num // 2
        if 0 <= rr < h and 0 <= cc < w:
            pts.append((int(rr), int(cc)))
    return tuple(sorted(set(pts)))


def _find_candidate_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blocks = []
    for color in np.unique(g):
        color = int(color)
        if color == MASK:
            continue
        coords = np.argwhere(g == color)
        if coords.shape[0] != 4 or not _is_solid_2x2(coords):
            continue
        r0, c0, _r1, _c1 = _bbox_from_coords(coords)
        blocks.append(
            {
                "color": color,
                "bbox": (r0, c0, r0 + 1, c0 + 1),
                "sr": 2 * r0 + 1,
                "sc": 2 * c0 + 1,
            }
        )
    blocks.sort(key=lambda block: block["bbox"])
    assert len(blocks) == 2, f"expected exactly two 2x2 marker blocks, got {len(blocks)}"
    return blocks


def _visible_pattern_mask(grid: np.ndarray, marker_colors: Iterable[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    visible = g != MASK
    for color in marker_colors:
        visible &= g != int(color)
    return visible


def _center_consistent(
    grid: np.ndarray, sr: int, sc: int, marker_colors: set[int]
) -> tuple[bool, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visible = _visible_pattern_mask(g, marker_colors)
    seen = set()
    evidence = 0
    for r, c in np.argwhere(visible):
        key = _orbit_positions(int(r), int(c), sr, sc, h, w)
        if key in seen:
            continue
        seen.add(key)
        vals = [int(g[rr, cc]) for rr, cc in key if visible[rr, cc]]
        if len(set(vals)) > 1:
            return False, evidence
        if len(vals) >= 2:
            evidence += len(vals) - 1
    return evidence > 0, evidence


def _true_center(grid: np.ndarray, blocks: list[dict]) -> tuple[int, int, set[int]]:
    marker_colors = {int(block["color"]) for block in blocks}
    valid = []
    for block in blocks:
        ok, evidence = _center_consistent(grid, int(block["sr"]), int(block["sc"]), marker_colors)
        if ok:
            valid.append((int(block["sr"]), int(block["sc"]), evidence))
    assert len(valid) == 1, f"expected one self-consistent rotation center, got {valid}"
    sr, sc, _evidence = valid[0]
    return sr, sc, marker_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify which 2x2 marker block is the true quarter-turn center and fill each pink mask from its rotation orbit.

    Input structure:
    - The grid contains a colored pattern plus some pink masked cells.
    - Exactly two colors each form one solid 2x2 marker block; one block marks the true
      center of a 90-degree rotational symmetry and the other is a decoy.
    - The visible pattern pixels exclude the pink masks and the two marker blocks.

    Rule:
    - Test both 2x2 block centers by rotating every visible pattern pixel by 90, 180,
      and 270 degrees and checking that all in-grid visible orbit mates agree in color.
    - Select the unique center whose visible pattern orbits are globally consistent.
    - For every pink cell, collect the in-grid visible pattern cells in its orbit around
      that center and replace the pink cell with their unique shared color.

    Key insight:
    - The marker blocks are only candidate coordinates; the true center is the one whose
      quarter-turn orbits make all visible non-marker pixels agree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == MASK), "input must contain pink masked pixels"

    blocks = _find_candidate_blocks(g)
    sr, sc, marker_colors = _true_center(g, blocks)

    out = g.copy()
    h, w = g.shape
    for r, c in np.argwhere(g == MASK):
        vals = {
            int(g[rr, cc])
            for rr, cc in _orbit_positions(int(r), int(c), sr, sc, h, w)
            if g[rr, cc] != MASK and int(g[rr, cc]) not in marker_colors
        }
        assert len(vals) == 1, f"masked cell {(int(r), int(c))} must have one visible orbit color"
        out[int(r), int(c)] = vals.pop()
    return out


# --------------- generator ---------------

# Grid, palette, and retry bounds.
GEN_ATTEMPTS = 500
GEN_H_MIN, GEN_H_MAX = 12, 22
GEN_W_MIN, GEN_W_MAX = 12, 22
GEN_PATTERN_COLOR_MIN, GEN_PATTERN_COLOR_MAX = 3, 5

# Mask and layout density.
GEN_MASKED_ORBITS_MIN, GEN_MASKED_ORBITS_MAX = 4, 10
GEN_MAX_MASKS_PER_ORBIT = 2
GEN_BLOCK_GAP = 1

# Seed-pattern modes.
GEN_MODES = ("blocks", "bands", "checker", "mixed")


def _boxes_too_close(a: tuple[int, int], b: tuple[int, int], gap: int = GEN_BLOCK_GAP) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 1, ac0 + 1
    br1, bc1 = br0 + 1, bc0 + 1
    rows_overlap = not (ar1 + gap < br0 or br1 + gap < ar0)
    cols_overlap = not (ac1 + gap < bc0 or bc1 + gap < ac0)
    return rows_overlap and cols_overlap


def _paint_block(grid: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)


def _choose_centers(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    options = [
        (r, c)
        for r in range(1, h - 2)
        for c in range(1, w - 2)
        if (2 * r + 1, 2 * c + 1) != (h - 1, w - 1)
    ]
    random.shuffle(options)
    for true_top in options:
        decoys = [
            other
            for other in options
            if other != true_top and not _boxes_too_close(true_top, other)
        ]
        if decoys:
            return true_top, random.choice(decoys)
    raise RuntimeError("failed to place marker blocks")


def _make_seed_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    mode = random.choice(GEN_MODES)
    palette_arr = np.asarray(palette, dtype=int)

    if mode == "checker":
        bh = random.randint(1, min(4, h))
        bw = random.randint(1, min(4, w))
        phase = random.randrange(len(palette))
        rows, cols = np.indices((h, w))
        seed = palette_arr[((rows // bh) + (cols // bw) + phase) % len(palette)]
        if random.random() < 0.5:
            seed = np.fliplr(seed)
    else:
        bh = random.randint(2, min(5, h))
        bw = random.randint(2, min(5, w))
        coarse_h = (h + bh - 1) // bh
        coarse_w = (w + bw - 1) // bw
        coarse = np.random.choice(palette_arr, size=(coarse_h, coarse_w))
        if mode in ("bands", "mixed") and random.random() < 0.8:
            for r in range(coarse_h):
                coarse[r, :] = random.choice(palette)
        if mode in ("bands", "mixed") and random.random() < 0.8:
            for c in range(coarse_w):
                if random.random() < 0.7:
                    coarse[:, c] = random.choice(palette)
        seed = np.repeat(np.repeat(coarse, bh, axis=0), bw, axis=1)[:h, :w]

        for _ in range(random.randint(2, 6)):
            rh = random.randint(1, max(2, h // 2))
            rw = random.randint(1, max(2, w // 2))
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            seed[r0 : r0 + rh, c0 : c0 + rw] = random.choice(palette)

        if mode == "mixed":
            band = random.randint(1, 3)
            phase = random.randrange(len(palette))
            rows, cols = np.indices((h, w))
            overlay = palette_arr[((rows + cols) // band + phase) % len(palette)]
            overlay_mask = np.random.random((h, w)) < 0.2
            seed[overlay_mask] = overlay[overlay_mask]

    # Force every palette color to appear in the seed at least once.
    for color in palette:
        rr = random.randrange(h)
        cc = random.randrange(w)
        seed[rr, cc] = color
    return seed.astype(int)


def _build_symmetric_pattern(
    h: int,
    w: int,
    sr: int,
    sc: int,
    reserved: set[tuple[int, int]],
    palette: list[int],
) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]]:
    seed = _make_seed_pattern(h, w, palette)
    pattern = np.zeros((h, w), dtype=int)
    orbit_color: dict[tuple[tuple[int, int], ...], int] = {}
    orbit_members: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}

    for r in range(h):
        for c in range(w):
            if (r, c) in reserved:
                continue
            key = _orbit_positions(r, c, sr, sc, h, w)
            if key not in orbit_color:
                members = tuple(sorted((rr, cc) for rr, cc in key if (rr, cc) not in reserved))
                assert members
                orbit_members[key] = members
                rep_r, rep_c = min(members)
                orbit_color[key] = int(seed[rep_r, rep_c])
            pattern[r, c] = orbit_color[key]
    return pattern, list(orbit_members.values())


def _mask_cells(
    solved: np.ndarray, orbit_groups: list[tuple[tuple[int, int], ...]]
) -> np.ndarray | None:
    inp = solved.copy()
    eligible = [members for members in orbit_groups if len(members) >= 2]
    if len(eligible) < GEN_MASKED_ORBITS_MIN:
        return None
    random.shuffle(eligible)
    orbit_count = random.randint(
        GEN_MASKED_ORBITS_MIN, min(GEN_MASKED_ORBITS_MAX, len(eligible))
    )
    masked = 0
    for members in eligible[:orbit_count]:
        max_hide = min(GEN_MAX_MASKS_PER_ORBIT, len(members) - 1)
        hide_count = random.randint(1, max_hide)
        for r, c in random.sample(list(members), hide_count):
            inp[r, c] = MASK
            masked += 1
    if masked < GEN_MASKED_ORBITS_MIN:
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_H_MIN, GEN_H_MAX)
    w = random.randint(GEN_W_MIN, GEN_W_MAX)

    color_pool = [color for color in ALL_COLORS if color != MASK]
    random.shuffle(color_pool)
    pattern_color_count = random.randint(GEN_PATTERN_COLOR_MIN, GEN_PATTERN_COLOR_MAX)
    palette = color_pool[:pattern_color_count]
    true_color = color_pool[pattern_color_count]
    decoy_color = color_pool[pattern_color_count + 1]

    true_top, decoy_top = _choose_centers(h, w)
    sr, sc = 2 * true_top[0] + 1, 2 * true_top[1] + 1

    reserved = {
        (r, c)
        for top in (true_top, decoy_top)
        for r in range(top[0], top[0] + 2)
        for c in range(top[1], top[1] + 2)
    }
    pattern, orbit_groups = _build_symmetric_pattern(h, w, sr, sc, reserved, palette)

    solved = pattern.copy()
    _paint_block(solved, true_top, true_color)
    _paint_block(solved, decoy_top, decoy_color)

    inp = _mask_cells(solved, orbit_groups)
    if inp is None or not np.any(inp == MASK):
        return None

    # Solver-facing validity checks.
    try:
        blocks = _find_candidate_blocks(inp)
        got_sr, got_sc, _marker_colors = _true_center(inp, blocks)
        if (got_sr, got_sc) != (sr, sc):
            return None
        if not np.array_equal(main(inp), solved):
            return None
    except AssertionError:
        return None

    # Keep the scene readable and non-trivial.
    visible_pattern = inp != MASK
    visible_pattern &= inp != true_color
    visible_pattern &= inp != decoy_color
    if np.unique(solved[visible_pattern]).size < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
