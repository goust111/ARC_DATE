from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


ALL_COLORS = tuple(range(10))
GEN_RETRIES = 500


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tile_from_periods(grid: np.ndarray, pr: int, pc: int) -> tuple[np.ndarray, int]:
    tile = np.zeros((pr, pc), dtype=int)
    matches = 0
    for rr in range(pr):
        for cc in range(pc):
            vals = list(map(int, grid[rr::pr, cc::pc].ravel()))
            color = Counter(vals).most_common(1)[0][0]
            tile[rr, cc] = int(color)
            matches += sum(v == color for v in vals)
    return tile, int(matches)


def _render_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    pr, pc = tile.shape
    rows = np.arange(h) % pr
    cols = np.arange(w) % pc
    return tile[rows[:, None], cols[None, :]]


def _infer_periodic_pattern(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    h, w = map(int, grid.shape)
    max_pr = max(1, h // 2)
    max_pc = max(1, w // 2)

    best_key = None
    best_tile = None
    for pr in range(1, max_pr + 1):
        for pc in range(1, max_pc + 1):
            tile, matches = _tile_from_periods(grid, pr, pc)
            key = (matches, -(pr * pc), -(pr + pc))
            if best_key is None or key > best_key:
                best_key = key
                best_tile = tile

    assert best_tile is not None
    predicted = _render_tile(best_tile, grid.shape)
    return best_tile, best_tile.shape, predicted


def _is_needle_triplet(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 3 or len(set(coords)) != 3:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    for r, c in coords:
        if not (r == r0 or r == r1 or c == c0 or c == c1):
            return False
    return True


def _find_needle_bbox(grid: np.ndarray, mismatch_mask: np.ndarray) -> tuple[int, int, int, int]:
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(mismatch_mask):
        by_color[int(grid[r, c])].append((int(r), int(c)))

    candidates = []
    for color, coords in by_color.items():
        if len(coords) == 3 and _is_needle_triplet(coords):
            candidates.append((color, _bbox(coords)))

    assert len(candidates) == 1, f"expected one needle color, got {candidates}"
    return candidates[0][1]


def _solve(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    tile, (_pr, _pc), predicted = _infer_periodic_pattern(g)
    mismatch_mask = g != predicted
    assert mismatch_mask.any(), "expected at least one corrupted cell"
    r0, c0, r1, c1 = _find_needle_bbox(g, mismatch_mask)
    out = predicted[r0 + 1 : r1, c0 + 1 : c1].copy()
    assert out.size > 0, "needle window must have a non-empty interior"
    return out, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the hidden needle window and repair its interior from the global periodic pattern.

    Input structure:
    - The whole grid is a repetition of a smaller multi-color tile along both rows and columns.
    - Some cells are corrupted stains whose colors do not belong to the underlying tile palette.
    - Exactly one stain color appears three times and those three cells all lie on the border of their shared bounding box; other stain colors are distractors.

    Rule:
    - Infer the repeated global pattern and mark every cell whose visible color disagrees with that periodic pattern as corrupted.
    - Among corrupted colors, find the unique one that occurs exactly three times and whose three cells sit on the boundary of one non-degenerate rectangle; use its bounding box as the hidden window.
    - Return the strict interior of that box, with every corrupted cell inside repaired back to the periodic color expected at that global phase.

    Key insight:
    - The three needle pixels do not draw a frame; they only identify which rectangle to crop, while the actual repair comes from matching each interior position to clean cells with the same row/column residues in the repeated background.
    """
    out, _bbox_value = _solve(grid)
    return out


def _fundamental_period_along_axis(tile: np.ndarray, axis: int) -> int:
    size = tile.shape[axis]
    for p in range(1, size):
        if np.array_equal(tile, np.roll(tile, -p, axis=axis)):
            return p
    return size


def _random_tile(pr: int, pc: int) -> np.ndarray | None:
    n_colors = random.randint(2, min(4, len(ALL_COLORS) - 4))
    palette = random.sample(ALL_COLORS, n_colors)
    tile = np.array(
        [[random.choice(palette) for _ in range(pc)] for _ in range(pr)],
        dtype=int,
    )
    if len(set(map(int, tile.ravel()))) < 2:
        return None
    if _fundamental_period_along_axis(tile, 0) != pr:
        return None
    if _fundamental_period_along_axis(tile, 1) != pc:
        return None
    return tile


def _pick_needle_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        return [(r0, c0), (r1, random.randint(c0 + 1, c1 - 1)), (random.randint(r0 + 1, r1 - 1), c1)]
    if corner == "tr":
        return [(r0, c1), (r1, random.randint(c0 + 1, c1 - 1)), (random.randint(r0 + 1, r1 - 1), c0)]
    if corner == "bl":
        return [(r1, c0), (r0, random.randint(c0 + 1, c1 - 1)), (random.randint(r0 + 1, r1 - 1), c1)]
    return [(r1, c1), (r0, random.randint(c0 + 1, c1 - 1)), (random.randint(r0 + 1, r1 - 1), c0)]


def _sample_positions(
    available: list[tuple[int, int]],
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    choices = [pos for pos in available if pos not in occupied]
    if len(choices) < count:
        return None
    return random.sample(choices, count)


def _pick_false_triplet(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    for _ in range(80):
        rm = random.randint(1, h - 2)
        cm = random.randint(1, w - 2)
        r0 = random.randint(0, rm - 1)
        r1 = random.randint(rm + 1, h - 1)
        c0 = random.randint(0, cm - 1)
        c1 = random.randint(cm + 1, w - 1)
        pts = [(r0, c0), (rm, cm), (r1, c1)]
        if any(pt in occupied for pt in pts):
            continue
        if _is_needle_triplet(pts):
            continue
        return pts
    return None


def _place_stains(
    clean: np.ndarray,
    bbox_value: tuple[int, int, int, int],
    stain_colors: list[int],
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]] | None:
    h, w = map(int, clean.shape)
    r0, c0, r1, c1 = bbox_value
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    exterior = [(r, c) for r in range(h) for c in range(w) if not (r0 < r < r1 and c0 < c < c1)]

    grid = clean.copy()
    occupied: set[tuple[int, int]] = set()
    placements: dict[int, list[tuple[int, int]]] = {}

    needle_color = stain_colors[0]
    needle_pts = _pick_needle_positions(r0, c0, r1, c1)
    placements[needle_color] = needle_pts
    occupied.update(needle_pts)

    needs_false_triplet = len(stain_colors) >= 3 and random.random() < 0.7
    next_idx = 1
    if needs_false_triplet:
        false_triplet = _pick_false_triplet(clean.shape, occupied)
        if false_triplet is None:
            return None
        placements[stain_colors[next_idx]] = false_triplet
        occupied.update(false_triplet)
        next_idx += 1

    mandatory_inside = random.randint(1, min(4, max(1, len(interior) // 4)))
    inside_pts = _sample_positions(interior, mandatory_inside, occupied)
    if inside_pts is None:
        return None

    mandatory_outside = random.randint(2, 5)
    outside_pts = _sample_positions(exterior, mandatory_outside, occupied | set(inside_pts))
    if outside_pts is None:
        return None

    filler_colors = stain_colors[next_idx:] or [stain_colors[1]]
    random.shuffle(inside_pts)
    random.shuffle(outside_pts)

    all_groups: list[tuple[int, list[tuple[int, int]]]] = []
    for pt in inside_pts:
        all_groups.append((random.choice(filler_colors), [pt]))
    for pt in outside_pts:
        all_groups.append((random.choice(filler_colors), [pt]))

    if random.random() < 0.6 and len(filler_colors) >= 1:
        bulk_color = random.choice(filler_colors)
        bulk_count = random.choice((2, 4, 5))
        bulk_pts = _sample_positions(
            [(r, c) for r in range(h) for c in range(w)],
            bulk_count,
            occupied | set(inside_pts) | set(outside_pts),
        )
        if bulk_pts is None:
            return None
        all_groups.append((bulk_color, bulk_pts))

    for color, pts in all_groups:
        placements.setdefault(int(color), []).extend(pts)

    for color, pts in placements.items():
        if len(pts) == 3 and color != needle_color and _is_needle_triplet(pts):
            return None
        for r, c in pts:
            grid[r, c] = int(color)

    if not any(r0 < r < r1 and c0 < c < c1 for pts in placements.values() for r, c in pts if int(grid[r, c]) != needle_color):
        return None

    return grid, placements


def _gen_attempt() -> np.ndarray | None:
    pr = random.randint(2, 5)
    pc = random.randint(2, 5)
    rep_r = random.randint(4, min(6, 30 // pr))
    rep_c = random.randint(4, min(6, 30 // pc))
    h = pr * rep_r
    w = pc * rep_c

    tile = _random_tile(pr, pc)
    if tile is None:
        return None
    clean = _render_tile(tile, (h, w))

    max_outer_h = min(h - 2, 12)
    max_outer_w = min(w - 2, 12)
    if max_outer_h < 5 or max_outer_w < 5:
        return None
    outer_h = random.randint(5, max_outer_h)
    outer_w = random.randint(5, max_outer_w)
    r0 = random.randint(1, h - outer_h - 1)
    c0 = random.randint(1, w - outer_w - 1)
    bbox_value = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)

    pattern_colors = set(map(int, tile.ravel()))
    stain_pool = [color for color in ALL_COLORS if color not in pattern_colors]
    if len(stain_pool) < 3:
        return None
    random.shuffle(stain_pool)
    stain_colors = stain_pool[: random.randint(3, min(5, len(stain_pool)))]

    placed = _place_stains(clean, bbox_value, stain_colors)
    if placed is None:
        return None
    grid, placements = placed

    predicted_output = clean[r0 + 1 : r0 + outer_h - 1, c0 + 1 : c0 + outer_w - 1]
    try:
        actual_output, found_bbox = _solve(grid)
    except AssertionError:
        return None
    if found_bbox != bbox_value:
        return None
    if not np.array_equal(actual_output, predicted_output):
        return None

    _tile, _periods, predicted = _infer_periodic_pattern(grid)
    mismatch_coords = {tuple(map(int, p)) for p in np.argwhere(grid != predicted)}
    stain_coords = {coord for pts in placements.values() for coord in pts}
    if mismatch_coords != stain_coords:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
