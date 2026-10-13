from __future__ import annotations

from itertools import combinations
import random

import numpy as np


ARC_COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 600

# The profile cycles make both separator orientations and both panel orders
# appear in any short run of generated examples.
_GENERATION_PROFILE = 0
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _colors(grid: np.ndarray) -> list[int]:
    """Return present colors; ids are labels only, never an ordering signal."""
    return list(map(int, np.unique(grid)))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bbox in the common arc_utils (row, column) convention."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _split_options(grid: np.ndarray):
    """Yield every visible, one-cell, full-line split and its two regions."""
    h, w = grid.shape
    for r in range(1, h - 1):
        color = int(grid[r, 0])
        if bool(np.all(grid[r, :] == color)) and int(np.count_nonzero(grid == color)) == w:
            yield "horizontal", r, color, grid[:r, :], grid[r + 1 :, :]
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if bool(np.all(grid[:, c] == color)) and int(np.count_nonzero(grid == color)) == h:
            yield "vertical", c, color, grid[:, :c], grid[:, c + 1 :]


def _cross_candidates(region: np.ndarray):
    """Find all two-color masks that exactly fill active-rows x active-cols."""
    h, w = region.shape
    for pair_tuple in combinations(_colors(region), 2):
        pair = frozenset(pair_tuple)
        mask = np.isin(region, pair_tuple)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        rh, cw = int(rows.sum()), int(cols.sum())

        # Both deletions must matter; this also rejects a whole-region panel.
        if not (0 < rh < h and 0 < cw < w):
            continue
        if int(mask.sum()) != rh * cw:
            continue
        template = region[np.ix_(rows, cols)]
        if frozenset(_colors(template)) != pair:
            continue
        yield {
            "pair": pair,
            "rows": rows,
            "cols": cols,
            "template": template.copy(),
        }


def _copy_candidates(region: np.ndarray):
    """Extract a rectangular two-color block from a uniform surrounding field."""
    for surround in _colors(region):
        box = _bbox(region != surround)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r0 == 0 or c0 == 0 or r1 == region.shape[0] - 1 or c1 == region.shape[1] - 1:
            continue

        outside = np.ones(region.shape, dtype=bool)
        outside[r0 : r1 + 1, c0 : c1 + 1] = False
        if not bool(np.all(region[outside] == surround)):
            continue
        tile = region[r0 : r1 + 1, c0 : c1 + 1]
        tile_colors = frozenset(_colors(tile))
        if len(tile_colors) != 2 or surround in tile_colors:
            continue
        yield {
            "surround": int(surround),
            "bbox": box,
            "tile": tile.copy(),
            "colors": tile_colors,
        }


def _solution_candidates(grid: np.ndarray):
    """Enumerate complete interpretations, including the one-way damage test."""
    present = set(_colors(grid))
    missing = [color for color in ARC_COLORS if color not in present]
    if len(missing) != 1:
        return []
    marker = missing[0]

    solutions = []
    for orientation, line, separator, first, second in _split_options(grid):
        for master_side, (region_a, region_b) in enumerate(((first, second), (second, first))):
            crosses = list(_cross_candidates(region_a))
            copies = list(_copy_candidates(region_b))
            for cross in crosses:
                master = cross["template"]
                for copy_info in copies:
                    copy = copy_info["tile"]
                    if copy.shape != master.shape or copy_info["colors"] != cross["pair"]:
                        continue

                    different = master != copy
                    if not bool(different.any()):
                        continue
                    master_changed = set(map(int, master[different]))
                    copy_changed = set(map(int, copy[different]))
                    if len(master_changed) != 1 or len(copy_changed) != 1:
                        continue
                    foreground = next(iter(master_changed))
                    background = next(iter(copy_changed))
                    if foreground == background:
                        continue

                    # Every disagreement must be foreground -> copy background;
                    # all other cells already agree.  This uniquely determines
                    # which of the two template colors is the copy background.
                    damage = (master == foreground) & (copy == background)
                    if not np.array_equal(different, damage):
                        continue

                    output = master.copy()
                    output[damage] = marker
                    solutions.append(
                        {
                            "orientation": orientation,
                            "line": int(line),
                            "separator": int(separator),
                            "master_side": int(master_side),
                            "region_a": region_a,
                            "region_b": region_b,
                            "cross": cross,
                            "copy_info": copy_info,
                            "master": master,
                            "copy": copy,
                            "background": int(background),
                            "foreground": int(foreground),
                            "damage": damage,
                            "marker": int(marker),
                            "output": output,
                        }
                    )
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract a hidden bicolor master and mark the missing pixels in its damaged copy.

    Input structure:
    - One full separator line divides the grid into two regions.
    - In one region, exactly one color pair fills the Cartesian product of all
      rows and columns touched by that pair; other colors are clutter.
    - The other region contains one rectangular two-color damaged copy on a
      uniform surround.  Exactly one of the ten ARC colors is absent.

    Rule:
    - Keep only the active rows and active columns of the unique Cartesian-product
      color pair to form the master, then compare it cell by cell with the copy.
    - Preserve agreeing master cells.  Wherever the copy has its background but
      the master has foreground, replace the master pixel with the absent color.

    Key insight:
    - Neither color ids nor a bounding box identify the master: its scattered
      active-row/active-column product must be compressed before the one-way
      foreground loss in the copy reveals the repair marks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    solutions = _solution_candidates(g)
    assert len(solutions) == 1, "the visible clues must determine exactly one repair"
    return solutions[0]["output"].copy()


def _connected_foreground(h: int, w: int) -> np.ndarray | None:
    """Grow a compact, irregular 4-connected foreground with ample background."""
    target = random.randint(max(7, (h * w) // 4), max(8, (h * w) // 2))
    target = min(target, h * w - 6)
    if target < 6:
        return None

    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            r, c = int(rr), int(cc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        mask[random.choice(tuple(frontier))] = True

    coords = np.argwhere(mask)
    if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
        return None
    return mask


def _active_indices(n_active: int, total: int) -> np.ndarray:
    """Choose active coordinates with outer margins and forced internal gaps."""
    inactive_count = total - n_active
    assert inactive_count >= 3
    internal = list(range(1, total - 1))
    for _ in range(100):
        inactive = {0, total - 1}
        inactive.update(random.sample(internal, inactive_count - 2))
        active = np.array([x for x in range(total) if x not in inactive], dtype=int)
        if active.size >= 2 and bool(np.any(np.diff(active) > 1)):
            return active
    raise RuntimeError("failed to choose non-contiguous active coordinates")


def _choose_damage(fg_mask: np.ndarray) -> np.ndarray | None:
    """Remove several foreground cells spanning multiple rows and columns."""
    coords = [tuple(map(int, p)) for p in np.argwhere(fg_mask)]
    max_damage = min(len(coords) - 3, max(3, len(coords) // 3))
    if max_damage < 3:
        return None
    count = random.randint(3, max_damage)
    for _ in range(100):
        picked = random.sample(coords, count)
        if len({r for r, _ in picked}) >= 2 and len({c for _, c in picked}) >= 2:
            damage = np.zeros(fg_mask.shape, dtype=bool)
            for r, c in picked:
                damage[r, c] = True
            return damage
    return None


def _make_vertical_case(profile: int):
    """Construct an output-first vertical split; callers may transpose it."""
    h, w = random.randint(4, 7), random.randint(4, 7)
    region_h = h + random.randint(4, 6)
    region_a_w = w + random.randint(4, 6)
    region_b_w = w + random.randint(4, 6)

    palette = list(ARC_COLORS)
    marker = random.choice(palette)
    available = [c for c in palette if c != marker]
    separator = random.choice(available)
    available.remove(separator)
    color_a, color_b = random.sample(available, 2)
    filler_colors = [c for c in available if c not in (color_a, color_b)]
    assert len(filler_colors) == 6
    panel_color = random.choice(filler_colors)

    fg_mask = _connected_foreground(h, w)
    if fg_mask is None:
        return None
    background, foreground = random.sample((color_a, color_b), 2)
    master = np.full((h, w), background, dtype=int)
    master[fg_mask] = foreground
    damage = _choose_damage(fg_mask)
    if damage is None:
        return None
    copy = master.copy()
    copy[damage] = background
    expected = master.copy()
    expected[damage] = marker

    active_rows = _active_indices(h, region_h)
    active_cols = _active_indices(w, region_a_w)
    region_a = np.empty((region_h, region_a_w), dtype=int)
    outside = np.ones(region_a.shape, dtype=bool)
    outside[np.ix_(active_rows, active_cols)] = False
    outside_coords = [tuple(map(int, p)) for p in np.argwhere(outside)]
    for r, c in outside_coords:
        region_a[r, c] = random.choice(filler_colors)
    random.shuffle(outside_coords)
    for color, (r, c) in zip(filler_colors, outside_coords):
        region_a[r, c] = color
    region_a[np.ix_(active_rows, active_cols)] = master

    region_b = np.full((region_h, region_b_w), panel_color, dtype=int)
    top = random.randint(1, region_h - h - 1)
    left = random.randint(1, region_b_w - w - 1)
    region_b[top : top + h, left : left + w] = copy

    # Profile parity on this branch controls which side contains the master.
    master_first = ((profile // 2) % 2 == 0)
    if master_first:
        grid = np.concatenate(
            (region_a, np.full((region_h, 1), separator, dtype=int), region_b), axis=1
        )
    else:
        grid = np.concatenate(
            (region_b, np.full((region_h, 1), separator, dtype=int), region_a), axis=1
        )
    return grid, expected, marker


def _has_internal_gap(active: np.ndarray) -> bool:
    ids = np.flatnonzero(active)
    return bool(ids.size >= 2 and np.any(np.diff(ids) > 1))


def _semantic_validation(grid: np.ndarray, expected: np.ndarray, marker: int) -> bool:
    """Validate stated semantics and anti-shortcut constraints, not just closure."""
    if grid.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if set(_colors(grid)) != set(ARC_COLORS) - {marker}:
        return False

    # The divider itself and the full interpretation must both be unique.
    splits = list(_split_options(grid))
    if len(splits) != 1:
        return False
    solutions = _solution_candidates(grid)
    if len(solutions) != 1:
        return False
    solution = solutions[0]
    if solution["marker"] != marker or not np.array_equal(solution["output"], expected):
        return False

    # Independently check that no other pair in region A has the product mask.
    crosses = list(_cross_candidates(solution["region_a"]))
    if len(crosses) != 1:
        return False
    rows, cols = crosses[0]["rows"], crosses[0]["cols"]
    if rows[0] or rows[-1] or cols[0] or cols[-1]:
        return False
    if not _has_internal_gap(rows) or not _has_internal_gap(cols):
        return False

    # The block extraction, background direction, and all marked cells are unique.
    copies = list(_copy_candidates(solution["region_b"]))
    if len(copies) != 1:
        return False
    damage = solution["damage"]
    coords = np.argwhere(damage)
    if int(damage.sum()) < 3:
        return False
    if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
        return False

    master, copy = solution["master"], solution["copy"]
    if np.array_equal(expected, master) or np.array_equal(expected, copy):
        return False
    if not bool((~damage & (master == solution["foreground"])).any()):
        return False
    if not bool((master == solution["background"]).any()):
        return False

    try:
        solved = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    return np.array_equal(solved, expected)


def _gen_attempt(profile: int) -> tuple[np.ndarray, np.ndarray] | None:
    made = _make_vertical_case(profile)
    if made is None:
        return None
    grid, expected, marker = made

    # Even profiles are vertical; odd profiles transpose every coordinate and
    # therefore genuinely exercise the horizontal separator/reference frame.
    if profile % 2 == 1:
        grid = grid.T.copy()
        expected = expected.T.copy()

    if not _semantic_validation(grid, expected, marker):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate a diverse, uniquely solvable hidden-master repair instance."""
    global _GENERATION_PROFILE
    profile = _GENERATION_PROFILE
    for _ in range(GENERATION_ATTEMPTS):
        made = _gen_attempt(profile)
        if made is None:
            continue
        grid, expected = made
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        _GENERATION_PROFILE += 1
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
