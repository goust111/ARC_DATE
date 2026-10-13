from __future__ import annotations

import itertools
import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

REGION_POSITIONS = ("tl", "tr", "bl", "br")
PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _rotate_cw(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, -int(turns))


def _split_regions(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    row_divs = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_divs = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(row_divs) == 1 and len(col_divs) == 1, "expected one grey divider row and one grey divider column"

    div_r = row_divs[0]
    div_c = col_divs[0]
    side = div_r
    assert div_c == side
    assert grid.shape == (2 * side + 1, 2 * side + 1)

    regions = [
        grid[:div_r, :div_c].copy(),
        grid[:div_r, div_c + 1 :].copy(),
        grid[div_r + 1 :, :div_c].copy(),
        grid[div_r + 1 :, div_c + 1 :].copy(),
    ]
    assert all(region.shape == (side, side) for region in regions)
    return side, regions


def _consistent_assignments(regions: list[np.ndarray]) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    side = regions[0].shape[0]
    consistent: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for perm in itertools.permutations(range(4)):
        merged = np.zeros((side, side), dtype=int)
        known = np.zeros((side, side), dtype=bool)
        ok = True
        for region, cw_turns in zip(regions, perm):
            canonical = np.rot90(region, int(cw_turns))
            visible = canonical != BLACK
            if np.any(known & visible & (merged != canonical)):
                ok = False
                break
            merged[visible] = canonical[visible]
            known |= visible
        if ok:
            consistent.append((tuple(int(v) for v in perm), merged))
    return consistent


def _render_from_solution(grid: np.ndarray, regions: list[np.ndarray], assignment: tuple[int, int, int, int], template: np.ndarray) -> np.ndarray:
    side = template.shape[0]
    out = grid.copy()
    slices = [
        (slice(0, side), slice(0, side)),
        (slice(0, side), slice(side + 1, 2 * side + 1)),
        (slice(side + 1, 2 * side + 1), slice(0, side)),
        (slice(side + 1, 2 * side + 1), slice(side + 1, 2 * side + 1)),
    ]

    for idx, cw_turns in enumerate(assignment):
        full_region = _rotate_cw(template, cw_turns)
        current = regions[idx]
        filled = current.copy()
        missing = (current == BLACK) & (full_region != BLACK)
        filled[missing] = YELLOW
        out[slices[idx]] = filled

    return out


def _solve_regions(grid: np.ndarray, regions: list[np.ndarray]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    consistent = _consistent_assignments(regions)
    assert consistent, "expected at least one consistent rotation assignment"

    rendered = [_render_from_solution(grid, regions, assignment, template) for assignment, template in consistent]
    first = rendered[0]
    assert all(np.array_equal(candidate, first) for candidate in rendered[1:]), "consistent assignments must imply one unique output"
    return consistent[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared template behind four rotated incomplete regions and mark each recovered missing pixel in yellow.

    Input structure:
    - A single gray row and a single gray column split the grid into four equal square regions.
    - Each region contains the same red-and-blue template after one of the four 90-degree rotations, on a black background.
    - Different parts of each rotated copy are missing and shown as black.

    Rule:
    - Try all 24 ways to assign the four rotations {0, 90, 180, 270} to the four regions.
    - Keep the globally consistent assignments whose visible non-black pixels become mutually compatible after inverse rotation into one shared orientation.
    - Merge those inverse-rotated visible pixels to reconstruct the full template, then rotate that template back into each region; all valid assignments must imply the same final filled grid.
    - Wherever a region currently has black but the rotated full template has a non-black pixel, fill that cell with yellow; keep all existing non-black pixels and gray dividers unchanged.

    Key insight:
    - The regions cannot be solved independently: rotation identity and missing content must be inferred together by cross-checking all four partial views in one shared coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    side, regions = _split_regions(g)
    assignment, template = _solve_regions(g, regions)
    return _render_from_solution(g, regions, assignment, template)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _grow_connected_mask(side: int, count: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        start = (rng.randrange(side), rng.randrange(side))
        cells = {start}
        while len(cells) < count:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != count:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) == min(rows) or max(cols) == min(cols):
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _is_connected(mask):
            return mask
    raise RuntimeError("failed to create connected template mask")


def _color_template(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = [tuple(int(v) for v in rc) for rc in np.argwhere(mask)]
    total = len(coords)
    blue_target = rng.randint(max(3, total // 3), min(total - 3, (2 * total) // 3))

    blue = {rng.choice(coords)}
    while len(blue) < blue_target:
        frontier = []
        for r, c in blue:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in blue:
                    frontier.append((nr, nc))
        if frontier:
            blue.add(rng.choice(frontier))
        else:
            remaining = [cell for cell in coords if cell not in blue]
            blue.add(rng.choice(remaining))

    template = np.zeros(mask.shape, dtype=int)
    template[mask] = RED
    for r, c in blue:
        template[r, c] = BLUE
    return template


def _template_has_four_distinct_rotations(template: np.ndarray) -> bool:
    rotations = [_rotate_cw(template, k) for k in range(4)]
    return len({rot.tobytes() for rot in rotations}) == 4


def _make_template(side: int, rng: random.Random) -> np.ndarray:
    min_count = 12
    max_count = min(16, int(math.floor(side * side * 0.58)))
    for _ in range(300):
        count = rng.randint(min_count, max_count)
        mask = _grow_connected_mask(side, count, rng)
        template = _color_template(mask, rng)
        if not _template_has_four_distinct_rotations(template):
            continue
        if len(set(int(v) for v in template[mask])) != 2:
            continue
        return template
    raise RuntimeError("failed to create asymmetric template")


def _make_visibility_masks(template: np.ndarray, rng: random.Random) -> list[np.ndarray]:
    coords = [tuple(int(v) for v in rc) for rc in np.argwhere(template != BLACK)]
    count = len(coords)
    indices = list(range(count))
    rng.shuffle(indices)

    cap = count - math.ceil(count * 0.30)
    min_visible = max(5, math.ceil(count * 0.38))
    assert count >= 10

    visible_sets = [set() for _ in range(4)]
    cursor = 0
    for region in range(4):
        visible_sets[region].add(indices[cursor])
        cursor += 1
    for a, b in PAIR_ORDER:
        idx = indices[cursor]
        cursor += 1
        visible_sets[a].add(idx)
        visible_sets[b].add(idx)

    targets = [rng.randint(min_visible, cap) for _ in range(4)]
    counts = [len(s) for s in visible_sets]
    remaining = indices[cursor:]
    rng.shuffle(remaining)

    for idx in remaining:
        options = []
        for subset_size in (2, 3, 4):
            for subset in itertools.combinations(range(4), subset_size):
                if all(counts[region] < cap for region in subset):
                    score = sum(max(0, targets[region] - counts[region]) for region in subset)
                    options.append((score, rng.random(), subset))
        if not options:
            return []
        best_score = max(score for score, _tie, _subset in options)
        best_subsets = [subset for score, _tie, subset in options if score == best_score]
        chosen = rng.choice(best_subsets)
        for region in chosen:
            visible_sets[region].add(idx)
            counts[region] += 1

    for region in range(4):
        while counts[region] < min_visible:
            candidates = [
                idx
                for idx in range(count)
                if idx not in visible_sets[region]
                and counts[region] < cap
            ]
            if not candidates:
                return []
            idx = rng.choice(candidates)
            visible_sets[region].add(idx)
            counts[region] += 1

    masks = [np.zeros(template.shape, dtype=bool) for _ in range(4)]
    for region in range(4):
        for idx in visible_sets[region]:
            r, c = coords[idx]
            masks[region][r, c] = True

    return masks


def _build_input(template: np.ndarray, visibility_masks: list[np.ndarray], assignment: tuple[int, int, int, int]) -> np.ndarray:
    side = template.shape[0]
    grid = np.zeros((2 * side + 1, 2 * side + 1), dtype=int)
    grid[side, :] = GREY
    grid[:, side] = GREY

    slices = [
        (slice(0, side), slice(0, side)),
        (slice(0, side), slice(side + 1, 2 * side + 1)),
        (slice(side + 1, 2 * side + 1), slice(0, side)),
        (slice(side + 1, 2 * side + 1), slice(side + 1, 2 * side + 1)),
    ]

    for idx, cw_turns in enumerate(assignment):
        partial = np.where(visibility_masks[idx], template, BLACK)
        grid[slices[idx]] = _rotate_cw(partial, cw_turns)

    return grid


def _expected_output(template: np.ndarray, inp: np.ndarray, assignment: tuple[int, int, int, int]) -> np.ndarray:
    _side, regions = _split_regions(inp)
    return _render_from_solution(inp, regions, assignment, template)


def _visibility_stats(template: np.ndarray, visibility_masks: list[np.ndarray]) -> list[int]:
    template_cells = template != BLACK
    return [int(np.count_nonzero(mask & template_cells)) for mask in visibility_masks]


def _validate_example(template: np.ndarray, visibility_masks: list[np.ndarray], assignment: tuple[int, int, int, int], inp: np.ndarray) -> bool:
    template_cells = template != BLACK
    total = int(np.count_nonzero(template_cells))
    min_missing = math.ceil(total * 0.30)

    counts = _visibility_stats(template, visibility_masks)
    if any(total - count < min_missing for count in counts):
        return False

    union = np.zeros(template.shape, dtype=bool)
    for mask in visibility_masks:
        union |= mask
    if not np.array_equal(union, template_cells):
        return False

    for region in range(4):
        exclusive = visibility_masks[region] & ~np.logical_or.reduce(
            [visibility_masks[other] for other in range(4) if other != region]
        )
        if not np.any(exclusive):
            return False

    for a, b in PAIR_ORDER:
        if not np.any(visibility_masks[a] & visibility_masks[b]):
            return False

    consistent = _consistent_assignments(_split_regions(inp)[1])
    if len(consistent) != 4:
        return False

    rendered = [_render_from_solution(inp, _split_regions(inp)[1], found_assignment, merged) for found_assignment, merged in consistent]
    if not all(np.array_equal(rendered[0], candidate) for candidate in rendered[1:]):
        return False
    if not any(tuple(found_assignment) == tuple(assignment) and np.array_equal(merged, template) for found_assignment, merged in consistent):
        return False

    out = main(inp)
    expected = _expected_output(template, inp, assignment)
    if not np.array_equal(out, expected):
        return False

    if np.count_nonzero(out == YELLOW) < 8:
        return False
    if any(np.count_nonzero(out_region == YELLOW) == 0 for out_region in _split_regions(out)[1]):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        side = rng.choice((5, 6))
        template = _make_template(side, rng)
        visibility_masks = _make_visibility_masks(template, rng)
        if len(visibility_masks) != 4:
            continue
        assignment = tuple(rng.sample(range(4), 4))
        inp = _build_input(template, visibility_masks, assignment)
        if _validate_example(template, visibility_masks, assignment, inp):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
