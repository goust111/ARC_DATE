from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 400
PANEL_SIZES = (9, 10, 11, 12, 13)
TARGET_RED_RANGE = (2, 4)
TARGET_AREA_RANGE = (14, 25)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of a boolean mask as (r0, c0, r1, c1), inclusive."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _orbit_positions(r: int, c: int, box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = box
    pts = {
        (r, c),
        (r0 + r1 - r, c),
        (r, c0 + c1 - c),
        (r0 + r1 - r, c0 + c1 - c),
    }
    return tuple(sorted((int(rr), int(cc)) for rr, cc in pts))


def _orbit_weight(r: int, c: int, h: int, w: int) -> int:
    row_mult = 1 if r == h - 1 - r else 2
    col_mult = 1 if c == w - 1 - c else 2
    return int(row_mult * col_mult)


def _divider_column(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLUE)]
    assert len(cols) == 1, "expected exactly one full blue divider column"
    return int(cols[0])


def _spec_output(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    union = left_mask | right_mask
    out = np.zeros_like(left_mask, dtype=int)
    out[left_mask & right_mask] = GREEN
    out[left_mask ^ right_mask] = YELLOW

    box = bbox(union)
    if box is None:
        return out

    r0, c0, r1, c1 = box
    visited: set[tuple[tuple[int, int], ...]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            orbit = _orbit_positions(r, c, box)
            if orbit in visited:
                continue
            visited.add(orbit)
            if not any(union[rr, cc] for rr, cc in orbit):
                continue
            for rr, cc in orbit:
                if not left_mask[rr, cc] and not right_mask[rr, cc]:
                    out[rr, cc] = RED
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Merge the two divider-separated panels, then mark shared, single-sided, and symmetry-inferred pixels.

    Input structure:
    - A square left panel and an equally sized square right panel are separated by one full blue column.
    - Each panel contains only black background and yellow pattern pixels.
    - After aligning the two panels cell-by-cell, their yellow union belongs to a shape whose bbox defines horizontal and vertical symmetry axes.

    Rule:
    - Remove the blue divider and compare the aligned left and right panels.
    - Paint green where both panels contain yellow and yellow where exactly one panel contains yellow.
    - Inside the merged pattern bbox, use the horizontal and vertical reflection orbit of each cell; if an orbit already has a known merged yellow cell, then every still-missing cell in that orbit is inferred and painted red.

    Key insight:
    - The final object is not read from either panel alone: first take the aligned union, then use only the union bbox and its two reflection axes to recover the jointly missing cells.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    div = _divider_column(grid)
    left = grid[:, :div]
    right = grid[:, div + 1 :]
    assert left.shape == right.shape and left.shape[0] <= 30 and left.shape[1] <= 30
    assert np.all(np.isin(left, (BLACK, YELLOW)))
    assert np.all(np.isin(right, (BLACK, YELLOW)))

    return _spec_output(left == YELLOW, right == YELLOW)


def _seed_sides(seed: set[tuple[int, int]], qh: int, qw: int) -> set[str]:
    sides = set()
    for r, c in seed:
        if r == 0:
            sides.add("top")
        if c == 0:
            sides.add("left")
        if r == qh - 1:
            sides.add("bottom")
        if c == qw - 1:
            sides.add("right")
    return sides


def _full_mask_from_seed(seed: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    local_box = (0, 0, h - 1, w - 1)
    for r, c in seed:
        for rr, cc in _orbit_positions(r, c, local_box):
            mask[rr, cc] = True
    return mask


def _seed_area(seed: set[tuple[int, int]], h: int, w: int) -> int:
    return int(sum(_orbit_weight(r, c, h, w) for r, c in seed))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _build_seed(h: int, w: int, area_lo: int, area_hi: int) -> set[tuple[int, int]] | None:
    qh = (h + 1) // 2
    qw = (w + 1) // 2
    for _ in range(120):
        seed = {(qh - 1, qw - 1)}
        while True:
            area = _seed_area(seed, h, w)
            sides = _seed_sides(seed, qh, qw)
            if area_lo <= area <= area_hi and len(sides) == 4:
                return seed
            if area > area_hi:
                break

            frontier = set()
            for r, c in seed:
                frontier.update(_neighbors4(r, c, qh, qw))
            frontier.difference_update(seed)
            if not frontier:
                break

            missing = {"top", "left", "bottom", "right"} - sides

            def score(cell: tuple[int, int]) -> float:
                r, c = cell
                new_sides = len(_seed_sides(seed | {cell}, qh, qw) - sides)
                s = 5.0 * new_sides + random.random()
                if "top" in missing:
                    s += float(qh - 1 - r)
                if "left" in missing:
                    s += float(qw - 1 - c)
                if "bottom" in missing:
                    s += float(r)
                if "right" in missing:
                    s += float(c)
                s += 0.2 * (_orbit_weight(r, c, h, w) == 4)
                return s

            ranked = sorted(frontier, key=score, reverse=True)
            top_k = ranked[: min(3, len(ranked))]
            seed.add(random.choice(top_k))
    return None


def _occupied_orbits(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    box = bbox(mask)
    assert box is not None
    groups: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}
    for r, c in np.argwhere(mask):
        orbit = _orbit_positions(int(r), int(c), box)
        groups[orbit] = orbit
    return list(groups.values())


def _pick_joint_gaps(full_mask: np.ndarray) -> set[tuple[int, int]] | None:
    all_orbits = [orbit for orbit in _occupied_orbits(full_mask) if len(orbit) >= 2]
    capacity = sum(len(orbit) - 1 for orbit in all_orbits)
    if capacity < TARGET_RED_RANGE[0]:
        return None

    target = random.randint(TARGET_RED_RANGE[0], min(TARGET_RED_RANGE[1], capacity))
    chosen: set[tuple[int, int]] = set()
    orbit_counts = {orbit: 0 for orbit in all_orbits}

    candidates = [orbit for orbit in all_orbits if len(orbit) > 1]
    while len(chosen) < target and candidates:
        orbit = random.choice(candidates)
        remaining = [cell for cell in orbit if cell not in chosen]
        if not remaining:
            candidates = [g for g in candidates if any(cell not in chosen for cell in g)]
            continue
        cell = random.choice(remaining)
        chosen.add(cell)
        orbit_counts[orbit] += 1
        candidates = [g for g in candidates if orbit_counts[g] < len(g) - 1]

    if len(chosen) != target:
        return None

    union = full_mask.copy()
    for r, c in chosen:
        union[r, c] = False
    full_box = bbox(full_mask)
    union_box = bbox(union)
    if union_box != full_box:
        return None
    return chosen


def _split_panels(full_mask: np.ndarray, joint_gaps: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray] | None:
    occupied = [tuple(map(int, cell)) for cell in np.argwhere(full_mask)]
    remaining = [cell for cell in occupied if cell not in joint_gaps]
    if len(remaining) < 4:
        return None

    max_single = min(len(remaining) - 2, max(4, len(remaining) * 3 // 5))
    if max_single < 2:
        return None
    single_count = random.randint(2, max_single)
    singles = random.sample(remaining, single_count)
    split = random.randint(1, single_count - 1)
    left_only = set(random.sample(singles, split))
    right_only = set(singles) - left_only
    if not left_only or not right_only:
        return None

    left = full_mask.copy()
    right = full_mask.copy()

    for r, c in joint_gaps:
        left[r, c] = False
        right[r, c] = False
    for r, c in left_only:
        right[r, c] = False
    for r, c in right_only:
        left[r, c] = False

    if not np.any(left & right):
        return None
    if np.sum(full_mask & ~left) < 2 or np.sum(full_mask & ~right) < 2:
        return None
    return left, right


def _embed_panel(mask: np.ndarray, panel_size: int) -> np.ndarray:
    panel = np.zeros((panel_size, panel_size), dtype=int)
    h, w = mask.shape
    top = random.randint(0, panel_size - h)
    left = random.randint(0, panel_size - w)
    panel[top : top + h, left : left + w][mask] = YELLOW
    return panel


def _input_from_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    panel_size = left.shape[0]
    grid = np.zeros((panel_size, panel_size * 2 + 1), dtype=int)
    grid[:, panel_size] = BLUE
    grid[:, :panel_size] = left
    grid[:, panel_size + 1 :] = right
    return grid


def _validate_instance(grid: np.ndarray, full_panel_mask: np.ndarray, left: np.ndarray, right: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        return False
    if grid.shape != (left.shape[0], left.shape[1] * 2 + 1):
        return False

    union = left | right
    full_box = bbox(full_panel_mask)
    union_box = bbox(union)
    if full_box is None or union_box != full_box:
        return False

    r0, c0, r1, c1 = full_box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            orbit = _orbit_positions(r, c, full_box)
            vals = {bool(full_panel_mask[rr, cc]) for rr, cc in orbit}
            if len(vals) != 1:
                return False

    joint = full_panel_mask & ~(left | right)
    if not (TARGET_RED_RANGE[0] <= int(joint.sum()) <= TARGET_RED_RANGE[1]):
        return False
    if not np.any(left & right):
        return False
    if not np.any(left & ~right):
        return False
    if not np.any(right & ~left):
        return False

    expected = np.zeros_like(left, dtype=int)
    expected[left & right] = GREEN
    expected[left ^ right] = YELLOW
    expected[joint] = RED

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False

    box = bbox(union)
    assert box is not None
    for r, c in np.argwhere(joint):
        orbit = _orbit_positions(int(r), int(c), box)
        if not any(union[rr, cc] for rr, cc in orbit):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    panel_size = random.choice(PANEL_SIZES)
    box_h = random.randint(5, min(8, panel_size))
    box_w = random.randint(5, min(8, panel_size))
    area_hi = min(TARGET_AREA_RANGE[1], box_h * box_w - 1)
    area_lo = min(TARGET_AREA_RANGE[0], area_hi)
    if area_hi < area_lo:
        return None

    seed = _build_seed(box_h, box_w, area_lo, area_hi)
    if seed is None:
        return None
    local_full = _full_mask_from_seed(seed, box_h, box_w)

    joint_gaps = _pick_joint_gaps(local_full)
    if joint_gaps is None:
        return None
    split = _split_panels(local_full, joint_gaps)
    if split is None:
        return None
    local_left, local_right = split

    top = random.randint(0, panel_size - box_h)
    left = random.randint(0, panel_size - box_w)
    full_panel_mask = np.zeros((panel_size, panel_size), dtype=bool)
    left_panel = np.zeros((panel_size, panel_size), dtype=bool)
    right_panel = np.zeros((panel_size, panel_size), dtype=bool)
    full_panel_mask[top : top + box_h, left : left + box_w] = local_full
    left_panel[top : top + box_h, left : left + box_w] = local_left
    right_panel[top : top + box_h, left : left + box_w] = local_right

    inp = _input_from_panels(left_panel.astype(int) * YELLOW, right_panel.astype(int) * YELLOW)
    if not _validate_instance(inp, full_panel_mask, left_panel, right_panel):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-panel symmetry-gap puzzle")
