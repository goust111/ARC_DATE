from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
PALETTE = tuple(c for c in range(1, 10) if c != ORANGE)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(len(flags))
    while i < n:
        if not bool(flags[i]):
            i += 1
            continue
        start = i
        while i < n and bool(flags[i]):
            i += 1
        runs.append((start, i))
    return runs


def _full_projection(mask: np.ndarray) -> bool:
    return bool(mask.any() and np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)))


def _parse_sprite_regions(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    colors = sorted(set(map(int, np.unique(grid))) - {BLACK})
    assert len(colors) == 1, "expected one sprite color on black"
    sprite_color = int(colors[0])
    assert sprite_color != ORANGE, "input sprite color must differ from orange fill"

    occupied = grid != BLACK
    row_runs = _runs(np.any(occupied, axis=1))
    col_runs = _runs(np.any(occupied, axis=0))
    assert row_runs and col_runs, "expected at least one sprite region"

    regions: list[dict[str, object]] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            patch = grid[r0:r1, c0:c1]
            if not np.any(patch != BLACK):
                continue
            assert np.all((patch == BLACK) | (patch == sprite_color)), "region contains an unexpected color"
            mask = patch == sprite_color
            assert _full_projection(mask), "each sprite mask must span its rectangle"
            regions.append({"bbox": (r0, c0, r1, c1), "mask": mask})

    assert len(regions) >= 3, "expected at least three sprite regions"
    shapes = {tuple(region["mask"].shape) for region in regions}
    assert len(shapes) == 1, "all sprite regions must have equal size"
    return sprite_color, regions


def _candidate_triples(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            left = masks[i]
            right = masks[j]
            if not bool(np.any(left & ~right) and np.any(right & ~left)):
                continue
            union = left | right
            for k, target in enumerate(masks):
                if k in (i, j):
                    continue
                if bool(np.all(~target | union) and np.any(union & ~target)):
                    candidates.append((i, j, k))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the sprite region that is a damaged version of the pixel union of two other sprites.

    Input structure:
    - A black background contains several same-color, equal-size rectangular sprite regions.
    - The regions are separated by empty rows and columns, and each region is read in its own local coordinates.
    - Exactly one target region is a strict subset of the union of two other regions, while both operands have at least one pixel not shared by the other operand.

    Rule:
    - Compare every pair of sprite masks by their relative positions inside the equal rectangles.
    - Find the unique pair A and B and target C such that C is a proper subset of A union B.
    - In C's region only, recolor every pixel present in A union B but missing from C to orange; leave all existing sprite pixels and all other regions unchanged.

    Key insight:
    - The missing pixels are identified by a three-object union relation, not by filling a whole bounding box or matching one operand alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    _sprite_color, regions = _parse_sprite_regions(g)
    masks = [region["mask"] for region in regions]
    candidates = _candidate_triples(masks)
    assert len(candidates) == 1, "expected a unique union-subset triple"

    a_idx, b_idx, target_idx = candidates[0]
    union = masks[a_idx] | masks[b_idx]
    target = masks[target_idx]
    missing = union & ~target
    assert np.any(missing), "target must be a strict subset of the operand union"

    r0, c0, r1, c1 = regions[target_idx]["bbox"]
    out = g.copy()
    patch = out[int(r0) : int(r1), int(c0) : int(c1)]
    patch[missing] = ORANGE
    return out


# --------------- generator ---------------

_MAX_ATTEMPTS = 700
_MASK_ATTEMPTS = 250


def _random_full_mask(h: int, w: int, low_area: int, high_area: int) -> np.ndarray | None:
    high_area = min(high_area, h * w)
    low_area = max(low_area, max(h, w))
    if low_area > high_area:
        return None

    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(_MASK_ATTEMPTS):
        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            mask[r, random.randrange(w)] = True
        for c in range(w):
            mask[random.randrange(h), c] = True
        base = int(mask.sum())
        if base > high_area:
            continue

        target_area = random.randint(max(low_area, base), high_area)
        remaining = [(r, c) for r, c in cells if not mask[r, c]]
        random.shuffle(remaining)
        for r, c in remaining[: target_area - base]:
            mask[r, c] = True
        if _full_projection(mask):
            return mask
    return None


def _choose_target_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    union = left | right
    a_only = left & ~right
    b_only = right & ~left
    if int(a_only.sum()) < 2 or int(b_only.sum()) < 2 or np.all(union):
        return None

    a_cells = [tuple(map(int, rc)) for rc in np.argwhere(a_only)]
    b_cells = [tuple(map(int, rc)) for rc in np.argwhere(b_only)]
    union_cells = [tuple(map(int, rc)) for rc in np.argwhere(union)]

    for _ in range(_MASK_ATTEMPTS):
        missing: set[tuple[int, int]] = {random.choice(a_cells), random.choice(b_cells)}
        extra_limit = min(3, len(union_cells) - len(missing) - 1)
        if extra_limit > 0:
            pool = [cell for cell in union_cells if cell not in missing]
            missing.update(random.sample(pool, random.randint(0, extra_limit)))

        target = union.copy()
        for r, c in missing:
            target[r, c] = False

        if not _full_projection(target):
            continue
        if not bool(np.any(target & a_only) and np.any(target & b_only)):
            continue
        if not bool(np.any((union & ~target) & a_only) and np.any((union & ~target) & b_only)):
            continue
        return target
    return None


def _make_operand_triple(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    area = h * w
    low = max(h, w, int(round(area * 0.28)))
    high = max(low, int(round(area * 0.58)))
    for _ in range(_MASK_ATTEMPTS):
        left = _random_full_mask(h, w, low, high)
        right = _random_full_mask(h, w, low, high)
        if left is None or right is None:
            continue
        if np.array_equal(left, right):
            continue
        if not bool(np.any(left & ~right) and np.any(right & ~left)):
            continue
        union = left | right
        if np.all(union) or int(union.sum()) <= max(h, w) + 3:
            continue
        target = _choose_target_mask(left, right)
        if target is None:
            continue
        return left, right, target
    return None


def _unique_masks(masks: list[np.ndarray]) -> bool:
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.array_equal(masks[i], masks[j]):
                return False
    return True


def _make_masks(h: int, w: int, count: int) -> list[np.ndarray] | None:
    assert count >= 4
    area = h * w
    low = max(h, w, int(round(area * 0.25)))
    high = max(low, int(round(area * 0.62)))

    triple = _make_operand_triple(h, w)
    if triple is None:
        return None
    masks = [triple[0], triple[1], triple[2]]
    if _candidate_triples(masks) != [(0, 1, 2)]:
        return None

    while len(masks) < count:
        accepted = False
        for _ in range(_MASK_ATTEMPTS):
            candidate = _random_full_mask(h, w, low, high)
            if candidate is None:
                continue
            trial = masks + [candidate]
            if not _unique_masks(trial):
                continue
            if _candidate_triples(trial) == [(0, 1, 2)]:
                masks.append(candidate)
                accepted = True
                break
        if not accepted:
            return None

    random.shuffle(masks)
    if len(_candidate_triples(masks)) != 1:
        return None
    return masks


def _layout_options(h: int, w: int) -> list[tuple[int, int, int, int, int, int]]:
    options: list[tuple[int, int, int, int, int, int]] = []
    for rows in range(1, 5):
        for cols in range(1, 7):
            count = rows * cols
            if count < 4 or count > 9:
                continue
            for sep_r in (1, 2):
                for sep_c in (1, 2):
                    for top in (0, 1, 2):
                        for left in (0, 1, 2):
                            height = top + rows * h + (rows - 1) * sep_r + random.randint(0, 2)
                            width = left + cols * w + (cols - 1) * sep_c + random.randint(0, 2)
                            if height <= 30 and width <= 30:
                                options.append((rows, cols, sep_r, sep_c, top, left))
    return options


def _render_grid(
    masks: list[np.ndarray],
    h: int,
    w: int,
    layout: tuple[int, int, int, int, int, int],
    color: int,
) -> np.ndarray:
    rows, cols, sep_r, sep_c, top, left = layout
    bottom = random.randint(0, 2)
    right = random.randint(0, 2)
    height = top + rows * h + (rows - 1) * sep_r + bottom
    width = left + cols * w + (cols - 1) * sep_c + right
    grid = np.full((height, width), BLACK, dtype=int)

    for idx, mask in enumerate(masks):
        rr = idx // cols
        cc = idx % cols
        r0 = top + rr * (h + sep_r)
        c0 = left + cc * (w + sep_c)
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[mask] = color
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        color, regions = _parse_sprite_regions(grid)
        if color == ORANGE:
            return False
        masks = [region["mask"] for region in regions]
        triples = _candidate_triples(masks)
        if len(triples) != 1:
            return False

        a_idx, b_idx, target_idx = triples[0]
        union = masks[a_idx] | masks[b_idx]
        target = masks[target_idx]
        missing = union & ~target
        a_only = masks[a_idx] & ~masks[b_idx]
        b_only = masks[b_idx] & ~masks[a_idx]

        if not bool(np.any(missing & a_only) and np.any(missing & b_only)):
            return False
        if not bool(np.any(target & a_only) and np.any(target & b_only)):
            return False
        if np.all(union):
            return False

        solved = main(grid)
        if solved.shape != grid.shape or np.array_equal(solved, grid):
            return False
        changed = solved != grid
        expected_changed = np.zeros(grid.shape, dtype=bool)
        r0, c0, r1, c1 = regions[target_idx]["bbox"]
        expected_changed[int(r0) : int(r1), int(c0) : int(c1)] = missing
        if not np.array_equal(changed, expected_changed):
            return False
        if not np.all(solved[expected_changed] == ORANGE):
            return False

        bbox_fill = grid.copy()
        patch = bbox_fill[int(r0) : int(r1), int(c0) : int(c1)]
        patch[patch == BLACK] = ORANGE
        if np.array_equal(bbox_fill, solved):
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        layouts = _layout_options(h, w)
        if not layouts:
            continue
        layout = random.choice(layouts)
        count = layout[0] * layout[1]
        masks = _make_masks(h, w, count)
        if masks is None:
            continue

        color = random.choice(PALETTE)
        grid = _render_grid(masks, h, w, layout, color)
        if max(grid.shape) <= 30 and _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid missing-union input")
