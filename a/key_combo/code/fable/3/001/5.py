from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, find_components


BLACK = int(Color.BLACK)


def _component_masks(g: np.ndarray, color: int) -> list[np.ndarray]:
    comps = find_components(
        g,
        bg=BLACK,
        connectivity=8,
        monochromatic=True,
        colors=[int(color)],
        min_area=1,
    )
    comps = sorted(comps, key=lambda obj: tuple(map(int, obj["bbox"])))
    if len(comps) < 2:
        return []

    dims = []
    masks = []
    for obj in comps:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        dims.append((r1 - r0 + 1, c1 - c0 + 1))
        masks.append(g[r0 : r1 + 1, c0 : c1 + 1] == int(color))

    if len(set(dims)) != 1:
        return []
    return masks


def _find_template(masks: list[np.ndarray]) -> int | None:
    winners: list[int] = []
    for i, mask in enumerate(masks):
        ok = True
        for j, other in enumerate(masks):
            if i == j:
                continue
            if not np.all(np.logical_or(~other, mask)):
                ok = False
                break
            if not np.any(mask & ~other):
                ok = False
                break
        if ok:
            winners.append(i)
    if len(winners) != 1:
        return None
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Aggregate every partial sprite's missing cells relative to the complete template.

    Input structure:
    - A black background contains several same-color sprite bodies with identical
      bounding-box size, plus sparse single-cell distractors in other colors.
    - After bbox alignment, exactly one same-color sprite mask is a strict
      superset of every other same-color sprite mask.

    Rule:
    - Find the unique complete template sprite by the strict-superset relation.
    - For each remaining same-color sprite, compute the cells where the template
      has the sprite color and the copy does not.
    - Return a black image with the template bbox size, coloring the union of all
      those missing cells in the sprite color.

    Key insight:
    - The answer is not any visible sprite and not the repaired copies; it is the
      transparent overlay of all template-minus-copy difference masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.any(g == BLACK), "expected a black background"

    candidates: list[tuple[int, list[np.ndarray], int]] = []
    for raw_color in sorted(int(c) for c in np.unique(g) if int(c) != BLACK):
        masks = _component_masks(g, raw_color)
        if not masks:
            continue
        template_idx = _find_template(masks)
        if template_idx is not None:
            candidates.append((raw_color, masks, template_idx))

    assert len(candidates) == 1, "expected one sprite color with a unique full template"
    color, masks, template_idx = candidates[0]
    template = masks[template_idx]

    missing_union = np.zeros_like(template, dtype=bool)
    for i, mask in enumerate(masks):
        if i == template_idx:
            continue
        missing = template & ~mask
        assert np.any(missing), "each copy must be a strict partial copy"
        missing_union |= missing

    assert np.any(missing_union), "expected at least one missing cell"
    out = np.zeros(template.shape, dtype=int)
    out[missing_union] = int(color)
    return out


# --------------- generator ---------------

GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
TEMPLATE_H_RANGE = (4, 7)
TEMPLATE_W_RANGE = (4, 7)
COPY_COUNT_RANGE = (2, 4)
DISTRACTOR_COUNT_RANGE = (3, 8)
GENERATION_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 200
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _paint_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = map(int, start)
    tr, tc = map(int, end)
    mask[r, c] = True
    axes = [0, 1]
    random.shuffle(axes)
    for axis in axes:
        if axis == 0:
            while r != tr:
                r += 1 if tr > r else -1
                mask[r, c] = True
        else:
            while c != tc:
                c += 1 if tc > c else -1
                mask[r, c] = True


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for dr, dc in ORTHOGONAL_STEPS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _grow_connected(mask: np.ndarray, target: int) -> None:
    while int(mask.sum()) < int(target):
        cells = _frontier(mask)
        if not cells:
            return
        r, c = random.choice(cells)
        mask[r, c] = True


def _is_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return len(comps) == 1


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_template_mask(h: int, w: int) -> np.ndarray:
    for _ in range(100):
        mask = np.zeros((h, w), dtype=bool)
        edge_points = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(edge_points)
        start = edge_points[0]
        for end in edge_points[1:]:
            _paint_path(mask, start, end)
            start = end

        lo = max(int(mask.sum()), int(np.ceil(0.48 * h * w)))
        hi = max(lo, int(np.floor(0.76 * h * w)))
        _grow_connected(mask, random.randint(lo, hi))
        if _touches_all_edges(mask) and _is_connected(mask) and not np.all(mask):
            return mask
    raise RuntimeError("failed to make a template mask")


def _copy_is_valid(template: np.ndarray, missing: set[tuple[int, int]]) -> bool:
    copy = template.copy()
    for r, c in missing:
        copy[r, c] = False
    return bool(copy.any() and _touches_all_edges(copy) and _is_connected(copy))


def _make_missing_sets(template: np.ndarray, count: int) -> list[frozenset[tuple[int, int]]] | None:
    cells = [tuple(map(int, p)) for p in np.argwhere(template)]
    removable = [cell for cell in cells if _copy_is_valid(template, {cell})]
    if len(removable) < count:
        return None

    for _ in range(250):
        random.shuffle(removable)
        bases = list(removable[:count])
        base_set = set(bases)
        missing: list[set[tuple[int, int]]] = [{cell} for cell in bases]
        optional = [cell for cell in cells if cell not in base_set]
        random.shuffle(optional)

        if count >= 3 and random.random() < 0.65:
            for cell in optional:
                receivers = random.sample(range(count), random.randint(2, count))
                if all(_copy_is_valid(template, missing[i] | {cell}) for i in receivers):
                    for i in receivers:
                        missing[i].add(cell)
                    break

        max_missing = max(1, min(5, int(template.sum()) // 4))
        for i in range(count):
            target = random.randint(1, max_missing)
            random.shuffle(optional)
            for cell in optional:
                if len(missing[i]) >= target:
                    break
                trial = missing[i] | {cell}
                if _copy_is_valid(template, trial):
                    missing[i].add(cell)

        frozen = [frozenset(s) for s in missing]
        union = set().union(*frozen)
        if len(union) >= int(template.sum()):
            continue
        if not any(len(s) > 1 for s in frozen) and len(optional) >= count:
            continue
        if any(not (set(s) - set().union(*(set(t) for j, t in enumerate(frozen) if j != i))) for i, s in enumerate(frozen)):
            continue
        copy_masks = []
        for s in frozen:
            copy = template.copy()
            for r, c in s:
                copy[r, c] = False
            copy_masks.append(copy)
        union_mask = np.zeros_like(template, dtype=bool)
        for r, c in union:
            union_mask[r, c] = True
        if any(np.array_equal(union_mask, copy) for copy in copy_masks):
            continue
        return frozen
    return None


def _can_reserve(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = map(int, occupied.shape)
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(H, r + h + pad)
    c1 = min(W, c + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _reserve(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    H, W = map(int, occupied.shape)
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(H, r + h + pad)
    c1 = min(W, c + w + pad)
    occupied[r0:r1, c0:c1] = True


def _random_location(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = map(int, occupied.shape)
    if h > H or w > W:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        r = random.randint(0, H - h)
        c = random.randint(0, W - w)
        if _can_reserve(occupied, r, c, h, w):
            return r, c
    return None


def _paint_sprite(grid: np.ndarray, mask: np.ndarray, color: int, origin: tuple[int, int]) -> None:
    r, c = map(int, origin)
    patch = grid[r : r + mask.shape[0], c : c + mask.shape[1]]
    patch[mask] = int(color)


def _add_distractors(
    grid: np.ndarray,
    sprite_color: int,
    preferred: list[tuple[int, int]] | None = None,
) -> None:
    target = random.randint(*DISTRACTOR_COUNT_RANGE)
    colors = [int(c) for c in Color.NOT_BLACK if int(c) != int(sprite_color)]
    positions: set[tuple[int, int]] = set()

    def place_if_clear(r: int, c: int) -> bool:
        if int(grid[r, c]) != BLACK:
            return False
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in positions):
            return False
        grid[r, c] = random.choice(colors)
        positions.add((r, c))
        return True

    preferred = list(preferred or [])
    random.shuffle(preferred)
    preferred_target = min(target, len(preferred), random.randint(1, 2) if preferred else 0)
    for r, c in preferred:
        if len(positions) >= preferred_target:
            break
        place_if_clear(int(r), int(c))

    black_cells = [tuple(map(int, p)) for p in np.argwhere(grid == BLACK)]
    random.shuffle(black_cells)
    for r, c in black_cells:
        if len(positions) >= target:
            break
        place_if_clear(int(r), int(c))


def _expected_output(template: np.ndarray, missing_sets: list[frozenset[tuple[int, int]]], color: int) -> np.ndarray:
    out = np.zeros(template.shape, dtype=int)
    for missing in missing_sets:
        for r, c in missing:
            out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*TEMPLATE_H_RANGE)
    w = random.randint(*TEMPLATE_W_RANGE)
    copy_count = random.randint(*COPY_COUNT_RANGE)
    template = _make_template_mask(h, w)
    missing_sets = _make_missing_sets(template, copy_count)
    if missing_sets is None:
        return None

    sprite_color = random.choice(list(Color.NOT_BLACK))
    H = random.randint(max(GRID_H_RANGE[0], h + 4), GRID_H_RANGE[1])
    W = random.randint(max(GRID_W_RANGE[0], w + 4), GRID_W_RANGE[1])
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    masks: list[np.ndarray] = [template.copy()]
    for missing in missing_sets:
        copy = template.copy()
        for r, c in missing:
            copy[r, c] = False
        masks.append(copy)
    random.shuffle(masks)

    preferred_distractors: list[tuple[int, int]] = []
    for mask in masks:
        loc = _random_location(occupied, h, w)
        if loc is None:
            return None
        _paint_sprite(grid, mask, sprite_color, loc)
        for dr, dc in np.argwhere(~mask):
            preferred_distractors.append((int(loc[0] + dr), int(loc[1] + dc)))
        _reserve(occupied, loc[0], loc[1], h, w)

    _add_distractors(grid, sprite_color, preferred_distractors)
    expected = _expected_output(template, missing_sets, sprite_color)

    computed = main(grid)
    if not np.array_equal(computed, expected):
        return None
    out_mask = expected == int(sprite_color)
    if not np.any(out_mask) or np.array_equal(out_mask, template):
        return None
    for i, missing in enumerate(missing_sets):
        others = set().union(*(set(s) for j, s in enumerate(missing_sets) if j != i))
        if not (set(missing) - others):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for fable/3/001/5")
