from __future__ import annotations

from collections import Counter
import random

import numpy as np


DISTRACTOR_BASES = [
    np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
        dtype=bool,
    ),
]

AMBIGUOUS_BASES = [
    np.array(
        [
            [0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
]

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int).tolist())


def _crop_color_mask(grid: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair every monochrome sprite with its main-diagonal transpose and emit the missing partner of the lone unmatched sprite.

    Input structure:
    - The grid has one dominant background color and several pairwise separated monochrome sprites.
    - Every sprite is a single-color shape read inside its own minimal bounding box, and sprite colors are all distinct.
    - All but one sprite have exactly one partner whose cropped boolean mask is the transpose of their own mask.

    Rule:
    - Ignore sprite colors and positions, and compare only cropped shapes.
    - Pair sprites whose cropped masks are exact main-diagonal transposes of one another.
    - Find the unique unpaired sprite and output its transposed mask in its own color, cropped to the minimal bounding box of that missing partner and filled with the input background elsewhere.

    Key insight:
    - The relation is exact transpose, not rotation or ordinary reflection, so pairing must be checked cell-for-cell on cropped masks.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(c) for c in np.unique(g) if int(c) != int(bg))
    assert colors

    sprites = [{"color": color, "mask": _crop_color_mask(g, color)} for color in colors]
    key_to_indices: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    for idx, sprite in enumerate(sprites):
        key_to_indices.setdefault(_mask_key(sprite["mask"]), []).append(idx)

    used: set[int] = set()
    unmatched_idx: int | None = None
    for idx, sprite in enumerate(sprites):
        if idx in used:
            continue
        tkey = _mask_key(sprite["mask"].T)
        partners = [j for j in key_to_indices.get(tkey, []) if j != idx and j not in used]
        if len(partners) == 1:
            used.add(idx)
            used.add(int(partners[0]))
            continue
        assert len(partners) == 0
        assert unmatched_idx is None
        unmatched_idx = idx
        used.add(idx)

    assert unmatched_idx is not None
    assert len(used) == len(sprites)

    unmatched = sprites[int(unmatched_idx)]
    partner_mask = unmatched["mask"].T
    out = np.full(partner_mask.shape, int(bg), dtype=int)
    out[partner_mask] = int(unmatched["color"])
    return out


def _transform_variants(mask: np.ndarray, swap_dims: bool | None = None) -> list[np.ndarray]:
    raw = [
        mask.copy(),
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
        mask.T.copy(),
        np.fliplr(mask.T),
    ]
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for cand in raw:
        if swap_dims is not None and ((cand.shape != mask.shape) != bool(swap_dims)):
            continue
        key = _mask_key(cand)
        if key in seen:
            continue
        seen.add(key)
        variants.append(cand.astype(bool, copy=True))
    return variants


def _random_transform(mask: np.ndarray, rng: random.Random, swap_dims: bool | None = None) -> np.ndarray:
    variants = _transform_variants(mask, swap_dims=swap_dims)
    assert variants
    return variants[rng.randrange(len(variants))].copy()


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_axis_asymmetric(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            return False
        anti_diag = np.fliplr(np.flipud(mask)).T
        if np.array_equal(mask, anti_diag):
            return False
    return True


def _random_connected_mask(
    h: int,
    w: int,
    area: int,
    rng: random.Random,
    *,
    near_horizontal: bool = False,
) -> np.ndarray | None:
    for _ in range(500):
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True

        if not _touches_all_sides(mask):
            continue
        if not _is_axis_asymmetric(mask):
            continue
        if near_horizontal and float(np.mean(mask == np.fliplr(mask))) < 0.6:
            continue
        return mask
    return None


def _conflicts(mask: np.ndarray, used_keys: set[tuple[tuple[int, ...], ...]]) -> bool:
    return _mask_key(mask) in used_keys or _mask_key(mask.T) in used_keys


def _sample_random_shape(
    rng: random.Random,
    used_keys: set[tuple[tuple[int, ...], ...]],
    *,
    prefer_dims: tuple[int, int] | None = None,
    near_horizontal: bool = False,
) -> np.ndarray | None:
    for _ in range(400):
        if prefer_dims is None:
            h = rng.randint(3, 6)
            w = rng.randint(3, 7)
            if h == w:
                continue
        else:
            h, w = prefer_dims
        min_area = max(h + w - 2, (h * w + 2) // 3)
        max_area = min(h * w - 1, (2 * h * w) // 3 + 1)
        if min_area > max_area:
            continue
        area = rng.randint(min_area, max_area)
        mask = _random_connected_mask(h, w, area, rng, near_horizontal=near_horizontal)
        if mask is None:
            continue
        mask = _random_transform(mask, rng)
        if _conflicts(mask, used_keys):
            continue
        return mask
    return None


def _mark_reserved(reserved: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    r0 = max(0, int(r) - 1)
    c0 = max(0, int(c) - 1)
    r1 = min(reserved.shape[0], int(r) + int(h) + 1)
    c1 = min(reserved.shape[1], int(c) + int(w) + 1)
    reserved[r0:r1, c0:c1] = True


def _place_objects(
    masks: list[np.ndarray],
    colors: list[int],
    shape: tuple[int, int],
    bg: int,
    rng: random.Random,
) -> np.ndarray | None:
    H, W = shape
    grid = np.full((H, W), int(bg), dtype=int)
    reserved = np.zeros((H, W), dtype=bool)
    order = list(range(len(masks)))
    rng.shuffle(order)

    for idx in order:
        mask = masks[idx]
        color = colors[idx]
        h, w = mask.shape
        placed = False
        for _ in range(600):
            r = rng.randint(0, H - h)
            c = rng.randint(0, W - w)
            if reserved[r : r + h, c : c + w].any():
                continue
            patch = grid[r : r + h, c : c + w]
            if np.any(patch[mask] != int(bg)):
                continue
            patch[mask] = int(color)
            _mark_reserved(reserved, r, c, h, w)
            placed = True
            break
        if not placed:
            return None
    return grid


def _build_shape_plan(rng: random.Random, pair_count: int) -> list[np.ndarray] | None:
    used_keys: set[tuple[tuple[int, ...], ...]] = set()
    bases: list[np.ndarray] = []

    distractor_swap = rng.choice([False, True])
    first = _random_transform(DISTRACTOR_BASES[0], rng, swap_dims=distractor_swap)
    second = _random_transform(DISTRACTOR_BASES[1], rng, swap_dims=distractor_swap)
    if _conflicts(first, used_keys):
        return None
    used_keys.add(_mask_key(first))
    used_keys.add(_mask_key(first.T))
    if _conflicts(second, used_keys):
        return None
    used_keys.add(_mask_key(second))
    used_keys.add(_mask_key(second.T))
    bases.extend([first, second])

    ambiguous = _random_transform(rng.choice(AMBIGUOUS_BASES), rng)
    if _conflicts(ambiguous, used_keys):
        return None
    used_keys.add(_mask_key(ambiguous))
    used_keys.add(_mask_key(ambiguous.T))
    bases.append(ambiguous)

    while len(bases) < pair_count:
        extra = _sample_random_shape(rng, used_keys)
        if extra is None:
            return None
        used_keys.add(_mask_key(extra))
        used_keys.add(_mask_key(extra.T))
        bases.append(extra)

    prefer_dims = rng.choice([bases[0].shape, bases[1].shape, ambiguous.shape])
    unmatched = _sample_random_shape(
        rng,
        used_keys,
        prefer_dims=prefer_dims if rng.random() < 0.6 else None,
        near_horizontal=rng.random() < 0.35,
    )
    if unmatched is None:
        return None
    bases.append(unmatched)
    return bases


def _expected_output(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    partner = mask.T
    out = np.full(partner.shape, int(bg), dtype=int)
    out[partner] = int(color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(3, 4)
    bases = _build_shape_plan(rng, pair_count)
    if bases is None:
        return None

    unmatched_mask = bases[-1]
    masks: list[np.ndarray] = []
    for base in bases[:-1]:
        masks.append(base)
        masks.append(base.T.copy())
    masks.append(unmatched_mask)

    object_count = len(masks)
    palette = list(range(10))
    bg = rng.choice(palette)
    fg_colors = [c for c in palette if c != bg]
    rng.shuffle(fg_colors)
    colors = fg_colors[:object_count]

    min_side = 24 if pair_count == 4 else 21
    H = rng.randint(min_side, 30)
    W = rng.randint(min_side, 30)
    if sum(int(mask.sum()) for mask in masks) * 2 >= H * W:
        return None

    grid = _place_objects(masks, colors, (H, W), bg, rng)
    if grid is None:
        return None

    expected = _expected_output(unmatched_mask, colors[-1], bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")
