from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NON_ID_INVOLUTIONS = ("rot180", "flip_lr", "flip_ud", "transpose", "anti")
ALL_D4 = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti",
)
GEN_ATTEMPTS = 500
PLACEMENT_PADDING = 1


def _signature(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in arr)


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1)
    if name == "rot180":
        return np.rot90(arr, 2)
    if name == "rot270":
        return np.rot90(arr, 3)
    if name == "flip_lr":
        return np.fliplr(arr)
    if name == "flip_ud":
        return np.flipud(arr)
    if name == "transpose":
        return arr.T.copy()
    if name == "anti":
        return np.transpose(arr[::-1, ::-1]).copy()
    raise ValueError(f"unknown transform: {name}")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, b: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = b
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    comps.sort(key=lambda comp: tuple(map(int, comp["bbox"][:2])))
    return comps


def _component_crops(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert len(comps) >= 3 and len(comps) % 2 == 1, "expected an odd number of sprites"
    crops = []
    for comp in comps:
        crop = _crop_bbox(g, comp["bbox"])
        assert int(np.count_nonzero(crop != bg)) == int(comp["area"])
        crops.append(crop)
    return crops


def _candidate_edges(crops: list[np.ndarray], transform: str) -> list[int]:
    n = len(crops)
    edges = [0] * n
    transformed = [_apply_transform(crop, transform) for crop in crops]
    for i in range(n):
        for j in range(i + 1, n):
            if transformed[i].shape == crops[j].shape and np.array_equal(transformed[i], crops[j]):
                edges[i] |= 1 << j
                edges[j] |= 1 << i
    return edges


def _unique_unmatched_for_transform(crops: list[np.ndarray], transform: str) -> int | None:
    n = len(crops)
    full_mask = (1 << n) - 1
    edges = _candidate_edges(crops, transform)

    @lru_cache(maxsize=None)
    def has_perfect_matching(mask: int) -> bool:
        if mask == 0:
            return True
        lowbit = mask & -mask
        i = lowbit.bit_length() - 1
        rest = mask ^ (1 << i)
        options = edges[i] & rest
        while options:
            pick = options & -options
            j = pick.bit_length() - 1
            if has_perfect_matching(rest ^ (1 << j)):
                return True
            options ^= pick
        return False

    unmatched = [idx for idx in range(n) if has_perfect_matching(full_mask ^ (1 << idx))]
    if len(unmatched) == 1:
        return unmatched[0]
    return None


def _infer_transform_and_unmatched(crops: list[np.ndarray]) -> tuple[str, int]:
    valid: list[tuple[str, int]] = []
    for transform in NON_ID_INVOLUTIONS:
        unmatched = _unique_unmatched_for_transform(crops, transform)
        if unmatched is not None:
            valid.append((transform, unmatched))
    assert len(valid) == 1, f"expected one global pairing transform, got {valid}"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the missing partner of the lone sprite by extrapolating the scene-wide shared dihedral pairing transform.

    Input structure:
    - A dominant background contains several separated 4-connected multicolor sprites.
    - All but one sprite can be partitioned into pairs whose tight crops match under one shared non-identity dihedral involution.
    - The remaining sprite is the only one whose transformed companion is absent from the grid.

    Rule:
    - Crop every sprite to its tight bounding box and test non-identity dihedral involutions as exact pairing rules.
    - Identify the unique involution that lets all but one sprite pair off and locate that lone unmatched sprite.
    - Return the unmatched sprite transformed by the shared rule, cropped to its own tight box.

    Key insight:
    - The visible odd sprite is not the answer itself; it is the source exemplar whose never-shown companion must be synthesized using the transform inferred from the already completed pairs.
    """
    crops = _component_crops(grid)
    transform, unmatched = _infer_transform_and_unmatched(crops)
    out = _apply_transform(crops[unmatched], transform)
    input_sigs = {_signature(crop) for crop in crops}
    assert _signature(out) not in input_sigs, "missing companion must be novel"
    return out


def _connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    cells = [start]
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
        cells.append((nr, nc))
    b = _bbox(mask)
    if b != (0, 0, h - 1, w - 1):
        return None
    if mask.all():
        return None
    return mask


def _colorize_mask(mask: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    crop = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    n_colors = random.randint(2, min(3, len(palette)))
    colors = random.sample(list(palette), n_colors)
    random.shuffle(coords)
    splits = sorted(random.sample(range(1, len(coords)), n_colors - 1))
    groups = []
    prev = 0
    for split in splits + [len(coords)]:
        groups.append(coords[prev:split])
        prev = split
    for color, group in zip(colors, groups):
        for r, c in group:
            crop[r, c] = color
    return crop


def _orbit_signatures(crop: np.ndarray) -> dict[str, tuple[tuple[int, ...], ...]]:
    return {name: _signature(_apply_transform(crop, name)) for name in ALL_D4}


def _gen_seed_crop(h: int, w: int, palette: tuple[int, ...], used: set[tuple[tuple[int, ...], ...]]) -> np.ndarray | None:
    min_area = max(5, (h * w * 3) // 5)
    max_area = min(h * w - 1, (h * w * 4) // 5 + 1)
    if min_area > max_area:
        return None
    for _ in range(120):
        area = random.randint(min_area, max_area)
        mask = _connected_mask(h, w, area)
        if mask is None:
            continue
        crop = _colorize_mask(mask, palette)
        orbit = _orbit_signatures(crop)
        if len(set(orbit.values())) != len(ALL_D4):
            continue
        if any(sig in used for sig in orbit.values()):
            continue
        return crop
    return None


def _candidate_positions(reserved: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = reserved.shape
    h, w = shape
    positions = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - PLACEMENT_PADDING)
            c0 = max(0, c - PLACEMENT_PADDING)
            r1 = min(H, r + h + PLACEMENT_PADDING)
            c1 = min(W, c + w + PLACEMENT_PADDING)
            if not reserved[r0:r1, c0:c1].any():
                positions.append((r, c))
    random.shuffle(positions)
    return positions


def _place_crops(crops: list[np.ndarray], H: int, W: int) -> np.ndarray | None:
    for _ in range(80):
        reserved = np.zeros((H, W), dtype=bool)
        locs: list[tuple[int, int] | None] = [None] * len(crops)
        order = list(range(len(crops)))
        random.shuffle(order)
        order.sort(key=lambda i: crops[i].shape[0] * crops[i].shape[1], reverse=True)

        ok = True
        for idx in order:
            h, w = crops[idx].shape
            positions = _candidate_positions(reserved, (h, w))
            if not positions:
                ok = False
                break
            r, c = positions[0]
            reserved[r : r + h, c : c + w] = True
            locs[idx] = (r, c)

        if not ok:
            continue

        grid = np.full((H, W), BG, dtype=int)
        for crop, loc in zip(crops, locs):
            assert loc is not None
            r, c = loc
            h, w = crop.shape
            block = grid[r : r + h, c : c + w]
            mask = crop != BG
            block[mask] = crop[mask]
        return grid
    return None


def _semantic_checks(grid: np.ndarray, expected_transform: str, visible_unmatched: np.ndarray) -> bool:
    crops = _component_crops(grid)
    try:
        transform, unmatched = _infer_transform_and_unmatched(crops)
    except AssertionError:
        return False
    if transform != expected_transform:
        return False
    if not np.array_equal(crops[unmatched], visible_unmatched):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(out, _apply_transform(visible_unmatched, expected_transform)):
        return False

    sigs = Counter(_signature(crop) for crop in crops)
    if sum(count == 1 for count in sigs.values()) < 3:
        return False
    if _signature(out) in sigs:
        return False

    shapes = Counter(tuple(map(int, crop.shape)) for crop in crops)
    color_sets = Counter(
        tuple(sorted({int(v) for v in np.unique(crop) if int(v) != BG}))
        for crop in crops
    )
    unmatched_shape = tuple(map(int, visible_unmatched.shape))
    unmatched_colors = tuple(sorted({int(v) for v in np.unique(visible_unmatched) if int(v) != BG}))
    if shapes[unmatched_shape] < 2:
        return False
    if color_sets[unmatched_colors] < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    transform = random.choice(list(NON_ID_INVOLUTIONS))
    pair_count = random.randint(2, 4)
    if transform in {"transpose", "anti"} and random.random() < 0.75:
        h, w = random.choice([(3, 4), (3, 5), (4, 5)])
    else:
        h = random.randint(3, 5)
        w = random.randint(3, 5)

    palette = tuple(random.sample(PALETTE, random.randint(3, 4)))
    seed_count = pair_count + 1
    used: set[tuple[tuple[int, ...], ...]] = set()
    seeds: list[np.ndarray] = []
    for _ in range(seed_count):
        crop = _gen_seed_crop(h, w, palette, used)
        if crop is None:
            return None
        orbit = _orbit_signatures(crop)
        used.update(orbit.values())
        seeds.append(crop)

    paired = []
    for seed in seeds[:-1]:
        paired.extend([seed, _apply_transform(seed, transform)])

    unmatched_seed = seeds[-1]
    if random.random() < 0.5:
        visible_unmatched = unmatched_seed
    else:
        visible_unmatched = _apply_transform(unmatched_seed, transform)
    if np.array_equal(visible_unmatched, _apply_transform(visible_unmatched, transform)):
        return None

    crops = paired + [visible_unmatched]
    random.shuffle(crops)

    max_h = max(crop.shape[0] for crop in crops)
    max_w = max(crop.shape[1] for crop in crops)
    H = random.randint(max(15, max_h + 4), 28)
    W = random.randint(max(15, max_w + 4), 28)
    grid = _place_crops(crops, H, W)
    if grid is None:
        return None

    comps = _component_crops(grid)
    matched = [crop for crop in comps if np.array_equal(crop, visible_unmatched)]
    if len(matched) != 1:
        return None
    if not _semantic_checks(grid, transform, visible_unmatched):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
