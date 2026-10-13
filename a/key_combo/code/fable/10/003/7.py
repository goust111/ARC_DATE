from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 2000
PALETTE = list(range(10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def dihedral(arr: np.ndarray, code: int) -> np.ndarray:
    code = int(code)
    if code < 4:
        return np.rot90(arr, code)
    return np.fliplr(np.rot90(arr, code - 4))


def _corner_pairs(side: int) -> list[set[tuple[int, int]]]:
    s = int(side) - 1
    return [
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0)},
        {(0, s), (0, s - 1)},
        {(0, s), (1, s)},
        {(s, 0), (s - 1, 0)},
        {(s, 0), (s, 1)},
        {(s, s), (s - 1, s)},
        {(s, s), (s, s - 1)},
    ]


def _find_anchor_color(patch: np.ndarray, base: int) -> int:
    side = int(patch.shape[0])
    assert patch.shape == (side, side)
    patterns = _corner_pairs(side)
    candidates = []
    for color in sorted(set(map(int, patch.ravel())) - {int(base)}):
        coords = {tuple(map(int, rc)) for rc in np.argwhere(patch == int(color))}
        if len(coords) == 2 and coords in patterns:
            candidates.append(int(color))
    assert len(candidates) == 1, "expected a unique 2-pixel corner anchor color"
    return candidates[0]


def _normalize_patch(patch: np.ndarray, base: int, anchor: int) -> np.ndarray:
    hits = []
    for code in range(8):
        v = dihedral(patch, code)
        if int(v[0, 0]) == int(anchor) and int(v[0, 1]) == int(anchor):
            if int(np.sum(v == int(anchor))) == 2:
                hits.append(v)
    assert len(hits) == 1, "expected a unique normalization"
    return hits[0]


def _extract_layers(grid: np.ndarray) -> tuple[list[np.ndarray], int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg)
    assert len(comps) >= 2, "expected multiple square layers"

    patches = []
    side = None
    base = None
    anchor = None
    marker = None

    for coords in comps:
        box = bbox_coords(coords)
        patch = crop_bbox(g, box)
        h, w = map(int, patch.shape)
        assert h == w, "each layer must be square"
        assert np.all(patch != bg), "layers must be filled squares"

        counts = Counter(map(int, patch.ravel()))
        base_color, base_count = counts.most_common(1)[0]
        assert base_count > 2, "base color must dominate each layer"
        anchor_color = _find_anchor_color(patch, int(base_color))
        norm = _normalize_patch(patch, int(base_color), int(anchor_color))
        extra = sorted(set(map(int, norm.ravel())) - {int(base_color), int(anchor_color)})
        assert len(extra) == 1, "expected one marker color"

        if side is None:
            side = h
            base = int(base_color)
            anchor = int(anchor_color)
            marker = int(extra[0])
        else:
            assert side == h, "all layers must share size"
            assert base == int(base_color), "all layers must share base color"
            assert anchor == int(anchor_color), "all layers must share anchor color"
            assert marker == int(extra[0]), "all layers must share marker color"
        patches.append(norm)

    assert side is not None and base is not None and anchor is not None and marker is not None
    return patches, int(base), int(anchor), int(marker)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize square layers by their anchor arms, then xor-overlay the marker pixels into one board.

    Input structure:
    - The background surrounds several separated filled square layers of the same size.
    - Every layer shares one base color, contains one 2-pixel corner anchor arm in a second color,
      and contains scattered marker pixels of a third color.
    - Different layers may be rotated or mirrored versions of the same coordinate system.

    Rule:
    - For each layer, use the 2-pixel corner arm to normalize orientation so the anchor sits at the
      top-left corner and its short arm points right.
    - Align all normalized layers on the same square board.
    - Output that board filled with the shared base color, keep the normalized anchor arm at the
      top-left, and paint a marker cell exactly where an odd number of normalized layers contain one.

    Key insight:
    - The marker merge is parity-based rather than overwrite-based, and the anchor arm defines the
      only valid coordinate frame before counting odd/even overlaps.
    """
    layers, base, anchor, marker = _extract_layers(np.asarray(grid, dtype=int))
    side = int(layers[0].shape[0])
    out = np.full((side, side), int(base), dtype=int)
    parity = np.zeros((side, side), dtype=bool)
    for patch in layers:
        parity ^= patch == int(marker)

    assert not parity[0, 0] and not parity[0, 1], "markers must not occupy anchor-arm cells"
    out[0, 0] = int(anchor)
    out[0, 1] = int(anchor)
    out[parity] = int(marker)
    return out


def _choose_transforms(count: int) -> list[int]:
    ids = list(range(count))
    transforms = [0] * count
    rot_idx, flip_idx = random.sample(ids, 2)
    transforms[rot_idx] = random.choice([1, 2, 3])
    transforms[flip_idx] = random.choice([4, 5, 6, 7])
    for i in ids:
        if i not in {rot_idx, flip_idx}:
            transforms[i] = random.randrange(8)
    random.shuffle(transforms)
    assert any(t in {1, 2, 3} for t in transforms)
    assert any(t >= 4 for t in transforms)
    return transforms


def _membership_attempt(side: int, count: int) -> tuple[list[np.ndarray], np.ndarray] | None:
    usable = [
        (r, c)
        for r in range(1, side - 1)
        for c in range(1, side - 1)
        if (r, c) not in {(1, 1), (1, side - 2)}
    ]
    random.shuffle(usable)

    memberships: dict[tuple[int, int], set[int]] = {}
    for layer_id in range(count):
        if not usable:
            return None
        pos = usable.pop()
        odd_sizes = [1]
        if count >= 3:
            odd_sizes.append(3)
        if count >= 5:
            odd_sizes.append(5)
        size = random.choice([s for s in odd_sizes if s <= count])
        others = [i for i in range(count) if i != layer_id]
        chosen = {layer_id, *random.sample(others, size - 1)}
        memberships[pos] = chosen

    cancel_count = random.randint(1, min(3, len(usable)))
    for _ in range(cancel_count):
        pos = usable.pop()
        even_sizes = [2]
        if count >= 4:
            even_sizes.append(4)
        size = random.choice([s for s in even_sizes if s <= count])
        memberships[pos] = set(random.sample(range(count), size))

    extra_count = random.randint(0, min(2, len(usable)))
    for _ in range(extra_count):
        pos = usable.pop()
        parity = random.choice(["odd", "even"])
        if parity == "odd":
            sizes = [1]
            if count >= 3:
                sizes.append(3)
            if count >= 5:
                sizes.append(5)
        else:
            sizes = [2]
            if count >= 4:
                sizes.append(4)
        size = random.choice([s for s in sizes if s <= count])
        memberships[pos] = set(random.sample(range(count), size))

    layer_masks = [np.zeros((side, side), dtype=bool) for _ in range(count)]
    for pos, subset in memberships.items():
        r, c = pos
        for layer_id in subset:
            layer_masks[layer_id][r, c] = True

    counts = [int(mask.sum()) for mask in layer_masks]
    if not all(2 <= n <= 5 for n in counts):
        return None

    parity_mask = np.zeros((side, side), dtype=bool)
    for mask in layer_masks:
        parity_mask ^= mask

    out_count = int(parity_mask.sum())
    if out_count < 2 or out_count > 6:
        return None
    if not any(len(subset) % 2 == 0 for subset in memberships.values()):
        return None
    return layer_masks, parity_mask


def _make_canonical_layers(side: int, count: int, base: int, anchor: int, marker: int):
    built = _membership_attempt(side, count)
    if built is None:
        return None
    marker_masks, parity_mask = built
    layers = []
    for mask in marker_masks:
        patch = np.full((side, side), int(base), dtype=int)
        patch[0, 0] = int(anchor)
        patch[0, 1] = int(anchor)
        patch[mask] = int(marker)
        layers.append(patch)
    expected = np.full((side, side), int(base), dtype=int)
    expected[0, 0] = int(anchor)
    expected[0, 1] = int(anchor)
    expected[parity_mask] = int(marker)
    return layers, marker_masks, expected


def _raw_overlay_without_normalization(patches: list[np.ndarray], base: int, anchor: int, marker: int) -> np.ndarray:
    side = int(patches[0].shape[0])
    out = np.full((side, side), int(base), dtype=int)
    parity = np.zeros((side, side), dtype=bool)
    anchor_mask = np.zeros((side, side), dtype=bool)
    for patch in patches:
        parity ^= patch == int(marker)
        anchor_mask |= patch == int(anchor)
    out[anchor_mask] = int(anchor)
    out[parity] = int(marker)
    return out


def _place_layers(layers: list[np.ndarray], bg: int) -> np.ndarray | None:
    side = int(layers[0].shape[0])
    count = len(layers)
    occupied = count * side * side
    canvas_min = max(2 * side + 3, int(math.ceil(math.sqrt(2 * occupied + 24))))
    canvas_max = min(30, canvas_min + 6)
    canvas = random.randint(canvas_min, canvas_max)
    grid = np.full((canvas, canvas), int(bg), dtype=int)
    blocked = np.zeros((canvas, canvas), dtype=bool)
    order = list(range(count))
    random.shuffle(order)

    for idx in order:
        placed = False
        for _ in range(400):
            r = random.randint(1, canvas - side - 1)
            c = random.randint(1, canvas - side - 1)
            rr0 = max(0, r - 1)
            rr1 = min(canvas, r + side + 1)
            cc0 = max(0, c - 1)
            cc1 = min(canvas, c + side + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            grid[r : r + side, c : c + side] = layers[idx]
            blocked[r : r + side, c : c + side] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(5, 7)
    count = random.randint(2, 4)
    colors = random.sample(PALETTE, 4)
    bg, base, anchor, marker = map(int, colors)

    made = _make_canonical_layers(side, count, base, anchor, marker)
    if made is None:
        return None
    canonical_layers, marker_masks, expected = made

    transforms = _choose_transforms(count)
    transformed_layers = [dihedral(layer, code) for layer, code in zip(canonical_layers, transforms)]
    grid = _place_layers(transformed_layers, bg)
    if grid is None:
        return None

    if get_bg(grid) != int(bg):
        return None
    if np.array_equal(_raw_overlay_without_normalization(transformed_layers, base, anchor, marker), expected):
        return None
    if not any(code != 0 for code in transforms):
        return None
    if not all(mask.any() for mask in marker_masks):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-parity ARC input")
