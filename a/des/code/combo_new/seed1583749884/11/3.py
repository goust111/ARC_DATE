from __future__ import annotations

from collections import deque

import numpy as np
import random


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

RAINBOW = (RED, ORANGE, YELLOW, GREEN, BLUE)
PAIR_ORDER = tuple(zip(RAINBOW, RAINBOW[1:]))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 80
GRID_LIMIT = 30


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    ph, pw = padded.shape
    outside = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(ph):
        for c in (0, pw - 1):
            if not padded[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if not padded[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ph and 0 <= nc < pw and not padded[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = (~padded) & (~outside)
    return (padded | holes)[1:-1, 1:-1]


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                out[r, c] = True
                continue
            if any(not mask[r + dr, c + dc] for dr, dc in DIRS4):
                out[r, c] = True
    return out


def _color_coverage(mask: np.ndarray) -> np.ndarray:
    cover = np.zeros_like(mask, dtype=bool)
    for comp in _components(mask):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        sub = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        cover[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] |= _fill_holes(sub)
    return cover


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the regions shared by rainbow-adjacent colors, recoloring them grey.

    Input structure:
    - The background is black.
    - Foreground colors come from the fixed rainbow order Red, Orange, Yellow, Green, Blue.
    - Each color may form one or more 4-connected solid blobs or hollow outlines; a hollow outline stands for the filled region it encloses.

    Rule:
    - Recover every present color's full covered area by filling the enclosed holes of each same-color component.
    - For each adjacent pair in the fixed order Red-Orange-Yellow-Green-Blue, if both colors appear, compute the intersection of their recovered areas.
    - Recolor the union of all such intersections to grey.
    - Erase every other foreground pixel to the black background.

    Key insight:
    - The important pairing is the fixed rainbow order rather than spatial proximity, and hollow outlines still count by the area they cover.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    coverages = {
        color: _color_coverage(g == color)
        for color in RAINBOW
        if np.any(g == color)
    }

    keep = np.zeros_like(g, dtype=bool)
    for a, b in PAIR_ORDER:
        if a in coverages and b in coverages:
            keep |= coverages[a] & coverages[b]

    out = np.full(g.shape, BLACK, dtype=int)
    out[keep] = GREY
    return out


def _random_blob() -> np.ndarray:
    if random.random() < 0.35:
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        return np.ones((h, w), dtype=bool)

    for _ in range(120):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        target = random.randint(max(4, (h * w) // 2), h * w)
        mask = np.zeros((h, w), dtype=bool)

        seed_h = 2 if h >= 2 and random.random() < 0.7 else 1
        seed_w = 2 if w >= 2 and random.random() < 0.7 else 1
        r0 = random.randint(0, h - seed_h)
        c0 = random.randint(0, w - seed_w)
        mask[r0 : r0 + seed_h, c0 : c0 + seed_w] = True

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        trimmed = _trim(mask)
        if trimmed.shape[0] < 2 or trimmed.shape[1] < 2:
            continue
        if not np.array_equal(trimmed, _fill_holes(trimmed)):
            continue
        return trimmed

    return np.ones((3, 3), dtype=bool)


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, k=random.randrange(4))
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _pair_patch(color_a: int, color_b: int) -> np.ndarray:
    core = _random_blob()
    inner = _boundary(core)
    if not inner.any():
        inner = core.copy()

    ch, cw = core.shape
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    h = ch + top + bottom
    w = cw + left + right
    patch = np.full((h, w), BLACK, dtype=int)

    outer_color, inner_color = random.choice(((color_a, color_b), (color_b, color_a)))
    patch[0, :] = outer_color
    patch[-1, :] = outer_color
    patch[:, 0] = outer_color
    patch[:, -1] = outer_color
    patch[top : top + ch, left : left + cw][inner] = inner_color
    return _transform_patch(patch)


def _isolated_patch(color: int) -> np.ndarray:
    shape = _random_blob()
    visible = _boundary(shape) if random.random() < 0.6 else shape
    patch = np.full(shape.shape, BLACK, dtype=int)
    patch[visible] = color
    return _transform_patch(patch)


def _present_colors() -> tuple[int, ...]:
    options: list[tuple[int, ...]] = []
    for mask in range(1, 1 << len(RAINBOW)):
        colors = tuple(RAINBOW[i] for i in range(len(RAINBOW)) if mask & (1 << i))
        if len(colors) < 2:
            continue
        if any(a in colors and b in colors for a, b in PAIR_ORDER):
            options.append(colors)
    return random.choice(options)


def _active_pairs(colors: tuple[int, ...]) -> list[tuple[int, int]]:
    present = set(colors)
    return [(a, b) for a, b in PAIR_ORDER if a in present and b in present]


def _boxes_conflict(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _build_patches(colors: tuple[int, ...]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    pairs = _active_pairs(colors)
    for a, b in pairs:
        patches.append(_pair_patch(a, b))

    paired_colors = {color for pair in pairs for color in pair}
    for color in colors:
        if color not in paired_colors:
            patches.append(_isolated_patch(color))

    extra_count = random.randint(0, min(2, len(colors)))
    for color in random.sample(list(colors), extra_count):
        patches.append(_isolated_patch(color))

    assert patches
    random.shuffle(patches)
    return patches


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    patches = sorted(patches, key=lambda p: p.shape[0] * p.shape[1], reverse=True)
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    total_area = sum(p.shape[0] * p.shape[1] for p in patches)
    base = max(max_h, max_w, int(np.ceil(np.sqrt(total_area))) + 3)

    for _ in range(PLACEMENT_ATTEMPTS):
        h_low = max(base, max_h + 2, 12)
        w_low = max(base, max_w + 2, 12)
        if h_low > GRID_LIMIT or w_low > GRID_LIMIT:
            return None

        h = random.randint(h_low, GRID_LIMIT)
        w = random.randint(w_low, GRID_LIMIT)
        grid = np.full((h, w), BLACK, dtype=int)
        boxes: list[tuple[int, int, int, int]] = []
        ok = True

        for patch in patches:
            ph, pw = patch.shape
            positions = []
            for r in range(h - ph + 1):
                for c in range(w - pw + 1):
                    box = (r, c, r + ph - 1, c + pw - 1)
                    if any(_boxes_conflict(box, other) for other in boxes):
                        continue
                    positions.append((r, c))
            if not positions:
                ok = False
                break

            r, c = random.choice(positions)
            region = patch != BLACK
            grid[r : r + ph, c : c + pw][region] = patch[region]
            boxes.append((r, c, r + ph - 1, c + pw - 1))

        if ok:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        colors = _present_colors()
        patches = _build_patches(colors)
        grid = _place_patches(patches)
        if grid is None:
            continue

        out = main(grid)
        if np.count_nonzero(out == GREY) == 0:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
