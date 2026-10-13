from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 500
PLACE_RETRIES = 200
MAX_SCALE = 12


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_patch(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                out[rr, cc] = True
                break
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask.astype(bool), factor, axis=0), factor, axis=1)


def _separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert cols, "expected a full-height grey separator"
    return int(cols[0])


def _seed_patch(grid: np.ndarray, sep_col: int) -> np.ndarray:
    left = grid[:, :sep_col]
    mask = left != BG
    assert mask.any(), "left region must contain the seed object"
    comps = _components(mask)
    assert len(comps) == 1, "left region must contain exactly one connected seed object"
    comp_mask = np.zeros_like(left, dtype=bool)
    pts = comps[0]
    comp_mask[pts[:, 0], pts[:, 1]] = True
    return _crop_patch(left, comp_mask)


def _sprites_by_color(grid: np.ndarray, sep_col: int) -> dict[int, np.ndarray]:
    right = grid[:, sep_col + 1 :]
    sprites: dict[int, np.ndarray] = {}
    for color in {int(v) for v in np.unique(right) if int(v) != BG}:
        comps = _components(right == color)
        assert len(comps) == 1, "each sprite color must form exactly one component"
        comp_mask = np.zeros_like(right, dtype=bool)
        pts = comps[0]
        comp_mask[pts[:, 0], pts[:, 1]] = True
        sprites[color] = _crop_mask(comp_mask)
    return sprites


def _enclosure_order(seed: np.ndarray) -> list[int]:
    colors = [int(v) for v in np.unique(seed) if int(v) != BG]
    assert len(colors) >= 2
    padded = np.pad(seed, 1, constant_values=BG)
    h, w = padded.shape
    encloses: dict[int, set[int]] = {color: set() for color in colors}

    for color in colors:
        passable = padded != color
        seen = np.zeros((h, w), dtype=bool)
        q: deque[tuple[int, int]] = deque()

        for r in range(h):
            for c in (0, w - 1):
                if passable[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    q.append((r, c))
        for c in range(w):
            for r in (0, h - 1):
                if passable[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

        for other in colors:
            if other == color:
                continue
            if not np.any(seen & (padded == other)):
                encloses[color].add(other)

    indegree = {color: 0 for color in colors}
    for outer, inner_set in encloses.items():
        for inner in inner_set:
            indegree[inner] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert ready, "seed colors must form an enclosure chain"
        ready.sort(key=lambda color: (-len(encloses[color]), colors.index(color)))
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for inner in encloses[color]:
            indegree[inner] -= 1

    assert len(order) == len(colors)
    return order


def _solid_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = color
    return out


def _wrap_with_frame(mask: np.ndarray, color: int, inner_patch: np.ndarray) -> np.ndarray:
    occupied = inner_patch != BG
    ph, pw = inner_patch.shape
    for scale in range(2, MAX_SCALE + 1):
        scaled = _scale_mask(mask, scale)
        sh, sw = scaled.shape
        if ph > sh or pw > sw:
            continue
        boundary = _boundary(scaled)
        interior = scaled & ~boundary
        top = (sh - ph) // 2
        left = (sw - pw) // 2
        if np.any(occupied & ~interior[top : top + ph, left : left + pw]):
            continue

        out = np.zeros((sh, sw), dtype=int)
        out[boundary] = color
        window = out[top : top + ph, left : left + pw]
        window[occupied] = inner_patch[occupied]
        return out
    raise AssertionError("could not scale a sprite enough to wrap the inner layers")


def _build_output(order: list[int], sprites: dict[int, np.ndarray]) -> np.ndarray:
    assert order, "need at least one color layer"
    patch = _solid_sprite(sprites[order[-1]], order[-1])
    for color in reversed(order[:-1]):
        patch = _wrap_with_frame(sprites[color], color, patch)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a seed color-order and rebuild it as nested hollow frames made from same-color sprites.

    Input structure:
    - The background is black, and one full-height grey column splits the grid into a left seed region and a right sprite region.
    - The left region contains one connected multicolor seed object whose colors form a strict outside-to-inside enclosure chain.
    - The right region contains one isolated monochrome sprite per seed color, each a small connected 2x2 to 4x4 shape.

    Rule:
    - Read the seed object's color nesting order from outermost to innermost.
    - Match every seed color to the monochrome sprite of the same color on the right.
    - Draw the innermost layer as that sprite's original solid shape.
    - For each remaining layer, enlarge its sprite by the smallest integer scale factor greater than 1 whose centered interior can contain everything already built, then keep only the one-pixel outer boundary of that scaled shape and center the previous inner layers inside it.
    - Return only the rebuilt nested structure; discard the separator, the seed object, and the loose sprites.

    Key insight:
    - The seed object contributes only a color containment order, while the visible geometry comes from the separate same-color sprites; solving the task requires combining those two clues.
    """
    g = np.asarray(grid, dtype=int)
    sep_col = _separator_col(g)
    seed = _seed_patch(g, sep_col)
    order = _enclosure_order(seed)
    sprites = _sprites_by_color(g, sep_col)
    assert set(order) == set(sprites), "seed colors and sprite colors must match exactly"
    return _build_output(order, sprites)


def _diamond(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    center = radius
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) <= radius:
                mask[r, c] = True
    return mask


def _shape_library() -> list[np.ndarray]:
    return [
        np.ones((2, 2), dtype=bool),
        np.ones((2, 3), dtype=bool),
        np.ones((3, 2), dtype=bool),
        np.ones((2, 4), dtype=bool),
        np.ones((4, 2), dtype=bool),
        np.ones((3, 3), dtype=bool),
        np.ones((3, 4), dtype=bool),
        np.ones((4, 3), dtype=bool),
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 0],
            ],
            dtype=bool,
        ),
    ]


def _random_seed(colors: list[int], rng: random.Random) -> np.ndarray:
    layers = len(colors)
    if rng.random() < 0.5:
        if layers == 2:
            inner_h = rng.randint(1, 3)
            inner_w = rng.randint(1, 3)
        elif layers == 3:
            inner_h = rng.randint(1, 2)
            inner_w = rng.randint(1, 2)
        else:
            inner_h = 1
            inner_w = 1
        mask = np.ones((inner_h + 2 * (layers - 1), inner_w + 2 * (layers - 1)), dtype=bool)
    else:
        inner_radius = 0 if layers >= 3 else rng.randint(0, 1)
        mask = _diamond(inner_radius + layers - 1)

    seed = np.zeros(mask.shape, dtype=int)
    remaining = mask.copy()
    for color in colors[:-1]:
        ring = _boundary(remaining)
        seed[ring] = color
        remaining &= ~ring
    seed[remaining] = colors[-1]
    return seed


def _sample_sprites(colors: list[int], rng: random.Random) -> dict[int, np.ndarray]:
    library = _shape_library()
    chosen = rng.sample(library, len(colors))
    rng.shuffle(chosen)
    return {color: chosen[idx].copy() for idx, color in enumerate(colors)}


def _place_colored_masks(
    height: int,
    width: int,
    items: list[tuple[int, np.ndarray]],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    order = items[:]
    rng.shuffle(order)

    for color, mask in order:
        mh, mw = mask.shape
        if mh > height or mw > width:
            return None

        positions: list[tuple[int, int]] = []
        for r0 in range(height - mh + 1):
            for c0 in range(width - mw + 1):
                if np.any(blocked[r0 : r0 + mh, c0 : c0 + mw] & mask):
                    continue
                positions.append((r0, c0))
        if not positions:
            return None

        r0, c0 = rng.choice(positions)
        patch = np.zeros_like(grid, dtype=bool)
        patch[r0 : r0 + mh, c0 : c0 + mw][mask] = True
        grid[patch] = color
        blocked |= _dilate(patch)
    return grid


def _random_right_region(sprites: dict[int, np.ndarray], rng: random.Random) -> np.ndarray | None:
    items = list(sprites.items())
    max_h = max(mask.shape[0] for _, mask in items)
    max_w = max(mask.shape[1] for _, mask in items)
    area = sum(int(mask.sum()) for _, mask in items)

    min_h = max_h + 2
    min_w = max_w + 2
    max_canvas_h = min(14, max(min_h, min_h + len(items) + 2))
    max_canvas_w = min(15, max(min_w, min_w + len(items) * 2 + 1))

    for _ in range(PLACE_RETRIES):
        h = rng.randint(min_h, max_canvas_h)
        w = rng.randint(min_w, max_canvas_w)
        if h * w < area + 4 * len(items):
            continue
        placed = _place_colored_masks(h, w, items, rng)
        if placed is not None:
            return placed
    return None


def _embed_seed(seed: np.ndarray, rng: random.Random) -> np.ndarray:
    sh, sw = seed.shape
    canvas_h = sh + rng.randint(2, 5)
    canvas_w = sw + rng.randint(2, 5)
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    top = rng.randint(1, canvas_h - sh - 1)
    left = rng.randint(1, canvas_w - sw - 1)
    canvas[top : top + sh, left : left + sw] = seed
    return canvas


def _compose_input(left: np.ndarray, right: np.ndarray, rng: random.Random) -> np.ndarray | None:
    h = max(left.shape[0], right.shape[0]) + rng.randint(0, 2)
    w = left.shape[1] + 1 + right.shape[1]
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    sep_col = left.shape[1]
    grid[:, sep_col] = SEP

    left_top = rng.randint(0, h - left.shape[0])
    right_top = rng.randint(0, h - right.shape[0])
    grid[left_top : left_top + left.shape[0], : sep_col] = left
    grid[right_top : right_top + right.shape[0], sep_col + 1 :] = right
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.choice((2, 3, 4))
    colors = rng.sample(COLOR_POOL, layer_count)
    seed = _random_seed(colors, rng)
    sprites = _sample_sprites(colors, rng)

    try:
        expected = _build_output(colors, sprites)
    except AssertionError:
        return None

    if max(expected.shape) > 30:
        return None

    left = _embed_seed(seed, rng)
    right = _random_right_region(sprites, rng)
    if right is None:
        return None

    grid = _compose_input(left, right, rng)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if grid.ndim != 2 or max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-sprite puzzle")
