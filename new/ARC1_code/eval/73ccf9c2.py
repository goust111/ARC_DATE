from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, is_contiguous, paste
import math
import random
from collections import deque


def _component_crop(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    crop[rr, cc] = grid[coords[:, 0], coords[:, 1]]
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique foreground object whose mask is not mirror-symmetric left-to-right.

    Input structure
    - A dominant background surrounds several same-color 8-connected foreground objects.
    - Exactly one foreground color appears.
    - All but one foreground object are mirror-symmetric left-to-right inside their tight bboxes.

    Rule
    - Split the non-background pixels into 8-connected objects.
    - Build each object's tight bbox crop over the background.
    - Return the single crop whose foreground mask differs from its left-right reflection.

    Key insight
    - The input is a set of candidate sprites, and vertical-axis asymmetry identifies which one to extract regardless of color or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected one foreground color"

    asymmetric_crops: list[np.ndarray] = []
    for coords in components_mask(g != bg, connectivity=8):
        crop = _component_crop(g, coords, bg)
        mask = crop != bg
        if not np.array_equal(mask, np.fliplr(mask)):
            asymmetric_crops.append(crop)

    assert len(asymmetric_crops) == 1, "expected exactly one left-right-asymmetric object"
    return asymmetric_crops[0]

# --------------- generator (o5x) ---------------

# Canvas and palette.
BACKGROUND_COLOR = 0
CANVAS_SIZE_MIN = 20
CANVAS_SIZE_MAX = 23
FOREGROUND_COLORS = tuple(range(1, 10))

# Sprite bbox and fill constraints.
SPRITE_HEIGHT_MIN = 3
SPRITE_HEIGHT_MAX = 7
SPRITE_WIDTH_MIN = 4
SPRITE_WIDTH_MAX = 9
SPRITE_AREA_MIN = 15
SPRITE_AREA_MAX = 36
SPRITE_FILL_MIN = 0.40
SPRITE_FILL_MAX = 0.66

# Object counts and placement spacing.
SYMMETRIC_OBJECT_COUNT = 3
BBOX_GAP = 2

# Shape construction choices.
MIRROR_ORBIT_SINGLE_WEIGHT = 1
MIRROR_ORBIT_PAIR_WEIGHT = 2
CONNECT_START_LIMIT = 5
RANDOM_FLIP_PROB = 0.5
CARDINAL_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
EDGE_NAMES = ("top", "bottom", "left", "right")

# Generation retry limit.
MAX_GENERATION_ATTEMPTS = 300


def _gen_area_bounds(h: int, w: int) -> tuple[int, int]:
    cells = int(h) * int(w)
    lo = max(SPRITE_AREA_MIN, math.ceil(SPRITE_FILL_MIN * cells))
    hi = min(SPRITE_AREA_MAX, math.floor(SPRITE_FILL_MAX * cells))
    return lo, hi


def _gen_valid_dim_pairs() -> tuple[tuple[int, int], ...]:
    pairs = []
    for h in range(SPRITE_HEIGHT_MIN, SPRITE_HEIGHT_MAX + 1):
        for w in range(SPRITE_WIDTH_MIN, SPRITE_WIDTH_MAX + 1):
            lo, hi = _gen_area_bounds(h, w)
            if lo <= hi:
                pairs.append((h, w))
    return tuple(pairs)


VALID_SPRITE_DIMS = _gen_valid_dim_pairs()


def _gen_pick_dims() -> tuple[int, int]:
    pairs = VALID_SPRITE_DIMS
    if random.random() < 0.5:
        heights = sorted({h for h, _w in pairs})
        h = random.choice(heights)
        w = random.choice([ww for hh, ww in pairs if hh == h])
    else:
        widths = sorted({w for _h, w in pairs})
        w = random.choice(widths)
        h = random.choice([hh for hh, ww in pairs if ww == w])
    return int(h), int(w)


def _gen_orbit_weight(c: int, w: int) -> int:
    return MIRROR_ORBIT_SINGLE_WEIGHT if c == w - 1 - c else MIRROR_ORBIT_PAIR_WEIGHT


def _gen_half_area(half: np.ndarray, w: int) -> int:
    weights = np.array([_gen_orbit_weight(c, w) for c in range(half.shape[1])], dtype=int)
    return int((half * weights).sum())


def _gen_frontier_cells(mask: np.ndarray, forbidden: tuple[int, int] | None = None) -> set[tuple[int, int]]:
    """Return unfilled 8-neighbor cells around the current mask."""
    h, w = mask.shape
    candidates = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    if forbidden is None or (nr, nc) != forbidden:
                        candidates.add((nr, nc))
    return candidates


def _gen_sprite_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BACKGROUND_COLOR, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _gen_mark_half_path(half: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r, c = a
    tr, tc = b
    half[r, c] = True
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        half[r, c] = True


def _gen_grow_half(half: np.ndarray, w: int, target: int) -> bool:
    while _gen_half_area(half, w) < target:
        area = _gen_half_area(half, w)
        candidates = {
            (r, c)
            for r, c in _gen_frontier_cells(half)
            if _gen_orbit_weight(c, w) <= target - area
        }
        if not candidates:
            return False

        pair_candidates = [
            p for p in candidates if _gen_orbit_weight(p[1], w) == MIRROR_ORBIT_PAIR_WEIGHT
        ]
        single_candidates = [
            p for p in candidates if _gen_orbit_weight(p[1], w) == MIRROR_ORBIT_SINGLE_WEIGHT
        ]
        remaining = target - area
        if remaining % MIRROR_ORBIT_PAIR_WEIGHT == 0 and pair_candidates:
            nr, nc = random.choice(pair_candidates)
        elif single_candidates:
            nr, nc = random.choice(single_candidates)
        else:
            nr, nc = random.choice(list(candidates))
        half[nr, nc] = True
    return True


def _gen_full_from_half(half: np.ndarray, w: int) -> np.ndarray:
    h, q = half.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :q] |= half
    mask[:, w - q :] |= np.fliplr(half)
    return mask


def _gen_symmetric_sprite(color: int) -> np.ndarray | None:
    h, w = _gen_pick_dims()
    q = (w + 1) // 2
    half = np.zeros((h, q), dtype=bool)

    bridge = (random.randrange(h), q - 1)
    anchors = [
        (random.randrange(h), 0),
        (0, random.randrange(q)),
        (h - 1, random.randrange(q)),
    ]
    for anchor in anchors:
        _gen_mark_half_path(half, bridge, anchor)

    lo, hi = _gen_area_bounds(h, w)
    base = _gen_half_area(half, w)
    targets = [
        a
        for a in range(max(lo, base), hi + 1)
        if (a - base) % MIRROR_ORBIT_PAIR_WEIGHT == 0
    ]
    if not targets:
        return None

    target = random.choice(targets)
    if not _gen_grow_half(half, w, target):
        return None

    mask = _gen_full_from_half(half, w)
    if not np.array_equal(mask, np.fliplr(mask)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=8):
        return None

    return _gen_sprite_from_mask(mask, color)


def _gen_random_edge_cell(h: int, w: int, edge: str, forbidden: tuple[int, int]) -> tuple[int, int]:
    if edge == "top":
        cells = [(0, c) for c in range(w)]
    elif edge == "bottom":
        cells = [(h - 1, c) for c in range(w)]
    elif edge == "left":
        cells = [(r, 0) for r in range(h)]
    else:
        cells = [(r, w - 1) for r in range(h)]
    cells = [p for p in cells if p != forbidden]
    return random.choice(cells)


def _gen_bfs_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    forbidden: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        moves = list(CARDINAL_DELTAS)
        random.shuffle(moves)
        for dr, dc in moves:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if nxt == forbidden or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)
    return None


def _gen_connect_point(mask: np.ndarray, point: tuple[int, int], forbidden: tuple[int, int]) -> bool:
    if mask[point]:
        return True

    filled = [tuple(map(int, p)) for p in np.argwhere(mask)]
    filled.sort(key=lambda p: abs(p[0] - point[0]) + abs(p[1] - point[1]))
    starts = filled[: min(CONNECT_START_LIMIT, len(filled))]
    random.shuffle(starts)

    for start in starts:
        path = _gen_bfs_path(mask.shape, start, point, forbidden)
        if path is not None:
            for r, c in path:
                mask[r, c] = True
            return True
    return False


def _gen_grow_mask(mask: np.ndarray, forbidden: tuple[int, int], target: int) -> bool:
    while int(mask.sum()) < target:
        candidates = _gen_frontier_cells(mask, forbidden)
        if not candidates:
            return False
        r, c = random.choice(list(candidates))
        mask[r, c] = True
    return True


def _gen_asymmetric_sprite(color: int) -> np.ndarray | None:
    h, w = _gen_pick_dims()
    mask = np.zeros((h, w), dtype=bool)

    witness = (random.randrange(h), random.randrange(w // 2))
    forbidden = (witness[0], w - 1 - witness[1])
    core_choices = [(r, c) for r in range(h) for c in range(w) if (r, c) != forbidden]
    core = random.choice(core_choices)
    mask[core] = True

    points = [witness] + [_gen_random_edge_cell(h, w, edge, forbidden) for edge in EDGE_NAMES]
    random.shuffle(points)
    for point in points:
        if not _gen_connect_point(mask, point, forbidden):
            return None

    lo, hi = _gen_area_bounds(h, w)
    base = int(mask.sum())
    targets = list(range(max(lo, base), hi + 1))
    if not targets:
        return None
    target = random.choice(targets)
    if not _gen_grow_mask(mask, forbidden, target):
        return None

    if random.random() < RANDOM_FLIP_PROB:
        mask = np.flipud(mask)
    if random.random() < RANDOM_FLIP_PROB:
        mask = np.fliplr(mask)

    if np.array_equal(mask, np.fliplr(mask)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=8):
        return None

    return _gen_sprite_from_mask(mask, color)


def _gen_bboxes_far_enough(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = BBOX_GAP,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return bool(ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _gen_valid_positions(
    n: int,
    sprite: np.ndarray,
    placed: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = sprite.shape
    out = []
    for r in range(n - h + 1):
        for c in range(n - w + 1):
            box = (r, c, r + h - 1, c + w - 1)
            if all(_gen_bboxes_far_enough(box, other) for other in placed):
                out.append((r, c))
    random.shuffle(out)
    return out


def _gen_choose_canvas_and_color() -> tuple[int, int]:
    n = random.randint(CANVAS_SIZE_MIN, CANVAS_SIZE_MAX)
    color = random.choice(FOREGROUND_COLORS)
    return n, int(color)


def _gen_build_sprites(color: int) -> tuple[list[np.ndarray], np.ndarray] | None:
    symmetric = []
    for _ in range(SYMMETRIC_OBJECT_COUNT):
        sprite = _gen_symmetric_sprite(color)
        if sprite is None:
            return None
        symmetric.append(sprite)

    asymmetric = _gen_asymmetric_sprite(color)
    if asymmetric is None:
        return None
    return symmetric, asymmetric


def _gen_place_sprites(n: int, sprites: list[np.ndarray]) -> np.ndarray | None:
    items = list(sprites)
    random.shuffle(items)

    canvas = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    for sprite in items:
        positions = _gen_valid_positions(n, sprite, placed)
        if not positions:
            return None
        r, c = random.choice(positions)
        paste(canvas, sprite, r, c, transparent=BACKGROUND_COLOR)
        placed.append((r, c, r + sprite.shape[0] - 1, c + sprite.shape[1] - 1))
    return canvas


def _gen_oracle_accepts(canvas: np.ndarray, expected: np.ndarray) -> bool:
    try:
        observed = main(canvas)
    except AssertionError:
        return False
    return bool(np.array_equal(observed, expected))


def _gen_attempt() -> np.ndarray | None:
    n, color = _gen_choose_canvas_and_color()
    sprites = _gen_build_sprites(color)
    if sprites is None:
        return None

    symmetric, asymmetric = sprites
    canvas = _gen_place_sprites(n, symmetric + [asymmetric])
    if canvas is None:
        return None
    if not _gen_oracle_accepts(canvas, asymmetric):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 73ccf9c2")
