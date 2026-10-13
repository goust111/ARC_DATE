import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the only connected foreground sprite whose exact colored crop appears twice in the input.

    Input structure:
    - The background is the most common color.
    - Foreground cells form several separated 4-connected multicolor sprites.
    - Exactly one sprite pattern, including background holes inside its bbox, occurs twice.

    Rule:
    - Crop each foreground sprite to its tight bounding box.
    - Compare the full colored crops, preserving internal background cells.
    - Return the unique crop that appears exactly twice.

    Key insight: The requested output is selected by repetition of the whole sprite pattern, not by a fixed color, size, or position.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) >= 2, "expected multiple foreground sprites"

    sprites: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    counts: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    for component in components:
        bbox = component["bbox"]
        coords = component["coords"]
        sprite = np.full(crop_bbox(g, bbox).shape, bg, dtype=int)
        r0, c0, _r1, _c1 = bbox
        sprite[coords[:, 0] - r0, coords[:, 1] - c0] = g[coords[:, 0], coords[:, 1]]

        key = (sprite.shape, tuple(int(x) for x in sprite.ravel()))
        sprites[key] = sprite
        counts[key] = counts.get(key, 0) + 1

    duplicate_keys = [key for key, count in counts.items() if count == 2]
    assert len(duplicate_keys) == 1, "expected exactly one sprite pattern to occur twice"
    return sprites[duplicate_keys[0]].copy()

# --------------- generator (o5x) ---------------
# Canvas and sprite count bounds.
CANVAS_H_RANGE, CANVAS_W_RANGE = (10, 15), (13, 15)
DISTRACTOR_COUNT_RANGE = (2, 4)
# Sprite bbox and foreground occupancy bounds.
SPRITE_H_RANGE, SPRITE_W_RANGE, SPRITE_FG_RANGE = (2, 5), (2, 4), (5, 9)
# Palette and per-sprite color bounds.
COLOR_VALUE_RANGE, TOTAL_COLOR_RANGE = (1, 9), (5, 8)
MIN_COLORS_PER_SPRITE, DUPLICATE_COLOR_MAX, DISTRACTOR_COLOR_MAX = 2, 3, 4
# Rejection sampling limit and 4-neighbor geometry.
GENERATION_ATTEMPTS = 300
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_sprite_triples() -> list[tuple[int, int, int]]:
    triples = []
    for h in range(SPRITE_H_RANGE[0], SPRITE_H_RANGE[1] + 1):
        for w in range(SPRITE_W_RANGE[0], SPRITE_W_RANGE[1] + 1):
            lo = max(SPRITE_FG_RANGE[0], h + w - 1)
            hi = min(SPRITE_FG_RANGE[1], h * w - 1)
            for k in range(lo, hi + 1):
                triples.append((h, w, k))
    return triples


def _opposite_corner_walk_options(h: int, w: int):
    return [
        ((0, 0), (h - 1, w - 1), [(1, 0), (0, 1)]),
        ((0, w - 1), (h - 1, 0), [(1, 0), (0, -1)]),
        ((h - 1, 0), (0, w - 1), [(-1, 0), (0, 1)]),
        ((h - 1, w - 1), (0, 0), [(-1, 0), (0, -1)]),
    ]


def _gen_connected_mask(h: int, w: int, k: int) -> np.ndarray:
    (r, c), end, moves = random.choice(_opposite_corner_walk_options(h, w))
    steps = [moves[0]] * abs(end[0] - r) + [moves[1]] * abs(end[1] - c)
    random.shuffle(steps)

    cells = {(r, c)}
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))

    while len(cells) < k:
        frontier = []
        for rr, cc in cells:
            for dr, dc in FOUR_NEIGHBORS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    return mask


def _gen_color_sprite(mask: np.ndarray, colors: set[int]) -> np.ndarray:
    coords = [tuple(rc) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    ordered_colors = list(colors)
    random.shuffle(ordered_colors)

    sprite = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, ordered_colors):
        sprite[r, c] = color
    for r, c in coords[len(ordered_colors) :]:
        sprite[r, c] = random.choice(ordered_colors)
    return sprite


def _gen_color_sets(n_distractors: int) -> list[set[int]]:
    total_colors = random.randint(*TOTAL_COLOR_RANGE)
    palette = random.sample(range(COLOR_VALUE_RANGE[0], COLOR_VALUE_RANGE[1] + 1), total_colors)
    max_counts = [DUPLICATE_COLOR_MAX] + [DISTRACTOR_COLOR_MAX] * n_distractors
    counts = [random.randint(MIN_COLORS_PER_SPRITE, max_counts[i]) for i in range(n_distractors + 1)]
    while sum(counts) < total_colors:
        choices = [i for i, count in enumerate(counts) if count < max_counts[i]]
        counts[random.choice(choices)] += 1

    color_sets = [set() for _ in counts]
    slots = []
    for i, count in enumerate(counts):
        slots.extend([i] * count)
    random.shuffle(slots)
    for color, i in zip(palette, slots):
        color_sets[i].add(color)

    for i, count in enumerate(counts):
        while len(color_sets[i]) < count:
            color_sets[i].add(random.choice(palette))
    return color_sets


def _gen_nonoverlapping_location(
    grid: np.ndarray, sprite: np.ndarray, placed_bboxes: list[tuple[int, int, int, int]]
) -> tuple[int, int] | None:
    H, W = grid.shape
    h, w = sprite.shape
    mask_coords = [tuple(rc) for rc in np.argwhere(sprite != 0)]
    occupied = grid != 0
    candidates = []

    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            r1, c1 = r0 + h - 1, c0 + w - 1
            if any(not (r1 < br0 or br1 < r0 or c1 < bc0 or bc1 < c0) for br0, bc0, br1, bc1 in placed_bboxes):
                continue

            ok = True
            for rr, cc in mask_coords:
                r, c = r0 + rr, c0 + cc
                if occupied[r, c]:
                    ok = False
                    break
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))

    if not candidates:
        return None
    return random.choice(candidates)


def _build_sprites(n_distractors: int) -> tuple[np.ndarray, list[np.ndarray]]:
    triples = _gen_sprite_triples()
    chosen = random.sample(triples, n_distractors + 1)
    duplicate_h, duplicate_w, duplicate_k = chosen[0]
    color_sets = _gen_color_sets(n_distractors)

    duplicate = _gen_color_sprite(_gen_connected_mask(duplicate_h, duplicate_w, duplicate_k), color_sets[0])

    sprites = [duplicate.copy(), duplicate.copy()]
    for (h, w, k), colors in zip(chosen[1:], color_sets[1:]):
        sprites.append(_gen_color_sprite(_gen_connected_mask(h, w, k), colors))

    return duplicate, sprites


def _place_sprites(H: int, W: int, sprites: list[np.ndarray]) -> np.ndarray | None:
    grid = np.zeros((H, W), dtype=int)
    placed_bboxes: list[tuple[int, int, int, int]] = []
    for sprite in sprites:
        loc = _gen_nonoverlapping_location(grid, sprite, placed_bboxes)
        if loc is None:
            return None
        r0, c0 = loc
        paste(grid, sprite, r0, c0, transparent=0)
        h, w = sprite.shape
        placed_bboxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return grid


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*CANVAS_H_RANGE)
    W = random.randint(*CANVAS_W_RANGE)
    n_distractors = random.randint(*DISTRACTOR_COUNT_RANGE)

    duplicate, sprites = _build_sprites(n_distractors)
    random.shuffle(sprites)
    sprites.sort(key=lambda s: int(s.size), reverse=True)

    grid = _place_sprites(H, W, sprites)
    if grid is None:
        return None

    try:
        if np.array_equal(main(grid), duplicate):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
