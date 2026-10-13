from collections import Counter

import numpy as np

from arc_utils import bbox, crop_bbox, find_components, get_bg, is_contiguous, paste
import random
from itertools import combinations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the 4-connected foreground sprite whose colored pattern occurs only once among otherwise duplicated sprites.

    Input structure
    - The background is the dominant color.
    - Foreground pixels form several separated 4-connected sprites.
    - Every distractor sprite has an identical colored bounding-box crop elsewhere in the grid.

    Rule
    - Crop each 4-connected foreground sprite to its bounding box.
    - Compare the exact colored crops, including background pixels inside each crop.
    - Return the only crop whose exact pattern occurs once.

    Key insight
    - The answer is the outlier sprite by repetition count, not by position, color id, or size alone.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)

    assert len(objects) >= 2

    crops: list[tuple[tuple[tuple[int, ...], ...], np.ndarray]] = []
    for obj in objects:
        crop = crop_bbox(g, obj["bbox"])
        assert int(np.sum(crop != bg)) == int(obj["area"])
        signature = tuple(tuple(map(int, row)) for row in crop)
        crops.append((signature, crop))

    counts = Counter(signature for signature, _ in crops)
    assert len(counts) >= 2
    assert any(n > 1 for n in counts.values())

    unique_signatures = [signature for signature, n in counts.items() if n == 1]
    assert len(unique_signatures) == 1

    matches = [crop for signature, crop in crops if signature == unique_signatures[0]]
    assert len(matches) == 1
    return matches[0].copy()

# --------------- generator (o5x) ---------------

# Canvas and background bounds.
BG_COLOR = 0
CANVAS_MIN_SIZE = 11
CANVAS_MAX_SIZE = 13

# Sprite geometry bounds.
SPRITE_MIN_HEIGHT = 1
SPRITE_MAX_HEIGHT = 5
SPRITE_MIN_WIDTH = 1
SPRITE_MAX_WIDTH = 3
SPRITE_MIN_AREA = 3
SPRITE_MAX_AREA = 6

# Palette and color assignment.
FOREGROUND_COLOR_MIN = 1
FOREGROUND_COLOR_MAX = 9
PALETTE_SIZE_CHOICES = (3, 4)
BASE_SPRITE_COUNT = 3
MAX_COLORS_PER_SPRITE = 2
EXTRA_COLOR_PROB = 0.45

# Two duplicate pairs plus the singleton in base sprite order.
SPRITE_COPY_PATTERN = (0, 0, 1, 1, 2)

# Mask selection and placement behavior.
MASK_FILTER_MODES = ("height", "width", "area", "free")
PLACEMENT_PADDING = 1
INTERIOR_PLACEMENT_FIRST_PROB = 0.65

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_cd3c21df_mask_catalog() -> dict[tuple[int, int, int], list[np.ndarray]]:
    catalog: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for h in range(SPRITE_MIN_HEIGHT, SPRITE_MAX_HEIGHT + 1):
        for w in range(SPRITE_MIN_WIDTH, SPRITE_MAX_WIDTH + 1):
            cells = h * w
            for area in range(SPRITE_MIN_AREA, min(SPRITE_MAX_AREA, cells) + 1):
                key = (h, w, area)
                for flat_coords in combinations(range(cells), area):
                    mask = np.zeros((h, w), dtype=int)
                    for k in flat_coords:
                        mask[k // w, k % w] = 1
                    if bbox(mask.astype(bool)) != (0, 0, h - 1, w - 1):
                        continue
                    if is_contiguous(mask, background=0, connectivity=4):
                        catalog.setdefault(key, []).append(mask.astype(bool))
    return catalog


_GEN_CD3C21DF_MASKS = _gen_cd3c21df_mask_catalog()
_GEN_CD3C21DF_KEYS = list(_GEN_CD3C21DF_MASKS)


def _gen_cd3c21df_choose_mask() -> np.ndarray:
    mode = random.choice(MASK_FILTER_MODES)
    if mode == "height":
        h = random.randint(SPRITE_MIN_HEIGHT, SPRITE_MAX_HEIGHT)
        keys = [key for key in _GEN_CD3C21DF_KEYS if key[0] == h]
    elif mode == "width":
        w = random.randint(SPRITE_MIN_WIDTH, SPRITE_MAX_WIDTH)
        keys = [key for key in _GEN_CD3C21DF_KEYS if key[1] == w]
    elif mode == "area":
        area = random.randint(SPRITE_MIN_AREA, SPRITE_MAX_AREA)
        keys = [key for key in _GEN_CD3C21DF_KEYS if key[2] == area]
    else:
        keys = _GEN_CD3C21DF_KEYS
    key = random.choice(keys)
    return random.choice(_GEN_CD3C21DF_MASKS[key]).copy()


def _gen_cd3c21df_color_sets(palette: list[int]) -> list[tuple[int, ...]]:
    colors = palette[:]
    random.shuffle(colors)
    sets = [{colors[i]} for i in range(BASE_SPRITE_COUNT)]
    for color in colors[BASE_SPRITE_COUNT:]:
        choices = [i for i, s in enumerate(sets) if len(s) < MAX_COLORS_PER_SPRITE]
        sets[random.choice(choices)].add(color)
    for s in sets:
        if len(s) < MAX_COLORS_PER_SPRITE and random.random() < EXTRA_COLOR_PROB:
            choices = [color for color in palette if color not in s]
            if choices:
                s.add(random.choice(choices))
    random.shuffle(sets)
    return [tuple(sorted(s)) for s in sets]


def _gen_cd3c21df_colorize(mask: np.ndarray, colors: tuple[int, ...]) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    cols = list(colors)
    random.shuffle(cols)
    if len(cols) == 1:
        sprite[mask] = cols[0]
        return sprite

    random.shuffle(coords)
    split = random.randint(1, len(coords) - 1)
    first = np.array(coords[:split], dtype=int)
    second = np.array(coords[split:], dtype=int)
    sprite[first[:, 0], first[:, 1]] = cols[0]
    sprite[second[:, 0], second[:, 1]] = cols[1]
    return sprite


def _gen_cd3c21df_signature(sprite: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in sprite)


def _gen_cd3c21df_locations(reserved: np.ndarray, sprite_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = sprite_shape
    H, W = reserved.shape
    locs: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - PLACEMENT_PADDING)
            c0 = max(0, c - PLACEMENT_PADDING)
            r1 = min(H, r + h + PLACEMENT_PADDING)
            c1 = min(W, c + w + PLACEMENT_PADDING)
            if not bool(reserved[r0:r1, c0:c1].any()):
                locs.append((r, c))
    random.shuffle(locs)
    if random.random() < INTERIOR_PLACEMENT_FIRST_PROB:
        locs.sort(key=lambda rc: 0 if 0 < rc[0] and 0 < rc[1] and rc[0] + h < H and rc[1] + w < W else 1)
    return locs


def _gen_cd3c21df_place_sprites(H: int, W: int, sprites: list[np.ndarray]) -> np.ndarray | None:
    reserved = np.zeros((H, W), dtype=int)
    locations: list[tuple[int, int] | None] = [None] * len(sprites)
    order = list(range(len(sprites)))
    random.shuffle(order)
    order.sort(key=lambda i: sprites[i].shape[0] * sprites[i].shape[1], reverse=True)

    def place_at(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        sprite = sprites[idx]
        h, w = sprite.shape
        for r, c in _gen_cd3c21df_locations(reserved, (h, w)):
            reserved[r : r + h, c : c + w] = 1
            locations[idx] = (r, c)
            if place_at(k + 1):
                return True
            locations[idx] = None
            reserved[r : r + h, c : c + w] = 0
        return False

    if not place_at(0):
        return None

    grid = np.zeros((H, W), dtype=int)
    for sprite, loc in zip(sprites, locations):
        assert loc is not None
        paste(grid, sprite, loc[0], loc[1], transparent=BG_COLOR)
    return grid


def _gen_cd3c21df_make_base_sprites(color_sets: list[tuple[int, ...]]) -> list[np.ndarray] | None:
    base_sprites: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, ...], ...]] = set()
    for colors in color_sets:
        sprite = _gen_cd3c21df_colorize(_gen_cd3c21df_choose_mask(), colors)
        sig = _gen_cd3c21df_signature(sprite)
        if sig in signatures:
            return None
        signatures.add(sig)
        base_sprites.append(sprite)
    return base_sprites


def _gen_cd3c21df_passes_oracle(grid: np.ndarray, singleton: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, singleton)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(CANVAS_MIN_SIZE, CANVAS_MAX_SIZE)
    W = random.randint(CANVAS_MIN_SIZE, CANVAS_MAX_SIZE)
    palette = random.sample(range(FOREGROUND_COLOR_MIN, FOREGROUND_COLOR_MAX + 1), random.choice(PALETTE_SIZE_CHOICES))
    color_sets = _gen_cd3c21df_color_sets(palette)

    base_sprites = _gen_cd3c21df_make_base_sprites(color_sets)
    if base_sprites is None:
        return None

    sprites = [base_sprites[i] for i in SPRITE_COPY_PATTERN]
    random.shuffle(sprites)

    grid = _gen_cd3c21df_place_sprites(H, W, sprites)
    if grid is None or not _gen_cd3c21df_passes_oracle(grid, base_sprites[SPRITE_COPY_PATTERN[-1]]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed for cd3c21df")
