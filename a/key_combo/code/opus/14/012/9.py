import random
from collections import Counter

import numpy as np


BG = 0
GUIDE = 6
SPRITE_PALETTE = (1, 2, 3, 4, 5)

GRID_H_RANGE = (16, 20)
GRID_W_RANGE = (16, 20)
SPRITE_COUNT_CHOICES = (3, 4)
SELECTED_COLOR_COUNT_CHOICES = (3, 4)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, max(b[0] - a[2] - 1, a[0] - b[2] - 1))
    col_gap = max(0, max(b[1] - a[3] - 1, a[1] - b[3] - 1))
    if row_gap == 0:
        return col_gap
    if col_gap == 0:
        return row_gap
    return min(row_gap, col_gap)


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def find_sprites(grid: np.ndarray) -> list[dict]:
    comps = components_mask((grid != BG) & (grid != GUIDE))
    sprites = []
    for comp in comps:
        box = bbox_from_coords(comp)
        if box[2] - box[0] == 2 and box[3] - box[1] == 2:
            sprites.append({"coords": comp, "bbox": box, "crop": crop_bbox(grid, box)})
    sprites.sort(key=lambda item: item["bbox"])
    return sprites


def ring_block(color: int, scale: int) -> np.ndarray:
    if color == BG:
        return np.zeros((scale, scale), dtype=int)
    block = np.zeros((scale, scale), dtype=int)
    for r in range(scale):
        for c in range(scale):
            depth = min(r, c, scale - 1 - r, scale - 1 - c)
            block[r, c] = color if depth % 2 == 0 else BG
    return block


def upscale_with_rings(sprite: np.ndarray, scale: int) -> np.ndarray:
    h, w = sprite.shape
    out = np.zeros((h * scale, w * scale), dtype=int)
    cache: dict[int, np.ndarray] = {BG: np.zeros((scale, scale), dtype=int)}
    for r in range(h):
        for c in range(w):
            color = int(sprite[r, c])
            if color not in cache:
                cache[color] = ring_block(color, scale)
            out[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = cache[color]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the sparse guide-mask bbox to select one 3x3 sprite, then upscale it by its own color count into concentric rings.

    Input structure:
    - The background is black.
    - Several separated multicolor 3x3 sprites use colors 1-5; some sprite cells may be black holes.
    - Sparse magenta guide pixels form a bbox that fully contains exactly one sprite and no other sprite.

    Rule:
    - Find the guide mask from color 6 and take its bounding box.
    - Find every non-guide sprite whose non-black cells form a compact 3x3 bounded component.
    - Select the unique sprite whose 3x3 bbox is fully contained in the guide bbox, crop that 3x3 sprite, and count its distinct non-black colors N.
    - Replace each original sprite pixel by an N x N block; black pixels stay all black, while non-black blocks are filled as concentric layers with even depths in the original color and odd positive depths in black.

    Key insight:
    - The guide chooses which sprite to crop, and the chosen sprite's own color diversity determines the upscale factor before the concentric recoloring can be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    guide_mask = g == GUIDE
    assert np.any(guide_mask), "expected a guide mask"
    guide_coords = [tuple(map(int, rc)) for rc in np.argwhere(guide_mask)]
    guide_box = bbox_from_coords(guide_coords)

    sprites = find_sprites(g)
    assert sprites, "expected 3x3 sprites"
    selected = [sprite for sprite in sprites if bbox_contains(guide_box, sprite["bbox"])]
    assert len(selected) == 1, "expected exactly one sprite inside the guide bbox"

    sprite = selected[0]["crop"]
    colors = sorted(set(map(int, sprite.flat)) - {BG})
    assert 2 <= len(colors) <= 4
    return upscale_with_rings(sprite, len(colors))


def _precompute_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << 9):
        mask = np.zeros((3, 3), dtype=int)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = 1
        if mask.sum() < 5:
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask == 1)]
        if bbox_from_coords(coords) != (0, 0, 2, 2):
            continue
        if len(components_mask(mask.astype(bool))) != 1:
            continue
        masks.append(mask.astype(bool))
    return masks


VALID_MASKS = _precompute_masks()
SELECTED_MASKS = [mask for mask in VALID_MASKS if int(mask.sum()) <= 8]


def _random_sprite(n_colors: int, *, require_black: bool, rng: random.Random) -> np.ndarray:
    mask_pool = SELECTED_MASKS if require_black else VALID_MASKS
    for _ in range(200):
        mask = rng.choice(mask_pool)
        if require_black and bool(mask.all()):
            continue
        cells = [(r, c) for r in range(3) for c in range(3) if mask[r, c]]
        colors = rng.sample(SPRITE_PALETTE, n_colors)
        sprite = np.zeros((3, 3), dtype=int)
        assigned = cells[:]
        rng.shuffle(assigned)
        for color, (r, c) in zip(colors, assigned):
            sprite[r, c] = color
        for r, c in assigned[n_colors:]:
            sprite[r, c] = rng.choice(colors)
        if len(set(map(int, sprite.flat)) - {BG}) != n_colors:
            continue
        return sprite
    raise RuntimeError("failed to build sprite")


def _guide_perimeter_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _sample_guide_pixels(box: tuple[int, int, int, int], rng: random.Random) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    top = [(r0, c) for c in range(c0, c1 + 1)]
    bottom = [(r1, c) for c in range(c0, c1 + 1)]
    left = [(r, c0) for r in range(r0 + 1, r1)]
    right = [(r, c1) for r in range(r0 + 1, r1)]
    full = _guide_perimeter_cells(box)
    for _ in range(200):
        chosen = {cell for cell in full if rng.random() < 0.52}
        for side in (top, bottom, left, right):
            if side and not any(cell in chosen for cell in side):
                chosen.add(rng.choice(side))
        fill = len(chosen) / len(full)
        if 0.35 <= fill <= 0.7:
            return sorted(chosen)
    return sorted(set(full))


def _sample_positions(
    h: int,
    w: int,
    count: int,
    guide_box: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    candidates = []
    for r in range(h - 2):
        for c in range(w - 2):
            box = (r, c, r + 2, c + 2)
            if bbox_contains(guide_box, box):
                continue
            if not (box[2] < guide_box[0] or box[0] > guide_box[2] or box[3] < guide_box[1] or box[1] > guide_box[3]):
                continue
            if bbox_gap(box, guide_box) < 1:
                continue
            candidates.append(box)
    rng.shuffle(candidates)
    placed: list[tuple[int, int, int, int]] = []
    for box in candidates:
        if all(bbox_gap(box, prev) >= 2 for prev in placed):
            placed.append(box)
            if len(placed) == count:
                return placed
    return None


def _count_sprite_colors(sprite: np.ndarray) -> int:
    return len(set(map(int, sprite.flat)) - {BG})


def _validate_semantics(grid: np.ndarray) -> bool:
    sprites = find_sprites(grid)
    if len(sprites) not in SPRITE_COUNT_CHOICES:
        return False
    crops = [tuple(map(int, sprite["crop"].flat)) for sprite in sprites]
    if len(set(crops)) != len(crops):
        return False

    guide_mask = grid == GUIDE
    if not np.any(guide_mask):
        return False
    guide_box = bbox_from_coords([tuple(map(int, rc)) for rc in np.argwhere(guide_mask)])
    selected = [sprite for sprite in sprites if bbox_contains(guide_box, sprite["bbox"])]
    if len(selected) != 1:
        return False

    counts = [_count_sprite_colors(sprite["crop"]) for sprite in sprites]
    if len(set(counts)) < 2:
        return False

    selected_sprite = selected[0]["crop"]
    scale = _count_sprite_colors(selected_sprite)
    if scale not in SELECTED_COLOR_COUNT_CHOICES:
        return False
    if np.array_equal(upscale_with_rings(selected_sprite, scale), np.kron(selected_sprite, np.ones((scale, scale), dtype=int))):
        return False

    out = main(grid)
    if out.shape != (3 * scale, 3 * scale):
        return False
    if max(out.shape) > 30:
        return False

    perimeter = _guide_perimeter_cells(guide_box)
    if not any(grid[cell] == GUIDE for cell in [(guide_box[0], c) for c in range(guide_box[1], guide_box[3] + 1)]):
        return False
    if not any(grid[cell] == GUIDE for cell in [(guide_box[2], c) for c in range(guide_box[1], guide_box[3] + 1)]):
        return False
    if not any(grid[cell] == GUIDE for cell in [(r, guide_box[1]) for r in range(guide_box[0], guide_box[2] + 1)]):
        return False
    if not any(grid[cell] == GUIDE for cell in [(r, guide_box[3]) for r in range(guide_box[0], guide_box[2] + 1)]):
        return False
    if not all(grid[r, c] in (BG, GUIDE) for r, c in perimeter if not bbox_contains(selected[0]["bbox"], (r, c, r, c))):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    sprite_count = rng.choice(SPRITE_COUNT_CHOICES)
    selected_count = rng.choice(SELECTED_COLOR_COUNT_CHOICES)

    if sprite_count == 3:
        other_counts = [2, 7 - selected_count]
    else:
        other_counts = [2, 7 - selected_count, 2]
    rng.shuffle(other_counts)

    margin_top = rng.randint(1, 3)
    margin_bottom = rng.randint(1, 3)
    margin_left = rng.randint(1, 3)
    margin_right = rng.randint(1, 3)
    guide_h = 3 + margin_top + margin_bottom
    guide_w = 3 + margin_left + margin_right
    if guide_h > h or guide_w > w:
        return None

    gr0 = rng.randint(0, h - guide_h)
    gc0 = rng.randint(0, w - guide_w)
    guide_box = (gr0, gc0, gr0 + guide_h - 1, gc0 + guide_w - 1)
    selected_box = (gr0 + margin_top, gc0 + margin_left, gr0 + margin_top + 2, gc0 + margin_left + 2)

    other_boxes = _sample_positions(h, w, sprite_count - 1, guide_box, rng)
    if other_boxes is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    sprites: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    selected_sprite = _random_sprite(selected_count, require_black=True, rng=rng)
    sprites.append((selected_box, selected_sprite))
    for count, box in zip(other_counts, other_boxes):
        require_black = rng.random() < 0.55
        sprites.append((box, _random_sprite(count, require_black=require_black, rng=rng)))

    for box, sprite in sprites:
        r0, c0, _, _ = box
        grid[r0 : r0 + 3, c0 : c0 + 3] = sprite

    guide_pixels = _sample_guide_pixels(guide_box, rng)
    for r, c in guide_pixels:
        if selected_box[0] <= r <= selected_box[2] and selected_box[1] <= c <= selected_box[3]:
            return None
        grid[r, c] = GUIDE

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
