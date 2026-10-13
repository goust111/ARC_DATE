from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BG = 0
FRAME = 2
MARKER = 5
SPRITE_COLORS = (1, 3, 4, 7, 9)
SPRITE_GAP = 2
GEN_RETRIES = 500

_SIZE_BAG: list[int] = []
_COUNT_BAG: list[int] = []


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_component_from_coords(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _expand_self_mask(sprite: np.ndarray) -> np.ndarray:
    sprite = np.asarray(sprite, dtype=int)
    h, w = sprite.shape
    out = np.zeros((h * h, w * w), dtype=int)
    masks = {int(color): np.where(sprite == int(color), int(color), 0) for color in np.unique(sprite) if int(color) != BG}
    for r in range(h):
        for c in range(w):
            color = int(sprite[r, c])
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = masks[color]
    return out


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame_mask = grid == FRAME
    box = _bbox(frame_mask)
    assert box is not None, "expected one red frame"
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"
    border = _rect_border_mask(grid.shape, box)
    assert int(frame_mask.sum()) == int(border.sum()), "frame must be a hollow rectangle"
    assert np.all(frame_mask[border]), "frame border must be complete"
    return box


def _extract_sprites(grid: np.ndarray) -> list[np.ndarray]:
    sprite_mask = ~np.isin(grid, [BG, FRAME, MARKER])
    sprites: list[tuple[int, int, np.ndarray]] = []
    for coords in _components(sprite_mask, connectivity=8):
        crop = _crop_component_from_coords(grid, coords)
        r0, c0 = coords.min(axis=0)
        sprites.append((int(c0), int(r0), crop))
    sprites.sort(key=lambda item: (item[0], item[1]))
    ordered = [crop for _c0, _r0, crop in sprites]
    assert 3 <= len(ordered) <= 5, "expected 3-5 candidate sprites"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count gray markers inside the red frame, use that count to select a sprite, then magnify the selected sprite by self-color masks.

    Input structure:
    - Black background contains one 1-cell-thick red rectangular frame in the upper region.
    - Gray single-pixel markers appear both inside and outside that frame.
    - A lower horizontal row contains 3-5 separated small multicolor sprites ordered left-to-right.

    Rule:
    - Count the gray markers strictly inside the red frame; ignore gray pixels outside the frame.
    - Use that count N as a 1-based left-to-right index into the sprite row and crop the Nth sprite.
    - Replace each pixel of the selected sprite by a block the same shape as the sprite, keeping the pixel color only where the original sprite also had that color and writing black elsewhere.

    Key insight:
    - The frame does not produce the output directly; it only converts distractor-laden marker placement into an index that determines which sprite is then transformed by the self-mask expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    r0, c0, r1, c1 = _find_frame(g)
    inside_count = int(np.count_nonzero(g[r0 + 1 : r1, c0 + 1 : c1] == MARKER))

    sprites = _extract_sprites(g)
    assert 1 <= inside_count <= len(sprites), "inside count must select a valid sprite"
    return _expand_self_mask(sprites[inside_count - 1])


def _bag_pick(options: tuple[int, ...] | list[int], bag: list[int]) -> int:
    if not bag:
        bag.extend(random.sample(list(options), len(options)))
    return int(bag.pop())


def _sample_index(count: int) -> int:
    if count == 1:
        return 1
    roll = random.random()
    if roll < 0.35:
        return 1
    if roll < 0.70:
        return count
    return random.randint(2, count - 1) if count > 2 else random.randint(1, count)


def _sprite_signature(sprite: np.ndarray) -> bytes:
    return sprite.tobytes() + str(sprite.shape).encode()


def _expanded_signature(sprite: np.ndarray) -> bytes:
    expanded = _expand_self_mask(sprite)
    return expanded.tobytes() + str(expanded.shape).encode()


def _random_sprite(size: int, color_set: tuple[int, ...]) -> np.ndarray | None:
    cells = size * size
    need_each = 2 if cells >= 2 * len(color_set) else 1
    flat: list[int] = []
    for color in color_set:
        flat.extend([int(color)] * need_each)
    flat.extend(random.choices(color_set, k=cells - len(flat)))
    random.shuffle(flat)
    sprite = np.array(flat, dtype=int).reshape(size, size)

    if len(np.unique(sprite)) != len(color_set):
        return None
    if len({tuple(row.tolist()) for row in sprite}) == 1:
        return None
    if len({tuple(col.tolist()) for col in sprite.T}) == 1:
        return None
    return sprite


def _generate_sprites(count: int, size: int) -> list[np.ndarray] | None:
    combo_sizes = (2,) if size == 2 else (2, 3)
    color_sets = []
    for combo_size in combo_sizes:
        for combo in itertools.combinations(SPRITE_COLORS, combo_size):
            color_sets.append(tuple(int(c) for c in combo))
    if len(color_sets) < count:
        return None
    for _ in range(100):
        chosen_sets = random.sample(color_sets, count)
        sprites: list[np.ndarray] = []
        seen_sprites: set[bytes] = set()
        seen_expanded: set[bytes] = set()
        ok = True
        for color_set in chosen_sets:
            sprite = None
            for _ in range(80):
                candidate = _random_sprite(size, color_set)
                if candidate is None:
                    continue
                sprite_sig = _sprite_signature(candidate)
                expanded_sig = _expanded_signature(candidate)
                if sprite_sig in seen_sprites or expanded_sig in seen_expanded:
                    continue
                sprite = candidate
                seen_sprites.add(sprite_sig)
                seen_expanded.add(expanded_sig)
                break
            if sprite is None:
                ok = False
                break
            sprites.append(sprite)
        if ok:
            return sprites
    return None


def _place_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    grid[top : top + h, left : left + w] = sprite


def _ring_outside_frame(shape: tuple[int, int], box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = box
    cells = []
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if min(abs(r - r0), abs(r - r1), abs(c - c0), abs(c - c1)) == 0 or (
                r in (r0 - 1, r1 + 1) and c0 - 1 <= c <= c1 + 1
            ) or (
                c in (c0 - 1, c1 + 1) and r0 - 1 <= r <= r1 + 1
            ):
                cells.append((r, c))
    return cells


def _sample_distinct(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if len(cells) < count:
        return None
    return random.sample(cells, count)


def _build_canvas(
    sprites: list[np.ndarray],
    index: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    size = int(sprites[0].shape[0])
    count = len(sprites)
    row_width = count * size + (count - 1) * SPRITE_GAP

    frame_h = random.randint(7, 10)
    frame_w = random.randint(7, 14)
    top_margin = random.randint(1, 2)
    mid_gap = random.randint(3, 5)
    bottom_margin = random.randint(1, 2)
    height = top_margin + frame_h + mid_gap + size + bottom_margin

    width_min = max(frame_w, row_width) + random.randint(2, 6)
    if width_min > 29:
        return None
    width = random.randint(width_min, min(29, width_min + 3))
    if height > 29 or width > 29:
        return None

    grid = np.zeros((height, width), dtype=int)
    frame_top = top_margin
    frame_left = random.randint(1, width - frame_w - 1)
    frame_box = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
    _place_frame(grid, frame_box)

    sprite_top = frame_box[2] + 1 + mid_gap
    sprite_left = random.randint(1, width - row_width - 1)
    sprite_boxes: list[tuple[int, int, int, int]] = []
    cur_left = sprite_left
    for sprite in sprites:
        _paste_sprite(grid, sprite, sprite_top, cur_left)
        sprite_boxes.append((sprite_top, cur_left, sprite_top + size - 1, cur_left + size - 1))
        cur_left += size + SPRITE_GAP

    r0, c0, r1, c1 = frame_box
    inside_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    chosen_inside = _sample_distinct(inside_cells, index)
    if chosen_inside is None:
        return None
    for r, c in chosen_inside:
        grid[r, c] = MARKER

    free_bg = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == BG)
        if not (r0 < int(r) < r1 and c0 < int(c) < c1)
    ]
    near_frame = [cell for cell in _ring_outside_frame(grid.shape, frame_box) if grid[cell] == BG]
    lower_noise = [cell for cell in free_bg if cell[0] >= sprite_top]
    random.shuffle(free_bg)
    outside_count = random.randint(3, 8)
    outside_cells: list[tuple[int, int]] = []

    if near_frame:
        outside_cells.append(random.choice(near_frame))
    if lower_noise and all(cell != outside_cells[0] for cell in lower_noise if outside_cells):
        pick = random.choice([cell for cell in lower_noise if cell not in outside_cells])
        outside_cells.append(pick)

    remaining_pool = [cell for cell in free_bg if cell not in outside_cells]
    if len(remaining_pool) < outside_count - len(outside_cells):
        return None
    outside_cells.extend(random.sample(remaining_pool, outside_count - len(outside_cells)))
    for r, c in outside_cells:
        grid[r, c] = MARKER

    expected = _expand_self_mask(sprites[index - 1])
    return grid, expected


def _attempt() -> np.ndarray | None:
    size = _bag_pick((2, 3, 4), _SIZE_BAG)
    count = _bag_pick((3, 4, 5), _COUNT_BAG)
    if count * size + (count - 1) * SPRITE_GAP + 2 > 29:
        return None

    sprites = _generate_sprites(count, size)
    if sprites is None:
        return None
    index = _sample_index(count)

    built = _build_canvas(sprites, index)
    if built is None:
        return None
    grid, expected = built

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    parsed_sprites = _extract_sprites(grid)
    if len(parsed_sprites) != count:
        return None
    if not all(np.array_equal(a, b) for a, b in zip(parsed_sprites, sprites)):
        return None

    frame_box = _find_frame(grid)
    r0, c0, r1, c1 = frame_box
    inside_count = int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == MARKER))
    if inside_count != index:
        return None
    if int(np.count_nonzero(grid == MARKER)) <= inside_count:
        return None

    expanded_sigs = {_expanded_signature(sprite) for sprite in sprites}
    if len(expanded_sigs) != count:
        return None
    if max(actual.shape) > 30 or max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
