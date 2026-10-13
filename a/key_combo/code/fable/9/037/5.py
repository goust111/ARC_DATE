from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SLAB = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    norm = coords - np.array([r0, c0], dtype=int)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[norm[:, 0], norm[:, 1]] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return mask.shape, tuple(int(x) for x in mask.astype(int).ravel())


def _shape_signature_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return _mask_signature(_coords_to_mask(coords))


def _largest_nonblack_component(grid: np.ndarray) -> tuple[int, np.ndarray]:
    best_color = None
    best_coords = None
    best_area = -1
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != BG):
        for coords in _component_coords(grid == color):
            area = int(coords.shape[0])
            if area > best_area:
                best_area = area
                best_color = color
                best_coords = coords
    assert best_color is not None and best_coords is not None
    return int(best_color), best_coords


def _extract_cavities(grid: np.ndarray, slab_mask: np.ndarray) -> list[np.ndarray]:
    cavities: list[np.ndarray] = []
    slab_set = {tuple(map(int, rc)) for rc in np.argwhere(slab_mask)}
    for coords in _component_coords(grid == BG):
        cells = {tuple(map(int, rc)) for rc in coords}
        if any(_is_border(grid.shape, r, c) for r, c in cells):
            continue
        enclosed = True
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells and (nr, nc) not in slab_set:
                    enclosed = False
                    break
            if not enclosed:
                break
        if enclosed:
            cavities.append(coords)
    return cavities


def _extract_sprites_and_anchors(
    grid: np.ndarray, slab_color: int
) -> tuple[list[tuple[int, np.ndarray]], list[int]]:
    sprites: list[tuple[int, np.ndarray]] = []
    anchors: list[int] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in (BG, slab_color)):
        for coords in _component_coords(grid == color):
            cells = [tuple(map(int, rc)) for rc in coords]
            if coords.shape[0] == 1 and _is_border(grid.shape, cells[0][0], cells[0][1]):
                anchors.append(color)
            else:
                assert all(not _is_border(grid.shape, r, c) for r, c in cells)
                sprites.append((color, coords))
    return sprites, anchors


def _target_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    slab_color, slab_coords = _largest_nonblack_component(g)
    slab_mask = np.zeros(g.shape, dtype=bool)
    slab_mask[slab_coords[:, 0], slab_coords[:, 1]] = True

    cavities = _extract_cavities(g, slab_mask)
    sprites, anchors = _extract_sprites_and_anchors(g, slab_color)

    cavity_masks = [_coords_to_mask(coords) for coords in cavities]
    cavity_keys = [_mask_signature(mask) for mask in cavity_masks]
    sprite_keys = [_shape_signature_from_coords(coords) for _color, coords in sprites]
    sprite_colors = [int(color) for color, _coords in sprites]

    assert len(cavity_keys) >= 2
    assert len(set(cavity_keys)) == len(cavity_keys)
    assert len(set(sprite_keys)) == len(sprite_keys)
    assert len(set(anchors)) == len(anchors)
    assert len(set(sprite_colors)) == len(sprite_colors)
    assert len(sprites) + 1 == len(cavities)
    assert len(anchors) == len(cavities)

    missing_shapes = [i for i, key in enumerate(cavity_keys) if key not in set(sprite_keys)]
    missing_colors = [color for color in anchors if color not in set(sprite_colors)]
    assert len(missing_shapes) == 1
    assert len(missing_colors) == 1

    missing_mask = cavity_masks[missing_shapes[0]]
    out = np.zeros(missing_mask.shape, dtype=int)
    out[missing_mask] = int(missing_colors[0])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the missing sprite by combining the unmatched cavity shape with the unmatched anchor color.

    Input structure:
    - A large monochrome slab contains several enclosed black cavities, each a distinct connected shape.
    - Outside the slab sit monochrome sprite components whose cropped shapes exactly match all but one cavity.
    - Isolated border anchor pixels have distinct colors; the sprite colors match all but one of those anchor colors.

    Rule:
    - Identify the slab and collect the black cavities fully enclosed by it.
    - Match sprite components to cavities by exact cropped shape under translation only, leaving one cavity unmatched.
    - Match sprite colors to border-anchor colors, leaving one anchor color unused.
    - Output the unmatched cavity's tight crop, filling its cells with the unused anchor color and leaving the crop background black.

    Key insight:
    - The answer never appears directly: shape is recovered from the only unpaired hole in the slab, and color is recovered from the only unpaired border anchor.
    """
    return _target_from_grid(grid)


def _grow_shape(area: int) -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, min(5, area))
        w = random.randint(2, min(5, area))
        if h * w < area:
            continue
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask = _coords_to_mask(coords)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if random.random() < 0.6 and mask.all():
            continue
        if mask.sum() < 3:
            continue
        return mask
    raise RuntimeError("failed to grow shape")


def _generate_shapes(count: int, missing_idx: int) -> list[np.ndarray]:
    areas = [random.randint(3, 8) for _ in range(count)]
    if count >= 3 and random.random() < 0.75:
        peer = random.choice([i for i in range(count) if i != missing_idx])
        areas[peer] = areas[missing_idx]

    shapes: list[np.ndarray] = []
    used = set()
    for i in range(count):
        for _ in range(300):
            mask = _grow_shape(areas[i])
            sig = _mask_signature(mask)
            if sig in used:
                continue
            used.add(sig)
            shapes.append(mask)
            break
        else:
            raise RuntimeError("failed to generate unique shapes")
    return shapes


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, require_interior: bool) -> bool:
    h, w = occupied.shape
    cells = np.argwhere(mask)
    for r, c in cells:
        rr = top + int(r)
        cc = left + int(c)
        if not _in_bounds(occupied.shape, rr, cc):
            return False
        if require_interior and _is_border((h, w), rr, cc):
            return False
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if _in_bounds(occupied.shape, nr, nc) and occupied[nr, nc]:
                    return False
    return True


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    for r, c in coords:
        grid[top + int(r), left + int(c)] = int(color)


def _place_masks_in_slab(slab_h: int, slab_w: int, masks: list[np.ndarray]) -> list[tuple[int, int]]:
    occupied = np.zeros((slab_h, slab_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    order = list(range(len(masks)))
    order.sort(key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
    results = [None] * len(masks)
    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        candidates = [
            (r, c)
            for r in range(1, slab_h - mh)
            for c in range(1, slab_w - mw)
            if _can_place(occupied, mask, r, c, require_interior=False)
        ]
        if not candidates:
            raise RuntimeError("failed to place cavity")
        top, left = random.choice(candidates)
        _stamp_mask(occupied, mask, top, left, True)
        results[idx] = (top, left)
    return [(int(r), int(c)) for r, c in results]


def _place_anchors(grid: np.ndarray, occupied: np.ndarray, colors: list[int]) -> None:
    border_cells = []
    h, w = grid.shape
    for c in range(w):
        border_cells.append((0, c))
        border_cells.append((h - 1, c))
    for r in range(1, h - 1):
        border_cells.append((r, 0))
        border_cells.append((r, w - 1))
    random.shuffle(border_cells)

    for color in colors:
        for r, c in border_cells:
            one = np.array([[True]], dtype=bool)
            if _can_place(occupied, one, r, c, require_interior=False):
                grid[r, c] = int(color)
                occupied[r, c] = True
                break
        else:
            raise RuntimeError("failed to place anchor")


def _place_sprites(
    grid: np.ndarray,
    occupied: np.ndarray,
    sprite_specs: list[tuple[int, np.ndarray]],
) -> None:
    h, w = grid.shape
    order = list(range(len(sprite_specs)))
    order.sort(key=lambda i: sprite_specs[i][1].shape[0] * sprite_specs[i][1].shape[1], reverse=True)
    for idx in order:
        color, mask = sprite_specs[idx]
        mh, mw = mask.shape
        candidates = [
            (r, c)
            for r in range(1, h - mh)
            for c in range(1, w - mw)
            if _can_place(occupied, mask, r, c, require_interior=True)
        ]
        if not candidates:
            raise RuntimeError("failed to place sprite")
        top, left = random.choice(candidates)
        _stamp_mask(grid, mask, top, left, color)
        _stamp_mask(occupied, mask, top, left, True)


def _build_expected(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def _semantic_check(grid: np.ndarray, expected: np.ndarray) -> None:
    got = main(grid)
    assert np.array_equal(got, expected)

    slab_color, slab_coords = _largest_nonblack_component(grid)
    assert slab_color == SLAB
    slab_mask = np.zeros(grid.shape, dtype=bool)
    slab_mask[slab_coords[:, 0], slab_coords[:, 1]] = True
    cavities = _extract_cavities(grid, slab_mask)
    sprites, anchors = _extract_sprites_and_anchors(grid, slab_color)

    cavity_keys = [_shape_signature_from_coords(coords) for coords in cavities]
    sprite_keys = [_shape_signature_from_coords(coords) for _color, coords in sprites]
    sprite_colors = [color for color, _coords in sprites]

    assert len(cavities) == len(anchors)
    assert len(sprites) + 1 == len(cavities)
    assert len(set(cavity_keys)) == len(cavity_keys)
    assert len(set(sprite_keys)) == len(sprite_keys)
    assert len(set(anchors)) == len(anchors)
    assert len(set(sprite_colors)) == len(sprite_colors)
    assert len([key for key in cavity_keys if key not in set(sprite_keys)]) == 1
    assert len([color for color in anchors if color not in set(sprite_colors)]) == 1


def _attempt_generate() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(21, 28)
    w = random.randint(21, 28)
    slab_h = random.randint(10, min(16, h - 6))
    slab_w = random.randint(10, min(16, w - 6))
    slab_top = random.randint(3, h - slab_h - 3)
    slab_left = random.randint(3, w - slab_w - 3)

    cavity_count = random.randint(3, 5)
    missing_idx = random.randrange(cavity_count)
    shapes = _generate_shapes(cavity_count, missing_idx)
    anchor_colors = random.sample(PALETTE, cavity_count)
    expected = _build_expected(shapes[missing_idx], anchor_colors[missing_idx])

    grid = np.zeros((h, w), dtype=int)
    grid[slab_top : slab_top + slab_h, slab_left : slab_left + slab_w] = SLAB

    cavity_positions = _place_masks_in_slab(slab_h, slab_w, shapes)
    for mask, (r0, c0) in zip(shapes, cavity_positions):
        _stamp_mask(grid, mask, slab_top + r0, slab_left + c0, BG)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[slab_top : slab_top + slab_h, slab_left : slab_left + slab_w] = True
    _place_anchors(grid, occupied, anchor_colors)

    sprite_specs = [
        (anchor_colors[i], shapes[i])
        for i in range(cavity_count)
        if i != missing_idx
    ]
    random.shuffle(sprite_specs)
    _place_sprites(grid, occupied, sprite_specs)

    _semantic_check(grid, expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, _expected = _attempt_generate()
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate valid input")
