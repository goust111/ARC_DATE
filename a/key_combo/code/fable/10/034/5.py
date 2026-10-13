from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    rel = np.asarray(coords, dtype=int) - np.min(coords, axis=0, keepdims=True)
    order = np.lexsort((rel[:, 1], rel[:, 0]))
    return tuple(map(tuple, rel[order]))


def _mask_to_coords(mask: np.ndarray, top: int = 0, left: int = 0) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = _mask_to_coords(mask, top, left)
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _find_hole_rect(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[0, :] == GREY):
            continue
        if not np.all(patch[-1, :] == GREY):
            continue
        if not np.all(patch[:, 0] == GREY):
            continue
        if not np.all(patch[:, -1] == GREY):
            continue
        if np.any((patch != GREY) & (patch != BLACK)):
            continue
        inner_black = np.argwhere(patch[1:-1, 1:-1] == BLACK)
        if inner_black.size == 0:
            continue
        inner_black[:, 0] += r0 + 1
        inner_black[:, 1] += c0 + 1
        candidates.append(((r0, c0, r1, c1), inner_black))
    assert len(candidates) == 1, "expected a unique grey rectangle with a black hole"
    return candidates[0]


def _matching_layer_colors(grid: np.ndarray, shape: tuple[tuple[int, int], ...]) -> list[int]:
    matches: list[int] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BLACK, GREY}):
        coords = np.argwhere(grid == color)
        if coords.shape[0] != len(shape):
            continue
        if _normalize(coords) == shape:
            matches.append(int(color))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the only non-grey color layer whose full pixel set matches the hole inside a grey rectangle, then move that color into the hole.

    Input structure:
    - The grid is a dense multicolor noise field.
    - One grey rectangular block contains a black hole shape strictly inside its border.
    - Elsewhere, one grey connected trap block has the same shape as the hole, while colored noise layers may contain misleading same-shape fragments.

    Rule:
    - Detect the unique grey rectangle whose bounding-box border is fully grey and read the black cells inside it as the hole mask.
    - For each non-black, non-grey color, collect all pixels of that color across the whole grid and compare that full set to the hole mask up to translation.
    - Choose the unique matching color, fill the hole with it, and repaint every original pixel of that color to black.

    Key insight:
    - The correct target is not a connected component; it is the whole layer of one color, so same-shape grey or colored components can be decoys.
    """
    g = np.asarray(grid, dtype=int)
    _, hole_coords = _find_hole_rect(g)
    hole_shape = _normalize(hole_coords)
    matches = _matching_layer_colors(g, hole_shape)
    assert len(matches) == 1, "expected exactly one matching color layer"
    target = matches[0]

    out = g.copy()
    out[out == target] = BLACK
    out[hole_coords[:, 0], hole_coords[:, 1]] = target
    return out


def _random_connected_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        target_area = random.randint(max(4, h + w - 1), h * w - 1)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set(_neighbors4(*start))
        while len(cells) < target_area and frontier:
            choices = [(r, c) for r, c in frontier if 0 <= r < h and 0 <= c < w and (r, c) not in cells]
            if not choices:
                break
            pick = random.choice(choices)
            cells.add(pick)
            frontier.remove(pick)
            for nbr in _neighbors4(*pick):
                if nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != target_area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        if (r0, c0, r1, c1) != (0, 0, h - 1, w - 1):
            continue
        if mask.sum() == h * w:
            continue
        border_count = 2 * h + 2 * w - 4
        if (
            int(mask[0, :].sum())
            + int(mask[-1, :].sum())
            + int(mask[1:-1, 0].sum())
            + int(mask[1:-1, -1].sum())
        ) == border_count:
            continue
        return mask
    raise RuntimeError("failed to generate a connected hole mask")


def _reserve_box(reserved: np.ndarray, box_h: int, box_w: int, padding: int) -> tuple[int, int] | None:
    h, w = reserved.shape
    tops = list(range(h - box_h + 1))
    lefts = list(range(w - box_w + 1))
    random.shuffle(tops)
    random.shuffle(lefts)
    for top in tops:
        for left in lefts:
            r0 = max(0, top - padding)
            c0 = max(0, left - padding)
            r1 = min(h, top + box_h + padding)
            c1 = min(w, left + box_w + padding)
            if np.any(reserved[r0:r1, c0:c1]):
                continue
            reserved[r0:r1, c0:c1] = True
            return top, left
    return None


def _place_extra_pixels(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    count: int,
    forbidden_neighbor_color: int,
) -> bool:
    h, w = grid.shape
    placed = 0
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for r, c in cells:
        if grid[r, c] != -1 or reserved[r, c]:
            continue
        bad = False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == forbidden_neighbor_color:
                bad = True
                break
        if bad:
            continue
        grid[r, c] = color
        reserved[r, c] = True
        placed += 1
        if placed == count:
            return True
    return False


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        rect_bbox, hole_coords = _find_hole_rect(inp)
        hole_shape = _normalize(hole_coords)
        matches = _matching_layer_colors(inp, hole_shape)
        if len(matches) != 1:
            return False
        target = matches[0]
        grey_components = _components(inp == GREY)
        trap_count = 0
        for coords in grey_components:
            if _bbox(coords) == rect_bbox:
                continue
            if _normalize(coords) == hole_shape:
                trap_count += 1
        if trap_count != 1:
            return False

        has_color_component_decoy = False
        for color in sorted(set(map(int, np.unique(inp))) - {BLACK, GREY, target}):
            all_coords = np.argwhere(inp == color)
            if all_coords.shape[0] == len(hole_shape) and _normalize(all_coords) == hole_shape:
                return False
            for comp in _components(inp == color):
                if _normalize(comp) == hole_shape:
                    has_color_component_decoy = True
                    break
        if not has_color_component_decoy:
            return False

        out = main(inp)
        if np.array_equal(inp, out):
            return False
        if not np.all(out[hole_coords[:, 0], hole_coords[:, 1]] == target):
            return False
        original_target = np.argwhere(inp == target)
        if np.any(out[original_target[:, 0], original_target[:, 1]] != BLACK):
            return False
        trap_before = np.zeros_like(inp, dtype=bool)
        trap_after = np.zeros_like(inp, dtype=bool)
        for coords in grey_components:
            if _bbox(coords) != rect_bbox and _normalize(coords) == hole_shape:
                trap_before[coords[:, 0], coords[:, 1]] = True
        grey_after = _components(out == GREY)
        for coords in grey_after:
            if _normalize(coords) == hole_shape:
                trap_after[coords[:, 0], coords[:, 1]] = True
        if not np.array_equal(trap_before, trap_after):
            return False
        return True
    except AssertionError:
        return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    colors = list(PALETTE)
    random.shuffle(colors)
    n_decoys = random.choice([1, 2])
    target_color = colors[0]
    decoy_colors = colors[1 : 1 + n_decoys]
    noise_colors = colors[1 + n_decoys : 4 + n_decoys]
    hole_mask = _random_connected_mask()
    hole_h, hole_w = hole_mask.shape

    margin_top = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_right = random.randint(1, 2)
    rect_h = hole_h + margin_top + margin_bottom
    rect_w = hole_w + margin_left + margin_right

    grid = np.full((h, w), -1, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    rect_pos = _reserve_box(reserved, rect_h, rect_w, padding=2)
    trap_pos = _reserve_box(reserved, hole_h, hole_w, padding=2)
    target_pos = _reserve_box(reserved, hole_h, hole_w, padding=2)
    if rect_pos is None or trap_pos is None or target_pos is None:
        return None

    decoy_positions: list[tuple[int, int]] = []
    for _ in decoy_colors:
        pos = _reserve_box(reserved, hole_h, hole_w, padding=2)
        if pos is None:
            return None
        decoy_positions.append(pos)

    rect_top, rect_left = rect_pos
    grid[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = GREY
    hole_top = rect_top + margin_top
    hole_left = rect_left + margin_left
    hole_abs = _mask_to_coords(hole_mask, hole_top, hole_left)
    grid[hole_abs[:, 0], hole_abs[:, 1]] = BLACK

    trap_top, trap_left = trap_pos
    _place_mask(grid, hole_mask, trap_top, trap_left, GREY)

    target_top, target_left = target_pos
    _place_mask(grid, hole_mask, target_top, target_left, target_color)

    for color, (top, left) in zip(decoy_colors, decoy_positions):
        _place_mask(grid, hole_mask, top, left, color)

    for color in decoy_colors:
        if not _place_extra_pixels(
            grid,
            reserved,
            color=color,
            count=random.randint(1, 3),
            forbidden_neighbor_color=color,
        ):
            return None

    remaining = np.argwhere(grid == -1)
    random.shuffle(remaining)
    fill_prob = random.uniform(0.76, 0.88)
    for r, c in remaining:
        if random.random() < fill_prob:
            grid[r, c] = random.choice(noise_colors)
        else:
            grid[r, c] = BLACK

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scattered-set ARC input")
