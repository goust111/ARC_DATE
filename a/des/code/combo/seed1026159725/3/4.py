from __future__ import annotations

from collections import Counter, deque

import numpy as np
import random


RED = 2
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400

# Sprite / canvas variability.
SPRITE_SIDE_MIN = 4
SPRITE_SIDE_MAX = 6
COPY_COUNT_MIN = 3
COPY_COUNT_MAX = 6
PALETTE_SIZE_MIN = 2
PALETTE_SIZE_MAX = 4
SLOT_GAP_MIN = 1
SLOT_GAP_MAX = 2
MARGIN_MIN = 1
MARGIN_MAX = 2
LAYOUT_ROWS_MAX = 3
LAYOUT_COLS_MAX = 3


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _extract_sprite_patches(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        patches.append(patch)
    assert len(patches) >= 3, "need at least three sprite copies"
    shapes = {patch.shape for patch in patches}
    assert len(shapes) == 1, "all sprite copies must share one bbox shape"
    return patches


def _consensus_template(patches: list[np.ndarray]) -> np.ndarray:
    count = len(patches)
    h, w = patches[0].shape
    stack = np.stack(patches, axis=0)
    consensus = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            order = np.argsort(-counts)
            assert counts[order[0]] > count // 2, "local majority must be unique"
            consensus[r, c] = int(values[order[0]])
    return consensus


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a set of translated same-sized sprite copies, find the one modified copy
    and output a mask of the modified local cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    assert bg != RED, "background cannot be the red output color"

    patches = _extract_sprite_patches(g, bg)
    consensus = _consensus_template(patches)

    diff_masks = [patch != consensus for patch in patches]
    anomalous = [mask for mask in diff_masks if bool(mask.any())]
    assert len(anomalous) == 1, "there must be exactly one modified sprite"

    out = np.full(consensus.shape, bg, dtype=int)
    out[anomalous[0]] = RED
    return out


def _choose_layout(count: int) -> tuple[int, int]:
    layouts = [
        (rows, cols)
        for rows in range(1, LAYOUT_ROWS_MAX + 1)
        for cols in range(1, LAYOUT_COLS_MAX + 1)
        if rows * cols >= count
    ]
    return random.choice(layouts)


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_support_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    min_area = max(height + width - 1, int(np.ceil(0.45 * area)))
    max_area = max(min_area, int(np.floor(0.80 * area)))
    for _ in range(150):
        target_area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(height, width, target_area)
        if not mask[0, :].any():
            continue
        if not mask[-1, :].any():
            continue
        if not mask[:, 0].any():
            continue
        if not mask[:, -1].any():
            continue
        if mask.all():
            continue
        return mask
    return None


def _assign_template_colors(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    h, w = mask.shape
    support = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    seeds = random.sample(support, len(palette))
    arr = np.full((h, w), -1, dtype=int)
    for color, seed in zip(palette, seeds):
        arr[seed] = color

    color_of_seed = {seed: color for color, seed in zip(palette, seeds)}
    for r, c in support:
        best_dist = None
        best_colors: list[int] = []
        for sr, sc in seeds:
            dist = abs(r - sr) + abs(c - sc)
            color = color_of_seed[(sr, sc)]
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_colors = [color]
            elif dist == best_dist:
                best_colors.append(color)
        arr[r, c] = random.choice(best_colors)
    return arr


def _make_template(height: int, width: int, bg: int, palette: list[int]) -> np.ndarray | None:
    support = _make_support_mask(height, width)
    if support is None:
        return None
    color_mask = _assign_template_colors(support, palette)
    template = np.full((height, width), bg, dtype=int)
    template[support] = color_mask[support]
    return template


def _place_copies(
    template: np.ndarray,
    count: int,
    bg: int,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    height, width = template.shape
    rows, cols = _choose_layout(count)
    gap_y = random.randint(SLOT_GAP_MIN, SLOT_GAP_MAX)
    gap_x = random.randint(SLOT_GAP_MIN, SLOT_GAP_MAX)
    margin_top = random.randint(MARGIN_MIN, MARGIN_MAX)
    margin_bottom = random.randint(MARGIN_MIN, MARGIN_MAX)
    margin_left = random.randint(MARGIN_MIN, MARGIN_MAX)
    margin_right = random.randint(MARGIN_MIN, MARGIN_MAX)

    canvas_h = margin_top + rows * (height + gap_y) + margin_bottom
    canvas_w = margin_left + cols * (width + gap_x) + margin_right
    grid = np.full((canvas_h, canvas_w), bg, dtype=int)

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen_slots = random.sample(slots, count)
    tops_lefts: list[tuple[int, int]] = []
    for slot_r, slot_c in chosen_slots:
        oy = random.randint(0, gap_y - 1)
        ox = random.randint(0, gap_x - 1)
        top = margin_top + slot_r * (height + gap_y) + oy
        left = margin_left + slot_c * (width + gap_x) + ox
        patch = grid[top : top + height, left : left + width]
        assert np.all(patch == bg)
        sprite_mask = template != bg
        patch[sprite_mask] = template[sprite_mask]
        tops_lefts.append((top, left))

    if _mode_color(grid) != bg:
        return None
    return grid, tops_lefts


def _mutate_template(
    template: np.ndarray,
    bg: int,
    palette: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    support = [tuple(map(int, cell)) for cell in np.argwhere(template != bg)]
    if not support:
        return None
    max_changes = min(5, max(1, len(support) // 3))
    change_count = random.randint(1, max_changes)
    changed = random.sample(support, change_count)

    mutated = template.copy()
    diff_mask = np.zeros(template.shape, dtype=bool)
    for r, c in changed:
        options = [color for color in palette if color != int(template[r, c])]
        if not options:
            return None
        mutated[r, c] = random.choice(options)
        diff_mask[r, c] = True
    return mutated, diff_mask


def _expected_output(shape: tuple[int, int], bg: int, diff_mask: np.ndarray) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[diff_mask] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [color for color in range(10) if color != RED]
    bg = random.choice(bg_choices)

    height = random.randint(SPRITE_SIDE_MIN, SPRITE_SIDE_MAX)
    width = random.randint(SPRITE_SIDE_MIN, SPRITE_SIDE_MAX)
    copy_count = random.randint(COPY_COUNT_MIN, COPY_COUNT_MAX)

    color_pool = [color for color in range(10) if color not in {bg, RED}]
    palette_size = random.randint(PALETTE_SIZE_MIN, min(PALETTE_SIZE_MAX, len(color_pool)))
    palette = random.sample(color_pool, palette_size)

    template = _make_template(height, width, bg, palette)
    if template is None:
        return None

    placed = _place_copies(template, copy_count, bg)
    if placed is None:
        return None
    grid, locations = placed

    modified_index = random.randrange(copy_count)
    mutated = _mutate_template(template, bg, palette)
    if mutated is None:
        return None
    mutated_patch, diff_mask = mutated

    top, left = locations[modified_index]
    sprite_mask = template != bg
    grid[top : top + height, left : left + width][sprite_mask] = mutated_patch[sprite_mask]

    expected = _expected_output((height, width), bg, diff_mask)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sprite-anomaly input")
