from __future__ import annotations

import random

import numpy as np


PLAIN_PALETTES: tuple[tuple[int, ...], ...] = (
    (1, 3),
    (2, 5),
    (1, 2, 4),
)
WRAP_PALETTES: tuple[tuple[int, ...], ...] = (
    (3, 8),
    (4, 8),
    (1, 4, 8),
    (2, 4, 7),
)
ZERO_PALETTES: tuple[tuple[int, ...], ...] = (
    (3, 7),
    (4, 6),
    (1, 2, 7),
    (1, 3, 6),
)
GEN_MAX_ATTEMPTS = 300


def _read_labels_and_scene(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 4 and g.shape[1] >= 4, "grid must contain labels and a scene"
    assert np.all(g[1, :] == 0), "expected one empty separator row below the label strip"

    labels = [int(v) for v in g[0, :] if int(v) != 0]
    assert 2 <= len(labels) <= 3, "expected two or three labeled layers"
    assert len(labels) == len(set(labels)), "layer labels must be distinct"

    scene = g[2:, :]
    assert scene.size > 0, "scene must be nonempty"
    return labels, scene


def _subset_table(colors: list[int]) -> list[int]:
    k = len(colors)
    table = [0] * 10
    for mask in range(1 << k):
        total = sum(colors[i] for i in range(k) if (mask >> i) & 1) % 10
        table[total] |= 1 << mask
    return table


def _prefix_sums(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0:
        total -= int(prefix[r0 - 1, c1])
    if c0:
        total -= int(prefix[r1, c0 - 1])
    if r0 and c0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _initial_cell_bits(scene: np.ndarray, colors: list[int]) -> np.ndarray:
    table = _subset_table(colors)
    bits = np.zeros(scene.shape, dtype=np.uint16)
    for r in range(scene.shape[0]):
        for c in range(scene.shape[1]):
            allowed = table[int(scene[r, c])]
            assert allowed, "every visible color must match at least one subset of layer labels"
            bits[r, c] = allowed
    return bits


def _layer_candidates(cell_bits: np.ndarray, layer: int, n_layers: int) -> list[tuple[int, int, int, int]]:
    h, w = map(int, cell_bits.shape)
    present = np.zeros((h, w), dtype=bool)
    absent = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            cell = int(cell_bits[r, c])
            has_present = False
            has_absent = False
            for mask in range(1 << n_layers):
                if not ((cell >> mask) & 1):
                    continue
                if (mask >> layer) & 1:
                    has_present = True
                else:
                    has_absent = True
            present[r, c] = has_present
            absent[r, c] = has_absent

    must = present & ~absent
    must_coords = np.argwhere(must)
    assert len(must_coords) > 0, "every layer must remain directly visible somewhere"

    forbidden = ~present
    min_r, min_c = map(int, must_coords.min(axis=0))
    max_r, max_c = map(int, must_coords.max(axis=0))
    prefix = _prefix_sums(forbidden)

    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(min_r + 1):
        for c0 in range(min_c + 1):
            for r1 in range(max_r, h):
                for c1 in range(max_c, w):
                    if _rect_sum(prefix, r0, c0, r1, c1) == 0:
                        candidates.append((int(r0), int(c0), int(r1), int(c1)))
    assert candidates, "each layer must admit at least one rectangle candidate"
    return candidates


def _solve_rectangles(scene: np.ndarray, colors: list[int]) -> list[tuple[int, int, int, int]]:
    h, w = map(int, scene.shape)
    n_layers = len(colors)
    cell_bits = _initial_cell_bits(scene, colors)

    candidates = [_layer_candidates(cell_bits, layer, n_layers) for layer in range(n_layers)]
    order = sorted(range(n_layers), key=lambda layer: (len(candidates[layer]), layer))

    cover_filters = [0] * n_layers
    clear_filters = [0] * n_layers
    for layer in range(n_layers):
        for mask in range(1 << n_layers):
            if (mask >> layer) & 1:
                cover_filters[layer] |= 1 << mask
            else:
                clear_filters[layer] |= 1 << mask

    assigned: list[tuple[int, int, int, int] | None] = [None] * n_layers
    solutions: list[list[tuple[int, int, int, int]]] = []

    def search(depth: int, current_bits: np.ndarray) -> None:
        if len(solutions) >= 2:
            return
        if depth == len(order):
            solutions.append([rect for rect in assigned if rect is not None])
            return

        layer = order[depth]
        for rect in candidates[layer]:
            r0, c0, r1, c1 = rect
            next_bits = current_bits.copy()
            valid = True

            for r in range(h):
                row_inside = r0 <= r <= r1
                for c in range(w):
                    inside = row_inside and c0 <= c <= c1
                    filt = cover_filters[layer] if inside else clear_filters[layer]
                    pruned = int(next_bits[r, c]) & int(filt)
                    if pruned == 0:
                        valid = False
                        break
                    next_bits[r, c] = pruned
                if not valid:
                    break

            if not valid:
                continue

            assigned[layer] = rect
            search(depth + 1, next_bits)
            assigned[layer] = None

    search(0, cell_bits)
    assert len(solutions) == 1, "layer rectangles must be uniquely determined"
    return solutions[0]


def _packed_output(rectangles: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    heights = [int(r1 - r0 + 1) for r0, c0, r1, c1 in rectangles]
    widths = [int(c1 - c0 + 1) for r0, c0, r1, c1 in rectangles]
    out = np.zeros((max(heights), sum(widths) + len(widths) - 1), dtype=int)

    col = 0
    for color, height, width in zip(colors, heights, widths):
        out[:height, col : col + width] = int(color)
        col += width + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate an additive overlap scene back into its labeled solid rectangles.

    Input structure:
    - The top edge contains a 1xM layer-label strip of nonzero colors, listed from bottom layer to top layer.
    - One empty background row separates that strip from the scene below.
    - The scene is a black background with two or three solid rectangles whose overlap colors are the sum of the covered label colors modulo 10.

    Rule:
    - Read the label strip to get the exact layer colors.
    - For each scene cell, enumerate which subsets of labeled layers could have produced its displayed color.
    - Recover the unique axis-aligned rectangle for each layer by finding the only set of filled rectangles consistent with all subset constraints.
    - Output the recovered rectangles in label-strip order, packed left-to-right with one black separator column and top-aligned.

    Key insight:
    - A single overlap color can be ambiguous, but the label-strip color prior plus the fact that every layer is one filled rectangle makes the subset choices collapse to one rectangle per layer.
    """
    colors, scene = _read_labels_and_scene(grid)
    rectangles = _solve_rectangles(scene, colors)
    return _packed_output(rectangles, colors)


def _render_scene(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    accum = np.zeros((h, w), dtype=int)
    for (r0, c0, r1, c1), color in zip(rectangles, colors):
        accum[r0 : r1 + 1, c0 : c1 + 1] += int(color)
    return accum % 10


def _coverage(rectangles: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    masks = np.zeros((len(rectangles), h, w), dtype=bool)
    for i, (r0, c0, r1, c1) in enumerate(rectangles):
        masks[i, r0 : r1 + 1, c0 : c1 + 1] = True
    return masks


def _branch_palette() -> tuple[str, list[int]]:
    category = random.choice(("plain", "wrap", "zero"))
    if category == "plain":
        palette = list(random.choice(PLAIN_PALETTES))
    elif category == "wrap":
        palette = list(random.choice(WRAP_PALETTES))
    else:
        palette = list(random.choice(ZERO_PALETTES))
    random.shuffle(palette)
    return category, palette


def _sample_core(scene_h: int, scene_w: int) -> tuple[int, int, int, int]:
    core_h = random.randint(1, 2)
    core_w = random.randint(1, 2)
    core_r0 = random.randint(1, scene_h - core_h - 2)
    core_c0 = random.randint(1, scene_w - core_w - 2)
    return (
        int(core_r0),
        int(core_c0),
        int(core_r0 + core_h - 1),
        int(core_c0 + core_w - 1),
    )


def _sample_rectangles(scene_h: int, scene_w: int, n_layers: int) -> list[tuple[int, int, int, int]]:
    core_r0, core_c0, core_r1, core_c1 = _sample_core(scene_h, scene_w)
    directions = random.sample(("tl", "tr", "bl", "br"), n_layers)
    rectangles: list[tuple[int, int, int, int]] = []

    for direction in directions:
        if direction == "tl":
            r0 = random.randint(0, core_r0)
            c0 = random.randint(0, core_c0)
            rectangles.append((int(r0), int(c0), int(core_r1), int(core_c1)))
        elif direction == "tr":
            r0 = random.randint(0, core_r0)
            c1 = random.randint(core_c1, scene_w - 1)
            rectangles.append((int(r0), int(core_c0), int(core_r1), int(c1)))
        elif direction == "bl":
            r1 = random.randint(core_r1, scene_h - 1)
            c0 = random.randint(0, core_c0)
            rectangles.append((int(core_r0), int(c0), int(r1), int(core_c1)))
        else:
            r1 = random.randint(core_r1, scene_h - 1)
            c1 = random.randint(core_c1, scene_w - 1)
            rectangles.append((int(core_r0), int(core_c0), int(r1), int(c1)))
    return rectangles


def _branch_ok(category: str, colors: list[int]) -> bool:
    full_sum = sum(colors)

    if category == "plain":
        return full_sum <= 9 and full_sum % 10 != 0
    if category == "wrap":
        return full_sum > 9 and full_sum % 10 != 0
    return full_sum % 10 == 0


def _valid_instance(category: str, rectangles: list[tuple[int, int, int, int]], colors: list[int], scene: np.ndarray) -> bool:
    masks = _coverage(rectangles, scene.shape)
    counts = masks.sum(axis=0)

    if not np.any(counts >= 2):
        return False
    for i in range(len(rectangles)):
        if not np.any(masks[i] & (counts == 1)):
            return False

    core_rows = [max(rect[0] for rect in rectangles), min(rect[2] for rect in rectangles)]
    core_cols = [max(rect[1] for rect in rectangles), min(rect[3] for rect in rectangles)]
    if core_rows[0] > core_rows[1] or core_cols[0] > core_cols[1]:
        return False

    core_value = int(scene[core_rows[0], core_cols[0]])
    if category == "plain" and core_value == 0:
        return False
    if category == "wrap" and (sum(colors) <= 9 or core_value == 0):
        return False
    if category == "zero" and core_value != 0:
        return False
    return True


def _build_input(scene: np.ndarray, colors: list[int]) -> np.ndarray:
    inp = np.zeros((scene.shape[0] + 2, scene.shape[1]), dtype=int)
    inp[0, : len(colors)] = np.asarray(colors, dtype=int)
    inp[2:, :] = scene
    return inp


def _generate_attempt() -> np.ndarray | None:
    category, colors = _branch_palette()
    scene_h = random.randint(6, 8)
    scene_w = random.randint(max(6, len(colors) + 2), 8)
    rectangles = _sample_rectangles(scene_h, scene_w, len(colors))
    scene = _render_scene((scene_h, scene_w), rectangles, colors)

    if not _valid_instance(category, rectangles, colors, scene):
        return None
    if not _branch_ok(category, colors):
        return None

    inp = _build_input(scene, colors)
    expected = _packed_output(rectangles, colors)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid additive-rectangle scene")
