from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIVIDER_COLOR = 5
GEN_RETRIES = 300
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append({"coords": coords, "bbox": _bbox(coords)})
    return out


def _find_divider(grid: np.ndarray) -> tuple[str, int, int]:
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(values, counts, strict=False)}
    candidates: list[tuple[str, int, int]] = []

    for r in range(h):
        color = int(grid[r, 0])
        if np.all(grid[r, :] == color) and color_counts[color] == w and 0 < r < h - 1:
            candidates.append(("row", r, color))

    for c in range(w):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color) and color_counts[color] == h and 0 < c < w - 1:
            candidates.append(("col", c, color))

    assert len(candidates) == 1, "expected a unique full divider line"
    return candidates[0]


def _extract_mapping(example: np.ndarray, bg: int) -> dict[int, int]:
    comps = _components(example != bg)
    assert len(comps) == 2, "example region must contain exactly two objects"

    ordered = sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    before = ordered[0]["bbox"]
    after = ordered[1]["bbox"]
    assert before[2] < after[0], "example objects must be stacked vertically"

    br0, bc0, br1, bc1 = before
    ar0, ac0, ar1, ac1 = after
    before_patch = example[br0 : br1 + 1, bc0 : bc1 + 1]
    after_patch = example[ar0 : ar1 + 1, ac0 : ac1 + 1]

    before_mask = before_patch != bg
    after_mask = after_patch != bg
    assert before_mask.shape == after_mask.shape, "example objects must share a bounding box shape"
    assert np.array_equal(before_mask, after_mask), "example objects must have identical occupied cells"

    mapping: dict[int, int] = {}
    for r, c in np.argwhere(before_mask):
        src = int(before_patch[int(r), int(c)])
        dst = int(after_patch[int(r), int(c)])
        if src in mapping:
            assert mapping[src] == dst, "example mapping must be consistent"
        else:
            mapping[src] = dst

    assert len(mapping) == 2, "expected exactly two source colors in the example"
    assert len(set(mapping.values())) == 2, "expected exactly two target colors in the example"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a two-color replacement rule from the example panel and apply it only to the operation panel.

    Input structure:
    - The grid is split into an example region and an operation region by one full gray divider row or column.
    - The example region contains exactly two vertically stacked objects with the same shape; the upper object uses two source colors and the lower object shows the recolored result with two target colors.
    - The operation region contains several separated objects that may use the example's two source colors alongside other unrelated colors.

    Rule:
    - Detect the divider and keep it unchanged.
    - Compare the upper and lower example objects cell by cell to infer the two source→target color substitutions.
    - Apply those substitutions only inside the operation region, replacing source-color pixels while leaving all other colors and the entire example region unchanged.

    Key insight:
    - The left or top panel is a legend that teaches a color mapping; it is not itself transformed, and only matching colors on the data side should be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    kind, index, _divider_color = _find_divider(g)

    if kind == "col":
        example = g[:, :index]
        operation = g[:, index + 1 :]
        assert example.size > 0 and operation.size > 0, "divider must split two non-empty regions"
        op_slices = (slice(None), slice(index + 1, g.shape[1]))
    else:
        example = g[:index, :]
        operation = g[index + 1 :, :]
        assert example.size > 0 and operation.size > 0, "divider must split two non-empty regions"
        op_slices = (slice(index + 1, g.shape[0]), slice(None))

    mapping = _extract_mapping(example, bg)
    region = operation.copy()
    transformed = region.copy()
    for src, dst in mapping.items():
        transformed[region == src] = dst

    out = g.copy()
    out[op_slices] = transformed
    return out


def _grow_mask(height: int, width: int, cell_count: int) -> np.ndarray | None:
    if cell_count < 1 or cell_count > height * width:
        return None

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < cell_count:
        if not frontier:
            return None
        frontier_list = list(frontier)
        frontier_list.sort(
            key=lambda rc: sum((rc[0] + dr, rc[1] + dc) in cells for dr, dc in DIRS4),
            reverse=True,
        )
        top_k = max(1, min(4, len(frontier_list)))
        chosen = random.choice(frontier_list[:top_k] if random.random() < 0.7 else frontier_list)
        frontier.remove(chosen)
        cells.add(chosen)
        add_frontier(*chosen)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _colorize_mask(mask: np.ndarray, bg: int, colors: list[int], required: list[int]) -> np.ndarray | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if len(coords) < max(1, len(required)):
        return None

    patch = np.full(mask.shape, bg, dtype=int)
    shuffled = coords[:]
    random.shuffle(shuffled)
    used = 0
    for color in required:
        r, c = shuffled[used]
        patch[r, c] = int(color)
        used += 1

    palette = list(dict.fromkeys(required + colors))
    for r, c in shuffled[used:]:
        patch[r, c] = int(random.choice(palette))

    if required and any(not np.any(patch == color) for color in required):
        return None
    return patch


def _find_positions(panel: np.ndarray, patch_shape: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = panel.shape
    ph, pw = patch_shape
    positions = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            r1 = r0 + ph
            c1 = c0 + pw
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r1 + 1)
            cc1 = min(w, c1 + 1)
            if not blocked[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_patch(panel: np.ndarray, patch: np.ndarray, blocked: np.ndarray, bg: int) -> bool:
    positions = _find_positions(panel, patch.shape, blocked)
    if not positions:
        return False
    r0, c0 = random.choice(positions)
    mask = patch != bg
    panel[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(panel.shape[0], r0 + patch.shape[0] + 1)
    cc1 = min(panel.shape[1], c0 + patch.shape[1] + 1)
    blocked[rr0:rr1, cc0:cc1] = True
    return True


def _make_example_panel(height: int, width: int, bg: int, src_a: int, src_b: int, dst_a: int, dst_b: int) -> np.ndarray | None:
    if height < 7 or width < 4:
        return None

    max_obj_h = min(6, (height - 3) // 2)
    max_obj_w = min(6, width - 2)
    if max_obj_h < 2 or max_obj_w < 2:
        return None

    obj_h = random.randint(2, max_obj_h)
    obj_w = random.randint(2, max_obj_w)
    max_cells = min(obj_h * obj_w, max(4, obj_h * obj_w - 1))
    min_cells = max(4, (obj_h * obj_w + 1) // 2)
    cell_count = random.randint(min_cells, max_cells)
    mask = _grow_mask(obj_h, obj_w, cell_count)
    if mask is None:
        return None

    before_patch = _colorize_mask(mask, bg, [src_a, src_b], [src_a, src_b])
    if before_patch is None:
        return None
    after_patch = before_patch.copy()
    after_patch[before_patch == src_a] = dst_a
    after_patch[before_patch == src_b] = dst_b

    panel = np.full((height, width), bg, dtype=int)
    left = random.randint(1, width - obj_w - 1)
    max_top = height - 2 * obj_h - 3
    if max_top < 1:
        return None
    top_r = random.randint(1, max_top)
    bottom_r = random.randint(top_r + obj_h + 2, height - obj_h - 1)

    panel[top_r : top_r + obj_h, left : left + obj_w][mask] = before_patch[mask]
    panel[bottom_r : bottom_r + obj_h, left : left + obj_w][mask] = after_patch[mask]
    return panel


def _random_object_patch(panel_h: int, panel_w: int, bg: int, extra_colors: list[int], required: list[int]) -> np.ndarray | None:
    max_h = min(6, panel_h - 2)
    max_w = min(6, panel_w - 2)
    if max_h < 2 or max_w < 2:
        return None

    patch_h = random.randint(2, max_h)
    patch_w = random.randint(2, max_w)
    area = patch_h * patch_w
    min_cells = max(3, len(required))
    max_cells = min(area, max(4, area - 1))
    if min_cells > max_cells:
        return None
    cell_count = random.randint(min_cells, max_cells)
    mask = _grow_mask(patch_h, patch_w, cell_count)
    if mask is None:
        return None

    pool = required[:]
    available_extras = [color for color in extra_colors if color not in pool]
    random.shuffle(available_extras)
    extra_pick_count = random.randint(0 if required else 1, min(2, len(available_extras)))
    pool.extend(available_extras[:extra_pick_count])
    if not pool:
        return None
    patch = _colorize_mask(mask, bg, pool, required)
    if patch is None:
        return None
    return patch


def _make_operation_panel(height: int, width: int, bg: int, src_a: int, src_b: int, extra_colors: list[int]) -> np.ndarray | None:
    panel = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    required_sets: list[list[int]] = [[src_a], [src_b]]
    extra_object_count = random.randint(1, 4)
    options = [[], [src_a], [src_b], [src_a, src_b]]
    for _ in range(extra_object_count):
        required_sets.append(random.choice(options))
    random.shuffle(required_sets)

    for required in required_sets:
        patch = None
        for _ in range(40):
            patch = _random_object_patch(height, width, bg, extra_colors, required)
            if patch is not None and _place_patch(panel, patch, blocked, bg):
                break
            patch = None
        if patch is None:
            return None

    assert np.any(panel == src_a) and np.any(panel == src_b), "operation panel must use both mapped colors"
    return panel


def _expected_output(inp: np.ndarray, src_a: int, src_b: int, dst_a: int, dst_b: int, vertical: bool, example_extent: int) -> np.ndarray:
    out = inp.copy()
    if vertical:
        region = inp[:, example_extent + 1 :]
        transformed = region.copy()
        transformed[region == src_a] = dst_a
        transformed[region == src_b] = dst_b
        out[:, example_extent + 1 :] = transformed
    else:
        region = inp[example_extent + 1 :, :]
        transformed = region.copy()
        transformed[region == src_a] = dst_a
        transformed[region == src_b] = dst_b
        out[example_extent + 1 :, :] = transformed
    return out


def _gen_vertical_case(bg: int, src_a: int, src_b: int, dst_a: int, dst_b: int, extra_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    total_h = random.randint(13, 26)
    example_w = random.randint(6, 11)
    max_operation_w = min(18, 29 - example_w)
    if max_operation_w < 9:
        return None
    operation_w = random.randint(9, max_operation_w)

    example = _make_example_panel(total_h, example_w, bg, src_a, src_b, dst_a, dst_b)
    if example is None:
        return None
    operation = _make_operation_panel(total_h, operation_w, bg, src_a, src_b, extra_colors)
    if operation is None:
        return None

    grid = np.full((total_h, example_w + 1 + operation_w), bg, dtype=int)
    grid[:, :example_w] = example
    grid[:, example_w] = DIVIDER_COLOR
    grid[:, example_w + 1 :] = operation
    expected = _expected_output(grid, src_a, src_b, dst_a, dst_b, True, example_w)
    return grid, expected


def _gen_horizontal_case(bg: int, src_a: int, src_b: int, dst_a: int, dst_b: int, extra_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    total_w = random.randint(13, 26)
    example_h = random.randint(8, 13)
    max_operation_h = min(16, 29 - example_h)
    if max_operation_h < 7:
        return None
    operation_h = random.randint(7, max_operation_h)

    example = _make_example_panel(example_h, total_w, bg, src_a, src_b, dst_a, dst_b)
    if example is None:
        return None
    operation = _make_operation_panel(operation_h, total_w, bg, src_a, src_b, extra_colors)
    if operation is None:
        return None

    grid = np.full((example_h + 1 + operation_h, total_w), bg, dtype=int)
    grid[:example_h, :] = example
    grid[example_h, :] = DIVIDER_COLOR
    grid[example_h + 1 :, :] = operation
    expected = _expected_output(grid, src_a, src_b, dst_a, dst_b, False, example_h)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(COLOR_POOL, 7)
    bg, src_a, src_b, dst_a, dst_b, extra1, extra2 = colors
    extra_colors = [extra1, extra2]

    if random.random() < 0.5:
        candidate = _gen_vertical_case(bg, src_a, src_b, dst_a, dst_b, extra_colors)
    else:
        candidate = _gen_horizontal_case(bg, src_a, src_b, dst_a, dst_b, extra_colors)
    if candidate is None:
        return None

    inp, expected = candidate
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
