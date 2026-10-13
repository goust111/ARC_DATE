from __future__ import annotations

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
KEY_COLORS = (1, 2, 4, 6, 7, 8, 9)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

MAX_GENERATION_ATTEMPTS = 400
MIN_KEYS = 2
MAX_KEYS = 4
MAX_SHAPE_DIM = 4
PANEL_GUTTER = 1


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for variant in (rot, np.fliplr(rot)):
            trimmed = _trim_mask(np.asarray(variant, dtype=bool))
            key = (trimmed.tobytes(), tuple(map(int, trimmed.shape)))
            if key not in seen:
                seen.add(key)
                variants.append(trimmed)
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    return min((variant.tobytes(), tuple(map(int, variant.shape))) for variant in _shape_variants(mask))


def _detect_divider(grid: np.ndarray) -> tuple[str, int]:
    h, w = grid.shape
    grey_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    grey_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    if len(grey_cols) == 1 and not grey_rows:
        return "vertical", int(grey_cols[0])
    if len(grey_rows) == 1 and not grey_cols:
        return "horizontal", int(grey_rows[0])
    raise AssertionError("expected exactly one full grey divider line")


def _extract_keys(panel: np.ndarray) -> list[dict]:
    comps = _connected_components(panel != BLACK)
    keys: list[dict] = []
    for comp in comps:
        colors = {int(panel[r, c]) for r, c in comp}
        assert len(colors) == 1, "each key must be monochrome"
        color = next(iter(colors))
        assert color not in (BLACK, GREY, GREEN)
        mask = _mask_from_coords(comp)
        keys.append(
            {
                "color": color,
                "mask": mask,
                "shape_key": _canonical_shape_key(mask),
            }
        )
    return keys


def _extract_holes(panel: np.ndarray) -> list[dict]:
    h, w = panel.shape
    comps = _connected_components(panel == BLACK)
    holes: list[dict] = []
    for comp in comps:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        comp_set = set(comp)
        enclosed = True
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in comp_set:
                    continue
                if panel[nr, nc] != GREEN:
                    enclosed = False
                    break
            if not enclosed:
                break
        if not enclosed:
            continue
        holes.append(
            {
                "coords": comp,
                "mask": _mask_from_coords(comp),
                "shape_key": _canonical_shape_key(_mask_from_coords(comp)),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each lock hole with the color of the matching key shape from the other panel.

    Input structure:
    - The background is black, and one full grey divider line splits the grid into two panels, either left/right or top/bottom.
    - One panel contains several small monochrome key shapes in distinct non-black, non-grey, non-green colors.
    - The other panel contains a larger green filled region on black, with several enclosed black holes cut out of it.
    - Each hole is congruent to exactly one key shape up to rotation and horizontal reflection.

    Rule:
    - Detect the divider and separate the two panels.
    - Find the monochrome key components and canonicalize each shape under rotations and reflections.
    - Find the enclosed black holes inside the green region and canonicalize those shapes the same way.
    - Match keys to holes by canonical shape, then recolor every pixel of each hole with its key's color.

    Key insight:
    The black lock holes are not standalone objects; they must be extracted as enclosed background components inside the green region, then matched to the scattered keys by rotation/flip-invariant shape.
    """
    g = np.asarray(grid, dtype=int)
    orientation, index = _detect_divider(g)

    if orientation == "vertical":
        panel_a = g[:, :index]
        panel_b = g[:, index + 1 :]
    else:
        panel_a = g[:index, :]
        panel_b = g[index + 1 :, :]

    if int(np.sum(panel_a == GREEN)) > int(np.sum(panel_b == GREEN)):
        lock_panel = panel_a.copy()
        key_panel = panel_b
    else:
        lock_panel = panel_b.copy()
        key_panel = panel_a

    keys = _extract_keys(key_panel)
    holes = _extract_holes(lock_panel)

    assert len(keys) == len(holes) >= 1, "keys and holes must be in one-to-one correspondence"

    key_map: dict[tuple[bytes, tuple[int, int]], int] = {}
    for key in keys:
        shape_key = key["shape_key"]
        assert shape_key not in key_map, "key shapes must be unique"
        key_map[shape_key] = int(key["color"])

    for hole in holes:
        shape_key = hole["shape_key"]
        assert shape_key in key_map, "every hole must match a key"
        for r, c in hole["coords"]:
            lock_panel[r, c] = key_map[shape_key]

    return lock_panel


def _normalize_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _random_shape(area: int) -> np.ndarray:
    for _ in range(200):
        coords: set[tuple[int, int]] = {(0, 0)}
        while len(coords) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in coords:
                for dr, dc in DIR4:
                    candidate = (r + dr, c + dc)
                    if candidate not in coords and candidate not in frontier:
                        frontier.append(candidate)
            choice = frontier[int(RNG.integers(len(frontier)))]
            coords.add(choice)
            norm = _normalize_coords(coords)
            height = max(r for r, _ in norm) + 1
            width = max(c for _, c in norm) + 1
            if max(height, width) > MAX_SHAPE_DIM:
                break
        if len(coords) != area:
            continue
        norm = _normalize_coords(coords)
        height = max(r for r, _ in norm) + 1
        width = max(c for _, c in norm) + 1
        if max(height, width) > MAX_SHAPE_DIM:
            continue
        if area >= 5 and min(height, width) == 1:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        bbox_area = height * width
        if bbox_area == area and area >= 4 and RNG.random() < 0.7:
            continue
        return mask
    raise RuntimeError("failed to generate a compact shape")


def _random_transform(mask: np.ndarray) -> np.ndarray:
    variants = _shape_variants(mask)
    return variants[int(RNG.integers(len(variants)))].copy()


def _choose_grid(n_items: int, mode: str) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for rows in range(1, n_items + 1):
        cols = (n_items + rows - 1) // rows
        if mode == "vertical":
            if cols <= 2:
                choices.append((rows, cols))
        else:
            if rows <= 2:
                choices.append((rows, cols))
    return choices[int(RNG.integers(len(choices)))]


def _build_lock_core(
    hole_masks: list[np.ndarray], colors: list[int], mode: str
) -> tuple[np.ndarray, np.ndarray]:
    n = len(hole_masks)
    rows, cols = _choose_grid(n, mode)
    order = np.array(range(n))
    RNG.shuffle(order)

    max_h = max(mask.shape[0] for mask in hole_masks)
    max_w = max(mask.shape[1] for mask in hole_masks)
    cell_h = max_h + int(RNG.integers(0, 2))
    cell_w = max_w + int(RNG.integers(0, 2))

    inner_h = rows * cell_h + (rows + 1) * PANEL_GUTTER
    inner_w = cols * cell_w + (cols + 1) * PANEL_GUTTER

    margin_top = int(RNG.integers(1, 3))
    margin_bottom = int(RNG.integers(1, 3))
    margin_left = int(RNG.integers(1, 3))
    margin_right = int(RNG.integers(1, 3))

    core_h = inner_h + margin_top + margin_bottom
    core_w = inner_w + margin_left + margin_right
    lock_in = np.full((core_h, core_w), BLACK, dtype=int)
    lock_out = np.full((core_h, core_w), BLACK, dtype=int)

    green_r0 = margin_top
    green_c0 = margin_left
    lock_in[green_r0 : green_r0 + inner_h, green_c0 : green_c0 + inner_w] = GREEN
    lock_out[green_r0 : green_r0 + inner_h, green_c0 : green_c0 + inner_w] = GREEN

    for slot_idx, hole_idx in enumerate(order.tolist()):
        slot_r = slot_idx // cols
        slot_c = slot_idx % cols
        mask = hole_masks[hole_idx]
        color = colors[hole_idx]
        h, w = mask.shape
        slot_top = green_r0 + PANEL_GUTTER + slot_r * (cell_h + PANEL_GUTTER)
        slot_left = green_c0 + PANEL_GUTTER + slot_c * (cell_w + PANEL_GUTTER)
        off_r = int(RNG.integers(cell_h - h + 1))
        off_c = int(RNG.integers(cell_w - w + 1))
        top = slot_top + off_r
        left = slot_left + off_c
        for rr, cc in np.argwhere(mask):
            r = top + int(rr)
            c = left + int(cc)
            lock_in[r, c] = BLACK
            lock_out[r, c] = color

    return lock_in, lock_out


def _build_key_core(key_masks: list[np.ndarray], colors: list[int], mode: str) -> np.ndarray:
    n = len(key_masks)
    rows, cols = _choose_grid(n, mode)
    order = np.array(range(n))
    RNG.shuffle(order)

    max_h = max(mask.shape[0] for mask in key_masks)
    max_w = max(mask.shape[1] for mask in key_masks)
    cell_h = max_h + int(RNG.integers(0, 2))
    cell_w = max_w + int(RNG.integers(0, 2))

    inner_h = rows * cell_h + (rows + 1) * PANEL_GUTTER
    inner_w = cols * cell_w + (cols + 1) * PANEL_GUTTER

    margin_top = int(RNG.integers(1, 3))
    margin_bottom = int(RNG.integers(1, 3))
    margin_left = int(RNG.integers(1, 3))
    margin_right = int(RNG.integers(1, 3))

    core_h = inner_h + margin_top + margin_bottom
    core_w = inner_w + margin_left + margin_right
    key_core = np.full((core_h, core_w), BLACK, dtype=int)

    for slot_idx, key_idx in enumerate(order.tolist()):
        slot_r = slot_idx // cols
        slot_c = slot_idx % cols
        mask = key_masks[key_idx]
        color = colors[key_idx]
        h, w = mask.shape
        slot_top = margin_top + PANEL_GUTTER + slot_r * (cell_h + PANEL_GUTTER)
        slot_left = margin_left + PANEL_GUTTER + slot_c * (cell_w + PANEL_GUTTER)
        off_r = int(RNG.integers(cell_h - h + 1))
        off_c = int(RNG.integers(cell_w - w + 1))
        top = slot_top + off_r
        left = slot_left + off_c
        for rr, cc in np.argwhere(mask):
            key_core[top + int(rr), left + int(cc)] = color

    return key_core


def _embed_vertical(key_core: np.ndarray, lock_in_core: np.ndarray, lock_out_core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extra_h = int(RNG.integers(0, 3))
    full_h = max(key_core.shape[0], lock_in_core.shape[0]) + extra_h
    key_w = key_core.shape[1]
    lock_w = lock_in_core.shape[1]
    if full_h > 30 or key_w + 1 + lock_w > 30:
        raise ValueError("vertical layout exceeds ARC size limit")

    key_panel = np.full((full_h, key_w), BLACK, dtype=int)
    lock_panel_in = np.full((full_h, lock_w), BLACK, dtype=int)
    lock_panel_out = np.full((full_h, lock_w), BLACK, dtype=int)

    key_top = int(RNG.integers(full_h - key_core.shape[0] + 1))
    lock_top = int(RNG.integers(full_h - lock_in_core.shape[0] + 1))

    key_panel[key_top : key_top + key_core.shape[0], :] = key_core
    lock_panel_in[lock_top : lock_top + lock_in_core.shape[0], :] = lock_in_core
    lock_panel_out[lock_top : lock_top + lock_out_core.shape[0], :] = lock_out_core

    grid = np.full((full_h, key_w + 1 + lock_w), BLACK, dtype=int)
    grid[:, :key_w] = key_panel
    grid[:, key_w] = GREY
    grid[:, key_w + 1 :] = lock_panel_in
    return grid, lock_panel_out


def _embed_horizontal(
    key_core: np.ndarray, lock_in_core: np.ndarray, lock_out_core: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    extra_w = int(RNG.integers(0, 3))
    full_w = max(key_core.shape[1], lock_in_core.shape[1]) + extra_w
    key_h = key_core.shape[0]
    lock_h = lock_in_core.shape[0]
    if full_w > 30 or key_h + 1 + lock_h > 30:
        raise ValueError("horizontal layout exceeds ARC size limit")

    key_panel = np.full((key_h, full_w), BLACK, dtype=int)
    lock_panel_in = np.full((lock_h, full_w), BLACK, dtype=int)
    lock_panel_out = np.full((lock_h, full_w), BLACK, dtype=int)

    key_left = int(RNG.integers(full_w - key_core.shape[1] + 1))
    lock_left = int(RNG.integers(full_w - lock_in_core.shape[1] + 1))

    key_panel[:, key_left : key_left + key_core.shape[1]] = key_core
    lock_panel_in[:, lock_left : lock_left + lock_in_core.shape[1]] = lock_in_core
    lock_panel_out[:, lock_left : lock_left + lock_out_core.shape[1]] = lock_out_core

    grid = np.full((key_h + 1 + lock_h, full_w), BLACK, dtype=int)
    grid[:key_h, :] = key_panel
    grid[key_h, :] = GREY
    grid[key_h + 1 :, :] = lock_panel_in
    return grid, lock_panel_out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    orientation = "vertical" if RNG.random() < 0.5 else "horizontal"
    n_keys = int(RNG.integers(MIN_KEYS, MAX_KEYS + 1))
    colors = list(map(int, RNG.choice(KEY_COLORS, size=n_keys, replace=False)))

    base_shapes: list[np.ndarray] = []
    seen_shapes: set[tuple[bytes, tuple[int, int]]] = set()
    while len(base_shapes) < n_keys:
        area = int(RNG.integers(3, 7))
        mask = _random_shape(area)
        shape_key = _canonical_shape_key(mask)
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        base_shapes.append(mask)

    key_masks = [_random_transform(mask) for mask in base_shapes]
    hole_masks = [_random_transform(mask) for mask in base_shapes]

    lock_in_core, lock_out_core = _build_lock_core(hole_masks, colors, orientation)
    key_core = _build_key_core(key_masks, colors, orientation)

    if orientation == "vertical":
        grid, expected = _embed_vertical(key_core, lock_in_core, lock_out_core)
    else:
        grid, expected = _embed_horizontal(key_core, lock_in_core, lock_out_core)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        raise ValueError("grid exceeds ARC size limit")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected = _generate_case()
        except (AssertionError, RuntimeError, ValueError):
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
