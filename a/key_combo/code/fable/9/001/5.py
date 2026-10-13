import numpy as np


BACKGROUND = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 200
TILE_ATTEMPTS = 200


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    assert coords.size > 0, "expected a non-background object"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _uniform_bands(grid: np.ndarray, axis: int, bg: int) -> list[tuple[int, int, int]]:
    bands: list[tuple[int, int, int]] = []
    limit = grid.shape[axis]
    start = None
    color = None
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        uniform = bool(np.all(line == line[0]))
        line_color = int(line[0])
        is_sep = uniform and line_color != bg
        if is_sep:
            if start is None:
                start = idx
                color = line_color
            elif idx != bands[-1][1] + 1 if bands else False:
                pass
        if start is not None:
            contiguous_break = idx + 1 == limit
            if not is_sep:
                contiguous_break = True
            elif color != line_color:
                contiguous_break = True
            elif idx + 1 < limit:
                next_line = grid[idx + 1, :] if axis == 0 else grid[:, idx + 1]
                next_uniform = bool(np.all(next_line == next_line[0]))
                if not (next_uniform and int(next_line[0]) == color and int(next_line[0]) != bg):
                    contiguous_break = True
            if contiguous_break:
                end = idx if is_sep and color == line_color else idx - 1
                bands.append((int(start), int(end), int(color)))
                start = None
                color = None
    return bands


def _split_panels(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    row_bands = _uniform_bands(grid, axis=0, bg=bg)
    col_bands = _uniform_bands(grid, axis=1, bg=bg)

    valid_rows = [(r0, r1) for r0, r1, _ in row_bands if r0 > 0 and r1 < grid.shape[0] - 1]
    valid_cols = [(c0, c1) for c0, c1, _ in col_bands if c0 > 0 and c1 < grid.shape[1] - 1]
    assert not (valid_rows and valid_cols), "expected a single separator orientation"

    if valid_rows:
        r0, r1 = valid_rows[0]
        top = _crop_non_bg(grid[:r0, :], bg)
        bottom = _crop_non_bg(grid[r1 + 1 :, :], bg)
        return top, bottom

    assert valid_cols, "expected a separator band"
    c0, c1 = valid_cols[0]
    left = _crop_non_bg(grid[:, :c0], bg)
    right = _crop_non_bg(grid[:, c1 + 1 :], bg)
    return left, right


def _classify_panels(panel_a: np.ndarray, panel_b: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    panels = [panel_a, panel_b]
    roles = []
    for panel in panels:
        colors = sorted(set(int(v) for v in np.unique(panel)) - {bg})
        roles.append((len(colors), panel, colors))

    mask_candidates = [panel for color_count, panel, _ in roles if color_count == 1]
    tile_candidates = [panel for color_count, panel, _ in roles if color_count >= 2]
    assert len(mask_candidates) == 1 and len(tile_candidates) == 1, "need one mask panel and one tile panel"

    mask_panel = mask_candidates[0]
    tile_panel = tile_candidates[0]
    assert tile_panel.shape[0] == tile_panel.shape[1], "tile must be square"
    return mask_panel, tile_panel


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the inner tile onto each foreground mask cell with a phase-dependent quarter-turn rotation.

    Input structure:
    - The grid contains two separator-delimited panels on a common background.
    - One panel is a binary outer mask; the other is a square multicolor inner tile with no rotational ambiguity.
    - The separator may be horizontal or vertical, and the two panels may appear in either order.

    Rule:
    - Identify the binary panel as the outer layout and the multicolor square as the stamp tile.
    - For each foreground cell at mask coordinate (i, j), place the tile rotated clockwise by `((i + j) mod 4) * 90` degrees into the corresponding output block.
    - Background mask cells become all-background blocks, so the output size is `mask_shape * tile_side`.

    Key insight:
    - The layout comes from the binary mask, but the orientation of each copy is not fixed; it is determined solely by the sum of the mask cell coordinates, so every anti-diagonal shares one rotation phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    panel_a, panel_b = _split_panels(g, bg)
    mask_panel, tile = _classify_panels(panel_a, panel_b, bg)

    mask = mask_panel != bg
    mask_h, mask_w = mask.shape
    tile_side = tile.shape[0]

    out = np.full((mask_h * tile_side, mask_w * tile_side), bg, dtype=int)
    for i in range(mask_h):
        for j in range(mask_w):
            if not mask[i, j]:
                continue
            phase = (i + j) % 4
            rotated = np.rot90(tile, -phase)
            r0 = i * tile_side
            c0 = j * tile_side
            out[r0 : r0 + tile_side, c0 : c0 + tile_side] = rotated
    return out


def _sample_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    phases = (np.arange(height)[:, None] + np.arange(width)[None, :]) % 4
    min_fill = max(4, int(np.ceil(height * width * 0.35)))
    max_fill = height * width - 1
    for _ in range(MASK_ATTEMPTS):
        fill_prob = rng.uniform(0.4, 0.75)
        mask = rng.random((height, width)) < fill_prob
        if int(mask.sum()) < min_fill or int(mask.sum()) > max_fill:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if any(not np.any(mask & (phases == phase)) for phase in range(4)):
            continue
        return mask
    return None


def _sample_tile(rng: np.random.Generator, side: int, colors: np.ndarray) -> np.ndarray | None:
    max_colors = min(5, len(colors))
    for _ in range(TILE_ATTEMPTS):
        color_count = int(rng.integers(2, max_colors + 1))
        chosen = rng.choice(colors, size=color_count, replace=False)
        tile = rng.choice(chosen, size=(side, side), replace=True)
        if len(np.unique(tile)) < 2:
            continue
        rotations = [np.rot90(tile, -k) for k in range(4)]
        signatures = {rot.tobytes() for rot in rotations}
        if len(signatures) != 4:
            continue
        return tile.astype(int)
    return None


def _place_vertical(
    rng: np.random.Generator,
    left_obj: np.ndarray,
    right_obj: np.ndarray,
    sep_color: int,
) -> np.ndarray:
    left_h, left_w = left_obj.shape
    right_h, right_w = right_obj.shape
    left_panel_h, left_panel_w = left_h + 2, left_w + 2
    right_panel_h, right_panel_w = right_h + 2, right_w + 2
    core_h = max(left_panel_h, right_panel_h)

    top_margin = int(rng.integers(0, 3))
    bottom_margin = int(rng.integers(0, 3))
    left_margin = int(rng.integers(0, 3))
    right_margin = int(rng.integers(0, 3))
    sep_w = int(rng.integers(1, 3))

    grid_h = top_margin + core_h + bottom_margin
    grid_w = left_margin + left_panel_w + sep_w + right_panel_w + right_margin
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    sep_c0 = left_margin + left_panel_w
    grid[:, sep_c0 : sep_c0 + sep_w] = sep_color

    left_r0 = top_margin + int(rng.integers(0, core_h - left_panel_h + 1))
    right_r0 = top_margin + int(rng.integers(0, core_h - right_panel_h + 1))
    left_c0 = left_margin
    right_c0 = sep_c0 + sep_w

    grid[left_r0 + 1 : left_r0 + 1 + left_h, left_c0 + 1 : left_c0 + 1 + left_w] = left_obj
    grid[right_r0 + 1 : right_r0 + 1 + right_h, right_c0 + 1 : right_c0 + 1 + right_w] = right_obj
    return grid


def _place_horizontal(
    rng: np.random.Generator,
    top_obj: np.ndarray,
    bottom_obj: np.ndarray,
    sep_color: int,
) -> np.ndarray:
    top_h, top_w = top_obj.shape
    bottom_h, bottom_w = bottom_obj.shape
    top_panel_h, top_panel_w = top_h + 2, top_w + 2
    bottom_panel_h, bottom_panel_w = bottom_h + 2, bottom_w + 2
    core_w = max(top_panel_w, bottom_panel_w)

    top_margin = int(rng.integers(0, 3))
    bottom_margin = int(rng.integers(0, 3))
    left_margin = int(rng.integers(0, 3))
    right_margin = int(rng.integers(0, 3))
    sep_h = int(rng.integers(1, 3))

    grid_h = top_margin + top_panel_h + sep_h + bottom_panel_h + bottom_margin
    grid_w = left_margin + core_w + right_margin
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    sep_r0 = top_margin + top_panel_h
    grid[sep_r0 : sep_r0 + sep_h, :] = sep_color

    top_c0 = left_margin + int(rng.integers(0, core_w - top_panel_w + 1))
    bottom_c0 = left_margin + int(rng.integers(0, core_w - bottom_panel_w + 1))
    top_r0 = top_margin
    bottom_r0 = sep_r0 + sep_h

    grid[top_r0 + 1 : top_r0 + 1 + top_h, top_c0 + 1 : top_c0 + 1 + top_w] = top_obj
    grid[bottom_r0 + 1 : bottom_r0 + 1 + bottom_h, bottom_c0 + 1 : bottom_c0 + 1 + bottom_w] = bottom_obj
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile_side = int(rng.integers(2, 6))
    mask_h = int(rng.integers(2, 7))
    mask_w = int(rng.integers(2, 7))
    if max(mask_h * tile_side, mask_w * tile_side) > 30:
        return None

    colors = rng.choice(PALETTE, size=len(PALETTE), replace=False)
    mask_color = int(colors[0])
    sep_color = int(colors[1])
    tile_palette = colors[2:]

    mask = _sample_mask(rng, mask_h, mask_w)
    if mask is None:
        return None
    tile = _sample_tile(rng, tile_side, tile_palette)
    if tile is None:
        return None

    mask_obj = np.full((mask_h, mask_w), BACKGROUND, dtype=int)
    mask_obj[mask] = mask_color

    orientation = rng.choice(["vertical", "horizontal"])
    if orientation == "vertical":
        if bool(rng.integers(0, 2)):
            grid = _place_vertical(rng, mask_obj, tile, sep_color)
        else:
            grid = _place_vertical(rng, tile, mask_obj, sep_color)
    else:
        if bool(rng.integers(0, 2)):
            grid = _place_horizontal(rng, mask_obj, tile, sep_color)
        else:
            grid = _place_horizontal(rng, tile, mask_obj, sep_color)

    if max(grid.shape) > 30:
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
