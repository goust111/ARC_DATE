from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": len(coords),
                }
            )
    return comps


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    patch = np.zeros((h, w), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = True
    border = np.ones((h, w), dtype=bool)
    border[1:-1, 1:-1] = False
    return int(comp["area"]) == int(border.sum()) and np.array_equal(patch, border)


def _find_vertical_split(grid: np.ndarray) -> tuple[int, int]:
    occupied = np.where((grid != BG).any(axis=0))[0]
    assert occupied.size >= 2, "expected two separated non-background regions"

    diffs = occupied[1:] - occupied[:-1]
    candidates = np.where(diffs >= 3)[0]
    assert candidates.size >= 1, "expected at least two blank columns between left and right structures"
    idx = int(candidates[np.argmax(diffs[candidates])])
    return int(occupied[idx] + 1), int(occupied[idx + 1] - 1)


def _extract_palette(left_grid: np.ndarray) -> list[int]:
    frames = [comp for comp in _monochrome_components(left_grid) if _is_rect_frame(comp)]
    assert 2 <= len(frames) <= 5, "expected 2 to 5 rectangular frame layers"

    frames.sort(
        key=lambda comp: (
            -(comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1),
            comp["bbox"][0],
            comp["bbox"][1],
        )
    )
    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "frame layers must be strictly nested"
    return [int(comp["color"]) for comp in frames]


def _outer_layer(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbor_outside = (
        ~padded[:-2, 1:-1]
        | ~padded[2:, 1:-1]
        | ~padded[1:-1, :-2]
        | ~padded[1:-1, 2:]
    )
    return center & neighbor_outside


def _layer_indices(mask: np.ndarray) -> tuple[np.ndarray, int]:
    remaining = mask.copy()
    layers = np.full(mask.shape, -1, dtype=int)
    depth = 0
    while remaining.any():
        frontier = _outer_layer(remaining)
        assert frontier.any(), "shape peeling stalled"
        layers[frontier] = depth
        remaining[frontier] = False
        depth += 1
    return layers, depth


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _colorize_shape(mask: np.ndarray, palette: list[int], base_grid: np.ndarray, top: int, left: int) -> np.ndarray:
    out = base_grid.copy()
    layers, layer_count = _layer_indices(mask)
    assert layer_count >= 1
    for idx, color in enumerate(palette):
        assert color != BG
    for r, c in np.argwhere(mask):
        out[top + int(r), left + int(c)] = int(palette[int(layers[r, c]) % len(palette)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the right-hand gray shape by peeling its layers according to the left nested-frame palette.

    Input structure:
    - The background is black.
    - The left side contains 2 to 5 concentric one-pixel-thick rectangular frame layers, each with a distinct visible color.
    - The right side contains one connected monochrome gray shape, and the two sides are separated by at least two blank columns.
    - The innermost frame encloses black cells rather than another foreground object.

    Rule:
    - Read the left palette from the outermost rectangular frame inward to the innermost frame.
    - On the right shape, repeatedly remove the current outer layer: a pixel belongs to the current layer if one of its 4-neighbors is black or already removed.
    - Color layer 1 with the first palette color, layer 2 with the second color, and continue cycling through the palette if the shape has more layers than the palette length.
    - Leave the left palette, the separating blank columns, and the rest of the background unchanged.

    Key insight:
    The left object is not to be transformed; it encodes an ordered palette by geometric nesting, while the right object must first be decomposed by iterative 4-neighbor erosion before those colors can be assigned.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    gap_left, gap_right = _find_vertical_split(g)
    palette = _extract_palette(g[:, :gap_left])

    right = g[:, gap_right + 1 :]
    mask = right != BG
    assert mask.any(), "expected a right-hand foreground shape"

    comps = _monochrome_components(right)
    assert len(comps) == 1, "expected exactly one right-hand connected shape"
    assert comps[0]["area"] == int(mask.sum()), "right-hand foreground must be a single monochrome component"

    out = g.copy()
    layers, _layer_count = _layer_indices(mask)
    for r, c in np.argwhere(mask):
        out[int(r), gap_right + 1 + int(c)] = int(palette[int(layers[r, c]) % len(palette)])
    return out


def _random_connected_coarse() -> np.ndarray | None:
    for _ in range(120):
        ch = random.randint(2, 4)
        cw = random.randint(2, 4)
        max_area = ch * cw - 1
        min_area = max(3, max(ch, cw))
        if min_area > max_area:
            continue

        target_area = random.randint(min_area, max_area)
        start = (random.randrange(ch), random.randrange(cw))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ch and 0 <= nc < cw and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue

        mask = np.zeros((ch, cw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask.all():
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue

        k = random.randint(0, 3)
        mask = np.rot90(mask, k)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask
    return None


def _scaled_shape_mask(use_cycle_branch: bool, palette_len: int) -> tuple[np.ndarray, int] | None:
    for _ in range(180):
        coarse = _random_connected_coarse()
        if coarse is None:
            continue

        if use_cycle_branch:
            scale_choices = list(range(max(3, palette_len), 6))
        else:
            scale_choices = list(range(2, 5))
        if not scale_choices:
            continue

        scale = random.choice(scale_choices)
        mask = np.kron(coarse, np.ones((scale, scale), dtype=bool))
        layers, depth = _layer_indices(mask)
        _ = layers
        if depth < 2:
            continue
        if use_cycle_branch and depth <= palette_len:
            continue
        if not use_cycle_branch and depth > palette_len:
            continue
        return mask, depth
    return None


def _sample_palette(n_layers: int) -> list[int]:
    colors = [c for c in range(1, 10) if c != GRAY]
    return random.sample(colors, n_layers)


def _build_input_and_output() -> tuple[np.ndarray, np.ndarray] | None:
    use_cycle_branch = random.random() < 0.5
    palette_len = random.randint(2, 4 if use_cycle_branch else 5)
    palette = _sample_palette(palette_len)

    shape_data = _scaled_shape_mask(use_cycle_branch, palette_len)
    if shape_data is None:
        return None
    shape_mask, depth = shape_data
    shape_h, shape_w = shape_mask.shape

    gap = random.randint(2, 5)
    feasible_steps = []
    for step in (1, 2):
        min_frame = 3 + 2 * step * (palette_len - 1)
        min_width = 1 + min_frame + gap + shape_w + 1
        min_height = 1 + max(min_frame, shape_h) + 1
        if min_width <= 30 and min_height <= 30:
            feasible_steps.append(step)
    if not feasible_steps:
        return None
    step = random.choice(feasible_steps)

    base_frame = 3 + 2 * step * (palette_len - 1)
    max_frame_w = 30 - gap - shape_w - 2
    max_frame_h = 28
    if base_frame > max_frame_w or base_frame > max_frame_h:
        return None

    frame_w = random.randint(base_frame, max_frame_w)
    frame_h = random.randint(base_frame, max_frame_h)
    inner_w = frame_w - 2 * step * (palette_len - 1)
    inner_h = frame_h - 2 * step * (palette_len - 1)
    if inner_w < 3 or inner_h < 3:
        return None

    min_width = 1 + frame_w + gap + shape_w + 1
    min_height = 1 + max(frame_h, shape_h) + 1
    width = random.randint(min_width, 30)
    height = random.randint(min_height, 30)

    extra_left = width - min_width
    left = 1 + random.randint(0, extra_left)
    shape_left = left + frame_w + gap
    extra_top_frame = height - 2 - frame_h
    extra_top_shape = height - 2 - shape_h
    frame_top = 1 + random.randint(0, extra_top_frame)
    shape_top = 1 + random.randint(0, extra_top_shape)

    inp = np.full((height, width), BG, dtype=int)
    for idx, color in enumerate(palette):
        inset = idx * step
        _paint_frame(
            inp,
            frame_top + inset,
            left + inset,
            frame_top + frame_h - 1 - inset,
            left + frame_w - 1 - inset,
            color,
        )

    inp[shape_top : shape_top + shape_h, shape_left : shape_left + shape_w][shape_mask] = GRAY

    out = inp.copy()
    layers, layer_count = _layer_indices(shape_mask)
    assert layer_count == depth
    for r, c in np.argwhere(shape_mask):
        out[shape_top + int(r), shape_left + int(c)] = int(palette[int(layers[r, c]) % len(palette)])
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_input_and_output()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")
