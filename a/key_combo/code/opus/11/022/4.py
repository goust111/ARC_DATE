import random
from collections import Counter

import numpy as np


BG = 0
SEP = 7
PANEL_SIZE = 9
GRID_SHAPE = (9, 29)
PALETTE = (1, 2, 3, 4, 6)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _panel_foreground_mask(panel: np.ndarray) -> np.ndarray:
    return (panel != BG) & (panel != SEP)


def _find_separator_columns(grid: np.ndarray) -> list[int]:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 2, "expected exactly two separator columns"
    assert cols[0] + 1 < cols[1], "separator columns must be distinct"
    return cols


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    sep_cols = _find_separator_columns(grid)
    a = grid[:, : sep_cols[0]]
    b = grid[:, sep_cols[0] + 1 : sep_cols[1]]
    c = grid[:, sep_cols[1] + 1 :]
    assert a.shape == b.shape == c.shape == (PANEL_SIZE, PANEL_SIZE)
    return [a, b, c]


def _single_panel_color(panel: np.ndarray) -> int:
    colors = sorted(set(map(int, panel.flatten())) - {BG, SEP})
    assert len(colors) == 1, "each panel must use exactly one foreground color"
    return int(colors[0])


def _normalized_mask_offsets(panel: np.ndarray) -> np.ndarray:
    coords = np.argwhere(_panel_foreground_mask(panel))
    assert coords.size > 0
    r0, c0, _r1, _c1 = _bbox_coords(coords)
    offsets = coords - np.array([r0, c0], dtype=int)
    return offsets.astype(int)


def _render(shape_offsets: np.ndarray, anchors: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    for ar, ac in anchors:
        for dr, dc in shape_offsets:
            rr = int(ar) + int(dr)
            cc = int(ac) + int(dc)
            if 0 <= rr < PANEL_SIZE and 0 <= cc < PANEL_SIZE:
                out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assign shape, position, and color roles to three separated panels by foreground-pixel count, then stamp the chosen shape onto the chosen positions.

    Input structure:
    - The grid is 9x29 with black background and two full-height orange separator columns splitting it into three 9x9 panels.
    - Each panel contains exactly one non-black, non-orange foreground color.
    - One panel holds a 4-connected irregular sprite, one holds 3-4 isolated points, and one holds a compact 3-4 pixel block.

    Rule:
    - Split the grid into the three panels using the separator columns and count each panel's non-background, non-separator foreground pixels.
    - The panel with the most pixels is the shape source, the panel with the fewest pixels is the position source, and the remaining panel is the color source.
    - Normalize the shape source to offsets from its foreground bounding-box top-left; for every foreground cell of the position source, stamp those offsets into a 9x9 black output using the color source's foreground color, clipping out-of-bounds cells.

    Key insight:
    - Pixel-count ranking is the hidden instruction channel: the same visible panel types swap semantic roles when their foreground counts swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE
    panels = _split_panels(g)

    counts = [int(_panel_foreground_mask(panel).sum()) for panel in panels]
    assert len(set(counts)) == 3, "panel counts must be unique"

    shape_idx = int(np.argmax(counts))
    position_idx = int(np.argmin(counts))
    color_idx = [idx for idx in range(3) if idx not in {shape_idx, position_idx}][0]

    shape_offsets = _normalized_mask_offsets(panels[shape_idx])
    anchors = np.argwhere(_panel_foreground_mask(panels[position_idx])).astype(int)
    color = _single_panel_color(panels[color_idx])

    return _render(shape_offsets, anchors, color)


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = 1
    return mask


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask > 0).astype(int)


def _random_polyomino(area: int) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(random.choice(frontier))

        coords = _normalize_coords(cells)
        mask = _mask_from_coords(coords)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        coords = _coords_from_mask(mask)
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w < 2:
            continue
        if area == h * w:
            continue

        degrees = []
        cell_set = {tuple(map(int, xy)) for xy in coords}
        for r, c in cell_set:
            deg = sum((r + dr, c + dc) in cell_set for dr, dc in DIRS4)
            degrees.append(deg)
        if max(degrees) < 2 or min(degrees) > 2:
            continue
        if h > 5 or w > 5:
            continue
        return coords
    raise RuntimeError("failed to build irregular sprite")


def _place_shape(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    margin_mode = random.random()
    if margin_mode < 0.35:
        top = random.choice([0, PANEL_SIZE - h])
        left = random.randint(0, PANEL_SIZE - w)
    elif margin_mode < 0.70:
        top = random.randint(0, PANEL_SIZE - h)
        left = random.choice([0, PANEL_SIZE - w])
    else:
        top = random.randint(0, PANEL_SIZE - h)
        left = random.randint(0, PANEL_SIZE - w)
    return coords + np.array([top, left], dtype=int)


def _choose_clip_anchor(shape_h: int, shape_w: int) -> tuple[int, int]:
    choices = []
    for r in range(PANEL_SIZE):
        for c in range(PANEL_SIZE):
            row_clip = r + shape_h > PANEL_SIZE
            col_clip = c + shape_w > PANEL_SIZE
            if row_clip or col_clip:
                choices.append((r, c))
    assert choices
    return random.choice(choices)


def _choose_safe_anchor(shape_h: int, shape_w: int) -> tuple[int, int]:
    return (
        random.randint(0, PANEL_SIZE - shape_h),
        random.randint(0, PANEL_SIZE - shape_w),
    )


def _anchor_is_clipped(anchor: tuple[int, int], shape_h: int, shape_w: int) -> bool:
    r, c = map(int, anchor)
    return r + shape_h > PANEL_SIZE or c + shape_w > PANEL_SIZE


def _sample_scattered_points(
    count: int,
    shape_h: int,
    shape_w: int,
    clip_mode: bool,
) -> np.ndarray:
    if clip_mode:
        candidate_pool = [(r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE)]
        required_pool = [pt for pt in candidate_pool if _anchor_is_clipped(pt, shape_h, shape_w)]
    else:
        candidate_pool = [
            (r, c)
            for r in range(PANEL_SIZE - shape_h + 1)
            for c in range(PANEL_SIZE - shape_w + 1)
        ]
        required_pool = candidate_pool
    if not required_pool:
        raise RuntimeError("no legal point anchors")
    for _ in range(300):
        points: list[tuple[int, int]] = []
        first = random.choice(required_pool)
        points.append(first)
        while len(points) < count:
            candidate = random.choice(candidate_pool)
            if candidate in points:
                continue
            if any(max(abs(candidate[0] - r), abs(candidate[1] - c)) <= 1 for r, c in points):
                continue
            points.append(candidate)
        rows = {r for r, _c in points}
        cols = {c for _r, c in points}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return np.array(sorted(points), dtype=int)
    raise RuntimeError("failed to sample scattered points")


def _sample_block(count: int, role: str, shape_h: int, shape_w: int, clip_mode: bool) -> np.ndarray:
    assert count in (3, 4)
    shape_masks = [np.ones((2, 2), dtype=int)] if count == 4 else [np.ones((1, 3), dtype=int), np.ones((3, 1), dtype=int)]
    random.shuffle(shape_masks)
    for mask in shape_masks:
        base = np.argwhere(mask > 0).astype(int)
        h, w = mask.shape
        placements: list[np.ndarray] = []
        for row in range(PANEL_SIZE - h + 1):
            for col in range(PANEL_SIZE - w + 1):
                coords = base + np.array([row, col], dtype=int)
                if role == "position":
                    clipped = any(_anchor_is_clipped((int(r), int(c)), shape_h, shape_w) for r, c in coords)
                    if clip_mode and not clipped:
                        continue
                    if not clip_mode and clipped:
                        continue
                placements.append(coords)
        if placements:
            return random.choice(placements)
    raise RuntimeError("failed to sample block")


def _paint_panel(coords: np.ndarray, color: int) -> np.ndarray:
    panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    panel[coords[:, 0], coords[:, 1]] = int(color)
    return panel


def _assemble_input(panels: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[:, 9] = SEP
    grid[:, 19] = SEP
    starts = [0, 10, 20]
    for panel, start in zip(panels, starts):
        grid[:, start : start + PANEL_SIZE] = panel
    return grid


def _count_panel_pixels(panel: np.ndarray) -> int:
    return int(_panel_foreground_mask(panel).sum())


def _nondegenerate_output(out: np.ndarray, color: int) -> bool:
    fg = np.argwhere(out == color)
    if fg.size == 0:
        return False
    if fg.shape[0] >= 32:
        return False
    rows = len(set(map(int, fg[:, 0])))
    cols = len(set(map(int, fg[:, 1])))
    return rows >= 2 and cols >= 2


def _alternative_outputs_differ(panels: list[np.ndarray], true_output: np.ndarray) -> bool:
    counts = [_count_panel_pixels(panel) for panel in panels]
    idxs = range(3)
    true_roles = (int(np.argmax(counts)), int(np.argmin(counts)))
    true_color_idx = [i for i in idxs if i not in true_roles][0]
    shape_offsets = [_normalized_mask_offsets(panel) for panel in panels]
    anchors = [np.argwhere(_panel_foreground_mask(panel)).astype(int) for panel in panels]
    colors = [_single_panel_color(panel) for panel in panels]
    different = 0
    for s in idxs:
        for p in idxs:
            if p == s:
                continue
            c = [i for i in idxs if i not in {s, p}][0]
            candidate = _render(shape_offsets[s], anchors[p], colors[c])
            if not np.array_equal(candidate, true_output):
                different += 1
    return different == 5


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(PALETTE, 3)
    sprite_area = random.randint(5, 8)
    sprite = _place_shape(_random_polyomino(sprite_area))
    sr0, sc0, sr1, sc1 = _bbox_coords(sprite)
    shape_h = sr1 - sr0 + 1
    shape_w = sc1 - sc0 + 1

    points_are_position = random.random() < 0.5
    point_count = 3 if points_are_position else 4
    block_count = 4 if points_are_position else 3
    clip_mode = random.random() < 0.55

    try:
        point_coords = _sample_scattered_points(
            point_count,
            shape_h,
            shape_w,
            clip_mode if points_are_position else False,
        )
        block_coords = _sample_block(
            block_count,
            "color" if points_are_position else "position",
            shape_h,
            shape_w,
            clip_mode if not points_are_position else False,
        )
    except RuntimeError:
        return None

    sprite_panel = _paint_panel(sprite, colors[0])
    points_panel = _paint_panel(point_coords, colors[1])
    block_panel = _paint_panel(block_coords, colors[2])

    typed_panels = [
        ("sprite", sprite_panel),
        ("points", points_panel),
        ("block", block_panel),
    ]
    random.shuffle(typed_panels)
    panels = [panel for _name, panel in typed_panels]
    grid = _assemble_input(panels)

    try:
        out = main(grid)
    except AssertionError:
        return None

    parsed_panels = _split_panels(grid)
    counts = [_count_panel_pixels(panel) for panel in parsed_panels]
    if sorted(counts) != sorted([sprite_area, point_count, block_count]):
        return None
    if len(set(counts)) != 3:
        return None

    shape_idx = int(np.argmax(counts))
    position_idx = int(np.argmin(counts))
    color_idx = [i for i in range(3) if i not in {shape_idx, position_idx}][0]
    color = _single_panel_color(parsed_panels[color_idx])
    anchors = np.argwhere(_panel_foreground_mask(parsed_panels[position_idx]))
    shape_offsets = _normalized_mask_offsets(parsed_panels[shape_idx])

    if shape_offsets.shape[0] != sprite_area or anchors.shape[0] != min(counts):
        return None
    if not _nondegenerate_output(out, color):
        return None
    if clip_mode:
        clipped = any(
            int(ar) + int(dr) >= PANEL_SIZE or int(ac) + int(dc) >= PANEL_SIZE
            for ar, ac in anchors
            for dr, dc in shape_offsets
        )
        if not clipped:
            return None
    else:
        clipped = any(
            int(ar) + int(dr) >= PANEL_SIZE or int(ac) + int(dc) >= PANEL_SIZE
            for ar, ac in anchors
            for dr, dc in shape_offsets
        )
        if clipped:
            return None

    if not _alternative_outputs_differ(parsed_panels, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
