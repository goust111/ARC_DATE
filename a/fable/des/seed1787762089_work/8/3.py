import random

import numpy as np


BG_COLOR = 0
DIVIDER_COLOR = 5
MARKER_COLOR = 6
GRID_HEIGHTS = tuple(range(13, 20, 2))
GRID_WIDTHS = tuple(range(15, 22, 2))
OBJECT_SIZE_RANGE = (5, 12)
ANCHOR_COUNTS = (2, 3, 4)
PALETTE = tuple(color for color in range(10) if color not in {BG_COLOR, DIVIDER_COLOR, MARKER_COLOR})
MAX_GENERATION_ATTEMPTS = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    norm = {(r - r0, c - c0) for r, c in coords}
    height = max(rows) - r0 + 1
    width = max(cols) - c0 + 1
    return norm, height, width


def _has_forbidden_local_symmetry(coords: set[tuple[int, int]]) -> bool:
    norm, height, width = _normalize(coords)
    if height == 1 or width == 1:
        return True
    rot180 = {(height - 1 - r, width - 1 - c) for r, c in norm}
    mirror_lr = {(r, width - 1 - c) for r, c in norm}
    return norm == rot180 or norm == mirror_lr


def _is_row_reflection_symmetric(coords: set[tuple[int, int]], center_row: int) -> bool:
    return coords == {(2 * center_row - r, c) for r, c in coords}


def _random_connected_mask(height: int, width: int, target_size: int) -> np.ndarray | None:
    if target_size < 1 or target_size > height * width:
        return None

    mask = np.zeros((height, width), dtype=bool)
    sr = random.randrange(height)
    sc = random.randrange(width)
    mask[sr, sc] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(sr, sc)
    while int(mask.sum()) < target_size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        add_frontier(r, c)

    return mask


def _sample_source_object(height: int, width: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    cr = height // 2
    cc = width // 2
    max_box_h = min(6, height - 2)
    max_box_w = min(6, cc - 1)
    if max_box_h < 2 or max_box_w < 2:
        return None

    size = random.randint(*OBJECT_SIZE_RANGE)
    for _ in range(120):
        box_h = random.randint(2, max_box_h)
        box_w = random.randint(2, max_box_w)
        if box_h * box_w < size:
            continue

        local = _random_connected_mask(box_h, box_w, size)
        if local is None:
            continue

        local_coords = {(int(r), int(c)) for r, c in np.argwhere(local)}
        if _has_forbidden_local_symmetry(local_coords):
            continue

        top = random.randint(1, height - box_h - 1)
        left_max = cc - box_w - 1
        if left_max < 1:
            continue
        left = random.randint(1, left_max)

        source = {(top + r, left + c) for r, c in local_coords}
        if _is_row_reflection_symmetric(source, cr):
            continue

        reflected = {(2 * cr - r, 2 * cc - c) for r, c in source}
        if any(not (0 <= rr < height and 0 <= rc < width) for rr, rc in reflected):
            continue
        if any(rc <= cc for _, rc in reflected):
            continue
        if any(c >= cc for _, c in source):
            continue
        return source, reflected

    return None


def _expand4(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    blocked = set(cells)
    for r, c in cells:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked.add((nr, nc))
    return blocked


def _place_anchor_positions(
    reflected: set[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    cc = width // 2
    hit = random.choice(tuple(reflected))
    anchor_positions = [hit]

    blocked = _expand4(reflected, (height, width))
    right_cells = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(cc + 2, width - 1)
        if (r, c) not in blocked
    ]

    needed = random.choice(ANCHOR_COUNTS) - 1
    for _ in range(needed):
        if not right_cells:
            return None
        cell = random.choice(right_cells)
        anchor_positions.append(cell)
        newly_blocked = _expand4({cell}, (height, width))
        blocked |= newly_blocked
        right_cells = [pos for pos in right_cells if pos not in blocked]

    return anchor_positions, hit


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1
    cr = h // 2
    cc = w // 2

    bg = _background_color(g)
    marker_color = int(g[cr, cc])
    divider_colors = {int(g[r, cc]) for r in range(h) if r != cr}
    assert len(divider_colors) == 1
    divider_color = divider_colors.pop()
    assert divider_color != marker_color

    left_mask = (g != bg) & (g != divider_color) & (g != marker_color)
    left_mask[:, cc:] = False
    source_components = _components(left_mask)
    assert len(source_components) == 1
    source = source_components[0]
    source_colors = {int(g[r, c]) for r, c in source}
    assert len(source_colors) == 1

    right_mask = (g != bg) & (g != divider_color) & (g != marker_color)
    right_mask[:, : cc + 1] = False
    anchor_components = _components(right_mask)
    assert 2 <= len(anchor_components) <= 4

    anchor_positions: list[tuple[int, int]] = []
    anchor_colors: list[int] = []
    for comp in anchor_components:
        assert len(comp) == 1
        r, c = comp[0]
        anchor_positions.append((r, c))
        anchor_colors.append(int(g[r, c]))
    assert len(set(anchor_colors)) == len(anchor_colors)

    reflected = []
    for r, c in source:
        rr = 2 * cr - r
        rc = 2 * cc - c
        assert 0 <= rr < h and 0 <= rc < w
        assert rc > cc
        reflected.append((rr, rc))

    hits = [(r, c) for r, c in reflected if (r, c) in set(anchor_positions)]
    assert len(hits) == 1
    fill_color = int(g[hits[0]])

    out = g.copy()
    for r, c in anchor_positions:
        out[r, c] = bg
    for r, c in reflected:
        out[r, c] = fill_color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(GRID_HEIGHTS)
    width = random.choice(GRID_WIDTHS)
    center_row = height // 2
    center_col = width // 2

    sampled = _sample_source_object(height, width)
    if sampled is None:
        return None
    source, reflected = sampled

    anchors = _place_anchor_positions(reflected, height, width)
    if anchors is None:
        return None
    anchor_positions, hit_pos = anchors

    object_color = random.choice(PALETTE)
    anchor_palette = [color for color in PALETTE if color != object_color]
    anchor_colors = random.sample(anchor_palette, len(anchor_positions))
    hit_color = anchor_colors[0]

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[:, center_col] = DIVIDER_COLOR
    grid[center_row, center_col] = MARKER_COLOR

    for r, c in source:
        grid[r, c] = object_color

    for idx, (r, c) in enumerate(anchor_positions):
        grid[r, c] = anchor_colors[idx]

    expected = grid.copy()
    for r, c in anchor_positions:
        expected[r, c] = BG_COLOR
    for r, c in reflected:
        expected[r, c] = hit_color

    if tuple(hit_pos) != tuple(anchor_positions[0]):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
