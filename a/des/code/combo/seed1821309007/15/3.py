import numpy as np


BACKGROUND = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()


def _connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (_RNG.integers(height), _RNG.integers(width))
    cells = {tuple(map(int, start))}
    frontier = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = start[0] + dr
        cc = start[1] + dc
        if 0 <= rr < height and 0 <= cc < width:
            frontier.add((rr, cc))

    while len(cells) < area and frontier:
        rr, cc = frontier.pop()
        cells.add((rr, cc))
        neighbors = [(rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)]
        _RNG.shuffle(neighbors)
        for nr, nc in neighbors:
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != area:
        return np.zeros((height, width), dtype=bool)

    for rr, cc in cells:
        mask[rr, cc] = True
    return mask


def _find_separator(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    valid = []
    for c in candidates:
        if c == 0 or c == grid.shape[1] - 1:
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        if not np.all(np.isin(left, [BACKGROUND, GREY])):
            continue
        row_colors = []
        ok = True
        for r in range(right.shape[0]):
            colors = np.unique(right[r])
            if len(colors) != 1:
                ok = False
                break
            row_colors.append(int(colors[0]))
        if ok:
            valid.append(c)
    assert len(valid) == 1, "expected exactly one valid grey separator column"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine a grey template mask with row-wise color bands from the data panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _find_separator(g)
    template = g[:, :sep]
    data = g[:, sep + 1 :]
    assert template.shape[0] == data.shape[0]

    out = np.full(template.shape, BACKGROUND, dtype=int)
    for r in range(template.shape[0]):
        row_colors = np.unique(data[r])
        assert len(row_colors) == 1, "each data-panel row must be a single color band"
        out[r, template[r] == GREY] = int(row_colors[0])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(3, 9))
    left_w = int(_RNG.integers(4, 11))
    right_w = int(_RNG.integers(2, 8))

    if height > len(PALETTE):
        return None

    bbox_h = int(_RNG.integers(max(2, height - 3), height + 1))
    bbox_w = int(_RNG.integers(2, left_w + 1))
    top = int(_RNG.integers(0, height - bbox_h + 1))
    left = int(_RNG.integers(0, left_w - bbox_w + 1))

    bbox_area = bbox_h * bbox_w
    min_area = max(2, (bbox_area + 3) // 4)
    max_area = min(bbox_area - 1, (bbox_area * 3) // 4 + 1)
    if min_area > max_area:
        return None
    area = int(_RNG.integers(min_area, max_area + 1))

    local_mask = _connected_mask(bbox_h, bbox_w, area)
    if not local_mask.any() or local_mask.all():
        return None
    if np.any(local_mask.all(axis=0)):
        return None
    if local_mask.sum(axis=0).max() < 2:
        return None
    if local_mask.sum(axis=1).max() < 2:
        return None

    template = np.full((height, left_w), BACKGROUND, dtype=int)
    template[top : top + bbox_h, left : left + bbox_w][local_mask] = GREY

    used_rows = np.flatnonzero(np.any(template == GREY, axis=1))
    if len(used_rows) < 2:
        return None

    row_colors = _RNG.choice(PALETTE, size=height, replace=False)
    data = np.repeat(row_colors[:, None], right_w, axis=1)

    sep = np.full((height, 1), GREY, dtype=int)
    grid = np.hstack([template, sep, data])

    separators = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if len(separators) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
