import random
from collections import Counter

import numpy as np


PANEL_SIZE = 5
GRID_SHAPE = (5, 17)
BACKGROUND = 8
SEPARATOR = 0
FOREGROUND_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _panel_bounds(grid: np.ndarray) -> tuple[int, list[int]]:
    height, width = grid.shape
    assert width == 3 * height + 2, "expected three equal panels split by two columns"
    sep_cols = [height, 2 * height + 1]
    sep_color = int(grid[0, sep_cols[0]])
    for col in sep_cols:
        assert np.all(grid[:, col] == sep_color), "separator columns must be uniform"
    return sep_color, sep_cols


def _background_color(grid: np.ndarray, sep_cols: list[int]) -> int:
    panel_cells = np.delete(grid, sep_cols, axis=1).reshape(-1)
    counts = Counter(map(int, panel_cells))
    return int(counts.most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """Keep each panel's largest component and overlay them on one canvas in panel order."""
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, "expected a 5x17 grid"

    _sep_color, sep_cols = _panel_bounds(g)
    bg = _background_color(g, sep_cols)
    out = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)

    for start in (0, PANEL_SIZE + 1, 2 * PANEL_SIZE + 2):
        panel = g[:, start : start + PANEL_SIZE]
        fg_colors = [int(color) for color in np.unique(panel) if int(color) != bg]
        assert len(fg_colors) == 1, "each panel must have exactly one foreground color"
        fg = fg_colors[0]
        comps = _components_from_mask(panel == fg)
        assert len(comps) == 2, "each panel must contain exactly two foreground components"
        areas = [len(comp) for comp in comps]
        largest = max(areas)
        assert areas.count(largest) == 1, "largest component must be unique"
        for r, c in comps[areas.index(largest)]:
            out[r, c] = fg

    return out


def _grow_connected_mask(size: int, rng: random.Random) -> np.ndarray:
    assert 1 <= size <= PANEL_SIZE * PANEL_SIZE
    cells = {(rng.randrange(PANEL_SIZE), rng.randrange(PANEL_SIZE))}
    while len(cells) < size:
        r, c = rng.choice(tuple(cells))
        nbrs = list(_neighbors4(r, c, PANEL_SIZE, PANEL_SIZE))
        rng.shuffle(nbrs)
        for rr, cc in nbrs:
            if (rr, cc) not in cells:
                cells.add((rr, cc))
                break
    mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mask_is_connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask)
    return len(comps) == 1


def _non_adjacent(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    for r, c in zip(*np.where(mask_a)):
        for rr, cc in _neighbors4(int(r), int(c), PANEL_SIZE, PANEL_SIZE):
            if mask_b[rr, cc]:
                return False
    return not np.any(mask_a & mask_b)


def _place_small_component(big_mask: np.ndarray, small_size: int, rng: random.Random) -> np.ndarray | None:
    forbidden = big_mask.copy()
    for r, c in zip(*np.where(big_mask)):
        for rr, cc in _neighbors4(int(r), int(c), PANEL_SIZE, PANEL_SIZE):
            forbidden[rr, cc] = True
    allowed = ~forbidden
    if int(np.sum(allowed)) < small_size:
        return None
    allowed_cells = [(int(r), int(c)) for r, c in zip(*np.where(allowed))]
    for _ in range(120):
        start = rng.choice(allowed_cells)
        cells = {start}
        frontier = {start}
        while len(cells) < small_size and frontier:
            base = rng.choice(tuple(frontier))
            nbrs = [
                (rr, cc)
                for rr, cc in _neighbors4(base[0], base[1], PANEL_SIZE, PANEL_SIZE)
                if allowed[rr, cc] and (rr, cc) not in cells
            ]
            rng.shuffle(nbrs)
            if nbrs:
                nxt = nbrs[0]
                cells.add(nxt)
                frontier.add(nxt)
            else:
                frontier.remove(base)
        if len(cells) != small_size:
            continue
        mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _mask_is_connected(mask) and _non_adjacent(big_mask, mask):
            return mask
    return None


def _sample_big_masks(rng: random.Random) -> list[np.ndarray] | None:
    mode = rng.choice(["none", "left", "right", "chain"])
    sizes = [rng.randint(3, 7) for _ in range(3)]
    masks = [_grow_connected_mask(size, rng) for size in sizes]

    if mode == "none":
        if any(np.any(masks[i] & masks[j]) for i in range(3) for j in range(i + 1, 3)):
            return None
    elif mode == "left":
        if not np.any(masks[0] & masks[1]):
            return None
    elif mode == "right":
        if not np.any(masks[1] & masks[2]):
            return None
    else:
        if not np.any(masks[0] & masks[1]) or not np.any(masks[1] & masks[2]):
            return None

    return masks


def _build_panel(big_mask: np.ndarray, color: int, rng: random.Random) -> np.ndarray | None:
    big_size = int(np.sum(big_mask))
    small_size = rng.randint(1, min(4, big_size - 1))
    small_mask = _place_small_component(big_mask, small_size, rng)
    if small_mask is None:
        return None
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BACKGROUND, dtype=int)
    panel[big_mask | small_mask] = color
    comps = _components_from_mask(panel == color)
    if len(comps) != 2:
        return None
    areas = sorted(len(comp) for comp in comps)
    if areas[0] >= areas[1]:
        return None
    return panel


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    big_masks = _sample_big_masks(rng)
    if big_masks is None:
        return None

    colors = rng.sample(FOREGROUND_POOL, 3)
    panels = []
    for big_mask, color in zip(big_masks, colors):
        panel = _build_panel(big_mask, color, rng)
        if panel is None:
            return None
        panels.append(panel)

    grid = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
    grid[:, PANEL_SIZE] = SEPARATOR
    grid[:, 2 * PANEL_SIZE + 1] = SEPARATOR
    for idx, panel in enumerate(panels):
        start = idx * (PANEL_SIZE + 1)
        grid[:, start : start + PANEL_SIZE] = panel

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != (PANEL_SIZE, PANEL_SIZE):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
