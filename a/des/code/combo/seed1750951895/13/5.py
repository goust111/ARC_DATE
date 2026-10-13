from __future__ import annotations

import random

import numpy as np


PANEL_SIZE = 5
NUM_PANELS = 3
GRID_SHAPE = (5, 17)
PANEL_STARTS = (0, 6, 12)
SEPARATOR_COLS = (5, 11)

BG = 8
SEP = 0
FG_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in CARDINAL_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _valid_shift(cells: list[tuple[int, int]], occupied: np.ndarray, dr: int, dc: int) -> bool:
    shifted = [(r + dr, c + dc) for r, c in cells]
    for r, c in shifted:
        if not (0 <= r < PANEL_SIZE and 0 <= c < PANEL_SIZE):
            return False
        if occupied[r, c]:
            return False
    return True


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in cells]


def _paint(canvas: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        canvas[r, c] = color


def _mode(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int).ravel(), minlength=10)
    return int(np.argmax(counts))


def main(grid: np.ndarray) -> np.ndarray:
    """Keep the largest component from each 5x5 panel and overlay them onto one 5x5 canvas."""
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, "expected a 5x17 grid"
    assert np.all(g[:, SEPARATOR_COLS[0]] == g[0, SEPARATOR_COLS[0]])
    assert np.all(g[:, SEPARATOR_COLS[1]] == g[0, SEPARATOR_COLS[1]])

    sep = int(g[0, SEPARATOR_COLS[0]])
    assert sep == int(g[0, SEPARATOR_COLS[1]])

    panel_pixels = np.concatenate(
        [g[:, start : start + PANEL_SIZE].ravel() for start in PANEL_STARTS]
    )
    bg = _mode(panel_pixels)
    out = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    occupied = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)

    for start in PANEL_STARTS:
        panel = g[:, start : start + PANEL_SIZE]
        fg_colors = [int(v) for v in np.unique(panel) if int(v) != bg]
        assert len(fg_colors) == 1, "each panel must use one foreground color"
        color = fg_colors[0]

        comps = _components(panel == color)
        assert len(comps) == 2, "each panel must contain exactly two foreground components"
        sizes = [len(comp) for comp in comps]
        largest = max(sizes)
        assert sizes.count(largest) == 1, "largest component must be unique"
        sprite = comps[sizes.index(largest)]

        if any(occupied[r, c] for r, c in sprite):
            valid_dirs = [(dr, dc) for dr, dc in CARDINAL_DIRS if _valid_shift(sprite, occupied, dr, dc)]
            assert valid_dirs, "overlapping sprite must be movable by one step"
            dr, dc = valid_dirs[0]
            sprite = _shift_cells(sprite, dr, dc)

        _paint(out, sprite, color)
        for r, c in sprite:
            occupied[r, c] = True

    return out


def _random_polyomino(size: int, forbidden: set[tuple[int, int]] | None = None) -> list[tuple[int, int]] | None:
    if forbidden is None:
        forbidden = set()
    free = [(r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE) if (r, c) not in forbidden]
    if not free:
        return None

    start = random.choice(free)
    cells = {start}
    frontier = set()
    for dr, dc in CARDINAL_DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE and (nr, nc) not in forbidden:
            frontier.add((nr, nc))

    while len(cells) < size and frontier:
        nxt = random.choice(list(frontier))
        frontier.discard(nxt)
        cells.add(nxt)
        for dr, dc in CARDINAL_DIRS:
            nr, nc = nxt[0] + dr, nxt[1] + dc
            if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE and (nr, nc) not in forbidden and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != size:
        return None
    return list(cells)


def _adjacent_forbidden(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    forbidden = set(cells)
    for r, c in cells:
        for dr, dc in CARDINAL_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE:
                forbidden.add((nr, nc))
    return forbidden


def _sample_output_sprites(colors: list[int]) -> list[tuple[int, list[tuple[int, int]]]] | None:
    occupied: set[tuple[int, int]] = set()
    sprites: list[tuple[int, list[tuple[int, int]]]] = []
    for idx, color in enumerate(colors):
        remaining = NUM_PANELS - idx - 1
        max_size = min(7, PANEL_SIZE * PANEL_SIZE - len(occupied) - 2 * remaining)
        if max_size < 2:
            return None
        size = random.randint(2, max_size)
        sprite = _random_polyomino(size, forbidden=occupied)
        if sprite is None:
            return None
        sprites.append((color, sprite))
        occupied.update(sprite)
    return sprites


def _one_step_options(cells: list[tuple[int, int]], occupied: np.ndarray) -> list[tuple[int, int]]:
    return [(dr, dc) for dr, dc in CARDINAL_DIRS if _valid_shift(cells, occupied, dr, dc)]


def _choose_input_sprite_cells(
    sprites: list[tuple[int, list[tuple[int, int]]]], force_shift: bool
) -> list[list[tuple[int, int]]] | None:
    result = [list(cells) for _color, cells in sprites]
    if not force_shift:
        return result

    candidates: list[tuple[int, tuple[int, int], list[tuple[int, int]]]] = []
    occupied = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for idx, (_color, cells) in enumerate(sprites):
        if idx == 0:
            for r, c in cells:
                occupied[r, c] = True
            continue

        for dr, dc in CARDINAL_DIRS:
            shifted = _shift_cells(cells, -dr, -dc)
            if not all(0 <= r < PANEL_SIZE and 0 <= c < PANEL_SIZE for r, c in shifted):
                continue
            if not any(occupied[r, c] for r, c in shifted):
                continue
            valid = _one_step_options(shifted, occupied)
            if valid == [(dr, dc)]:
                candidates.append((idx, (dr, dc), shifted))

        for r, c in cells:
            occupied[r, c] = True

    if not candidates:
        return None

    idx, _direction, shifted = random.choice(candidates)
    result[idx] = shifted
    return result


def _build_panel(color: int, sprite_cells: list[tuple[int, int]]) -> np.ndarray | None:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
    _paint(panel, sprite_cells, color)

    max_distractor = min(4, len(sprite_cells) - 1)
    if max_distractor < 1:
        return None

    forbidden = _adjacent_forbidden(sprite_cells)
    distractor_size = random.randint(1, max_distractor)
    distractor = _random_polyomino(distractor_size, forbidden=forbidden)
    if distractor is None:
        return None
    _paint(panel, distractor, color)

    comps = _components(panel == color)
    if len(comps) != 2:
        return None
    areas = sorted(len(comp) for comp in comps)
    if not (areas[0] < areas[1]):
        return None
    return panel


def _assemble_input(panels: list[np.ndarray]) -> np.ndarray:
    grid = np.full(GRID_SHAPE, BG, dtype=int)
    for start, panel in zip(PANEL_STARTS, panels):
        grid[:, start : start + PANEL_SIZE] = panel
    for col in SEPARATOR_COLS:
        grid[:, col] = SEP
    return grid


def _render_output(sprites: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray:
    out = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
    for color, cells in sprites:
        _paint(out, cells, color)
    return out


def _attempt_generate(force_shift: bool) -> np.ndarray | None:
    colors = random.sample(FG_COLORS, NUM_PANELS)
    sprites = _sample_output_sprites(colors)
    if sprites is None:
        return None

    input_cells = _choose_input_sprite_cells(sprites, force_shift=force_shift)
    if input_cells is None:
        return None

    panels = []
    for idx, (color, _target_cells) in enumerate(sprites):
        panel = _build_panel(color, input_cells[idx])
        if panel is None:
            return None
        panels.append(panel)

    grid = _assemble_input(panels)
    expected = _render_output(sprites)
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        force_shift = random.random() < 0.5
        grid = _attempt_generate(force_shift=force_shift)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
