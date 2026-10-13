from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg
import random
from scipy.ndimage import label


def _nearest_non_overlapping_shift(coords: np.ndarray, occupied: np.ndarray) -> tuple[int, int]:
    H, W = occupied.shape

    def valid(dr: int, dc: int) -> bool:
        rr = coords[:, 0] + int(dr)
        cc = coords[:, 1] + int(dc)
        return (
            int(rr.min()) >= 0
            and int(cc.min()) >= 0
            and int(rr.max()) < H
            and int(cc.max()) < W
            and not bool(occupied[rr, cc].any())
        )

    if valid(0, 0):
        return 0, 0

    candidates: list[tuple[int, int, int]] = []
    for dr in range(-(H - 1), H):
        for dc in range(-(W - 1), W):
            if valid(dr, dc):
                candidates.append((abs(int(dr)) + abs(int(dc)), int(dr), int(dc)))

    assert candidates, "sprite must have a non-overlapping placement"
    best_distance = min(dist for dist, _dr, _dc in candidates)
    best = [(dr, dc) for dist, dr, dc in candidates if dist == best_distance]
    assert len(best) == 1, "nearest non-overlapping placement must be unique"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the larger sprite from each separated panel and overlay those sprites into one 5x5 canvas.

    Input structure:
    - The input is a 5x17 grid: three 5x5 panels separated by one-cell black columns.
    - Each panel has the same background color and exactly one foreground color split into two 4-connected components.

    Rule:
    - In each panel, discard the smaller foreground component and keep the unique largest component as the sprite.
    - Process panels left to right, placing each sprite at its panel coordinates on a 5x5 background canvas.
    - If a sprite would cover an already placed sprite, slide it by the unique nearest in-bounds offset that avoids overlap.

    Key insight:
    - The extra same-colored component in each panel is a distractor; the final answer is the non-overlapping overlay of the three largest components.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    panel_size = 5
    assert H == panel_size and W == 3 * panel_size + 2, "expected three 5x5 panels"

    bg = int(get_bg(g))
    sep = int(g[0, panel_size])
    assert sep != bg, "separator color must differ from panel background"
    assert np.all(g[:, panel_size] == sep) and np.all(
        g[:, 2 * panel_size + 1] == sep
    ), "panels must be separated by uniform columns"

    out = np.full((panel_size, panel_size), bg, dtype=int)
    occupied = np.zeros_like(out, dtype=bool)
    sprite_colors: list[int] = []

    for c0 in (0, panel_size + 1, 2 * panel_size + 2):
        panel = g[:, c0 : c0 + panel_size]
        panel_colors = [int(c) for c in np.unique(panel) if int(c) != bg]
        assert len(panel_colors) == 1, "each panel must contain one foreground color"
        color = panel_colors[0]
        sprite_colors.append(color)

        comps = find_components(panel, bg=bg, connectivity=4, monochromatic=True, colors=[color])
        assert len(comps) == 2, "each panel must contain exactly two components"
        areas = [int(comp["area"]) for comp in comps]
        largest = max(areas)
        assert areas.count(largest) == 1, "largest component must be unique"
        sprite = comps[areas.index(largest)]
        coords = np.asarray(sprite["coords"], dtype=int)

        dr, dc = _nearest_non_overlapping_shift(coords, occupied)
        rr = coords[:, 0] + int(dr)
        cc = coords[:, 1] + int(dc)
        out[rr, cc] = color
        occupied[rr, cc] = True

    assert len(set(sprite_colors)) == len(sprite_colors), "panel sprite colors must be distinct"
    return out

# --------------- generator (refactored) ---------------

# Canvas and panel geometry.
BG = 8
SEP = 0
PANEL_SIZE = 5
GRID_WIDTH = 17
PANEL_STARTS = (0, 6, 12)
SEPARATOR_COLS = (5, 11)
NUM_PANELS = 3

# Colors and connected-component rules.
AVAILABLE_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
ORTHOGONAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
REQUIRED_PANEL_COMPONENTS = 2

# Sprite and distractor size bounds.
MIN_SPRITE_SIZE = 2
MAX_SPRITE_SIZE = 9
MAX_DISTRACTOR_SIZE = 4

# Shift branch and retry budget.
SHIFT_BRANCH_PROBABILITY = 0.20
OVERLAP_OFFSETS = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (2, 0), (-2, 0))
GENERATION_ATTEMPTS = 300


def _coords_in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    return (
        int(coords[:, 0].min()) >= 0
        and int(coords[:, 1].min()) >= 0
        and int(coords[:, 0].max()) < shape[0]
        and int(coords[:, 1].max()) < shape[1]
    )


def _shift_is_valid(coords: np.ndarray, occupied: np.ndarray, dr: int, dc: int) -> bool:
    rr = coords[:, 0] + dr
    cc = coords[:, 1] + dc
    return (
        int(rr.min()) >= 0
        and int(cc.min()) >= 0
        and int(rr.max()) < occupied.shape[0]
        and int(cc.max()) < occupied.shape[1]
        and not bool(occupied[rr, cc].any())
    )


def _nearest_non_overlapping_shift(coords: np.ndarray, occupied: np.ndarray) -> tuple[int, int] | None:
    if _shift_is_valid(coords, occupied, 0, 0):
        return 0, 0

    candidates = []
    for dr in range(-(occupied.shape[0] - 1), occupied.shape[0]):
        for dc in range(-(occupied.shape[1] - 1), occupied.shape[1]):
            if _shift_is_valid(coords, occupied, dr, dc):
                candidates.append((abs(dr) + abs(dc), dr, dc))
    if not candidates:
        return None

    best_distance = min(dist for dist, _dr, _dc in candidates)
    best = [(dr, dc) for dist, dr, dc in candidates if dist == best_distance]
    return best[0] if len(best) == 1 else None


def _paint_cells(canvas: np.ndarray, cells: list[tuple[int, int]], value: int | bool) -> None:
    for r, c in cells:
        canvas[r, c] = value


def _overlap_input_cells(sprite_cells: list[tuple[int, int]], occupied: np.ndarray) -> list[tuple[int, int]] | None:
    """Return shifted input cells whose solver shift restores the output position."""
    coords = np.asarray(sprite_cells, dtype=int)
    offsets = list(OVERLAP_OFFSETS)
    random.shuffle(offsets)
    offsets.sort(key=lambda d: (abs(d[0]) + abs(d[1]), d != (0, 1)))

    for in_dr, in_dc in offsets:
        shifted = coords + np.array([in_dr, in_dc], dtype=int)
        if not _coords_in_bounds(shifted, occupied.shape):
            continue
        if not bool(occupied[shifted[:, 0], shifted[:, 1]].any()):
            continue
        if _nearest_non_overlapping_shift(shifted, occupied) == (-in_dr, -in_dc):
            return [(int(r), int(c)) for r, c in shifted]
    return None


def _random_polyomino(
    size: int, grid_h: int = PANEL_SIZE, grid_w: int = PANEL_SIZE, forbidden: set | None = None
) -> list[tuple[int, int]]:
    """Grow a random 4-connected polyomino, or return [] if growth gets stuck."""
    if forbidden is None:
        forbidden = set()

    free = [(r, c) for r in range(grid_h) for c in range(grid_w) if (r, c) not in forbidden]
    if not free:
        return []

    start = random.choice(free)
    cells = {start}
    frontier = set()
    for dr, dc in ORTHOGONAL_STEPS:
        nr, nc = start[0] + dr, start[1] + dc
        if (nr, nc) not in forbidden:
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                frontier.add((nr, nc))

    while len(cells) < size and frontier:
        nxt = random.choice(list(frontier))
        cells.add(nxt)
        frontier.discard(nxt)
        for dr, dc in ORTHOGONAL_STEPS:
            nr, nc = nxt[0] + dr, nxt[1] + dc
            if 0 <= nr < grid_h and 0 <= nc < grid_w and (nr, nc) not in cells and (nr, nc) not in forbidden:
                frontier.add((nr, nc))

    if len(cells) < size:
        return []
    return list(cells)


def _sample_sprites(colors: list[int]) -> list[tuple[int, list[tuple[int, int]]]] | None:
    """Choose non-overlapping output sprites and keep room for later panels."""
    occupied = set()
    sprites = []

    for idx, color in enumerate(colors):
        remaining_panels = NUM_PANELS - idx - 1
        max_possible = min(MAX_SPRITE_SIZE, PANEL_SIZE * PANEL_SIZE - len(occupied) - remaining_panels * MIN_SPRITE_SIZE)
        if max_possible < MIN_SPRITE_SIZE:
            return None

        sprite_size = random.randint(MIN_SPRITE_SIZE, max_possible)
        cells = _random_polyomino(sprite_size, forbidden=occupied)
        if not cells:
            return None

        sprites.append((color, cells))
        occupied.update(cells)

    return sprites


def _render_output(sprites: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray:
    output = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
    for color, cells in sprites:
        _paint_cells(output, cells, color)
    return output


def _choose_panel_sprite_cells(
    sprites: list[tuple[int, list[tuple[int, int]]]]
) -> list[list[tuple[int, int]]] | None:
    panel_sprite_cells = [list(cells) for _color, cells in sprites]
    if random.random() >= SHIFT_BRANCH_PROBABILITY:
        return panel_sprite_cells

    earlier = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for _color, cells in sprites[:2]:
        _paint_cells(earlier, cells, True)

    shifted = _overlap_input_cells(sprites[2][1], earlier)
    if shifted is None:
        return None

    panel_sprite_cells[2] = shifted
    return panel_sprite_cells


def _distractor_forbidden_cells(sprite_cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    sprite_set = set(sprite_cells)
    sprite_adjacent = set()
    for r, c in sprite_cells:
        for dr, dc in ORTHOGONAL_STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE:
                sprite_adjacent.add((nr, nc))
    return sprite_set | sprite_adjacent


def _build_panel(color: int, sprite_cells: list[tuple[int, int]]) -> np.ndarray | None:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
    _paint_cells(panel, sprite_cells, color)

    max_distractor = min(len(sprite_cells) - 1, MAX_DISTRACTOR_SIZE)
    if max_distractor < 1:
        return None

    distractor_size = random.randint(1, max_distractor)
    distractor_cells = _random_polyomino(distractor_size, forbidden=_distractor_forbidden_cells(sprite_cells))
    if not distractor_cells:
        return None
    _paint_cells(panel, distractor_cells, color)

    _, n_comps = label(panel == color, structure=STRUCT4)
    if n_comps != REQUIRED_PANEL_COMPONENTS:
        return None
    return panel


def _build_panels(
    sprites: list[tuple[int, list[tuple[int, int]]]], panel_sprite_cells: list[list[tuple[int, int]]]
) -> list[np.ndarray] | None:
    panels = []
    for idx, (color, _output_cells) in enumerate(sprites):
        panel = _build_panel(color, panel_sprite_cells[idx])
        if panel is None:
            return None
        panels.append(panel)
    return panels


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    grid = np.full((PANEL_SIZE, GRID_WIDTH), BG, dtype=int)
    for start, panel in zip(PANEL_STARTS, panels):
        grid[:, start : start + PANEL_SIZE] = panel
    for col in SEPARATOR_COLS:
        grid[:, col] = SEP
    return grid


def _oracle_accepts(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(grid), expected_output))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(AVAILABLE_COLORS, NUM_PANELS)
    sprites = _sample_sprites(colors)
    if sprites is None:
        return None

    output = _render_output(sprites)
    panel_sprite_cells = _choose_panel_sprite_cells(sprites)
    if panel_sprite_cells is None:
        return None

    panels = _build_panels(sprites, panel_sprite_cells)
    if panels is None:
        return None

    grid = _assemble_grid(panels)
    return grid if _oracle_accepts(grid, output) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
