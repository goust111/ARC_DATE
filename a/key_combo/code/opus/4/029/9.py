from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np

try:
    from arc_utils import Color, find_components, in_bounds, neighbors4, neighbors8
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_shared",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    Color = _ARC_UTILS.Color
    find_components = _ARC_UTILS.find_components
    in_bounds = _ARC_UTILS.in_bounds
    neighbors4 = _ARC_UTILS.neighbors4
    neighbors8 = _ARC_UTILS.neighbors8


FG_COLORS = (Color.BLUE, Color.RED)
MARKER_COLOR = Color.GREEN
BG = Color.BLACK

BASE_TILE = np.array(
    [
        [Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE],
        [Color.RED, Color.RED, Color.RED, Color.BLUE],
        [Color.BLACK, Color.RED, Color.RED, Color.RED],
        [Color.RED, Color.RED, Color.RED, Color.RED],
        [Color.RED, Color.RED, Color.RED, Color.RED],
    ],
    dtype=int,
)

GEN_REPEATS = (3, 5)
GEN_ERASE_FRAC = (0.28, 0.55)
GEN_RETRIES = 400


def _swap_blue_red(tile: np.ndarray) -> np.ndarray:
    out = tile.copy()
    out[tile == Color.BLUE] = Color.RED
    out[tile == Color.RED] = Color.BLUE
    return out


def _transform_tile(tile: np.ndarray) -> np.ndarray:
    out = tile.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = _swap_blue_red(out)
    return np.asarray(out, dtype=int)


def _period_candidates(grid: np.ndarray) -> list[tuple[int, int, dict[tuple[int, int], int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []

    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            phase_to_color: dict[tuple[int, int], int] = {}
            consistent = True
            saw_fg = False
            for r in range(h):
                if not consistent:
                    break
                for c in range(w):
                    color = int(g[r, c])
                    if color not in FG_COLORS:
                        continue
                    saw_fg = True
                    key = (r % pr, c % pc)
                    known = phase_to_color.get(key)
                    if known is None:
                        phase_to_color[key] = color
                    elif known != color:
                        consistent = False
                        break
            if consistent and saw_fg:
                candidates.append((pr, pc, phase_to_color))
    return candidates


def _infer_period(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int]]:
    candidates = _period_candidates(grid)
    assert candidates, "expected at least one foreground-consistent period"
    min_area = min(pr * pc for pr, pc, _phase in candidates)
    smallest = [(pr, pc, phase) for pr, pc, phase in candidates if pr * pc == min_area]
    assert len(smallest) == 1, "expected a unique smallest period"
    return smallest[0]


def _complete_wallpaper(grid: np.ndarray) -> np.ndarray:
    pr, pc, phase_to_color = _infer_period(grid)
    h, w = map(int, grid.shape)
    out = np.full((h, w), BG, dtype=int)
    for r in range(h):
        for c in range(w):
            color = phase_to_color.get((r % pr, c % pc))
            if color is not None:
                out[r, c] = int(color)
    return out


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r) + 1
        cc = int(c) + 1
        out[int(r), int(c)] = bool(padded[rr - 1 : rr + 2, cc - 1 : cc + 2].all())
    return out


def _component_infos(grid: np.ndarray) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for comp in find_components(grid, bg=BG, colors=FG_COLORS, connectivity=4):
        mask = np.zeros_like(grid, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        interior = _interior_mask(mask)
        candidate_holes: list[tuple[int, int]] = []
        for r in range(1, grid.shape[0] - 1):
            for c in range(1, grid.shape[1] - 1):
                if int(grid[r, c]) != BG:
                    continue
                if all(mask[nr, nc] for nr, nc in neighbors4(r, c)):
                    candidate_holes.append((int(r), int(c)))
        infos.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "mask": mask,
                "interior": interior,
                "candidate_holes": candidate_holes,
                "area": int(comp["area"]),
            }
        )
    return infos


def _selected_component_indices(
    components: list[dict[str, object]],
    marker_positions: list[tuple[int, int]],
) -> set[int]:
    selected: set[int] = set()
    for idx, info in enumerate(components):
        mask = info["mask"]
        for r, c in marker_positions:
            if any(in_bounds(mask.shape, nr, nc) and mask[nr, nc] for nr, nc in neighbors4(r, c)):
                selected.add(idx)
                break
    return selected


def _paint_selected_interiors(
    base_grid: np.ndarray,
    marker_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(base_grid, dtype=int).copy()
    components = _component_infos(out)
    for idx in _selected_component_indices(components, marker_positions):
        out[components[idx]["interior"]] = MARKER_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the periodic wallpaper and recolor only the marked component interiors.

    Input structure:
    - The grid contains a blue/red periodic wallpaper with some foreground cells erased to black.
    - One or two green marker pixels sit on black cavity cells inside completed same-color components.
    - Green markers are not wallpaper evidence; only blue and red pixels define the repeat phase classes.

    Rule:
    - Ignore black and green cells when inferring the unique smallest row and column periods for blue/red evidence.
    - Fill every residue class that has foreground evidence across the whole grid, leaving unevidenced classes black.
    - In the completed wallpaper, find same-color 4-connected components whose interior cavity contains a green marker.
    - Recolor only those components' strict 3x3-solid interior pixels to green, then erase the original marker cells to black.

    Key insight:
    - The marker chooses which completed component to edit, but the component must be reconstructed first because erased wallpaper pixels would otherwise destroy the interior test.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    completed = _complete_wallpaper(g)
    marker_positions = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER_COLOR)]
    out = _paint_selected_interiors(completed, marker_positions)
    out[g == MARKER_COLOR] = BG
    return out


def _phase_occurrences(
    wallpaper: np.ndarray,
    tile_shape: tuple[int, int],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    pr, pc = map(int, tile_shape)
    phases: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(wallpaper.shape[0]):
        for c in range(wallpaper.shape[1]):
            if int(wallpaper[r, c]) not in FG_COLORS:
                continue
            phases.setdefault((r % pr, c % pc), []).append((int(r), int(c)))
    return phases


def _erase_foreground(
    wallpaper: np.ndarray,
    selected_components: list[dict[str, object]],
    marker_positions: list[tuple[int, int]],
    tile_shape: tuple[int, int],
) -> np.ndarray:
    inp = wallpaper.copy()
    for r, c in marker_positions:
        inp[r, c] = MARKER_COLOR

    phase_occ = _phase_occurrences(wallpaper, tile_shape)
    protected: set[tuple[int, int]] = set()

    for coords in phase_occ.values():
        protected.add(random.choice(coords))

    for info in selected_components:
        interior_coords = [tuple(map(int, rc)) for rc in np.argwhere(info["interior"])]
        random.shuffle(interior_coords)
        for coord in interior_coords:
            if coord not in protected:
                break
        else:
            coord = None
        if coord is not None:
            protected.add(coord)

    erase_prob = random.uniform(*GEN_ERASE_FRAC)
    for coords in phase_occ.values():
        for coord in coords:
            if coord in protected:
                continue
            if random.random() < erase_prob:
                inp[coord] = BG
    return inp


def _shortcut_without_completion(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sparse = g.copy()
    marker_positions = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER_COLOR)]
    sparse[sparse == MARKER_COLOR] = BG
    out = _paint_selected_interiors(sparse, marker_positions)
    out[g == MARKER_COLOR] = BG
    return out


def _validate_case(
    inp: np.ndarray,
    wallpaper: np.ndarray,
    expected: np.ndarray,
    marker_positions: list[tuple[int, int]],
    tile_shape: tuple[int, int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    without_markers = inp.copy()
    without_markers[without_markers == MARKER_COLOR] = BG
    try:
        pr, pc, phase_to_color = _infer_period(without_markers)
    except AssertionError:
        return False
    if (pr, pc) != tuple(map(int, tile_shape)):
        return False

    seen_phases = set(phase_to_color)
    true_phases = {
        (r % tile_shape[0], c % tile_shape[1])
        for r in range(wallpaper.shape[0])
        for c in range(wallpaper.shape[1])
        if int(wallpaper[r, c]) in FG_COLORS
    }
    if seen_phases != true_phases:
        return False

    components = _component_infos(wallpaper)
    selected = _selected_component_indices(components, marker_positions)
    if len(selected) != len(marker_positions):
        return False

    if sum(int(components[idx]["interior"].any()) for idx in selected) != len(selected):
        return False

    unselected_with_interior = [
        info
        for idx, info in enumerate(components)
        if idx not in selected and bool(info["interior"].any())
    ]
    if not unselected_with_interior:
        return False

    if np.array_equal(_shortcut_without_completion(inp), expected):
        return False

    for r, c in marker_positions:
        if int(wallpaper[r, c]) != BG:
            return False
    return True


def _build_case() -> np.ndarray | None:
    tile = _transform_tile(BASE_TILE)
    rep_r = random.randint(*GEN_REPEATS)
    rep_c = random.randint(*GEN_REPEATS)
    wallpaper = np.tile(tile, (rep_r, rep_c))

    components = _component_infos(wallpaper)
    selectable = [
        info
        for info in components
        if bool(info["interior"].any()) and len(info["candidate_holes"]) > 0
    ]
    if len(selectable) < 2:
        return None

    max_markers = min(2, len(selectable) - 1)
    marker_count = 2 if max_markers >= 2 and random.random() < 0.45 else 1
    chosen = random.sample(selectable, marker_count)
    marker_positions = [random.choice(info["candidate_holes"]) for info in chosen]
    if len(set(marker_positions)) != marker_count:
        return None

    expected = wallpaper.copy()
    for info in chosen:
        expected[info["interior"]] = MARKER_COLOR

    inp = _erase_foreground(
        wallpaper,
        chosen,
        marker_positions,
        tile.shape,
    )
    if _validate_case(inp, wallpaper, expected, marker_positions, tile.shape):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic selective-recolor ARC input")

