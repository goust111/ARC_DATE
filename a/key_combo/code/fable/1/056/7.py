from __future__ import annotations

import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_shared_arc_utils_fable_1_056_7", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"unable to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BG = 0
BAR_COLOR = 5
PALETTE = tuple(color for color in range(1, 10) if color != BAR_COLOR)

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (14, 24)
COUNT_RANGE = (1, 4)
MAX_ATTEMPTS = 800

Coord = tuple[int, int]


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = map(int, cell)
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _left_height(bar_h: int, step: int, distance: int, rows: int) -> int:
    return min(int(rows), int(bar_h) + int(step) * int(distance))


def _right_height(bar_h: int, step: int, distance: int) -> int:
    return int(bar_h) - int(step) * int(distance)


def _right_positive_columns(bar_h: int, step: int, width: int) -> int:
    count = 0
    for distance in range(1, int(width) + 1):
        if _right_height(bar_h, step, distance) <= 0:
            break
        count += 1
    return count


def _render_output(
    shape: tuple[int, int],
    bar_c: int,
    bar_h: int,
    left_color: int,
    left_step: int,
    right_color: int,
    right_step: int,
    *,
    right_mode: str = "decay",
) -> np.ndarray:
    rows, cols = map(int, shape)
    out = np.full((rows, cols), BG, dtype=int)

    for c in range(0, int(bar_c)):
        distance = int(bar_c) - int(c)
        height = _left_height(bar_h, left_step, distance, rows)
        out[:height, c] = int(left_color)

    for c in range(int(bar_c) + 1, cols):
        distance = int(c) - int(bar_c)
        if right_mode == "decay":
            height = _right_height(bar_h, right_step, distance)
            if height <= 0:
                break
        elif right_mode == "grow":
            height = min(rows, int(bar_h) + int(right_step) * distance)
        else:
            raise ValueError(f"unknown right_mode: {right_mode}")
        out[: min(rows, height), c] = int(right_color)

    out[: int(bar_h), int(bar_c)] = BAR_COLOR
    return out


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rows, cols = map(int, g.shape)
    bg = int(get_bg(g))

    bar_candidates: list[dict] = []
    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[BAR_COLOR],
        return_format="dict",
    ):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 != 0 or c0 != c1:
            continue
        if int(comp["area"]) != int(r1 - r0 + 1):
            continue
        if np.any(g[r1 + 1 :, c0] == BAR_COLOR):
            continue
        bar_candidates.append({"c": c0, "h": r1 - r0 + 1})

    assert len(bar_candidates) == 1, "expected exactly one top-anchored grey vertical bar"
    bar = bar_candidates[0]
    bar_c = int(bar["c"])
    bar_h = int(bar["h"])
    assert 0 < bar_h <= rows

    other = np.where((g != bg) & (g != BAR_COLOR), g, bg)
    clusters = find_components(
        other,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert len(clusters) == 2, "expected exactly two parameter clusters"

    parsed = {"bg": bg, "bar_c": bar_c, "bar_h": bar_h}
    for comp in clusters:
        coords = np.asarray(comp["coords"], dtype=int)
        rs = coords[:, 0]
        cs = coords[:, 1]
        colors = sorted(set(map(int, g[rs, cs])))
        assert len(colors) == 1 and colors[0] not in {bg, BAR_COLOR}

        if int(cs.max()) <= int(bar_c) - 2:
            side = "left"
        elif int(cs.min()) >= int(bar_c) + 2:
            side = "right"
        else:
            raise AssertionError("parameter clusters must stay off the bar column and its adjacent columns")

        assert side not in parsed, f"duplicate {side} cluster"
        parsed[side] = {
            "color": int(colors[0]),
            "count": int(coords.shape[0]),
            "coords": [tuple(map(int, cell)) for cell in coords],
        }

    assert "left" in parsed and "right" in parsed, "expected one cluster on each side of the bar"
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the grey top bar into opposite left/right column gradients whose rates and colors come from two remote pixel clusters.

    Input structure:
    - The background is black.
    - A single grey width-1 vertical bar starts on the top edge and runs downward contiguously for height H.
    - Exactly two separate monochrome parameter clusters remain: one wholly on the left of the bar and one wholly on the right, with neither cluster touching the bar column or its adjacent columns.

    Rule:
    - Keep the grey bar.
    - Let the left cluster have color cL and area nL: every column at distance d to the left of the bar is filled from the top downward with cL for height H + nL*d, clipped at the bottom edge.
    - Let the right cluster have color cR and area nR: every column at distance d to the right of the bar is filled from the top downward with cR for height H - nR*d; once this height is non-positive, that column and all farther right columns stay black.
    - Erase both parameter clusters from the output.

    Key insight:
    - Each side cluster is a two-part legend at once: its visible color says what to paint on that side, while its pixel count sets that side's per-column growth or decay rate.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _render_output(
        tuple(map(int, grid.shape)),
        int(scene["bar_c"]),
        int(scene["bar_h"]),
        int(scene["left"]["color"]),
        int(scene["left"]["count"]),
        int(scene["right"]["color"]),
        int(scene["right"]["count"]),
    )


def _sample_parameters() -> dict | None:
    rows = random.randint(*GRID_H_RANGE)
    cols = random.randint(*GRID_W_RANGE)
    bar_c = random.randint(4, cols - 5)
    left_width = int(bar_c)
    right_width = int(cols - bar_c - 1)
    left_truncates = random.choice([True, False])
    right_stops_early = random.choice([True, False])

    for _ in range(300):
        bar_h = random.randint(4, rows - 4)
        left_step = random.randint(*COUNT_RANGE)
        right_step = random.randint(*COUNT_RANGE)

        if bar_h + 2 * left_step >= rows:
            continue
        if left_truncates:
            if bar_h + left_step >= rows:
                continue
            if bar_h + left_step * left_width < rows:
                continue
        else:
            if bar_h + left_step * left_width >= rows:
                continue

        positive_right = _right_positive_columns(bar_h, right_step, right_width)
        if positive_right < 2:
            continue
        if right_stops_early:
            if not (2 <= positive_right < right_width):
                continue
        else:
            if positive_right != right_width:
                continue

        left_color, right_color = random.sample(PALETTE, 2)
        return {
            "rows": rows,
            "cols": cols,
            "bar_c": bar_c,
            "bar_h": bar_h,
            "left_color": left_color,
            "left_count": left_step,
            "right_color": right_color,
            "right_count": right_step,
            "left_truncates": left_truncates,
            "right_stops_early": right_stops_early,
        }

    return None


def _paint_masks(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = int(scene["rows"])
    cols = int(scene["cols"])
    bar_c = int(scene["bar_c"])
    bar_h = int(scene["bar_h"])
    left_step = int(scene["left_count"])
    right_step = int(scene["right_count"])

    left_mask = np.zeros((rows, cols), dtype=bool)
    right_mask = np.zeros((rows, cols), dtype=bool)

    for c in range(0, bar_c):
        distance = bar_c - c
        left_mask[: _left_height(bar_h, left_step, distance, rows), c] = True

    for c in range(bar_c + 1, cols):
        distance = c - bar_c
        height = _right_height(bar_h, right_step, distance)
        if height <= 0:
            break
        right_mask[:height, c] = True

    return left_mask, right_mask


def _candidate_mask(scene: dict, side: str) -> np.ndarray:
    rows = int(scene["rows"])
    cols = int(scene["cols"])
    bar_c = int(scene["bar_c"])
    bar_h = int(scene["bar_h"])
    left_mask, right_mask = _paint_masks(scene)

    mask = np.zeros((rows, cols), dtype=bool)
    if side == "left":
        mask[bar_h + 1 :, : bar_c - 1] = True
        mask &= ~left_mask
    elif side == "right":
        mask[bar_h + 1 :, bar_c + 2 :] = True
        mask &= ~right_mask
    else:
        raise ValueError(f"unknown side: {side}")
    return mask


def _connected_cells(mask: np.ndarray, count: int) -> list[Coord] | None:
    coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    if len(coords) < int(count):
        return None

    allowed = set(coords)
    neighbors = {
        cell: [nbr for nbr in _neighbors4(cell) if nbr in allowed]
        for cell in coords
    }
    starts = [cell for cell in coords if int(count) == 1 or neighbors[cell]]
    if not starts:
        return None

    for _ in range(300):
        start = random.choice(starts)
        cells = {start}
        frontier = set(neighbors[start])

        while frontier and len(cells) < int(count):
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            if nxt in cells:
                continue
            cells.add(nxt)
            for nbr in neighbors[nxt]:
                if nbr not in cells:
                    frontier.add(nbr)

        if len(cells) == int(count):
            return sorted(cells)

    return None


def _build_input(scene: dict) -> np.ndarray | None:
    grid = np.full((int(scene["rows"]), int(scene["cols"])), BG, dtype=int)
    grid[: int(scene["bar_h"]), int(scene["bar_c"])] = BAR_COLOR

    left_cells = _connected_cells(_candidate_mask(scene, "left"), int(scene["left_count"]))
    if left_cells is None:
        return None
    right_cells = _connected_cells(_candidate_mask(scene, "right"), int(scene["right_count"]))
    if right_cells is None:
        return None

    for r, c in left_cells:
        grid[r, c] = int(scene["left_color"])
    for r, c in right_cells:
        grid[r, c] = int(scene["right_color"])

    scene["left_cells"] = left_cells
    scene["right_cells"] = right_cells
    return grid


def _matches_scene(scene: dict, parsed: dict) -> bool:
    return bool(
        int(scene["bar_c"]) == int(parsed["bar_c"])
        and int(scene["bar_h"]) == int(parsed["bar_h"])
        and int(scene["left_color"]) == int(parsed["left"]["color"])
        and int(scene["left_count"]) == int(parsed["left"]["count"])
        and int(scene["right_color"]) == int(parsed["right"]["color"])
        and int(scene["right_count"]) == int(parsed["right"]["count"])
    )


def _perturbed_step(scene: dict, key: str) -> int:
    value = int(scene[key])
    return value + 1 if value < COUNT_RANGE[1] else value - 1


def _scene_semantic_ok(grid: np.ndarray, scene: dict) -> bool:
    try:
        parsed = _parse_scene(grid)
    except Exception:
        return False
    if not _matches_scene(scene, parsed):
        return False

    expected = _render_output(
        tuple(map(int, grid.shape)),
        int(scene["bar_c"]),
        int(scene["bar_h"]),
        int(scene["left_color"]),
        int(scene["left_count"]),
        int(scene["right_color"]),
        int(scene["right_count"]),
    )
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if not all(int(actual[r, c]) == BG for r, c in scene["left_cells"]):
        return False
    if not all(int(actual[r, c]) == BG for r, c in scene["right_cells"]):
        return False

    alt_left_step = _render_output(
        tuple(map(int, grid.shape)),
        int(scene["bar_c"]),
        int(scene["bar_h"]),
        int(scene["left_color"]),
        _perturbed_step(scene, "left_count"),
        int(scene["right_color"]),
        int(scene["right_count"]),
    )
    if np.array_equal(actual, alt_left_step):
        return False

    alt_right_step = _render_output(
        tuple(map(int, grid.shape)),
        int(scene["bar_c"]),
        int(scene["bar_h"]),
        int(scene["left_color"]),
        int(scene["left_count"]),
        int(scene["right_color"]),
        _perturbed_step(scene, "right_count"),
    )
    if np.array_equal(actual, alt_right_step):
        return False

    alt_swap_colors = _render_output(
        tuple(map(int, grid.shape)),
        int(scene["bar_c"]),
        int(scene["bar_h"]),
        int(scene["right_color"]),
        int(scene["left_count"]),
        int(scene["left_color"]),
        int(scene["right_count"]),
    )
    if np.array_equal(actual, alt_swap_colors):
        return False

    alt_right_grows = _render_output(
        tuple(map(int, grid.shape)),
        int(scene["bar_c"]),
        int(scene["bar_h"]),
        int(scene["left_color"]),
        int(scene["left_count"]),
        int(scene["right_color"]),
        int(scene["right_count"]),
        right_mode="grow",
    )
    if np.array_equal(actual, alt_right_grows):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    scene = _sample_parameters()
    if scene is None:
        return None
    grid = _build_input(scene)
    if grid is None:
        return None
    if not _scene_semantic_ok(grid, scene):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
