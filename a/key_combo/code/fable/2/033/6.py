from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_SPEC is not None and _ARC_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds


BG = int(Color.BLACK)
BLOCK = int(Color.BLUE)
RED = int(Color.RED)
GRAY = int(Color.GRAY)
SOURCE_COLORS = (RED, GRAY)

GEN_RETRIES = 400
HEIGHT_RANGE = (8, 11)
SOURCE_COUNT_OPTIONS = (2, 3)
CORRIDOR_GAP_RANGE = (1, 3)
EXTRA_WIDTH_CAP = 6


def _flip_red_gray(color: int) -> int:
    assert color in SOURCE_COLORS
    return GRAY if int(color) == RED else RED


def _simulate_drop(
    block_mask: np.ndarray, start_col: int, start_color: int
) -> tuple[list[tuple[int, int]], list[int], list[str]]:
    h, w = map(int, block_mask.shape)
    r, c = 0, int(start_col)
    color = int(start_color)
    cells = [(r, c)]
    colors = [color]
    moves: list[str] = []

    while True:
        diag = (r + 1, c + 1)
        if in_bounds((h, w), *diag) and not bool(block_mask[diag]):
            r, c = diag
            color = _flip_red_gray(color)
            moves.append("D")
            cells.append((r, c))
            colors.append(color)
            continue

        down = (r + 1, c)
        if in_bounds((h, w), *down) and not bool(block_mask[down]):
            r, c = down
            moves.append("V")
            cells.append((r, c))
            colors.append(color)
            continue

        break

    return cells, colors, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each top-row red/gray source as a rain drop that prefers down-right steps and only keeps color through forced vertical falls.

    Input structure:
    - The top row contains one-pixel rain sources colored red or gray.
    - Blue single pixels elsewhere in the grid are blockers; the remaining cells are background.
    - Different sources are arranged so their rain trajectories do not overlap.

    Rule:
    - For every top-row source, simulate a drop from that source cell.
    - Each step first tries the down-right diagonal cell; if that cell is out of bounds or blue, the drop instead tries the cell directly below; if both are unavailable, the drop stops.
    - Writing color depends on direction: a diagonal step flips red and gray, while a vertical step keeps the previous color. Obstacles and untouched cells stay unchanged.

    Key insight:
    - The visible color phase is controlled by move direction rather than by elapsed time, so blue blockers reveal themselves by creating same-color vertical runs inside otherwise alternating diagonals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source_cols = [int(c) for c in np.flatnonzero(np.isin(g[0], SOURCE_COLORS))]
    assert source_cols, "expected at least one red/gray top-row source"

    out = g.copy()
    block_mask = (g == BLOCK)
    for start_col in source_cols:
        start_color = int(g[0, start_col])
        cells, colors, _moves = _simulate_drop(block_mask, start_col, start_color)
        for (r, c), color in zip(cells[1:], colors[1:]):
            out[r, c] = int(color)

    return out


def _positive_parts(total: int, count: int) -> list[int]:
    assert total >= count >= 1
    if count == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), count - 1))
    parts: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        parts.append(int(cut - prev))
        prev = cut
    return parts


def _bottom_diagonal_sequence(height: int) -> list[str]:
    return ["D"] * (int(height) - 1)


def _bottom_deflected_sequence(height: int) -> list[str]:
    length = int(height) - 1
    v_total = random.randint(2, min(4, length - 3))
    d_total = length - v_total
    assert d_total >= 3

    if v_total == 2:
        d1, d2 = _positive_parts(d_total, 2)
        return ["D"] * d1 + ["V"] * 2 + ["D"] * d2

    if v_total == 3:
        v_parts = [1, 2] if random.random() < 0.5 else [2, 1]
    else:
        v_parts = random.choice(([1, 3], [2, 2], [3, 1]))

    d1, d2, d3 = _positive_parts(d_total, 3)
    return (
        ["D"] * d1
        + ["V"] * v_parts[0]
        + ["D"] * d2
        + ["V"] * v_parts[1]
        + ["D"] * d3
    )


def _early_stop_sequence(height: int) -> list[str]:
    length = random.randint(4, int(height) - 3)
    v_total = random.randint(1, min(2, length - 2))
    d_total = length - v_total
    assert d_total >= 2
    d1, d2 = _positive_parts(d_total, 2)
    return ["D"] * d1 + ["V"] * v_total + ["D"] * d2


def _required_width(sequence: list[str], stop_early: bool) -> int:
    diag_steps = int(sequence.count("D"))
    needs_blocker_column = stop_early or ("V" in sequence)
    return diag_steps + (2 if needs_blocker_column else 1)


def _trace_planned_path(
    height: int, width: int, start_col: int, sequence: list[str], stop_early: bool
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    r, c = 0, int(start_col)
    cells = [(r, c)]
    blockers: set[tuple[int, int]] = set()

    for move in sequence:
        if move == "D":
            nr, nc = r + 1, c + 1
            if not in_bounds((height, width), nr, nc):
                raise ValueError("planned diagonal step leaves the grid")
            r, c = nr, nc
        else:
            blocker = (r + 1, c + 1)
            if not in_bounds((height, width), *blocker):
                raise ValueError("planned vertical step needs an in-bounds diagonal blocker")
            if not in_bounds((height, width), r + 1, c):
                raise ValueError("planned vertical step leaves the grid")
            blockers.add(blocker)
            r = r + 1
        cells.append((r, c))

    if stop_early:
        down_block = (r + 1, c)
        diag_block = (r + 1, c + 1)
        if not in_bounds((height, width), *down_block):
            raise ValueError("early-stop blocker below the path is out of bounds")
        if not in_bounds((height, width), *diag_block):
            raise ValueError("early-stop diagonal blocker is out of bounds")
        blockers.add(down_block)
        blockers.add(diag_block)

    return cells, blockers


def _specs_for_sample(height: int, source_count: int) -> list[dict]:
    major = random.choice(SOURCE_COLORS)
    minor = _flip_red_gray(major)

    specs = [
        {
            "kind": "diag",
            "sequence": _bottom_diagonal_sequence(height),
            "stop_early": False,
            "color": major if source_count == 3 else RED,
        },
        {
            "kind": "deflect",
            "sequence": _bottom_deflected_sequence(height),
            "stop_early": False,
            "color": major if source_count == 3 else GRAY,
        },
    ]
    if source_count == 3:
        specs.append(
            {
                "kind": "stop",
                "sequence": _early_stop_sequence(height),
                "stop_early": True,
                "color": minor,
            }
        )

    if source_count == 2 and random.random() < 0.5:
        specs[0]["color"], specs[1]["color"] = specs[1]["color"], specs[0]["color"]

    random.shuffle(specs)
    return specs


def _place_specs(height: int, specs: list[dict]) -> tuple[int, list[dict]] | None:
    widths = [_required_width(spec["sequence"], bool(spec["stop_early"])) for spec in specs]
    gaps = [random.randint(*CORRIDOR_GAP_RANGE) for _ in range(len(specs) - 1)]
    min_width = int(sum(widths) + sum(gaps))
    if min_width > 30:
        return None

    extra = random.randint(0, min(EXTRA_WIDTH_CAP, 30 - min_width))
    slack = [0] * (len(specs) + 1)
    for _ in range(extra):
        slack[random.randrange(len(slack))] += 1

    placed: list[dict] = []
    cursor = slack[0]
    for idx, spec in enumerate(specs):
        placed_spec = dict(spec)
        placed_spec["start_col"] = int(cursor)
        placed.append(placed_spec)
        cursor += widths[idx]
        if idx < len(specs) - 1:
            cursor += gaps[idx] + slack[idx + 1]

    width = int(cursor + slack[-1])
    if width > 30:
        return None
    return width, placed


def _expected_output_from_paths(grid: np.ndarray, path_specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in path_specs:
        color = int(spec["color"])
        for idx, (r, c) in enumerate(spec["cells"]):
            if idx == 0:
                continue
            if spec["sequence"][idx - 1] == "D":
                color = _flip_red_gray(color)
            out[r, c] = color
    return out


def _naive_toggle_output(grid: np.ndarray, path_specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in path_specs:
        color = int(spec["color"])
        for idx, (r, c) in enumerate(spec["cells"]):
            if idx == 0:
                continue
            color = _flip_red_gray(color)
            out[r, c] = color
    return out


def _clear_sky_path(height: int, width: int, start_col: int) -> list[tuple[int, int]]:
    clear_mask = np.zeros((height, width), dtype=bool)
    cells, _colors, _moves = _simulate_drop(clear_mask, start_col, RED)
    return cells


def _valid_sample(grid: np.ndarray, path_specs: list[dict]) -> bool:
    out = main(grid)
    if np.array_equal(out, grid):
        return False

    block_mask = (grid == BLOCK)
    occupied: set[tuple[int, int]] = set()
    any_blocker_changes_path = False
    any_vertical = False
    any_resume_after_vertical = False

    for spec in path_specs:
        start_col = int(spec["start_col"])
        color = int(spec["color"])
        cells, colors, moves = _simulate_drop(block_mask, start_col, color)
        expected_cells = list(spec["cells"])
        if cells != expected_cells:
            return False
        for pos, expected_color in zip(cells, colors):
            if int(out[pos]) != int(expected_color):
                return False
        for cell in cells:
            if cell in occupied:
                return False
            occupied.add(cell)

        sequence = list(spec["sequence"])
        any_vertical = any_vertical or ("V" in sequence)
        any_resume_after_vertical = any_resume_after_vertical or any(
            sequence[idx] == "V" and "D" in sequence[idx + 1 :]
            for idx in range(len(sequence))
        )
        if cells != _clear_sky_path(*grid.shape, start_col):
            any_blocker_changes_path = True

    if not any_vertical or not any_resume_after_vertical or not any_blocker_changes_path:
        return False

    if np.array_equal(out, _naive_toggle_output(grid, path_specs)):
        return False

    if not np.array_equal(out, _expected_output_from_paths(grid, path_specs)):
        return False

    source_colors = {int(grid[0, spec["start_col"]]) for spec in path_specs}
    if source_colors != {RED, GRAY}:
        return False

    if grid.shape[1] <= len(path_specs) + 4:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    source_count = random.choice(SOURCE_COUNT_OPTIONS)
    specs = _specs_for_sample(height, source_count)

    placed = _place_specs(height, specs)
    if placed is None:
        return None
    width, specs = placed

    all_cells: set[tuple[int, int]] = set()
    all_blockers: set[tuple[int, int]] = set()
    for spec in specs:
        try:
            cells, blockers = _trace_planned_path(
                height, width, int(spec["start_col"]), list(spec["sequence"]), bool(spec["stop_early"])
            )
        except ValueError:
            return None

        if any(cell in all_cells for cell in cells):
            return None
        if any(cell in all_blockers for cell in cells):
            return None
        if any(blocker in all_cells for blocker in blockers):
            return None

        spec["cells"] = cells
        spec["blockers"] = blockers
        all_cells.update(cells)
        all_blockers.update(blockers)

    grid = np.full((height, width), BG, dtype=int)
    for r, c in sorted(all_blockers):
        grid[r, c] = BLOCK
    for spec in specs:
        grid[0, int(spec["start_col"])] = int(spec["color"])

    if not _valid_sample(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rain-drop deflection sample")
