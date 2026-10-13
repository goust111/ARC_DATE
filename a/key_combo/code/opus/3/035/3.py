from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, in_bounds


BLUE = Color.BLUE
RED = Color.RED
BLACK = Color.BLACK

Frame = tuple[int, int, int, int]
Hint = dict[str, object]


def _orth_degree(mask: np.ndarray, row: int, col: int) -> int:
    deg = 0
    height, width = mask.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]):
            deg += 1
    return deg


def _has_diagonal_neighbor(mask: np.ndarray, row: int, col: int) -> bool:
    height, width = mask.shape
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]):
            return True
    return False


def _diagonal_hint_mask(mask: np.ndarray) -> np.ndarray:
    diag = np.zeros_like(mask, dtype=bool)
    for row, col in np.argwhere(mask):
        row = int(row)
        col = int(col)
        if _orth_degree(mask, row, col) == 0 and _has_diagonal_neighbor(mask, row, col):
            diag[row, col] = True
    return diag


def _segment_runs(values: np.ndarray) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(values[0])
    prev = int(values[0])
    for raw in values[1:]:
        value = int(raw)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _collect_hints(diag_mask: np.ndarray) -> list[Hint]:
    hints: list[Hint] = []
    for coords in components_mask(diag_mask, connectivity=8):
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        span = max(int(r1 - r0), int(c1 - c0))
        hints.append(
            {
                "coords": {(int(r), int(c)) for r, c in coords},
                "span": max(1, span),
            }
        )
    return hints


def _hint_supports_horizontal(hints: list[Hint], row: int, gap0: int, gap1: int) -> bool:
    length = int(gap1 - gap0 + 1)
    for hint in hints:
        if int(hint["span"]) < length:
            continue
        for rr, cc in hint["coords"]:
            if abs(int(rr) - row) <= 1 and gap0 - 1 <= int(cc) <= gap1 + 1:
                return True
    return False


def _hint_supports_vertical(hints: list[Hint], col: int, gap0: int, gap1: int) -> bool:
    length = int(gap1 - gap0 + 1)
    for hint in hints:
        if int(hint["span"]) < length:
            continue
        for rr, cc in hint["coords"]:
            if abs(int(cc) - col) <= 1 and gap0 - 1 <= int(rr) <= gap1 + 1:
                return True
    return False


def _repair_frames(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    blue_mask = out == BLUE
    diag_mask = _diagonal_hint_mask(blue_mask)
    out[diag_mask] = BLACK

    hints = _collect_hints(diag_mask)
    base_mask = out == BLUE
    height, width = out.shape

    for row in range(height):
        runs = _segment_runs(np.where(base_mask[row])[0])
        for (_a0, b0), (a1, _b1) in zip(runs, runs[1:]):
            gap0 = int(b0 + 1)
            gap1 = int(a1 - 1)
            length = int(gap1 - gap0 + 1)
            if 1 <= length <= 3 and _hint_supports_horizontal(hints, row, gap0, gap1):
                out[row, gap0 : gap1 + 1] = BLUE

    for col in range(width):
        runs = _segment_runs(np.where(base_mask[:, col])[0])
        for (_a0, b0), (a1, _b1) in zip(runs, runs[1:]):
            gap0 = int(b0 + 1)
            gap1 = int(a1 - 1)
            length = int(gap1 - gap0 + 1)
            if 1 <= length <= 3 and _hint_supports_vertical(hints, col, gap0, gap1):
                out[gap0 : gap1 + 1, col] = BLUE

    return out


def _is_rectangular_frame(shape: tuple[int, int], coords: np.ndarray) -> tuple[bool, Frame | None]:
    if coords.size == 0:
        return False, None
    r0, c0, r1, c1 = bbox_coords(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 4 or width < 4:
        return False, None
    if int(coords.shape[0]) != 2 * height + 2 * width - 4:
        return False, None

    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(local, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return False, None
    return True, (int(r0), int(c0), int(r1), int(c1))


def _fill_checkerboard(out: np.ndarray, frame: Frame) -> None:
    r0, c0, r1, c1 = frame
    anchor_r = r0 + 1
    anchor_c = c0 + 1
    for row in range(anchor_r, r1):
        for col in range(anchor_c, c1):
            if (row - anchor_r + col - anchor_c) % 2 == 0:
                out[row, col] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal hint chains, repair the broken blue rectangular frames, and fill each frame interior with an anchored checkerboard.

    Input structure:
    - A black background contains blue one-pixel axis-aligned line segments that come from 2-3 separate rectangular frames.
    - Each frame has 1-2 short gaps of length 1-3 on its border, and each gap is marked by a nearby blue diagonal-only chain.
    - The diagonal chains are 8-connected but have no orthogonal blue neighbors, while the real frame borders are horizontal or vertical.

    Rule:
    - Delete every blue pixel that has no orthogonal blue neighbor but does have a diagonal blue neighbor.
    - On the cleaned grid, fill any horizontal or vertical blue gap of length 1-3 when a nearby deleted diagonal chain has span at least that long.
    - Treat every repaired blue 4-connected component that is exactly a one-pixel rectangular border as a frame.
    - For each such frame, fill its interior with red on checkerboard-even cells anchored at the interior's top-left cell, leaving odd cells black.

    Key insight:
    - The diagonal noise is actually a local length marker for each missing border segment; once those hints are erased and interpreted, the completed rectangles expose the regions that should receive checkerboard fills.
    """
    repaired = _repair_frames(grid)
    out = repaired.copy()

    for coords in components_mask(repaired == BLUE, connectivity=4):
        ok, frame = _is_rectangular_frame(repaired.shape, coords)
        if ok and frame is not None:
            _fill_checkerboard(out, frame)

    return out


GEN_RETRIES = 400
GRID_MIN = 20
GRID_MAX = 28
FRAME_COUNT_CHOICES = (2, 2, 3)
FRAME_HEIGHT_RANGE = (5, 8)
FRAME_WIDTH_RANGE = (5, 8)
OUTER_MARGIN = 4
FRAME_CLEARANCE = 2


def _boxes_too_close(a: Frame, b: Frame, pad: int) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _draw_frame(grid: np.ndarray, frame: Frame) -> None:
    r0, c0, r1, c1 = frame
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE


def _frame_border_cells(frame: Frame) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    cells: set[tuple[int, int]] = set()
    for row in range(r0, r1 + 1):
        cells.add((row, c0))
        cells.add((row, c1))
    for col in range(c0, c1 + 1):
        cells.add((r0, col))
        cells.add((r1, col))
    return cells


def _render_expected(shape: tuple[int, int], frames: list[Frame]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame in frames:
        _draw_frame(out, frame)
        _fill_checkerboard(out, frame)
    return out


def _choose_frames(rng: random.Random, height: int, width: int, count: int) -> list[Frame] | None:
    frames: list[Frame] = []
    for _ in range(count):
        placed = False
        for _ in range(200):
            fh = rng.randint(*FRAME_HEIGHT_RANGE)
            fw = rng.randint(*FRAME_WIDTH_RANGE)
            max_r0 = height - OUTER_MARGIN - fh
            max_c0 = width - OUTER_MARGIN - fw
            if max_r0 < OUTER_MARGIN or max_c0 < OUTER_MARGIN:
                return None
            r0 = rng.randint(OUTER_MARGIN, max_r0)
            c0 = rng.randint(OUTER_MARGIN, max_c0)
            cand = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_too_close(cand, other, FRAME_CLEARANCE) for other in frames):
                continue
            frames.append(cand)
            placed = True
            break
        if not placed:
            return None
    return frames


def _gap_cells(frame: Frame, side: str, start: int, length: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    if side == "top":
        return [(r0, start + delta) for delta in range(length)]
    if side == "bottom":
        return [(r1, start + delta) for delta in range(length)]
    if side == "left":
        return [(start + delta, c0) for delta in range(length)]
    if side == "right":
        return [(start + delta, c1) for delta in range(length)]
    raise ValueError(f"unknown side: {side}")


def _chain_cells(frame: Frame, side: str, start: int, length: int, orient: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    coords: list[tuple[int, int]] = []
    if side == "top":
        for step in range(length + 1):
            col = start + step if orient == 0 else start + length - 1 - step
            coords.append((r0 - 1 - step, col))
        return coords
    if side == "bottom":
        for step in range(length + 1):
            col = start + step if orient == 0 else start + length - 1 - step
            coords.append((r1 + 1 + step, col))
        return coords
    if side == "left":
        for step in range(length + 1):
            row = start + step if orient == 0 else start + length - 1 - step
            coords.append((row, c0 - 1 - step))
        return coords
    if side == "right":
        for step in range(length + 1):
            row = start + step if orient == 0 else start + length - 1 - step
            coords.append((row, c1 + 1 + step))
        return coords
    raise ValueError(f"unknown side: {side}")


def _gap_options(frame: Frame, side: str) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame
    options: list[tuple[int, int, int]] = []
    if side in {"top", "bottom"}:
        side_len = c1 - c0 + 1
        max_gap = min(3, side_len - 2)
        for length in range(1, max_gap + 1):
            for start in range(c0 + 1, c1 - length + 1):
                for orient in (0, 1):
                    options.append((start, length, orient))
        return options
    side_len = r1 - r0 + 1
    max_gap = min(3, side_len - 2)
    for length in range(1, max_gap + 1):
        for start in range(r0 + 1, r1 - length + 1):
            for orient in (0, 1):
                options.append((start, length, orient))
    return options


def _valid_chain(shape: tuple[int, int], border_cells: set[tuple[int, int]], chain: list[tuple[int, int]]) -> bool:
    chain_set = set(chain)
    if len(chain_set) != len(chain):
        return False
    for row, col in chain:
        if not in_bounds(shape, row, col):
            return False
        if (row, col) in border_cells:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nbr = (row + dr, col + dc)
            if nbr in border_cells or nbr in chain_set:
                return False
    mask = np.zeros(shape, dtype=bool)
    for row, col in chain:
        mask[row, col] = True
    diag = _diagonal_hint_mask(mask)
    return bool(np.array_equal(mask, diag))


def _choose_breaks(
    rng: random.Random,
    shape: tuple[int, int],
    frames: list[Frame],
    frame: Frame,
    all_border_cells: set[tuple[int, int]],
    used_chain_cells: set[tuple[int, int]],
) -> list[tuple[str, int, int, int, list[tuple[int, int]]]] | None:
    break_count = 1 if rng.random() < 0.45 else 2
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    picked_sides = sides[:break_count]
    frame_border = _frame_border_cells(frame)
    other_borders = all_border_cells - frame_border
    breaks: list[tuple[str, int, int, int, list[tuple[int, int]]]] = []
    local_gap_cells: set[tuple[int, int]] = set()
    local_chain_cells: set[tuple[int, int]] = set()

    for side in picked_sides:
        options = _gap_options(frame, side)
        rng.shuffle(options)
        chosen = None
        for start, length, orient in options:
            gap = _gap_cells(frame, side, start, length)
            chain = _chain_cells(frame, side, start, length, orient)
            if any(cell in local_gap_cells for cell in gap):
                continue
            if any(cell in used_chain_cells or cell in local_chain_cells for cell in chain):
                continue
            if any(cell in other_borders for cell in chain):
                continue
            border_without_gap = frame_border - set(gap)
            if not _valid_chain(shape, border_without_gap, chain):
                continue
            chosen = (side, start, length, orient, chain)
            local_gap_cells.update(gap)
            local_chain_cells.update(chain)
            break
        if chosen is None:
            return None
        breaks.append(chosen)
    return breaks


def _build_input_from_frames(
    shape: tuple[int, int],
    frames: list[Frame],
    break_plans: list[list[tuple[str, int, int, int, list[tuple[int, int]]]]],
) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    for frame in frames:
        _draw_frame(inp, frame)
    for frame, plans in zip(frames, break_plans):
        for side, start, length, _orient, chain in plans:
            for row, col in _gap_cells(frame, side, start, length):
                inp[row, col] = BLACK
            for row, col in chain:
                inp[row, col] = BLUE
    return inp


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    frame_count = rng.choice(FRAME_COUNT_CHOICES)

    frames = _choose_frames(rng, height, width, frame_count)
    if frames is None:
        return None

    border_cells: set[tuple[int, int]] = set()
    for frame in frames:
        border_cells |= _frame_border_cells(frame)

    used_chain_cells: set[tuple[int, int]] = set()
    break_plans: list[list[tuple[str, int, int, int, list[tuple[int, int]]]]] = []
    for frame in frames:
        plans = _choose_breaks(rng, (height, width), frames, frame, border_cells, used_chain_cells)
        if plans is None:
            return None
        for _side, _start, _length, _orient, chain in plans:
            used_chain_cells.update(chain)
        break_plans.append(plans)

    inp = _build_input_from_frames((height, width), frames, break_plans)
    expected = _render_expected((height, width), frames)

    if np.array_equal(inp, expected):
        return None
    if len(np.unique(inp)) != 2:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 3")
