import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS = None
if _ARC_UTILS_PATH.exists():
    try:
        _spec = importlib.util.spec_from_file_location("_arc_utils_local", _ARC_UTILS_PATH)
        if _spec is not None and _spec.loader is not None:
            _ARC_UTILS = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_ARC_UTILS)
    except Exception:
        _ARC_UTILS = None


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    _bbox = _ARC_UTILS.bbox
    _in_bounds = _ARC_UTILS.in_bounds
    _neighbors4 = _ARC_UTILS.neighbors4
else:
    class Color:
        BLACK = 0
        RED = 2
        GREY = 5
        TEAL = 8

    def _bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def _in_bounds(shape, r: int, c: int) -> bool:
        h, w = shape
        return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)

    def _neighbors4(r: int, c: int):
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1


BG = Color.BLACK
MARKER = Color.RED
BLOCKER = Color.GREY
LATTICE = Color.TEAL

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (14, 18)
BOX_H_RANGE = (8, 10)
BOX_W_RANGE = (8, 10)
OBSTACLE_COUNT_RANGE = (3, 5)
AMPLITUDES = (0, 1, 2, 3)
GEN_ATTEMPTS = 400


def _lattice_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox((grid == LATTICE) | (grid == BLOCKER))
    assert box is not None, "expected a teal rectangle with grey blockers"
    return box


def _row_marker_counts(grid: np.ndarray, c0: int, r0: int, r1: int) -> list[int]:
    return [int(np.count_nonzero(grid[r, :c0] == MARKER)) for r in range(r0, r1 + 1)]


def _anchor_coords(grid: np.ndarray) -> set[tuple[int, int]]:
    anchors: set[tuple[int, int]] = set()
    for r, c in np.argwhere(grid == LATTICE):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLOCKER:
                anchors.add((rr, cc))
                break
    return anchors


def _shift_from_distance(
    distance_from_bottom: int,
    amplitude: int,
    *,
    phase: str = "bottom",
) -> int:
    dist = int(distance_from_bottom)
    amp = int(amplitude)
    if dist == 0 or amp == 0:
        return 0
    if phase == "bottom":
        return amp if dist % 2 == 1 else -amp
    if phase == "top":
        return amp if dist % 2 == 0 else -amp
    raise ValueError(f"unknown phase {phase}")


def _transform(
    grid: np.ndarray,
    *,
    use_anchors: bool = True,
    amplitude_mode: str = "count",
    phase: str = "bottom",
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, _c1 = _lattice_bbox(g)
    counts = _row_marker_counts(g, c0, r0, r1)
    anchors = _anchor_coords(g) if use_anchors else set()

    out = np.full((h, w), BG, dtype=int)
    for r, c in np.argwhere(g == LATTICE):
        rr = int(r)
        cc = int(c)
        if not (r0 <= rr <= r1):
            continue
        distance = r1 - rr
        amplitude = counts[rr - r0]
        if amplitude_mode == "unit_nonzero":
            amplitude = 1 if amplitude > 0 else 0
        elif amplitude_mode != "count":
            raise ValueError(f"unknown amplitude_mode {amplitude_mode}")

        shift = _shift_from_distance(distance, amplitude, phase=phase)
        target_c = cc if (rr, cc) in anchors else cc + shift
        if 0 <= target_c < w:
            out[rr, target_c] = LATTICE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bend the teal rectangle into a row-wise wave whose amplitude is encoded by left-edge red marker counts.

    Input structure:
    - Black is the background.
    - One central teal rectangle is the foreground lattice, with a few grey blocker pixels replacing teal cells inside it.
    - For each lattice row, 0 to 3 red marker pixels may appear immediately to the left of the rectangle, starting at column 0.

    Rule:
    - Find the bounding box of the teal rectangle together with the grey blocker cells.
    - Keep the bottom row of that box fixed. For every higher row, count the red markers to the left of the box; that count is the horizontal shift amplitude for teal pixels in the row.
    - Measure row parity from the bottom of the box: odd distances shift right, even distances shift left.
    - Exception: any teal pixel that is 4-neighbor adjacent to a grey blocker in the input is anchored and does not move.
    - Discard shifted teal pixels that leave the grid, and erase all red markers and grey blockers in the output.

    Key insight:
    - The final wave depends on three independent cues at once: per-row marker counts set the amplitude, bottom-based parity sets the direction, and blocker-adjacent teal cells locally resist the shift.
    """
    return _transform(grid, use_anchors=True, amplitude_mode="count", phase="bottom")


def _sample_counts(height: int) -> list[int] | None:
    for _ in range(120):
        counts = [random.choice(AMPLITUDES) for _ in range(height)]
        non_bottom = counts[:-1]
        if len(set(counts)) < 2:
            continue
        if len({v for v in counts if v > 0}) < 2:
            continue
        if not any(v == 0 for v in non_bottom):
            continue
        if not any(non_bottom[idx] > 0 and ((height - 1 - idx) % 2 == 1) for idx in range(len(non_bottom))):
            continue
        if not any(non_bottom[idx] > 0 and ((height - 1 - idx) % 2 == 0) for idx in range(len(non_bottom))):
            continue
        return counts
    return None


def _sample_obstacles(
    r0: int,
    c0: int,
    box_h: int,
    box_w: int,
    counts: list[int],
) -> list[tuple[int, int]] | None:
    r1 = r0 + box_h - 1
    candidates = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c0 + box_w - 1)
    ]
    for _ in range(160):
        k = random.randint(*OBSTACLE_COUNT_RANGE)
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for rc in candidates:
            if all(abs(rc[0] - rr) + abs(rc[1] - cc) >= 2 for rr, cc in chosen):
                chosen.append(rc)
            if len(chosen) == k:
                break
        if len(chosen) < k:
            continue

        anchored_rows: set[int] = set()
        effective_rows: set[int] = set()
        for br, bc in chosen:
            for nr, nc in _neighbors4(br, bc):
                if r0 <= nr <= r1 and c0 <= nc < c0 + box_w:
                    anchored_rows.add(int(nr))
                    if counts[nr - r0] > 0 and nr != r1:
                        effective_rows.add(int(nr))

        if len(anchored_rows) < 3:
            continue
        if len(effective_rows) < 2:
            continue
        return chosen
    return None


def _build_grid(
    height: int,
    width: int,
    r0: int,
    c0: int,
    box_h: int,
    box_w: int,
    counts: list[int],
    obstacles: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[r0 : r0 + box_h, c0 : c0 + box_w] = LATTICE
    for offset, count in enumerate(counts):
        row = r0 + offset
        if count > 0:
            grid[row, :count] = MARKER
    for r, c in obstacles:
        grid[r, c] = BLOCKER
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _transform(grid, use_anchors=False, amplitude_mode="count", phase="bottom")):
        return False
    if np.array_equal(out, _transform(grid, use_anchors=True, amplitude_mode="unit_nonzero", phase="bottom")):
        return False
    if np.array_equal(out, _transform(grid, use_anchors=True, amplitude_mode="count", phase="top")):
        return False

    r0, c0, r1, _c1 = _lattice_bbox(grid)
    counts = _row_marker_counts(grid, c0, r0, r1)
    anchors = _anchor_coords(grid)

    moved_pixels = 0
    anchored_active_rows: set[int] = set()
    for r, c in np.argwhere(grid == LATTICE):
        rr = int(r)
        cc = int(c)
        shift = _shift_from_distance(r1 - rr, counts[rr - r0], phase="bottom")
        if shift != 0:
            moved_pixels += 1
        if (rr, cc) in anchors and shift != 0:
            anchored_active_rows.add(rr)

    if moved_pixels < 12:
        return False
    if len(anchored_active_rows) < 2:
        return False
    if np.count_nonzero(out == LATTICE) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    max_box_h = min(BOX_H_RANGE[1], height - 3)
    max_box_w = min(BOX_W_RANGE[1], width - 7)
    if max_box_h < BOX_H_RANGE[0] or max_box_w < BOX_W_RANGE[0]:
        return None

    box_h = random.randint(BOX_H_RANGE[0], max_box_h)
    box_w = random.randint(BOX_W_RANGE[0], max_box_w)
    r0 = random.randint(1, height - box_h - 1)
    c0 = random.randint(3, width - box_w - 4)

    counts = _sample_counts(box_h)
    if counts is None:
        return None

    obstacles = _sample_obstacles(r0, c0, box_h, box_w, counts)
    if obstacles is None:
        return None

    grid = _build_grid(height, width, r0, c0, box_h, box_w, counts, obstacles)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
