from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import bbox_coords, bbox_h, bbox_w, find_components, get_bg


BG = 0
FRAME = 1
STAMP = 2
ANCHOR = 3

GRID_ROWS = (15, 24)
GRID_COLS = (15, 24)
FRAME_SIZE_CHOICES = (4, 5, 5)
MARKER_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 400


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _bbox_contains(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = map(int, box)
    return r0 <= int(r) <= r1 and c0 <= int(c) <= c1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _point_box_chebyshev(point: tuple[int, int], box: tuple[int, int, int, int]) -> int:
    r, c = map(int, point)
    r0, c0, r1, c1 = map(int, box)
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return max(dr, dc)


def _point_chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _find_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    frames: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME],
        return_format="dict",
    )
    for comp in comps:
        box = tuple(map(int, comp["bbox"]))
        height = int(bbox_h(box))
        width = int(bbox_w(box))
        if height < 4 or width < 4:
            continue
        r0, c0, r1, c1 = box
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        interior = sub[1:-1, 1:-1]
        if interior.size == 0:
            continue
        if int(comp["area"]) != int(border.sum()):
            continue
        if not np.all(sub[border] == FRAME):
            continue
        if not np.all(np.isin(interior, [BG, STAMP])):
            continue
        reds = np.argwhere(interior == STAMP)
        if not (2 <= len(reds) <= 4):
            continue
        frames.append((box, [(int(r), int(c)) for r, c in reds]))

    assert len(frames) == 1, "expected exactly one valid blue frame"
    return frames[0]


def _find_markers(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME, ANCHOR],
        return_format="dict",
    )
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        coord = tuple(map(int, comp["coords"][0]))
        if color == FRAME and tuple(map(int, comp["bbox"])) == frame_box and area > 1:
            continue
        assert area == 1, "all non-frame blue/green objects must be singleton markers"
        assert not _bbox_contains(frame_box, coord[0], coord[1]), "markers must lie outside the frame"
        markers.append((coord[0], coord[1], color))

    assert 4 <= len(markers) <= 6, "expected 4-6 singleton markers"
    assert {FRAME, ANCHOR}.issubset({color for _, _, color in markers}), "both marker colors must appear"
    return sorted(markers)


def _render_expected(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    frame_box, offsets = _find_frame(grid)
    markers = _find_markers(grid, frame_box)

    out = np.array(grid, dtype=int, copy=True)
    for r, c, _color in markers:
        for dr, dc in offsets:
            rr = int(r + dr)
            cc = int(c + dc)
            if out[rr, cc] == BG:
                out[rr, cc] = STAMP

    for r, c, color in markers:
        out[r, c] = BG if color == FRAME else FRAME

    return out, frame_box, offsets, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the red motif inside the blue frame and stamp it at singleton markers, then erase or recolor each marker by its color code.

    Input structure:
    - The grid has a black background, one complete blue rectangular frame, and a
      small red motif inside that frame.
    - Outside the frame there are 4-6 isolated singleton markers, each either blue
      or green, with enough spacing that their stamp regions do not overlap.
    - The red motif is interpreted on the frame interior's coordinate system, not by
      cropping the motif to its own tight bounding box.

    Rule:
    - Find the unique blue frame and record every red pixel's offset from the frame
      interior's top-left cell.
    - For each outside singleton marker at (r, c), stamp red pixels at (r+dr, c+dc)
      for all recorded offsets, but only onto background cells.
    - After stamping, erase blue markers to black and recolor green markers to blue.
      The original frame and its interior motif stay unchanged.

    Key insight:
    - The frame supplies the template, the singleton positions supply the stamp
      placements, and the singleton color supplies a post-stamp instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BG, "expected black to be the background"
    out, _frame_box, _offsets, _markers = _render_expected(g)
    return out


def _random_pattern(inner: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(inner) for c in range(inner)]
    want_margin = inner == 3 and random.random() < 0.7
    for _ in range(100):
        count = random.randint(2, min(4, len(cells)))
        chosen = random.sample(cells, count)
        coords = np.array(chosen, dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 == r0 or c1 == c0:
            continue
        full_bbox = r0 == 0 and c0 == 0 and r1 == inner - 1 and c1 == inner - 1
        if want_margin and full_bbox:
            continue
        return sorted((int(r), int(c)) for r, c in chosen)
    return None


def _candidate_markers(
    shape: tuple[int, int],
    stamp_size: int,
    frame_box: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    height, width = map(int, shape)
    candidates = [(r, c) for r in range(height - stamp_size + 1) for c in range(width - stamp_size + 1)]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    stamp_boxes: list[tuple[int, int, int, int]] = []
    for r, c in candidates:
        point = (int(r), int(c))
        box = (int(r), int(c), int(r + stamp_size - 1), int(c + stamp_size - 1))
        if _boxes_overlap(box, frame_box):
            continue
        if _point_box_chebyshev(point, frame_box) < stamp_size + 2:
            continue
        blocked = False
        for other_point, other_box in zip(chosen, stamp_boxes):
            if _point_chebyshev(point, other_point) < stamp_size + 2:
                blocked = True
                break
            if _boxes_overlap(box, other_box):
                blocked = True
                break
            if _bbox_contains(other_box, point[0], point[1]) or _bbox_contains(box, other_point[0], other_point[1]):
                blocked = True
                break
        if blocked:
            continue
        chosen.append(point)
        stamp_boxes.append(box)
        if len(chosen) == count:
            return chosen
    return None


def _build_input() -> np.ndarray | None:
    height = random.randint(*GRID_ROWS)
    width = random.randint(*GRID_COLS)
    grid = np.full((height, width), BG, dtype=int)

    frame_size = random.choice(FRAME_SIZE_CHOICES)
    inner = frame_size - 2
    pattern = _random_pattern(inner)
    if pattern is None:
        return None

    frame_r0 = random.randint(1, height - frame_size - 1)
    frame_c0 = random.randint(1, width - frame_size - 1)
    frame_box = (frame_r0, frame_c0, frame_r0 + frame_size - 1, frame_c0 + frame_size - 1)

    grid[frame_r0 : frame_r0 + frame_size, frame_c0 : frame_c0 + frame_size] = FRAME
    grid[frame_r0 + 1 : frame_r0 + frame_size - 1, frame_c0 + 1 : frame_c0 + frame_size - 1] = BG
    for dr, dc in pattern:
        grid[frame_r0 + 1 + dr, frame_c0 + 1 + dc] = STAMP

    marker_count = random.randint(*MARKER_COUNT_RANGE)
    marker_positions = _candidate_markers(grid.shape, inner, frame_box, marker_count)
    if marker_positions is None:
        return None

    marker_colors = [FRAME, ANCHOR]
    marker_colors.extend(random.choice([FRAME, ANCHOR]) for _ in range(marker_count - 2))
    random.shuffle(marker_colors)
    for (r, c), color in zip(marker_positions, marker_colors):
        grid[r, c] = color

    return grid


def _validate_case(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))).issubset({BG, FRAME, STAMP, ANCHOR})

    expected, frame_box, offsets, markers = _render_expected(grid)
    out = main(grid)
    assert np.array_equal(out, expected), "solver output must match the semantic reconstruction"

    r0, c0, r1, c1 = frame_box
    assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], grid[r0 : r1 + 1, c0 : c1 + 1]), "frame area must stay unchanged"

    frame_size = int(r1 - r0 + 1)
    stamp_size = int(frame_size - 2)
    stamp_boxes = []
    blue_markers = 0
    green_markers = 0
    for r, c, color in markers:
        box = (r, c, r + stamp_size - 1, c + stamp_size - 1)
        stamp_boxes.append(box)
        assert not _boxes_overlap(box, frame_box)
        assert _point_box_chebyshev((r, c), frame_box) >= frame_size
        if color == FRAME:
            blue_markers += 1
            assert int(out[r, c]) == BG
        else:
            green_markers += 1
            assert int(out[r, c]) == FRAME

    for i in range(len(markers)):
        for j in range(i + 1, len(markers)):
            assert _point_chebyshev(markers[i][:2], markers[j][:2]) >= frame_size
            assert not _boxes_overlap(stamp_boxes[i], stamp_boxes[j])

    assert blue_markers >= 1 and green_markers >= 1
    assert any(int(grid[r, c]) != int(out[r, c]) for r, c, _ in markers), "marker post-processing must matter"
    assert np.count_nonzero((grid != STAMP) & (out == STAMP)) > 0, "stamping must add new red pixels"
    assert len(offsets) >= 2


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        try:
            _validate_case(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
