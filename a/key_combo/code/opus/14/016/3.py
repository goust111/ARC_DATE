from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
TOP = 1
LEFT = 2
BOTTOM = 3
RIGHT = 4

SIDE_ORDER = ("top", "bottom", "left", "right")
SIDE_COLOR = {"top": TOP, "bottom": BOTTOM, "left": LEFT, "right": RIGHT}

_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SELF_TESTS_DONE = False


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _marker_distance(side: str, bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    r, c = cell
    if side == "top":
        return int(r - r0)
    if side == "bottom":
        return int(r1 - r)
    if side == "left":
        return int(c - c0)
    if side == "right":
        return int(c1 - c)
    raise ValueError(f"unknown side: {side}")


def _edge_distances(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    r, c = cell
    return {
        "top": int(r - r0),
        "bottom": int(r1 - r),
        "left": int(c - c0),
        "right": int(c1 - c),
    }


def _draw_frame(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros(shape, dtype=int)
    out[r0, c0 + 1 : c1] = TOP
    out[r1, c0 + 1 : c1] = BOTTOM
    out[r0 + 1 : r1, c0] = LEFT
    out[r0 + 1 : r1, c1] = RIGHT
    return out


def _candidate_frame_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    blue_rows = sorted({int(r) for r, _c in np.argwhere(g == TOP)})
    green_rows = sorted({int(r) for r, _c in np.argwhere(g == BOTTOM)})
    red_cols = sorted({int(c) for _r, c in np.argwhere(g == LEFT)})
    yellow_cols = sorted({int(c) for _r, c in np.argwhere(g == RIGHT)})

    candidates: list[tuple[int, int, int, int]] = []
    for r0 in blue_rows:
        for r1 in green_rows:
            if r1 - r0 < 4:
                continue
            for c0 in red_cols:
                for c1 in yellow_cols:
                    if c1 - c0 < 4:
                        continue
                    if not (g[r0, c0] == g[r0, c1] == g[r1, c0] == g[r1, c1] == BG):
                        continue
                    if not np.all(g[r0, c0 + 1 : c1] == TOP):
                        continue
                    if not np.all(g[r1, c0 + 1 : c1] == BOTTOM):
                        continue
                    if not np.all(g[r0 + 1 : r1, c0] == LEFT):
                        continue
                    if not np.all(g[r0 + 1 : r1, c1] == RIGHT):
                        continue
                    candidates.append((int(r0), int(c0), int(r1), int(c1)))
    return candidates


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = _candidate_frame_boxes(grid)
    assert len(candidates) == 1, f"expected exactly one frame, found {len(candidates)}"
    return candidates[0]


def _parse_markers(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[list[tuple[str, tuple[int, int], int]], dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = bbox
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    markers: list[tuple[str, tuple[int, int], int]] = []
    stretches = {side: 0 for side in SIDE_ORDER}

    for rr, cc in np.argwhere(interior != BG):
        r = int(rr) + r0 + 1
        c = int(cc) + c0 + 1
        color = int(g[r, c])
        side = {
            TOP: "top",
            BOTTOM: "bottom",
            LEFT: "left",
            RIGHT: "right",
        }.get(color)
        assert side is not None, f"unexpected interior color: {color}"
        d = _marker_distance(side, bbox, (r, c))
        assert d >= 1
        markers.append((side, (r, c), d))
        stretches[side] = max(stretches[side], d)
    assert markers, "expected at least one interior marker"
    return markers, stretches


def main(grid: np.ndarray) -> np.ndarray:
    """Move interior color-coded markers to their matching border conceptually, then stretch each chosen side by the marker's recorded distance.

    Input structure:
    - A black background contains one hollow rectangular frame whose corners are black.
    - The top, bottom, left, and right frame sides use fixed colors blue, green, red, and yellow respectively.
    - The frame interior is otherwise black except for 1-4 single-pixel markers whose colors match one of the four frame sides.

    Rule:
    - Find the unique hollow frame and, for each interior marker, read its color as the target side and its offset from that side as a distance.
    - Erase all interior markers; if multiple markers target the same side, keep only the maximum recorded distance for that side.
    - Stretch each targeted side outward by its recorded distance while untargeted sides stay fixed, then redraw the hollow frame with the same side colors, black corners, and black interior.

    Key insight:
    - Each marker simultaneously encodes direction by color and stretch magnitude by position, so both attributes must be extracted before the marker disappears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bbox = _find_frame(g)
    _markers, stretches = _parse_markers(g, bbox)
    r0, c0, r1, c1 = bbox

    nr0 = r0 - stretches["top"]
    nr1 = r1 + stretches["bottom"]
    nc0 = c0 - stretches["left"]
    nc1 = c1 + stretches["right"]
    assert 0 <= nr0 < nr1 < g.shape[0]
    assert 0 <= nc0 < nc1 < g.shape[1]

    return _draw_frame(g.shape, (nr0, nc0, nr1, nc1))


def _choose_active_sides() -> list[str]:
    count = random.choice([1, 2, 2, 3, 3, 4])
    return random.sample(list(SIDE_ORDER), count)


def _max_distance_for_side(
    side: str,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> int:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    H, W = shape
    if side == "top":
        return min(r0, h - 2, 6)
    if side == "bottom":
        return min(H - 1 - r1, h - 2, 6)
    if side == "left":
        return min(c0, w - 2, 6)
    if side == "right":
        return min(W - 1 - c1, w - 2, 6)
    raise ValueError(f"unknown side: {side}")


def _far_enough(cell: tuple[int, int], occupied: Iterable[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in occupied)


def _candidate_cells_for_side(
    side: str,
    bbox: tuple[int, int, int, int],
    d: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0 + d, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1 - d, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0 + d) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1 - d) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown side: {side}")


def _is_confusing_cell(
    side: str,
    bbox: tuple[int, int, int, int],
    cell: tuple[int, int],
) -> bool:
    dists = _edge_distances(bbox, cell)
    target = dists[side]
    others = [dist for name, dist in dists.items() if name != side]
    return min(others) < target


def _choose_marker_cell(
    side: str,
    bbox: tuple[int, int, int, int],
    d: int,
    occupied: set[tuple[int, int]],
    prefer_confusing: bool,
) -> tuple[int, int] | None:
    candidates = [
        cell
        for cell in _candidate_cells_for_side(side, bbox, d)
        if cell not in occupied and _far_enough(cell, occupied)
    ]
    if prefer_confusing:
        confusing = [cell for cell in candidates if _is_confusing_cell(side, bbox, cell)]
        if confusing:
            candidates = confusing
    if not candidates:
        return None
    return random.choice(candidates)


def _build_input(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    placements: list[tuple[str, tuple[int, int]]],
) -> np.ndarray:
    grid = _draw_frame(shape, bbox)
    for side, cell in placements:
        grid[cell] = SIDE_COLOR[side]
    return grid


def _expected_bbox(
    bbox: tuple[int, int, int, int],
    distances: dict[str, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        r0 - distances["top"],
        c0 - distances["left"],
        r1 + distances["bottom"],
        c1 + distances["right"],
    )


def _validate_generated_case(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    placements: list[tuple[str, tuple[int, int]]],
    distances: dict[str, int],
) -> bool:
    try:
        solved = main(grid)
    except Exception:
        return False

    if len(_candidate_frame_boxes(grid)) != 1:
        return False

    for side, cell in placements:
        if _marker_distance(side, bbox, cell) != distances[side]:
            return False

    expected = _draw_frame(grid.shape, _expected_bbox(bbox, distances))
    if not np.array_equal(solved, expected):
        return False

    solved_bbox = _find_frame(solved)
    if solved_bbox != _expected_bbox(bbox, distances):
        return False

    if len(placements) >= 2 and len({distances[side] for side, _cell in placements}) == 1:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(14, 22)
    W = random.randint(14, 22)
    h = random.randint(7, min(11, H - 4))
    w = random.randint(7, min(11, W - 4))

    r0 = random.randint(2, H - h - 2)
    c0 = random.randint(2, W - w - 2)
    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)

    active_sides = _choose_active_sides()
    max_distances = {side: _max_distance_for_side(side, bbox, (H, W)) for side in SIDE_ORDER}
    if any(max_distances[side] < 2 for side in active_sides):
        return None

    distances = {side: 0 for side in SIDE_ORDER}
    active_d_values = []
    for side in active_sides:
        distances[side] = random.randint(2, max_distances[side])
        active_d_values.append(distances[side])
    if max(active_d_values) == 2 and len(active_d_values) > 1:
        return None

    placements: list[tuple[str, tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    confusing_side = random.choice(active_sides)
    order = active_sides[:]
    random.shuffle(order)
    order.sort(key=lambda side: side != confusing_side)

    for side in order:
        cell = _choose_marker_cell(
            side,
            bbox,
            distances[side],
            occupied,
            prefer_confusing=(side == confusing_side),
        )
        if cell is None:
            return None
        placements.append((side, cell))
        occupied.add(cell)

    grid = _build_input((H, W), bbox, placements)
    if not _validate_generated_case(grid, bbox, placements, distances):
        return None

    output_sig = _signature(main(grid))
    if output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def _run_internal_self_tests_once() -> None:
    global _SELF_TESTS_DONE
    if _SELF_TESTS_DONE:
        return

    def render_case(
        shape: tuple[int, int],
        bbox: tuple[int, int, int, int],
        markers: list[tuple[str, tuple[int, int]]],
    ) -> np.ndarray:
        return _build_input(shape, bbox, markers)

    bbox = (5, 4, 10, 10)
    case1 = render_case(
        (16, 16),
        bbox,
        [
            ("top", (7, 5)),
            ("top", (9, 8)),
            ("left", (6, 6)),
        ],
    )
    out1 = main(case1)
    assert _find_frame(out1) == (1, 2, 10, 10)

    bbox2 = (4, 4, 10, 11)
    case2 = render_case(
        (17, 17),
        bbox2,
        [
            ("top", (6, 5)),
            ("right", (8, 9)),
        ],
    )
    out2 = main(case2)
    assert _find_frame(out2) == (2, 4, 10, 13)

    bbox3 = (6, 6, 12, 12)
    case3 = render_case(
        (22, 22),
        bbox3,
        [
            ("top", (8, 7)),
            ("bottom", (10, 11)),
            ("left", (9, 9)),
            ("right", (8, 10)),
        ],
    )
    out3 = main(case3)
    assert _find_frame(out3) == (4, 3, 14, 14)

    bbox4 = (5, 5, 11, 11)
    case4 = render_case((18, 18), bbox4, [("bottom", (7, 8))])
    out4 = main(case4)
    assert _find_frame(out4) == (5, 5, 15, 11)

    _SELF_TESTS_DONE = True


def generate_input() -> np.ndarray:
    _run_internal_self_tests_once()
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
