from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

ROLE_FROM_SHAPE = {
    ((0, 0), (0, 1), (1, 0)): "TL",
    ((0, 0), (0, 1), (1, 1)): "TR",
    ((0, 0), (1, 0), (1, 1)): "BL",
    ((0, 1), (1, 0), (1, 1)): "BR",
}

ORTH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORTH8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

GRID_ROWS = (22, 28)
GRID_COLS = (22, 28)
FRAME_COUNT_CHOICES = (2, 3)
FRAME_SIZE_MIN = 8
FRAME_SIZE_MAX = 11
FRAME_GAP = 2
GEN_ATTEMPTS = 400


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ORTH4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _corner_anchor(role: str, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if role == "TL":
        return r0, c0
    if role == "TR":
        return r0, c1
    if role == "BL":
        return r1, c0
    return r1, c1


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    corners: dict[str, set[tuple[int, int]]] = {role: set() for role in ("TL", "TR", "BL", "BR")}
    comps = _components_of_color(grid, GREEN)
    assert comps, "expected green corner markers"

    for comp in comps:
        assert len(comp) == 3, "green components must be 2x2 L triominoes"
        r0, c0, r1, c1 = _bbox_coords(comp)
        assert (r1 - r0, c1 - c0) == (1, 1), "green corner markers must occupy 2x2 boxes"
        rel = tuple(sorted((r - r0, c - c0) for r, c in comp))
        assert rel in ROLE_FROM_SHAPE, "invalid green corner marker"
        role = ROLE_FROM_SHAPE[rel]
        corners[role].add(_corner_anchor(role, (r0, c0, r1, c1)))

    frames: list[tuple[int, int, int, int]] = []
    used: set[tuple[str, tuple[int, int]]] = set()
    for top, left in sorted(corners["TL"]):
        for top2, right in sorted(corners["TR"]):
            if top2 != top or right <= left:
                continue
            for bottom, left2 in sorted(corners["BL"]):
                if left2 != left or bottom <= top:
                    continue
                if (bottom, right) in corners["BR"]:
                    frame = (top, left, bottom, right)
                    frames.append(frame)
                    used.update(
                        {
                            ("TL", (top, left)),
                            ("TR", (top, right)),
                            ("BL", (bottom, left)),
                            ("BR", (bottom, right)),
                        }
                    )

    total_corners = sum(len(v) for v in corners.values())
    assert total_corners == len(used), "green markers must form disjoint complete frames"
    assert frames, "no valid frame found"
    return sorted(frames)


def _classify_red_pixels(grid: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[int, int]:
    top, left, bottom, right = frame
    nw_se = 0
    ne_sw = 0
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if int(grid[r, c]) != RED:
                continue
            has_nw_se = (
                (r - 1 >= top + 1 and c - 1 >= left + 1 and int(grid[r - 1, c - 1]) == RED)
                or (r + 1 <= bottom - 1 and c + 1 <= right - 1 and int(grid[r + 1, c + 1]) == RED)
            )
            has_ne_sw = (
                (r - 1 >= top + 1 and c + 1 <= right - 1 and int(grid[r - 1, c + 1]) == RED)
                or (r + 1 <= bottom - 1 and c - 1 >= left + 1 and int(grid[r + 1, c - 1]) == RED)
            )
            assert has_nw_se ^ has_ne_sw, "every red pixel must belong to exactly one diagonal orientation"
            if has_nw_se:
                nw_se += 1
            else:
                ne_sw += 1
    assert nw_se != ne_sw, "red orientation counts must be unequal"
    return nw_se, ne_sw


def _axis_from_red_counts(grid: np.ndarray, frame: tuple[int, int, int, int]) -> str:
    nw_se, ne_sw = _classify_red_pixels(grid, frame)
    return "vertical" if nw_se > ne_sw else "horizontal"


def _mirror_coord(cell: tuple[int, int], frame: tuple[int, int, int, int], axis: str) -> tuple[int, int]:
    r, c = cell
    top, left, bottom, right = frame
    if axis == "vertical":
        return r, left + right - c
    return top + bottom - r, c


def _frame_output(grid: np.ndarray, frame: tuple[int, int, int, int], axis: str) -> np.ndarray:
    out = grid.copy()
    top, left, bottom, right = frame
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if int(grid[r, c]) in (BLUE, RED):
                out[r, c] = BLACK
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if int(grid[r, c]) != BLUE:
                continue
            out[r, c] = BLUE
            mr, mc = _mirror_coord((r, c), frame, axis)
            if int(grid[mr, mc]) == BLACK:
                out[mr, mc] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use each frame's majority red diagonal orientation to choose the blue pattern's mirror-completion axis, then erase the red code and outside blue noise.

    Input structure:
    - A black background contains 2-3 non-overlapping rectangular frames identified by four green 2x2 L-corner triominoes.
    - Inside each frame there is a blue pattern missing one mirrored quadrant copy plus several short red diagonal segments.
    - Red segments are isolated 2-3 pixel diagonals whose majority orientation is either NW-SE or NE-SW inside that frame.
    - A few extra blue pixels may appear outside all frames as noise.

    Rule:
    - Detect each green-corner frame and inspect only the red pixels inside that frame.
    - Classify every red pixel by whether it has a red neighbor on the NW-SE diagonal or on the NE-SW diagonal.
    - If NW-SE red pixels are the majority, complete the blue pattern by mirroring it left-right across the frame's vertical center axis; otherwise mirror it top-bottom across the horizontal center axis.
    - Add mirrored blue only onto cells that were black in the input, then erase all red pixels and all blue pixels outside every frame while keeping the green corners unchanged.

    Key insight:
    - The frame geometry does not choose the symmetry axis; the hidden instruction is encoded independently by the majority orientation of the red diagonal stripes inside each frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _find_frames(g)

    out = g.copy()
    out[g == BLUE] = BLACK
    out[g == RED] = BLACK
    for frame in frames:
        axis = _axis_from_red_counts(g, frame)
        frame_completed = _frame_output(g, frame, axis)
        top, left, bottom, right = frame
        out[top : bottom + 1, left : right + 1] = frame_completed[top : bottom + 1, left : right + 1]
    return out


def _pair_lists(start: int, end: int) -> tuple[list[int], list[int]]:
    first = [v for v in range(start, end + 1) if v < start + end - v]
    second = [start + end - v for v in first]
    return first, second


def _grow_cells(candidates: list[tuple[int, int]], count: int) -> set[tuple[int, int]]:
    pool = list(candidates)
    chosen = {random.choice(pool)}
    while len(chosen) < count:
        frontier = [
            cell
            for cell in pool
            if cell not in chosen
            and any(max(abs(cell[0] - r), abs(cell[1] - c)) <= 1 for r, c in chosen)
        ]
        options = frontier if frontier else [cell for cell in pool if cell not in chosen]
        chosen.add(random.choice(options))
    return chosen


def _choose_two_distinct(lo: int, hi: int) -> tuple[int, int]:
    assert lo < hi
    a = random.randint(lo, hi)
    b_choices = [v for v in range(lo, hi + 1) if v != a]
    return a, random.choice(b_choices)


def _build_blue_pattern(frame: tuple[int, int, int, int], axis: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]], str]:
    top, left, bottom, right = frame
    row_a, row_b = _pair_lists(top + 1, bottom - 1)
    col_a, col_b = _pair_lists(left + 1, right - 1)
    assert len(row_a) >= 2 and len(col_a) >= 2

    if axis == "vertical":
        target_side = random.choice(("left", "right"))
        source_cols = col_b if target_side == "left" else col_a
        target_cols = col_a if target_side == "left" else col_b
        top_rows, bottom_rows = row_a, row_b
        top_count, bottom_count = _choose_two_distinct(2, min(4, len(source_cols) * len(top_rows) - 1))
        top_source = _grow_cells([(r, c) for r in top_rows for c in source_cols], top_count)
        bottom_source = _grow_cells([(r, c) for r in bottom_rows for c in source_cols], bottom_count)
        source = top_source | bottom_source
        full = set(source)
        for cell in source:
            full.add(_mirror_coord(cell, frame, axis))
        missing_part = random.choice(("top", "bottom"))
        missing_rows = set(top_rows if missing_part == "top" else bottom_rows)
        missing_cols = set(target_cols)
    else:
        target_side = random.choice(("top", "bottom"))
        source_rows = row_b if target_side == "top" else row_a
        target_rows = row_a if target_side == "top" else row_b
        left_cols, right_cols = col_a, col_b
        left_count, right_count = _choose_two_distinct(2, min(4, len(source_rows) * len(left_cols) - 1))
        left_source = _grow_cells([(r, c) for r in source_rows for c in left_cols], left_count)
        right_source = _grow_cells([(r, c) for r in source_rows for c in right_cols], right_count)
        source = left_source | right_source
        full = set(source)
        for cell in source:
            full.add(_mirror_coord(cell, frame, axis))
        missing_part = random.choice(("left", "right"))
        missing_rows = set(target_rows)
        missing_cols = set(left_cols if missing_part == "left" else right_cols)

    removed = {cell for cell in full if cell[0] in missing_rows and cell[1] in missing_cols}
    kept = full - removed
    assert removed, "pattern must miss one mirrored quadrant copy"
    assert any(cell[0] in missing_rows for cell in removed)
    assert any(cell[1] in missing_cols for cell in removed)
    return full, kept, axis


def _segment_candidates(
    frame: tuple[int, int, int, int],
    orientation: str,
    forbidden: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], ...]]:
    top, left, bottom, right = frame
    candidates: list[tuple[tuple[int, int], ...]] = []
    for length in (2, 3):
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if orientation == "nw_se":
                    coords = tuple((r + i, c + i) for i in range(length))
                else:
                    coords = tuple((r + i, c - i) for i in range(length))
                if not all(top + 1 <= rr <= bottom - 1 and left + 1 <= cc <= right - 1 for rr, cc in coords):
                    continue
                if any(cell in forbidden or cell in blocked for cell in coords):
                    continue
                candidates.append(coords)
    random.shuffle(candidates)
    return candidates


def _place_red_segments(
    frame: tuple[int, int, int, int],
    axis: str,
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    majority = "nw_se" if axis == "vertical" else "ne_sw"
    minority = "ne_sw" if majority == "nw_se" else "nw_se"
    maj_count, min_count = random.choice([(3, 1), (4, 1), (4, 2), (5, 0)])

    placed: list[tuple[tuple[int, int], ...]] = []
    blocked: set[tuple[int, int]] = set()
    for orientation, count in ((majority, maj_count), (minority, min_count)):
        for _ in range(count):
            candidates = _segment_candidates(frame, orientation, forbidden, blocked)
            if not candidates:
                raise ValueError("unable to place red segments")
            seg = candidates[0]
            placed.append(seg)
            for r, c in seg:
                for dr, dc in ORTH8:
                    blocked.add((r + dr, c + dc))
                blocked.add((r, c))

    red_cells = {cell for seg in placed for cell in seg}
    input_grid = np.zeros((frame[2] + 1, frame[3] + 1), dtype=int)
    for r, c in red_cells:
        input_grid[r, c] = RED
    nw_se, ne_sw = _classify_red_pixels(input_grid, frame)
    assert (nw_se > ne_sw) == (axis == "vertical")
    return red_cells


def _paint_frame_corners(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = frame
    for cell in ((top, left), (top, left + 1), (top + 1, left)):
        grid[cell] = GREEN
    for cell in ((top, right), (top, right - 1), (top + 1, right)):
        grid[cell] = GREEN
    for cell in ((bottom, left), (bottom - 1, left), (bottom, left + 1)):
        grid[cell] = GREEN
    for cell in ((bottom, right), (bottom - 1, right), (bottom, right - 1)):
        grid[cell] = GREEN


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] + FRAME_GAP < b[0] or b[2] + FRAME_GAP < a[0] or a[3] + FRAME_GAP < b[1] or b[3] + FRAME_GAP < a[1]


def _place_boxes(rows: int, cols: int, count: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    used_tops: set[int] = set()
    used_bottoms: set[int] = set()
    used_lefts: set[int] = set()
    used_rights: set[int] = set()

    for _ in range(count):
        candidates: list[tuple[int, int, int, int]] = []
        for h in range(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1):
            for w in range(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1):
                for top in range(1, rows - h):
                    bottom = top + h - 1
                    if top in used_tops or bottom in used_bottoms:
                        continue
                    for left in range(1, cols - w):
                        right = left + w - 1
                        if left in used_lefts or right in used_rights:
                            continue
                        box = (top, left, bottom, right)
                        if all(_boxes_far_enough(box, other) for other in boxes):
                            candidates.append(box)
        if not candidates:
            return None
        box = random.choice(candidates)
        boxes.append(box)
        used_tops.add(box[0])
        used_bottoms.add(box[2])
        used_lefts.add(box[1])
        used_rights.add(box[3])
    return boxes


def _assign_axes(boxes: list[tuple[int, int, int, int]]) -> list[str]:
    axes = ["vertical", "horizontal"]
    while len(axes) < len(boxes):
        axes.append(random.choice(("vertical", "horizontal")))
    random.shuffle(axes)
    info = list(zip(boxes, axes))
    has_contradiction = any(
        ((box[3] - box[1]) > (box[2] - box[0]) and axis == "horizontal")
        or ((box[2] - box[0]) > (box[3] - box[1]) and axis == "vertical")
        for box, axis in info
    )
    if not has_contradiction:
        for i, (box, axis) in enumerate(info):
            h = box[2] - box[0]
            w = box[3] - box[1]
            if w != h:
                info[i] = (box, "horizontal" if axis == "vertical" else "vertical")
                break
    return [axis for _, axis in info]


def _build_instance() -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    count = random.choice(FRAME_COUNT_CHOICES)
    boxes = _place_boxes(rows, cols, count)
    if boxes is None:
        raise ValueError("failed to place boxes")
    axes = _assign_axes(boxes)

    grid = np.zeros((rows, cols), dtype=int)
    expected = np.zeros((rows, cols), dtype=int)
    infos: list[dict[str, object]] = []
    occupied_outside: set[tuple[int, int]] = set()

    for frame, axis in zip(boxes, axes):
        _paint_frame_corners(grid, frame)
        _paint_frame_corners(expected, frame)
        full_blue, input_blue, _ = _build_blue_pattern(frame, axis)
        red_cells = _place_red_segments(frame, axis, full_blue)

        for cell in input_blue:
            grid[cell] = BLUE
        for cell in red_cells:
            grid[cell] = RED
        for cell in full_blue:
            expected[cell] = BLUE

        infos.append(
            {
                "frame": frame,
                "axis": axis,
                "full_blue": full_blue,
                "input_blue": input_blue,
                "red_cells": red_cells,
            }
        )

        top, left, bottom, right = frame
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                occupied_outside.add((r, c))

    noise_count = random.randint(1, 3)
    outside = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in occupied_outside and int(grid[r, c]) == BLACK
    ]
    if len(outside) < noise_count:
        raise ValueError("not enough room for outside noise")
    for cell in random.sample(outside, noise_count):
        grid[cell] = BLUE

    return grid, expected, infos


def _wrong_axis_result(grid: np.ndarray, frame: tuple[int, int, int, int], axis: str) -> np.ndarray:
    wrong = "horizontal" if axis == "vertical" else "vertical"
    return _frame_output(grid, frame, wrong)


def _validate_instance(grid: np.ndarray, expected: np.ndarray, infos: list[dict[str, object]]) -> bool:
    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    frames = _find_frames(grid)
    if len(frames) != len(infos):
        return False

    axes = [str(info["axis"]) for info in infos]
    if "vertical" not in axes or "horizontal" not in axes:
        return False
    if not any(
        ((frame[3] - frame[1]) > (frame[2] - frame[0]) and axis == "horizontal")
        or ((frame[2] - frame[0]) > (frame[3] - frame[1]) and axis == "vertical")
        for frame, axis in zip(frames, axes)
    ):
        return False

    if np.count_nonzero((grid == BLUE) & (expected == BLACK)) == 0:
        return False

    for info in infos:
        frame = info["frame"]
        axis = str(info["axis"])
        full_blue = set(info["full_blue"])
        input_blue = set(info["input_blue"])
        red_cells = set(info["red_cells"])
        top, left, bottom, right = frame

        if _axis_from_red_counts(grid, frame) != axis:
            return False
        if not red_cells:
            return False
        if input_blue == full_blue:
            return False

        frame_expected = expected[top : bottom + 1, left : right + 1]
        frame_solved = solved[top : bottom + 1, left : right + 1]
        if not np.array_equal(frame_expected, frame_solved):
            return False
        if np.any(frame_solved == RED):
            return False

        wrong = _wrong_axis_result(grid, frame, axis)
        if np.array_equal(
            wrong[top : bottom + 1, left : right + 1],
            frame_expected,
        ):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected, infos = _build_instance()
        except Exception:
            continue
        if _validate_instance(grid, expected, infos):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
