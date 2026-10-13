import hashlib
import itertools
import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

CELL_PITCH = 4
FRAME_SIZE = 3
CENTER_OFFSET = 1
MIN_CELLS = 3
MAX_CELLS = 7
MAX_ATTEMPTS = 400

_USED_OUTPUT_SIGNATURES: set[str] = set()
_GEN_COUNTER = 0


def _payload_centers(n_cells: int) -> list[int]:
    return [CELL_PITCH * idx + CENTER_OFFSET for idx in range(n_cells)]


def _build_scaffold(n_cells: int) -> np.ndarray:
    payload = np.zeros((CELL_PITCH * n_cells, CELL_PITCH * n_cells), dtype=int)
    for br in range(n_cells):
        for bc in range(n_cells):
            r0 = br * CELL_PITCH
            c0 = bc * CELL_PITCH
            payload[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = YELLOW
            payload[r0 + CENTER_OFFSET, c0 + CENTER_OFFSET] = BLACK
    return payload


def _control_offsets(grid: np.ndarray) -> tuple[int, int]:
    top_hits = np.flatnonzero(grid[0, 1:] == RED)
    left_hits = np.flatnonzero(grid[1:, 0] == RED)
    assert top_hits.size == 1 and left_hits.size == 1, "expected one red marker in each control band"
    return int(top_hits[0]), int(left_hits[0])


def _crop_payload(grid: np.ndarray) -> np.ndarray:
    payload = np.asarray(grid, dtype=int)[1:, 1:]
    assert payload.ndim == 2 and payload.shape[0] == payload.shape[1], "payload must be square"
    assert payload.shape[0] % CELL_PITCH == 0, "payload size must be divisible by 4"
    return payload


def _roll_payload(payload: np.ndarray, shift_h: int, shift_v: int) -> np.ndarray:
    return np.roll(payload, shift=(shift_v * CELL_PITCH, shift_h * CELL_PITCH), axis=(0, 1))


def _marker_cells(payload: np.ndarray, color: int) -> list[tuple[int, int]]:
    n_cells = payload.shape[0] // CELL_PITCH
    cells: list[tuple[int, int]] = []
    for br in range(n_cells):
        for bc in range(n_cells):
            rr = br * CELL_PITCH + CENTER_OFFSET
            cc = bc * CELL_PITCH + CENTER_OFFSET
            if int(payload[rr, cc]) == color:
                cells.append((br, bc))
    return cells


def _paint_frame(payload: np.ndarray, br: int, bc: int, color: int) -> None:
    r0 = br * CELL_PITCH
    c0 = bc * CELL_PITCH
    payload[r0, c0 : c0 + FRAME_SIZE] = color
    payload[r0 + FRAME_SIZE - 1, c0 : c0 + FRAME_SIZE] = color
    payload[r0 : r0 + FRAME_SIZE, c0] = color
    payload[r0 : r0 + FRAME_SIZE, c0 + FRAME_SIZE - 1] = color


def _expand_color(payload: np.ndarray, color: int) -> None:
    cells = _marker_cells(payload, color)
    assert len(cells) >= 2, f"expected at least two markers of color {color}"
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    for br in range(r0, r1 + 1):
        for bc in range(c0, c1 + 1):
            rr = br * CELL_PITCH + CENTER_OFFSET
            cc = bc * CELL_PITCH + CENTER_OFFSET
            payload[rr, cc] = color
            if br in (r0, r1) or bc in (c0, c1):
                _paint_frame(payload, br, bc, color)
                payload[rr, cc] = color


def _scaffold_like(payload: np.ndarray) -> bool:
    n_cells = payload.shape[0] // CELL_PITCH
    for br in range(n_cells):
        for bc in range(n_cells):
            r0 = br * CELL_PITCH
            c0 = bc * CELL_PITCH
            block = payload[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE]
            if block.shape != (FRAME_SIZE, FRAME_SIZE):
                return False
            border = block.copy()
            border[CENTER_OFFSET, CENTER_OFFSET] = YELLOW
            if not np.all(border == YELLOW):
                return False
            if int(block[CENTER_OFFSET, CENTER_OFFSET]) not in (BLACK, BLUE, GREEN):
                return False
    for idx in range(CELL_PITCH - 1, payload.shape[0], CELL_PITCH):
        if np.any(payload[idx, :] != BLACK):
            return False
        if np.any(payload[:, idx] != BLACK):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the border shifts, cyclically relocate the scaffold markers, then expand each marker color into a scaffold-aligned rectangle.

    Input structure:
    - The top row and left column are grey control bands, each containing one red marker pixel that encodes a cell shift.
    - The lower-right payload is a square lattice whose side length is a multiple of four.
    - Every 4x4 payload cell contains a yellow hollow 3x3 frame with a single center pixel, and some centers carry green or blue markers.

    Rule:
    - Read the red marker position in the top band as the horizontal cell shift, and the red marker position in the left band as the vertical cell shift.
    - Remove the control bands and cyclically roll the payload right/down by whole-cell steps of four pixels.
    - Re-detect the green and blue center markers on the shifted scaffold.
    - For each color, paint the smallest cell-aligned rectangle covering its markers: recolor the perimeter cell frames and fill every enclosed center. Apply green first, then blue so blue overwrites overlaps.

    Key insight:
    - The border markers do not point to the final rectangles directly; they first relocate the marker centers on the periodic scaffold, and only those shifted marker positions determine the expansion rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1] and g.shape[0] >= 1 + CELL_PITCH * MIN_CELLS
    assert np.all(g[0, 0] == GREY)
    assert np.all(np.isin(g[0, 1:], [GREY, RED]))
    assert np.all(np.isin(g[1:, 0], [GREY, RED]))

    shift_h, shift_v = _control_offsets(g)
    payload = _crop_payload(g)
    assert _scaffold_like(payload), "payload does not match the scaffold layout"
    shifted = _roll_payload(payload, shift_h, shift_v)

    out = shifted.copy()
    for color in (GREEN, BLUE):
        _expand_color(out, color)
    return out


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _sample_rect(n_cells: int, kind: str) -> tuple[int, int, int, int]:
    for _ in range(200):
        if kind == "line_h":
            width = random.randint(2, min(5, n_cells))
            row = random.randrange(n_cells)
            c0 = random.randint(0, n_cells - width)
            return (row, c0, row, c0 + width - 1)
        if kind == "line_v":
            height = random.randint(2, min(5, n_cells))
            col = random.randrange(n_cells)
            r0 = random.randint(0, n_cells - height)
            return (r0, col, r0 + height - 1, col)

        max_span = min(4, n_cells)
        height = random.randint(2, max_span)
        width = random.randint(2, max_span)
        r0 = random.randint(0, n_cells - height)
        c0 = random.randint(0, n_cells - width)
        rect = (r0, c0, r0 + height - 1, c0 + width - 1)
        if kind == "box":
            return rect
        if kind == "wide" and width > 1:
            return rect
        if kind == "tall" and height > 1:
            return rect
    raise RuntimeError("failed to sample rectangle")


def _sample_overlapping_pair(n_cells: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    for _ in range(400):
        a = _sample_rect(n_cells, random.choice(["box", "wide", "tall"]))
        b = _sample_rect(n_cells, random.choice(["box", "wide", "tall"]))
        cells_a = _rect_cells(a)
        cells_b = _rect_cells(b)
        inter = cells_a & cells_b
        if not inter:
            continue
        if inter == cells_a or inter == cells_b:
            continue
        return a, b
    raise RuntimeError("failed to sample overlapping rectangles")


def _sample_disjoint_pair(n_cells: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    for _ in range(400):
        kinds = [
            random.choice(["box", "line_h", "line_v"]),
            random.choice(["box", "line_h", "line_v"]),
        ]
        a = _sample_rect(n_cells, kinds[0])
        b = _sample_rect(n_cells, kinds[1])
        if _rect_cells(a).isdisjoint(_rect_cells(b)):
            return a, b
    raise RuntimeError("failed to sample disjoint rectangles")


def _rect_markers(rect: tuple[int, int, int, int], occupied: set[tuple[int, int]], mode: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    all_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    available = [cell for cell in all_cells if cell not in occupied]
    if len(available) < 2:
        raise RuntimeError("not enough free cells for marker placement")

    combos: list[tuple[tuple[int, int], ...]] = []
    for size in range(2, min(4, len(available)) + 1):
        for combo in itertools.combinations(available, size):
            rows = [r for r, _ in combo]
            cols = [c for _, c in combo]
            if min(rows) == r0 and max(rows) == r1 and min(cols) == c0 and max(cols) == c1:
                combos.append(combo)

    if not combos:
        raise RuntimeError("failed to find a bbox-defining marker set")

    random.shuffle(combos)
    combos = combos[mode % len(combos) :] + combos[: mode % len(combos)]
    return list(combos[0])
    raise RuntimeError("failed to place markers without overlap")


def _place_markers(payload: np.ndarray, color: int, markers: Iterable[tuple[int, int]]) -> None:
    for br, bc in markers:
        rr = br * CELL_PITCH + CENTER_OFFSET
        cc = bc * CELL_PITCH + CENTER_OFFSET
        payload[rr, cc] = color


def _render_shifted_payload(
    n_cells: int,
    green_markers: list[tuple[int, int]],
    blue_markers: list[tuple[int, int]],
) -> np.ndarray:
    payload = _build_scaffold(n_cells)
    _place_markers(payload, GREEN, green_markers)
    _place_markers(payload, BLUE, blue_markers)
    return payload


def _assemble_input(payload: np.ndarray, shift_h: int, shift_v: int) -> np.ndarray:
    n_cells = payload.shape[0] // CELL_PITCH
    grid = np.full((payload.shape[0] + 1, payload.shape[1] + 1), GREY, dtype=int)
    grid[1:, 1:] = payload
    grid[0, 1:] = GREY
    grid[1:, 0] = GREY
    grid[0, 1 + shift_h] = RED
    grid[1 + shift_v, 0] = RED
    assert 0 <= shift_h < n_cells and 0 <= shift_v < n_cells
    return grid


def _oracle_expand_without_shift(grid: np.ndarray) -> np.ndarray:
    payload = _crop_payload(grid)
    out = payload.copy()
    for color in (GREEN, BLUE):
        _expand_color(out, color)
    return out


def _rectangles_after_shift(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    shift_h, shift_v = _control_offsets(grid)
    payload = _roll_payload(_crop_payload(grid), shift_h, shift_v)
    result: dict[int, tuple[int, int, int, int]] = {}
    for color in (GREEN, BLUE):
        cells = _marker_cells(payload, color)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        result[color] = (min(rows), min(cols), max(rows), max(cols))
    return result


def _output_signature(grid: np.ndarray) -> str:
    out = main(grid)
    return hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        shift_h, shift_v = _control_offsets(grid)
        payload = _crop_payload(grid)
        n_cells = payload.shape[0] // CELL_PITCH
        if not (0 <= shift_h < n_cells and 0 <= shift_v < n_cells):
            return False
        if shift_h == 0 and shift_v == 0:
            return False

        shifted = _roll_payload(payload, shift_h, shift_v)
        out = main(grid)
        shortcut = _oracle_expand_without_shift(grid)
        if np.array_equal(out, shortcut):
            return False
        if np.array_equal(out, shifted):
            return False

        rects_shifted = _rectangles_after_shift(grid)
        rects_unshifted = {}
        for color in (GREEN, BLUE):
            cells = _marker_cells(payload, color)
            if len(cells) < 2:
                return False
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            rects_unshifted[color] = (min(rows), min(cols), max(rows), max(cols))
        if rects_unshifted == rects_shifted:
            return False

        if rects_shifted[GREEN] == rects_shifted[BLUE]:
            return False

        inter = _rect_cells(rects_shifted[GREEN]) & _rect_cells(rects_shifted[BLUE])
        if inter:
            green_only = _rect_cells(rects_shifted[GREEN]) - inter
            blue_only = _rect_cells(rects_shifted[BLUE]) - inter
            if not green_only or not blue_only:
                return False
            test_out = main(grid)
            for br, bc in inter:
                rr = br * CELL_PITCH + CENTER_OFFSET
                cc = bc * CELL_PITCH + CENTER_OFFSET
                if int(test_out[rr, cc]) != BLUE:
                    return False

        sig = _output_signature(grid)
        if sig in _USED_OUTPUT_SIGNATURES:
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    global _GEN_COUNTER
    mode = _GEN_COUNTER % 4

    n_cells = random.randint(MIN_CELLS, MAX_CELLS)
    shift_h = random.randrange(n_cells)
    shift_v = random.randrange(n_cells)
    if shift_h == 0 and shift_v == 0:
        shift_h = random.randrange(1, n_cells)

    try:
        if mode in (0, 2):
            rect_g, rect_b = _sample_overlapping_pair(n_cells)
        else:
            rect_g, rect_b = _sample_disjoint_pair(n_cells)

        occupied: set[tuple[int, int]] = set()
        green_markers = _rect_markers(rect_g, occupied, mode)
        occupied.update(green_markers)
        blue_markers = _rect_markers(rect_b, occupied, mode + 1)
        occupied.update(blue_markers)
    except RuntimeError:
        return None

    shifted_payload = _render_shifted_payload(n_cells, green_markers, blue_markers)
    input_payload = _roll_payload(shifted_payload, (-shift_h) % n_cells, (-shift_v) % n_cells)
    grid = _assemble_input(input_payload, shift_h, shift_v)

    if not _semantic_checks(grid):
        return None

    _USED_OUTPUT_SIGNATURES.add(_output_signature(grid))
    _GEN_COUNTER += 1
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
