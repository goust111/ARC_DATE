from __future__ import annotations

import math
import random

import numpy as np


BG = 0
FRAME = 5
SEP = 8
CELL = 5
INNER = CELL - 2
HEIGHT = CELL * 2 + 1
REF_COUNTS = (5, 6)
QUERY_COUNTS = (4, 5)
GLYPH_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_ATTEMPTS = 300


def _mask_to_array(mask: int) -> np.ndarray:
    bits = [(mask >> i) & 1 for i in range(INNER * INNER)]
    return np.array(bits, dtype=bool).reshape(INNER, INNER)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(int(v) for v in pts[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _build_pattern_masks() -> list[int]:
    out: list[int] = []
    for mask in range(1, 1 << (INNER * INNER)):
        arr = _mask_to_array(mask)
        filled = int(arr.sum())
        if not 3 <= filled <= 6:
            continue
        coords = np.argwhere(arr)
        if int(coords[:, 0].min()) != 0 or int(coords[:, 0].max()) != INNER - 1:
            continue
        if int(coords[:, 1].min()) != 0 or int(coords[:, 1].max()) != INNER - 1:
            continue
        if _is_connected(arr):
            out.append(mask)
    return out


PATTERN_MASKS = _build_pattern_masks()


def _render_inner(mask: int, color: int) -> np.ndarray:
    inner = np.full((INNER, INNER), BG, dtype=int)
    inner[_mask_to_array(mask)] = int(color)
    return inner


def _make_cell(inner: np.ndarray | None = None) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    cell[0, :] = FRAME
    cell[-1, :] = FRAME
    cell[:, 0] = FRAME
    cell[:, -1] = FRAME
    if inner is not None:
        cell[1:-1, 1:-1] = inner
    return cell


def _has_frame(cell: np.ndarray, frame_color: int) -> bool:
    return (
        cell.shape[0] >= 3
        and cell.shape[1] >= 3
        and np.all(cell[0, :] == frame_color)
        and np.all(cell[-1, :] == frame_color)
        and np.all(cell[:, 0] == frame_color)
        and np.all(cell[:, -1] == frame_color)
    )


def _cell_inner(cell: np.ndarray) -> np.ndarray:
    return np.asarray(cell[1:-1, 1:-1], dtype=int)


def _signature(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in arr.flatten())


def _uniform_rows(g: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(g[r, 0])) for r in range(g.shape[0]) if np.all(g[r, :] == g[r, 0])]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the last query cell with the next equal-stride jump around the reference glyph cycle.

    Input structure:
    - A full-width separator row splits the grid into a top query band and a bottom
      reference band of the same height.
    - The bottom band is a left-to-right row of equal cells; each cell contains one
      distinct colored glyph, and the row is interpreted cyclically.
    - The top band contains a shorter left-aligned row of the same cell type; its
      first cells contain copied glyphs from the reference cycle and its last cell is empty.

    Rule:
    - Match each filled query glyph to its unique position in the reference row.
    - The matched reference positions advance by one constant stride modulo the
      reference length.
    - Advance once more by the same stride from the last filled query glyph and copy
      that reference glyph into the empty query cell.

    Key insight:
    - The bottom row is not a simple lookup table but a ring, so the next glyph is
      determined by a cyclic constant-step walk rather than by adjacency or set difference.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    uniform_rows = _uniform_rows(g)
    color_counts: dict[int, int] = {}
    for _r, color in uniform_rows:
        color_counts[color] = color_counts.get(color, 0) + 1
    sep_candidates = [r for r, color in uniform_rows if 0 < r < h - 1 and color_counts[color] == 1]
    assert len(sep_candidates) == 1, "expected one separator row color"
    sep_row = int(sep_candidates[0])

    cell = sep_row
    assert h == 2 * cell + 1, "top and bottom bands must have the same height"
    assert cell >= 3 and w % cell == 0, "invalid cell geometry"

    frame_color = int(g[sep_row + 1, 0])
    assert np.all(g[sep_row + 1, :] == frame_color), "expected a full-width framed reference row"
    assert np.all(g[-1, :] == frame_color), "expected framed reference cells"

    top = g[:sep_row, :]
    bottom = g[sep_row + 1 :, :]
    ref_count = w // cell

    reference_inners: list[np.ndarray] = []
    reference_map: dict[tuple[int, ...], int] = {}
    for idx in range(ref_count):
        c0 = idx * cell
        ref_cell = bottom[:, c0 : c0 + cell]
        assert _has_frame(ref_cell, frame_color), "reference cells must share one frame style"
        inner = _cell_inner(ref_cell).copy()
        sig = _signature(inner)
        assert sig not in reference_map, "reference glyphs must be unique"
        reference_map[sig] = idx
        reference_inners.append(inner)

    query_cells: list[np.ndarray] = []
    seen_gap = False
    for idx in range(ref_count):
        c0 = idx * cell
        cell_arr = top[:, c0 : c0 + cell]
        if _has_frame(cell_arr, frame_color):
            assert not seen_gap, "query cells must be a single left-aligned run"
            query_cells.append(cell_arr.copy())
        else:
            seen_gap = True
    query_count = len(query_cells)
    assert query_count >= 3, "expected at least two filled query glyphs and one empty cell"

    filled_sigs = [_signature(_cell_inner(cell_arr)) for cell_arr in query_cells[:-1]]
    assert len(filled_sigs) >= 2, "need enough filled cells to infer a stride"
    for sig in filled_sigs:
        assert sig in reference_map, "query glyph must appear in the reference cycle"

    empty_inner = _cell_inner(query_cells[-1])
    assert np.all(empty_inner == empty_inner[0, 0]), "last query cell must be empty"

    positions = [reference_map[sig] for sig in filled_sigs]
    stride = (positions[1] - positions[0]) % ref_count
    assert stride != 0, "stride must move around the ring"
    for prev, cur in zip(positions, positions[1:]):
        assert (cur - prev) % ref_count == stride, "query glyphs must follow one constant stride"

    next_pos = (positions[-1] + stride) % ref_count
    out = g.copy()
    c0 = (query_count - 1) * cell
    out[1:cell - 1, c0 + 1 : c0 + cell - 1] = reference_inners[next_pos]
    return out


def _sequence_candidates(ref_count: int, query_count: int) -> dict[str, list[tuple[int, int]]]:
    kinds = {"no_wrap": [], "output_wrap": [], "visible_wrap": []}
    for stride in range(1, ref_count):
        period = ref_count // math.gcd(ref_count, stride)
        if period < query_count:
            continue
        for start in range(ref_count):
            last_filled_raw = start + (query_count - 2) * stride
            output_raw = start + (query_count - 1) * stride
            if output_raw < ref_count:
                kind = "no_wrap"
            elif last_filled_raw < ref_count:
                kind = "output_wrap"
            else:
                kind = "visible_wrap"
            kinds[kind].append((start, stride))
    return kinds


def _expected_output(grid: np.ndarray, ref_inners: list[np.ndarray], query_count: int, next_pos: int) -> np.ndarray:
    out = grid.copy()
    c0 = (query_count - 1) * CELL
    out[1:CELL - 1, c0 + 1 : c0 + CELL - 1] = ref_inners[next_pos]
    return out


def _gen_attempt() -> np.ndarray | None:
    ref_count = random.choice(REF_COUNTS)
    query_count = random.choice([q for q in QUERY_COUNTS if q < ref_count])

    sequence_kinds = _sequence_candidates(ref_count, query_count)
    preferred = random.choice(
        [kind for kind in ("visible_wrap", "output_wrap", "no_wrap", "visible_wrap") if sequence_kinds[kind]]
    )
    start, stride = random.choice(sequence_kinds[preferred])
    positions = [int((start + i * stride) % ref_count) for i in range(query_count)]

    masks = random.sample(PATTERN_MASKS, ref_count)
    color_pool = random.sample(GLYPH_COLORS, random.choice((3, 4)))
    ref_inners = [_render_inner(mask, random.choice(color_pool)) for mask in masks]

    width = ref_count * CELL
    grid = np.full((HEIGHT, width), BG, dtype=int)
    grid[CELL, :] = SEP

    for idx, inner in enumerate(ref_inners):
        c0 = idx * CELL
        grid[CELL + 1 :, c0 : c0 + CELL] = _make_cell(inner)

    for slot, pos in enumerate(positions[:-1]):
        c0 = slot * CELL
        grid[:CELL, c0 : c0 + CELL] = _make_cell(ref_inners[pos])

    empty_c0 = (query_count - 1) * CELL
    grid[:CELL, empty_c0 : empty_c0 + CELL] = _make_cell()

    expected = _expected_output(grid, ref_inners, query_count, positions[-1])
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
