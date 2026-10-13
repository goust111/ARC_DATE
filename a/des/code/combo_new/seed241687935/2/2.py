from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class EdgeProfile:
    kind: str
    key: tuple[int, ...] = ()


@dataclass
class PieceInfo:
    color: int
    mask: np.ndarray
    base_len: int
    lead: EdgeProfile
    trail: EdgeProfile
    lead_tab: int


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _find_pieces(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [r0 for r0, _ in cells]
            cols = [c0 for _, c0 in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                mask[rr - r0, cc - c0] = True
            pieces.append((color, mask))

    return pieces


def _classify_edge(gaps: np.ndarray) -> EdgeProfile:
    gaps = np.asarray(gaps, dtype=int)
    assert set(gaps.tolist()) <= {0, 1}, "edge depth must be 0/1"

    ones = int(gaps.sum())
    length = int(gaps.size)
    if ones == 0:
        return EdgeProfile("flat", ())
    if 0 < ones < length / 2:
        return EdgeProfile("notch", tuple(int(i) for i in np.where(gaps == 1)[0]))
    if ones > length / 2:
        return EdgeProfile("tab", tuple(int(i) for i in np.where(gaps == 0)[0]))
    raise AssertionError("ambiguous edge profile")


def _analyze_horizontal_piece(mask: np.ndarray, color: int) -> PieceInfo:
    h, w = mask.shape
    first = np.empty(h, dtype=int)
    last = np.empty(h, dtype=int)
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        assert cols.size > 0, "every horizontal piece row must be occupied"
        assert np.all(mask[r, cols[0] : cols[-1] + 1]), "piece rows must stay contiguous"
        first[r] = int(cols[0])
        last[r] = int(cols[-1])

    left = _classify_edge(first)
    right = _classify_edge((w - 1) - last)
    base_len = w - int(left.kind == "tab") - int(right.kind == "tab")
    assert base_len >= 2
    return PieceInfo(
        color=color,
        mask=mask,
        base_len=base_len,
        lead=left,
        trail=right,
        lead_tab=int(left.kind == "tab"),
    )


def _analyze_vertical_piece(mask: np.ndarray, color: int) -> PieceInfo:
    h, w = mask.shape
    first = np.empty(w, dtype=int)
    last = np.empty(w, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        assert rows.size > 0, "every vertical piece column must be occupied"
        assert np.all(mask[rows[0] : rows[-1] + 1, c]), "piece columns must stay contiguous"
        first[c] = int(rows[0])
        last[c] = int(rows[-1])

    top = _classify_edge(first)
    bottom = _classify_edge((h - 1) - last)
    base_len = h - int(top.kind == "tab") - int(bottom.kind == "tab")
    assert base_len >= 2
    return PieceInfo(
        color=color,
        mask=mask,
        base_len=base_len,
        lead=top,
        trail=bottom,
        lead_tab=int(top.kind == "tab"),
    )


def _profiles_match(a: EdgeProfile, b: EdgeProfile) -> bool:
    if a.key != b.key:
        return False
    return (a.kind == "tab" and b.kind == "notch") or (a.kind == "notch" and b.kind == "tab")


def _reconstruct_order(pieces: list[PieceInfo]) -> list[int]:
    n = len(pieces)
    outgoing = [[] for _ in range(n)]
    incoming = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _profiles_match(pieces[i].trail, pieces[j].lead):
                outgoing[i].append(j)
                incoming[j].append(i)

    starts = [i for i, piece in enumerate(pieces) if piece.lead.kind == "flat"]
    ends = [i for i, piece in enumerate(pieces) if piece.trail.kind == "flat"]
    assert len(starts) == 1 and len(ends) == 1, "assembly chain must have unique ends"
    start = starts[0]
    end = ends[0]

    for i in range(n):
        expected_in = 0 if i == start else 1
        expected_out = 0 if i == end else 1
        assert len(incoming[i]) == expected_in, "each piece needs a unique predecessor"
        assert len(outgoing[i]) == expected_out, "each piece needs a unique successor"

    order: list[int] = []
    cur = start
    while True:
        assert cur not in order, "piece chain must be acyclic"
        order.append(cur)
        if cur == end:
            break
        cur = outgoing[cur][0]

    assert len(order) == n
    return order


def _assemble_horizontal(pieces: list[PieceInfo], order: list[int], bg: int) -> np.ndarray:
    out_h = pieces[0].mask.shape[0]
    out_w = sum(pieces[i].base_len for i in order)
    out = np.full((out_h, out_w), bg, dtype=int)
    filled = np.zeros((out_h, out_w), dtype=bool)

    cursor = 0
    for idx in order:
        piece = pieces[idx]
        c0 = cursor - piece.lead_tab
        for r, c in zip(*np.where(piece.mask)):
            gc = c0 + int(c)
            assert 0 <= gc < out_w
            assert not filled[int(r), gc], "interlocked pieces must not overlap"
            filled[int(r), gc] = True
            out[int(r), gc] = piece.color
        cursor += piece.base_len

    assert np.all(filled), "assembled result must be a full rectangle"
    return out


def _assemble_vertical(pieces: list[PieceInfo], order: list[int], bg: int) -> np.ndarray:
    out_h = sum(pieces[i].base_len for i in order)
    out_w = pieces[0].mask.shape[1]
    out = np.full((out_h, out_w), bg, dtype=int)
    filled = np.zeros((out_h, out_w), dtype=bool)

    cursor = 0
    for idx in order:
        piece = pieces[idx]
        r0 = cursor - piece.lead_tab
        for r, c in zip(*np.where(piece.mask)):
            gr = r0 + int(r)
            assert 0 <= gr < out_h
            assert not filled[gr, int(c)], "interlocked pieces must not overlap"
            filled[gr, int(c)] = True
            out[gr, int(c)] = piece.color
        cursor += piece.base_len

    assert np.all(filled), "assembled result must be a full rectangle"
    return out


def _solve_horizontal(raw_pieces: list[tuple[int, np.ndarray]], bg: int) -> np.ndarray:
    heights = {mask.shape[0] for _, mask in raw_pieces}
    assert len(heights) == 1, "horizontal assembly needs a common height"
    pieces = [_analyze_horizontal_piece(mask, color) for color, mask in raw_pieces]
    order = _reconstruct_order(pieces)
    return _assemble_horizontal(pieces, order, bg)


def _solve_vertical(raw_pieces: list[tuple[int, np.ndarray]], bg: int) -> np.ndarray:
    widths = {mask.shape[1] for _, mask in raw_pieces}
    assert len(widths) == 1, "vertical assembly needs a common width"
    pieces = [_analyze_vertical_piece(mask, color) for color, mask in raw_pieces]
    order = _reconstruct_order(pieces)
    return _assemble_vertical(pieces, order, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered tab-and-notch fragments into their compact rectangle.

    Input structure:
    - The background is black.
    - Several disconnected monochrome fragments are scattered across the grid, each in a unique color.
    - Every fragment is almost rectangular, except that one or two sides may contain 1-cell tabs or 1-cell notches.
    - Complementary tabs and notches encode which fragment must attach to which neighbor.

    Rule:
    - Isolate the colored fragments and inspect their edge profiles.
    - Match a fragment's tabbed side to the unique fragment whose opposite side has the same notch pattern, or vice versa.
    - Follow those matches to reconstruct the horizontal or vertical chain of pieces.
    - Reassemble the pieces tightly so the interlocking edges cancel out, producing the completed rectangle while preserving every fragment color.

    Key insight:
    - The original positions are irrelevant; only the complementary edge silhouettes determine the ordering and exact placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    raw_pieces = _find_pieces(g, bg)
    assert len(raw_pieces) >= 2

    candidates: list[np.ndarray] = []
    try:
        candidates.append(_solve_horizontal(raw_pieces, bg))
    except Exception:
        pass
    try:
        candidates.append(_solve_vertical(raw_pieces, bg))
    except Exception:
        pass

    assert candidates, "no valid assembly orientation found"
    if len(candidates) == 1:
        return candidates[0]
    if all(np.array_equal(candidates[0], other) for other in candidates[1:]):
        return candidates[0]
    raise AssertionError("assembly orientation is ambiguous")


# --------------- generator ---------------

ARC_MAX = 30
GEN_RETRIES = 400
PIECE_COUNT_RANGE = (3, 5)
CROSS_RANGE = (5, 8)
BASE_LEN_RANGE = (3, 6)
TOTAL_LEN_RANGE = (12, 24)
CANVAS_MIN = 16
PLACEMENT_RETRIES = 200
CANVAS_RETRIES = 80
COLOR_POOL = tuple(range(1, 10))


def _pattern_options(length: int) -> list[tuple[int, ...]]:
    singles = [(i,) for i in range(1, length - 1)]
    doubles = [tuple(range(i, i + 2)) for i in range(1, length - 2)]
    return singles + doubles


def _sample_base_lengths(n: int) -> list[int] | None:
    for _ in range(200):
        lengths = [random.randint(*BASE_LEN_RANGE) for _ in range(n)]
        if TOTAL_LEN_RANGE[0] <= sum(lengths) <= TOTAL_LEN_RANGE[1] and len(set(lengths)) > 1:
            return lengths
    return None


def _build_horizontal_mask(height: int, base_width: int, left: EdgeProfile, right: EdgeProfile) -> np.ndarray:
    left_pad = int(left.kind == "tab")
    right_pad = int(right.kind == "tab")
    mask = np.zeros((height, base_width + left_pad + right_pad), dtype=bool)
    base_c0 = left_pad
    mask[:, base_c0 : base_c0 + base_width] = True

    if left.kind == "tab":
        mask[list(left.key), 0] = True
    elif left.kind == "notch":
        mask[list(left.key), base_c0] = False

    if right.kind == "tab":
        mask[list(right.key), base_c0 + base_width] = True
    elif right.kind == "notch":
        mask[list(right.key), base_c0 + base_width - 1] = False

    return mask


def _build_vertical_mask(base_height: int, width: int, top: EdgeProfile, bottom: EdgeProfile) -> np.ndarray:
    top_pad = int(top.kind == "tab")
    bottom_pad = int(bottom.kind == "tab")
    mask = np.zeros((base_height + top_pad + bottom_pad, width), dtype=bool)
    base_r0 = top_pad
    mask[base_r0 : base_r0 + base_height, :] = True

    if top.kind == "tab":
        mask[0, list(top.key)] = True
    elif top.kind == "notch":
        mask[base_r0, list(top.key)] = False

    if bottom.kind == "tab":
        mask[base_r0 + base_height, list(bottom.key)] = True
    elif bottom.kind == "notch":
        mask[base_r0 + base_height - 1, list(bottom.key)] = False

    return mask


def _sample_horizontal_instance() -> tuple[list[tuple[int, np.ndarray]], np.ndarray] | None:
    n = random.randint(*PIECE_COUNT_RANGE)
    height = random.randint(*CROSS_RANGE)
    patterns = _pattern_options(height)
    if len(patterns) < n - 1:
        return None

    base_widths = _sample_base_lengths(n)
    if base_widths is None:
        return None

    random.shuffle(patterns)
    patterns = patterns[: n - 1]
    colors = random.sample(COLOR_POOL, n)

    left_edges: list[EdgeProfile | None] = [None] * n
    right_edges: list[EdgeProfile | None] = [None] * n
    left_edges[0] = EdgeProfile("flat", ())
    right_edges[-1] = EdgeProfile("flat", ())

    for i, key in enumerate(patterns):
        if random.random() < 0.5:
            right_edges[i] = EdgeProfile("tab", key)
            left_edges[i + 1] = EdgeProfile("notch", key)
        else:
            right_edges[i] = EdgeProfile("notch", key)
            left_edges[i + 1] = EdgeProfile("tab", key)

    pieces = [
        (color, _build_horizontal_mask(height, base_widths[i], left_edges[i], right_edges[i]))
        for i, color in enumerate(colors)
    ]
    if len({mask.shape[1] for _, mask in pieces}) == 1:
        return None

    infos = [_analyze_horizontal_piece(mask, color) for color, mask in pieces]
    expected = _assemble_horizontal(infos, list(range(n)), 0)
    return pieces, expected


def _sample_vertical_instance() -> tuple[list[tuple[int, np.ndarray]], np.ndarray] | None:
    n = random.randint(*PIECE_COUNT_RANGE)
    width = random.randint(*CROSS_RANGE)
    patterns = _pattern_options(width)
    if len(patterns) < n - 1:
        return None

    base_heights = _sample_base_lengths(n)
    if base_heights is None:
        return None

    random.shuffle(patterns)
    patterns = patterns[: n - 1]
    colors = random.sample(COLOR_POOL, n)

    top_edges: list[EdgeProfile | None] = [None] * n
    bottom_edges: list[EdgeProfile | None] = [None] * n
    top_edges[0] = EdgeProfile("flat", ())
    bottom_edges[-1] = EdgeProfile("flat", ())

    for i, key in enumerate(patterns):
        if random.random() < 0.5:
            bottom_edges[i] = EdgeProfile("tab", key)
            top_edges[i + 1] = EdgeProfile("notch", key)
        else:
            bottom_edges[i] = EdgeProfile("notch", key)
            top_edges[i + 1] = EdgeProfile("tab", key)

    pieces = [
        (color, _build_vertical_mask(base_heights[i], width, top_edges[i], bottom_edges[i]))
        for i, color in enumerate(colors)
    ]
    if len({mask.shape[0] for _, mask in pieces}) == 1:
        return None

    infos = [_analyze_vertical_piece(mask, color) for color, mask in pieces]
    expected = _assemble_vertical(infos, list(range(n)), 0)
    return pieces, expected


def _scatter_pieces(pieces: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    max_h = max(mask.shape[0] for _, mask in pieces)
    max_w = max(mask.shape[1] for _, mask in pieces)

    for _ in range(CANVAS_RETRIES):
        grid_h = random.randint(max(CANVAS_MIN, max_h + 2), ARC_MAX)
        grid_w = random.randint(max(CANVAS_MIN, max_w + 2), ARC_MAX)
        if sum(mask.size for _, mask in pieces) > (grid_h * grid_w) // 2:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)
        order = sorted(range(len(pieces)), key=lambda i: pieces[i][1].size, reverse=True)
        success = True

        for idx in order:
            color, mask = pieces[idx]
            ph, pw = mask.shape
            placed = False

            for _ in range(PLACEMENT_RETRIES):
                r0 = random.randint(0, grid_h - ph)
                c0 = random.randint(0, grid_w - pw)
                rr0 = max(0, r0 - 1)
                rr1 = min(grid_h, r0 + ph + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(grid_w, c0 + pw + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue

                grid[r0 : r0 + ph, c0 : c0 + pw][mask] = color
                blocked[rr0:rr1, cc0:cc1] = True
                placed = True
                break

            if not placed:
                success = False
                break

        if success:
            return grid

    return None


def _oracle_matches(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    if random.random() < 0.5:
        sampled = _sample_horizontal_instance()
    else:
        sampled = _sample_vertical_instance()
    if sampled is None:
        return None

    pieces, expected = sampled
    inp = _scatter_pieces(pieces)
    if inp is None:
        return None
    if not _oracle_matches(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid puzzle after {GEN_RETRIES} attempts")
