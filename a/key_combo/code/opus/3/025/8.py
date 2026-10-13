from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        utils_dir = parent / "new" / "solvers"
        if (utils_dir / "arc_utils.py").exists():
            utils_str = str(utils_dir)
            if utils_str not in sys.path:
                sys.path.insert(0, utils_str)
            return
    raise ImportError("could not locate new/solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import Color, colors_present, find_components, get_bg  # noqa: E402


BG = Color.BLACK
YELLOW = Color.YELLOW
FG_PALETTE = tuple(color for color in Color.NOT_BLACK if color != YELLOW)
GEN_RETRIES = 300

Coord = tuple[int, int]
Segment = list[Coord]


def _normalize_segment(coords: np.ndarray) -> Segment:
    pts = sorted((int(r), int(c)) for r, c in coords)
    assert pts, "segment must be non-empty"
    r0, c0 = pts[0]
    diagonal_id = r0 - c0
    for index, (r, c) in enumerate(pts):
        assert r - c == diagonal_id, "foreground component must stay on one down-right diagonal"
        assert (r, c) == (r0 + index, c0 + index), "foreground component must be a gapless diagonal chain"
    return pts


def _analyze_input(grid: np.ndarray) -> tuple[int, int, list[Segment]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert YELLOW not in fg_colors, "input must not contain yellow pixels"
    assert len(fg_colors) == 1, "expected exactly one non-background foreground color"
    fg = int(fg_colors[0])

    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[fg],
    )
    segments = [_normalize_segment(comp["coords"]) for comp in comps]
    assert segments, "expected at least one diagonal segment"
    segments.sort(key=lambda seg: (seg[0][0], seg[0][1], len(seg)))
    return bg, fg, segments


def _paint_ray(out: np.ndarray, blockers: np.ndarray, start: Coord, step: Coord) -> None:
    h, w = out.shape
    r, c = start
    dr, dc = step
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return
        if blockers[r, c]:
            return
        out[r, c] = YELLOW


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, _fg, segments = _analyze_input(g)
    out = g.copy()
    blockers = g != bg
    triggers: list[Coord] = []

    for seg in segments:
        for index, (r, c) in enumerate(seg, start=1):
            if index % 2 == 0:
                out[r, c] = YELLOW
                triggers.append((r, c))

    for trigger in triggers:
        _paint_ray(out, blockers, trigger, (1, 1))
        _paint_ray(out, blockers, trigger, (-1, -1))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every even-indexed pixel on each down-right diagonal segment, then extend yellow rays along that diagonal.

    Input structure:
    - A black background contains pixels of exactly one other color, and yellow does not appear in the input.
    - The foreground pixels form several disconnected maximal segments, each a continuous chain along row+1/col+1.
    - Different segments are separated so they do not 8-touch each other.

    Rule:
    - Identify each maximal down-right diagonal segment and index its cells from the upper-left endpoint as 1, 2, 3, ...
    - Recolor every even-indexed cell to yellow while leaving odd-indexed cells in the original foreground color.
    - From every recolored yellow cell, extend a yellow ray one step at a time in the down-right direction until the grid edge or the next non-background cell.
    - Also extend a yellow ray from that same trigger in the opposite up-left direction with the same stopping rule.
    - Original foreground cells keep priority over any ray cells, so only background cells can be newly painted.

    Key insight:
    - The segment order determines which cells become yellow triggers, and the segment geometry itself determines the ray direction without any separate indicator pixel.
    """
    return _solve(grid)


def _segment_piece(length: int, color: int) -> dict:
    patch = np.full((length, length), BG, dtype=int)
    coords = [(k, k) for k in range(length)]
    for r, c in coords:
        patch[r, c] = color
    return {"patch": patch, "segments": [coords]}


def _blocked_pair_piece(source_len: int, gap: int, target_len: int, color: int) -> dict:
    size = source_len + gap + target_len
    patch = np.full((size, size), BG, dtype=int)
    source = [(k, k) for k in range(source_len)]
    target = [(source_len + gap + k, source_len + gap + k) for k in range(target_len)]
    for r, c in source + target:
        patch[r, c] = color
    return {"patch": patch, "segments": [source, target], "gap": gap}


def _placement_is_clear(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    for pr, pc in np.argwhere(mask):
        r = top + int(pr)
        c = left + int(pc)
        if occupied[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)].any():
            return False
    return True


def _place_piece(
    grid: np.ndarray,
    occupied: np.ndarray,
    used_diagonals: set[int],
    piece: dict,
    *,
    tail_min: int = 0,
) -> list[Segment] | None:
    patch = piece["patch"]
    mask = patch != BG
    h, w = grid.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    if not positions:
        return None

    order = np.random.permutation(len(positions))
    for idx in order:
        top, left = positions[int(idx)]
        diagonal_values = {top + seg[0][0] - (left + seg[0][1]) for seg in piece["segments"]}
        if diagonal_values & used_diagonals:
            continue
        if tail_min:
            violates_tail = False
            for seg in piece["segments"]:
                end_r = top + seg[-1][0]
                end_c = left + seg[-1][1]
                tail = min(h - 1 - end_r, w - 1 - end_c)
                if tail < tail_min:
                    violates_tail = True
                    break
            if violates_tail:
                continue
        if not _placement_is_clear(occupied, mask, top, left):
            continue

        grid[top : top + ph, left : left + pw][mask] = patch[mask]
        for pr, pc in np.argwhere(mask):
            occupied[top + int(pr), left + int(pc)] = True
        used_diagonals.update(diagonal_values)
        return [[(top + r, left + c) for r, c in seg] for seg in piece["segments"]]
    return None


def _tail_cells(shape: tuple[int, int], seg: Segment) -> list[Coord]:
    h, w = shape
    r, c = seg[-1]
    cells: list[Coord] = []
    while True:
        r += 1
        c += 1
        if not (0 <= r < h and 0 <= c < w):
            return cells
        cells.append((r, c))


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    try:
        bg, fg, segments = _analyze_input(grid)
    except AssertionError:
        return False

    if bg != BG or fg != meta["color"] or len(segments) < 4:
        return False

    observed = {tuple(seg) for seg in segments}
    for key in ("clear_even", "clear_odd", "blocked_source", "blocked_target"):
        if tuple(meta[key]) not in observed:
            return False

    for seg in segments:
        if not 3 <= len(seg) <= 6:
            return False

    out = _solve(grid)
    added = (grid == BG) & (out == YELLOW)
    if not bool(np.any(added)):
        return False
    if np.any((grid != BG) & (out == BG)):
        return False

    clear_even_tail = _tail_cells(grid.shape, meta["clear_even"])
    if len(clear_even_tail) < 2 or any(grid[r, c] != BG for r, c in clear_even_tail):
        return False
    if any(out[r, c] != YELLOW for r, c in clear_even_tail):
        return False

    clear_odd_tail = _tail_cells(grid.shape, meta["clear_odd"])
    if len(clear_odd_tail) < 2 or any(grid[r, c] != BG for r, c in clear_odd_tail):
        return False
    if any(out[r, c] != BG for r, c in clear_odd_tail):
        return False

    blocked_source = meta["blocked_source"]
    blocked_target = meta["blocked_target"]
    gap_cells = []
    r, c = blocked_source[-1]
    while True:
        r += 1
        c += 1
        if (r, c) == blocked_target[0]:
            break
        gap_cells.append((r, c))
    if len(gap_cells) != meta["blocked_gap"]:
        return False
    if not gap_cells or any(grid[r, c] != BG for r, c in gap_cells):
        return False
    if any(out[r, c] != YELLOW for r, c in gap_cells):
        return False
    target_start = blocked_target[0]
    if int(out[target_start]) != meta["color"]:
        return False

    even_coords = {coord for seg in segments for idx, coord in enumerate(seg, start=1) if idx % 2 == 0}
    if not even_coords:
        return False
    if any(int(out[r, c]) != YELLOW for r, c in even_coords):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(17, 23))
    w = int(np.random.randint(17, 23))
    color = int(np.random.choice(FG_PALETTE))
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    used_diagonals: set[int] = set()

    clear_even_len = int(np.random.choice([4, 6]))
    clear_odd_len = int(np.random.choice([3, 5]))
    blocked_source_len = int(np.random.choice([4, 6]))
    blocked_target_len = int(np.random.choice([3, 4, 5, 6]))
    blocked_gap = int(np.random.randint(1, 4))

    blocked_piece = _blocked_pair_piece(blocked_source_len, blocked_gap, blocked_target_len, color)
    clear_even_piece = _segment_piece(clear_even_len, color)
    clear_odd_piece = _segment_piece(clear_odd_len, color)

    blocked_segments = _place_piece(grid, occupied, used_diagonals, blocked_piece)
    if blocked_segments is None:
        return None
    clear_even_segments = _place_piece(grid, occupied, used_diagonals, clear_even_piece, tail_min=2)
    if clear_even_segments is None:
        return None
    clear_odd_segments = _place_piece(grid, occupied, used_diagonals, clear_odd_piece, tail_min=2)
    if clear_odd_segments is None:
        return None

    extra_count = int(np.random.randint(0, 3))
    for _ in range(extra_count):
        extra_len = int(np.random.choice([3, 4, 5, 6]))
        tail_min = 2 if np.random.random() < 0.4 else 0
        extra_piece = _segment_piece(extra_len, color)
        if _place_piece(grid, occupied, used_diagonals, extra_piece, tail_min=tail_min) is None:
            break

    meta = {
        "color": color,
        "clear_even": clear_even_segments[0],
        "clear_odd": clear_odd_segments[0],
        "blocked_source": blocked_segments[0],
        "blocked_target": blocked_segments[1],
        "blocked_gap": blocked_gap,
    }
    if not _validate_generated(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal recolor-ray input")
