from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
PIXEL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("up", "down")
MAX_RETRIES = 300


@dataclass(frozen=True)
class Frame:
    bbox: tuple[int, int, int, int]
    gap: tuple[int, int]
    gap_side: str
    source: tuple[int, int] | None
    source_color: int | None


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _gray_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != GREY:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _gap_side(r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]) -> str:
    gr, gc = gap
    if gr == r0:
        return "up"
    if gr == r1:
        return "down"
    if gc == c0:
        return "left"
    if gc == c1:
        return "right"
    raise AssertionError("gap must lie on the frame border")


def _extract_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    comps = _gray_components(g)
    assert comps, "expected at least one gray frame"

    frames: list[Frame] = []
    colored_inside: set[tuple[int, int]] = set()
    for comp in comps:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame bbox must have a strict interior"

        border = _border_cells(r0, c0, r1, c1)
        gap_cells = [(r, c) for r, c in border if int(g[r, c]) != GREY]
        assert len(gap_cells) == 1, "each frame must have exactly one border gap"
        gap = gap_cells[0]
        assert int(g[gap]) == BLACK, "frame gap must be empty"
        assert gap[0] not in (r0, r1) or gap[1] not in (c0, c1), "gap cannot be at a corner"
        assert len(comp) == len(border) - 1, "gray component must be a one-gap rectangular border"

        gap_side = _gap_side(r0, c0, r1, c1, gap)
        sources = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                val = int(g[r, c])
                assert val != GREY, "frame interior cannot contain gray pixels"
                if val != BLACK:
                    sources.append((r, c, val))
        assert len(sources) <= 1, "each frame may contain at most one colored source pixel"

        source = None
        source_color = None
        if sources:
            sr, sc, source_color = sources[0]
            if gap_side in ("up", "down"):
                assert sc == gap[1], "vertical gap requires the source pixel in the same column"
            else:
                assert sr == gap[0], "horizontal gap requires the source pixel in the same row"
            source = (sr, sc)
            colored_inside.add(source)

        frames.append(Frame((r0, c0, r1, c1), gap, gap_side, source, source_color))

    colored = {
        (r, c)
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if int(g[r, c]) not in (BLACK, GREY)
    }
    assert colored == colored_inside, "all colored pixels must be sources inside frames"
    return frames


def _entry_dir(frame: Frame) -> tuple[int, int]:
    dr, dc = DIRS[frame.gap_side]
    return -dr, -dc


def _entry_stop(frame: Frame) -> tuple[int, int]:
    r0, c0, r1, c1 = frame.bbox
    gr, gc = frame.gap
    if frame.gap_side == "up":
        return r1 - 1, gc
    if frame.gap_side == "down":
        return r0 + 1, gc
    if frame.gap_side == "left":
        return gr, c1 - 1
    return gr, c0 + 1


def _trace_move(grid: np.ndarray, frame: Frame, gap_lookup: dict[tuple[int, int], Frame]) -> tuple[tuple[int, int], str]:
    dr, dc = DIRS[frame.gap_side]
    prev = frame.gap
    r, c = frame.gap[0] + dr, frame.gap[1] + dc
    while True:
        if not _in_bounds(grid.shape, r, c):
            return prev, "edge"
        other = gap_lookup.get((r, c))
        if other is not None and other is not frame and _entry_dir(other) == (dr, dc):
            return _entry_stop(other), "enter"
        if int(grid[r, c]) == GREY:
            return prev, "wall"
        prev = (r, c)
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each interior pixel straight out through its own frame gap until it reaches a wall, another frame's gap, or the canvas edge.

    Input structure:
    - The grid has a black background and several gray hollow rectangular frames.
    - Each frame border has exactly one non-corner single-cell gap, and some frames contain one colored source pixel.
    - Every source pixel is aligned with its own frame gap in the same row or column.

    Rule:
    - For each source pixel, follow the direction from the source through its own frame gap and continue in a straight line outside the frame.
    - If the ray first meets another frame at that frame's gap from the outside, place the pixel just inside that frame against the opposite inner wall.
    - If the ray first meets a gray border wall, place the pixel on the adjacent outside cell before the wall; if nothing blocks it, place the pixel on the canvas edge.

    Key insight:
    - The frame gap serves both as the source frame's launch direction and as the only traversable doorway of a target frame, so the correct destination comes from stepwise wall-vs-door tracing along the ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    frames = _extract_frames(g)
    gap_lookup = {frame.gap: frame for frame in frames}

    out = g.copy()
    sources = sorted((frame for frame in frames if frame.source is not None), key=lambda f: f.source)
    for frame in sources:
        sr, sc = frame.source
        out[sr, sc] = BLACK

    for frame in sources:
        dest, _event = _trace_move(g, frame, gap_lookup)
        assert int(out[dest]) == BLACK, "pixel paths must not collide"
        out[dest] = int(frame.source_color)

    return out


def _gap_cell(spec: dict) -> tuple[int, int]:
    r0, c0, h, w, side, offset = (
        spec["r0"],
        spec["c0"],
        spec["h"],
        spec["w"],
        spec["gap_side"],
        spec["gap_offset"],
    )
    if side == "up":
        return r0, c0 + offset
    if side == "down":
        return r0 + h - 1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    return r0 + offset, c0 + w - 1


def _render_frame(grid: np.ndarray, spec: dict) -> None:
    r0, c0, h, w = spec["r0"], spec["c0"], spec["h"], spec["w"]
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    gr, gc = _gap_cell(spec)
    grid[gr, gc] = BLACK
    if spec["source"] is not None:
        sr, sc = spec["source"]
        grid[sr, sc] = spec["source_color"]


def _fits_with_padding(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    *,
    pad: int = 1,
) -> bool:
    h_grid, w_grid = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h > h_grid - 1 or c0 + w > w_grid - 1:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h_grid, r0 + h + pad)
    cc1 = min(w_grid, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, *, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _source_cell(r0: int, c0: int, h: int, w: int, side: str, offset: int, rng: random.Random) -> tuple[int, int]:
    if side in ("up", "down"):
        return r0 + rng.randint(1, h - 2), c0 + offset
    return r0 + offset, c0 + rng.randint(1, w - 2)


def _make_frame_spec(
    r0: int,
    c0: int,
    h: int,
    w: int,
    gap_side: str,
    gap_offset: int,
    source_color: int | None,
    rng: random.Random,
) -> dict:
    source = None
    if source_color is not None:
        source = _source_cell(r0, c0, h, w, gap_side, gap_offset, rng)
    return {
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "gap_side": gap_side,
        "gap_offset": gap_offset,
        "source": source,
        "source_color": source_color,
    }


def _horizontal_entry(
    rng: random.Random,
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_rows: set[int],
    source_color: int,
) -> tuple[int, list[dict]] | None:
    for _ in range(200):
        lane = rng.randint(2, H - 3)
        if lane in reserved_rows:
            continue
        h1, w1 = rng.randint(5, 7), rng.randint(5, 8)
        h2, w2 = rng.randint(5, 7), rng.randint(5, 8)
        off1 = rng.randint(1, h1 - 2)
        off2 = rng.randint(1, h2 - 2)
        sep = rng.randint(2, 6)
        direction = rng.choice(HORIZONTAL_SIDES)
        if direction == "right":
            r1 = lane - off1
            r2 = lane - off2
            max_c1 = W - 1 - (w1 + sep + w2)
            if r1 < 1 or r2 < 1 or r1 + h1 > H - 1 or r2 + h2 > H - 1 or max_c1 < 1:
                continue
            c1 = rng.randint(1, max_c1)
            c2 = c1 + w1 + sep
            spec_a = _make_frame_spec(r1, c1, h1, w1, "right", off1, source_color, rng)
            spec_b = _make_frame_spec(r2, c2, h2, w2, "left", off2, None, rng)
        else:
            r1 = lane - off1
            r2 = lane - off2
            max_c2 = W - 1 - (w2 + sep + w1)
            if r1 < 1 or r2 < 1 or r1 + h1 > H - 1 or r2 + h2 > H - 1 or max_c2 < 1:
                continue
            c2 = rng.randint(1, max_c2)
            c1 = c2 + w2 + sep
            spec_a = _make_frame_spec(r1, c1, h1, w1, "left", off1, source_color, rng)
            spec_b = _make_frame_spec(r2, c2, h2, w2, "right", off2, None, rng)

        if not _fits_with_padding(occupied, spec_a["r0"], spec_a["c0"], h1, w1):
            continue
        if not _fits_with_padding(occupied, spec_b["r0"], spec_b["c0"], h2, w2):
            continue
        _reserve_bbox(occupied, spec_a["r0"], spec_a["c0"], h1, w1)
        _reserve_bbox(occupied, spec_b["r0"], spec_b["c0"], h2, w2)
        reserved_rows.add(lane)
        return lane, [spec_a, spec_b]
    return None


def _horizontal_wall(
    rng: random.Random,
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_rows: set[int],
    source_color: int,
) -> tuple[int, list[dict]] | None:
    blocker_gap_choices = {
        "right": ("up", "down", "right"),
        "left": ("up", "down", "left"),
    }
    for _ in range(200):
        lane = rng.randint(2, H - 3)
        if lane in reserved_rows:
            continue
        h1, w1 = rng.randint(5, 7), rng.randint(5, 8)
        h2, w2 = rng.randint(5, 7), rng.randint(5, 8)
        off1 = rng.randint(1, h1 - 2)
        wall_off = rng.randint(1, h2 - 2)
        sep = rng.randint(2, 6)
        direction = rng.choice(HORIZONTAL_SIDES)
        blocker_side = rng.choice(blocker_gap_choices[direction])
        gap_off2 = rng.randint(1, (w2 - 2) if blocker_side in ("up", "down") else (h2 - 2))
        if direction == "right":
            r1 = lane - off1
            r2 = lane - wall_off
            max_c1 = W - 1 - (w1 + sep + w2)
            if r1 < 1 or r2 < 1 or r1 + h1 > H - 1 or r2 + h2 > H - 1 or max_c1 < 1:
                continue
            c1 = rng.randint(1, max_c1)
            c2 = c1 + w1 + sep
            spec_a = _make_frame_spec(r1, c1, h1, w1, "right", off1, source_color, rng)
            spec_b = _make_frame_spec(r2, c2, h2, w2, blocker_side, gap_off2, None, rng)
        else:
            r1 = lane - off1
            r2 = lane - wall_off
            max_c2 = W - 1 - (w2 + sep + w1)
            if r1 < 1 or r2 < 1 or r1 + h1 > H - 1 or r2 + h2 > H - 1 or max_c2 < 1:
                continue
            c2 = rng.randint(1, max_c2)
            c1 = c2 + w2 + sep
            spec_a = _make_frame_spec(r1, c1, h1, w1, "left", off1, source_color, rng)
            spec_b = _make_frame_spec(r2, c2, h2, w2, blocker_side, gap_off2, None, rng)

        if not _fits_with_padding(occupied, spec_a["r0"], spec_a["c0"], h1, w1):
            continue
        if not _fits_with_padding(occupied, spec_b["r0"], spec_b["c0"], h2, w2):
            continue
        _reserve_bbox(occupied, spec_a["r0"], spec_a["c0"], h1, w1)
        _reserve_bbox(occupied, spec_b["r0"], spec_b["c0"], h2, w2)
        reserved_rows.add(lane)
        return lane, [spec_a, spec_b]
    return None


def _horizontal_edge(
    rng: random.Random,
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_rows: set[int],
    source_color: int,
) -> tuple[int, list[dict]] | None:
    for _ in range(200):
        lane = rng.randint(2, H - 3)
        if lane in reserved_rows:
            continue
        h, w = rng.randint(5, 7), rng.randint(5, 8)
        off = rng.randint(1, h - 2)
        direction = rng.choice(HORIZONTAL_SIDES)
        r0 = lane - off
        if r0 < 1 or r0 + h > H - 1:
            continue
        if direction == "right":
            max_c0 = W - w - 3
            if max_c0 < 1:
                continue
            c0 = rng.randint(1, max_c0)
        else:
            min_c0 = 3
            max_c0 = W - w - 1
            if min_c0 > max_c0:
                continue
            c0 = rng.randint(min_c0, max_c0)
        spec = _make_frame_spec(r0, c0, h, w, direction, off, source_color, rng)
        if not _fits_with_padding(occupied, r0, c0, h, w):
            continue
        _reserve_bbox(occupied, r0, c0, h, w)
        reserved_rows.add(lane)
        return lane, [spec]
    return None


def _vertical_entry(
    rng: random.Random,
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_cols: set[int],
    source_color: int,
) -> tuple[int, list[dict]] | None:
    for _ in range(200):
        lane = rng.randint(2, W - 3)
        if lane in reserved_cols:
            continue
        h1, w1 = rng.randint(5, 8), rng.randint(5, 7)
        h2, w2 = rng.randint(5, 8), rng.randint(5, 7)
        off1 = rng.randint(1, w1 - 2)
        off2 = rng.randint(1, w2 - 2)
        sep = rng.randint(2, 6)
        direction = rng.choice(VERTICAL_SIDES)
        if direction == "down":
            c1 = lane - off1
            c2 = lane - off2
            max_r1 = H - 1 - (h1 + sep + h2)
            if c1 < 1 or c2 < 1 or c1 + w1 > W - 1 or c2 + w2 > W - 1 or max_r1 < 1:
                continue
            r1 = rng.randint(1, max_r1)
            r2 = r1 + h1 + sep
            spec_a = _make_frame_spec(r1, c1, h1, w1, "down", off1, source_color, rng)
            spec_b = _make_frame_spec(r2, c2, h2, w2, "up", off2, None, rng)
        else:
            c1 = lane - off1
            c2 = lane - off2
            max_r2 = H - 1 - (h2 + sep + h1)
            if c1 < 1 or c2 < 1 or c1 + w1 > W - 1 or c2 + w2 > W - 1 or max_r2 < 1:
                continue
            r2 = rng.randint(1, max_r2)
            r1 = r2 + h2 + sep
            spec_a = _make_frame_spec(r1, c1, h1, w1, "up", off1, source_color, rng)
            spec_b = _make_frame_spec(r2, c2, h2, w2, "down", off2, None, rng)

        if not _fits_with_padding(occupied, spec_a["r0"], spec_a["c0"], h1, w1):
            continue
        if not _fits_with_padding(occupied, spec_b["r0"], spec_b["c0"], h2, w2):
            continue
        _reserve_bbox(occupied, spec_a["r0"], spec_a["c0"], h1, w1)
        _reserve_bbox(occupied, spec_b["r0"], spec_b["c0"], h2, w2)
        reserved_cols.add(lane)
        return lane, [spec_a, spec_b]
    return None


def _vertical_wall(
    rng: random.Random,
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_cols: set[int],
    source_color: int,
) -> tuple[int, list[dict]] | None:
    blocker_gap_choices = {
        "down": ("left", "right", "down"),
        "up": ("left", "right", "up"),
    }
    for _ in range(200):
        lane = rng.randint(2, W - 3)
        if lane in reserved_cols:
            continue
        h1, w1 = rng.randint(5, 8), rng.randint(5, 7)
        h2, w2 = rng.randint(5, 8), rng.randint(5, 7)
        off1 = rng.randint(1, w1 - 2)
        wall_off = rng.randint(1, w2 - 2)
        sep = rng.randint(2, 6)
        direction = rng.choice(VERTICAL_SIDES)
        blocker_side = rng.choice(blocker_gap_choices[direction])
        gap_off2 = rng.randint(1, (h2 - 2) if blocker_side in ("left", "right") else (w2 - 2))
        if direction == "down":
            c1 = lane - off1
            c2 = lane - wall_off
            max_r1 = H - 1 - (h1 + sep + h2)
            if c1 < 1 or c2 < 1 or c1 + w1 > W - 1 or c2 + w2 > W - 1 or max_r1 < 1:
                continue
            r1 = rng.randint(1, max_r1)
            r2 = r1 + h1 + sep
            spec_a = _make_frame_spec(r1, c1, h1, w1, "down", off1, source_color, rng)
            spec_b = _make_frame_spec(r2, c2, h2, w2, blocker_side, gap_off2, None, rng)
        else:
            c1 = lane - off1
            c2 = lane - wall_off
            max_r2 = H - 1 - (h2 + sep + h1)
            if c1 < 1 or c2 < 1 or c1 + w1 > W - 1 or c2 + w2 > W - 1 or max_r2 < 1:
                continue
            r2 = rng.randint(1, max_r2)
            r1 = r2 + h2 + sep
            spec_a = _make_frame_spec(r1, c1, h1, w1, "up", off1, source_color, rng)
            spec_b = _make_frame_spec(r2, c2, h2, w2, blocker_side, gap_off2, None, rng)

        if not _fits_with_padding(occupied, spec_a["r0"], spec_a["c0"], h1, w1):
            continue
        if not _fits_with_padding(occupied, spec_b["r0"], spec_b["c0"], h2, w2):
            continue
        _reserve_bbox(occupied, spec_a["r0"], spec_a["c0"], h1, w1)
        _reserve_bbox(occupied, spec_b["r0"], spec_b["c0"], h2, w2)
        reserved_cols.add(lane)
        return lane, [spec_a, spec_b]
    return None


def _vertical_edge(
    rng: random.Random,
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_cols: set[int],
    source_color: int,
) -> tuple[int, list[dict]] | None:
    for _ in range(200):
        lane = rng.randint(2, W - 3)
        if lane in reserved_cols:
            continue
        h, w = rng.randint(5, 8), rng.randint(5, 7)
        off = rng.randint(1, w - 2)
        direction = rng.choice(VERTICAL_SIDES)
        c0 = lane - off
        if c0 < 1 or c0 + w > W - 1:
            continue
        if direction == "down":
            max_r0 = H - h - 3
            if max_r0 < 1:
                continue
            r0 = rng.randint(1, max_r0)
        else:
            min_r0 = 3
            max_r0 = H - h - 1
            if min_r0 > max_r0:
                continue
            r0 = rng.randint(min_r0, max_r0)
        spec = _make_frame_spec(r0, c0, h, w, direction, off, source_color, rng)
        if not _fits_with_padding(occupied, r0, c0, h, w):
            continue
        _reserve_bbox(occupied, r0, c0, h, w)
        reserved_cols.add(lane)
        return lane, [spec]
    return None


def _passive_frame(
    rng: random.Random,
    H: int,
    W: int,
    occupied: np.ndarray,
    reserved_rows: set[int],
    reserved_cols: set[int],
) -> dict | None:
    for _ in range(200):
        h = rng.randint(4, 7)
        w = rng.randint(4, 7)
        r0 = rng.randint(1, H - h - 1)
        c0 = rng.randint(1, W - w - 1)
        r1, c1 = r0 + h - 1, c0 + w - 1
        if any(r0 <= row <= r1 for row in reserved_rows):
            continue
        if any(c0 <= col <= c1 for col in reserved_cols):
            continue
        if not _fits_with_padding(occupied, r0, c0, h, w):
            continue
        side = rng.choice(tuple(DIRS))
        gap_off = rng.randint(1, (w - 2) if side in ("up", "down") else (h - 2))
        spec = _make_frame_spec(r0, c0, h, w, side, gap_off, None, rng)
        _reserve_bbox(occupied, r0, c0, h, w)
        return spec
    return None


def _build_grid(rng: random.Random) -> np.ndarray | None:
    axis = rng.choice(("horizontal", "vertical"))
    H = rng.randint(19, 29)
    W = rng.randint(19, 29)
    occupied = np.zeros((H, W), dtype=bool)
    reserved_rows: set[int] = set()
    reserved_cols: set[int] = set()
    specs: list[dict] = []

    colors = rng.sample(PIXEL_COLORS, 3)
    builders = {
        "horizontal": {
            "enter": _horizontal_entry,
            "wall": _horizontal_wall,
            "edge": _horizontal_edge,
        },
        "vertical": {
            "enter": _vertical_entry,
            "wall": _vertical_wall,
            "edge": _vertical_edge,
        },
    }[axis]

    for branch, color in zip(rng.sample(["enter", "wall", "edge"], 3), colors):
        if axis == "horizontal":
            built = builders[branch](rng, H, W, occupied, reserved_rows, color)
        else:
            built = builders[branch](rng, H, W, occupied, reserved_cols, color)
        if built is None:
            return None
        _lane, branch_specs = built
        specs.extend(branch_specs)

    for _ in range(rng.randint(0, 3)):
        passive = _passive_frame(rng, H, W, occupied, reserved_rows, reserved_cols)
        if passive is None:
            break
        specs.append(passive)

    grid = np.full((H, W), BLACK, dtype=int)
    for spec in specs:
        _render_frame(grid, spec)

    try:
        frames = _extract_frames(grid)
        gap_lookup = {frame.gap: frame for frame in frames}
        events = []
        for frame in frames:
            if frame.source is not None:
                _dest, event = _trace_move(grid, frame, gap_lookup)
                events.append(event)
        if sorted(events) != ["edge", "enter", "wall"]:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        rng = random.Random()
        grid = _build_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-delivery input")
