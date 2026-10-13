from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)
DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: BBox
    gap: Coord
    cells: frozenset[Coord]
    inward: Coord


@dataclass(frozen=True)
class Hit:
    marker: Coord
    marker_color: int
    frame_index: int
    path: tuple[Coord, ...]  # marker-to-gap background cells, excluding the gap


def _component(grid: np.ndarray, start: Coord, seen: set[Coord]) -> set[Coord]:
    """Return one same-color 4-connected component."""
    h, w = grid.shape
    color = int(grid[start])
    stack = [start]
    seen.add(start)
    cells: set[Coord] = set()
    while stack:
        r, c = stack.pop()
        cells.add((r, c))
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            p = (rr, cc)
            if 0 <= rr < h and 0 <= cc < w and p not in seen and int(grid[p]) == color:
                seen.add(p)
                stack.append(p)
    return cells


def _border_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    return (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )


def _gap_inward(gap: Coord, box: BBox) -> Coord | None:
    r, c = gap
    r0, c0, r1, c1 = box
    if r == r0 and c0 < c < c1:
        return (1, 0)
    if r == r1 and c0 < c < c1:
        return (-1, 0)
    if c == c0 and r0 < r < r1:
        return (0, 1)
    if c == c1 and r0 < r < r1:
        return (0, -1)
    return None


def _find_frames(grid: np.ndarray) -> list[Frame]:
    """Find monochrome rectangular perimeters having exactly one non-corner gap."""
    seen: set[Coord] = set()
    frames: list[Frame] = []
    for rr, cc in np.argwhere(grid != 0):
        start = (int(rr), int(cc))
        if start in seen:
            continue
        cells = _component(grid, start, seen)
        if len(cells) < 9:
            continue
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        box = (min(rows), min(cols), max(rows), max(cols))
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        border = _border_cells(box)
        missing = border - cells
        if len(missing) != 1 or not cells <= border:
            continue
        gap = next(iter(missing))
        inward = _gap_inward(gap, box)
        if inward is None:
            continue
        frames.append(Frame(int(grid[start]), box, gap, frozenset(cells), inward))
    frames.sort(key=lambda f: f.bbox)
    return frames


def _markers(grid: np.ndarray, frames: list[Frame]) -> list[Coord]:
    frame_cells = set().union(*(f.cells for f in frames)) if frames else set()
    return [
        (int(r), int(c))
        for r, c in np.argwhere(grid != 0)
        if (int(r), int(c)) not in frame_cells
    ]


def _hits(grid: np.ndarray, frames: list[Frame], markers: list[Coord]) -> list[Hit]:
    """Trace all rays in the input grid; every nonblack input cell is opaque."""
    h, w = grid.shape
    gap_lookup = {f.gap: i for i, f in enumerate(frames)}
    hits: list[Hit] = []
    for marker in markers:
        marker_color = int(grid[marker])
        for dr, dc in DIRS:
            r, c = marker
            clear: list[Coord] = []
            while True:
                r += dr
                c += dc
                if not (0 <= r < h and 0 <= c < w):
                    break
                p = (r, c)
                if int(grid[p]) != 0:
                    break
                frame_index = gap_lookup.get(p)
                if frame_index is not None and frames[frame_index].inward == (dr, dc):
                    # The cell immediately beyond the gap must really be empty interior.
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == 0:
                        hits.append(Hit(marker, marker_color, frame_index, tuple(clear)))
                    break
                clear.append(p)
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """Send marker rays through frame gaps, filling the entered frames and sealing their gaps.

    Input structure:
    - A black background contains one or more separated, monochrome, one-cell-thick
      rectangular frames, each missing exactly one non-corner border cell.
    - Single colored marker pixels lie outside the frames; every nonblack input cell
      (including another marker or another frame) blocks a ray.

    Rule:
    - Trace four orthogonal rays from every marker on the original input. If an
      unobstructed ray approaches a frame from outside through its unique gap, fill
      that frame's entire strict interior with the marker color and color the clear
      path from the marker up to (but not over) the gap with the marker color.
    - Restore the gap with the frame's own color. Markers whose rays never enter a
      gap, including aligned markers blocked first by another frame, remain unchanged.

    Key insight:
    - A gap acts as a directional keyhole: marker-gap alignment is insufficient;
      the ray must reach that gap before encountering any nonblack blocker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30x30"
    frames = _find_frames(g)
    assert frames, "expected at least one one-gap rectangular frame"
    markers = _markers(g, frames)
    hits = _hits(g, frames, markers)
    assert len({hit.frame_index for hit in hits}) == len(hits), "a frame must have one visible key"

    out = g.copy()
    for hit in hits:
        frame = frames[hit.frame_index]
        r0, c0, r1, c1 = frame.bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = hit.marker_color
        for p in hit.path:
            assert int(g[p]) == 0
            out[p] = hit.marker_color
        out[frame.gap] = frame.color
    return out


def _paint_frame(grid: np.ndarray, box: BBox, color: int, gap: Coord) -> None:
    for p in _border_cells(box):
        grid[p] = color
    grid[gap] = 0


def _rotate(grid: np.ndarray) -> np.ndarray:
    k = random.randrange(4)
    out = np.rot90(grid, k=k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _aligned_with_any_gap(marker: Coord, frames: list[Frame]) -> bool:
    return any(marker[0] == f.gap[0] or marker[1] == f.gap[1] for f in frames)


def _blocked_by_other_frame(grid: np.ndarray, marker: Coord, target: Frame, frames: list[Frame]) -> bool:
    """Check the explicit hard negative: aligned toward target, but another frame is hit first."""
    mr, mc = marker
    gr, gc = target.gap
    if mr == gr:
        step = (0, 1 if gc > mc else -1)
    elif mc == gc:
        step = (1 if gr > mr else -1, 0)
    else:
        return False
    other_cells = set().union(*(f.cells for f in frames if f != target))
    r, c = marker
    while (r, c) != target.gap:
        r += step[0]
        c += step[1]
        if (r, c) in other_cells:
            return True
        if int(grid[r, c]) != 0:
            return False
    return False


def _semantic_check(grid: np.ndarray) -> bool:
    """Validate uniqueness, active branches, and the intended blocking counterexample."""
    frames = _find_frames(grid)
    if len(frames) not in (2, 3) or len({f.gap for f in frames}) != len(frames):
        return False
    markers = _markers(grid, frames)
    hits = _hits(grid, frames, markers)
    if len(hits) != len(frames) or len({h.frame_index for h in hits}) != len(frames):
        return False
    hit_markers = {h.marker for h in hits}
    ineffective = [m for m in markers if m not in hit_markers]
    blocked = any(
        _blocked_by_other_frame(grid, marker, target, frames)
        for marker in ineffective
        for target in frames
    )
    unaligned = any(not _aligned_with_any_gap(marker, frames) for marker in ineffective)
    if not blocked or not unaligned or any(len(h.path) == 0 for h in hits):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    for hit in hits:
        frame = frames[hit.frame_index]
        r0, c0, r1, c1 = frame.bbox
        if not np.all(out[r0 + 1 : r1, c0 + 1 : c1] == hit.marker_color):
            return False
        if int(out[frame.gap]) != frame.color:
            return False
        if any(int(out[p]) != hit.marker_color for p in hit.path):
            return False
    return all(int(out[m]) == int(grid[m]) for m in ineffective)


def _generate_attempt() -> np.ndarray | None:
    # Canonical layout: the left frame physically blocks a decoy ray aimed at
    # the right frame, while a nearer key on the far side of the blocker succeeds.
    h = random.randint(23, 30)
    w = random.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)
    palette = random.sample(range(1, 10), 6)
    left_color, right_color, left_key, right_key, blocked_color, stray_color = palette

    shared_row = random.randint(7, 11)
    lh = random.randint(5, 8)
    lr0 = random.randint(max(2, shared_row - lh + 2), shared_row - 1)
    lr1 = lr0 + lh - 1
    lc0 = random.randint(4, 6)
    lc1 = lc0 + random.randint(4, 5)

    rh = random.randint(5, 8)
    rr0 = random.randint(max(2, shared_row - rh + 2), shared_row - 1)
    rr1 = rr0 + rh - 1
    rc0 = random.randint(lc1 + 5, min(w - 7, lc1 + 8))
    rc1 = rc0 + random.randint(4, min(7, w - rc0 - 2))
    if not (lr0 < shared_row < lr1 and rr0 < shared_row < rr1):
        return None

    right_gap = (shared_row, rc0)
    left_side = random.choice(("top", "bottom"))
    left_gap_col = random.randint(lc0 + 1, lc1 - 1)
    left_gap = (lr0, left_gap_col) if left_side == "top" else (lr1, left_gap_col)
    _paint_frame(grid, (lr0, lc0, lr1, lc1), left_color, left_gap)
    _paint_frame(grid, (rr0, rc0, rr1, rc1), right_color, right_gap)

    # Active right key is between the blocking frame and its target gap.
    right_marker = (shared_row, random.randint(lc1 + 2, rc0 - 2))
    grid[right_marker] = right_key
    # A farther aligned marker is stopped by the left frame before reaching that keyhole.
    blocked_marker = (shared_row, random.randint(0, lc0 - 2))
    grid[blocked_marker] = blocked_color

    # The blocking frame has its own perpendicular active key.
    if left_side == "top":
        left_marker = (random.randint(0, lr0 - 2), left_gap_col)
    else:
        left_marker = (random.randint(lr1 + 2, min(h - 1, lr1 + 5)), left_gap_col)
    grid[left_marker] = left_key

    # Add an optional third active frame in the lower-right region.
    if random.random() < 0.5:
        tr0 = max(lr1, rr1) + 4
        th = random.randint(4, min(6, h - tr0 - 2)) if h - tr0 - 2 >= 4 else 0
        tw = random.randint(4, 6)
        tc0 = random.randint(max(2, w // 2 - 3), w - tw - 2)
        if th:
            tr1, tc1 = tr0 + th - 1, tc0 + tw - 1
            third_color = stray_color
            third_key_color = random.choice([c for c in range(1, 10) if c != third_color])
            third_gap_col = random.randint(tc0 + 1, tc1 - 1)
            third_gap = (tr1, third_gap_col)
            _paint_frame(grid, (tr0, tc0, tr1, tc1), third_color, third_gap)
            marker_row = random.randint(tr1 + 2, h - 1)
            grid[marker_row, third_gap_col] = third_key_color

    # A genuinely unaligned distractor rules out "change every marker" shortcuts.
    frames_now = _find_frames(grid)
    free = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == 0
        and all(r != f.gap[0] and c != f.gap[1] for f in frames_now)
        and all(not (f.bbox[0] <= r <= f.bbox[2] and f.bbox[1] <= c <= f.bbox[3]) for f in frames_now)
    ]
    if not free:
        return None
    grid[random.choice(free)] = stray_color

    grid = _rotate(grid)
    return grid if _semantic_check(grid) else None


def generate_input() -> np.ndarray:
    """Generate varied two/three-frame examples with active and blocked ray branches."""
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a nondegenerate lock-and-ray input")
