from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
GREY = 5
FRAME_W = 4
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int
    gap: int
    direction: int
    seed: int
    seed_pos: tuple[int, int]

    @property
    def spill_r(self) -> int:
        return self.r0 - 1


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expected_grey_cells(r0: int, c0: int, r1: int, c1: int, gap: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1) if c != gap}
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r1, c))
    return cells


def _detect_frames(grid: np.ndarray, bg: int) -> list[Frame]:
    frames: list[Frame] = []
    for comp in _components4(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        assert r0 > 0, "each frame needs one background row above its top"
        assert c1 - c0 + 1 == FRAME_W, "expected four-column capped frames"
        assert r1 - r0 + 1 >= 4, "frame must have room for an interior seed"

        top = grid[r0, c0 : c1 + 1]
        gap_offsets = np.flatnonzero(top == bg)
        assert len(gap_offsets) == 1, "top edge must have exactly one background gap"
        gap_off = int(gap_offsets[0])
        gap = c0 + gap_off
        assert np.count_nonzero(top == GREY) == FRAME_W - 1, "top edge must be grey except the gap"

        left_cap = gap_off == 2 and top[0] == GREY and top[1] == GREY
        right_cap = gap_off == 1 and top[2] == GREY and top[3] == GREY
        assert left_cap != right_cap, "the two-pixel cap must choose exactly one horizontal direction"
        direction = -1 if left_cap else 1

        expected = _expected_grey_cells(r0, c0, r1, c1, gap)
        assert set(comp) == expected, "grey component must be exactly one open-top frame"

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_local = np.argwhere((interior != bg) & (interior != GREY))
        assert len(seed_local) == 1, "each frame must contain exactly one colored seed"
        sr = r0 + 1 + int(seed_local[0, 0])
        sc = c0 + 1 + int(seed_local[0, 1])
        frames.append(Frame(r0, c0, r1, c1, gap, direction, int(grid[sr, sc]), (sr, sc)))

    assert frames, "expected at least one grey frame"
    seed_positions = {f.seed_pos for f in frames}
    colored_positions = {
        (int(r), int(c))
        for r, c in np.argwhere((grid != bg) & (grid != GREY))
    }
    assert colored_positions == seed_positions, "all non-grey foreground pixels must be frame seeds"
    return sorted(frames, key=lambda f: (f.r0, f.c0, f.gap))


def _build_edges(frames: list[Frame]) -> tuple[list[int | None], list[int]]:
    edges: list[int | None] = [None] * len(frames)
    indeg = [0] * len(frames)

    for i, src in enumerate(frames):
        best_j: int | None = None
        best_dist: int | None = None
        for j, dst in enumerate(frames):
            if i == j or dst.spill_r != src.spill_r:
                continue
            delta = dst.gap - src.gap
            if src.direction * delta <= 0:
                continue
            dist = abs(delta)
            if best_dist is None or dist < best_dist:
                best_j = j
                best_dist = dist
        if best_j is not None:
            edges[i] = best_j
            indeg[best_j] += 1

    assert all(k <= 1 for k in indeg), "a frame gap may receive water from only one upstream frame"
    for start in range(len(frames)):
        seen: set[int] = set()
        cur: int | None = start
        while cur is not None:
            assert cur not in seen, "frame spill graph must be acyclic"
            seen.add(cur)
            cur = edges[cur]
    return edges, indeg


def _fill_frame(out: np.ndarray, original: np.ndarray, frame: Frame, color: int) -> None:
    region = out[frame.r0 : frame.r1, frame.c0 + 1 : frame.c1]
    src = original[frame.r0 : frame.r1, frame.c0 + 1 : frame.c1]
    region[src != GREY] = int(color)


def _paint_ray(out: np.ndarray, bg: int, frame: Frame, end_col: int, color: int) -> None:
    step = 1 if frame.direction > 0 else -1
    c = frame.gap
    while True:
        cur = int(out[frame.spill_r, c])
        assert cur in (int(bg), int(color)), "spill rays must not cross other objects or colors"
        out[frame.spill_r, c] = int(color)
        if c == end_col:
            break
        c += step


def _solve_from_frames(grid: np.ndarray, bg: int, frames: list[Frame]) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    edges, indeg = _build_edges(frames)

    colors: list[int | None] = [None] * len(frames)
    heads = [i for i, k in enumerate(indeg) if k == 0]
    assert heads, "at least one chain head is required"
    for i in heads:
        colors[i] = frames[i].seed

    q: deque[int] = deque(sorted(heads, key=lambda i: (frames[i].r0, frames[i].gap)))
    done: set[int] = set()
    while q:
        i = q.popleft()
        if i in done:
            continue
        color = colors[i]
        assert color is not None, "frame color must be known before it spills"
        frame = frames[i]
        _fill_frame(out, grid, frame, int(color))

        j = edges[i]
        end_col = frames[j].gap if j is not None else (w - 1 if frame.direction > 0 else 0)
        assert 0 <= frame.spill_r < h and 0 <= end_col < w
        _paint_ray(out, bg, frame, end_col, int(color))

        done.add(i)
        if j is not None:
            if colors[j] is None:
                colors[j] = int(color)
            else:
                assert colors[j] == int(color), "downstream color must be unique"
            q.append(j)

    assert len(done) == len(frames), "every frame must be reachable from a chain head"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill capped grey frames and pass the fill color along spill-ray chains.

    Input structure:
    - The background is the dominant color, with grey four-column open-top rectangular frames.
    - Each frame has complete left, right, and bottom sides, one top gap, and a two-pixel grey cap on exactly one side of that gap.
    - Every frame interior contains one non-grey seed pixel; frames whose gap is directly under an upstream spill ray may have decoy seed colors.

    Rule:
    - A frame with no incoming spill is filled through its interior and top gap using its own seed color.
    - From each filled frame, draw a same-color horizontal spill ray on the row above its top, moving toward the side marked by the two-pixel cap.
    - If the ray reaches another frame's gap column on that same overhead row, stop there and fill that downstream frame with the incoming color, overriding its seed; otherwise continue to the grid edge.

    Key insight:
    - Only chain heads keep their seed color; the cap-directed overhead ray is the pipe that transfers that color through the downstream frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)
    assert bg != GREY, "grey is reserved for the frames"
    frames = _detect_frames(g, bg)
    return _solve_from_frames(g, bg, frames)


# --------------- generator ---------------


def _split_extra(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _frame_cells(frame: Frame) -> set[tuple[int, int]]:
    cells = _expected_grey_cells(frame.r0, frame.c0, frame.r1, frame.c1, frame.gap)
    cells.add(frame.seed_pos)
    return cells


def _make_frame(r0: int, c0: int, height: int, direction: int, seed: int) -> Frame:
    r1 = r0 + height - 1
    c1 = c0 + FRAME_W - 1
    gap = c0 + (1 if direction > 0 else 2)
    sr = random.randint(r0 + 1, r1 - 1)
    sc = random.randint(c0 + 1, c1 - 1)
    return Frame(r0, c0, r1, c1, gap, direction, int(seed), (sr, sc))


def _draw_frame(grid: np.ndarray, frame: Frame) -> None:
    for r, c in _expected_grey_cells(frame.r0, frame.c0, frame.r1, frame.c1, frame.gap):
        assert int(grid[r, c]) == BG
        grid[r, c] = GREY
    sr, sc = frame.seed_pos
    assert int(grid[sr, sc]) == BG
    grid[sr, sc] = frame.seed


def _group_columns(width: int, length: int, direction: int) -> list[int] | None:
    if length == 1:
        candidates = []
        for c0 in range(0, width - FRAME_W + 1):
            gap = c0 + (1 if direction > 0 else 2)
            ray_len = width - gap if direction > 0 else gap + 1
            if ray_len >= 5:
                candidates.append(c0)
        return [random.choice(candidates)] if candidates else None

    for _ in range(80):
        spaces = [random.randint(2, 5) for _ in range(length - 1)]
        total_w = FRAME_W * length + sum(spaces)
        if total_w > width:
            continue
        starts: list[int] = []
        for left in range(0, width - total_w + 1):
            c0s = [left]
            cur = left
            for sp in spaces:
                cur += FRAME_W + sp
                c0s.append(cur)
            if direction > 0:
                tail_gap = c0s[-1] + 1
                if width - tail_gap < 5:
                    continue
            else:
                tail_gap = c0s[0] + 2
                if tail_gap + 1 < 5:
                    continue
            starts.append(left)
        if not starts:
            continue
        left = random.choice(starts)
        c0s = [left]
        cur = left
        for sp in spaces:
            cur += FRAME_W + sp
            c0s.append(cur)
        return c0s
    return None


def _group_plan() -> list[dict[str, int]]:
    chain_dir = random.choice((-1, 1))
    groups = [
        {"length": random.choice((2, 2, 3)), "direction": chain_dir},
        {"length": 1, "direction": -chain_dir},
    ]
    if random.random() < 0.55:
        groups.append({"length": random.choice((1, 2)), "direction": random.choice((-1, 1))})
    if random.random() < 0.25:
        groups.append({"length": 1, "direction": random.choice((-1, 1))})
    random.shuffle(groups)
    return groups


def _vertical_layout(height: int, max_heights: list[int]) -> list[int] | None:
    n = len(max_heights)
    band_heights = [mh + 1 for mh in max_heights]
    min_total = sum(band_heights) + max(0, n - 1)
    if min_total > height:
        return None
    gaps = _split_extra(height - min_total, n + 1)
    starts: list[int] = []
    row = gaps[0]
    for i, band_h in enumerate(band_heights):
        starts.append(row + 1)
        row += band_h
        if i < n - 1:
            row += 1
        row += gaps[i + 1]
    return starts


def _render(frames: list[Frame], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    used: set[tuple[int, int]] = set()
    for frame in frames:
        cells = _frame_cells(frame)
        assert not (used & cells)
        used |= cells
        _draw_frame(grid, frame)
    return grid


def _propagated_colors(frames: list[Frame]) -> tuple[list[int | None], list[int | None], list[int]]:
    edges, indeg = _build_edges(frames)
    colors: list[int | None] = [None] * len(frames)
    q: deque[int] = deque()
    for i, k in enumerate(indeg):
        if k == 0:
            colors[i] = frames[i].seed
            q.append(i)
    seen: set[int] = set()
    while q:
        i = q.popleft()
        if i in seen:
            continue
        seen.add(i)
        j = edges[i]
        if j is not None:
            colors[j] = colors[i]
            q.append(j)
    return colors, edges, indeg


def _semantic_checks(inp: np.ndarray, out: np.ndarray, frames: list[Frame]) -> bool:
    colors, edges, indeg = _propagated_colors(frames)
    if not any(edge is not None for edge in edges):
        return False
    if not any(edge is None and indeg[i] == 0 for i, edge in enumerate(edges)):
        return False
    if not any(f.direction > 0 for f in frames) or not any(f.direction < 0 for f in frames):
        return False

    overridden = False
    for i, edge in enumerate(edges):
        if edge is None:
            continue
        assert colors[i] is not None and colors[edge] == colors[i]
        sr, sc = frames[edge].seed_pos
        if int(inp[sr, sc]) != int(colors[i]) and int(out[sr, sc]) == int(colors[i]):
            overridden = True
    if not overridden:
        return False

    for i, frame in enumerate(frames):
        color = colors[i]
        if color is None:
            return False
        patch = out[frame.r0 : frame.r1, frame.c0 + 1 : frame.c1]
        original = inp[frame.r0 : frame.r1, frame.c0 + 1 : frame.c1]
        if not np.all(patch[original != GREY] == int(color)):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(18, 30)
    groups = _group_plan()
    lengths = [g["length"] for g in groups]
    total_frames = sum(lengths)
    if total_frames > len(SEED_COLORS):
        return None

    group_heights: list[list[int]] = []
    max_heights: list[int] = []
    for length in lengths:
        hs = [random.randint(4, 7) for _ in range(length)]
        group_heights.append(hs)
        max_heights.append(max(hs))

    top_rows = _vertical_layout(h, max_heights)
    if top_rows is None:
        return None

    colors = random.sample(SEED_COLORS, total_frames)
    color_i = 0
    frames: list[Frame] = []
    for group, heights, r0 in zip(groups, group_heights, top_rows):
        direction = group["direction"]
        c0s = _group_columns(w, group["length"], direction)
        if c0s is None:
            return None
        for c0, fh in zip(c0s, heights):
            frames.append(_make_frame(r0, c0, fh, direction, colors[color_i]))
            color_i += 1

    try:
        inp = _render(frames, (h, w))
        expected = _solve_from_frames(inp, BG, frames)
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(inp, actual, frames):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overflow-pipeline input")
