from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


BG = 0
FRAME = 5
CANVAS = 30
FRAME_SIZES = (5, 7, 9)
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


@dataclass
class FrameInfo:
    bbox: tuple[int, int, int, int]
    seed_pos: tuple[int, int]
    seed_color: int
    gap_side: str | None = None


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _midpoint_cell(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "up":
        return (r0, (c0 + c1) // 2)
    if side == "down":
        return (r1, (c0 + c1) // 2)
    if side == "left":
        return ((r0 + r1) // 2, c0)
    if side == "right":
        return ((r0 + r1) // 2, c1)
    raise ValueError(f"unknown side: {side}")


def _frame_border_cells(
    bbox: tuple[int, int, int, int], gap_side: str | None = None
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    if gap_side is not None:
        cells.remove(_midpoint_cell(bbox, gap_side))
    return cells


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _gap_side_from_missing(
    bbox: tuple[int, int, int, int], missing: tuple[int, int] | None
) -> str | None:
    if missing is None:
        return None
    r0, c0, r1, c1 = bbox
    r, c = missing
    if r == r0 and c0 < c < c1 and c == (c0 + c1) // 2:
        return "up"
    if r == r1 and c0 < c < c1 and c == (c0 + c1) // 2:
        return "down"
    if c == c0 and r0 < r < r1 and r == (r0 + r1) // 2:
        return "left"
    if c == c1 and r0 < r < r1 and r == (r0 + r1) // 2:
        return "right"
    return None


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _detect_frames(grid: np.ndarray) -> tuple[int, list[FrameInfo]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors, counts = np.unique(g[g != bg], return_counts=True)
    if len(colors) == 0:
        return FRAME, []
    frame_color = int(colors[np.argmax(counts)])
    frames: list[FrameInfo] = []
    for comp in _component_sets(g == frame_color):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        perimeter = _frame_border_cells(bbox)
        extra = comp - perimeter
        if extra:
            continue
        missing = list(perimeter - comp)
        if len(missing) > 1:
            continue
        gap_side = _gap_side_from_missing(bbox, missing[0] if missing else None)
        if len(missing) == 1 and gap_side is None:
            continue
        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere((inside != bg) & (inside != frame_color))
        if seed_coords.shape[0] != 1:
            continue
        sr, sc = map(int, seed_coords[0])
        seed_pos = (r0 + 1 + sr, c0 + 1 + sc)
        frames.append(
            FrameInfo(
                bbox=bbox,
                seed_pos=seed_pos,
                seed_color=int(g[seed_pos]),
                gap_side=gap_side,
            )
        )
    frames.sort(key=lambda fr: fr.bbox)
    return frame_color, frames


def _detect_edges(
    grid: np.ndarray, frame_color: int, frames: list[FrameInfo]
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    owner: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in _frame_border_cells(frame.bbox, frame.gap_side):
            owner[cell] = idx

    edges: list[tuple[int, int]] = []
    for idx, frame in enumerate(frames):
        if frame.gap_side is None:
            continue
        gr, gc = _midpoint_cell(frame.bbox, frame.gap_side)
        dr, dc = DIRS[frame.gap_side]
        r, c = gr + dr, gc + dc
        length = 0
        while 0 <= r < h and 0 <= c < w and int(g[r, c]) == frame.seed_color:
            length += 1
            r += dr
            c += dc
        if length == 0:
            continue
        if 0 <= r < h and 0 <= c < w and int(g[r, c]) == frame_color and (r, c) in owner:
            target = owner[(r, c)]
            if target != idx:
                edges.append((idx, target))
    return edges


def _propagated_colors(frames: list[FrameInfo], edges: list[tuple[int, int]]) -> list[int]:
    n = len(frames)
    colors = [frame.seed_color for frame in frames]
    outgoing = [[] for _ in range(n)]
    indeg = [0] * n
    for src, dst in edges:
        outgoing[src].append(dst)
        indeg[dst] += 1

    q = deque(i for i in range(n) if indeg[i] == 0)
    seen = 0
    while q:
        cur = q.popleft()
        seen += 1
        for nxt in outgoing[cur]:
            colors[nxt] = colors[cur]
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if seen < n:
        for _ in range(n):
            changed = False
            for src, dst in edges:
                if colors[dst] != colors[src]:
                    colors[dst] = colors[src]
                    changed = True
            if not changed:
                break
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed into a cross inside its gray frame, then propagate colors along gap-arrow links between frames.

    Input structure:
    - The background is black.
    - Several gray 1-pixel-thick rectangular frames appear on the grid; each frame may be complete or may have one midpoint gap on one side.
    - Each frame contains exactly one non-black, non-gray seed pixel, and the seed colors are distinct.
    - A gapped frame may emit a straight 1-pixel-wide colored arrow segment from its gap toward another frame's outer wall.

    Rule:
    - First, inside every frame, extend the seed color horizontally and vertically across the whole interior row and column, forming a cross centered on the seed.
    - Then, for each arrow, treat the gapped frame as the source and recolor the entire cross inside the target frame to the source frame's current propagated color.
    - Apply propagation in arrow order from roots toward descendants, so chain links pass the upstream color onward.
    - Keep the gray frames, the gaps, and the external arrow segments unchanged.

    Key insight:
    - The puzzle combines two separate mechanisms: a local cross-expansion inside each rectangular enclosure, and a global directed color-transfer graph encoded by gap-aligned arrow segments.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frame_color, frames = _detect_frames(g)
    edges = _detect_edges(g, frame_color, frames)
    final_colors = _propagated_colors(frames, edges)

    for frame, color in zip(frames, final_colors):
        r0, c0, r1, c1 = frame.bbox
        sr, sc = frame.seed_pos
        out[sr, c0 + 1 : c1] = color
        out[r0 + 1 : r1, sc] = color
    return out


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _can_place(
    cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int
) -> bool:
    if not _in_bounds(cells, h, w):
        return False
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _sample_frame(rng: random.Random, color: int) -> FrameInfo:
    h = rng.choice(FRAME_SIZES)
    w = rng.choice(FRAME_SIZES)
    r0 = rng.randint(1, CANVAS - h - 2)
    c0 = rng.randint(1, CANVAS - w - 2)
    sr = rng.randint(r0 + 1, r0 + h - 2)
    sc = rng.randint(c0 + 1, c0 + w - 2)
    return FrameInfo(bbox=(r0, c0, r0 + h - 1, c0 + w - 1), seed_pos=(sr, sc), seed_color=color)


def _frame_cells(frame: FrameInfo) -> set[tuple[int, int]]:
    return _frame_border_cells(frame.bbox, frame.gap_side) | {frame.seed_pos}


def _place_free_frame(
    rng: random.Random, occupied: set[tuple[int, int]], color: int
) -> FrameInfo | None:
    for _ in range(PLACEMENT_RETRIES):
        frame = _sample_frame(rng, color)
        if _can_place(_frame_cells(frame), occupied, CANVAS, CANVAS):
            return frame
    return None


def _arrow_cells_from_gap(
    bbox: tuple[int, int, int, int], side: str, length: int
) -> set[tuple[int, int]]:
    gr, gc = _midpoint_cell(bbox, side)
    dr, dc = DIRS[side]
    return {(gr + dr * step, gc + dc * step) for step in range(1, length + 1)}


def _extend_chain(
    rng: random.Random,
    source: FrameInfo,
    occupied: set[tuple[int, int]],
    color: int,
    banned_dir: str | None,
) -> tuple[FrameInfo, set[tuple[int, int]], str] | None:
    directions = list(DIRS)
    rng.shuffle(directions)
    if banned_dir is not None:
        directions = [d for d in directions if d != banned_dir]

    for side in directions:
        source_gap = _midpoint_cell(source.bbox, side)
        occupied_wo_gap = set(occupied)
        occupied_wo_gap.discard(source_gap)
        for _ in range(PLACEMENT_RETRIES):
            h = rng.choice(FRAME_SIZES)
            w = rng.choice(FRAME_SIZES)
            arrow_len = rng.randint(2, 4)
            gr, gc = source_gap

            if side == "right":
                r0 = gr - rng.randint(1, h - 2)
                c0 = gc + arrow_len + 1
            elif side == "left":
                r0 = gr - rng.randint(1, h - 2)
                c0 = gc - arrow_len - w
            elif side == "down":
                r0 = gr + arrow_len + 1
                c0 = gc - rng.randint(1, w - 2)
            else:
                r0 = gr - arrow_len - h
                c0 = gc - rng.randint(1, w - 2)

            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            target = FrameInfo(
                bbox=bbox,
                seed_pos=(rng.randint(r0 + 1, r0 + h - 2), rng.randint(c0 + 1, c0 + w - 2)),
                seed_color=color,
            )
            arrow = _arrow_cells_from_gap(source.bbox, side, arrow_len)
            candidate = _frame_cells(target) | arrow
            if _can_place(candidate, occupied_wo_gap, CANVAS, CANVAS):
                return target, arrow, side
    return None


def _scenario_chain_lengths(rng: random.Random) -> list[int]:
    scenario = rng.choice(("single_edge", "long_chain", "two_chains"))
    if scenario == "single_edge":
        total = rng.randint(2, 5)
        return [2] + [1] * (total - 2)
    if scenario == "long_chain":
        total = rng.randint(3, 5)
        main_len = rng.randint(3, total)
        return [main_len] + [1] * (total - main_len)
    total = rng.randint(4, 5)
    first = rng.randint(2, total - 2)
    return [first, total - first]


def _crop_with_margin(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = min(int(r0), rng.randint(1, 3))
    left = min(int(c0), rng.randint(1, 3))
    bottom = min(int(grid.shape[0] - 1 - r1), rng.randint(1, 3))
    right = min(int(grid.shape[1] - 1 - c1), rng.randint(1, 3))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.full((CANVAS, CANVAS), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    frames: list[FrameInfo] = []
    arrows: list[tuple[set[tuple[int, int]], int]] = []
    edges: list[tuple[int, int]] = []

    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    lengths = sorted(_scenario_chain_lengths(rng), reverse=True)
    needed_frames = sum(lengths)
    if needed_frames > len(colors):
        return None

    color_idx = 0
    for chain_len in lengths:
        root = _place_free_frame(rng, occupied, colors[color_idx])
        color_idx += 1
        if root is None:
            return None
        root_idx = len(frames)
        frames.append(root)
        occupied |= _frame_cells(root)
        cur_idx = root_idx
        prev_dir: str | None = None

        for _ in range(chain_len - 1):
            result = _extend_chain(
                rng,
                frames[cur_idx],
                occupied,
                colors[color_idx],
                OPPOSITE[prev_dir] if prev_dir is not None else None,
            )
            color_idx += 1
            if result is None:
                return None
            target, arrow_cells, side = result
            gap = _midpoint_cell(frames[cur_idx].bbox, side)
            occupied.discard(gap)
            frames[cur_idx].gap_side = side
            occupied |= _frame_cells(target)
            occupied |= arrow_cells
            arrows.append((arrow_cells, frames[cur_idx].seed_color))
            next_idx = len(frames)
            frames.append(target)
            edges.append((cur_idx, next_idx))
            cur_idx = next_idx
            prev_dir = side

    for frame in frames:
        for r, c in _frame_border_cells(frame.bbox, frame.gap_side):
            grid[r, c] = FRAME
        sr, sc = frame.seed_pos
        grid[sr, sc] = frame.seed_color
    for arrow_cells, color in arrows:
        for r, c in arrow_cells:
            grid[r, c] = color

    cropped = _crop_with_margin(grid, rng)
    _, detected_frames = _detect_frames(cropped)
    if len(detected_frames) != len(frames):
        return None
    if len(_detect_edges(cropped, FRAME, detected_frames)) != len(edges):
        return None
    if not np.any(main(cropped) != cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
