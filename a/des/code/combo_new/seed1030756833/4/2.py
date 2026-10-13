from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_frame_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return (r0, c0, r1, c1) if set(cells) == expected else None


def _find_frames(g: np.ndarray) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, GRAY)):
        for comp in _components(g == color):
            bbox = _rect_frame_bbox(comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            frames.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    frames.sort(key=lambda f: (f["r0"], f["c0"], f["r1"], f["c1"], f["color"]))
    return frames


def _straight_line(comp: list[tuple[int, int]]):
    rows = sorted({r for r, _c in comp})
    cols = sorted({c for _r, c in comp})
    if len(rows) == 1 and len(cols) == len(comp) and cols == list(range(cols[0], cols[-1] + 1)):
        return "h", rows[0], cols[0], cols[-1]
    if len(cols) == 1 and len(rows) == len(comp) and rows == list(range(rows[0], rows[-1] + 1)):
        return "v", cols[0], rows[0], rows[-1]
    return None


def _parse_arrows(
    g: np.ndarray,
    frame: dict[str, int],
    seed: tuple[int, int],
) -> list[tuple[tuple[int, int], int]]:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    gray = (g == GRAY).copy()
    gray[: r0 + 1, :] = False
    gray[r1:, :] = False
    gray[:, : c0 + 1] = False
    gray[:, c1:] = False

    comps = _components(gray)
    heads = {comp[0] for comp in comps if len(comp) == 1}
    tails = [(comp, _straight_line(comp)) for comp in comps if len(comp) >= 2]
    tails = [(comp, info) for comp, info in tails if info is not None]

    arrows: list[tuple[tuple[int, int], int]] = []
    used_heads: set[tuple[int, int]] = set()
    for comp, info in tails:
        found: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
        if info[0] == "h":
            _axis, row, minc, maxc = info
            candidates = [
                ((0, -1), (row, minc - 1), (row, minc - 2)),
                ((0, 1), (row, maxc + 1), (row, maxc + 2)),
            ]
        else:
            _axis, col, minr, maxr = info
            candidates = [
                ((-1, 0), (minr - 1, col), (minr - 2, col)),
                ((1, 0), (maxr + 1, col), (maxr + 2, col)),
            ]

        for direction, gap, head in candidates:
            gr, gc = gap
            if head not in heads or head in used_heads:
                continue
            if not (r0 < gr < r1 and c0 < gc < c1):
                continue
            if int(g[gr, gc]) != BG:
                continue
            found.append((direction, len(comp) + 1, head))
        assert len(found) <= 1, "an arrow tail must point to at most one head pixel"
        if found:
            direction, length, head = found[0]
            arrows.append((direction, length))
            used_heads.add(head)

    if arrows:
        return arrows

    # Compatibility for connected straight bars: when no detached head pixels are
    # present, choose the line endpoint farther from the seed as the head.
    if heads or len(tails) != len(comps):
        return []
    sr, sc = seed
    for comp, info in tails:
        if info[0] == "h":
            _axis, row, minc, maxc = info
            ends = [(row, minc), (row, maxc)]
        else:
            _axis, col, minr, maxr = info
            ends = [(minr, col), (maxr, col)]
        distances = [abs(sr - er) + abs(sc - ec) for er, ec in ends]
        if distances[0] == distances[1]:
            return []
        head = ends[0] if distances[0] > distances[1] else ends[1]
        tail = ends[1] if head == ends[0] else ends[0]
        dr = 0 if head[0] == tail[0] else (1 if head[0] > tail[0] else -1)
        dc = 0 if head[1] == tail[1] else (1 if head[1] > tail[1] else -1)
        arrows.append(((dr, dc), len(comp)))
    return arrows


def _seed_for_frame(
    g: np.ndarray,
    frame: dict[str, int],
    frame_colors: set[int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    seeds: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            v = int(g[r, c])
            if v != BG and v != GRAY and v not in frame_colors:
                seeds.append((r, c))
    assert len(seeds) == 1, "each frame must contain exactly one colored seed"
    return seeds[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow frame-colored line segments from colored seeds using grey arrows as direction and length cues.

    Input structure:
    - Black background with several non-overlapping, one-pixel-thick rectangular frames in distinct colors.
    - Each frame interior contains one colored seed pixel whose color is not a frame color.
    - Each frame interior also contains one or more grey arrows, each made from a one-pixel head aligned with a straight tail segment.

    Rule:
    - For each frame, assign the seed and grey arrows by spatial containment inside that frame.
    - Read every arrow direction from its tail toward its head, and read its length as all grey pixels in that arrow.
    - Erase the grey arrows to black.
    - Starting at the seed, draw a line of that arrow length in the arrow direction, including the seed cell.
    - Paint the grown line with the containing frame's border color; the rectangular frames themselves remain in place.

    Key insight: The seed gives the start cell, the grey arrow gives direction and distance, and the enclosing frame supplies the output color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"
    frame_colors = {int(f["color"]) for f in frames}

    out[g == GRAY] = BG
    for frame in frames:
        seed = _seed_for_frame(g, frame, frame_colors)
        arrows = _parse_arrows(g, frame, seed)
        assert arrows, "each frame must contain at least one parseable arrow"
        params = {(direction, length) for direction, length in arrows}
        assert len(params) == 1, "arrows in one frame must agree on direction and length"
        (dr, dc), length = next(iter(params))
        sr, sc = seed
        for step in range(length):
            rr = sr + dr * step
            cc = sc + dc * step
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = int(frame["color"])
    return out


def _mark_rect_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _expanded(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _place_frame_boxes(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _try in range(160):
            fh = random.randint(7, min(12, h - 2))
            fw = random.randint(7, min(12, w - 2))
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            er0, ec0, er1, ec1 = _expanded(box, 1, h, w)
            if np.any(occupied[er0 : er1 + 1, ec0 : ec1 + 1]):
                continue
            occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _arrow_candidate(
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
    length: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    dr, dc = direction
    for _ in range(100):
        fr = random.randint(ir0, ir1)
        fc = random.randint(ic0, ic1)
        points = [(fr + dr * k, fc + dc * k) for k in range(length + 1)]
        if not all(ir0 <= r <= ir1 and ic0 <= c <= ic1 for r, c in points):
            continue
        tail = points[: length - 1]
        gap = points[length - 1]
        head = points[length]
        return tail, gap, head
    return None


def _too_close(cell: tuple[int, int], protected: set[tuple[int, int]]) -> bool:
    r, c = cell
    return any(abs(r - pr) <= 1 and abs(c - pc) <= 1 for pr, pc in protected)


def _place_arrows(
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
    length: int,
    count: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    gray_cells: set[tuple[int, int]] = set()
    gaps: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    for _ in range(count):
        placed = False
        for _try in range(160):
            cand = _arrow_candidate(box, direction, length)
            if cand is None:
                continue
            tail, gap, head = cand
            arrow_gray = set(tail + [head])
            if any(_too_close(cell, protected) for cell in arrow_gray | {gap}):
                continue
            gray_cells |= arrow_gray
            gaps.add(gap)
            protected |= arrow_gray | {gap}
            placed = True
            break
        if not placed:
            return None
    return gray_cells, gaps


def _in_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _line_cells(seed: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    sr, sc = seed
    dr, dc = direction
    return [(sr + dr * k, sc + dc * k) for k in range(length)]


def _choose_seed(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    box_index: int,
    direction: tuple[int, int],
    length: int,
    gray_cells: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = grid.shape
    box = boxes[box_index]
    r0, c0, r1, c1 = box
    candidates: list[tuple[int, int]] = []
    crossing: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != BG or (r, c) in gray_cells:
                continue
            if _too_close((r, c), gray_cells):
                continue
            cells = _line_cells((r, c), direction, length)
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                continue
            if any(cell in gray_cells or cell in reserved_output for cell in cells[1:]):
                continue
            bad = False
            for cell in cells[1:]:
                for j, other in enumerate(boxes):
                    if j != box_index and _in_box(cell, other):
                        bad = True
                        break
                if bad:
                    break
                rr, cc = cell
                if int(grid[rr, cc]) not in (BG, int(grid[r0, c0])):
                    bad = True
                    break
            if bad:
                continue
            candidates.append((r, c))
            if any(
                rr in (r0, r1) or cc in (c0, c1)
                for rr, cc in cells[1:]
                if _in_box((rr, cc), box)
            ):
                crossing.append((r, c))
    if crossing and random.random() < 0.55:
        return random.choice(crossing)
    if candidates:
        return random.choice(candidates)
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    n_frames = random.choice([1, 2, 2, 3])
    colors = PALETTE[:]
    random.shuffle(colors)
    border_colors = colors[:n_frames]
    seed_colors = colors[n_frames : 2 * n_frames]
    if len(seed_colors) < n_frames:
        return None

    boxes = _place_frame_boxes(h, w, n_frames)
    if boxes is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for box, color in zip(boxes, border_colors):
        _mark_rect_frame(grid, box, color)

    reserved_output: set[tuple[int, int]] = set()
    for i, (box, seed_color) in enumerate(zip(boxes, seed_colors)):
        r0, c0, r1, c1 = box
        placed = False
        for _try in range(120):
            direction = random.choice(DIRS)
            axis_span = (r1 - r0 - 1) if direction[1] == 0 else (c1 - c0 - 1)
            max_len = min(6, axis_span - 1)
            if max_len < 3:
                continue
            length = random.randint(3, max_len)
            arrow_count = random.choice([1, 1, 2])
            arrows = _place_arrows(box, direction, length, arrow_count)
            if arrows is None:
                continue
            gray_cells, _gaps = arrows
            seed = _choose_seed(grid, boxes, i, direction, length, gray_cells, reserved_output)
            if seed is None:
                continue
            for rr, cc in gray_cells:
                grid[rr, cc] = GRAY
            grid[seed] = seed_color
            reserved_output.update(_line_cells(seed, direction, length))
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
