import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# Generator configuration.
MIN_HEIGHT = 8
MAX_HEIGHT = 24
MIN_WIDTH = 8
MAX_WIDTH = 24
MIN_RECT_SIDE = 3
MIN_RECTS = 3
MAX_RECTS = 8
EDGE_SELECT_PROB = 0.7
GEN_RETRIES = 400


def _candidate_fill(grid: np.ndarray, r: int, c: int) -> int | None:
    counts: dict[int, int] = {}
    h, w = grid.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            color = int(grid[nr, nc])
            if color != BG:
                counts[color] = counts.get(color, 0) + 1

    if not counts:
        return None

    best_count = max(counts.values())
    if best_count < 2:
        return None

    winners = [color for color, count in counts.items() if count == best_count]
    if len(winners) != 1:
        return None
    return winners[0]


def _repair_gaps(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape

    while True:
        fills: list[tuple[int, int, int]] = []
        for r in range(h):
            for c in range(w):
                if int(out[r, c]) != BG:
                    continue
                color = _candidate_fill(out, r, c)
                if color is not None:
                    fills.append((r, c, color))
        if not fills:
            break
        for r, c, color in fills:
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore damaged rectangle partitions by repainting black gap pixels with their owning region colors.

    Input structure:
    - Black is the apparent background color.
    - The non-black area is tiled by several adjacent monochrome rectangles.
    - Some pixels along shared rectangle borders are replaced by black in complementary pairs, creating local notches on both sides of the seam.

    Rule:
    - Treat each black pixel as a candidate missing border cell.
    - If a candidate black pixel has one uniquely most-supported non-black color among its 4-neighbors, and that color appears at least twice, repaint the pixel with that color.
    - Repeat until no more black pixels can be assigned; the repaired grid is the output.

    Key insight:
    - The black cells are not true empty background: each damaged border pixel is locally surrounded on two or three sides by the color of its original rectangle, so local neighbor consensus is enough to recover the full rectangle tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _repair_gaps(g)


def _split_rectangles(height: int, width: int) -> list[tuple[int, int, int, int]]:
    rects = [(0, 0, height, width)]
    max_rects = min(MAX_RECTS, (height * width) // (MIN_RECT_SIDE * MIN_RECT_SIDE))
    target = random.randint(MIN_RECTS, max_rects)

    while len(rects) < target:
        options: list[tuple[int, str]] = []
        for idx, (r0, c0, r1, c1) in enumerate(rects):
            h = r1 - r0
            w = c1 - c0
            if h >= 2 * MIN_RECT_SIDE:
                options.append((idx, "h"))
            if w >= 2 * MIN_RECT_SIDE:
                options.append((idx, "v"))
        if not options:
            break

        idx, axis = random.choice(options)
        r0, c0, r1, c1 = rects.pop(idx)
        if axis == "h":
            cut = random.randint(r0 + MIN_RECT_SIDE, r1 - MIN_RECT_SIDE)
            rects.extend([(r0, c0, cut, c1), (cut, c0, r1, c1)])
        else:
            cut = random.randint(c0 + MIN_RECT_SIDE, c1 - MIN_RECT_SIDE)
            rects.extend([(r0, c0, r1, cut), (r0, cut, r1, c1)])

    return rects


def _adjacency(
    rects: list[tuple[int, int, int, int]]
) -> tuple[list[dict[str, int]], list[set[int]]]:
    edges: list[dict[str, int]] = []
    neighbors = [set() for _ in rects]

    for i, (r0, c0, r1, c1) in enumerate(rects):
        for j in range(i + 1, len(rects)):
            rr0, cc0, rr1, cc1 = rects[j]

            if c1 == cc0 or cc1 == c0:
                start = max(r0, rr0)
                end = min(r1, rr1)
                if start < end:
                    if c1 == cc0:
                        left, right = i, j
                        left_col, right_col = c1 - 1, cc0
                    else:
                        left, right = j, i
                        left_col, right_col = cc1 - 1, c0
                    edges.append(
                        {
                            "axis": 0,
                            "a": left,
                            "b": right,
                            "start": start,
                            "end": end,
                            "p0": left_col,
                            "p1": right_col,
                        }
                    )
                    neighbors[i].add(j)
                    neighbors[j].add(i)

            if r1 == rr0 or rr1 == r0:
                start = max(c0, cc0)
                end = min(c1, cc1)
                if start < end:
                    if r1 == rr0:
                        top, bottom = i, j
                        top_row, bottom_row = r1 - 1, rr0
                    else:
                        top, bottom = j, i
                        top_row, bottom_row = rr1 - 1, r0
                    edges.append(
                        {
                            "axis": 1,
                            "a": top,
                            "b": bottom,
                            "start": start,
                            "end": end,
                            "p0": top_row,
                            "p1": bottom_row,
                        }
                    )
                    neighbors[i].add(j)
                    neighbors[j].add(i)

    return edges, neighbors


def _assign_colors(neighbors: list[set[int]]) -> list[int] | None:
    order = sorted(range(len(neighbors)), key=lambda idx: len(neighbors[idx]), reverse=True)
    assigned: dict[int, int] = {}
    for idx in order:
        forbidden = {assigned[nbr] for nbr in neighbors[idx] if nbr in assigned}
        choices = [color for color in COLORS if color not in forbidden]
        if not choices:
            return None
        assigned[idx] = random.choice(choices)
    return [assigned[idx] for idx in range(len(neighbors))]


def _paint_rectangles(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for (r0, c0, r1, c1), color in zip(rects, colors):
        grid[r0:r1, c0:c1] = color
    return grid


def _sample_gap_mask(length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    mode = random.randrange(3)

    if mode == 0:
        count = random.randint(1, max(1, min(length, 1 + length // 3)))
        positions = random.sample(range(length), count)
        mask[positions] = True
    elif mode == 1:
        n_segments = random.randint(1, 2 if length >= 4 else 1)
        for _ in range(n_segments):
            seg_len = random.randint(1, min(4, length))
            start = random.randint(0, length - seg_len)
            mask[start : start + seg_len] = True
    else:
        step = random.choice((2, 3))
        offset = random.randint(0, step - 1)
        mask[offset:length:step] = True
        if length >= 5 and random.random() < 0.5:
            start = random.randint(0, length - 2)
            mask[start : start + 2] = True

    if not mask.any():
        mask[random.randrange(length)] = True
    if length > 2 and mask.all():
        mask[random.randrange(length)] = False
    return mask


def _carve_gaps(grid: np.ndarray, edges: list[dict[str, int]]) -> tuple[np.ndarray, int]:
    out = grid.copy()
    if not edges:
        return out, 0

    indices = [idx for idx in range(len(edges)) if random.random() < EDGE_SELECT_PROB]
    if not indices:
        indices = [random.randrange(len(edges))]

    present_axes = {edges[idx]["axis"] for idx in indices}
    all_axes = {edge["axis"] for edge in edges}
    missing_axes = list(all_axes - present_axes)
    if missing_axes:
        extra_candidates = [idx for idx, edge in enumerate(edges) if edge["axis"] in missing_axes]
        extra = random.choice(extra_candidates)
        if extra not in indices:
            indices.append(extra)

    gap_cells = 0
    for idx in indices:
        edge = edges[idx]
        length = edge["end"] - edge["start"]
        mask = _sample_gap_mask(length)
        coords = np.arange(edge["start"], edge["end"])[mask]
        if edge["axis"] == 0:
            out[coords, edge["p0"]] = BG
            out[coords, edge["p1"]] = BG
        else:
            out[edge["p0"], coords] = BG
            out[edge["p1"], coords] = BG
        gap_cells += 2 * int(mask.sum())

    return out, gap_cells


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)

    rects = _split_rectangles(height, width)
    if len(rects) < MIN_RECTS:
        return None

    edges, neighbors = _adjacency(rects)
    if not edges:
        return None

    colors = _assign_colors(neighbors)
    if colors is None:
        return None

    target = _paint_rectangles(height, width, rects, colors)
    puzzle, gap_cells = _carve_gaps(target, edges)
    if gap_cells == 0:
        return None

    repaired = main(puzzle)
    if not np.array_equal(repaired, target):
        return None

    return puzzle


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
