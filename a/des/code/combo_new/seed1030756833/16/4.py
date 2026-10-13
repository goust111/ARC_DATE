import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            seen[sr, sc] = True
            q = deque([(sr, sc)])
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(br0: int, bc0: int, br1: int, bc1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(bc0, bc1 + 1):
        cells.append((br0, c))
        if br1 != br0:
            cells.append((br1, c))
    for r in range(br0 + 1, br1):
        cells.append((r, bc0))
        if bc1 != bc0:
            cells.append((r, bc1))
    return cells


def _find_regions(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    passable = (g == BLACK) | (g == AZURE)
    regions: list[dict] = []
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        ir0, ic0, ir1, ic1 = _bbox(comp)
        br0, bc0, br1, bc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        if br0 < 0 or bc0 < 0 or br1 >= h or bc1 >= w:
            continue
        regions.append(
            {
                "interior": set(comp),
                "ibox": (ir0, ic0, ir1, ic1),
                "bbox": (br0, bc0, br1, bc1),
                "border": _border_cells(br0, bc0, br1, bc1),
            }
        )
    regions.sort(key=lambda reg: reg["ibox"])
    return regions


def _decode_arrow(points: list[tuple[int, int]]) -> str:
    pts = set(points)
    if len(pts) != 4:
        raise AssertionError("each region must contain one 4-pixel azure arrow")

    rows = sorted({r for r, _ in pts})
    for r in rows:
        cols = sorted(c for rr, c in pts if rr == r)
        for i in range(len(cols) - 2):
            run = cols[i : i + 3]
            if run[1] == run[0] + 1 and run[2] == run[1] + 1:
                end_c = run[2]
                if (r - 1, end_c) in pts:
                    return "up"
                if (r + 1, end_c) in pts:
                    return "down"

    cols = sorted({c for _, c in pts})
    for c in cols:
        rows_here = sorted(r for r, cc in pts if cc == c)
        for i in range(len(rows_here) - 2):
            run = rows_here[i : i + 3]
            if run[1] == run[0] + 1 and run[2] == run[1] + 1:
                end_r = run[2]
                if (end_r, c - 1) in pts:
                    return "left"
                if (end_r, c + 1) in pts:
                    return "right"

    raise AssertionError("cannot decode L-shaped arrow direction")


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _neighbor_index(regions: list[dict], idx: int, direction: str) -> int | None:
    br0, bc0, br1, bc1 = regions[idx]["bbox"]
    candidates: list[tuple[int, int]] = []
    for j, reg in enumerate(regions):
        if j == idx:
            continue
        nr0, nc0, nr1, nc1 = reg["bbox"]
        if direction == "up" and nr1 == br0:
            overlap = _overlap_len(bc0, bc1, nc0, nc1)
        elif direction == "down" and nr0 == br1:
            overlap = _overlap_len(bc0, bc1, nc0, nc1)
        elif direction == "left" and nc1 == bc0:
            overlap = _overlap_len(br0, br1, nr0, nr1)
        elif direction == "right" and nc0 == bc1:
            overlap = _overlap_len(br0, br1, nr0, nr1)
        else:
            overlap = 0
        if overlap >= 2:
            candidates.append((overlap, j))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the special border color through adjacent rectangular regions by following each region's L-shaped arrow.

    Input structure:
    - The background is black, and grey one-pixel lines form rectangular closed regions whose interiors contain black cells and one azure arrow.
    - Exactly one region has its complete rectangular border recolored from grey to a single non-grey color.
    - Each azure arrow is a 4-pixel L shape: a 3-pixel horizontal or vertical stem plus one side pixel at the stem's terminal end.

    Rule:
    - Find the uniquely all-non-grey bordered start region and remember its border color.
    - Decode the arrow inside the current region, move to the region sharing the border in that direction, and recolor that neighbor's whole rectangular border with the start color.
    - Continue reading arrows and recoloring the next neighbor until the arrow points off the region grid; all unvisited region borders remain grey.

    Key insight: The azure L geometry gives a local direction, and the only special border supplies the color that is carried along the directed chain of neighboring panes.
    """
    g = np.asarray(grid, dtype=int)
    regions = _find_regions(g)
    assert regions, "expected at least one enclosed rectangular region"

    arrows: list[str] = []
    start_candidates: list[tuple[int, int]] = []
    for i, reg in enumerate(regions):
        arrow_pts = [(r, c) for r, c in reg["interior"] if int(g[r, c]) == AZURE]
        arrows.append(_decode_arrow(arrow_pts))

        border_vals = [int(g[r, c]) for r, c in reg["border"]]
        first = border_vals[0]
        if first not in (BLACK, GREY, AZURE) and all(v == first for v in border_vals):
            start_candidates.append((i, first))

    assert len(start_candidates) == 1, "expected exactly one uniformly special bordered start region"
    current, color = start_candidates[0]

    out = g.copy()
    seen: set[int] = set()
    while current not in seen:
        seen.add(current)
        nxt = _neighbor_index(regions, current, arrows[current])
        if nxt is None:
            break
        for r, c in regions[nxt]["border"]:
            out[r, c] = color
        current = nxt
    return out


def _line_positions(lengths: list[int]) -> list[int]:
    pos = 0
    lines = [0]
    for length in lengths:
        pos += length + 1
        lines.append(pos)
    return lines


def _region_boxes(row_heights: list[int], col_widths: list[int]) -> tuple[list[int], list[int], dict[tuple[int, int], tuple[int, int, int, int]]]:
    r_lines = _line_positions(row_heights)
    c_lines = _line_positions(col_widths)
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(len(row_heights)):
        for c in range(len(col_widths)):
            boxes[(r, c)] = (r_lines[r], c_lines[c], r_lines[r + 1], c_lines[c + 1])
    return r_lines, c_lines, boxes


def _paint_region_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_arrow(grid: np.ndarray, box: tuple[int, int, int, int], direction: str) -> None:
    r0, c0, r1, c1 = box
    top, left = r0 + 1, c0 + 1
    bottom, right = r1 - 1, c1 - 1

    if direction in ("up", "down"):
        start_c = random.randint(left, right - 2)
        if direction == "up":
            row = random.randint(top + 1, bottom)
            side = (row - 1, start_c + 2)
        else:
            row = random.randint(top, bottom - 1)
            side = (row + 1, start_c + 2)
        grid[row, start_c : start_c + 3] = AZURE
        grid[side] = AZURE
    else:
        start_r = random.randint(top, bottom - 2)
        if direction == "left":
            col = random.randint(left + 1, right)
            side = (start_r + 2, col - 1)
        else:
            col = random.randint(left, right - 1)
            side = (start_r + 2, col + 1)
        grid[start_r : start_r + 3, col] = AZURE
        grid[side] = AZURE


def _inside_cell(cell: tuple[int, int], nr: int, nc: int) -> bool:
    r, c = cell
    return 0 <= r < nr and 0 <= c < nc


def _move_cell(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    r, c = cell
    return r + dr, c + dc


def _outside_dirs(cell: tuple[int, int], nr: int, nc: int) -> list[str]:
    return [name for name in DIRS if not _inside_cell(_move_cell(cell, name), nr, nc)]


def _make_path(nr: int, nc: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str]] | None:
    total = nr * nc
    for _ in range(300):
        start = (random.randrange(nr), random.randrange(nc))
        target_len = random.randint(1, total)
        path = [start]
        dirs: dict[tuple[int, int], str] = {}
        seen = {start}
        cur = start

        for _step in range(total + 4):
            exits = _outside_dirs(cur, nr, nc)
            if exits and len(path) >= target_len:
                dirs[cur] = random.choice(exits)
                return path, dirs

            choices = []
            names = list(DIRS)
            random.shuffle(names)
            for name in names:
                nxt = _move_cell(cur, name)
                if _inside_cell(nxt, nr, nc) and nxt not in seen:
                    choices.append((name, nxt))

            if not choices:
                if exits:
                    dirs[cur] = random.choice(exits)
                    return path, dirs
                break

            name, nxt = random.choice(choices)
            dirs[cur] = name
            cur = nxt
            path.append(cur)
            seen.add(cur)
    return None


def _expected_from_path(
    inp: np.ndarray,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    start: tuple[int, int],
    dirs: dict[tuple[int, int], str],
    color: int,
    nr: int,
    nc: int,
) -> np.ndarray:
    out = inp.copy()
    cur = start
    seen: set[tuple[int, int]] = set()
    while cur not in seen:
        seen.add(cur)
        nxt = _move_cell(cur, dirs[cur])
        if not _inside_cell(nxt, nr, nc):
            break
        _paint_region_border(out, boxes[nxt], color)
        cur = nxt
    return out


def _gen_attempt() -> np.ndarray | None:
    nr = random.randint(2, 4)
    nc = random.randint(2, 4)
    row_heights = [random.randint(3, 6) for _ in range(nr)]
    col_widths = [random.randint(3, 6) for _ in range(nc)]
    height = sum(row_heights) + nr + 1
    width = sum(col_widths) + nc + 1
    if height > 30 or width > 30:
        return None

    r_lines, c_lines, boxes = _region_boxes(row_heights, col_widths)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[r_lines, :] = GREY
    grid[:, c_lines] = GREY

    path_info = _make_path(nr, nc)
    if path_info is None:
        return None
    path, path_dirs = path_info
    start = path[0]
    special = random.choice([1, 2, 3, 4, 6, 7, 9])
    _paint_region_border(grid, boxes[start], special)

    for cell, box in boxes.items():
        direction = path_dirs.get(cell, random.choice(list(DIRS)))
        _draw_arrow(grid, box, direction)

    expected = _expected_from_path(grid, boxes, start, path_dirs, special, nr, nc)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-propagation grid")
