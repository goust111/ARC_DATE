from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
GREY = 5


@dataclass(frozen=True)
class Container:
    box: tuple[int, int, int, int]
    interior: tuple[int, int, int, int]
    side: str
    gap: tuple[tuple[int, int], ...]
    port: tuple[int, int]
    color: int


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _bounded_run(grid: np.ndarray, r: int, c: int, axis: str) -> tuple[int, int, int] | None:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w) or grid[r, c] != BG:
        return None
    if axis == "v":
        a = r
        while a - 1 >= 0 and grid[a - 1, c] == BG:
            a -= 1
        b = r
        while b + 1 < h and grid[b + 1, c] == BG:
            b += 1
        if a - 1 < 0 or b + 1 >= h:
            return None
        if grid[a - 1, c] != GREY or grid[b + 1, c] != GREY:
            return None
    else:
        a = c
        while a - 1 >= 0 and grid[r, a - 1] == BG:
            a -= 1
        b = c
        while b + 1 < w and grid[r, b + 1] == BG:
            b += 1
        if a - 1 < 0 or b + 1 >= w:
            return None
        if grid[r, a - 1] != GREY or grid[r, b + 1] != GREY:
            return None
    width = b - a + 1
    if width not in (1, 3):
        return None
    center = a + width // 2
    return a, b, center


def _h_width(grid: np.ndarray, r: int, c: int) -> int:
    run = _bounded_run(grid, r, c, "v")
    if run is None:
        return 0
    a, b, center = run
    return b - a + 1 if r == center else 0


def _v_width(grid: np.ndarray, r: int, c: int) -> int:
    run = _bounded_run(grid, r, c, "h")
    if run is None:
        return 0
    a, b, center = run
    return b - a + 1 if c == center else 0


def _is_pipe_center(grid: np.ndarray, r: int, c: int, ports: set[tuple[int, int]]) -> bool:
    return (r, c) in ports or _h_width(grid, r, c) > 0 or _v_width(grid, r, c) > 0


def _edge_width(grid: np.ndarray, r: int, c: int, d: str) -> int:
    if d in ("L", "R"):
        return _h_width(grid, r, c)
    return _v_width(grid, r, c)


def _find_containers(grid: np.ndarray) -> list[Container]:
    h, w = grid.shape
    containers: list[Container] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r1 in range(h - 3):
        for c1 in range(w - 3):
            if grid[r1, c1] != GREY:
                continue
            for r2 in range(r1 + 3, min(h, r1 + 9)):
                for c2 in range(c1 + 3, min(w, c1 + 9)):
                    box = (r1, c1, r2, c2)
                    if box in seen or grid[r2, c2] != GREY:
                        continue
                    gaps: list[tuple[str, list[tuple[int, int]]]] = []
                    ok = True
                    for side in ("T", "B", "L", "R"):
                        if side == "T":
                            cells = [(r1, c) for c in range(c1, c2 + 1)]
                        elif side == "B":
                            cells = [(r2, c) for c in range(c1, c2 + 1)]
                        elif side == "L":
                            cells = [(r, c1) for r in range(r1, r2 + 1)]
                        else:
                            cells = [(r, c2) for r in range(r1, r2 + 1)]
                        non_grey = [(r, c) for r, c in cells if grid[r, c] != GREY]
                        if not non_grey:
                            continue
                        idxs = [cells.index(x) for x in non_grey]
                        if len(non_grey) not in (1, 3) or idxs != list(range(idxs[0], idxs[0] + len(idxs))):
                            ok = False
                            break
                        if any(x in (0, len(cells) - 1) for x in idxs):
                            ok = False
                            break
                        gaps.append((side, non_grey))
                    if not ok or len(gaps) != 1:
                        continue
                    interior = grid[r1 + 1 : r2, c1 + 1 : c2]
                    if interior.size == 0 or np.any(interior == GREY):
                        continue
                    vals = [int(v) for v in np.unique(interior) if v not in (BG, GREY)]
                    color = vals[0] if len(vals) == 1 else BG
                    side, gap = gaps[0]
                    gr, gc = gap[len(gap) // 2]
                    if side == "T":
                        port = (gr - 1, gc)
                        side_name = "U"
                    elif side == "B":
                        port = (gr + 1, gc)
                        side_name = "D"
                    elif side == "L":
                        port = (gr, gc - 1)
                        side_name = "L"
                    else:
                        port = (gr, gc + 1)
                        side_name = "R"
                    if not (0 <= port[0] < h and 0 <= port[1] < w):
                        continue
                    seen.add(box)
                    containers.append(
                        Container(
                            box=box,
                            interior=(r1 + 1, c1 + 1, r2 - 1, c2 - 1),
                            side=side_name,
                            gap=tuple(gap),
                            port=port,
                            color=color,
                        )
                    )
    # Drop rectangles nested inside a larger detected frame.
    result: list[Container] = []
    for c in containers:
        r1, c1, r2, c2 = c.box
        if any(or1 <= r1 and oc1 <= c1 and r2 <= or2 and c2 <= oc2 and other.box != c.box for other in containers for or1, oc1, or2, oc2 in [other.box]):
            continue
        result.append(c)
    return result


def _fill_interior(out: np.ndarray, container: Container, color: int) -> None:
    r1, c1, r2, c2 = container.interior
    out[r1 : r2 + 1, c1 : c2 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    containers = _find_containers(grid)
    if not containers:
        return out

    sources = [c for c in containers if c.color not in (BG, GREY)]
    targets_by_port = {c.port: c for c in containers if c.color == BG}
    ports = {c.port for c in containers}

    for source in sources:
        start = source.port
        color = source.color
        if not _is_pipe_center(grid, *start, ports):
            continue
        queue = [(start[0], start[1], source.side)]
        seen: set[tuple[int, int, str]] = set()
        while queue:
            r, c, incoming_dir = queue.pop(0)
            state = (r, c, incoming_dir)
            if state in seen:
                continue
            seen.add(state)

            target = targets_by_port.get((r, c))
            if target is not None:
                _fill_interior(out, target, color)
                continue

            choices: list[tuple[str, int, int, int]] = []
            for d, (dr, dc) in DIRS.items():
                if d == OPP[incoming_dir]:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                    continue
                if _is_pipe_center(grid, nr, nc, ports):
                    width = _edge_width(grid, nr, nc, d)
                    if (nr, nc) in ports:
                        width = max(width, 1)
                    if width:
                        choices.append((d, nr, nc, width))
            if not choices:
                continue
            max_width = max(width for _, _, _, width in choices)
            for d, nr, nc, width in choices:
                if width == max_width:
                    queue.append((nr, nc, d))
    return out


def _draw_container(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int, side: str, gap_width: int, color: int = BG) -> None:
    grid[r1, c1 : c2 + 1] = GREY
    grid[r2, c1 : c2 + 1] = GREY
    grid[r1 : r2 + 1, c1] = GREY
    grid[r1 : r2 + 1, c2] = GREY
    grid[r1 + 1 : r2, c1 + 1 : c2] = color
    if side in ("U", "D"):
        center = (c1 + c2) // 2
        row = r1 if side == "U" else r2
        grid[row, center - gap_width // 2 : center + gap_width // 2 + 1] = BG
    else:
        center = (r1 + r2) // 2
        col = c1 if side == "L" else c2
        grid[center - gap_width // 2 : center + gap_width // 2 + 1, col] = BG


def _draw_h_pipe(grid: np.ndarray, row: int, c1: int, c2: int, width: int) -> None:
    half = width // 2
    grid[row - half - 1, c1 : c2 + 1] = GREY
    grid[row + half + 1, c1 : c2 + 1] = GREY
    grid[row - half : row + half + 1, c1 : c2 + 1] = BG


def _draw_v_pipe(grid: np.ndarray, col: int, r1: int, r2: int, width: int) -> None:
    half = width // 2
    grid[r1 : r2 + 1, col - half - 1] = GREY
    grid[r1 : r2 + 1, col + half + 1] = GREY
    grid[r1 : r2 + 1, col - half : col + half + 1] = BG


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    color = random.choice(colors)
    grid = np.zeros((30, 30), dtype=int)

    # One source feeds a tree with three outgoing branches. Branch widths are
    # visible in the grey pipe walls; max-width branches receive the color.
    up_w, right_w, down_w = [random.choice([1, 3]) for _ in range(3)]
    if random.random() < 0.25:
        up_w = right_w = down_w = 1

    _draw_container(grid, 12, 2, 16, 6, "R", 1, color)
    _draw_container(grid, 2, 13, 6, 17, "D", up_w, BG)
    _draw_container(grid, 12, 23, 16, 27, "L", right_w, BG)
    _draw_container(grid, 23, 13, 27, 17, "U", down_w, BG)

    _draw_h_pipe(grid, 14, 7, 15, 1)
    _draw_h_pipe(grid, 14, 15, 22, right_w)
    _draw_v_pipe(grid, 15, 7, 14, up_w)
    _draw_v_pipe(grid, 15, 14, 22, down_w)

    # Restore container walls after pipe drawing, preserving their openings.
    _draw_container(grid, 12, 2, 16, 6, "R", 1, color)
    _draw_container(grid, 2, 13, 6, 17, "D", up_w, BG)
    _draw_container(grid, 12, 23, 16, 27, "L", right_w, BG)
    _draw_container(grid, 23, 13, 27, 17, "U", down_w, BG)
    return grid

