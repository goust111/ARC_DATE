import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _border_bg(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                out[rr, cc] = True
    return out


def _legend_candidates(g: np.ndarray, bg: int) -> list[tuple[int, str, int, int]]:
    h, w = g.shape
    candidates: list[tuple[int, str, int, int]] = []

    def isolated(cells: list[tuple[int, int]]) -> bool:
        cellset = set(cells)
        for r, c in cells:
            if int(g[r, c]) == bg:
                return False
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cellset and int(g[rr, cc]) != bg:
                    return False
        return True

    if h >= 2:
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                cargo = [int(x) for x in g[0, c0:c1].tolist()]
                pipe = [int(x) for x in g[1, c0:c1].tolist()]
                if len(set(cargo)) != len(cargo) or len(set(pipe)) != len(pipe):
                    continue
                cells = [(0, c) for c in range(c0, c1)] + [(1, c) for c in range(c0, c1)]
                if isolated(cells):
                    candidates.append((c1 - c0, "top", c0, c1))
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                cargo = [int(x) for x in g[h - 1, c0:c1].tolist()]
                pipe = [int(x) for x in g[h - 2, c0:c1].tolist()]
                if len(set(cargo)) != len(cargo) or len(set(pipe)) != len(pipe):
                    continue
                cells = [(h - 1, c) for c in range(c0, c1)] + [(h - 2, c) for c in range(c0, c1)]
                if isolated(cells):
                    candidates.append((c1 - c0, "bottom", c0, c1))

    if w >= 2:
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                cargo = [int(x) for x in g[r0:r1, 0].tolist()]
                pipe = [int(x) for x in g[r0:r1, 1].tolist()]
                if len(set(cargo)) != len(cargo) or len(set(pipe)) != len(pipe):
                    continue
                cells = [(r, 0) for r in range(r0, r1)] + [(r, 1) for r in range(r0, r1)]
                if isolated(cells):
                    candidates.append((r1 - r0, "left", r0, r1))
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                cargo = [int(x) for x in g[r0:r1, w - 1].tolist()]
                pipe = [int(x) for x in g[r0:r1, w - 2].tolist()]
                if len(set(cargo)) != len(cargo) or len(set(pipe)) != len(pipe):
                    continue
                cells = [(r, w - 1) for r in range(r0, r1)] + [(r, w - 2) for r in range(r0, r1)]
                if isolated(cells):
                    candidates.append((r1 - r0, "right", r0, r1))

    return candidates


def _find_legend(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    candidates = _legend_candidates(g, bg)
    assert candidates, "expected an isolated edge legend"
    _, side, start, end = max(candidates, key=lambda x: (x[0], x[1]))

    h, w = g.shape
    mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        mask[0:2, start:end] = True
        cargo_line = g[0, start:end]
        pipe_line = g[1, start:end]
    elif side == "bottom":
        mask[h - 2 : h, start:end] = True
        cargo_line = g[h - 1, start:end]
        pipe_line = g[h - 2, start:end]
    elif side == "left":
        mask[start:end, 0:2] = True
        cargo_line = g[start:end, 0]
        pipe_line = g[start:end, 1]
    else:
        mask[start:end, w - 2 : w] = True
        cargo_line = g[start:end, w - 1]
        pipe_line = g[start:end, w - 2]

    for cargo, pipe in zip(cargo_line.tolist(), pipe_line.tolist(), strict=False):
        mapping[int(cargo)] = int(pipe)
    return mapping, mask


def _warehouse_fill_order(r0: int, c0: int, side: str) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)]
    mid_r = r0 + 1
    mid_c = c0 + 1
    if side == "top":
        return sorted(cells, key=lambda p: (p[0], abs(p[1] - mid_c), p[1]))
    if side == "bottom":
        return sorted(cells, key=lambda p: (-p[0], abs(p[1] - mid_c), p[1]))
    if side == "left":
        return sorted(cells, key=lambda p: (p[1], abs(p[0] - mid_r), p[0]))
    return sorted(cells, key=lambda p: (-p[1], abs(p[0] - mid_r), p[0]))


def _find_pipe_infos(g: np.ndarray, legend_mask: np.ndarray, bg: int) -> dict[int, dict]:
    h, w = g.shape
    color_cells: dict[int, set[tuple[int, int]]] = {}
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        pts = {
            (int(r), int(c))
            for r, c in np.argwhere((g == color) & (~legend_mask))
        }
        if pts:
            color_cells[color] = pts

    infos: dict[int, dict] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if legend_mask[r0 : r0 + 3, c0 : c0 + 3].any():
                continue
            color = int(g[r0, c0])
            if color == bg:
                continue
            if not bool((g[r0 : r0 + 3, c0 : c0 + 3] == color).all()):
                continue

            exits: list[tuple[str, tuple[int, int]]] = []
            if r0 - 1 >= 0 and not legend_mask[r0 - 1, c0 + 1] and int(g[r0 - 1, c0 + 1]) == color:
                exits.append(("top", (r0 - 1, c0 + 1)))
            if r0 + 3 < h and not legend_mask[r0 + 3, c0 + 1] and int(g[r0 + 3, c0 + 1]) == color:
                exits.append(("bottom", (r0 + 3, c0 + 1)))
            if c0 - 1 >= 0 and not legend_mask[r0 + 1, c0 - 1] and int(g[r0 + 1, c0 - 1]) == color:
                exits.append(("left", (r0 + 1, c0 - 1)))
            if c0 + 3 < w and not legend_mask[r0 + 1, c0 + 3] and int(g[r0 + 1, c0 + 3]) == color:
                exits.append(("right", (r0 + 1, c0 + 3)))
            if len(exits) != 1 or color in infos:
                continue

            warehouse = {(rr, cc) for rr in range(r0, r0 + 3) for cc in range(c0, c0 + 3)}
            cells = color_cells.get(color, set())
            path_cells = cells - warehouse
            if exits[0][1] not in path_cells:
                continue
            if not path_cells:
                continue

            open_ends = []
            for rr, cc in path_cells:
                deg = 0
                for dr, dc in DIR4:
                    if (rr + dr, cc + dc) in cells:
                        deg += 1
                if deg == 1:
                    open_ends.append((rr, cc))
            if len(open_ends) != 1:
                continue

            side = exits[0][0]
            infos[color] = {
                "warehouse_top_left": (r0, c0),
                "warehouse_cells": warehouse,
                "entry_side": side,
                "open_end": open_ends[0],
                "fill_order": _warehouse_fill_order(r0, c0, side),
            }

    return infos


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _border_bg(g)
    mapping, legend_mask = _find_legend(g, bg)
    pipes = _find_pipe_infos(g, legend_mask, bg)
    assert pipes, "expected at least one pipe-and-warehouse structure"

    out = g.copy()
    by_pipe: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)

    for cargo_color, pipe_color in mapping.items():
        if pipe_color not in pipes:
            continue
        mouth_r, mouth_c = pipes[pipe_color]["open_end"]
        for r, c in np.argwhere((g == cargo_color) & (~legend_mask)):
            r = int(r)
            c = int(c)
            by_pipe[pipe_color].append((abs(r - mouth_r) + abs(c - mouth_c), r, c, cargo_color))
            out[r, c] = bg

    for pipe_color, payloads in by_pipe.items():
        payloads.sort()
        slots = pipes[pipe_color]["fill_order"]
        assert len(payloads) <= len(slots), "warehouse capacity exceeded"
        for idx, (_, _, _, cargo_color) in enumerate(payloads):
            rr, cc = slots[idx]
            out[rr, cc] = cargo_color

    return out


def _random_shortest_path(
    start: tuple[int, int], goal: tuple[int, int], blocked: np.ndarray
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        nbrs = list(DIR4)
        random.shuffle(nbrs)
        for dr, dc in nbrs:
            nr = cur[0] + dr
            nc = cur[1] + dc
            nxt = (nr, nc)
            if not _in_bounds(h, w, nr, nc) or blocked[nr, nc] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if goal not in prev:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _legend_layout(
    h: int,
    w: int,
    pairs: list[tuple[int, int]],
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((h, w), dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    n = len(pairs)
    if side in ("top", "bottom"):
        start = random.randint(1, w - n - 1)
        for i, (cargo, pipe) in enumerate(pairs):
            c = start + i
            if side == "top":
                grid[0, c] = cargo
                grid[1, c] = pipe
                mask[0:2, c] = True
            else:
                grid[h - 1, c] = cargo
                grid[h - 2, c] = pipe
                mask[h - 2 : h, c] = True
    else:
        start = random.randint(1, h - n - 1)
        for i, (cargo, pipe) in enumerate(pairs):
            r = start + i
            if side == "left":
                grid[r, 0] = cargo
                grid[r, 1] = pipe
                mask[r, 0:2] = True
            else:
                grid[r, w - 1] = cargo
                grid[r, w - 2] = pipe
                mask[r, w - 2 : w] = True
    return grid, mask


def _side_geometry(r0: int, c0: int, side: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (r0 - 1, c0 + 1), (r0 - 2, c0 + 1), (-1, 0)
    if side == "bottom":
        return (r0 + 3, c0 + 1), (r0 + 4, c0 + 1), (1, 0)
    if side == "left":
        return (r0 + 1, c0 - 1), (r0 + 1, c0 - 2), (0, -1)
    return (r0 + 1, c0 + 3), (r0 + 1, c0 + 4), (0, 1)


def _cargo_cells_for_mouth(
    h: int,
    w: int,
    mouth: tuple[int, int],
    direction: tuple[int, int],
    reserved: np.ndarray,
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    pr, pc = -dc, dr
    offsets = [
        (dr, dc),
        (dr + pr, dc + pc),
        (dr - pr, dc - pc),
    ]
    patterns = {
        1: [[0]],
        2: [[0, 1], [0, 2]],
        3: [[0, 1, 2]],
    }
    size = random.choice([1, 2, 3])
    pattern_list = patterns[size][:]
    random.shuffle(pattern_list)

    for pat in pattern_list:
        cells = []
        ok = True
        for idx in pat:
            rr = mouth[0] + offsets[idx][0]
            cc = mouth[1] + offsets[idx][1]
            if not _in_bounds(h, w, rr, cc) or reserved[rr, cc]:
                ok = False
                break
            cells.append((rr, cc))
        if ok:
            return cells
    return None


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _try_place_pipe(
    grid: np.ndarray,
    used_mask: np.ndarray,
    pipe_color: int,
    cargo_color: int,
) -> bool:
    h, w = grid.shape
    blocked_existing = _dilate8(used_mask)
    free_mouth_cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]

    for _ in range(180):
        side = random.choice(SIDES)

        r_min = 1
        r_max = h - 4
        c_min = 1
        c_max = w - 4
        if side == "top":
            r_min = 2
        elif side == "bottom":
            r_max = h - 5
        elif side == "left":
            c_min = 2
        else:
            c_max = w - 5
        if r_min > r_max or c_min > c_max:
            continue

        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        warehouse = {(rr, cc) for rr in range(r0, r0 + 3) for cc in range(c0, c0 + 3)}
        if any(blocked_existing[rr, cc] for rr, cc in warehouse):
            continue

        entry, anchor, _ = _side_geometry(r0, c0, side)
        if not _in_bounds(h, w, anchor[0], anchor[1]) or blocked_existing[entry] or blocked_existing[anchor]:
            continue

        local_blocked = _dilate8(used_mask)
        for rr, cc in warehouse:
            local_blocked[rr, cc] = True
        local_blocked = _dilate8(local_blocked)
        local_blocked[anchor] = False

        mouths = free_mouth_cells[:]
        random.shuffle(mouths)
        for mouth in mouths[:80]:
            if local_blocked[mouth]:
                continue
            if abs(mouth[0] - anchor[0]) + abs(mouth[1] - anchor[1]) < 3:
                continue
            path_tail = _random_shortest_path(anchor, mouth, local_blocked)
            if path_tail is None or len(path_tail) < 2:
                continue
            path = [entry] + path_tail
            prev = path[-2]
            direction = (mouth[0] - prev[0], mouth[1] - prev[1])
            reserved = used_mask.copy()
            _paint_cells(reserved, list(warehouse), True)
            _paint_cells(reserved, path, True)
            cargo_cells = _cargo_cells_for_mouth(h, w, mouth, direction, reserved)
            if cargo_cells is None:
                continue

            _paint_cells(grid, warehouse, pipe_color)
            _paint_cells(grid, path, pipe_color)
            _paint_cells(grid, cargo_cells, cargo_color)
            _paint_cells(used_mask, warehouse, True)
            _paint_cells(used_mask, path, True)
            _paint_cells(used_mask, cargo_cells, True)
            return True

    return False


def _validate_generated_grid(g: np.ndarray, expected_pairs: dict[int, int]) -> bool:
    try:
        bg = _border_bg(g)
        mapping, legend_mask = _find_legend(g, bg)
        if mapping != expected_pairs:
            return False
        pipes = _find_pipe_infos(g, legend_mask, bg)
        if set(pipes) != set(mapping.values()):
            return False
        for cargo_color, pipe_color in mapping.items():
            pts = [(int(r), int(c)) for r, c in np.argwhere((g == cargo_color) & (~legend_mask))]
            if not (1 <= len(pts) <= 3):
                return False
            mouth = pipes[pipe_color]["open_end"]
            if min(abs(r - mouth[0]) + abs(c - mouth[1]) for r, c in pts) > 2:
                return False
        out = main(g)
        if np.array_equal(out, g):
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_pipes = random.choice([2, 3, 4])
        h = random.randint(14 + 2 * n_pipes, 20 + 2 * n_pipes)
        w = random.randint(14 + 2 * n_pipes, 20 + 2 * n_pipes)
        h = min(h, 30)
        w = min(w, 30)

        palette = random.sample(list(range(1, 10)), 2 * n_pipes)
        pipe_colors = palette[:n_pipes]
        cargo_colors = palette[n_pipes:]
        shuffled_pipes = pipe_colors[:]
        random.shuffle(shuffled_pipes)
        mapping = {cargo: pipe for cargo, pipe in zip(cargo_colors, shuffled_pipes, strict=False)}

        legend_pairs = list(mapping.items())
        random.shuffle(legend_pairs)
        legend_side = random.choice(SIDES)
        grid, legend_mask = _legend_layout(h, w, legend_pairs, legend_side)
        used_mask = legend_mask.copy()

        ok = True
        for cargo_color, pipe_color in legend_pairs:
            if not _try_place_pipe(grid, used_mask, pipe_color, cargo_color):
                ok = False
                break
        if not ok:
            continue

        if _validate_generated_grid(grid, mapping):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
