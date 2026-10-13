import random
from typing import Iterable

import numpy as np


BLACK = 0
GRAY = 5
MAROON = 9
EMITTER_COLORS = [1, 2, 3, 4, 6, 7, 8]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = [UP, DOWN, LEFT, RIGHT]

TURN_LEFT = {
    UP: LEFT,
    LEFT: DOWN,
    DOWN: RIGHT,
    RIGHT: UP,
}
TURN_RIGHT = {
    UP: RIGHT,
    RIGHT: DOWN,
    DOWN: LEFT,
    LEFT: UP,
}
REFLECT = {
    ("\\", RIGHT): DOWN,
    ("\\", DOWN): RIGHT,
    ("\\", LEFT): UP,
    ("\\", UP): LEFT,
    ("/", RIGHT): UP,
    ("/", UP): RIGHT,
    ("/", LEFT): DOWN,
    ("/", DOWN): LEFT,
}

GEN_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(cells)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _emitter_from_cells(color: int, cells: list[tuple[int, int]]) -> dict | None:
    if len(cells) != 3:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    pts = set(cells)
    missing = list(corners - pts)
    if len(missing) != 1:
        return None
    deg = {}
    for r, c in pts:
        cnt = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in pts:
                cnt += 1
        deg[(r, c)] = cnt
    elbows = [p for p, d in deg.items() if d == 2]
    if len(elbows) != 1:
        return None
    start = elbows[0]
    mr, mc = missing[0]
    if (mr, mc) == (r0, c1):
        direction = RIGHT
    elif (mr, mc) == (r0, c0):
        direction = LEFT
    elif (mr, mc) == (r1, c1):
        direction = DOWN
    else:
        direction = UP
    return {
        "color": int(color),
        "cells": pts,
        "start": (int(start[0]), int(start[1])),
        "direction": direction,
    }


def _mirror_orientation(cells: list[tuple[int, int]]) -> str | None:
    if not (2 <= len(cells) <= 4):
        return None
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) != len(cells) or len(cols) != len(cells):
        return None
    pts = sorted(cells)
    if len({r - c for r, c in pts}) == 1:
        if all(pts[i + 1][0] - pts[i][0] == 1 and pts[i + 1][1] - pts[i][1] == 1 for i in range(len(pts) - 1)):
            return "\\"
    if len({r + c for r, c in pts}) == 1:
        if all(pts[i + 1][0] - pts[i][0] == 1 and pts[i + 1][1] - pts[i][1] == -1 for i in range(len(pts) - 1)):
            return "/"
    return None


def _target_blocks(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    out: list[set[tuple[int, int]]] = []
    for comp in _components(grid == MAROON, connectivity=4):
        if len(comp) != 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        block = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if set(comp) == block:
            out.append(block)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    emitters: list[dict] = []
    emitter_owner: dict[tuple[int, int], int] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GRAY, MAROON):
            continue
        for comp in _components(g == color, connectivity=4):
            emitter = _emitter_from_cells(color, comp)
            if emitter is None:
                continue
            emitter["id"] = len(emitters)
            emitters.append(emitter)
            for cell in emitter["cells"]:
                emitter_owner[cell] = emitter["id"]

    mirror_lookup: dict[tuple[int, int], str] = {}
    for comp in _components(g == GRAY, connectivity=8):
        orient = _mirror_orientation(comp)
        if orient is None:
            continue
        for cell in comp:
            mirror_lookup[cell] = orient

    targets = _target_blocks(g)
    target_lookup: dict[tuple[int, int], int] = {}
    for idx, block in enumerate(targets):
        for cell in block:
            target_lookup[cell] = idx

    for emitter in emitters:
        color = int(emitter["color"])
        cur = emitter["start"]
        direction = emitter["direction"]
        seen: set[tuple[int, int, int, int]] = set()
        while True:
            state = (cur[0], cur[1], direction[0], direction[1])
            if state in seen:
                break
            seen.add(state)
            nr = cur[0] + direction[0]
            nc = cur[1] + direction[1]
            if not (0 <= nr < h and 0 <= nc < w):
                break
            nxt = (int(nr), int(nc))
            if nxt in target_lookup:
                for tr, tc in targets[target_lookup[nxt]]:
                    out[tr, tc] = color
                break
            if nxt in mirror_lookup:
                out[nr, nc] = color
                cur = nxt
                direction = REFLECT[(mirror_lookup[nxt], direction)]
                continue
            owner = emitter_owner.get(nxt)
            if owner is not None and owner != emitter["id"]:
                break
            if g[nr, nc] == BLACK:
                out[nr, nc] = color
                cur = nxt
                continue
            if owner == emitter["id"]:
                cur = nxt
                continue
            break
    return out


def _distance_to_edge(pos: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = pos
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _emitter_cells_from_start(start: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = start
    if direction == RIGHT:
        rel = [(0, 0), (-1, 0), (0, 1)]
    elif direction == LEFT:
        rel = [(0, 0), (-1, 0), (0, -1)]
    elif direction == DOWN:
        rel = [(0, 0), (0, 1), (1, 0)]
    else:
        rel = [(0, 0), (0, -1), (1, 0)]
    return {(r + dr, c + dc) for dr, dc in rel}


def _start_candidates(h: int, w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if direction == RIGHT:
        cols = [c for c in range(1, min(4, w - 2))]
        for r in range(1, h - 1):
            for c in cols:
                out.append((r, c))
    elif direction == LEFT:
        cols = [c for c in range(max(1, w - 4), w - 1)]
        for r in range(1, h - 1):
            for c in cols:
                out.append((r, c))
    elif direction == DOWN:
        rows = [r for r in range(1, min(4, h - 2))]
        for r in rows:
            for c in range(1, w - 1):
                out.append((r, c))
    else:
        rows = [r for r in range(max(1, h - 4), h - 1)]
        for r in rows:
            for c in range(1, w - 1):
                out.append((r, c))
    return out


def _turn_orientation(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> str:
    for orient, idir in REFLECT:
        if idir == in_dir and REFLECT[(orient, in_dir)] == out_dir:
            return orient
    raise ValueError("invalid turn")


def _mirror_shapes(
    contact: tuple[int, int],
    orient: str,
    h: int,
    w: int,
) -> list[set[tuple[int, int]]]:
    if orient == "\\":
        d1, d2 = (-1, -1), (1, 1)
    else:
        d1, d2 = (-1, 1), (1, -1)
    seen: set[frozenset[tuple[int, int]]] = set()
    out: list[set[tuple[int, int]]] = []
    for a in range(4):
        for b in range(4):
            total = 1 + a + b
            if total < 2 or total > 4:
                continue
            cells = {contact}
            ok = True
            for step in range(1, a + 1):
                rr = contact[0] + d1[0] * step
                cc = contact[1] + d1[1] * step
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                cells.add((rr, cc))
            if not ok:
                continue
            for step in range(1, b + 1):
                rr = contact[0] + d2[0] * step
                cc = contact[1] + d2[1] * step
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                cells.add((rr, cc))
            if not ok:
                continue
            key = frozenset(cells)
            if key not in seen:
                seen.add(key)
                out.append(set(cells))
    return out


def _target_options(hit: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[set[tuple[int, int]]]:
    r, c = hit
    candidates: list[set[tuple[int, int]]] = []
    if direction == RIGHT:
        candidates = [
            {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)},
            {(r - 1, c), (r - 1, c + 1), (r, c), (r, c + 1)},
        ]
    elif direction == LEFT:
        candidates = [
            {(r, c - 1), (r, c), (r + 1, c - 1), (r + 1, c)},
            {(r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)},
        ]
    elif direction == DOWN:
        candidates = [
            {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)},
            {(r, c - 1), (r, c), (r + 1, c - 1), (r + 1, c)},
        ]
    else:
        candidates = [
            {(r - 1, c), (r - 1, c + 1), (r, c), (r, c + 1)},
            {(r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)},
        ]
    out: list[set[tuple[int, int]]] = []
    for block in candidates:
        if all(0 <= rr < h and 0 <= cc < w for rr, cc in block):
            out.append(block)
    return out


def _touches_existing_gray(candidate: set[tuple[int, int]], gray_cells: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in gray_cells and (rr, cc) not in candidate:
                    return True
    return False


def _plan_tile(rng: random.Random, h: int, w: int, color: int) -> dict | None:
    direction = rng.choice(DIRS)
    start_cands = _start_candidates(h, w, direction)
    rng.shuffle(start_cands)
    n_reflections = rng.randint(0, 3)

    for start in start_cands:
        emitter_cells = _emitter_cells_from_start(start, direction)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in emitter_cells):
            continue

        def dfs(
            pos: tuple[int, int],
            cur_dir: tuple[int, int],
            stage: int,
            ray_cells: set[tuple[int, int]],
            mirrors: list[dict],
            gray_cells: set[tuple[int, int]],
        ) -> dict | None:
            dist = _distance_to_edge(pos, cur_dir, h, w)
            max_len = dist - 1
            if max_len < 2:
                return None
            lengths = list(range(2, max_len + 1))
            rng.shuffle(lengths)

            for length in lengths:
                traverse = [(pos[0] + cur_dir[0] * step, pos[1] + cur_dir[1] * step) for step in range(1, length + 1)]
                hit = (pos[0] + cur_dir[0] * (length + 1), pos[1] + cur_dir[1] * (length + 1))
                if any(cell in ray_cells or cell in gray_cells for cell in traverse):
                    continue
                if stage > 0 and any(cell in emitter_cells for cell in traverse):
                    continue

                new_ray = ray_cells | set(traverse)
                if stage == n_reflections:
                    targets = _target_options(hit, cur_dir, h, w)
                    rng.shuffle(targets)
                    for target in targets:
                        if any(cell in emitter_cells or cell in new_ray or cell in gray_cells for cell in target):
                            continue
                        black_painted = sum(1 for cell in new_ray if cell not in emitter_cells)
                        if black_painted < 2:
                            continue
                        return {
                            "color": int(color),
                            "emitter_cells": set(emitter_cells),
                            "start": start,
                            "direction": cur_dir if stage == 0 else direction,
                            "ray_cells": set(new_ray),
                            "mirrors": list(mirrors),
                            "target": set(target),
                        }
                    continue

                if hit in emitter_cells or hit in new_ray or hit in gray_cells:
                    continue

                next_dirs = [TURN_LEFT[cur_dir], TURN_RIGHT[cur_dir]]
                rng.shuffle(next_dirs)
                for out_dir in next_dirs:
                    orient = _turn_orientation(cur_dir, out_dir)
                    shapes = _mirror_shapes(hit, orient, h, w)
                    rng.shuffle(shapes)
                    for shape in shapes:
                        if any(cell in emitter_cells or cell in new_ray or cell in gray_cells for cell in shape):
                            continue
                        if _touches_existing_gray(shape, gray_cells):
                            continue
                        plan = dfs(
                            hit,
                            out_dir,
                            stage + 1,
                            new_ray,
                            mirrors + [{"cells": set(shape), "contact": hit, "orient": orient}],
                            gray_cells | set(shape),
                        )
                        if plan is not None:
                            return plan
            return None

        plan = dfs(start, direction, 0, set(), [], set())
        if plan is not None:
            return plan
    return None


def _layout_boxes(rng: random.Random, n_emitters: int) -> list[tuple[int, int, int, int]]:
    if n_emitters == 1:
        rows, cols = 1, 1
        slots = [(0, 0)]
    elif n_emitters == 2:
        if rng.random() < 0.5:
            rows, cols = 1, 2
            slots = [(0, 0), (0, 1)]
        else:
            rows, cols = 2, 1
            slots = [(0, 0), (1, 0)]
    elif n_emitters == 3:
        rows, cols = 2, 2
        slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
        rng.shuffle(slots)
        slots = slots[:3]
    else:
        rows, cols = 2, 2
        slots = [(0, 0), (0, 1), (1, 0), (1, 1)]

    row_heights = [rng.randint(10, 13) for _ in range(rows)]
    col_widths = [rng.randint(10, 13) for _ in range(cols)]
    gap = rng.randint(1, 2)
    margin = rng.randint(1, 2)

    row_starts = []
    cur = margin
    for h in row_heights:
        row_starts.append(cur)
        cur += h + gap

    col_starts = []
    cur = margin
    for w in col_widths:
        col_starts.append(cur)
        cur += w + gap

    boxes: list[tuple[int, int, int, int]] = []
    for sr, sc in slots:
        r0 = row_starts[sr]
        c0 = col_starts[sc]
        boxes.append((r0, c0, row_heights[sr], col_widths[sc]))
    return boxes


def _translate(plan: dict, r0: int, c0: int) -> dict:
    def shift(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r + r0, c + c0) for r, c in cells}

    return {
        "color": int(plan["color"]),
        "emitter_cells": shift(plan["emitter_cells"]),
        "ray_cells": shift(plan["ray_cells"]),
        "mirrors": [
            {
                "cells": shift(mirror["cells"]),
                "contact": (mirror["contact"][0] + r0, mirror["contact"][1] + c0),
                "orient": mirror["orient"],
            }
            for mirror in plan["mirrors"]
        ],
        "target": shift(plan["target"]),
    }


def _render(plans: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_cells: set[tuple[int, int]] = set()
    for plan in plans:
        all_cells |= plan["emitter_cells"]
        all_cells |= plan["ray_cells"]
        all_cells |= plan["target"]
        for mirror in plan["mirrors"]:
            all_cells |= mirror["cells"]

    max_r = max(r for r, _ in all_cells) + 2
    max_c = max(c for _, c in all_cells) + 2
    grid = np.full((max_r, max_c), BLACK, dtype=int)
    out = np.full((max_r, max_c), BLACK, dtype=int)

    for plan in plans:
        color = int(plan["color"])
        for r, c in plan["emitter_cells"]:
            grid[r, c] = color
            out[r, c] = color
        for mirror in plan["mirrors"]:
            for r, c in mirror["cells"]:
                grid[r, c] = GRAY
                out[r, c] = GRAY
        for r, c in plan["target"]:
            grid[r, c] = MAROON
            out[r, c] = MAROON
        for r, c in plan["ray_cells"]:
            out[r, c] = color
        for mirror in plan["mirrors"]:
            r, c = mirror["contact"]
            out[r, c] = color
        for r, c in plan["target"]:
            out[r, c] = color

    used_rows = np.where((grid != BLACK).any(axis=1))[0]
    used_cols = np.where((grid != BLACK).any(axis=0))[0]
    r0 = max(0, int(used_rows.min()) - 1)
    r1 = min(grid.shape[0], int(used_rows.max()) + 2)
    c0 = max(0, int(used_cols.min()) - 1)
    c1 = min(grid.shape[1], int(used_cols.max()) + 2)
    return grid[r0:r1, c0:c1], out[r0:r1, c0:c1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_emitters = rng.randint(1, 4)
    colors = rng.sample(EMITTER_COLORS, n_emitters)
    boxes = _layout_boxes(rng, n_emitters)
    plans: list[dict] = []
    for color, (r0, c0, h, w) in zip(colors, boxes):
        plan = _plan_tile(rng, h, w, color)
        if plan is None:
            return None
        plans.append(_translate(plan, r0, c0))

    grid, expected = _render(plans)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
