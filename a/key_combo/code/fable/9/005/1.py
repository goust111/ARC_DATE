from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
MAGENTA = 6

DIAG_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _sign(x: int) -> int:
    return 1 if x > 0 else -1


def _diag_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = _sign(b[0] - a[0])
    dc = _sign(b[1] - a[1])
    steps = abs(b[0] - a[0])
    return [(a[0] + k * dr, a[1] + k * dc) for k in range(steps + 1)]


def _component_coords(mask: np.ndarray) -> list[set[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> bool:
    bset = set(b)
    for r, c in a:
        for dr, dc in ((0, 0),) + NEIGHBORS8:
            if (r + dr, c + dc) in bset:
                return True
    return False


def _red_blocks(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    comps = _component_coords(grid == RED)
    good: list[set[tuple[int, int]]] = []
    for comp in comps:
        rows = sorted(r for r, _ in comp)
        cols = sorted(c for _, c in comp)
        if len(comp) == 4 and rows[-1] - rows[0] == 1 and cols[-1] - cols[0] == 1:
            good.append(comp)
    assert len(good) == 2, "expected exactly two 2x2 red blocks"
    return good


def _segment_slope(a: tuple[int, int], b: tuple[int, int]) -> int:
    return 1 if (b[0] - a[0]) * (b[1] - a[1]) > 0 else -1


def _segments(grid: np.ndarray) -> list[dict]:
    blue_pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    blue_set = set(blue_pts)
    segs: list[dict] = []
    for i in range(len(blue_pts)):
        for j in range(i + 1, len(blue_pts)):
            a = blue_pts[i]
            b = blue_pts[j]
            dr = b[0] - a[0]
            dc = b[1] - a[1]
            if dr == 0 or abs(dr) != abs(dc):
                continue
            path = _diag_path(a, b)
            if any(cell in blue_set for cell in path[1:-1]):
                continue
            segs.append(
                {
                    "kind": "segment",
                    "color": BLUE,
                    "cells": set(path),
                    "path": path,
                    "endpoints": (a, b),
                    "slope": _segment_slope(a, b),
                }
            )
    assert segs, "expected at least one blue diagonal endpoint pair"
    return segs


def _ray_dirs_for_slope(slope: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if slope == 1:
        return ((1, -1), (-1, 1))
    return ((1, 1), (-1, -1))


def _rays(
    grid: np.ndarray,
    segs: list[dict],
    red_blocks: list[set[tuple[int, int]]],
) -> list[dict]:
    h, w = grid.shape
    red_cells = set().union(*red_blocks)
    magenta_pts = {tuple(map(int, rc)) for rc in np.argwhere(grid == MAGENTA)}
    out: list[dict] = []
    seen_keys: set[tuple[int, int, int]] = set()
    for seg in segs:
        ray_slope = -int(seg["slope"])
        dirs = _ray_dirs_for_slope(int(seg["slope"]))
        for trigger in seg["path"]:
            if trigger not in magenta_pts:
                continue
            key = (trigger[0], trigger[1], ray_slope)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cells = {trigger}
            for dr, dc in dirs:
                r, c = trigger[0] + dr, trigger[1] + dc
                while 0 <= r < h and 0 <= c < w and (r, c) not in red_cells:
                    cells.add((r, c))
                    r += dr
                    c += dc
            out.append(
                {
                    "kind": "ray",
                    "color": MAGENTA,
                    "cells": cells,
                    "trigger": trigger,
                    "slope": ray_slope,
                }
            )
    return out


def _unique_chain(pieces: list[dict], red_blocks: list[set[tuple[int, int]]]) -> list[int]:
    n = len(pieces)
    graph = [set() for _ in range(n + 2)]
    src = n
    dst = n + 1
    for i in range(n):
        for j in range(i + 1, n):
            if _touches(pieces[i]["cells"], pieces[j]["cells"]):
                graph[i].add(j)
                graph[j].add(i)
    for i, piece in enumerate(pieces):
        if _touches(piece["cells"], red_blocks[0]):
            graph[src].add(i)
            graph[i].add(src)
        if _touches(piece["cells"], red_blocks[1]):
            graph[dst].add(i)
            graph[i].add(dst)
    assert graph[src] and graph[dst], "both red blocks must touch candidate pieces"

    found: list[list[int]] = []

    def dfs(node: int, path: list[int], seen: set[int]) -> None:
        if len(found) >= 2:
            return
        if node == dst:
            found.append(path[:])
            return
        for nxt in sorted(graph[node]):
            if nxt in seen:
                continue
            if nxt == src:
                continue
            if nxt == dst:
                dfs(nxt, path, seen | {nxt})
            else:
                dfs(nxt, path + [nxt], seen | {nxt})

    dfs(src, [], {src})
    assert len(found) == 1, "expected exactly one source-target piece chain"
    return found[0]


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    red_blocks = _red_blocks(g)
    segs = _segments(g)
    rays = _rays(g, segs, red_blocks)
    pieces = segs + rays
    path = _unique_chain(pieces, red_blocks)
    return {
        "grid": g,
        "red_blocks": red_blocks,
        "segments": segs,
        "rays": rays,
        "pieces": pieces,
        "path": path,
    }


def _render_chain(grid: np.ndarray, pieces: list[dict], path: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for idx in path:
        piece = pieces[idx]
        for r, c in piece["cells"]:
            if out[r, c] == BG:
                out[r, c] = int(piece["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal only the unique hidden light-chain whose candidate diagonal segments and split rays connect the two red blocks.

    Input structure:
    - A black background contains exactly two distant 2x2 red blocks.
    - Blue pixels appear as endpoint pairs on shared diagonals; each pair defines one hidden candidate diagonal segment.
    - Magenta trigger pixels are scattered; a trigger that lies on a candidate segment emits a pair of hidden perpendicular diagonal rays.

    Rule:
    - Reconstruct every candidate blue diagonal segment from each diagonal blue endpoint pair.
    - For every magenta trigger lying on such a segment, extend magenta rays in the two perpendicular diagonal directions until the grid boundary or a red block.
    - Build the 8-connected contact graph of all candidate segments and rays; exactly one chain of touching pieces connects the two red blocks.
    - Output the input grid plus only that chain: fill hidden segment cells in blue and hidden ray cells in magenta, while preserving red blocks and all original visible markers.

    Key insight:
    - The decisive structure is not any single visible line fragment but the unique connectivity chain inside the full invisible candidate network; decoy endpoint pairs and triggers stay irrelevant because they do not help connect the red blocks.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    return _render_chain(info["grid"], info["pieces"], info["path"])


GEN_ATTEMPTS = 600
CHAIN_SEGMENTS_RANGE = (2, 4)
CHAIN_PRE_RANGE = (2, 5)
CHAIN_POST_RANGE = (2, 4)
CHAIN_FINAL_RANGE = (4, 8)
CHAIN_RAY_RANGE = (3, 6)
MARGIN_RANGE = (2, 5)
EXTRA_OFFSEG_TRIGGER_RANGE = (0, 2)
OFFGRID_TRIGGER_RANGE = (0, 3)
ISOLATED_SEGMENT_PROB = 0.45


def _block_behind(point: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    rows = [point[0] - 2, point[0] - 1] if direction[0] == 1 else [point[0] + 1, point[0] + 2]
    cols = [point[1] - 2, point[1] - 1] if direction[1] == 1 else [point[1] + 1, point[1] + 2]
    return {(r, c) for r in rows for c in cols}


def _block_ahead(point: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    rows = [point[0] + 1, point[0] + 2] if direction[0] == 1 else [point[0] - 2, point[0] - 1]
    cols = [point[1] + 1, point[1] + 2] if direction[1] == 1 else [point[1] - 2, point[1] - 1]
    return {(r, c) for r in rows for c in cols}


def _build_chain_plan() -> dict | None:
    segment_family = random.choice((1, -1))
    if segment_family == 1:
        segment_dirs = ((1, 1), (-1, -1))
        ray_dirs = ((1, -1), (-1, 1))
    else:
        segment_dirs = ((1, -1), (-1, 1))
        ray_dirs = ((1, 1), (-1, -1))

    count = random.randint(*CHAIN_SEGMENTS_RANGE)
    segments = []
    start = (0, 0)
    used_points = {start}

    for i in range(count):
        direction = random.choice(segment_dirs)
        if i < count - 1:
            pre = random.randint(*CHAIN_PRE_RANGE)
            post = random.randint(*CHAIN_POST_RANGE)
            trigger = (start[0] + pre * direction[0], start[1] + pre * direction[1])
            end = (trigger[0] + post * direction[0], trigger[1] + post * direction[1])
            ray_dir = random.choice(ray_dirs)
            ray_len = random.randint(*CHAIN_RAY_RANGE)
            next_start = (trigger[0] + ray_len * ray_dir[0], trigger[1] + ray_len * ray_dir[1])
            segments.append(
                {
                    "start": start,
                    "trigger": trigger,
                    "end": end,
                    "direction": direction,
                    "ray_dir": ray_dir,
                    "next_start": next_start,
                }
            )
            start = next_start
            if start in used_points:
                return None
            used_points.add(start)
        else:
            total = random.randint(*CHAIN_FINAL_RANGE)
            end = (start[0] + total * direction[0], start[1] + total * direction[1])
            segments.append(
                {
                    "start": start,
                    "trigger": None,
                    "end": end,
                    "direction": direction,
                    "ray_dir": None,
                    "next_start": None,
                }
            )

    red1 = _block_behind(segments[0]["start"], segments[0]["direction"])
    red2 = _block_ahead(segments[-1]["end"], segments[-1]["direction"])

    all_cells = set(red1) | set(red2)
    for seg in segments:
        all_cells.update(_diag_path(seg["start"], seg["end"]))
        if seg["trigger"] is not None:
            all_cells.add(seg["next_start"])

    if red1 & red2:
        return None
    return {"segments": segments, "red1": red1, "red2": red2}


def _shift_plan(plan: dict) -> tuple[dict, tuple[int, int]] | tuple[None, None]:
    all_coords: list[tuple[int, int]] = []
    for seg in plan["segments"]:
        all_coords.extend(_diag_path(seg["start"], seg["end"]))
        if seg["trigger"] is not None:
            all_coords.extend([seg["trigger"], seg["next_start"]])
    all_coords.extend(plan["red1"])
    all_coords.extend(plan["red2"])

    rs = [r for r, _ in all_coords]
    cs = [c for _, c in all_coords]
    top = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    shift_r = top - min(rs)
    shift_c = left - min(cs)
    h = max(rs) - min(rs) + 1 + top + bottom
    w = max(cs) - min(cs) + 1 + left + right
    if h > 30 or w > 30 or h < 10 or w < 10:
        return None, None

    def move(pt: tuple[int, int]) -> tuple[int, int]:
        return pt[0] + shift_r, pt[1] + shift_c

    shifted_segments = []
    for seg in plan["segments"]:
        shifted_segments.append(
            {
                "start": move(seg["start"]),
                "trigger": None if seg["trigger"] is None else move(seg["trigger"]),
                "end": move(seg["end"]),
                "direction": seg["direction"],
                "ray_dir": seg["ray_dir"],
                "next_start": None if seg["next_start"] is None else move(seg["next_start"]),
            }
        )
    shifted = {
        "segments": shifted_segments,
        "red1": {move(pt) for pt in plan["red1"]},
        "red2": {move(pt) for pt in plan["red2"]},
    }
    return shifted, (h, w)


def _render_input(plan: dict, shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in plan["red1"] | plan["red2"]:
        grid[r, c] = RED
    for seg in plan["segments"]:
        grid[seg["start"]] = BLUE
        grid[seg["end"]] = BLUE
        if seg["trigger"] is not None:
            grid[seg["trigger"]] = MAGENTA
    return grid


def _cells_with_clearance(
    candidates: Iterable[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    clearance: int = 1,
) -> list[tuple[int, int]]:
    good = []
    for r, c in candidates:
        ok = True
        for rr, cc in forbidden:
            if max(abs(r - rr), abs(c - cc)) <= clearance:
                ok = False
                break
        if ok:
            good.append((r, c))
    return good


def _try_add_branch_trigger(grid: np.ndarray, plan: dict) -> np.ndarray | None:
    candidates: list[tuple[int, int]] = []
    existing = {tuple(map(int, rc)) for rc in np.argwhere(grid == MAGENTA)}
    for seg in plan["segments"]:
        path = _diag_path(seg["start"], seg["end"])
        for cell in path[1:-1]:
            if cell not in existing:
                candidates.append(cell)
    random.shuffle(candidates)
    for cell in candidates:
        temp = grid.copy()
        temp[cell] = MAGENTA
        try:
            info = _analyze(temp)
        except AssertionError:
            continue
        path = info["path"]
        kinds = [info["pieces"][idx]["kind"] for idx in path]
        if "ray" in kinds and kinds.count("segment") >= 2 and len(path) < len(info["pieces"]):
            return temp
    return None


def _try_add_isolated_segment(grid: np.ndarray) -> np.ndarray | None:
    h, w = grid.shape
    current_nonbg = {tuple(map(int, rc)) for rc in np.argwhere(grid != BG)}
    for _ in range(80):
        direction = random.choice(DIAG_DIRS)
        length = random.randint(3, 7)
        r0 = random.randint(0, h - 1)
        c0 = random.randint(0, w - 1)
        r1 = r0 + length * direction[0]
        c1 = c0 + length * direction[1]
        if not (0 <= r1 < h and 0 <= c1 < w):
            continue
        path = _diag_path((r0, c0), (r1, c1))
        if any(cell in current_nonbg for cell in path):
            continue
        if any(
            max(abs(r - rr), abs(c - cc)) <= 1
            for r, c in path
            for rr, cc in current_nonbg
        ):
            continue
        temp = grid.copy()
        temp[r0, c0] = BLUE
        temp[r1, c1] = BLUE
        if random.random() < 0.75 and len(path) >= 5:
            trigger = random.choice(path[1:-1])
            temp[trigger] = MAGENTA
        try:
            info = _analyze(temp)
        except AssertionError:
            continue
        if len(info["path"]) < len(info["pieces"]):
            return temp
    return None


def _add_offpath_triggers(grid: np.ndarray, info: dict) -> np.ndarray:
    temp = grid.copy()
    forbidden = set().union(*(piece["cells"] for piece in info["pieces"])) | set().union(*info["red_blocks"])
    h, w = temp.shape
    candidates = [(r, c) for r in range(h) for c in range(w) if temp[r, c] == BG]
    candidates = _cells_with_clearance(candidates, forbidden, clearance=0)
    random.shuffle(candidates)
    want = random.randint(*OFFGRID_TRIGGER_RANGE)
    placed = 0
    for r, c in candidates:
        temp[r, c] = MAGENTA
        placed += 1
        if placed == want:
            break
    return temp


def _generated_ok(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False
    path = info["path"]
    kinds = [info["pieces"][idx]["kind"] for idx in path]
    if "ray" not in kinds or kinds.count("segment") < 2:
        return False
    if len(path) >= len(info["pieces"]):
        return False
    out = _render_chain(grid, info["pieces"], path)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, out):
        return False
    if not (np.count_nonzero(grid == RED) == 8 and len(_component_coords(grid == RED)) == 2):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    plan = _build_chain_plan()
    if plan is None:
        return None
    shifted, shape = _shift_plan(plan)
    if shifted is None:
        return None
    grid = _render_input(shifted, shape)

    branch = _try_add_branch_trigger(grid, shifted)
    if branch is not None:
        grid = branch
    elif random.random() < 0.8:
        iso = _try_add_isolated_segment(grid)
        if iso is not None:
            grid = iso

    if random.random() < ISOLATED_SEGMENT_PROB:
        iso = _try_add_isolated_segment(grid)
        if iso is not None:
            grid = iso

    try:
        info = _analyze(grid)
    except AssertionError:
        return None

    extra_onseg = random.randint(*EXTRA_OFFSEG_TRIGGER_RANGE)
    if extra_onseg:
        for _ in range(extra_onseg):
            branched = _try_add_branch_trigger(grid, shifted)
            if branched is not None:
                grid = branched

    try:
        info = _analyze(grid)
    except AssertionError:
        return None

    grid = _add_offpath_triggers(grid, info)
    return grid if _generated_ok(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unique-chain puzzle")
