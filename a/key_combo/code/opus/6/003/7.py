import random
from collections import deque

import numpy as np


GREY = 5
SOURCE = 2
PALETTE = [1, 2, 3, 4, 7]
MAX_GENERATE_ATTEMPTS = 400
_GEN_COUNTER = 0


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _neighbors(slot: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = slot
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            out.append((nr, nc))
    return out


def _parse_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    grey = patch == GREY
    row_counts = grey.sum(axis=1)
    col_counts = grey.sum(axis=0)
    wall_rows = [i for i, count in enumerate(row_counts) if int(count) > 4]
    wall_cols = [i for i, count in enumerate(col_counts) if int(count) > 4]
    assert len(wall_rows) == 4 and len(wall_cols) == 4

    slots: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for sr in range(3):
        for sc in range(3):
            slots[(sr, sc)] = (
                r0 + wall_rows[sr] + 1,
                c0 + wall_cols[sc] + 1,
                r0 + wall_rows[sr + 1] - 1,
                c0 + wall_cols[sc + 1] - 1,
            )

    adj = {slot: set() for slot in slots}
    gap_count = 0
    for sr in range(3):
        for sc in range(2):
            wall_c = c0 + wall_cols[sc + 1]
            rr0 = r0 + wall_rows[sr] + 1
            rr1 = r0 + wall_rows[sr + 1]
            if np.any(grid[rr0:rr1, wall_c] != GREY):
                a, b = (sr, sc), (sr, sc + 1)
                adj[a].add(b)
                adj[b].add(a)
                gap_count += 1
    for sr in range(2):
        for sc in range(3):
            wall_r = r0 + wall_rows[sr + 1]
            cc0 = c0 + wall_cols[sc] + 1
            cc1 = c0 + wall_cols[sc + 1]
            if np.any(grid[wall_r, cc0:cc1] != GREY):
                a, b = (sr, sc), (sr + 1, sc)
                adj[a].add(b)
                adj[b].add(a)
                gap_count += 1

    return {
        "bbox": bbox,
        "wall_rows": [r0 + x for x in wall_rows],
        "wall_cols": [c0 + x for x in wall_cols],
        "slots": slots,
        "adj": adj,
        "gap_count": gap_count,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == GREY)
    frames = []
    for coords in comps:
        bbox = _bbox_coords(coords)
        try:
            frames.append(_parse_frame(grid, bbox))
        except AssertionError:
            continue
    frames.sort(key=lambda item: item["bbox"][1])
    assert len(frames) == 2
    return frames


def _slot_color(grid: np.ndarray, slot_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = slot_bbox
    vals = set(map(int, np.unique(grid[r0 : r1 + 1, c0 : c1 + 1])))
    vals.discard(0)
    vals.discard(GREY)
    if not vals:
        return 0
    assert len(vals) == 1
    return next(iter(vals))


def _reachable_and_dist(adj: dict[tuple[int, int], set[tuple[int, int]]]) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    start = (0, 0)
    q = deque([start])
    dist = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in sorted(adj[cur]):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return set(dist), dist


def _oracle(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    left_candidates = [frame for frame in frames if frame["gap_count"] > 0]
    right_candidates = [frame for frame in frames if frame["gap_count"] == 0]
    assert len(left_candidates) == 1 and len(right_candidates) == 1
    left = left_candidates[0]
    right = right_candidates[0]

    reachable, dist = _reachable_and_dist(left["adj"])
    right_colors = {slot: _slot_color(g, right["slots"][slot]) for slot in right["slots"]}

    out = g.copy()
    for slot in reachable:
        r0, c0, r1, c1 = left["slots"][slot]
        out[r0 : r1 + 1, c0 : c1 + 1] = right_colors[slot]

    info = {
        "left": left,
        "right": right,
        "reachable": reachable,
        "dist": dist,
        "right_colors": right_colors,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood the gapped left 3x3 frame and recolor reached slots from their position-matched peers in the intact right frame.

    Input structure:
    - The background is black, and two grey frame components divide the left and right halves into parallel 3x3 slot grids.
    - The left frame contains internal one-pixel wall gaps and a single red source pixel inside its top-left slot.
    - The right frame has no gaps; some of its slots are fully prefilled with colors chosen from blue, red, green, yellow, and orange.

    Rule:
    - Treat each left-frame slot as a node, and connect orthogonally adjacent slots exactly when the shared grey wall segment contains a gap.
    - Starting from the top-left left-frame slot, breadth-first reach every slot connected through those gaps.
    - For each reached slot, overwrite that entire slot interior with the color found in the same slot position of the right frame; if the right slot is empty, clear the reached left slot to black.

    Key insight:
    - The left frame contributes only the flood topology, while the right frame contributes only the slot colors, so the answer requires combining two independent spatial cues.
    """
    out, _ = _oracle(np.asarray(grid, dtype=int))
    return out


def _all_edges() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for r in range(3):
        for c in range(3):
            a = (r, c)
            if r + 1 < 3:
                edges.append((a, (r + 1, c)))
            if c + 1 < 3:
                edges.append((a, (r, c + 1)))
    return edges


ALL_EDGES = _all_edges()


def _build_reachable_graph(rng: random.Random, mode: int) -> tuple[set[tuple[int, int]], set[frozenset[tuple[int, int]]]]:
    target_size = rng.randint(5, 8)
    reachable = {(0, 0)}
    open_edges: set[frozenset[tuple[int, int]]] = set()

    branch_first = mode != 2 or target_size >= 6
    source_nbrs = [(0, 1), (1, 0)]
    rng.shuffle(source_nbrs)
    initial = 2 if branch_first and target_size >= 3 else 1
    for idx in range(initial):
        nxt = source_nbrs[idx]
        reachable.add(nxt)
        open_edges.add(frozenset({(0, 0), nxt}))

    while len(reachable) < target_size:
        frontier = []
        for cell in sorted(reachable):
            for nbr in _neighbors(cell):
                if nbr not in reachable:
                    frontier.append((cell, nbr))
        assert frontier
        parent, child = rng.choice(frontier)
        reachable.add(child)
        open_edges.add(frozenset({parent, child}))

    extra_candidates = [
        frozenset({a, b})
        for a, b in ALL_EDGES
        if a in reachable and b in reachable and frozenset({a, b}) not in open_edges
    ]
    rng.shuffle(extra_candidates)
    extra_budget = 1 if mode == 2 else rng.randint(0, min(2, len(extra_candidates)))
    for edge in extra_candidates[:extra_budget]:
        open_edges.add(edge)

    return reachable, open_edges


def _line_positions(top: int, left: int, cell_h: int, cell_w: int) -> tuple[list[int], list[int]]:
    rows = [top + k * (cell_h + 1) for k in range(4)]
    cols = [left + k * (cell_w + 1) for k in range(4)]
    return rows, cols


def _draw_frame(grid: np.ndarray, top: int, left: int, cell_h: int, cell_w: int) -> tuple[list[int], list[int]]:
    rows, cols = _line_positions(top, left, cell_h, cell_w)
    for rr in rows:
        grid[rr, cols[0] : cols[-1] + 1] = GREY
    for cc in cols:
        grid[rows[0] : rows[-1] + 1, cc] = GREY
    return rows, cols


def _carve_gap(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    edge: frozenset[tuple[int, int]],
    rng: random.Random,
) -> None:
    a, b = sorted(edge)
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        wall_c = cols[min(c1, c2) + 1]
        rr = rng.randint(rows[r1] + 1, rows[r1 + 1] - 1)
        grid[rr, wall_c] = 0
    else:
        wall_r = rows[min(r1, r2) + 1]
        cc = rng.randint(cols[c1] + 1, cols[c1 + 1] - 1)
        grid[wall_r, cc] = 0


def _slot_bbox(rows: list[int], cols: list[int], slot: tuple[int, int]) -> tuple[int, int, int, int]:
    sr, sc = slot
    return rows[sr] + 1, cols[sc] + 1, rows[sr + 1] - 1, cols[sc + 1] - 1


def _fill_slot(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _choose_colored_slots(
    reachable: set[tuple[int, int]],
    open_edges: set[frozenset[tuple[int, int]]],
    rng: random.Random,
    mode: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    adj = {slot: set() for slot in reachable}
    for edge in open_edges:
        a, b = tuple(edge)
        if a in reachable and b in reachable:
            adj[a].add(b)
            adj[b].add(a)
    _, dist = _reachable_and_dist(adj)
    count = 4 if len(reachable) <= 5 else (5 if mode == 2 else rng.choice([4, 5]))
    count = min(count, len(reachable) - 1)

    include_source = mode == 1
    pool = list(reachable)
    if not include_source:
        pool = [slot for slot in pool if slot != (0, 0)]
    farthest = max(pool, key=lambda slot: (dist[slot], slot))
    near = min(pool, key=lambda slot: (dist[slot], slot))
    chosen = {farthest, near}
    while len(chosen) < count:
        chosen.add(rng.choice(pool))

    colors = PALETTE[:]
    rng.shuffle(colors)
    mapping = {slot: colors[idx] for idx, slot in enumerate(sorted(chosen))}
    return chosen, mapping


def _gen_attempt(mode: int) -> np.ndarray | None:
    rng = random.Random()

    cell_h = rng.randint(2, 4)
    cell_w = rng.randint(2, 3)
    frame_h = 3 * cell_h + 4
    frame_w = 3 * cell_w + 4

    max_gap = 30 - 2 * frame_w - 2
    if max_gap < 1:
        return None
    inter_gap = rng.randint(1, min(5, max_gap))
    total_w_min = 2 * frame_w + inter_gap + 2
    total_h_min = frame_h + 2
    width = rng.randint(total_w_min, min(30, total_w_min + 4))
    height = rng.randint(total_h_min, min(22, total_h_min + 6))

    left_margin = rng.randint(1, width - (2 * frame_w + inter_gap) - 1)
    right_margin = width - left_margin - 2 * frame_w - inter_gap
    top = rng.randint(1, height - frame_h - 1)
    left = left_margin
    right = left + frame_w + inter_gap

    grid = np.zeros((height, width), dtype=int)

    left_rows, left_cols = _draw_frame(grid, top, left, cell_h, cell_w)
    right_rows, right_cols = _draw_frame(grid, top, right, cell_h, cell_w)

    reachable, open_edges = _build_reachable_graph(rng, mode)
    for edge in open_edges:
        _carve_gap(grid, left_rows, left_cols, edge, rng)

    source_bbox = _slot_bbox(left_rows, left_cols, (0, 0))
    sr = rng.randint(source_bbox[0], source_bbox[2])
    sc = rng.randint(source_bbox[1], source_bbox[3])
    grid[sr, sc] = SOURCE

    _, right_mapping = _choose_colored_slots(reachable, open_edges, rng, mode)
    for slot, color in right_mapping.items():
        _fill_slot(grid, _slot_bbox(right_rows, right_cols, slot), color)

    try:
        out, info = _oracle(grid)
    except AssertionError:
        return None

    if not np.array_equal(main(grid), out):
        return None

    if len(info["reachable"]) < 5 or len(info["reachable"]) >= 9:
        return None

    if sum(color != 0 for color in info["right_colors"].values()) not in {4, 5}:
        return None

    if any(slot not in info["reachable"] for slot, color in info["right_colors"].items() if color != 0):
        return None

    if info["right_colors"][(0, 0)] != 0 and mode == 0:
        return None

    if info["right_colors"][(0, 0)] == 0 and mode == 1:
        return None

    if not any(info["right_colors"][slot] != 0 and info["dist"][slot] >= 2 for slot in info["reachable"]):
        return None

    if not any(info["right_colors"][slot] == 0 for slot in info["reachable"]):
        return None

    colored = [slot for slot, color in info["right_colors"].items() if color != 0]
    if len({info["right_colors"][slot] for slot in colored}) != len(colored):
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(MAX_GENERATE_ATTEMPTS):
        mode = _GEN_COUNTER % 3
        _GEN_COUNTER += 1
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
