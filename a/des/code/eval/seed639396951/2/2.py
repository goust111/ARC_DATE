import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _prefix(mask: np.ndarray) -> np.ndarray:
    ps = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    ps[1:, 1:] = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return ps


def _rect_sum(ps: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _touch_type(a: dict, b: dict) -> str:
    ar0, ac0, ar1, ac1 = a["r0"], a["c0"], a["r1"], a["c1"]
    br0, bc0, br1, bc1 = b["r0"], b["c0"], b["r1"], b["c1"]

    if _overlap_len(ar0, ar1, br0, br1) > 0 and _overlap_len(ac0, ac1, bc0, bc1) > 0:
        return "overlap"

    vertical_edge = (ar1 + 1 == br0 or br1 + 1 == ar0) and _overlap_len(ac0, ac1, bc0, bc1) > 0
    horizontal_edge = (ac1 + 1 == bc0 or bc1 + 1 == ac0) and _overlap_len(ar0, ar1, br0, br1) > 0
    if vertical_edge or horizontal_edge:
        return "edge"

    corner = (ar1 + 1 == br0 or br1 + 1 == ar0) and (ac1 + 1 == bc0 or bc1 + 1 == ac0)
    if corner:
        return "corner"

    return "separate"


def _adjacent(a: dict, b: dict) -> bool:
    return _touch_type(a, b) == "edge"


def _rect_border_cells(rect: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _rect_base_color(
    ps_occ: np.ndarray,
    ps_red: np.ndarray,
    ps_grey: np.ndarray,
    ps_yellow: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> int | None:
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    occ = _rect_sum(ps_occ, r0, r1, c0, c1)
    if occ != area:
        return None

    red = _rect_sum(ps_red, r0, r1, c0, c1)
    grey = _rect_sum(ps_grey, r0, r1, c0, c1)
    yellow = _rect_sum(ps_yellow, r0, r1, c0, c1)
    if red + grey + yellow != area:
        return None
    if yellow > 1:
        return None
    if red > 0 and grey == 0 and yellow == 0:
        return RED
    if grey > 0 and red == 0:
        return GREY
    return None


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    work = np.asarray(grid, dtype=int).copy()
    h, w = work.shape
    rects = []

    while np.any(work != BLACK):
        occ = work != BLACK
        ps_occ = _prefix(occ)
        ps_red = _prefix(work == RED)
        ps_grey = _prefix(work == GREY)
        ps_yellow = _prefix(work == YELLOW)

        best = None
        best_key = None
        for r0 in range(h):
            for r1 in range(r0, h):
                for c0 in range(w):
                    for c1 in range(c0, w):
                        base = _rect_base_color(ps_occ, ps_red, ps_grey, ps_yellow, r0, r1, c0, c1)
                        if base is None:
                            continue

                        ys = np.argwhere(work[r0 : r1 + 1, c0 : c1 + 1] == YELLOW)
                        marker = None
                        if ys.size:
                            rr, cc = map(int, ys[0])
                            marker = (r0 + rr, c0 + cc)
                            if not (marker[0] in (r0, r1) or marker[1] in (c0, c1)):
                                continue

                        top_dirty = 0 if r0 == 0 else int(np.count_nonzero(occ[r0 - 1, c0 : c1 + 1]))
                        bottom_dirty = 0 if r1 + 1 == h else int(np.count_nonzero(occ[r1 + 1, c0 : c1 + 1]))
                        left_dirty = 0 if c0 == 0 else int(np.count_nonzero(occ[r0 : r1 + 1, c0 - 1]))
                        right_dirty = 0 if c1 + 1 == w else int(np.count_nonzero(occ[r0 : r1 + 1, c1 + 1]))

                        dirty = {
                            "top": top_dirty,
                            "bottom": bottom_dirty,
                            "left": left_dirty,
                            "right": right_dirty,
                        }
                        dirty_sides = [side for side, count in dirty.items() if count > 0]
                        if len(dirty_sides) > 1:
                            continue

                        if dirty_sides:
                            side = dirty_sides[0]
                            side_len = (c1 - c0 + 1) if side in ("top", "bottom") else (r1 - r0 + 1)
                            if dirty[side] >= side_len:
                                continue

                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        key = (area, 1 if marker is not None else 0, -r0, -c0)
                        if best is None or key > best_key:
                            best = {
                                "r0": r0,
                                "c0": c0,
                                "r1": r1,
                                "c1": c1,
                                "base": base,
                                "marker": marker,
                            }
                            best_key = key

        if best is None:
            raise AssertionError("failed to recover rectangle cover")

        rects.append(best)
        work[best["r0"] : best["r1"] + 1, best["c0"] : best["c1"] + 1] = BLACK

    rects.sort(key=lambda r: (r["r0"], r["c0"], r["r1"], r["c1"]))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rects = _extract_rectangles(g)

    source_ids = [i for i, rect in enumerate(rects) if rect["base"] == RED]
    assert len(source_ids) == 1, "expected exactly one red source rectangle"
    source = source_ids[0]

    adj = [[] for _ in rects]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _adjacent(rects[i], rects[j]):
                adj[i].append(j)
                adj[j].append(i)

    dist = [-1] * len(rects)
    dist[source] = 0
    q = deque([source])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if dist[nxt] == -1:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)

    out = np.full_like(g, BLACK)
    for i, rect in enumerate(rects):
        if rect["base"] == RED:
            color = RED
        elif dist[i] == -1:
            color = GREY
        elif dist[i] % 2 == 1:
            color = BLUE
        else:
            color = GREEN

        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = color
        if rect["marker"] is not None:
            mr, mc = rect["marker"]
            out[mr, mc] = MAROON

    return out


MIN_SIZE = 2
MAX_SIZE = 5
MAX_TOTAL_RECTS = 9
MAX_SPAN = 26
MARGIN_RANGE = (1, 3)
GEN_RETRIES = 500
ATTACH_RETRIES = 120
ISOLATED_RETRIES = 160
PLACEMENT_SPAN = 12


def _sample_dims(rng: random.Random) -> tuple[int, int]:
    h = rng.randint(MIN_SIZE, MAX_SIZE)
    w = rng.randint(MIN_SIZE, MAX_SIZE)
    return h, w


def _make_rect(r0: int, c0: int, h: int, w: int) -> dict:
    return {"r0": r0, "c0": c0, "r1": r0 + h - 1, "c1": c0 + w - 1}


def _starts_with_exact_overlap(a0: int, a1: int, length: int, overlap: int) -> list[int]:
    starts = []
    for start in range(a0 - length + 1, a1 + 1):
        if _overlap_len(start, start + length - 1, a0, a1) == overlap:
            starts.append(start)
    return starts


def _attached_candidate(parent: dict, h: int, w: int, side: str, rng: random.Random) -> dict | None:
    if side in ("N", "S"):
        max_overlap = min(parent["c1"] - parent["c0"] + 1, w) - 1
        if max_overlap < 1:
            return None
        overlap = rng.randint(1, max_overlap)
        starts = _starts_with_exact_overlap(parent["c0"], parent["c1"], w, overlap)
        if not starts:
            return None
        c0 = rng.choice(starts)
        r0 = parent["r0"] - h if side == "N" else parent["r1"] + 1
        return _make_rect(r0, c0, h, w)

    max_overlap = min(parent["r1"] - parent["r0"] + 1, h) - 1
    if max_overlap < 1:
        return None
    overlap = rng.randint(1, max_overlap)
    starts = _starts_with_exact_overlap(parent["r0"], parent["r1"], h, overlap)
    if not starts:
        return None
    r0 = rng.choice(starts)
    c0 = parent["c0"] - w if side == "W" else parent["c1"] + 1
    return _make_rect(r0, c0, h, w)


def _try_attach(rects: list[dict], parent_idx: int, rng: random.Random) -> dict | None:
    parent = rects[parent_idx]
    for _ in range(ATTACH_RETRIES):
        h, w = _sample_dims(rng)
        side = rng.choice(["N", "S", "E", "W"])
        cand = _attached_candidate(parent, h, w, side, rng)
        if cand is None:
            continue

        edge_neighbors = []
        ok = True
        for idx, rect in enumerate(rects):
            rel = _touch_type(cand, rect)
            if rel in ("overlap", "corner"):
                ok = False
                break
            if rel == "edge":
                edge_neighbors.append(idx)
        if ok and edge_neighbors == [parent_idx]:
            return cand
    return None


def _try_isolated(rects: list[dict], rng: random.Random) -> dict | None:
    for _ in range(ISOLATED_RETRIES):
        h, w = _sample_dims(rng)
        cand = _make_rect(
            rng.randint(-PLACEMENT_SPAN, PLACEMENT_SPAN),
            rng.randint(-PLACEMENT_SPAN, PLACEMENT_SPAN),
            h,
            w,
        )
        if all(_touch_type(cand, rect) == "separate" for rect in rects):
            return cand
    return None


def _component_distances(adj: list[set[int]], source: int) -> list[int]:
    dist = [-1] * len(adj)
    dist[source] = 0
    q = deque([source])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if dist[nxt] == -1:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def _shift_layout(rects: list[dict], top: int, left: int) -> list[dict]:
    shifted = []
    for rect in rects:
        shifted.append(
            {
                "r0": rect["r0"] + top,
                "c0": rect["c0"] + left,
                "r1": rect["r1"] + top,
                "c1": rect["c1"] + left,
            }
        )
    return shifted


def _choose_terminal_cells(
    grid: np.ndarray,
    rects: list[dict],
    terminal_ids: set[int],
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    markers = {}
    for idx in terminal_ids:
        rect = rects[idx]
        exposed = []
        for r, c in _rect_border_cells(rect):
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                inside = rect["r0"] <= nr <= rect["r1"] and rect["c0"] <= nc <= rect["c1"]
                if inside:
                    continue
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == BLACK:
                    exposed.append((r, c))
                    break
        if not exposed:
            return None
        markers[idx] = rng.choice(exposed)
    return markers


def _expected_from_layout(
    shape: tuple[int, int],
    rects: list[dict],
    adj: list[set[int]],
    source: int,
    markers: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    dist = _component_distances(adj, source)
    for idx, rect in enumerate(rects):
        if idx == source:
            color = RED
        elif dist[idx] == -1:
            color = GREY
        elif dist[idx] % 2 == 1:
            color = BLUE
        else:
            color = GREEN
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = color
    for pos in markers.values():
        out[pos] = MAROON
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    reachable_count = rng.randint(3, 6)
    extra_components = rng.choices([0, 1, 2], weights=[2, 3, 2])[0]
    extra_sizes = [rng.randint(1, 3) for _ in range(extra_components)]
    total_rects = reachable_count + sum(extra_sizes)
    if total_rects > MAX_TOTAL_RECTS:
        return None

    rects = [_make_rect(0, 0, *_sample_dims(rng))]
    adj = [set()]
    source = 0
    source_component = [0]

    forced_parent = 0
    for step in range(1, reachable_count):
        if step == 1:
            parent_idx = 0
        elif step == 2:
            parent_idx = forced_parent
        else:
            parent_idx = rng.choice(source_component)

        cand = _try_attach(rects, parent_idx, rng)
        if cand is None:
            return None
        rects.append(cand)
        adj.append(set())
        new_idx = len(rects) - 1
        adj[parent_idx].add(new_idx)
        adj[new_idx].add(parent_idx)
        source_component.append(new_idx)
        if step == 1:
            forced_parent = new_idx

    components = [source_component]
    for comp_size in extra_sizes:
        root = _try_isolated(rects, rng)
        if root is None:
            return None
        rects.append(root)
        adj.append(set())
        comp_indices = [len(rects) - 1]
        for _ in range(comp_size - 1):
            parent_idx = rng.choice(comp_indices)
            cand = _try_attach(rects, parent_idx, rng)
            if cand is None:
                return None
            rects.append(cand)
            adj.append(set())
            new_idx = len(rects) - 1
            adj[parent_idx].add(new_idx)
            adj[new_idx].add(parent_idx)
            comp_indices.append(new_idx)
        components.append(comp_indices)

    dist = _component_distances(adj, source)
    reachable_grey = [idx for idx in range(1, len(rects)) if dist[idx] >= 0]
    if not any(dist[idx] % 2 == 1 for idx in reachable_grey):
        return None
    if not any(dist[idx] % 2 == 0 and dist[idx] > 0 for idx in reachable_grey):
        return None

    grey_ids = list(range(1, len(rects)))
    unreachable_ids = [idx for idx in grey_ids if dist[idx] == -1]
    terminal_count = rng.randint(1, max(1, min(4, len(grey_ids))))
    terminal_ids = set(rng.sample(grey_ids, terminal_count))
    if not terminal_ids & set(reachable_grey):
        terminal_ids.add(rng.choice(reachable_grey))
    if unreachable_ids and rng.random() < 0.6 and not terminal_ids & set(unreachable_ids):
        terminal_ids.add(rng.choice(unreachable_ids))

    min_r = min(rect["r0"] for rect in rects)
    min_c = min(rect["c0"] for rect in rects)
    max_r = max(rect["r1"] for rect in rects)
    max_c = max(rect["c1"] for rect in rects)

    top_margin = rng.randint(*MARGIN_RANGE)
    left_margin = rng.randint(*MARGIN_RANGE)
    bottom_margin = rng.randint(*MARGIN_RANGE)
    right_margin = rng.randint(*MARGIN_RANGE)

    top_shift = top_margin - min_r
    left_shift = left_margin - min_c
    h = max_r - min_r + 1 + top_margin + bottom_margin
    w = max_c - min_c + 1 + left_margin + right_margin
    if h > MAX_SPAN or w > MAX_SPAN:
        return None

    shifted = _shift_layout(rects, top_shift, left_shift)
    grid = np.full((h, w), BLACK, dtype=int)
    for idx, rect in enumerate(shifted):
        color = RED if idx == source else GREY
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = color

    markers = _choose_terminal_cells(grid, shifted, terminal_ids, rng)
    if markers is None:
        return None
    for pos in markers.values():
        grid[pos] = YELLOW

    expected = _expected_from_layout(grid.shape, shifted, adj, source, markers)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
