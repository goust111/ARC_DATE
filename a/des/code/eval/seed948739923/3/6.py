import random
from collections import Counter, defaultdict, deque

import numpy as np


GREY = 5
RED = 2
ORANGE = 7
AZURE = 8

DIRS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("top", "right", "bottom", "left")
DIR_TO_K = {name: k for k, name in enumerate(DIR_ORDER)}
OPPOSITE = {"top": "bottom", "right": "left", "bottom": "top", "left": "right"}

GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _frame_info(grid: np.ndarray, color: int, pts: list[tuple[int, int]]):
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 5 or side_h % 2 == 0:
        return None
    local = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    target = np.zeros((side_h, side_h), dtype=bool)
    target[0, :] = True
    target[-1, :] = True
    target[:, 0] = True
    target[:, -1] = True
    if len(pts) != int(target.sum()):
        return None
    if not np.array_equal(local, target):
        return None
    return (r0, c0, r1, c1, side_h)


def _anchor_map(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    return {
        "top": (r0 - 1, cm),
        "right": (rm, c1 + 1),
        "bottom": (r1 + 1, cm),
        "left": (rm, c0 - 1),
    }


def _detect_clocks(grid: np.ndarray) -> list[dict]:
    bg = _most_common_color(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) not in (bg, RED, ORANGE, AZURE)]
    by_color: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for color in colors:
        infos = []
        for comp in _components(grid == color):
            info = _frame_info(grid, color, comp)
            if info is not None:
                infos.append(info)
        if infos:
            size = Counter(info[4] for info in infos).most_common(1)[0][0]
            by_color[color] = [info for info in infos if info[4] == size]
    if not by_color:
        raise AssertionError("no square frame color found")
    frame_color, infos = max(by_color.items(), key=lambda item: (len(item[1]), item[0]))
    clocks = []
    for idx, info in enumerate(sorted(infos)):
        r0, c0, r1, c1, side = info
        anchors = _anchor_map((r0, c0, r1, c1))
        pointer_hits = []
        for name, (ar, ac) in anchors.items():
            if 0 <= ar < grid.shape[0] and 0 <= ac < grid.shape[1]:
                val = int(grid[ar, ac])
                if val in (RED, ORANGE):
                    pointer_hits.append((name, val))
        if len(pointer_hits) != 1:
            raise AssertionError("each clock must have exactly one red/orange pointer")
        side_name, pointer_color = pointer_hits[0]
        clocks.append(
            {
                "id": idx,
                "frame_color": frame_color,
                "bbox": (r0, c0, r1, c1),
                "side": side,
                "anchors": anchors,
                "pointer_side": side_name,
                "pointer_color": pointer_color,
                "is_start": pointer_color == RED,
            }
        )
    return clocks


def _build_graph(grid: np.ndarray, clocks: list[dict]) -> dict[int, set[int]]:
    anchor_to_clock: dict[tuple[int, int], int] = {}
    for clock in clocks:
        for pos in clock["anchors"].values():
            anchor_to_clock[pos] = clock["id"]
    graph: dict[int, set[int]] = defaultdict(set)
    for clock in clocks:
        graph[clock["id"]] = set()
    for comp in _components(grid == AZURE):
        touched = sorted({anchor_to_clock[pos] for pos in comp if pos in anchor_to_clock})
        if len(touched) == 2:
            a, b = touched
            graph[a].add(b)
            graph[b].add(a)
    return graph


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    clocks = _detect_clocks(g)
    starts = [clock["id"] for clock in clocks if clock["is_start"]]
    if len(starts) != 1:
        raise AssertionError("expected exactly one start clock")
    start = starts[0]

    own = {clock["id"]: DIR_TO_K[clock["pointer_side"]] for clock in clocks}
    graph = _build_graph(g, clocks)

    total: dict[int, int] = {start: own[start]}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in sorted(graph[cur]):
            if nxt in total:
                continue
            total[nxt] = (total[cur] + own[nxt]) % 4
            q.append(nxt)
    for clock in clocks:
        total.setdefault(clock["id"], own[clock["id"]])

    out = g.copy()
    for clock in clocks:
        r0, c0, r1, c1 = clock["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(inner, -total[clock["id"]])
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, side: int, color: int) -> tuple[int, int, int, int]:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    return (r0, c0, r1, c1)


def _line_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(r, c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, c) for r in range(r0, r1 + 1)]
    raise AssertionError("line endpoints must be aligned")


def _step_side(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if abs(dr) + abs(dc) != 1:
        raise AssertionError("path steps must use slot-adjacent clocks")
    if dr == -1:
        return "top"
    if dr == 1:
        return "bottom"
    if dc == 1:
        return "right"
    return "left"


def _random_path(rng: random.Random, rows: int, cols: int, length: int):
    def grow(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
        if len(path) == length:
            return True
        r, c = path[-1]
        nbrs = []
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in used:
                nbrs.append((nr, nc))
        rng.shuffle(nbrs)
        for nb in nbrs:
            used.add(nb)
            path.append(nb)
            if grow(path, used):
                return True
            path.pop()
            used.remove(nb)
        return False

    for _ in range(120):
        start = (rng.randrange(rows), rng.randrange(cols))
        path = [start]
        used = {start}
        if grow(path, used):
            return path
    return None


def _random_pattern(rng: random.Random, side: int, bg: int, palette: list[int]) -> np.ndarray:
    cells = [(r, c) for r in range(side) for c in range(side)]
    for _ in range(200):
        arr = np.full((side, side), bg, dtype=int)
        fill_lo = max(2, side)
        fill_hi = min(side * side - 1, side * side // 2 + side)
        fill_n = rng.randint(fill_lo, fill_hi)
        picks = rng.sample(cells, fill_n)
        color_n = rng.randint(1, min(3, len(palette)))
        colors = rng.sample(palette, color_n)
        for r, c in picks:
            arr[r, c] = rng.choice(colors)
        sigs = {np.rot90(arr, k).tobytes() for k in range(4)}
        if len(sigs) >= 3:
            return arr
    arr = np.full((side, side), bg, dtype=int)
    fallback = palette[: max(1, min(3, len(palette)))]
    arr[0, 0] = fallback[0]
    arr[min(1, side - 1), max(0, side - 2)] = fallback[min(1, len(fallback) - 1)]
    arr[side - 1, side // 2] = fallback[min(2, len(fallback) - 1)]
    return arr


def _pick_layout(rng: random.Random, total_clocks: int, clock_side: int):
    for _ in range(120):
        rows = rng.randint(2, 4)
        cols = rng.randint(2, 4)
        if rows * cols < total_clocks:
            continue
        margin_r = rng.randint(1, 2)
        margin_c = rng.randint(1, 2)
        if rows == 1:
            max_gap_r = 4
        else:
            max_gap_r = (29 - 2 * margin_r - rows * clock_side) // (rows - 1)
        if cols == 1:
            max_gap_c = 4
        else:
            max_gap_c = (29 - 2 * margin_c - cols * clock_side) // (cols - 1)
        max_gap = min(max_gap_r, max_gap_c, 4)
        if max_gap < 2:
            continue
        gap = rng.randint(2, max_gap)
        height = 2 * margin_r + rows * clock_side + (rows - 1) * gap
        width = 2 * margin_c + cols * clock_side + (cols - 1) * gap
        if height <= 29 and width <= 29:
            return rows, cols, gap, margin_r, margin_c, height, width
    return None


def _spec_output(grid: np.ndarray, clocks: list[dict], chain_ids: list[int]) -> np.ndarray:
    out = grid.copy()
    totals: dict[int, int] = {}
    running = 0
    for cid in chain_ids:
        running = (running + DIR_TO_K[clocks[cid]["pointer_side"]]) % 4
        totals[cid] = running
    for clock in clocks:
        cid = clock["id"]
        totals.setdefault(cid, DIR_TO_K[clock["pointer_side"]])
        r0, c0, r1, c1 = clock["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(clock["pattern"], -totals[cid])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    clock_side = rng.choice([5, 7])
    chain_len = rng.randint(1, 5)
    isolated_n = rng.randint(0, 3)
    if chain_len + isolated_n < 2:
        isolated_n = 1
    total_clocks = chain_len + isolated_n

    layout = _pick_layout(rng, total_clocks, clock_side)
    if layout is None:
        return None
    rows, cols, gap, margin_r, margin_c, height, width = layout
    path_slots = _random_path(rng, rows, cols, chain_len)
    if path_slots is None:
        return None

    all_slots = [(r, c) for r in range(rows) for c in range(cols)]
    free_slots = [slot for slot in all_slots if slot not in path_slots]
    if len(free_slots) < isolated_n:
        return None
    rng.shuffle(free_slots)
    isolated_slots = free_slots[:isolated_n]

    bg = rng.choice([0, 1, 3, 4, 6, 9])
    palette = [c for c in range(10) if c not in (bg, GREY, RED, ORANGE, AZURE)]
    rng.shuffle(palette)
    grid = np.full((height, width), bg, dtype=int)

    slot_to_id: dict[tuple[int, int], int] = {}
    clocks: list[dict] = []
    placed_slots = path_slots + isolated_slots
    for slot in placed_slots:
        sr, sc = slot
        r0 = margin_r + sr * (clock_side + gap)
        c0 = margin_c + sc * (clock_side + gap)
        bbox = _draw_frame(grid, r0, c0, clock_side, GREY)
        pattern = _random_pattern(rng, clock_side - 2, bg, palette)
        r1, c1 = bbox[2], bbox[3]
        grid[r0 + 1 : r1, c0 + 1 : c1] = pattern
        cid = len(clocks)
        slot_to_id[slot] = cid
        clocks.append(
            {
                "id": cid,
                "slot": slot,
                "bbox": bbox,
                "anchors": _anchor_map(bbox),
                "used_sides": set(),
                "pattern": pattern,
            }
        )

    for a_slot, b_slot in zip(path_slots, path_slots[1:]):
        a_id = slot_to_id[a_slot]
        b_id = slot_to_id[b_slot]
        side = _step_side(a_slot, b_slot)
        clocks[a_id]["used_sides"].add(side)
        clocks[b_id]["used_sides"].add(OPPOSITE[side])

    chain_ids = [slot_to_id[slot] for slot in path_slots]
    for i, cid in enumerate(chain_ids):
        clock = clocks[cid]
        available = [name for name in DIR_ORDER if name not in clock["used_sides"]]
        if not available:
            return None
        pointer_side = rng.choice(available)
        pointer_color = RED if i == 0 else ORANGE
        pr, pc = clock["anchors"][pointer_side]
        if not (0 <= pr < height and 0 <= pc < width):
            return None
        if grid[pr, pc] != bg:
            return None
        grid[pr, pc] = pointer_color
        clock["pointer_side"] = pointer_side
        clock["pointer_color"] = pointer_color

    for slot in isolated_slots:
        cid = slot_to_id[slot]
        clock = clocks[cid]
        pointer_side = rng.choice(list(DIR_ORDER))
        pr, pc = clock["anchors"][pointer_side]
        if not (0 <= pr < height and 0 <= pc < width):
            return None
        if grid[pr, pc] != bg:
            return None
        grid[pr, pc] = ORANGE
        clock["pointer_side"] = pointer_side
        clock["pointer_color"] = ORANGE

    for a_slot, b_slot in zip(path_slots, path_slots[1:]):
        a_clock = clocks[slot_to_id[a_slot]]
        b_clock = clocks[slot_to_id[b_slot]]
        side = _step_side(a_slot, b_slot)
        pts = _line_between(a_clock["anchors"][side], b_clock["anchors"][OPPOSITE[side]])
        for r, c in pts:
            if grid[r, c] != bg:
                return None
        for r, c in pts:
            grid[r, c] = AZURE

    out = _spec_output(grid, clocks, chain_ids)
    if np.array_equal(grid, out):
        return None
    if len([clock for clock in clocks if clock["pointer_color"] == RED]) != 1:
        return None
    if _most_common_color(grid) != bg:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
