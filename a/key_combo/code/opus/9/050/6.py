import random
from collections import deque

import numpy as np


BG = 0
BAR = 6
RECT_COLORS = (1, 3, 4, 7)
BAR_LENGTHS = (3, 5, 7)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_TRIES = 600


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _line_coords(cr: int, cc: int, length: int, orient: str) -> list[tuple[int, int]]:
    half = length // 2
    if orient == "H":
        return [(cr, cc + dc) for dc in range(-half, half + 1)]
    return [(cr + dr, cc) for dr in range(-half, half + 1)]


def _rotated_coords(cr: int, cc: int, length: int, orient: str) -> list[tuple[int, int]]:
    return _line_coords(cr, cc, length, "V" if orient == "H" else "H")


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    colors_set = set(int(c) for c in colors)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors_set or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _count_holes_in_patch(patch: np.ndarray) -> int:
    zero_mask = patch == BG
    if not zero_mask.any():
        return 0
    h, w = patch.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if zero_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if zero_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and zero_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = zero_mask & ~outside
    return len(_label_components(enclosed))


def _count_rect_holes(grid: np.ndarray, rect: dict) -> int:
    r0, c0, r1, c1 = rect["bbox"]
    return _count_holes_in_patch(grid[r0 : r1 + 1, c0 : c1 + 1])


def _is_bar(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    length = max(h, w)
    return (h == 1 or w == 1) and int(comp["area"]) == length and length in BAR_LENGTHS


def _bar_geometry(comp: dict) -> tuple[int, int, int, str]:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    return cr, cc, max(h, w), "H" if h == 1 else "V"


def _analyze(grid: np.ndarray, *, require_bars: bool = True) -> tuple[list[dict], list[int], list[dict], list[dict]]:
    comps = _find_components(grid, RECT_COLORS + (BAR,))
    rects = []
    bars = []
    for comp in comps:
        if comp["color"] == BAR:
            if not _is_bar(comp):
                raise AssertionError("invalid magenta component")
            bars.append(comp)
        else:
            rects.append(comp)
    if not rects or (require_bars and not bars):
        raise AssertionError("missing required objects")

    rect_ids = np.full(grid.shape, -1, dtype=int)
    rect_holes = []
    for idx, rect in enumerate(rects):
        coords = rect["coords"]
        rect_ids[coords[:, 0], coords[:, 1]] = idx
        rect_holes.append(_count_rect_holes(grid, rect))

    infos = []
    for bar in bars:
        cr, cc, length, orient = _bar_geometry(bar)
        adjacent = set()
        for r, c in bar["coords"]:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    idx = int(rect_ids[nr, nc])
                    if idx >= 0:
                        adjacent.add(idx)
        if not adjacent:
            raise AssertionError("bar without adjacent rectangle")
        hole_sum = sum(rect_holes[idx] for idx in adjacent)
        out_coords = _line_coords(cr, cc, length, orient) if hole_sum % 2 == 0 else _rotated_coords(cr, cc, length, orient)
        infos.append(
            {
                "coords": {tuple(map(int, p)) for p in bar["coords"]},
                "adjacent": adjacent,
                "hole_sum": hole_sum,
                "rotates": bool(hole_sum % 2),
                "output": set(out_coords),
                "length": length,
                "orient": orient,
            }
        )
    return rects, rect_holes, bars, infos


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each magenta thin bar iff the total enclosed-hole count of its 4-neighbor rectangles is odd.

    Input structure:
    - Black is the background.
    - Non-magenta foreground objects are distinct-colored axis-aligned rectangles, some with enclosed black holes.
    - Magenta foreground objects are one-pixel-thick horizontal or vertical odd-length bars.

    Rule:
    - Identify every magenta bar and every colored rectangle.
    - For each bar, collect all rectangles that are 4-neighbor adjacent to at least one of its pixels.
    - Count the enclosed 4-connected black hole components inside each adjacent rectangle and sum them.
    - Keep the bar unchanged when the sum is even; rotate it 90 degrees around its center pixel when the sum is odd.
    - Leave every rectangle and every hole unchanged.

    Key insight:
    - The bar itself carries no decision signal; the action is determined only by aggregating topological hole counts from neighboring rectangles.
    """
    g = np.asarray(grid, dtype=int)
    rects, _rect_holes, bars, infos = _analyze(g)

    out = g.copy()
    for bar in bars:
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    rect_mask = np.zeros(g.shape, dtype=bool)
    for rect in rects:
        coords = rect["coords"]
        rect_mask[coords[:, 0], coords[:, 1]] = True

    occupied = set()
    for info in infos:
        for r, c in info["output"]:
            if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
                raise AssertionError("rotated bar left grid")
            if rect_mask[r, c]:
                raise AssertionError("bar overlaps rectangle after rotation")
            if (r, c) in occupied:
                raise AssertionError("bars overlap after rotation")
            occupied.add((r, c))
            out[r, c] = BAR
    return out


def _component_count(mask: np.ndarray) -> int:
    return len(_label_components(mask))


def _grow_component(interior_cells: list[tuple[int, int]], used: set[tuple[int, int]], size_lo: int, size_hi: int) -> set[tuple[int, int]]:
    start = random.choice(interior_cells)
    comp = {start}
    target = random.randint(size_lo, size_hi)
    for _ in range(target - 1):
        frontier = []
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in comp or (nr, nc) in used or (nr, nc) not in interior_cells:
                    continue
                frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            break
        comp.add(random.choice(frontier))
    return comp


def _build_hole_mask(ih: int, iw: int, hole_count: int) -> np.ndarray | None:
    if hole_count == 0:
        return np.zeros((ih, iw), dtype=bool)

    interior = [(r, c) for r in range(ih) for c in range(iw)]
    interior_set = set(interior)
    for _ in range(80):
        mask = np.zeros((ih, iw), dtype=bool)
        used: set[tuple[int, int]] = set()
        comps: list[set[tuple[int, int]]] = []

        if hole_count == 1:
            comp = _grow_component(interior, used, 1, min(3, ih * iw))
            comps = [comp]
        else:
            seeds = interior[:]
            random.shuffle(seeds)
            sep_pair = None
            for a in seeds:
                for b in seeds:
                    if a == b:
                        continue
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1:
                        sep_pair = (a, b)
                        break
                if sep_pair is not None:
                    break
            if sep_pair is None:
                continue
            comp_a = {sep_pair[0]}
            comp_b = {sep_pair[1]}
            used |= comp_a | comp_b
            for comp in (comp_a, comp_b):
                growth = random.randint(0, 1)
                for _ in range(growth):
                    frontier = []
                    for r, c in comp:
                        for nr, nc in _neighbors4(r, c):
                            if (nr, nc) not in interior_set or (nr, nc) in used:
                                continue
                            if any(abs(nr - rr) + abs(nc - cc) == 1 for rr, cc in (comp_a | comp_b) - comp):
                                continue
                            frontier.append((nr, nc))
                    frontier = list(dict.fromkeys(frontier))
                    if not frontier:
                        break
                    cell = random.choice(frontier)
                    comp.add(cell)
                    used.add(cell)
            comps = [comp_a, comp_b]

        all_cells = set().union(*comps)
        if len(all_cells) == ih * iw and ih * iw > 1:
            continue
        for r, c in all_cells:
            mask[r, c] = True
        if _component_count(mask) == hole_count:
            return mask
    return None


def _make_rectangle(color: int, hole_count: int) -> np.ndarray:
    if hole_count == 2:
        size_choices = [(h, w) for h in range(4, 6) for w in range(4, 6)]
    else:
        size_choices = [(h, w) for h in range(3, 6) for w in range(3, 6)]

    for _ in range(100):
        h, w = random.choice(size_choices)
        ih, iw = h - 2, w - 2
        hole_mask = _build_hole_mask(ih, iw, hole_count)
        if hole_mask is None:
            continue
        patch = np.full((h, w), color, dtype=int)
        patch[1:-1, 1:-1][hole_mask] = BG
        if _count_holes_in_patch(patch) == hole_count:
            return patch
    raise RuntimeError("failed to build rectangle")


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    dr = 0
    if a[2] < b[0]:
        dr = b[0] - a[2]
    elif b[2] < a[0]:
        dr = a[0] - b[2]
    dc = 0
    if a[3] < b[1]:
        dc = b[1] - a[3]
    elif b[3] < a[1]:
        dc = a[1] - b[3]
    return dr + dc


def _place_top_pair(grid: np.ndarray, patches: list[np.ndarray]) -> list[tuple[int, int, int, int]] | None:
    h, w = grid.shape
    left, right = patches[0], patches[1]
    pair_w = left.shape[1] + 1 + right.shape[1]
    pair_h = max(left.shape[0], right.shape[0])
    if pair_w > w - 2 or pair_h > h - 6:
        return None
    top = random.randint(1, max(1, h // 4))
    c0 = random.randint(1, w - pair_w - 1)
    c1 = c0 + left.shape[1] + 1

    _paste(grid, left, top, c0)
    _paste(grid, right, top, c1)
    return [
        (top, c0, top + left.shape[0] - 1, c0 + left.shape[1] - 1),
        (top, c1, top + right.shape[0] - 1, c1 + right.shape[1] - 1),
    ]


def _slot_candidates(h: int, w: int, patch: np.ndarray, row_lo: int, row_hi: int, col_lo: int, col_hi: int) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    r_hi = min(row_hi, h - ph - 1)
    c_hi = min(col_hi, w - pw - 1)
    r_lo = max(1, row_lo)
    c_lo = max(1, col_lo)
    if r_lo > r_hi or c_lo > c_hi:
        return []
    return [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]


def _place_remaining_rects(grid: np.ndarray, patches: list[np.ndarray], bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]] | None:
    h, w = grid.shape
    top_bottom = max(bb[2] for bb in bboxes)
    slots = [
        (top_bottom + 3, h - 2, 1, w // 2 - 2),
        (top_bottom + 3, h - 2, w // 2 + 1, w - 2),
        (top_bottom + 2, h - 2, 1, w - 2),
    ]

    for patch in patches[2:]:
        placed = False
        order = slots[:]
        random.shuffle(order)
        for row_lo, row_hi, col_lo, col_hi in order:
            candidates = _slot_candidates(h, w, patch, row_lo, row_hi, col_lo, col_hi)
            random.shuffle(candidates)
            for r0, c0 in candidates:
                bb = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)
                if any(_bbox_distance(bb, old) < 3 for old in bboxes):
                    continue
                if np.any(grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] != BG):
                    continue
                _paste(grid, patch, r0, c0)
                bboxes.append(bb)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return bboxes


def _build_rectangles() -> tuple[np.ndarray, list[int]] | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    num_rects = random.choice((3, 4))
    hole_pair = random.choice(((1, 1), (2, 0), (0, 2), (2, 2)))
    hole_counts = [hole_pair[0], hole_pair[1], 1]
    if num_rects == 4:
        hole_counts.append(random.choice((0, 1, 2)))
    colors = random.sample(list(RECT_COLORS), num_rects)
    patches = [_make_rectangle(colors[i], hole_counts[i]) for i in range(num_rects)]

    grid = np.full((h, w), BG, dtype=int)
    bboxes = _place_top_pair(grid, patches)
    if bboxes is None:
        return None
    bboxes = _place_remaining_rects(grid, patches, bboxes)
    if bboxes is None:
        return None
    return grid, hole_counts


def _enumerate_candidates(rect_grid: np.ndarray) -> list[dict]:
    rects, rect_holes, _bars, _infos = _analyze(rect_grid, require_bars=False)
    rect_ids = np.full(rect_grid.shape, -1, dtype=int)
    for idx, rect in enumerate(rects):
        coords = rect["coords"]
        rect_ids[coords[:, 0], coords[:, 1]] = idx

    candidates = []
    h, w = rect_grid.shape
    for length in BAR_LENGTHS:
        half = length // 2
        for orient in ("H", "V"):
            for cr in range(half, h - half):
                for cc in range(half, w - half):
                    coords = _line_coords(cr, cc, length, orient)
                    if any(int(rect_grid[r, c]) != BG for r, c in coords):
                        continue
                    adjacent = set()
                    for r, c in coords:
                        for nr, nc in _neighbors4(r, c):
                            if 0 <= nr < h and 0 <= nc < w:
                                idx = int(rect_ids[nr, nc])
                                if idx >= 0:
                                    adjacent.add(idx)
                    if not adjacent:
                        continue
                    hole_sum = sum(rect_holes[idx] for idx in adjacent)
                    out_coords = coords if hole_sum % 2 == 0 else _rotated_coords(cr, cc, length, orient)
                    if any(int(rect_grid[r, c]) in RECT_COLORS for r, c in out_coords):
                        continue
                    candidates.append(
                        {
                            "input": set(coords),
                            "output": set(out_coords),
                            "adjacent": adjacent,
                            "hole_sum": hole_sum,
                            "rotates": bool(hole_sum % 2),
                            "length": length,
                            "orient": orient,
                            "center": (cr, cc),
                        }
                    )
    return candidates


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _compatible(chosen: list[dict], cand: dict) -> bool:
    for other in chosen:
        if _touches(other["input"], cand["input"]):
            return False
        if other["output"] & cand["output"]:
            return False
    return True


def _good_bar_set(chosen: list[dict], *, strict: bool = False) -> bool:
    odd = [c for c in chosen if c["rotates"]]
    even = [c for c in chosen if not c["rotates"]]
    if not odd or not even:
        return False
    if not any(len(c["adjacent"]) >= 2 for c in chosen):
        return False
    if not any((not c["rotates"]) and c["hole_sum"] > 0 for c in chosen):
        return False
    if not any(len(c["adjacent"]) == 1 for c in chosen):
        return False
    if not strict:
        return True
    if len({c["orient"] for c in chosen}) < 2:
        return False
    return any(a["orient"] == b["orient"] and a["rotates"] != b["rotates"] for i, a in enumerate(chosen) for b in chosen[i + 1 :])


def _select_bars(candidates: list[dict]) -> list[dict] | None:
    multi_even = [c for c in candidates if len(c["adjacent"]) >= 2 and not c["rotates"] and c["hole_sum"] > 0]
    odd_any = [c for c in candidates if c["rotates"]]
    if not multi_even or not odd_any:
        return None

    first_pool = multi_even[:]
    random.shuffle(first_pool)
    base_pool = candidates[:]
    random.shuffle(base_pool)

    def _rank_tail(pool: list[dict], chosen: list[dict]) -> list[dict]:
        orient_set = {c["orient"] for c in chosen}
        ranked = pool[:]
        ranked.sort(
            key=lambda c: (
                len(c["adjacent"]) != 1,
                c["hole_sum"] == 0,
                c["orient"] in orient_set,
                c["length"],
            )
        )
        return ranked

    for strict in (True, False):
        for first in first_pool:
            odd_pool = odd_any[:]
            random.shuffle(odd_pool)
            odd_pool.sort(key=lambda c: (c["orient"] != first["orient"], len(c["adjacent"]) != 1, c["length"]))
            for second in odd_pool:
                if not _compatible([first], second):
                    continue
                chosen2 = [first, second]
                third_pool = [c for c in _rank_tail(base_pool, chosen2) if c not in chosen2 and _compatible(chosen2, c)]
                for third in third_pool:
                    chosen3 = [first, second, third]
                    if not _good_bar_set(chosen3, strict=strict):
                        continue
                    want_four = random.random() < 0.45
                    if want_four:
                        fourth_pool = [
                            c
                            for c in _rank_tail(base_pool, chosen3)
                            if c not in chosen3 and _compatible(chosen3, c)
                        ]
                        for fourth in fourth_pool:
                            chosen4 = chosen3 + [fourth]
                            if _good_bar_set(chosen4, strict=strict):
                                return chosen4
                    return chosen3
    return None


def _validate_instance(inp: np.ndarray) -> None:
    out = main(inp)
    rects, rect_holes, bars, infos = _analyze(inp)

    if not (3 <= len(rects) <= 4):
        raise AssertionError("rectangle count out of range")
    if not (3 <= len(bars) <= 4):
        raise AssertionError("bar count out of range")
    if any(not (0 <= h <= 2) for h in rect_holes):
        raise AssertionError("invalid hole count")
    if 1 not in rect_holes:
        raise AssertionError("missing odd-hole rectangle")
    if not any(h % 2 == 0 and h > 0 for h in rect_holes) and rect_holes.count(1) < 2:
        raise AssertionError("missing even counterexample rectangle")
    if not _good_bar_set(infos):
        raise AssertionError("bar set is too degenerate")
    if np.array_equal(inp, out):
        raise AssertionError("no visible transform")

    rect_mask = np.isin(inp, RECT_COLORS)
    if not np.array_equal(inp[rect_mask], out[rect_mask]):
        raise AssertionError("rectangles changed")

    for i, info_a in enumerate(infos):
        for info_b in infos[i + 1 :]:
            if _touches(info_a["coords"], info_b["coords"]):
                raise AssertionError("bars touch in input")
            if info_a["output"] & info_b["output"]:
                raise AssertionError("bars overlap in output")


def _gen_attempt() -> np.ndarray | None:
    built = _build_rectangles()
    if built is None:
        return None
    rect_grid, _hole_counts = built
    candidates = _enumerate_candidates(rect_grid)
    chosen = _select_bars(candidates)
    if chosen is None:
        return None

    inp = rect_grid.copy()
    for cand in chosen:
        for r, c in cand["input"]:
            inp[r, c] = BAR

    try:
        _validate_instance(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
