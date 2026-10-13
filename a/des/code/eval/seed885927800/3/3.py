import random
from collections import Counter, defaultdict, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERPENDICULAR = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}

BG = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 24
GRID_MAX = 30
GROUP_COUNT_RANGE = (2, 4)
CHAIN_RETRIES = 400
GEN_RETRIES = 400
PATH_MARGIN = 3
FRAME_INNER_RANGE = range(3, 7)


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components_for_color(grid: np.ndarray, color: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rect_frame(component, shape) -> bool:
    h, w = shape
    r0, c0, r1, c1 = _bbox(component)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
        return False
    want = set()
    for r in range(r0, r1 + 1):
        want.add((r, c0))
        want.add((r, c1))
    for c in range(c0, c1 + 1):
        want.add((r0, c))
        want.add((r1, c))
    return set(component) == want


def _fragment_signature(component):
    pts = sorted(set(component))
    if len(pts) < 2:
        return None
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    if len(rows) == 1:
        r = rows[0]
        seq = sorted(c for _, c in pts)
        if seq != list(range(seq[0], seq[-1] + 1)):
            return None
        return {
            "orientation": "h",
            "endpoints": [((r, seq[0]), (r, seq[0] - 1)), ((r, seq[-1]), (r, seq[-1] + 1))],
            "length": len(seq),
        }
    if len(cols) == 1:
        c = cols[0]
        seq = sorted(r for r, _ in pts)
        if seq != list(range(seq[0], seq[-1] + 1)):
            return None
        return {
            "orientation": "v",
            "endpoints": [((seq[0], c), (seq[0] - 1, c)), ((seq[-1], c), (seq[-1] + 1, c))],
            "length": len(seq),
        }
    return None


def _frame_slots(frame_bbox):
    r0, c0, r1, c1 = frame_bbox
    slots = []
    for r in range(r0 + 1, r1, 2):
        for c in range(c0 + 1, c1, 2):
            slots.append((r, c))
    return slots


def _count_turns(path_cells) -> int:
    cells = set(path_cells)
    turns = 0
    for r, c in cells:
        neigh = []
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                neigh.append((nr - r, nc - c))
        if len(neigh) != 2:
            continue
        if neigh[0][0] + neigh[1][0] != 0 or neigh[0][1] + neigh[1][1] != 0:
            turns += 1
    return turns


def _solve_details(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = g.copy()
    details = {}

    for color in [int(c) for c in np.unique(g) if int(c) != int(bg)]:
        comps = _components_for_color(g, color)
        frame_candidates = [comp for comp in comps if _is_rect_frame(comp, (h, w))]
        if not frame_candidates:
            continue
        frame = max(frame_candidates, key=len)
        fragments = [comp for comp in comps if set(comp) != set(frame)]
        if not fragments:
            continue

        gap_map = defaultdict(list)
        valid_fragments = []
        for frag_id, comp in enumerate(fragments):
            sig = _fragment_signature(comp)
            if sig is None:
                continue
            valid_fragments.append(comp)
            for endpoint, gap in sig["endpoints"]:
                gr, gc = gap
                if 0 <= gr < h and 0 <= gc < w and int(g[gr, gc]) == int(bg):
                    gap_map[(int(gr), int(gc))].append((int(frag_id), endpoint))

        fill_gaps = []
        for gap, recs in gap_map.items():
            if len(recs) == 2 and recs[0][0] != recs[1][0]:
                fill_gaps.append(gap)
                out[gap] = int(color)

        path_cells = set(fill_gaps)
        for comp in valid_fragments:
            path_cells.update(comp)
        turns = _count_turns(path_cells)

        frame_bbox = _bbox(frame)
        slots = _frame_slots(frame_bbox)
        for slot in slots[:turns]:
            out[slot] = int(color)

        details[int(color)] = {
            "frame_bbox": frame_bbox,
            "gaps": sorted(fill_gaps),
            "turns": int(turns),
            "slot_count": len(slots),
        }

    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect same-color line fragments, count turns in the resulting polyline,
    and display the count as spaced dots inside the matching edge frame.
    """
    out, _details = _solve_details(np.asarray(grid, dtype=int))
    return out


def _normalize_chain(fragments, gap_cells, ordered_cells):
    all_cells = [cell for frag in fragments for cell in frag] + list(gap_cells)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)

    def shift(cell):
        return cell[0] - min_r, cell[1] - min_c

    norm_fragments = [[shift(cell) for cell in frag] for frag in fragments]
    norm_gaps = [shift(cell) for cell in gap_cells]
    norm_ordered = [shift(cell) for cell in ordered_cells]
    rows = [r for r, _ in norm_ordered]
    cols = [c for _, c in norm_ordered]
    return norm_fragments, norm_gaps, norm_ordered, (max(rows) + 1, max(cols) + 1)


def _components_from_cells(cells):
    remaining = set(cells)
    comps = []
    while remaining:
        start = next(iter(remaining))
        q = deque([start])
        comp = set([start])
        remaining.remove(start)
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _build_chain_spec(rng: random.Random):
    for _ in range(CHAIN_RETRIES):
        frag_count = rng.randint(2, 5)
        lengths = [rng.randint(2, 5) for _ in range(frag_count)]
        turns_mask = [rng.choice([0, 1]) for _ in range(frag_count - 1)]
        direction = rng.choice(DIRS4)
        pos = (0, 0)
        fragments = []
        gap_cells = []
        ordered = []

        for idx in range(frag_count):
            frag = []
            for step in range(lengths[idx]):
                frag.append((pos[0] + step * direction[0], pos[1] + step * direction[1]))
            fragments.append(frag)
            ordered.extend(frag)

            if idx == frag_count - 1:
                continue
            gap = (frag[-1][0] + direction[0], frag[-1][1] + direction[1])
            gap_cells.append(gap)
            ordered.append(gap)
            if turns_mask[idx]:
                direction = rng.choice(PERPENDICULAR[direction])
            pos = (gap[0] + direction[0], gap[1] + direction[1])

        if len(set(ordered)) != len(ordered):
            continue

        full_cells = set(ordered)
        degs = []
        bad = False
        for r, c in full_cells:
            degree = sum((nr, nc) in full_cells for nr, nc in _neighbors4(r, c))
            degs.append(int(degree))
            if degree not in (1, 2):
                bad = True
                break
        if bad or degs.count(1) != 2:
            continue

        norm_fragments, norm_gaps, norm_ordered, shape = _normalize_chain(fragments, gap_cells, ordered)
        if shape[0] > 12 or shape[1] > 12:
            continue

        actual_cells = {cell for frag in norm_fragments for cell in frag}
        if len(actual_cells) != sum(len(frag) for frag in norm_fragments):
            continue
        if len(_components_from_cells(actual_cells)) != frag_count:
            continue

        gap_map = defaultdict(list)
        ok = True
        for frag_id, frag in enumerate(norm_fragments):
            sig = _fragment_signature(frag)
            if sig is None or not (2 <= sig["length"] <= 5):
                ok = False
                break
            for _endpoint, gap in sig["endpoints"]:
                gap_map[gap].append(frag_id)
        if not ok:
            continue

        matched = {gap for gap, ids in gap_map.items() if len(ids) == 2 and ids[0] != ids[1]}
        if matched != set(norm_gaps):
            continue

        turns = _count_turns(actual_cells | set(norm_gaps))
        if turns != sum(turns_mask):
            continue

        return {
            "fragments": norm_fragments,
            "gaps": norm_gaps,
            "path": norm_ordered,
            "shape": shape,
            "turns": turns,
        }
    return None


def _expand_cells(cells, shape, radius=1):
    h, w = shape
    expanded = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _translate_cells(cells, dr, dc):
    return [(r + dr, c + dc) for r, c in cells]


def _frame_border_cells(frame_bbox):
    r0, c0, r1, c1 = frame_bbox
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _frame_bbox_cells(frame_bbox):
    r0, c0, r1, c1 = frame_bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _reserve_free(occupied, cells):
    return all(not occupied[r, c] for r, c in cells)


def _mark_occupied(occupied, cells):
    for r, c in cells:
        occupied[r, c] = True


def _paint_cells(grid, cells, color):
    for r, c in cells:
        grid[r, c] = int(color)


def _place_chain(grid, occupied, color, spec, rng: random.Random):
    h, w = grid.shape
    chain_h, chain_w = spec["shape"]
    min_r = PATH_MARGIN
    max_r = h - chain_h - PATH_MARGIN - 1
    min_c = PATH_MARGIN
    max_c = w - chain_w - PATH_MARGIN - 1
    if max_r < min_r or max_c < min_c:
        return None

    positions = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
    rng.shuffle(positions)

    path_cells = spec["path"]
    actual_cells = [cell for frag in spec["fragments"] for cell in frag]
    gap_cells = spec["gaps"]
    for top, left in positions:
        moved_path = _translate_cells(path_cells, top, left)
        moved_actual = _translate_cells(actual_cells, top, left)
        moved_gaps = _translate_cells(gap_cells, top, left)
        reserve = _expand_cells(moved_path, grid.shape, radius=1)
        if not _reserve_free(occupied, reserve):
            continue
        _paint_cells(grid, moved_actual, color)
        _mark_occupied(occupied, reserve)
        return {
            "fragments": [
                _translate_cells(frag, top, left) for frag in spec["fragments"]
            ],
            "gaps": moved_gaps,
            "turns": spec["turns"],
        }
    return None


def _candidate_frame_dims(turns, rng: random.Random):
    dims = []
    for inner_h in FRAME_INNER_RANGE:
        for inner_w in FRAME_INNER_RANGE:
            capacity = ((inner_h + 1) // 2) * ((inner_w + 1) // 2)
            if capacity >= turns:
                dims.append((inner_h, inner_w))
    rng.shuffle(dims)
    return dims


def _sample_frame_bbox(shape, frame_size, mode, rng: random.Random):
    h, w = shape
    frame_h, frame_w = frame_size
    if frame_h > h or frame_w > w:
        return None

    if mode == "top":
        c0 = rng.randint(0, w - frame_w)
        return 0, c0, frame_h - 1, c0 + frame_w - 1
    if mode == "bottom":
        c0 = rng.randint(0, w - frame_w)
        return h - frame_h, c0, h - 1, c0 + frame_w - 1
    if mode == "left":
        r0 = rng.randint(0, h - frame_h)
        return r0, 0, r0 + frame_h - 1, frame_w - 1
    if mode == "right":
        r0 = rng.randint(0, h - frame_h)
        return r0, w - frame_w, r0 + frame_h - 1, w - 1
    if mode == "tl":
        return 0, 0, frame_h - 1, frame_w - 1
    if mode == "tr":
        return 0, w - frame_w, frame_h - 1, w - 1
    if mode == "bl":
        return h - frame_h, 0, h - 1, frame_w - 1
    if mode == "br":
        return h - frame_h, w - frame_w, h - 1, w - 1
    return None


def _place_frame(grid, occupied, color, turns, rng: random.Random):
    modes = ["top", "bottom", "left", "right", "tl", "tr", "bl", "br"]
    dims = _candidate_frame_dims(turns, rng)
    for inner_h, inner_w in dims:
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        for _ in range(80):
            mode = rng.choice(modes)
            bbox = _sample_frame_bbox(grid.shape, (frame_h, frame_w), mode, rng)
            if bbox is None:
                continue
            bbox_cells = _frame_bbox_cells(bbox)
            reserve = _expand_cells(bbox_cells, grid.shape, radius=1)
            if not _reserve_free(occupied, reserve):
                continue
            border = _frame_border_cells(bbox)
            _paint_cells(grid, border, color)
            _mark_occupied(occupied, reserve)
            return bbox
    return None


def _validate_generated(grid: np.ndarray, placed_specs) -> bool:
    out, details = _solve_details(grid)
    _ = out
    for color, spec in placed_specs.items():
        if color not in details:
            return False
        found = details[color]
        if tuple(spec["frame_bbox"]) != tuple(found["frame_bbox"]):
            return False
        if sorted(spec["gaps"]) != found["gaps"]:
            return False
        if int(spec["turns"]) != int(found["turns"]):
            return False
        if int(found["slot_count"]) < int(found["turns"]):
            return False

        r0, c0, r1, c1 = spec["frame_bbox"]
        if not _is_rect_frame(list(_frame_border_cells((r0, c0, r1, c1))), grid.shape):
            return False
        if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG):
            return False
    return True


def _gen_attempt(rng: random.Random):
    group_count = rng.randint(*GROUP_COUNT_RANGE)
    colors = rng.sample(PALETTE, group_count)
    chains = []
    for _ in range(group_count):
        spec = _build_chain_spec(rng)
        if spec is None:
            return None
        chains.append(spec)

    min_side = max(GRID_MIN, 20 + 2 * group_count)
    if min_side > GRID_MAX:
        return None
    h = rng.randint(min_side, GRID_MAX)
    w = rng.randint(min_side, GRID_MAX)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placed_specs = {}

    order = list(range(group_count))
    rng.shuffle(order)
    for idx in order:
        color = colors[idx]
        placed = _place_chain(grid, occupied, color, chains[idx], rng)
        if placed is None:
            return None
        placed_specs[color] = placed

    frame_order = colors[:]
    rng.shuffle(frame_order)
    for color in frame_order:
        bbox = _place_frame(grid, occupied, color, placed_specs[color]["turns"], rng)
        if bbox is None:
            return None
        placed_specs[color]["frame_bbox"] = bbox

    if not _validate_generated(grid, placed_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
