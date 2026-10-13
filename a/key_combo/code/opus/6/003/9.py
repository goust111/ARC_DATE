import random
from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


BG = 0
TOP_COLOR = 1
BOTTOM_COLOR = 2
SEP_COLOR = 4

Coord = Tuple[int, int]

TOP_PARTIAL_SHAPES = [
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
]

BOTTOM_BLOCKED_SHAPES = [
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
]


def neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_bfs(mask: np.ndarray) -> List[List[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: List[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _find_separator(grid: np.ndarray) -> Tuple[int, List[int]]:
    h, w = grid.shape
    candidates: List[Tuple[int, List[int]]] = []
    for r in range(h):
        row = grid[r]
        if np.all((row == SEP_COLOR) | (row == BG)):
            gaps = [c for c in range(w) if int(row[c]) == BG]
            if 2 <= len(gaps) <= 3 and len(gaps) + int(np.count_nonzero(row == SEP_COLOR)) == w:
                candidates.append((r, gaps))
    if len(candidates) != 1:
        raise ValueError("expected exactly one horizontal separator row with 2-3 gaps")
    return candidates[0]


def _sorted_coords(coords: Sequence[Coord]) -> Tuple[Coord, ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _body_records(grid: np.ndarray) -> List[Dict]:
    sep_r, gaps = _find_separator(grid)
    bodies: List[Dict] = []
    for color in (TOP_COLOR, BOTTOM_COLOR):
        for coords in _component_bfs(grid == color):
            rows = [r for r, _ in coords]
            cols = sorted({c for _, c in coords})
            area = len(coords)
            record = {
                "color": color,
                "coords": _sorted_coords(coords),
                "area": area,
                "rows": rows,
                "cols": cols,
            }
            if area == 1:
                record["kind"] = "distractor"
            else:
                if max(rows) < sep_r:
                    record["kind"] = "active"
                    record["side"] = "top"
                    record["gravity"] = 1
                    record["distance"] = sep_r - max(rows)
                elif min(rows) > sep_r:
                    record["kind"] = "active"
                    record["side"] = "bottom"
                    record["gravity"] = -1
                    record["distance"] = min(rows) - sep_r
                else:
                    raise ValueError("active body may not straddle the separator in input")
            bodies.append(record)
    return bodies


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    h, w = g.shape
    sep_r, gaps = _find_separator(g)
    gap_set = set(gaps)
    records = _body_records(g)
    active = [rec for rec in records if rec["kind"] == "active"]
    distractors = [rec for rec in records if rec["kind"] == "distractor"]
    order = sorted(
        active,
        key=lambda rec: (
            int(rec["distance"]),
            0 if rec["side"] == "top" else 1,
            min(r for r, _ in rec["coords"]),
            min(c for _, c in rec["coords"]),
        ),
    )

    fixed_separator: Set[Coord] = {(sep_r, c) for c in range(w) if c not in gap_set}
    active_occupied: Set[Coord] = set()
    placed: List[Dict] = []

    for rec in order:
        coords = list(rec["coords"])
        dr = int(rec["gravity"])
        current = coords[:]
        crossed = False
        stop_reason = "boundary"

        while True:
            trial = [(r + dr, c) for r, c in current]
            if any(r < 0 or r >= h for r, _ in trial):
                stop_reason = "boundary"
                break
            sep_hits = [cell for cell in trial if cell in fixed_separator]
            body_hits = [cell for cell in trial if cell in active_occupied]
            if sep_hits or body_hits:
                stop_reason = "body" if body_hits else "separator"
                break
            current = trial
            if rec["side"] == "top" and any(r > sep_r for r, _ in current):
                crossed = True
            if rec["side"] == "bottom" and any(r < sep_r for r, _ in current):
                crossed = True

        final_color = rec["color"]
        if crossed:
            final_color = BOTTOM_COLOR if rec["side"] == "top" else TOP_COLOR

        for cell in current:
            active_occupied.add(cell)
        placed.append(
            {
                "original": rec["coords"],
                "side": rec["side"],
                "distance": int(rec["distance"]),
                "columns": tuple(rec["cols"]),
                "crossed": crossed,
                "stop_reason": stop_reason,
                "final_color": int(final_color),
                "final_coords": _sorted_coords(current),
            }
        )

    out = np.full_like(g, BG)
    out[sep_r, :] = SEP_COLOR
    out[sep_r, gaps] = BG

    for rec in distractors:
        r, c = rec["coords"][0]
        if out[r, c] != BG:
            raise ValueError("distractor overlaps a non-background fixed structure")
        out[r, c] = rec["color"]

    for rec in placed:
        color = int(rec["final_color"])
        for r, c in rec["final_coords"]:
            if out[r, c] not in (BG,):
                raise ValueError("active body overlaps fixed pixel in output")
            out[r, c] = color

    if return_meta:
        return out, {"separator_row": sep_r, "gaps": tuple(gaps), "active": placed, "distractors": distractors}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move each non-singleton body toward its panel's gravity direction, letting only gap-aligned bodies cross and recolor.

    Input structure:
    - The grid has black background and one full-width yellow horizontal separator row with 2-3 one-cell gaps.
    - Above the separator are blue 4-connected bodies plus scattered red singleton distractors; below it are red 4-connected bodies plus scattered blue singleton distractors.

    Rule:
    - Treat every monochrome 4-connected component of area 1 as a fixed distractor and every component of area at least 2 as an active rigid body.
    - Process active bodies from nearest to the separator to farthest; top-panel bodies fall downward and bottom-panel bodies rise upward until the next step would hit a solid separator cell, a previously stopped active body, or the grid boundary.
    - If all columns occupied by a body coincide with separator gaps, the body passes through the separator and continues in the same gravity direction; once it has crossed, recolor it to the opposite panel's active color.

    Key insight:
    - Solving requires separating singleton noise from real bodies, using panel membership to choose gravity, and using the body's whole column set, not just one lucky pixel, to decide whether crossing and recoloring are allowed.
    """
    return _solve(grid)


def _pick_gap_columns(width: int, count: int) -> List[int]:
    for _ in range(200):
        cols = sorted(random.sample(range(2, width - 2), count))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(len(cols) - 1)):
            return cols
    raise RuntimeError("failed to place separator gaps")


def _top_bar(sep_r: int, col: int, dist: int, length: int) -> List[Coord]:
    max_r = sep_r - dist
    return [(max_r - length + 1 + i, col) for i in range(length)]


def _bottom_bar(sep_r: int, col: int, dist: int, length: int) -> List[Coord]:
    min_r = sep_r + dist
    return [(min_r + i, col) for i in range(length)]


def _anchor_shape(shape: Sequence[Coord], top_r: int, left_c: int) -> List[Coord]:
    return [(top_r + dr, left_c + dc) for dr, dc in shape]


def _top_partial(sep_r: int, gap_col: int, dist: int) -> List[Coord]:
    shape = random.choice(TOP_PARTIAL_SHAPES)
    max_r = sep_r - dist
    top_r = max_r - max(dr for dr, _ in shape)
    return _anchor_shape(shape, top_r, gap_col)


def _bottom_blocked(sep_r: int, left_c: int, dist: int) -> List[Coord]:
    shape = random.choice(BOTTOM_BLOCKED_SHAPES)
    min_r = sep_r + dist
    top_r = min_r
    return _anchor_shape(shape, top_r, left_c)


def _in_bounds(shape: Tuple[int, int], cells: Sequence[Coord]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _separated(existing: Sequence[Sequence[Coord]], cells: Sequence[Coord]) -> bool:
    cell_set = set(cells)
    for other in existing:
        other_set = set(other)
        if cell_set & other_set:
            return False
        for r, c in cell_set:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in other_set:
                    return False
    return True


def _paint(grid: np.ndarray, cells: Sequence[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _forbidden_distractor_cells(inp: np.ndarray, out: np.ndarray) -> Set[Coord]:
    blocked: Set[Coord] = set()
    h, w = inp.shape
    for r in range(h):
        for c in range(w):
            if inp[r, c] != BG or out[r, c] != BG:
                blocked.add((r, c))
    return blocked


def _scatter_singletons(inp: np.ndarray, out: np.ndarray, sep_r: int) -> None:
    h, w = inp.shape
    forbidden = _forbidden_distractor_cells(inp, out)
    top_candidates = [(r, c) for r in range(sep_r) for c in range(w) if (r, c) not in forbidden]
    bot_candidates = [(r, c) for r in range(sep_r + 1, h) for c in range(w) if (r, c) not in forbidden]
    random.shuffle(top_candidates)
    random.shuffle(bot_candidates)

    top_count = random.randint(2, 4)
    bot_count = random.randint(2, 4)

    placed: Set[Coord] = set()

    def can_use(cell: Coord) -> bool:
        if cell in placed:
            return False
        return all(nbr not in placed for nbr in neighbors4(*cell))

    used = 0
    for cell in top_candidates:
        if used >= top_count:
            break
        if can_use(cell):
            inp[cell] = BOTTOM_COLOR
            out[cell] = BOTTOM_COLOR
            placed.add(cell)
            used += 1
    if used < top_count:
        raise RuntimeError("failed to place top distractors")

    used = 0
    for cell in bot_candidates:
        if used >= bot_count:
            break
        if can_use(cell):
            inp[cell] = TOP_COLOR
            out[cell] = TOP_COLOR
            placed.add(cell)
            used += 1
    if used < bot_count:
        raise RuntimeError("failed to place bottom distractors")


def _active_signature(coords: Sequence[Coord]) -> Tuple[Coord, ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _validate_scene(inp: np.ndarray, plan: Dict[str, Dict]) -> bool:
    try:
        out, meta = _solve(inp, return_meta=True)
    except ValueError:
        return False
    active_map = {tuple(rec["original"]): rec for rec in meta["active"]}
    gap_set = set(meta["gaps"])

    if len(active_map) != len(plan):
        return False

    distances = [rec["distance"] for rec in meta["active"]]
    if len(distances) != len(set(distances)):
        return False

    if not any(rec["crossed"] for rec in meta["active"] if rec["side"] == "top"):
        return False
    if not any(rec["crossed"] for rec in meta["active"] if rec["side"] == "bottom"):
        return False
    if not any(rec["stop_reason"] == "body" for rec in meta["active"]):
        return False

    top_partial_ok = False
    bottom_blocked_ok = False
    for original, spec in plan.items():
        rec = active_map.get(original)
        if rec is None:
            return False
        if bool(rec["crossed"]) != bool(spec["should_cross"]):
            return False
        cols = set(rec["columns"])
        if spec["kind"] == "top_partial":
            if cols & gap_set and not cols <= gap_set and rec["stop_reason"] == "separator":
                top_partial_ok = True
        if spec["kind"] == "bottom_blocked":
            if cols.isdisjoint(gap_set) and rec["stop_reason"] == "separator":
                bottom_blocked_ok = True
    if not top_partial_ok or not bottom_blocked_ok:
        return False

    # Ensure the visible transformation is non-degenerate.
    if np.array_equal(inp, out):
        return False
    if int(np.count_nonzero(inp != out)) < 10:
        return False
    return True


def _build_scene() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(14, 19)
    sep_r = h // 2
    gap_count = random.randint(2, 3)
    gaps = _pick_gap_columns(w, gap_count)

    dist_pool = [1, 2, 3, 4, 5]
    if sep_r < 7 or h - sep_r - 1 < 7:
        return None

    top_near = 1
    top_partial_dist = 2
    bottom_cross_dist = 3
    bottom_block_dist = 4
    top_far = 5

    g1 = gaps[0]
    g2 = gaps[-1]
    partial_neighbor = g2 + 1 if g2 + 1 < w - 1 and g2 + 1 not in gaps else g2 - 1
    if partial_neighbor in gaps or partial_neighbor <= 0 or partial_neighbor >= w - 1:
        return None

    non_gap_cols = [c for c in range(2, w - 2) if c not in gaps]
    blocked_left_choices = [c for c in non_gap_cols if c not in {g1 - 1, g1, g1 + 1, g2 - 1, g2, g2 + 1}]
    if not blocked_left_choices:
        return None
    blocked_left = random.choice(blocked_left_choices)

    top_len_a = random.randint(3, 4)
    top_len_b = random.randint(3, 4)
    bot_len = random.randint(3, 4)

    top_a = _top_bar(sep_r, g1, top_near, top_len_a)
    top_b = _top_bar(sep_r, g1, top_far, top_len_b)
    top_c = _top_partial(sep_r, min(g2, partial_neighbor), top_partial_dist)
    bottom_d = _bottom_bar(sep_r, g2, bottom_cross_dist, bot_len)
    bottom_e = _bottom_blocked(sep_r, blocked_left, bottom_block_dist)

    active_cells = [top_a, top_b, top_c, bottom_d, bottom_e]
    if not all(_in_bounds((h, w), cells) for cells in active_cells):
        return None
    existing: List[List[Coord]] = []
    for cells in active_cells:
        if not _separated(existing, cells):
            return None
        existing.append(list(cells))

    inp = np.full((h, w), BG, dtype=int)
    inp[sep_r, :] = SEP_COLOR
    inp[sep_r, gaps] = BG
    _paint(inp, top_a, TOP_COLOR)
    _paint(inp, top_b, TOP_COLOR)
    _paint(inp, top_c, TOP_COLOR)
    _paint(inp, bottom_d, BOTTOM_COLOR)
    _paint(inp, bottom_e, BOTTOM_COLOR)

    try:
        out = _solve(inp)
    except ValueError:
        return None
    _scatter_singletons(inp, out, sep_r)

    plan = {
        _active_signature(top_a): {"kind": "top_cross_a", "should_cross": True},
        _active_signature(top_b): {"kind": "top_cross_b", "should_cross": True},
        _active_signature(top_c): {"kind": "top_partial", "should_cross": False},
        _active_signature(bottom_d): {"kind": "bottom_cross", "should_cross": True},
        _active_signature(bottom_e): {"kind": "bottom_blocked", "should_cross": False},
    }
    if not _validate_scene(inp, plan):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
