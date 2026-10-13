import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells)


def _find_cavity(g: np.ndarray) -> set[tuple[int, int]]:
    holes = []
    for comp in _components(g == BG):
        if _touches_border(comp, *g.shape):
            continue
        holes.append(comp)
    assert holes, "grid must contain an enclosed black cavity"
    return set(max(holes, key=len))


def _cavity_boundary(cavity: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    boundary = set()
    for r, c in cavity:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cavity:
                boundary.add((nr, nc))
    return boundary


def _entry_direction(entry: tuple[int, int], cavity: set[tuple[int, int]], g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    opts = []
    er, ec = entry
    for dr, dc in DIRS4:
        inside = (er + dr, ec + dc)
        outside = (er - dr, ec - dc)
        if inside not in cavity:
            continue
        if not (0 <= outside[0] < h and 0 <= outside[1] < w):
            continue
        if int(g[outside]) != BG:
            continue
        opts.append((dr, dc))
    assert len(opts) == 1, "each colored boundary point must define a unique entrance axis"
    return opts[0]


def _arrow_direction(arrow: tuple[int, int], fragment: set[tuple[int, int]], g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    ar, ac = arrow
    opts = []
    for dr, dc in DIRS4:
        gap = (ar + dr, ac + dc)
        hit = (ar + 2 * dr, ac + 2 * dc)
        if not (0 <= gap[0] < h and 0 <= gap[1] < w and 0 <= hit[0] < h and 0 <= hit[1] < w):
            continue
        if int(g[gap]) != BG:
            continue
        if hit in fragment:
            opts.append((dr, dc))
    assert len(opts) == 1, "arrow position must determine exactly one slide direction"
    return opts[0]


def _distance_to_entry(
    fragment: set[tuple[int, int]], entry: tuple[int, int], direction: tuple[int, int]
) -> int:
    er, ec = entry
    dr, dc = direction
    vals = []
    for r, c in fragment:
        if dr != 0:
            if c != ec:
                continue
            delta = er - r
            if delta * dr > 0:
                vals.append(abs(delta))
        else:
            if r != er:
                continue
            delta = ec - c
            if delta * dc > 0:
                vals.append(abs(delta))
    assert vals, "fragment must align with its entrance along the arrow axis"
    return min(vals)


def _shift(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _simulate_piece(
    start: set[tuple[int, int]],
    direction: tuple[int, int],
    entry: tuple[int, int],
    blocked: np.ndarray,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = blocked.shape
    dr, dc = direction
    cur = set(start)
    while True:
        nxt = _shift(cur, dr, dc)
        ok = True
        for r, c in nxt:
            if not (0 <= r < h and 0 <= c < w):
                ok = False
                break
            if (r, c) in occupied:
                ok = False
                break
            if blocked[r, c] and (r, c) != entry:
                ok = False
                break
        if not ok:
            return cur
        cur = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored fragments through matching border dots and pack them into the grey outline.

    Input structure:
    - Background is black.
    - A central closed grey outline surrounds one enclosed black cavity.
    - Several colored fragments sit outside the outline; each fragment has a same-colored single-pixel arrow
      separated from it by one black cell on the side opposite the movement direction.
    - The grey border contains one same-colored entry dot for each fragment.

    Rule:
    - For each fragment, infer the slide direction from the arrow position and match it to the same-colored
      entry dot on the outline.
    - Order the fragments by the number of translation steps needed for the fragment to first reach its entry.
    - Slide each fragment rigidly along its arrow direction; its own entry dot is temporarily passable, while
      grey border pixels and already placed fragments block motion.
    - Erase the original outside fragments and arrows, keep the fragment colors where they stop inside the cavity,
      and restore every colored entry dot back to grey.

    Key insight:
    - The puzzle combines two linked inferences: arrow placement encodes motion direction, and the final layout
      depends on processing the fragments in increasing outside-to-entry distance because earlier placements become
      obstacles for later ones.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    cavity = _find_cavity(g)
    boundary = _cavity_boundary(cavity, h, w)

    entries: dict[int, dict[str, object]] = {}
    for pos in boundary:
        col = int(g[pos])
        if col == WALL:
            continue
        if col == BG:
            continue
        entries[col] = {"entry": pos, "dir": _entry_direction(pos, cavity, g)}
    assert entries, "there must be at least one colored entrance"

    pieces = []
    for col, info in entries.items():
        comps = [set(comp) for comp in _components(g == col)]
        entry = info["entry"]
        external = [comp for comp in comps if entry not in comp]
        assert len(external) == 2, "each color should have one fragment and one arrow outside the contour"
        arrow_comp = [comp for comp in external if len(comp) == 1]
        frag_comp = [comp for comp in external if len(comp) > 1]
        assert len(arrow_comp) == 1 and len(frag_comp) == 1
        arrow = next(iter(arrow_comp[0]))
        fragment = frag_comp[0]
        direction = _arrow_direction(arrow, fragment, g)
        assert direction == info["dir"], "arrow direction must agree with the matched entry direction"
        pieces.append(
            {
                "color": col,
                "entry": entry,
                "dir": direction,
                "start": fragment,
                "dist": _distance_to_entry(fragment, entry, direction),
            }
        )

    pieces.sort(key=lambda item: int(item["dist"]))

    out = np.full_like(g, BG)
    for r, c in boundary:
        out[r, c] = WALL
    blocked = out != BG

    occupied: set[tuple[int, int]] = set()
    for piece in pieces:
        final_cells = _simulate_piece(
            piece["start"], piece["dir"], piece["entry"], blocked, occupied
        )
        occupied |= final_cells
        for r, c in final_cells:
            out[r, c] = int(piece["color"])

    for info in entries.values():
        r, c = info["entry"]
        out[r, c] = WALL
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rotate90(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(-c, r) for r, c in cells}


def _random_transform(cells: set[tuple[int, int]], rng: random.Random) -> set[tuple[int, int]]:
    cur = set(cells)
    for _ in range(rng.randrange(4)):
        cur = _rotate90(cur)
    if rng.random() < 0.5:
        cur = {(r, -c) for r, c in cur}
    return _normalize(cur)


def _grow_cavity(rng: random.Random) -> set[tuple[int, int]] | None:
    target = rng.randint(8, 15)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                new_cells = cells | {(nr, nc)}
                rows = [x for x, _ in new_cells]
                cols = [y for _, y in new_cells]
                if max(rows) - min(rows) + 1 > 6 or max(cols) - min(cols) + 1 > 6:
                    continue
                frontier.append((nr, nc))
        if not frontier:
            return None
        rng.shuffle(frontier)
        frontier.sort(key=lambda p: -sum((p[0] + dr, p[1] + dc) in cells for dr, dc in DIRS4))
        pick = frontier[0] if rng.random() < 0.65 else rng.choice(frontier)
        cells.add(pick)
    cells = _random_transform(cells, rng)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if max(rows) + 1 < 3 or max(cols) + 1 < 3:
        return None
    if len(cells) == (max(rows) + 1) * (max(cols) + 1):
        return None
    return cells


def _entry_candidates(cavity: set[tuple[int, int]]) -> list[dict[str, object]]:
    cset = set(cavity)
    cands = []
    seen = set()
    for r, c in cset:
        for dr, dc in DIRS4:
            entry = (r - dr, c - dc)
            if entry in cset:
                continue
            if sum((entry[0] + rr, entry[1] + cc) in cset for rr, cc in DIRS4) != 1:
                continue
            lane = []
            rr, cc = r, c
            while (rr, cc) in cset:
                lane.append((rr, cc))
                rr += dr
                cc += dc
            if len(lane) < 2:
                continue
            key = (entry, dr, dc)
            if key in seen:
                continue
            seen.add(key)
            cands.append(
                {
                    "entry": entry,
                    "dir": (dr, dc),
                    "lane": tuple(lane),
                    "lane_set": set(lane),
                }
            )
    return cands


def _lane_prefix(
    lane: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    prefix = []
    for cell in lane:
        if cell in occupied:
            break
        prefix.append(cell)
    return tuple(prefix)


def _select_plan(
    candidates: list[dict[str, object]], rng: random.Random
) -> list[dict[str, object]] | None:
    if len(candidates) < 2:
        return None
    mode = rng.choice(["independent", "blocking"])
    target = rng.randint(2, min(4, len(candidates), len(PIECE_COLORS)))
    occupied: set[tuple[int, int]] = set()
    used_entries = set()
    selected = []
    blocking_seen = False
    for step in range(target):
        options = []
        idxs = list(range(len(candidates)))
        rng.shuffle(idxs)
        for idx in idxs:
            cand = candidates[idx]
            entry = cand["entry"]
            if entry in used_entries:
                continue
            prefix = _lane_prefix(cand["lane"], occupied)
            if len(prefix) < 2:
                continue
            for length in range(2, min(4, len(prefix)) + 1):
                final = prefix[-length:]
                block_effect = final != cand["lane"][-length:]
                overlap = bool(cand["lane_set"] & occupied)
                options.append(
                    {
                        "cand": cand,
                        "length": length,
                        "final": final,
                        "block_effect": block_effect,
                        "overlap": overlap,
                    }
                )
        if not options:
            break
        preferred = options
        if mode == "blocking" and step > 0 and not blocking_seen:
            blocking = [opt for opt in options if opt["block_effect"]]
            if blocking:
                preferred = blocking
        elif mode == "independent":
            clean = [opt for opt in options if not opt["overlap"] and not opt["block_effect"]]
            if clean:
                preferred = clean
        choice = rng.choice(preferred)
        selected.append(choice)
        used_entries.add(choice["cand"]["entry"])
        occupied.update(choice["final"])
        blocking_seen = blocking_seen or bool(choice["block_effect"])
    if len(selected) != target:
        return None
    if mode == "blocking" and not blocking_seen:
        return None
    return selected


def _boundary_ring(cavity: set[tuple[int, int]]) -> set[tuple[int, int]]:
    ring = set()
    for r, c in cavity:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in cavity:
                ring.add((nr, nc))
    return ring


def _piece_start(entry: tuple[int, int], direction: tuple[int, int], length: int, dist: int):
    dr, dc = direction
    cells = []
    for k in range(length):
        cells.append((entry[0] - (dist + k) * dr, entry[1] - (dist + k) * dc))
    tail = cells[-1]
    arrow = (tail[0] - 2 * dr, tail[1] - 2 * dc)
    return tuple(cells), arrow


def _build_grid(
    cavity: set[tuple[int, int]], plan: list[dict[str, object]], rng: random.Random
) -> np.ndarray | None:
    colors = rng.sample(PIECE_COLORS, len(plan))
    dists = sorted(rng.sample(range(2, 9), len(plan)))
    boundary = _boundary_ring(cavity)

    pieces = []
    occupied_owner: dict[tuple[int, int], int] = {cell: -1 for cell in boundary}
    for idx, item in enumerate(plan):
        cand = item["cand"]
        start_cells, arrow = _piece_start(cand["entry"], cand["dir"], item["length"], dists[idx])
        frag_set = set(start_cells)
        if frag_set & boundary or arrow in boundary or arrow in cavity:
            return None
        owner = idx + 1
        for cell in frag_set:
            if cell in cavity or cell in occupied_owner:
                return None
        if arrow in occupied_owner:
            return None
        cells_all = frag_set | {arrow}
        for cell in cells_all:
            for nr, nc in _neighbors4(*cell):
                other = occupied_owner.get((nr, nc))
                if other is not None and other != owner:
                    return None
        for cell in frag_set:
            occupied_owner[cell] = owner
        occupied_owner[arrow] = owner
        pieces.append(
            {
                "color": colors[idx],
                "entry": cand["entry"],
                "dir": cand["dir"],
                "fragment": frag_set,
                "arrow": arrow,
            }
        )

    all_rows = [r for r, _ in boundary]
    all_cols = [c for _, c in boundary]
    for piece in pieces:
        for r, c in piece["fragment"] | {piece["arrow"]}:
            all_rows.append(r)
            all_cols.append(c)
    min_r, max_r = min(all_rows), max(all_rows)
    min_c, max_c = min(all_cols), max(all_cols)
    need_h = max_r - min_r + 1
    need_w = max_c - min_c + 1
    if need_h > 28 or need_w > 28:
        return None

    min_h = max(need_h + 2, 14)
    min_w = max(need_w + 2, 14)
    max_h = min(30, need_h + 8)
    max_w = min(30, need_w + 8)
    if min_h > max_h or min_w > max_w:
        return None
    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    shift_r = rng.randint(1 - min_r, h - 2 - max_r)
    shift_c = rng.randint(1 - min_c, w - 2 - max_c)

    grid = np.full((h, w), BG, dtype=int)
    for r, c in boundary:
        grid[r + shift_r, c + shift_c] = WALL
    for piece in pieces:
        er, ec = piece["entry"]
        grid[er + shift_r, ec + shift_c] = int(piece["color"])
        for r, c in piece["fragment"]:
            grid[r + shift_r, c + shift_c] = int(piece["color"])
        ar, ac = piece["arrow"]
        grid[ar + shift_r, ac + shift_c] = int(piece["color"])

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cavity = _grow_cavity(rng)
    if cavity is None:
        return None
    candidates = _entry_candidates(cavity)
    if len(candidates) < 2:
        return None
    rng.shuffle(candidates)
    plan = _select_plan(candidates, rng)
    if plan is None:
        return None
    return _build_grid(cavity, plan, rng)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
