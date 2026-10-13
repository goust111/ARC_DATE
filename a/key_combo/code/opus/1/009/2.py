import random
from functools import lru_cache

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
CYAN = 8

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_RETRIES = 300
_GEN_COUNTER = 0


def _find_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "X" for ch in row] for row in rows], dtype=bool)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            v = _trim_mask(np.rot90(base, k))
            key = tuple(map(tuple, v.astype(int)))
            if key not in seen:
                seen.add(key)
                variants.append(v)
    return variants


def _solve_component(mask: np.ndarray, origin_r: int, origin_c: int) -> dict:
    mask = np.asarray(mask, dtype=bool)
    cells = list(map(tuple, np.argwhere(mask)))
    cell_to_index = {cell: idx for idx, cell in enumerate(cells)}
    candidates = []

    for h, w, kind in ((2, 2, "square"), (1, 3, "rect"), (3, 1, "rect")):
        if mask.shape[0] < h or mask.shape[1] < w:
            continue
        for r in range(mask.shape[0] - h + 1):
            for c in range(mask.shape[1] - w + 1):
                patch = mask[r : r + h, c : c + w]
                if not bool(patch.all()):
                    continue
                bits = 0
                for rr in range(h):
                    for cc in range(w):
                        bits |= 1 << cell_to_index[(r + rr, c + cc)]
                candidates.append(
                    {
                        "bits": bits,
                        "kind": kind,
                        "r0": origin_r + r,
                        "c0": origin_c + c,
                        "h": h,
                        "w": w,
                        "local": (r, c, h, w),
                    }
                )

    by_cell = [[] for _ in cells]
    for cid, cand in enumerate(candidates):
        bits = int(cand["bits"])
        for idx in range(len(cells)):
            if (bits >> idx) & 1:
                by_cell[idx].append(cid)

    full_mask = (1 << len(cells)) - 1

    @lru_cache(None)
    def dfs(decided_mask: int) -> tuple[int, tuple[int, ...], int]:
        if decided_mask == full_mask:
            return 0, (), 1

        first = 0
        while (decided_mask >> first) & 1:
            first += 1

        best_cover, best_sol, best_count = dfs(decided_mask | (1 << first))
        for cid in by_cell[first]:
            bits = int(candidates[cid]["bits"])
            if bits & decided_mask:
                continue
            extra_cover, sol, count = dfs(decided_mask | bits)
            total_cover = bits.bit_count() + extra_cover
            total_sol = tuple(sorted((cid,) + sol))
            if total_cover > best_cover:
                best_cover, best_sol, best_count = total_cover, total_sol, count
            elif total_cover == best_cover:
                if total_sol < best_sol:
                    best_sol = total_sol
                best_count = min(2, best_count + count)
        return best_cover, best_sol, best_count

    covered_count, solution_ids, solution_count = dfs(0)
    covered_bits = 0
    primitives = []
    for cid in solution_ids:
        cand = candidates[cid]
        covered_bits |= int(cand["bits"])
        r0 = int(cand["r0"])
        c0 = int(cand["c0"])
        h = int(cand["h"])
        w = int(cand["w"])
        primitives.append(
            {
                "kind": cand["kind"],
                "r0": r0,
                "c0": c0,
                "r1": r0 + h - 1,
                "c1": c0 + w - 1,
                "h": h,
                "w": w,
            }
        )

    residual = []
    for idx, (r, c) in enumerate(cells):
        if not ((covered_bits >> idx) & 1):
            residual.append((origin_r + int(r), origin_c + int(c)))

    primitives.sort(key=lambda p: (p["r0"], p["c0"], p["h"], p["w"], p["kind"]))
    return {
        "area": int(mask.sum()),
        "covered_count": int(covered_count),
        "unique": bool(solution_count == 1),
        "primitives": primitives,
        "residual": residual,
    }


def _detect_bars(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_candidates = [r for r in range(h) if bool(np.all(g[r, :] == CYAN))]
    assert row_candidates, "expected one full-width cyan bar"
    bar_row = int(row_candidates[0])

    col_candidates = []
    for c in range(w):
        if g[bar_row, c] != CYAN:
            continue
        col = g[:, c]
        other_rows = np.delete(col, bar_row)
        if bool(np.all(other_rows == RED)):
            col_candidates.append(c)
    assert col_candidates, "expected one full-height red bar"
    return bar_row, int(col_candidates[0])


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bar_row, bar_col = _detect_bars(g)
    comps = _find_components(g == GRAY, connectivity=4)

    component_infos = []
    primitives = []
    for comp in comps:
        mask, r0, c0 = _coords_to_mask(comp)
        info = _solve_component(mask, r0, c0)
        component_infos.append(
            {
                "bbox": _bbox(comp),
                "area": info["area"],
                "covered_count": info["covered_count"],
                "unique": info["unique"],
                "residual": list(info["residual"]),
                "primitives": [dict(p) for p in info["primitives"]],
            }
        )
        primitives.extend(info["primitives"])

    return {
        "bar_row": int(bar_row),
        "bar_col": int(bar_col),
        "components": component_infos,
        "primitives": primitives,
    }


def _primitive_distance2(prim: dict, bar_row: int, bar_col: int) -> int:
    if prim["kind"] == "square":
        return abs((prim["r0"] + prim["r1"]) - 2 * bar_row)
    return abs((prim["c0"] + prim["c1"]) - 2 * bar_col)


def _primitive_color(prim: dict) -> int:
    return CYAN if prim["kind"] == "square" else RED


def _primitive_step(prim: dict, bar_row: int, bar_col: int) -> tuple[int, int]:
    if prim["kind"] == "square":
        assert prim["r1"] < bar_row or prim["r0"] > bar_row
        return (1, 0) if prim["r1"] < bar_row else (-1, 0)
    assert prim["c1"] < bar_col or prim["c0"] > bar_col
    return (0, 1) if prim["c1"] < bar_col else (0, -1)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _has_ambiguous_distance_tie(primitives: list[dict], bar_row: int, bar_col: int) -> bool:
    for i in range(len(primitives)):
        p = primitives[i]
        dp = _primitive_distance2(p, bar_row, bar_col)
        for j in range(i + 1, len(primitives)):
            q = primitives[j]
            if dp != _primitive_distance2(q, bar_row, bar_col):
                continue
            if p["kind"] == q["kind"] == "square":
                same_side = (p["r1"] < bar_row) == (q["r1"] < bar_row)
                if same_side and _intervals_overlap(p["c0"], p["c1"], q["c0"], q["c1"]):
                    return True
            elif p["kind"] == q["kind"] == "rect":
                same_side = (p["c1"] < bar_col) == (q["c1"] < bar_col)
                if same_side and _intervals_overlap(p["r0"], p["r1"], q["r0"], q["r1"]):
                    return True
    return False


def _solve_grid(grid: np.ndarray, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    analysis = _analyze_input(g)
    bar_row = int(analysis["bar_row"])
    bar_col = int(analysis["bar_col"])
    primitives = [dict(p) for p in analysis["primitives"]]

    for prim in primitives:
        prim["distance2"] = _primitive_distance2(prim, bar_row, bar_col)
        prim["color"] = _primitive_color(prim)
    primitives.sort(key=lambda p: (p["distance2"], p["r0"], p["c0"], p["h"], p["w"], p["kind"]))

    out = np.zeros_like(g)
    out[:, bar_col] = RED
    out[bar_row, :] = CYAN
    bar_mask = out != BLACK
    occupied = bar_mask.copy()
    trace = []

    for prim in primitives:
        r = int(prim["r0"])
        c = int(prim["c0"])
        ph = int(prim["h"])
        pw = int(prim["w"])
        dr, dc = _primitive_step(prim, bar_row, bar_col)
        steps = 0
        stop_reason = "edge"
        blocking_colors: set[int] = set()

        while True:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nc < 0 or nr + ph > h or nc + pw > w:
                break

            occ_patch = occupied[nr : nr + ph, nc : nc + pw]
            if bool(occ_patch.any()):
                bar_patch = bar_mask[nr : nr + ph, nc : nc + pw]
                piece_patch = occ_patch & ~bar_patch
                if bool(piece_patch.any()):
                    stop_reason = "piece"
                    blocking_colors = set(map(int, np.unique(out[nr : nr + ph, nc : nc + pw][piece_patch])))
                    blocking_colors.discard(BLACK)
                else:
                    stop_reason = "bar"
                break

            r, c = nr, nc
            steps += 1

        out[r : r + ph, c : c + pw] = int(prim["color"])
        occupied[r : r + ph, c : c + pw] = True
        trace.append(
            {
                "kind": prim["kind"],
                "color": int(prim["color"]),
                "distance2": int(prim["distance2"]),
                "start": (int(prim["r0"]), int(prim["c0"])),
                "final": (int(r), int(c)),
                "shape": (ph, pw),
                "steps": int(steps),
                "stop_reason": stop_reason,
                "blocking_colors": blocking_colors,
            }
        )

    if return_trace:
        return out, analysis, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose each grey composite into squares and rectangles, recolor them, then slide them onto the matching full-span bar.

    Input structure:
    - A black background contains one cyan full-width horizontal bar and one red full-height vertical bar; their crossing cell is cyan.
    - Elsewhere there are 2-3 grey 4-connected composite shapes made from touching 2x2 squares and 1x3 or 3x1 rectangles.
    - A composite may also contain a few leftover grey cells that do not belong to any allowed primitive.

    Rule:
    - Inside each grey component, find a non-overlapping decomposition into 2x2 squares and 1x3 / 3x1 rectangles that covers as many grey cells as possible.
    - Recolor each square cyan and each rectangle red, and erase any uncovered leftover grey cells to black.
    - Cyan pieces slide vertically toward the cyan horizontal bar, while red pieces slide horizontally toward the red vertical bar.
    - Process pieces from the smallest center-to-target-bar distance to the largest; each piece stops just before it would hit its target bar or a previously stopped piece.
    - Keep both full-span bars unchanged.

    Key insight:
    - The grey shapes must be decomposed first, because primitive shape determines both the recolor and the later slide direction.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


_BASE_COMPONENTS = [
    ("RS", _mask_from_rows(("XX.", "XX.", "XXX"))),
    ("SS", _mask_from_rows(("XXXX", "XXXX"))),
    ("SS", _mask_from_rows(("XX..", "XXXX", "..XX"))),
    ("RRR", _mask_from_rows(("XXX.", "XXX.", ".XXX"))),
    ("RRS", _mask_from_rows(("XXXX", "XXXX", "XX.."))),
    ("RSS", _mask_from_rows(("XXXX", "XXXX", "XXX."))),
    ("SSS", _mask_from_rows(("XXXX", "XXXX", "XX..", "XX.."))),
]


def _sample_component_specs(mode: int, n_components: int) -> list[tuple[str, np.ndarray]]:
    mixed = [item for item in _BASE_COMPONENTS if "S" in item[0] and "R" in item[0]]
    simple = [item for item in _BASE_COMPONENTS if len(item[0]) == 2]

    while True:
        picks: list[tuple[str, np.ndarray]] = []
        if mode == 2:
            picks.append(random.choice(mixed))
        while len(picks) < n_components:
            pool = _BASE_COMPONENTS
            if mode == 1 and len(picks) == 0:
                pool = simple + mixed
            picks.append(random.choice(pool))

        total = sum(len(sig) for sig, _ in picks)
        sigs = "".join(sig for sig, _ in picks)
        if total > 7:
            continue
        if "S" not in sigs or "R" not in sigs:
            continue
        return picks


def _choose_layout(mode: int, n_components: int) -> tuple[int, int, int, int, list[str]]:
    same_quadrant = mode in (1, 2)
    focus = random.choice(("NW", "NE", "SW", "SE"))
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    min_small = 4
    min_focus = 7 if same_quadrant else 4

    if focus in ("NW", "NE"):
        bar_row = random.randint(min_focus, h - 1 - min_small)
    else:
        bar_row = random.randint(min_small, h - 1 - min_focus)
    if focus in ("NW", "SW"):
        bar_col = random.randint(min_focus, w - 1 - min_small)
    else:
        bar_col = random.randint(min_small, w - 1 - min_focus)

    if same_quadrant:
        quadrants = [focus, focus]
        others = [q for q in ("NW", "NE", "SW", "SE") if q != focus]
        while len(quadrants) < n_components:
            quadrants.append(random.choice(others))
    else:
        quadrants = random.sample(("NW", "NE", "SW", "SE"), k=n_components)

    return h, w, bar_row, bar_col, quadrants


def _quadrant_candidates(
    quadrant: str, shape: tuple[int, int], h: int, w: int, bar_row: int, bar_col: int
) -> list[tuple[int, int]]:
    ph, pw = shape
    if quadrant == "NW":
        rmin, rmax = 0, bar_row - ph
        cmin, cmax = 0, bar_col - pw
    elif quadrant == "NE":
        rmin, rmax = 0, bar_row - ph
        cmin, cmax = bar_col + 1, w - pw
    elif quadrant == "SW":
        rmin, rmax = bar_row + 1, h - ph
        cmin, cmax = 0, bar_col - pw
    else:
        rmin, rmax = bar_row + 1, h - ph
        cmin, cmax = bar_col + 1, w - pw

    if rmin > rmax or cmin > cmax:
        return []
    return [(r, c) for r in range(rmin, rmax + 1) for c in range(cmin, cmax + 1)]


def _can_place(mask: np.ndarray, top: int, left: int, grey_mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    h, w = grey_mask.shape
    for r, c in coords:
        rr = top + int(r)
        cc = left + int(c)
        for dr, dc in DIR8 + ((0, 0),):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and grey_mask[nr, nc]:
                return False
    return True


def _stamp(mask: np.ndarray, top: int, left: int, grey_mask: np.ndarray) -> None:
    coords = np.argwhere(mask)
    grey_mask[top + coords[:, 0], left + coords[:, 1]] = True


def _maybe_add_residue(mask: np.ndarray) -> np.ndarray | None:
    coords = set(map(tuple, np.argwhere(mask)))
    base = np.asarray(mask, dtype=bool)
    candidates = set()
    for r, c in coords:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in coords:
                candidates.add((nr, nc))

    residue_masks = []
    for nr, nc in candidates:
        pts = list(coords) + [(nr, nc)]
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        r0, c0 = min(rs), min(cs)
        shifted = [(r - r0, c - c0) for r, c in pts]
        hh = max(r for r, _ in shifted) + 1
        ww = max(c for _, c in shifted) + 1
        if hh > 4 or ww > 4:
            continue
        trial = np.zeros((hh, ww), dtype=bool)
        for r, c in shifted:
            trial[r, c] = True
        solved = _solve_component(trial, 0, 0)
        if solved["unique"] and len(solved["residual"]) == 1:
            residue_masks.append(trial)

    if not residue_masks:
        return None
    return random.choice(residue_masks)


def _render_input(
    h: int, w: int, bar_row: int, bar_col: int, grey_mask: np.ndarray
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[:, bar_col] = RED
    grid[bar_row, :] = CYAN
    grid[grey_mask] = GRAY
    return grid


def _validate_generated_case(grid: np.ndarray, mode: int) -> bool:
    out, analysis, trace = _solve_grid(grid, return_trace=True)
    bar_row = int(analysis["bar_row"])
    bar_col = int(analysis["bar_col"])
    components = analysis["components"]
    primitives = analysis["primitives"]

    if len(components) not in (2, 3):
        return False
    if any(not info["unique"] for info in components):
        return False
    if any(not (7 <= info["area"] <= 13) for info in components):
        return False
    if any(len(info["residual"]) > 1 for info in components):
        return False
    if not primitives:
        return False
    if not any(p["kind"] == "square" for p in primitives):
        return False
    if not any(p["kind"] == "rect" for p in primitives):
        return False

    if _has_ambiguous_distance_tie(primitives, bar_row, bar_col):
        return False

    if not any(event["color"] == CYAN and event["steps"] >= 1 for event in trace):
        return False
    if not any(event["color"] == RED and event["steps"] >= 1 for event in trace):
        return False
    if max(event["steps"] for event in trace) < 2:
        return False

    collision_count = sum(event["stop_reason"] == "piece" for event in trace)
    residual_count = sum(len(info["residual"]) for info in components)

    if mode == 1 and collision_count == 0:
        return False
    if mode == 2 and (collision_count == 0 or residual_count == 0):
        return False

    if not bool(np.all(out[bar_row, :] == CYAN)):
        return False
    col = out[:, bar_col].copy()
    col[bar_row] = RED
    if not bool(np.all(col == RED)):
        return False
    if GRAY in np.unique(out):
        return False
    return True


def _generate_attempt(mode: int) -> np.ndarray | None:
    n_components = 2 if mode in (1, 2) else random.choice((2, 3))
    component_specs = _sample_component_specs(mode, n_components)
    h, w, bar_row, bar_col, quadrants = _choose_layout(mode, n_components)

    transformed_masks = []
    want_residue = mode == 2
    residue_done = False
    residue_index = random.randrange(n_components) if want_residue else -1

    for idx, (_sig, base_mask) in enumerate(component_specs):
        mask = random.choice(_all_transforms(base_mask))
        if idx == residue_index:
            augmented = _maybe_add_residue(mask)
            if augmented is not None:
                mask = augmented
                residue_done = True
        transformed_masks.append(mask)

    if want_residue and not residue_done:
        return None

    grey_mask = np.zeros((h, w), dtype=bool)
    placed_positions = {}

    for idx, (mask, quadrant) in enumerate(zip(transformed_masks, quadrants)):
        candidates = _quadrant_candidates(quadrant, mask.shape, h, w, bar_row, bar_col)
        if not candidates:
            return None

        if quadrant in placed_positions:
            anchor_r, anchor_c = placed_positions[quadrant][0]
            if mode == 2:
                candidates.sort(
                    key=lambda rc: (
                        abs(rc[0] - anchor_r) + abs(rc[1] - anchor_c),
                        abs((bar_row - rc[0]) if quadrant in ("NW", "NE") else (rc[0] - bar_row)),
                        abs((bar_col - rc[1]) if quadrant in ("NW", "SW") else (rc[1] - bar_col)),
                    )
                )
            else:
                candidates.sort(key=lambda rc: (abs(rc[0] - anchor_r) + abs(rc[1] - anchor_c), random.random()))
        else:
            if mode == 2 and quadrant == quadrants[0]:
                candidates.sort(
                    key=lambda rc: (
                        abs((bar_row - (rc[0] + mask.shape[0])) if quadrant in ("NW", "NE") else (rc[0] - bar_row)),
                        abs((bar_col - (rc[1] + mask.shape[1])) if quadrant in ("NW", "SW") else (rc[1] - bar_col)),
                        random.random(),
                    )
                )
            else:
                random.shuffle(candidates)

        placed = False
        for top, left in candidates[:40]:
            if _can_place(mask, top, left, grey_mask):
                _stamp(mask, top, left, grey_mask)
                placed_positions.setdefault(quadrant, []).append((top, left))
                placed = True
                break
        if not placed:
            return None

    grid = _render_input(h, w, bar_row, bar_col, grey_mask)
    if not _validate_generated_case(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    mode = _GEN_COUNTER % 4
    _GEN_COUNTER += 1

    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
