from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
VEC_TO_SIDE = {v: k for k, v in SIDE_TO_VEC.items()}
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}

GRID_ROWS_RANGE = (15, 19)
GRID_COLS_RANGE = (17, 21)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4_coords(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = int(r) + dr
        cc = int(c) + dc
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solid_rectangle_info(mask: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one red component"
    box = _bbox(comps[0])
    r0, c0, r1, c1 = box
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])), "red object must be a solid rectangle"
    return box


def _edge_distance(blue_coords: np.ndarray, edge_cells: list[tuple[int, int]]) -> int:
    best = 10**9
    for er, ec in edge_cells:
        d = np.abs(blue_coords[:, 0] - int(er)) + np.abs(blue_coords[:, 1] - int(ec))
        best = min(best, int(d.min()))
    return best


def _red_anchor_side(blue_mask: np.ndarray, red_mask: np.ndarray) -> str:
    blue_coords = np.argwhere(blue_mask)
    r0, c0, r1, c1 = _solid_rectangle_info(red_mask)
    edges = {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }
    scores = {side: _edge_distance(blue_coords, cells) for side, cells in edges.items()}
    best = min(scores.values())
    sides = [side for side, score in scores.items() if score == best]
    assert len(sides) == 1, f"red anchor edge must be unique, got {scores}"
    return sides[0]


def _project_opposite_edge(mask: np.ndarray, side: str, r: int, c: int) -> tuple[int, int]:
    if side in {"top", "bottom"}:
        rows = np.flatnonzero(mask[:, int(c)])
        assert rows.size > 0
        return (int(rows.max()), int(c)) if side == "top" else (int(rows.min()), int(c))
    cols = np.flatnonzero(mask[int(r), :])
    assert cols.size > 0
    return (int(r), int(cols.max())) if side == "left" else (int(r), int(cols.min()))


def _project_outside_opposite_edge(mask: np.ndarray, side: str, r: int, c: int) -> tuple[int, int] | None:
    if side in {"top", "bottom"}:
        rows = np.flatnonzero(mask[:, int(c)])
        assert rows.size > 0
        rr = int(rows.max()) + 1 if side == "top" else int(rows.min()) - 1
        return (rr, int(c)) if _in_bounds(mask.shape, rr, int(c)) else None
    cols = np.flatnonzero(mask[int(r), :])
    assert cols.size > 0
    cc = int(cols.max()) + 1 if side == "left" else int(cols.min()) - 1
    return (int(r), cc) if _in_bounds(mask.shape, int(r), cc) else None


def _tab_sources(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    tabs: list[dict] = []
    for r, c in np.argwhere(mask):
        neighbors = _neighbors4_coords(mask, int(r), int(c))
        if len(neighbors) != 1:
            continue
        nr, nc = neighbors[0]
        outward = (int(r) - int(nr), int(c) - int(nc))
        tabs.append(
            {
                "cell": (int(r), int(c)),
                "vec": outward,
                "side": VEC_TO_SIDE[outward],
            }
        )
    return tabs


def _notch_sources(mask: np.ndarray, tab_cells: set[tuple[int, int]]) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    notches: list[dict] = []
    for r in range(H):
        for c in range(W):
            if bool(mask[r, c]):
                continue
            fg_dirs: list[tuple[int, int]] = []
            bg_dirs: list[tuple[int, int]] = []
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
                    fg_dirs.append((dr, dc))
                else:
                    bg_dirs.append((dr, dc))
            if len(fg_dirs) != 3 or len(bg_dirs) != 1:
                continue
            outward = bg_dirs[0]
            dr, dc = outward
            if dr != 0:
                side_a = (r, c - 1)
                side_b = (r, c + 1)
            else:
                side_a = (r - 1, c)
                side_b = (r + 1, c)
            if side_a in tab_cells or side_b in tab_cells:
                continue
            notches.append(
                {
                    "cell": (int(r), int(c)),
                    "vec": outward,
                    "side": VEC_TO_SIDE[outward],
                }
            )
    return notches


def _analyze_blue(mask: np.ndarray) -> dict:
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one blue component"
    tabs = _tab_sources(mask)
    assert tabs, "expected at least one blue tab"
    tab_sides = {tab["side"] for tab in tabs}
    assert len(tab_sides) == 1, f"tabs must all lie on one side, got {tab_sides}"
    tab_side = next(iter(tab_sides))
    tab_cells = {tab["cell"] for tab in tabs}
    notches = _notch_sources(mask, tab_cells)
    if notches:
        notch_sides = {notch["side"] for notch in notches}
        assert notch_sides == {OPPOSITE_SIDE[tab_side]}, f"notches must be opposite tabs, got {notch_sides}"
    return {
        "coords": comps[0],
        "bbox": _bbox(comps[0]),
        "tabs": tabs,
        "notches": notches,
        "side": tab_side,
        "t": len(tabs),
    }


def _complete_blue(mask: np.ndarray) -> np.ndarray:
    info = _analyze_blue(mask)
    out = mask.copy()
    for tab in info["tabs"]:
        r, c = tab["cell"]
        target = _project_opposite_edge(mask, tab["side"], r, c)
        out[target] = False
    for notch in info["notches"]:
        r, c = notch["cell"]
        target = _project_outside_opposite_edge(mask, notch["side"], r, c)
        if target is not None:
            out[target] = True
    return out


def _shear_red(mask: np.ndarray, direction_side: str, step: int, anchor_side: str) -> np.ndarray:
    r0, c0, r1, c1 = _solid_rectangle_info(mask)
    dr, dc = SIDE_TO_VEC[direction_side]
    out = np.zeros_like(mask, dtype=bool)
    if anchor_side in {"top", "bottom"}:
        for r in range(r0, r1 + 1):
            d = r - r0 if anchor_side == "top" else r1 - r
            shift = d * int(step)
            for c in range(c0, c1 + 1):
                rr = r + dr * shift
                cc = c + dc * shift
                if _in_bounds(mask.shape, rr, cc):
                    out[rr, cc] = True
    else:
        for c in range(c0, c1 + 1):
            d = c - c0 if anchor_side == "left" else c1 - c
            shift = d * int(step)
            for r in range(r0, r1 + 1):
                rr = r + dr * shift
                cc = c + dc * shift
                if _in_bounds(mask.shape, rr, cc):
                    out[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the blue shape's one-sided tabs to encode the red rectangle's shear, then erase that asymmetry by completing the blue shape.

    Input structure:
    - The background is black, with one blue thick rectilinear shape and one separate red solid rectangle.
    - The blue shape has 1-3 one-pixel tabs on a single side and fewer one-pixel notches on the opposite side.
    - The red object is a filled rectangle, and its nearest edge to the blue shape acts as the shear anchor.

    Rule:
    - Find the blue tabs: each is a blue boundary pixel with exactly one 4-neighbor. Their common side gives the shear direction, and their count t gives the per-step offset.
    - Shear the red rectangle by moving each row or column that is distance d from its anchor edge by d*t cells along the tab side direction.
    - Complete the blue shape symmetrically: every tab removes the opposite edge pixel in the same row or column, and every original notch adds a tab just outside the opposite edge.
    - Render the sheared red rectangle and the completed blue shape on the same canvas, with blue overriding red on overlaps.

    Key insight:
    - The blue object's temporary boundary asymmetry stores two parameters at once: which way the red rectangle leans and how large each successive offset is.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    blue_mask = g == BLUE
    red_mask = g == RED
    assert len(_components(blue_mask)) == 1, "expected one blue object"
    assert len(_components(red_mask)) == 1, "expected one red object"

    blue_info = _analyze_blue(blue_mask)
    anchor_side = _red_anchor_side(blue_mask, red_mask)
    sheared_red = _shear_red(red_mask, blue_info["side"], blue_info["t"], anchor_side)
    completed_blue = _complete_blue(blue_mask)

    out = np.full_like(g, BG)
    out[sheared_red] = RED
    out[completed_blue] = BLUE
    return out


def _make_rect_body() -> np.ndarray:
    h = random.randint(5, 8)
    w = random.randint(5, 9)
    return np.ones((h, w), dtype=bool)


def _make_l_body() -> np.ndarray:
    h = random.randint(7, 10)
    w = random.randint(7, 10)
    arm_h = random.randint(3, min(4, h - 2))
    arm_w = random.randint(3, min(4, w - 2))
    missing = random.choice(["TL", "TR", "BL", "BR"])
    mask = np.zeros((h, w), dtype=bool)
    if missing == "TR":
        mask[:, :arm_w] = True
        mask[h - arm_h :, :] = True
    elif missing == "TL":
        mask[:, w - arm_w :] = True
        mask[h - arm_h :, :] = True
    elif missing == "BR":
        mask[:, :arm_w] = True
        mask[:arm_h, :] = True
    else:
        mask[:, w - arm_w :] = True
        mask[:arm_h, :] = True
    return mask


def _side_candidates(mask: np.ndarray, side: str) -> tuple[list[int], list[int]]:
    H, W = mask.shape
    tabs: list[int] = []
    notches: list[int] = []

    def tab_ok(rr: int, cc: int) -> bool:
        if not _in_bounds(mask.shape, rr, cc) or bool(mask[rr, cc]):
            return False
        count = 0
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]):
                count += 1
        return count == 1

    def notch_ok(rr: int, cc: int, outward: tuple[int, int]) -> bool:
        if not bool(mask[rr, cc]):
            return False
        tmp = mask.copy()
        tmp[rr, cc] = False
        fg_dirs: list[tuple[int, int]] = []
        bg_dirs: list[tuple[int, int]] = []
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if _in_bounds(tmp.shape, nr, nc) and bool(tmp[nr, nc]):
                fg_dirs.append((dr, dc))
            else:
                bg_dirs.append((dr, dc))
        return len(fg_dirs) == 3 and bg_dirs == [outward]

    if side in {"top", "bottom"}:
        for c in range(1, W - 1):
            rows = np.flatnonzero(mask[:, c])
            if rows.size == 0:
                continue
            top = int(rows.min())
            bottom = int(rows.max())
            if side == "top":
                if top > 0 and tab_ok(top - 1, c):
                    tabs.append(c)
                if top > 0 and notch_ok(bottom, c, (1, 0)):
                    notches.append(c)
            else:
                if bottom < H - 1 and tab_ok(bottom + 1, c):
                    tabs.append(c)
                if bottom < H - 1 and notch_ok(top, c, (-1, 0)):
                    notches.append(c)
    else:
        for r in range(1, H - 1):
            cols = np.flatnonzero(mask[r, :])
            if cols.size == 0:
                continue
            left = int(cols.min())
            right = int(cols.max())
            if side == "left":
                if left > 0 and tab_ok(r, left - 1):
                    tabs.append(r)
                if left > 0 and notch_ok(r, right, (0, 1)):
                    notches.append(r)
            else:
                if right < W - 1 and tab_ok(r, right + 1):
                    tabs.append(r)
                if right < W - 1 and notch_ok(r, left, (0, -1)):
                    notches.append(r)
    return tabs, notches


def _spaced_subsets(values: list[int], size: int) -> list[tuple[int, ...]]:
    values = sorted(set(int(v) for v in values))
    out: list[tuple[int, ...]] = []

    def rec(start: int, chosen: list[int]) -> None:
        if len(chosen) == size:
            out.append(tuple(chosen))
            return
        for i in range(start, len(values)):
            v = values[i]
            if chosen and abs(v - chosen[-1]) < 2:
                continue
            chosen.append(v)
            rec(i + 1, chosen)
            chosen.pop()

    rec(0, [])
    return out


def _apply_features(mask: np.ndarray, side: str, tab_slots: tuple[int, ...], notch_slots: tuple[int, ...]) -> np.ndarray:
    out = mask.copy()
    if side in {"top", "bottom"}:
        for c in tab_slots:
            rows = np.flatnonzero(out[:, int(c)])
            top = int(rows.min())
            bottom = int(rows.max())
            rr = top - 1 if side == "top" else bottom + 1
            out[rr, int(c)] = True
        opp = OPPOSITE_SIDE[side]
        for c in notch_slots:
            rows = np.flatnonzero(out[:, int(c)])
            top = int(rows.min())
            bottom = int(rows.max())
            rr = bottom if opp == "bottom" else top
            out[rr, int(c)] = False
    else:
        for r in tab_slots:
            cols = np.flatnonzero(out[int(r), :])
            left = int(cols.min())
            right = int(cols.max())
            cc = left - 1 if side == "left" else right + 1
            out[int(r), cc] = True
        opp = OPPOSITE_SIDE[side]
        for r in notch_slots:
            cols = np.flatnonzero(out[int(r), :])
            left = int(cols.min())
            right = int(cols.max())
            cc = right if opp == "right" else left
            out[int(r), cc] = False
    return out


def _same_side_gap(blue_box: tuple[int, int, int, int], red_box: tuple[int, int, int, int], side: str) -> int:
    br0, bc0, br1, bc1 = blue_box
    rr0, rc0, rr1, rc1 = red_box
    if side == "right":
        return rc0 - bc1 - 1
    if side == "left":
        return bc0 - rc1 - 1
    if side == "bottom":
        return rr0 - br1 - 1
    return br0 - rr1 - 1


def _fits_same_side(blue_box: tuple[int, int, int, int], red_box: tuple[int, int, int, int], side: str) -> bool:
    return _same_side_gap(blue_box, red_box, side) >= 2


def _valid_red_placements(
    H: int,
    W: int,
    blue_mask: np.ndarray,
    side: str,
    rect_h: int,
    rect_w: int,
    step: int,
) -> list[tuple[int, int, str]]:
    blue_box = _bbox(np.argwhere(blue_mask))
    placements: list[tuple[int, int, str]] = []
    for r0 in range(H - rect_h + 1):
        for c0 in range(W - rect_w + 1):
            red_mask = np.zeros((H, W), dtype=bool)
            red_mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
            if bool((red_mask & blue_mask).any()):
                continue
            red_box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _fits_same_side(blue_box, red_box, side):
                continue
            try:
                anchor = _red_anchor_side(blue_mask, red_mask)
            except AssertionError:
                continue
            sheared = _shear_red(red_mask, side, step, anchor)
            if int(sheared.sum()) != rect_h * rect_w:
                continue
            placements.append((r0, c0, anchor))
    return placements


def _build_blue_case(H: int, W: int) -> tuple[np.ndarray, dict] | None:
    body = _make_rect_body() if random.random() < 0.5 else _make_l_body()
    h, w = body.shape
    if h >= H - 2 or w >= W - 2:
        return None

    side_order = random.sample(["top", "bottom", "left", "right"], 4)
    for side in side_order:
        tab_candidates, notch_candidates = _side_candidates(body, side)
        possible_t = [t for t in (1, 2, 3) if _spaced_subsets(tab_candidates, t)]
        random.shuffle(possible_t)
        for t in possible_t:
            tab_choices = _spaced_subsets(tab_candidates, t)
            random.shuffle(tab_choices)
            for tab_slots in tab_choices[: min(6, len(tab_choices))]:
                notch_pool = [slot for slot in notch_candidates if slot not in tab_slots]
                possible_n = [n for n in range(min(2, t - 1), -1, -1) if n == 0 or _spaced_subsets(notch_pool, n)]
                random.shuffle(possible_n)
                for notch_count in possible_n:
                    notch_choices = [()] if notch_count == 0 else _spaced_subsets(notch_pool, notch_count)
                    random.shuffle(notch_choices)
                    for notch_slots in notch_choices[: min(4, len(notch_choices))]:
                        local = _apply_features(body, side, tab_slots, notch_slots)
                        try:
                            info = _analyze_blue(local)
                        except AssertionError:
                            continue
                        if info["t"] != t or info["side"] != side or len(info["notches"]) != notch_count:
                            continue
                        reserve_h = 8
                        reserve_v = 10
                        r_low, r_high = 1, H - h - 1
                        c_low, c_high = 1, W - w - 1
                        if side == "right":
                            c_high = min(c_high, W - w - 1 - reserve_h)
                        elif side == "left":
                            c_low = max(c_low, reserve_h + 1)
                        elif side == "bottom":
                            r_high = min(r_high, H - h - 1 - reserve_v)
                        else:
                            r_low = max(r_low, reserve_v + 1)
                        if r_low > r_high or c_low > c_high:
                            continue
                        for _ in range(8):
                            r0 = random.randint(r_low, r_high)
                            c0 = random.randint(c_low, c_high)
                            full = np.zeros((H, W), dtype=bool)
                            full[r0 : r0 + h, c0 : c0 + w] = local
                            return full, {"side": side, "t": t, "notch_count": notch_count}
    return None


def _semantic_validate(inp: np.ndarray, meta: dict) -> None:
    blue_mask = inp == BLUE
    red_mask = inp == RED
    blue = _analyze_blue(blue_mask)
    assert blue["side"] == meta["side"]
    assert blue["t"] == meta["t"]
    assert len(blue["notches"]) == meta["notch_count"]
    assert 1 <= blue["t"] <= 3
    assert len(blue["notches"]) < blue["t"]

    red_box = _solid_rectangle_info(red_mask)
    assert _same_side_gap(blue["bbox"], red_box, blue["side"]) >= 2
    _red_anchor_side(blue_mask, red_mask)

    out = main(inp)
    assert out.shape == inp.shape
    assert np.any((inp == BLUE) != (out == BLUE)), "blue completion must change the blue shape"
    assert np.any((inp == RED) != (out == RED)), "red shear must change the red shape"
    assert len(_components(out == BLUE)) == 1, "completed blue shape must stay connected"
    assert int(np.sum(out == RED)) == int(np.sum(inp == RED)), "shear should preserve red area"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        H = random.randint(*GRID_ROWS_RANGE)
        W = random.randint(*GRID_COLS_RANGE)
        blue_case = _build_blue_case(H, W)
        if blue_case is None:
            continue
        blue_mask, meta = blue_case

        size_options: list[tuple[int, int, list[tuple[int, int, str]]]] = []
        for rect_h in range(3, 6):
            for rect_w in range(2, 5):
                placements = _valid_red_placements(H, W, blue_mask, meta["side"], rect_h, rect_w, meta["t"])
                if placements:
                    size_options.append((rect_h, rect_w, placements))
        if not size_options:
            continue
        rect_h, rect_w, placements = random.choice(size_options)
        r0, c0, anchor = random.choice(placements)
        red_mask = np.zeros((H, W), dtype=bool)
        red_mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

        grid = np.zeros((H, W), dtype=int)
        grid[blue_mask] = BLUE
        grid[red_mask] = RED
        meta["anchor"] = anchor
        _semantic_validate(grid, meta)
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
