from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
FG = 5

BOUNDARY_BY_NEIGHBORS = {
    0: 2,  # red
    1: 3,  # green
    2: 1,  # blue
}
INTERIOR_BY_HOLES = {
    0: 9,  # maroon
    1: 7,  # orange
}

GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _hole_count(mask: np.ndarray, coords: list[tuple[int, int]]) -> int:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    holes = 0
    for hole in _components4(~sub):
        if any(
            rr == 0 or rr == sub.shape[0] - 1 or cc == 0 or cc == sub.shape[1] - 1
            for rr, cc in hole
        ):
            continue
        holes += 1
    return holes


def _analyze(grid: np.ndarray):
    fg_mask = grid == FG
    comps = _components4(fg_mask)
    h, w = grid.shape
    comp_id = -np.ones((h, w), dtype=int)
    for idx, comp in enumerate(comps):
        for r, c in comp:
            comp_id[r, c] = idx

    info = []
    for idx, comp in enumerate(comps):
        neighbor_ids: set[int] = set()
        boundary: list[tuple[int, int]] = []
        interior: list[tuple[int, int]] = []
        for r, c in comp:
            is_boundary = False
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    is_boundary = True
                    continue
                if grid[nr, nc] == BG:
                    is_boundary = True
                rr, cc = r + 2 * (nr - r), c + 2 * (nc - c)
                if (
                    0 <= rr < h
                    and 0 <= cc < w
                    and grid[nr, nc] == BG
                    and comp_id[rr, cc] >= 0
                    and comp_id[rr, cc] != idx
                ):
                    neighbor_ids.add(int(comp_id[rr, cc]))
            if is_boundary:
                boundary.append((r, c))
            else:
                interior.append((r, c))

        holes = _hole_count(fg_mask, comp)
        info.append(
            {
                "coords": comp,
                "neighbors": len(neighbor_ids),
                "holes": holes,
                "boundary": boundary,
                "interior": interior,
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object by separating boundary pixels from interior pixels.

    Boundary color depends on the number of orthogonally neighboring objects
    separated by exactly one background cell. Interior color depends on the
    number of enclosed holes inside the object.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)

    for comp in _analyze(g):
        m = int(comp["neighbors"])
        h = int(comp["holes"])
        boundary_color = BOUNDARY_BY_NEIGHBORS.get(m, 4)
        interior_color = INTERIOR_BY_HOLES.get(h, 6)
        for r, c in comp["boundary"]:
            out[r, c] = boundary_color
        for r, c in comp["interior"]:
            out[r, c] = interior_color
    return out


def _solid_rect() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    return np.ones((h, w), dtype=bool)


def _holey_rect(n_holes: int) -> np.ndarray:
    if n_holes == 1:
        h = random.randint(7, 9)
        w = random.randint(7, 9)
    else:
        h = random.randint(9, 11)
        w = random.randint(10, 13)
    mask = np.ones((h, w), dtype=bool)
    used: list[tuple[int, int, int, int]] = []
    attempts = 0
    target = n_holes if n_holes <= 2 else random.randint(2, 3)
    while len(used) < target and attempts < 200:
        attempts += 1
        hh = random.randint(1, 2)
        hw = random.randint(1, 2)
        r0 = random.randint(2, h - hh - 3)
        c0 = random.randint(2, w - hw - 3)
        r1 = r0 + hh - 1
        c1 = c0 + hw - 1
        ok = True
        for ur0, uc0, ur1, uc1 in used:
            if not (r1 + 2 < ur0 or ur1 + 2 < r0 or c1 + 2 < uc0 or uc1 + 2 < c0):
                ok = False
                break
        if not ok:
            continue
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
        used.append((r0, c0, r1, c1))
    if len(used) < target:
        return _holey_rect(n_holes)
    return mask


def _make_object(hole_class: int) -> np.ndarray:
    if hole_class == 0:
        return _solid_rect()
    if hole_class == 1:
        return _holey_rect(1)
    return _holey_rect(2)


def _paste(grid: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = FG


def _add_star_positions(sizes: dict[str, tuple[int, int]], top: int, left: int):
    ah, aw = sizes["A"]
    bh, bw = sizes["B"]
    ch, cw = sizes["C"]
    dh, dw = sizes["D"]

    a_top = top + bh + 1
    a_left = left + cw + 1

    b_left_min = a_left - bw + 1
    b_left_max = a_left + aw - 1
    c_top_min = a_top - ch + 1
    c_top_max = a_top + ah - 1
    d_top_min = a_top - dh + 1
    d_top_max = a_top + ah - 1

    return {
        "A": (a_top, a_left),
        "B": (top, random.randint(b_left_min, b_left_max)),
        "C": (random.randint(c_top_min, c_top_max), left),
        "D": (random.randint(d_top_min, d_top_max), a_left + aw + 1),
    }


def _add_chain_positions(sizes: dict[str, tuple[int, int]], top: int, left: int):
    fh, fw = sizes["F"]
    eh, ew = sizes["E"]
    gh, gw = sizes["G"]
    e_left = left + fw + 1
    g_left = e_left + ew + 1
    e_top = top
    f_top = random.randint(e_top - fh + 1, e_top + eh - 1)
    g_top = random.randint(e_top - gh + 1, e_top + eh - 1)
    return {
        "E": (e_top, e_left),
        "F": (f_top, left),
        "G": (g_top, g_left),
    }


def _build_grid(masks: dict[str, np.ndarray]) -> np.ndarray:
    star_sizes = {k: masks[k].shape for k in ("A", "B", "C", "D")}
    star_pos = _add_star_positions(star_sizes, 0, 0)
    star_min_r = min(top for top, _ in star_pos.values())
    star_min_c = min(left for _, left in star_pos.values())
    star_pos = {k: (top - star_min_r, left - star_min_c) for k, (top, left) in star_pos.items()}

    star_bottom = max(star_pos[k][0] + masks[k].shape[0] for k in star_pos)
    star_right = max(star_pos[k][1] + masks[k].shape[1] for k in star_pos)

    chain_sizes = {k: masks[k].shape for k in ("E", "F", "G")}
    chain_pos = _add_chain_positions(chain_sizes, 0, 0)
    chain_min_r = min(top for top, _ in chain_pos.values())
    chain_min_c = min(left for _, left in chain_pos.values())
    chain_pos = {k: (top - chain_min_r, left - chain_min_c) for k, (top, left) in chain_pos.items()}

    gap_row = random.randint(4, 7)
    chain_pos = {k: (top + star_bottom + gap_row, left) for k, (top, left) in chain_pos.items()}

    gap_col = random.randint(5, 8)
    iso_h, iso_w = masks["H"].shape
    chain_bottom = max(chain_pos[k][0] + masks[k].shape[0] for k in chain_pos)
    iso_top = chain_bottom - iso_h + random.randint(0, max(0, iso_h // 2))
    iso_left = star_right + gap_col

    positions = {**star_pos, **chain_pos, "H": (iso_top, iso_left)}
    min_r = min(top for top, _ in positions.values())
    min_c = min(left for _, left in positions.values())
    pad_top = random.randint(1, 3) - min_r
    pad_left = random.randint(1, 3) - min_c
    positions = {k: (top + pad_top, left + pad_left) for k, (top, left) in positions.items()}

    max_r = max(top + masks[k].shape[0] for k, (top, _) in positions.items())
    max_c = max(left + masks[k].shape[1] for k, (_, left) in positions.items())
    grid = np.full(
        (max_r + random.randint(1, 4), max_c + random.randint(1, 5)),
        BG,
        dtype=int,
    )

    for name, mask in masks.items():
        _paste(grid, mask, *positions[name])
    return grid


def _coverage_ok(grid: np.ndarray) -> bool:
    info = _analyze(grid)
    if len(info) != 8:
        return False
    neighbor_branches = {0, 1, 2, 3}
    seen_neighbors = {min(3, int(comp["neighbors"])) for comp in info}
    if seen_neighbors != neighbor_branches:
        return False
    hole_branches = {0, 1, 2}
    seen_holes = {0 if comp["holes"] == 0 else 1 if comp["holes"] == 1 else 2 for comp in info}
    if seen_holes != hole_branches:
        return False
    for comp in info:
        if not comp["boundary"] or not comp["interior"]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    classes = [0, 0, 0, 0, 0, 1, 2, random.choice([0, 1, 2])]
    random.shuffle(classes)
    names = list("ABCDEFGH")
    masks = {name: _make_object(cls) for name, cls in zip(names, classes, strict=False)}
    grid = _build_grid(masks)
    if not _coverage_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
