import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _rotate_rel(coords: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_coords(pts)


def _translate(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _neighbor_count(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cells for dr, dc in DIRS)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = [start]
    while q:
        r, c = q.pop()
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


def _dir_to_idx(vec: tuple[int, int]) -> int:
    return DIRS.index(vec)


def _detect_piece_direction(coords: tuple[tuple[int, int], ...]) -> int:
    cells = set(coords)
    candidates: list[int] = []
    for leaf in list(cells):
        if _neighbor_count(leaf, cells) != 1:
            continue
        lr, lc = leaf
        root = next((lr + dr, lc + dc) for dr, dc in DIRS if (lr + dr, lc + dc) in cells)
        nbrs = [(root[0] + dr, root[1] + dc) for dr, dc in DIRS if (root[0] + dr, root[1] + dc) in cells]
        if len(nbrs) != 2:
            continue
        body = nbrs[0] if nbrs[1] == leaf else nbrs[1]
        v_leaf = (leaf[0] - root[0], leaf[1] - root[1])
        v_body = (body[0] - root[0], body[1] - root[1])
        if v_leaf[0] == -v_body[0] and v_leaf[1] == -v_body[1]:
            continue
        remainder = set(cells)
        remainder.remove(leaf)
        remainder.remove(root)
        if body not in remainder or not _is_connected(remainder):
            continue
        candidates.append(_dir_to_idx((root[0] - body[0], root[1] - body[1])))
    uniq = sorted(set(candidates))
    assert len(uniq) == 1 and len(candidates) == 1, "piece must have a unique 2-pixel L protrusion"
    return uniq[0]


def _find_template_color(grid: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(grid[grid != bg], return_counts=True)
    assert vals.size, "grid must contain a non-background template color"
    return int(vals[np.argmax(cnts)])


def _decode_arrow(
    comp: np.ndarray, hole_by_cell: dict[tuple[int, int], int]
) -> tuple[int, int]:
    pts = [tuple(map(int, p)) for p in comp]
    roots: list[tuple[int, int]] = []
    for r, c in pts:
        hits = []
        for d, (dr, dc) in enumerate(DIRS):
            hid = hole_by_cell.get((r + dr, c + dc))
            if hid is not None:
                hits.append((d, hid))
        if len(hits) == 1:
            roots.append((hits[0][0], hits[0][1]))
    uniq = list(dict.fromkeys(roots))
    assert len(uniq) == 1, "arrow must have exactly one pointing arm"
    return uniq[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered colored fragments by arrow-indicated directions, then plug them back into the template holes.

    Input structure:
    - bg is the most common color. One large grey rectangle acts as a template.
    - Inside that rectangle, several background-colored 4-connected holes are carved out.
    - Next to each hole sits a 2-pixel L-shaped colored arrow. One arrow pixel touches the hole boundary and points into it.
    - Outside the template, each non-grey color also appears as one scattered monochrome fragment with a unique 2-pixel L protrusion.

    Rule:
    - Pair each scattered fragment with the arrow of the same color.
    - Read the arrow's target direction from the arrow pixel that touches the hole.
    - Read the fragment's original direction from its unique 2-pixel L protrusion.
    - Rotate the fragment by the required number of 90-degree clockwise turns, then place it into the hole touched by the matching arrow.
    - The output keeps the grey template, fills each hole with its matched fragment color, and erases scattered fragments and arrows to background.

    Key insight:
    Color chooses which hole belongs to which fragment, but the rotation is encoded separately by comparing two direction markers: the template-side L arrow and the fragment-side L protrusion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    grey = _find_template_color(g, bg)

    grey_coords = np.argwhere(g == grey)
    assert grey_coords.size, "template must contain grey pixels"
    tr0, tc0, tr1, tc1 = _bbox_coords(grey_coords)
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    th, tw = template.shape

    holes: dict[int, dict] = {}
    hole_by_cell: dict[tuple[int, int], int] = {}
    hid = 0
    for comp in _components_mask(template == bg):
        if any(r == 0 or c == 0 or r == th - 1 or c == tw - 1 for r, c in comp):
            continue
        abs_cells = [(tr0 + int(r), tc0 + int(c)) for r, c in comp]
        holes[hid] = {
            "cells": abs_cells,
            "shape": _normalize_coords(abs_cells),
        }
        for cell in abs_cells:
            hole_by_cell[cell] = hid
        hid += 1
    assert holes, "template must contain at least one interior hole"

    infos: dict[int, dict] = {}
    colors = [int(v) for v in np.unique(g) if int(v) not in {bg, grey}]
    for color in colors:
        comps = _components_mask(g == color)
        marker = None
        piece = None
        for comp in comps:
            comp_list = [tuple(map(int, p)) for p in comp]
            inside_template = all(tr0 <= r <= tr1 and tc0 <= c <= tc1 for r, c in comp_list)
            if inside_template and len(comp_list) == 2:
                marker = comp
            else:
                piece = comp
        assert marker is not None and piece is not None, "each color must have one arrow and one fragment"
        target_dir, target_hole = _decode_arrow(marker, hole_by_cell)
        piece_shape = _normalize_coords(piece)
        piece_dir = _detect_piece_direction(piece_shape)
        infos[color] = {
            "target_dir": target_dir,
            "hole": target_hole,
            "piece_shape": piece_shape,
            "piece_dir": piece_dir,
        }

    out = np.full_like(g, bg)
    out[g == grey] = grey
    for color, info in infos.items():
        turns = (info["target_dir"] - info["piece_dir"]) % 4
        rotated = _rotate_rel(info["piece_shape"], turns)
        hole = holes[info["hole"]]
        assert rotated == hole["shape"], "rotated fragment must match its target hole exactly"
        for r, c in hole["cells"]:
            out[r, c] = color
    return out


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _grow_core(rng: random.Random) -> set[tuple[int, int]]:
    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    target = rng.randint(max(3, (h * w) // 2), h * w)
    seed = (rng.randrange(h), rng.randrange(w))
    cells = {seed}
    while len(cells) < target:
        frontier = []
        for r, c in list(cells):
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return cells


def _make_shape(rng: random.Random) -> tuple[tuple[tuple[int, int], ...], int]:
    for _ in range(300):
        core = _grow_core(rng)
        if len({r for r, _ in core}) < 2 or len({c for _, c in core}) < 2:
            continue
        for d in rng.sample(range(4), 4):
            dr, dc = DIRS[d]
            turns = rng.sample([-1, 1], 2)
            for side in turns:
                perp = DIRS[(d + side) % 4]
                bodies = list(core)
                rng.shuffle(bodies)
                for br, bc in bodies:
                    root = (br + dr, bc + dc)
                    leaf = (root[0] + perp[0], root[1] + perp[1])
                    if root in core or leaf in core or root == leaf:
                        continue
                    full = set(core)
                    full.add(root)
                    full.add(leaf)
                    if _neighbor_count(root, full) != 2 or _neighbor_count(leaf, full) != 1:
                        continue
                    if any((leaf[0] + x, leaf[1] + y) in full for x, y in DIRS if (leaf[0] + x, leaf[1] + y) != root):
                        continue
                    shape = _normalize_coords(full)
                    rr = [r for r, _ in shape]
                    cc = [c for _, c in shape]
                    if max(rr) + 1 > 5 or max(cc) + 1 > 5:
                        continue
                    try:
                        found = _detect_piece_direction(shape)
                    except AssertionError:
                        continue
                    if found == d:
                        return shape, d
    raise RuntimeError("failed to generate a valid fragment shape")


def _arrow_candidates(
    hole_cells: set[tuple[int, int]], target_dir: int, th: int, tw: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    dr, dc = DIRS[target_dir]
    out = []
    for hr, hc in hole_cells:
        root = (hr - dr, hc - dc)
        if not (0 <= root[0] < th and 0 <= root[1] < tw):
            continue
        if root in hole_cells:
            continue
        for side in (-1, 1):
            pr, pc = DIRS[(target_dir + side) % 4]
            leaf = (root[0] + pr, root[1] + pc)
            if not (0 <= leaf[0] < th and 0 <= leaf[1] < tw):
                continue
            if leaf in hole_cells or leaf == root:
                continue
            root_hits = sum((root[0] + x, root[1] + y) in hole_cells for x, y in DIRS)
            leaf_hits = sum((leaf[0] + x, leaf[1] + y) in hole_cells for x, y in DIRS)
            if root_hits == 1 and leaf_hits == 0:
                out.append((root, leaf))
    uniq = []
    seen = set()
    for pair in out:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)
    return uniq


def _place_holes_and_arrows(rng: random.Random, specs: list[dict], th: int, tw: int):
    reserve: set[tuple[int, int]] = set()
    placements = []
    for spec in specs:
        shape = spec["hole_shape"]
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        positions = [(r, c) for r in range(1, th - h) for c in range(1, tw - w)]
        rng.shuffle(positions)
        placed = None
        for r0, c0 in positions:
            hole_cells = _translate(shape, r0, c0)
            if hole_cells & reserve:
                continue
            candidates = _arrow_candidates(hole_cells, spec["target_dir"], th, tw)
            rng.shuffle(candidates)
            for root, leaf in candidates:
                arrow = {root, leaf}
                if arrow & reserve:
                    continue
                used = hole_cells | arrow
                placed = {
                    "color": spec["color"],
                    "hole_cells": hole_cells,
                    "arrow_cells": arrow,
                    "piece_shape": spec["piece_shape"],
                }
                reserve |= _expand_cells(used, th, tw)
                break
            if placed is not None:
                break
        if placed is None:
            return None
        placements.append(placed)
    return placements


def _place_scattered_pieces(
    rng: random.Random,
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    placements: list[dict],
) -> bool:
    h, w = grid.shape
    tr0, tc0, tr1, tc1 = template_bbox
    reserve = _expand_cells(
        {(r, c) for r in range(tr0, tr1 + 1) for c in range(tc0, tc1 + 1)},
        h,
        w,
    )
    for item in placements:
        shape = item["piece_shape"]
        ph = max(r for r, _ in shape) + 1
        pw = max(c for _, c in shape) + 1
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        rng.shuffle(positions)
        done = False
        for r0, c0 in positions:
            cells = _translate(shape, r0, c0)
            if cells & reserve:
                continue
            done = True
            for r, c in cells:
                grid[r, c] = item["color"]
            reserve |= _expand_cells(cells, h, w)
            break
        if not done:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, n)

    specs = []
    for color in colors:
        shape, base_dir = _make_shape(rng)
        target_dir = rng.randrange(4)
        hole_shape = _rotate_rel(shape, (target_dir - base_dir) % 4)
        turns_needed = rng.randrange(4)
        piece_shape = _rotate_rel(hole_shape, (-turns_needed) % 4)
        if _detect_piece_direction(hole_shape) != target_dir:
            return None
        specs.append(
            {
                "color": color,
                "target_dir": target_dir,
                "hole_shape": hole_shape,
                "piece_shape": piece_shape,
            }
        )

    th = rng.randint(12 + (n > 2), 16)
    tw = rng.randint(12 + (n > 2), 16)
    placements = _place_holes_and_arrows(rng, specs, th, tw)
    if placements is None:
        return None

    mt, mb, ml, mr = [rng.randint(4, 6) for _ in range(4)]
    h = th + mt + mb
    w = tw + ml + mr
    if h > 30 or w > 30:
        return None
    grid = np.full((h, w), BG, dtype=int)
    tr0, tc0 = mt, ml
    grid[tr0 : tr0 + th, tc0 : tc0 + tw] = GREY

    for item in placements:
        for r, c in item["hole_cells"]:
            grid[tr0 + r, tc0 + c] = BG
        for r, c in item["arrow_cells"]:
            grid[tr0 + r, tc0 + c] = item["color"]

    global_placements = []
    for item in placements:
        global_placements.append(
            {
                "color": item["color"],
                "piece_shape": item["piece_shape"],
            }
        )
    if not _place_scattered_pieces(rng, grid, (tr0, tc0, tr0 + th - 1, tc0 + tw - 1), global_placements):
        return None

    if _get_bg(grid) != BG:
        return None
    if _find_template_color(grid, BG) != GREY:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
