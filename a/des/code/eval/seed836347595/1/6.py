from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 8
TEMPLATE = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)
NEIGH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGH4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _corner_distance(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return min(
        r0 + c0,
        r0 + (w - 1 - c1),
        (h - 1 - r1) + c0,
        (h - 1 - r1) + (w - 1 - c1),
    )


def _find_template(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray, list[tuple[int, int]]]:
    comps = _components(grid == TEMPLATE)
    assert comps, "expected a grey template"
    best = None
    for coords in comps:
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        th = r1 - r0 + 1
        tw = c1 - c0 + 1
        if not (3 <= th <= 5 and 3 <= tw <= 5):
            continue
        score = (_corner_distance(bbox, grid.shape), -(coords.shape[0]), r0, c0)
        if best is None or score < best[0]:
            best = (score, bbox, coords)
    assert best is not None, "no valid template component found"
    _, (r0, c0, r1, c1), coords = best
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch == TEMPLATE
    offsets = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    return (r0, c0), mask, offsets


def _nearest_anchor(target: tuple[int, int], offsets: list[tuple[int, int]]) -> tuple[int, int]:
    tr, tc = target
    ranked = sorted(
        (
            (dr - tr) * (dr - tr) + (dc - tc) * (dc - tc),
            abs(dr - tr) + abs(dc - tc),
            dr,
            dc,
        )
        for dr, dc in offsets
    )
    _, _, ar, ac = ranked[0]
    return int(ar), int(ac)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a corner template, choose each color pair's anchor from its midpoint modulo
    the template size, then stamp the full template at both seeds of that color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    (tr0, tc0), template_mask, offsets = _find_template(g)
    th, tw = template_mask.shape
    template_abs = np.zeros((h, w), dtype=bool)
    template_abs[tr0 : tr0 + th, tc0 : tc0 + tw] = template_mask

    color_order: list[int] = []
    seeds: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v in (BG, TEMPLATE):
                continue
            if v not in seeds:
                color_order.append(v)
            seeds[v].append((r, c))

    out = g.copy()
    for pts in seeds.values():
        for r, c in pts:
            out[r, c] = BG

    for color in color_order:
        pts = seeds[color]
        assert len(pts) == 2, "each seed color must occur exactly twice"
        (r1, c1), (r2, c2) = pts
        mr = (r1 + r2) // 2
        mc = (c1 + c2) // 2
        target = (mr % th, mc % tw)
        anchor = target if bool(template_mask[target]) else _nearest_anchor(target, offsets)
        ar, ac = anchor

        rel = [(dr - ar, dc - ac) for dr, dc in offsets]
        for sr, sc in pts:
            for rr, cc in rel:
                pr = sr + rr
                pc = sc + cc
                if 0 <= pr < h and 0 <= pc < w:
                    out[pr, pc] = color

    out[template_abs] = TEMPLATE
    return out


def _random_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> None:
    r, c = start
    mask[r, c] = True
    while (r, c) != end:
        steps = []
        if r < end[0]:
            steps.append((r + 1, c))
        elif r > end[0]:
            steps.append((r - 1, c))
        if c < end[1]:
            steps.append((r, c + 1))
        elif c > end[1]:
            steps.append((r, c - 1))
        r, c = rng.choice(steps)
        mask[r, c] = True


def _is_nonsymmetric(mask: np.ndarray) -> bool:
    transforms = [np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    if mask.shape[0] == mask.shape[1]:
        transforms.extend([np.rot90(mask, 1), np.rot90(mask, 3), mask.T, np.fliplr(np.rot90(mask, 1))])
    return all(not np.array_equal(mask, other) for other in transforms)


def _template_candidates(mask: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]], list[tuple[int, int]]]:
    offsets = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    reverse: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    indirect_cells: list[tuple[int, int]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                reverse[(r, c)].append((r, c))
                continue
            ranked = sorted(
                (
                    (dr - r) * (dr - r) + (dc - c) * (dc - c),
                    dr,
                    dc,
                )
                for dr, dc in offsets
            )
            if len(ranked) >= 2 and ranked[0][0] == ranked[1][0]:
                continue
            _, ar, ac = ranked[0]
            reverse[(int(ar), int(ac))].append((r, c))
            indirect_cells.append((r, c))
    return offsets, reverse, indirect_cells


def _make_template_mask(rng: random.Random) -> np.ndarray | None:
    th = rng.randint(3, 5)
    tw = rng.randint(3, 5)
    for _ in range(200):
        mask = np.zeros((th, tw), dtype=bool)
        edge_points = [
            (0, rng.randrange(tw)),
            (th - 1, rng.randrange(tw)),
            (rng.randrange(th), 0),
            (rng.randrange(th), tw - 1),
        ]
        order = edge_points[:]
        rng.shuffle(order)
        for a, b in zip(order, order[1:]):
            _random_path(mask, a, b, rng)

        target_lo = max(5, int(mask.sum()))
        target_hi = th * tw - 1
        if target_lo > target_hi:
            continue
        target = rng.randint(target_lo, target_hi)
        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in NEIGH4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < th and 0 <= nc < tw and not bool(mask[nr, nc]):
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(frontier)
            mask[nr, nc] = True

        if not _is_nonsymmetric(mask):
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        _, _, indirect_cells = _template_candidates(mask)
        if indirect_cells:
            return mask
    return None


def _corner_origin(grid_h: int, grid_w: int, th: int, tw: int, corner: str, margin: int) -> tuple[int, int]:
    if corner == "tl":
        return margin, margin
    if corner == "tr":
        return margin, grid_w - tw - margin
    if corner == "bl":
        return grid_h - th - margin, margin
    return grid_h - th - margin, grid_w - tw - margin


def _footprint_mask(
    grid_h: int,
    grid_w: int,
    seed: tuple[int, int],
    rel: list[tuple[int, int]],
) -> np.ndarray | None:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    sr, sc = seed
    for dr, dc in rel:
        rr = sr + dr
        cc = sc + dc
        if not (0 <= rr < grid_h and 0 <= cc < grid_w):
            return None
        mask[rr, cc] = True
    return mask


def _place_pair(
    grid_h: int,
    grid_w: int,
    template_offsets: list[tuple[int, int]],
    anchor: tuple[int, int],
    target_cell: tuple[int, int],
    template_shape: tuple[int, int],
    guard: np.ndarray,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], np.ndarray] | None:
    th, tw = template_shape
    ar, ac = anchor
    rel = [(dr - ar, dc - ac) for dr, dc in template_offsets]
    min_dr = min(dr for dr, _ in rel)
    max_dr = max(dr for dr, _ in rel)
    min_dc = min(dc for _, dc in rel)
    max_dc = max(dc for _, dc in rel)
    row_lo = -min_dr
    row_hi = grid_h - 1 - max_dr
    col_lo = -min_dc
    col_hi = grid_w - 1 - max_dc

    rows = [r for r in range(row_lo, row_hi + 1) if r % th == target_cell[0]]
    cols = [c for c in range(col_lo, col_hi + 1) if c % tw == target_cell[1]]
    if not rows or not cols:
        return None

    mids = [(r, c) for r in rows for c in cols]
    rng.shuffle(mids)

    delta_options = []
    max_sep_r = min(6, row_hi - row_lo)
    max_sep_c = min(6, col_hi - col_lo)
    for dr in range(max_sep_r + 1):
        for dc in range(max_sep_c + 1):
            for pr in (0, 1):
                for pc in (0, 1):
                    if dr == 0 and dc == 0 and pr == 0 and pc == 0:
                        continue
                    delta_options.append((dr, dc, pr, pc))
    rng.shuffle(delta_options)

    for mr, mc in mids:
        for dr, dc, pr, pc in delta_options:
            s1 = (mr - dr, mc - dc)
            s2 = (mr + dr + pr, mc + dc + pc)
            if s1 == s2:
                continue
            if not (row_lo <= s1[0] <= row_hi and row_lo <= s2[0] <= row_hi):
                continue
            if not (col_lo <= s1[1] <= col_hi and col_lo <= s2[1] <= col_hi):
                continue
            m1 = _footprint_mask(grid_h, grid_w, s1, rel)
            m2 = _footprint_mask(grid_h, grid_w, s2, rel)
            assert m1 is not None and m2 is not None
            if bool((m1 & m2).any()):
                continue
            if bool((_dilate8(m1) & m2).any()) or bool((_dilate8(m2) & m1).any()):
                continue
            combo = m1 | m2
            if bool((combo & guard).any()):
                continue
            return s1, s2, combo
    return None


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_colors = rng.randint(2, 4)
    grid_h = rng.randint(17 + n_colors, 25)
    grid_w = rng.randint(17 + n_colors, 25)

    template_mask = _make_template_mask(rng)
    if template_mask is None:
        return None
    th, tw = template_mask.shape

    corner = rng.choice(("tl", "tr", "bl", "br"))
    margin = rng.choice((0, 1))
    if grid_h < th + margin or grid_w < tw + margin:
        return None
    tr0, tc0 = _corner_origin(grid_h, grid_w, th, tw, corner, margin)

    template_offsets, reverse_map, indirect_cells = _template_candidates(template_mask)
    anchors = [tuple(map(int, rc)) for rc in np.argwhere(template_mask)]
    if not indirect_cells:
        return None

    if n_colors == 2:
        branches = ["direct", "indirect"]
    elif n_colors == 3:
        branches = ["direct", "direct", "indirect"] if rng.random() < 0.5 else ["direct", "indirect", "indirect"]
    else:
        branches = ["direct", "direct", "indirect", "indirect"]
    rng.shuffle(branches)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[tr0 : tr0 + th, tc0 : tc0 + tw][template_mask] = TEMPLATE

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[tr0 : tr0 + th, tc0 : tc0 + tw] = template_mask
    guard = _dilate8(occupied)

    colors = list(rng.sample(PALETTE, n_colors))
    for color, branch in zip(colors, branches, strict=False):
        for _ in range(120):
            anchor = rng.choice(anchors)
            cells = [cell for cell in reverse_map[anchor] if bool(template_mask[cell]) == (branch == "direct")]
            if not cells:
                continue
            target_cell = rng.choice(cells)
            placed = _place_pair(
                grid_h,
                grid_w,
                template_offsets,
                anchor,
                target_cell,
                (th, tw),
                guard,
                rng,
            )
            if placed is None:
                continue
            s1, s2, combo = placed
            grid[s1] = color
            grid[s2] = color
            occupied |= combo
            guard = _dilate8(occupied)
            break
        else:
            return None

    for color in colors:
        if int(np.sum(grid == color)) != 2:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if any(int(np.sum(out == color)) <= 2 for color in colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
