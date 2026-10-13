import numpy as np

from arc_utils import bbox_pts as _bbox_pts, cc4 as _cc4, get_bg as _mode_color
from arc_utils import DIR4


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(n**0.5) + 1):
        if n % h == 0:
            w = n // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    return out


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, w = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


def _dihedral_orientations(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    """
    Unique orientations under rotations + mirror (D4).
    Returned as (coords, h, w, rot_k, flipped), where the transform is:
      - if flipped: horizontal flip (mirror left-right)
      - then rot_k times 90° clockwise rotation
    coords are normalized to start at (0,0).
    """
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]] = []
    cur = coords
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), False))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), True))
        cur = _rot90(cur)[0]
    return out


def _find_legend_patch(
    g: np.ndarray, pts: list[tuple[int, int]], bg: int
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    """
    Find a 2x2 multi-color patch fully inside this component.
    Returns (score, tie_key, (tl,tr,bl,br)) where score is number of distinct colors in the 2x2.
    """
    r0, c0, r1, c1 = _bbox_pts(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    occ = np.zeros(sub.shape, dtype=bool)
    for r, c in pts:
        occ[r - r0, c - c0] = True
    best = None
    for rr in range(sub.shape[0] - 1):
        for cc in range(sub.shape[1] - 1):
            if not (occ[rr, cc] and occ[rr, cc + 1] and occ[rr + 1, cc] and occ[rr + 1, cc + 1]):
                continue
            vals = (
                int(sub[rr, cc]),
                int(sub[rr, cc + 1]),
                int(sub[rr + 1, cc]),
                int(sub[rr + 1, cc + 1]),
            )
            if any(v == bg for v in vals):
                continue
            distinct = len(set(vals))
            if distinct < 3:
                continue
            # prefer most distinct, then top-leftmost
            key = (distinct, -rr, -cc)
            if best is None or key > best[0]:
                best = (key, (vals[0], vals[1], vals[2], vals[3]))
    if best is None:
        return None
    key, colors = best
    score = key[0]
    tie = (-key[1]) * 1000 + (-key[2])
    return score, tie, colors


def _extract_legend_map(g: np.ndarray, pts: list[tuple[int, int]], k: int) -> np.ndarray | None:
    """
    Inside the legend component's bounding box, find a k×k color map block (may include 0).
    We pick the k×k window with the maximum number of distinct colors; ties go to top-left.
    """
    r0, c0, r1, c1 = _bbox_pts(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    if k <= 0 or hh < k or ww < k:
        return None
    best = None  # (distinct, -rr, -cc)
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    if best_rc is None:
        return None
    rr, cc = best_rc
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


def _fill_holes_in_component(g: np.ndarray, pts: list[tuple[int, int]], bg: int) -> tuple[tuple[tuple[int, int], ...], int]:
    """
    Treat background 'holes' fully enclosed inside the component bbox as part of the piece.
    Returns (filled_pts, hole_count), where filled_pts is a sorted tuple of absolute coords.
    """
    r0, c0, r1, c1 = _bbox_pts(pts)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    occ = np.zeros((hh, ww), dtype=bool)
    for r, c in pts:
        occ[r - r0, c - c0] = True

    bgmask = ~occ
    vis = np.zeros_like(bgmask, dtype=bool)
    st: list[tuple[int, int]] = []

    for rr in range(hh):
        for cc in range(ww):
            if rr in (0, hh - 1) or cc in (0, ww - 1):
                if bgmask[rr, cc] and not vis[rr, cc]:
                    vis[rr, cc] = True
                    st.append((rr, cc))
                    while st:
                        r, c = st.pop()
                        for dr, dc in DIR4:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < hh and 0 <= c2 < ww and bgmask[r2, c2] and not vis[r2, c2]:
                                vis[r2, c2] = True
                                st.append((r2, c2))

    holes = bgmask & ~vis
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(pts).union(hole_coords)))
    return filled, int(holes.sum())


def _assemble_exact_cover(
    g: np.ndarray,
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
    bg: int,
) -> np.ndarray:
    """
    Tile all pieces into a full rectangle (no holes), allowing rotations + reflections.
    Recolor each piece by quantizing its centroid into a k×k grid, using legend_map.
    """
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    dims.sort(key=lambda hw: (abs(hw[0] - hw[1]), hw[0] * hw[1], hw[0] + hw[1]))

    for H, W in dims:
        nbits = H * W
        all_mask = (1 << nbits) - 1

        # Precompute placements for non-legend pieces (legend is fixed at (0,0), but orientation varies).
        other_pis = [i for i in range(len(pieces)) if i != legend_idx]
        placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
        for pi in other_pis:
            opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
            for coords, ph, pw, _, _ in pieces[pi]["oris"]:
                for r0 in range(0, H - ph + 1):
                    for c0 in range(0, W - pw + 1):
                        m = 0
                        for dr, dc in coords:
                            idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                            m |= 1 << idx
                        opts.append((m, int(r0), int(c0), coords))
            placements_other.append(opts)

        def solve_with_fixed_legend(
            legend_coords: tuple[tuple[int, int], ...],
        ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
            legend_mask = 0
            for dr, dc in legend_coords:
                idx = int(dr) * int(W) + int(dc)
                legend_mask |= 1 << idx

            filtered: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
            for opts in placements_other:
                filtered.append([pl for pl in opts if (pl[0] & legend_mask) == 0])

            remaining = set(range(len(other_pis)))
            chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {
                legend_idx: (legend_mask, 0, 0, legend_coords)
            }

            def search(used: int) -> bool:
                if not remaining:
                    return used == all_mask
                empty = (~used) & all_mask
                if empty == 0:
                    return False
                cell = (empty & -empty).bit_length() - 1
                # Branch over *any* remaining piece that can cover this cell.
                # (Picking only one "best" piece can miss valid tilings.)
                candidates = []
                for k in sorted(remaining):
                    opts_k = []
                    for pl in filtered[k]:
                        m = pl[0]
                        if (m >> cell) & 1 and (m & used) == 0:
                            opts_k.append(pl)
                    if opts_k:
                        candidates.append((len(opts_k), k, opts_k))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, k, opts_k in candidates:
                    remaining.remove(k)
                    pi = other_pis[k]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(k)
                return False

            if not search(legend_mask):
                return None
            return chosen

        # Try each possible legend orientation at (0,0).
        chosen = None
        for coords, ph, pw, rot_k, flipped in pieces[legend_idx]["oris"]:
            if ph > H or pw > W:
                continue
            sol = solve_with_fixed_legend(coords)
            if sol is None:
                continue
            chosen = sol
            break
        if chosen is None:
            continue

        # Determine legend-map cell for each piece by centroid.
        piece_cell: dict[int, tuple[int, int]] = {}
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            # centroid: average over occupied cells
            rs = [r0 + dr for dr, _ in coords]
            cs = [c0 + dc for _, dc in coords]
            cr = float(sum(rs)) / float(len(rs))
            cc = float(sum(cs)) / float(len(cs))
            rr = int(cr * float(k) / float(H))
            cc2 = int(cc * float(k) / float(W))
            rr = max(0, min(int(k) - 1, rr))
            cc2 = max(0, min(int(k) - 1, cc2))
            piece_cell[pi] = (rr, cc2)

        # Build final output.
        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rr, cc2 = piece_cell[pi]
            col = int(legend_map[rr, cc2])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out

    # If we couldn't assemble, fall back to returning the non-bg tight crop.
    pts = np.argwhere(g != bg)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, recoloring each by a legend sprite's color map.

    Input structure:
    - bg = most common color.
    - Several monochrome non-bg sprites (the "pieces"), scattered across the grid.
    - One sprite is the legend: it contains a multi-color k×k patch (typically 2×2) encoding output colors.

    Rule:
    - Tile all pieces (rotations/reflections allowed) into a tight rectangle with no gaps.
    - The legend's k×k color patch defines a spatial color map over the rectangle.
    - Each piece is recolored to the legend-map color corresponding to its position in the assembled rectangle.

    Key insight: The legend piece doubles as both a jigsaw piece and a color key — its internal k×k palette assigns colors by spatial quadrant.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    comps = _cc4(g != bg)
    if not comps:
        return g.copy()

    # Identify legend component (most distinct colors) and infer k from number of pieces.
    k = int(round(len(comps) ** 0.5))
    if k * k != len(comps):
        k = 2

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    legend_idx = max(range(len(comps)), key=lambda i: (len(comp_colors[i]), len(comps[i])))
    legend_map = _extract_legend_map(g, comps[legend_idx], k)
    if legend_map is None and k == 2:
        patch = _find_legend_patch(g, comps[legend_idx], bg)
        if patch is not None:
            _, _, colors = patch
            legend_map = np.array([[colors[0], colors[1]], [colors[2], colors[3]]], dtype=int)

    # Build piece masks (coords in bbox space), including filled holes.
    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, r1, c1 = _bbox_pts(list(filled_abs))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append({"oris": oris, "area": len(filled_abs)})

    if legend_map is None:
        # No legend map: just return tight crop of non-bg.
        pts = np.argwhere(g != bg)
        if pts.size == 0:
            return np.zeros((0, 0), dtype=int)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return g[r0 : r1 + 1, c0 : c1 + 1].copy()

    return _assemble_exact_cover(g, pieces, legend_idx, legend_map, k, bg)
