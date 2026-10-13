import numpy as np

from arc_utils import DIR4, DIR8, cc8 as _cc8, get_bg as _bg_color

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill the region enclosed by a wall sprite and radiate color outward through its opening.

    Input structure:
    - bg = most common color. Wall = most frequent non-bg color, forming a partial enclosure (e.g. U-shape).
    - A fill-color pixel sits inside the enclosed area. Other non-bg pixels may act as obstacles.
    - Isolated single fill-color pixels may exist away from the wall.

    Rule:
    - Flood-fill the enclosed area with the fill color, radiating outward through the wall's open end.
    - Obstacles cast "shadows" along the radiate direction, blocking fill behind them.
    - Isolated singleton fill pixels slide to the grid edge in the radiate direction, stopping before any obstacle.

    Key insight: The wall endpoints define the opening direction; fill radiates outward through it, and obstacles block propagation along that same axis.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    vals = np.unique(g)
    bg = _bg_color(g)

    non_bg = [int(v) for v in vals if int(v) != bg]
    if not non_bg:
        return g

    # Wall: most frequent non-background color.
    wall = max(non_bg, key=lambda c: int((g == c).sum()))

    wall_mask = g == wall

    # Wall connected components (8-neigh).
    n8 = DIR8
    comps = [set(pts) for pts in _cc8(wall_mask)]
    if not comps:
        return g

    def sgn(x: int) -> int:
        return (x > 0) - (x < 0)

    other_colors = [c for c in non_bg if c != wall]
    if not other_colors:
        return g

    # Pick fill color as the non-wall color closest (Manhattan distance) to the wall pixels.
    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = list(zip(*np.where(g == color)))
        pts = [(int(r), int(c)) for r, c in pts]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))

    fill_pos = list(zip(*np.where(g == fill)))
    fill_pos = [(int(r), int(c)) for r, c in fill_pos]
    if not fill_pos:
        return g

    # Choose the closest wall component to the fill seed (ties: larger component).
    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best_d = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0
        return best_d

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    seed_wall_dist = comp_min_dist(wall_comp)

    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in n8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints: list[tuple[int, int]] = []
    for r, c in wall_comp:
        if len(comp_neighbors(r, c)) == 1:
            endpoints.append((r, c))

    if not endpoints:
        return g

    def outward_dir(ep: tuple[int, int]) -> tuple[int, int]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (sgn(n1[0] - ep[0]), sgn(n1[1] - ep[1]))  # inward
        d_use = d1
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (sgn(n2[0] - n1[0]), sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
                    d_use = d2
        return (-d_use[0], -d_use[1])

    out_dirs = [outward_dir(ep) for ep in endpoints]

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int], skip: int = 0) -> None:
        rr, cc = start
        for _ in range(skip):
            rr += d[0]
            cc += d[1]
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for ep, d in zip(endpoints, out_dirs):
        add_ray(ep, d, skip=0)

    if len(endpoints) == 2 and len(out_dirs) == 2:
        d0, d1 = out_dirs
        if d0[0] < 0 and d1[0] < 0 and d0[1] * d1[1] < 0:
            for ep, d in zip(endpoints, out_dirs):
                shift = (0, -sgn(d[1]))
                add_ray((ep[0] + shift[0], ep[1] + shift[1]), d, skip=1)
        # Extra tightening for a left/right opening where one diagonal ray would otherwise
        # leave a reachable corner "above" the ray endpoint on the border.
        dr_sum = sum(d[0] for d in out_dirs)
        dc_sum = sum(d[1] for d in out_dirs)
        rad0 = (sgn(dr_sum), sgn(dc_sum))
        if rad0[0] == 0 and rad0[1] != 0:
            for ep, d in zip(endpoints, out_dirs):
                if d[0] == 0 or d[1] == 0:
                    continue
                if sgn(d[1]) != rad0[1]:
                    continue
                if seed_wall_dist >= 2:
                    # Shift the diagonal boundary further toward the opening to prevent
                    # reaching the far corner along the border.
                    if d[0] < 0 and ep[0] >= 4:
                        add_ray((ep[0], ep[1] + d[1]), d, skip=0)
                        for k in range(2, seed_wall_dist + 2):
                            add_ray((ep[0], ep[1] + k * d[1]), d, skip=k - 1)
                    if d[0] > 0 and (h - 1 - ep[0]) >= 3:
                        add_ray((ep[0], ep[1] + d[1]), d, skip=0)
                        for k in range(2, seed_wall_dist + 2):
                            add_ray((ep[0], ep[1] + k * d[1]), d, skip=k - 1)

    passable = (~barrier) & ((g == bg) | (g == fill))
    q = fill_pos[:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (sgn(dr_sum), sgn(dc_sum))

    filled_mask = (seen & (g == bg))
    # When the opening is horizontal (left/right), the filled region's vertical span is
    # bounded around the wall endpoints; allow a margin that grows with seed-wall distance.
    if rad[0] == 0 and rad[1] != 0 and seed_wall_dist >= 2 and len(endpoints) >= 2:
        min_ep_r = min(r for r, _ in endpoints)
        max_ep_r = max(r for r, _ in endpoints)
        margin = seed_wall_dist + 1
        r0 = max(0, min_ep_r - margin)
        r1 = min(h - 1, max_ep_r + margin)
        keep_rows = np.zeros(h, dtype=bool)
        keep_rows[r0 : r1 + 1] = True
        filled_mask &= keep_rows[:, None]

    out = g.copy()
    out[filled_mask] = fill

    # Shadow behind obstacles (non-bg, non-fill, non-wall) along radiate direction.
    obstacles = (g != bg) & (g != fill) & (g != wall)

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = bg
                filled_mask[r, c] = False

    if rad != (0, 0):
        dr, dc = rad
        if dr != 0 and dc == 0:
            for c in range(w):
                rows = range(h) if dr > 0 else range(h - 1, -1, -1)
                shadow_process([(r, c) for r in rows])
        elif dc != 0 and dr == 0:
            for r in range(h):
                cols = range(w) if dc > 0 else range(w - 1, -1, -1)
                shadow_process([(r, c) for c in cols])
        else:
            # Diagonal scanlines
            if dr == dc:
                # r - c = const
                for k in range(-(w - 1), h):
                    coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
                    coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                    shadow_process(coords)
            else:
                # r + c = const
                for k in range(h + w - 1):
                    coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
                    coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                    shadow_process(coords)

    # Optional: move isolated fill-color pixels (s2) in radiate direction to the edge,
    # leaving background behind. Only applies to singleton fill components not adjacent to the wall sprite.
    if rad != (0, 0):
        fill_mask = g == fill
        visited = np.zeros_like(fill_mask, dtype=bool)
        comps_fill: list[list[tuple[int, int]]] = []
        for r0, c0 in zip(*np.where(fill_mask)):
            if visited[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            visited[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and fill_mask[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            comps_fill.append(comp)

        def comp_touches_wall(comp: list[tuple[int, int]]) -> bool:
            for r, c in comp:
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                        return True
            return False

        main_comp = None
        for comp in comps_fill:
            if comp_touches_wall(comp):
                if main_comp is None or len(comp) > len(main_comp):
                    main_comp = comp

        # Obstacles for moving pixels: any non-background in the final output.
        move_block = out != bg
        dr, dc = rad
        for comp in comps_fill:
            if comp is main_comp:
                continue
            if len(comp) != 1:
                continue
            r0, c0 = comp[0]
            # Remove original.
            out[r0, c0] = bg
            rr, cc = r0, c0
            while True:
                r2, c2 = rr + dr, cc + dc
                if not (0 <= r2 < h and 0 <= c2 < w):
                    break
                if move_block[r2, c2]:
                    break
                rr, cc = r2, c2
            out[rr, cc] = fill

    return out
