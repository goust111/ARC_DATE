import numpy as np

from arc_utils import bbox_pts as _bbox_pts, cc4 as _cc4, get_bg as _mode_color


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]] | None:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    best = None
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in _cc4(g == col):
            r0, c0, r1, c1 = _bbox_pts(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if best is None or area > best[0]:
                best = (area, col, (r0, c0, r1, c1))
    if best is None:
        return None
    return int(best[1]), best[2]


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region using placeholder positions and a color legend.

    Input structure:
    - bg background, a rectangular single-color frame, colored placeholder cells inside the frame.
    - Outside the frame: sprite shapes (one per placeholder color) and a small legend mapping each placeholder color to an output color.

    Rule:
    - Find the frame and its interior placeholders.
    - For each placeholder color, locate the matching-color sprite outside the frame.
    - Stamp each sprite at the placeholder's position inside the frame.
    - Recolor every stamped sprite to the output color defined by the legend.

    Key insight: The legend pairs (overlapping two-color glyphs outside the frame) dictate the final recoloring — placeholder color selects the shape, legend maps it to the output color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    fr = _find_frame(g, bg)
    if fr is None:
        return g.copy()
    frame_col, (fr0, fc0, fr1, fc1) = fr

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape

    # Placeholder components inside the frame interior.
    inner_mask = np.zeros((H, W), dtype=bool)
    if H >= 3 and W >= 3:
        inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in _cc4((crop == col) & inner_mask):
            r0, c0, r1, c1 = _bbox_pts(pts)
            placeholders.append((col, (r0, c0, r1, c1)))
    # Expected placement bbox size per placeholder color (mode).
    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        if dims:
            vals, cnts = np.unique(np.array(dims), axis=0, return_counts=True)
            phw[col] = tuple(map(int, vals[int(cnts.argmax())]))

    # Extract outside-frame components for sprite shapes and for legend color mapping.
    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    # Collect all outside components (with ids) for later pairing.
    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in _cc4((g == col) & outside):
            r0, c0, r1, c1 = _bbox_pts(pts)
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts),
                }
            )
            cid += 1

    def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ar0, ac0, ar1, ac1 = a
        br0, bc0, br1, bc1 = b
        return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)

    # Legend mapping: detect small two-color glyph regions where two components overlap in bbox
    # and the union bbox contains only those two colors (no background).
    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            # restrict to compact legend-like regions
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors:
                continue
            if len(sub_colors) != 2:
                continue
            cset = set(sub_colors)
            if ci["col"] not in cset or cj["col"] not in cset:
                continue

            # Decide source->target.
            # If exactly one color is used as a placeholder inside the frame, that one is source.
            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                # Otherwise: the larger component in this pair is treated as the source.
                if ci["area"] >= cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]

            if src in placeholder_colors and src not in color_map:
                color_map[src] = int(tgt)
                legend_ids.add(ci["id"])
                legend_ids.add(cj["id"])

    # Sprite shapes: for each placeholder color, pick a non-legend outside component (largest).
    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        if not all_candidates:
            all_candidates = [c for c in comps if c["col"] == src_col]
        if not all_candidates:
            continue
        base_candidates = list(all_candidates)
        # Prefer base candidates whose bbox matches the placeholder bbox size.
        if src_col in phw:
            want_h, want_w = phw[src_col]
            sized = [
                c
                for c in base_candidates
                if (c["bbox"][2] - c["bbox"][0] + 1) == want_h and (c["bbox"][3] - c["bbox"][1] + 1) == want_w
            ]
            if sized:
                base_candidates = sized

        base = max(base_candidates, key=lambda x: x["area"])
        sr0, sc0, sr1, sc1 = base["bbox"]

        # Some sprites can be split into multiple 4-connected components of the same color.
        # Merge any additional (non-legend) components fully contained in the chosen sprite bbox.
        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            if sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1:
                pts_set.update(c["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    # Build output crop: keep frame border, clear interior to bg, then paste recolored sprites.
    out = crop.copy()
    if H >= 3 and W >= 3:
        out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        if src_col not in sprite_coords or src_col not in color_map:
            continue
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        # Align sprite's bbox top-left to placeholder bbox top-left.
        off_r = pr0
        off_c = pc0
        for dr, dc in coords:
            rr, cc = off_r + dr, off_c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = tgt_col

    return out
