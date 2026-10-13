from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, label_mask, paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched horizontal bars with a color swap.

    Input structure:
    - bg = most common color.
    - Top area: small two-tone sprites (outer color + inner color).
    - Below: horizontal bars spanning most of the row, each with a fill color and border color on edges.

    Rule:
    - Match each sprite to the bar whose fill color equals the sprite's outer color.
    - Place the sprite at its original x-position, aligned to the bar's bottom.
    - Recolor: sprite outer → bar border, sprite inner → bar fill.
    - Erase original sprites from the top area.

    Key insight: Color matching is inverted — the sprite's dominant color identifies its target bar, then gets swapped to the bar's accent color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    bg = int(Counter(g.flatten()).most_common(1)[0][0])

    # Find bars.
    bars = []
    current_bar = None
    for r in range(H):
        row = g[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            c = Counter(non_bg)
            fill_color, count = c.most_common(1)[0]
            if int(count) > int(W * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar:
                        bars.append(current_bar)
                    current_bar = {"fill": int(fill_color), "border": int(border), "r_min": int(r), "r_max": int(r)}
            else:
                if current_bar:
                    bars.append(current_bar)
                    current_bar = None
        else:
            if current_bar:
                bars.append(current_bar)
                current_bar = None
    if current_bar:
        bars.append(current_bar)

    if not bars:
        return out

    pattern_area_end = int(min(int(b["r_min"]) for b in bars))

    # Find patterns in top area: monochrome components by color, but we need full bbox content.
    patterns = []
    top = g[:pattern_area_end, :]
    for col in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(col)
        if not bool(mask.any()):
            continue
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            b = bbox_coords(coords)
            r0, c0, r1, c1 = map(int, b)
            bbox = crop_bbox(g, (r0, c0, r1, c1))

            colors_in_bbox = Counter(map(int, bbox.flatten()))
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue
            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            inner_colors = [int(c) for c in colors_in_bbox.keys() if int(c) != int(outer_color)]
            inner_color = int(inner_colors[0]) if inner_colors else None

            patterns.append(
                {
                    "outer_color": int(outer_color),
                    "inner_color": inner_color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )

    # Clear patterns from output.
    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    # Match and place patterns.
    for bar in bars:
        matching = [p for p in patterns if int(p["outer_color"]) == int(bar["fill"])]
        for pat in matching:
            transformed = np.array(pat["content"], dtype=int).copy()
            for rr in range(int(pat["h"])):
                for cc in range(int(pat["w"])):
                    v = int(transformed[int(rr), int(cc)])
                    if v == int(pat["outer_color"]):
                        transformed[int(rr), int(cc)] = int(bar["border"])
                    elif pat["inner_color"] is not None and v == int(pat["inner_color"]):
                        transformed[int(rr), int(cc)] = int(bar["fill"])

            place_r = int(bar["r_max"] - pat["h"] + 1)
            place_c = int(pat["c0"])
            paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out

