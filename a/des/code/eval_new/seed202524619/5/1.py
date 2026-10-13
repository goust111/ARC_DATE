import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _four_neighbors(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _find_components(grid: np.ndarray, color: int) -> List[Tuple[List[Tuple[int, int]], bool, Tuple[int, int, int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if grid[sr, sc] != color or visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells: List[Tuple[int, int]] = []
            touches_border = False
            rmin = sr
            rmax = sr
            cmin = sc
            cmax = sc
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                if r < rmin:
                    rmin = r
                elif r > rmax:
                    rmax = r
                if c < cmin:
                    cmin = c
                elif c > cmax:
                    cmax = c
                for nr, nc in _four_neighbors(r, c, h, w):
                    if not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append((cells, touches_border, (rmin, rmax, cmin, cmax)))
    return comps


def _extract_palette(grid: np.ndarray, bg: int) -> Tuple[np.ndarray, Dict[int, np.ndarray], Tuple[int, int, int, int]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    palette_mask = np.zeros((h, w), dtype=bool)
    palette_map: Dict[int, np.ndarray] = {}
    chosen_box = (0, -1, 0, -1)

    for sr in range(h):
        for sc in range(w):
            if grid[sr, sc] == bg or visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells = []
            rmin = sr
            rmax = sr
            cmin = sc
            cmax = sc
            touches_corner = False
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_corner = True
                if r < rmin:
                    rmin = r
                elif r > rmax:
                    rmax = r
                if c < cmin:
                    cmin = c
                elif c > cmax:
                    cmax = c
                for nr, nc in _four_neighbors(r, c, h, w):
                    if not visited[nr, nc] and grid[nr, nc] != bg:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            if not touches_corner:
                continue

            if not ((rmin == 0 and cmin == 0) or (rmin == 0 and cmax == w - 1) or (rmax == h - 1 and cmin == 0) or (rmax == h - 1 and cmax == w - 1)):
                continue

            area = (rmax - rmin + 1) * (cmax - cmin + 1)
            if area != len(cells):
                continue
            width = cmax - cmin + 1
            height = rmax - rmin + 1
            if width < 3 or height < 1:
                continue

            pattern_w = width - 1
            if pattern_w < 2:
                continue

            valid = True
            tmp_map: Dict[int, np.ndarray] = {}
            row_colors = []
            for rr in range(rmin, rmax + 1):
                idx = int(grid[rr, cmin])
                if idx == bg:
                    valid = False
                    break
                row = grid[rr, cmin + 1 : cmax + 1]
                if row.shape[0] != pattern_w or row.min() == bg:
                    valid = False
                    break
                tmp_map[idx] = row.copy()
                row_colors.append(idx)
                if np.any(row == bg):
                    valid = False
                    break
            if not valid:
                continue

            # Choose the first credible corner sprite.
            # Keep deterministic tie-break order by area-first,
            # then top-left to avoid instability.
            palette_mask = np.zeros((h, w), dtype=bool)
            palette_mask[rmin : rmax + 1, cmin : cmax + 1] = True
            palette_map = tmp_map
            chosen_box = (rmin, rmax, cmin, cmax)
            return palette_mask, palette_map, chosen_box

    return palette_mask, palette_map, chosen_box


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each closed interior with a tile from palette rows selected by in-region seed color.
Input structure:
- bg is the most frequent color.
- Multiple one-pixel-wide, same-color closed contours define enclosed regions; each region interior has exactly one unique non-bg seed pixel.
- A palette sprite in a corner stores rows where column 0 is index color and the remaining columns are the row pattern.
Rule:
- Find closed-region interiors (bg pixels not connected to border), locate each region's unique seed color, find its matching palette row by index color, and tile that row left-to-right on every row inside the region interior (periodic horizontally).
- Only interior bg pixels are rewritten; contours and seed pixels remain unchanged.
- The palette sprite area is erased to bg in the output.
Key insight:
- The seed color is a lookup key, while its region is filled by periodic pattern replication instead of a solid fill.
"""
    grid = np.asarray(grid)
    out = np.array(grid, copy=True)
    h, w = out.shape
    bg = _bg_color(out)

    # Detect the palette sprite.
    palette_mask, palette_map, _ = _extract_palette(out, bg)

    # Find enclosed background components.
    components = _find_components(out, bg)

    # Quick index for palette lookup.
    palette_positions = list(palette_map.keys())

    for cells, touches_border, (rmin, rmax, cmin, cmax) in components:
        if touches_border or not cells:
            continue

        if not palette_map:
            continue

        # Collect non-bg pixels in this bbox excluding palette area.
        colors = defaultdict(list)
        for rr in range(rmin, rmax + 1):
            for cc in range(cmin, cmax + 1):
                v = int(out[rr, cc])
                if v == bg or palette_mask[rr, cc]:
                    continue
                colors[v].append((rr, cc))

        seed_color = None
        for key in palette_positions:
            if len(colors.get(key, [])) == 1:
                seed_color = key
                break
        if seed_color is None:
            continue

        pattern = palette_map.get(seed_color)
        if pattern is None or len(pattern) == 0:
            continue

        pattern_len = len(pattern)
        row_cells = defaultdict(list)
        for rr, cc in cells:
            row_cells[rr].append(cc)

        for rr, cols in row_cells.items():
            for cc in cols:
                out[rr, cc] = int(pattern[(cc - cmin) % pattern_len])

    if palette_mask.any():
        out[palette_mask] = bg

    return out


def _draw_rect_contour(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    bottom = top + h - 1
    right = left + w - 1
    for c in range(left, right + 1):
        grid[top, c] = color
        grid[bottom, c] = color
    for r in range(top, bottom + 1):
        grid[r, left] = color
        grid[r, right] = color


def _palette_rows_for_colors(row_colors: List[int], pattern_w: int) -> Dict[int, np.ndarray]:
    palette_rows = {}
    for idx in row_colors:
        if random.random() < 0.5:
            c1 = random.randint(0, 9)
            c2 = (c1 + 1) % 10
            pattern = [c1, c2] * (pattern_w // 2 + 1)
            pattern = pattern[:pattern_w]
        elif random.random() < 0.85:
            a = random.randint(1, 9)
            d = random.choice([1, 2, 3])
            pattern = [(a + d * k) % 10 for k in range(pattern_w)]
            pattern = [p if p != 0 else 1 for p in pattern]
        else:
            base = random.randint(1, 9)
            pattern = [random.randint(1, 9) for _ in range(pattern_w)]
        palette_rows[int(idx)] = np.array(pattern, dtype=int)
    return palette_rows


def _build_palette(grid: np.ndarray, corner: str, region_count: int, bg: int) -> None:
    h, w = grid.shape
    colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(colors)

    pattern_w = random.choice([2, 3, 4])
    row_count = region_count + random.randint(0, 2)
    if row_count < 1:
        row_count = 1
    if row_count > len(colors) - 1:
        row_count = len(colors) - 1

    # Ensure the palette fits.
    if pattern_w + 1 >= w:
        pattern_w = max(2, w - 3)
    row_count = min(row_count, h - 3)

    idx_colors = colors[:row_count]

    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - (pattern_w + 1)
    elif corner == "bl":
        r0, c0 = h - row_count, 0
    else:
        r0, c0 = h - row_count, w - (pattern_w + 1)

    patterns = _palette_rows_for_colors(idx_colors, pattern_w)
    for i, idx in enumerate(idx_colors):
        rr = r0 + i
        grid[rr, c0] = idx
        grid[rr, c0 + 1 : c0 + 1 + pattern_w] = patterns[idx]


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(16, 30)
        w = random.randint(16, 30)

        # Background color kept constant in each sample.
        bg = 0
        grid = np.full((h, w), bg, dtype=int)

        all_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.shuffle(all_colors)
        contour_color = all_colors[0]

        region_count = random.randint(1, 4)
        corner = random.choice(["tl", "tr", "bl", "br"])
        _build_palette(grid, corner, region_count, bg)

        # Occupancy for non-background pixels.
        used = np.zeros_like(grid, dtype=bool)
        if corner in ("tl", "tr"):
            pr0 = 0
            pw = 4
            ph = region_count + random.randint(0, 2)
            ph = min(max(ph, 1), h - 1)
            if corner == "tl":
                pc0 = 0
            else:
                pc0 = w - (3 + 1)
            used[0 : ph + 1, pc0 : pc0 + 1 + 4] = True
        else:
            ph = region_count + random.randint(0, 2)
            ph = min(max(ph, 1), h - 1)
            pw = 4
            if corner == "bl":
                pr0 = h - ph
                pc0 = 0
            else:
                pr0 = h - ph
                pc0 = w - (pw + 1)
            used[pr0 : pr0 + ph + 1, pc0 : pc0 + pw + 1] = True

        # Mark palette as occupied.
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg:
                    used[r, c] = True

        idx_colors = [c for c in all_colors[1:] if c != contour_color]
        random.shuffle(idx_colors)

        success = 0
        min_area = 16
        tries = 0
        while success < region_count and tries < 500:
            tries += 1
            inner_h = random.randint(4, 9)
            inner_w = random.randint(4, 9)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h >= h - 1 or outer_w >= w - 1:
                continue

            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            if ((r1 - r0 + 1) * (c1 - c0 + 1)) < min_area:
                continue

            pad = 1
            rr0 = max(0, r0 - pad)
            rr1 = min(h - 1, r1 + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(w - 1, c1 + pad)

            if used[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            # Draw a one-pixel-wide closed contour rectangle.
            _draw_rect_contour(grid, r0, c0, outer_h, outer_w, contour_color)
            used[r0 : r1 + 1, c0 : c1 + 1] = True

            # Optional tiny branch: add a one-cell inward protrusion spike, keeping closure.
            if random.random() < 0.4:
                side = random.choice(["top", "bottom", "left", "right"])
                if side == "top":
                    cc = random.randint(c0 + 2, c1 - 2)
                    rr = r0 + 1
                    grid[rr, cc] = contour_color
                    used[rr, cc] = True
                elif side == "bottom":
                    cc = random.randint(c0 + 2, c1 - 2)
                    rr = r1 - 1
                    grid[rr, cc] = contour_color
                    used[rr, cc] = True
                elif side == "left":
                    rr = random.randint(r0 + 2, r1 - 2)
                    cc = c0 + 1
                    grid[rr, cc] = contour_color
                    used[rr, cc] = True
                else:
                    rr = random.randint(r0 + 2, r1 - 2)
                    cc = c1 - 1
                    grid[rr, cc] = contour_color
                    used[rr, cc] = True

            # Place unique seed and map to a palette row color.
            if idx_colors:
                seed_color = idx_colors.pop()
            else:
                seed_color = all_colors[random.randint(1, len(all_colors) - 1)]

            sr = random.randint(r0 + 1, r1 - 1)
            sc = random.randint(c0 + 1, c1 - 1)
            if used[sr, sc] and grid[sr, sc] != contour_color:
                # Extremely unlikely with current constraints; skip and redraw.
                for rr in range(r0 + 1, r1):
                    for cc in range(c0 + 1, c1):
                        if not used[rr, cc]:
                            sr, sc = rr, cc
                            break
                    else:
                        continue
                    break
            grid[sr, sc] = seed_color
            used[sr, sc] = True
            success += 1

        if success >= region_count:
            return grid
