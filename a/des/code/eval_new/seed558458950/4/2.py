import random
from collections import deque

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _component_mask(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    labels = -np.ones((h, w), dtype=int)
    comps = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            cid = len(comps)
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            touches_border = False
            min_r, min_c = r, c
            max_r, max_c = r, c

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                min_r = min(min_r, cr)
                min_c = min(min_c, cc)
                max_r = max(max_r, cr)
                max_c = max(max_c, cc)

                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            for rr, cc in cells:
                labels[rr, cc] = cid

            comps.append(
                {
                    "pixels": cells,
                    "size": len(cells),
                    "touches_border": touches_border,
                    "bbox": (min_r, min_c, max_r, max_c),
                }
            )

    return labels, comps


def _is_solid_rectangle(cells: list[tuple[int, int]], h: int, w: int, min_r: int, min_c: int, max_r: int, max_c: int) -> bool:
    if not cells:
        return False
    if (max_r - min_r + 1) * (max_c - min_c + 1) != len(cells):
        return False
    if not (0 <= min_r < h and 0 <= min_c < w and 0 <= max_r < h and 0 <= max_c < w):
        return False
    return True


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            return True
    return False


def _find_marker_component(grid: np.ndarray, bg: int):
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    marker_info = None

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            min_r, min_c = r, c
            max_r, max_c = r, c

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                min_r = min(min_r, cr)
                min_c = min(min_c, cc)
                max_r = max(max_r, cr)
                max_c = max(max_c, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            bbox_h = max_r - min_r + 1
            bbox_w = max_c - min_c + 1
            area = (max_r - min_r + 1) * (max_c - min_c + 1)
            touches = _touches_border(cells, h, w)
            solid_rect = _is_solid_rectangle(cells, h, w, min_r, min_c, max_r, max_c)

            if touches and solid_rect:
                # marker is usually small; choose the smallest valid edge-rectangle.
                if marker_info is None or area < marker_info["area"]:
                    marker_info = {
                        "color": color,
                        "bbox": (min_r, min_c, max_r, max_c),
                        "height": bbox_h,
                        "width": bbox_w,
                        "area": area,
                    }

    return marker_info


def _can_escape_to_border(
    start: tuple[int, int],
    blocked: np.ndarray,
    h: int,
    w: int,
) -> bool:
    sr, sc = start
    if blocked[sr, sc]:
        return False

    q = deque([(sr, sc)])
    vis = np.zeros((h, w), dtype=bool)
    vis[sr, sc] = True

    while q:
        r, c = q.popleft()
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            return True

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc] or vis[nr, nc]:
                continue
            vis[nr, nc] = True
            q.append((nr, nc))

    return False


def _collect_ring_components(grid: np.ndarray, bg: int, marker_color: int):
    h, w = grid.shape
    mask = (grid != bg) & (grid != marker_color)
    comps = []
    visited = np.zeros_like(mask, dtype=bool)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            pixels = []
            min_r, min_c = r, c
            max_r, max_c = r, c
            colors = set()

            while q:
                cr, cc = q.popleft()
                pixels.append((cr, cc))
                color = int(grid[cr, cc])
                colors.add(color)
                min_r = min(min_r, cr)
                min_c = min(min_c, cc)
                max_r = max(max_r, cr)
                max_c = max(max_c, cc)
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            if len(colors) != 1:
                continue
            color = colors.pop()

            comps.append(
                {
                    "color": color,
                    "pixels": pixels,
                    "bbox": (min_r, min_c, max_r, max_c),
                    "touches_border": _touches_border(pixels, h, w),
                }
            )

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Transform nested loop nesting into nested rectangular frames.
Input structure:
- bg is the most common color.
- Several single-color closed 1-pixel-wide loop components are present in the interior.
- One small gray marker rectangle sits on the grid border and encodes border thickness.
- Non-bg colors are background, loops, and marker; only loops define nesting levels.
Rule:
- Find the marker rectangle; its width (x extent) is used as horizontal border thickness and its height (y extent) as vertical thickness.
- Compute each loop's nesting level by how many other loop components strictly contain it.
- Map level 1 (outermost) to level N (innermost) loop colors to concentric rectangle frames.
- Adjacent frames are separated by 1 background pixel and the innermost frame is solid-filled with its own color.
Key insight:
- The exact loop geometry is irrelevant; only the topological containment order among loop boundaries matters.
"""
    grid = np.asarray(grid)
    h, w = grid.shape

    if h == 0 or w == 0:
        return np.array([[0]], dtype=int)

    bg = _most_common_color(grid)

    marker = _find_marker_component(grid, bg)
    if marker is None:
        return grid.copy()

    marker_color = marker["color"]
    # Horizontal thickness is marker width; vertical thickness is marker height.
    frame_w = int(marker["width"])
    frame_h = int(marker["height"])
    frame_w = max(1, frame_w)
    frame_h = max(1, frame_h)

    # Extract loop components (excluding background and marker color).
    rings = _collect_ring_components(grid, bg, marker_color)
    if not rings:
        return grid.copy()

    # Compute bg component ids to select inside-adjacent bg for each loop.
    _, bg_comps = _component_mask(grid == bg)
    bg_label = np.zeros_like(grid, dtype=int)
    # Re-label once for fast lookups.
    bg_label[:] = -1
    for idx, comp in enumerate(bg_comps):
        for rr, cc in comp["pixels"]:
            bg_label[rr, cc] = idx

    ring_inside_ref = [None] * len(rings)
    for i, ring in enumerate(rings):
        refs = []
        for r, c in ring["pixels"]:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg:
                    refs.append((nr, nc))

        chosen = None
        for rr, cc in refs:
            lid = bg_label[rr, cc]
            if lid >= 0 and not bg_comps[lid]["touches_border"]:
                chosen = (rr, cc)
                break

        if chosen is None and refs:
            chosen = refs[0]

        ring_inside_ref[i] = chosen

    levels = []
    for i, ring in enumerate(rings):
        # Block only this loop to test topological containment of other ref points.
        blocked = np.zeros_like(grid, dtype=bool)
        for rr, cc in ring["pixels"]:
            blocked[rr, cc] = True

        encl_count = 0
        for j, ref in enumerate(ring_inside_ref):
            if j == i or ref is None:
                continue
            if not _can_escape_to_border(ref, blocked, h, w):
                encl_count += 1

        level = encl_count + 1
        levels.append((level, ring["color"]))

    # Build level -> color mapping; keep one color per level, prefer outer to inner order.
    levels.sort(key=lambda t: t[0])
    level_to_color = {}
    for lv, color in levels:
        if lv not in level_to_color:
            level_to_color[lv] = color

    if not level_to_color:
        return grid.copy()

    ordered_levels = sorted(level_to_color.items(), key=lambda t: t[0])

    # Start from innermost as a 1-cell solid, then expand outward.
    core = np.full((1, 1), ordered_levels[-1][1], dtype=grid.dtype)
    for idx in range(len(ordered_levels) - 2, -1, -1):
        color = ordered_levels[idx][1]
        out_h = core.shape[0] + 2 * frame_h + 2
        out_w = core.shape[1] + 2 * frame_w + 2
        out = np.full((out_h, out_w), bg, dtype=grid.dtype)

        out[:frame_h, :] = color
        out[-frame_h:, :] = color
        out[:, :frame_w] = color
        out[:, -frame_w:] = color

        rr = frame_h + 1
        cc = frame_w + 1
        out[rr : rr + core.shape[0], cc : cc + core.shape[1]] = core
        core = out

    return core


def _draw_rect_loop(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int, color: int) -> None:
    for c in range(c1, c2 + 1):
        grid[r1, c] = color
        grid[r2, c] = color
    for r in range(r1 + 1, r2):
        grid[r, c1] = color
        grid[r, c2] = color


def _place_marker(
    grid: np.ndarray,
    marker_color: int,
    marker_h: int,
    marker_w: int,
) -> None:
    h, w = grid.shape
    marker_h = max(1, min(marker_h, h))
    marker_w = max(1, min(marker_w, w))

    side = random.randint(0, 3)  # 0 top, 1 bottom, 2 left, 3 right
    if side == 0:
        row = 0
        col = random.randint(0, max(0, w - marker_w))
        col = min(col, w - marker_w)
        grid[row : row + marker_h, col : col + marker_w] = marker_color
    elif side == 1:
        row = h - marker_h
        col = random.randint(0, max(0, w - marker_w))
        col = min(col, w - marker_w)
        grid[row : row + marker_h, col : col + marker_w] = marker_color
    elif side == 2:
        col = 0
        row = random.randint(0, max(0, h - marker_h))
        row = min(row, h - marker_h)
        grid[row : row + marker_h, col : col + marker_w] = marker_color
    else:
        col = w - marker_w
        row = random.randint(0, max(0, h - marker_h))
        row = min(row, h - marker_h)
        grid[row : row + marker_h, col : col + marker_w] = marker_color


def generate_input() -> np.ndarray:
    while True:
        layers = random.randint(2, 4)

        if layers == 4:
            max_mark = 3
        elif layers == 3:
            max_mark = 4
        else:
            max_mark = 5

        marker_w = random.randint(1, max_mark)
        marker_h = random.randint(1, max_mark)
        # Keep shape varied; avoid all-equal too often.
        if random.random() < 0.5 and marker_w < max_mark:
            marker_w += 1 if random.random() < 0.5 else 0
        marker_w = max(1, min(marker_w, max_mark))
        if random.random() < 0.5 and marker_h < max_mark:
            marker_h += 1 if random.random() < 0.5 else 0
        marker_h = max(1, min(marker_h, max_mark))

        # Build nested loop sizes (outer -> inner), preserving strong gap between levels.
        min_outer = 5 + 4 * (layers - 1)
        outer_h = random.randint(min_outer, 30)
        outer_w = random.randint(min_outer, 30)

        heights = [outer_h]
        widths = [outer_w]
        for _ in range(1, layers):
            prev_h = heights[-1]
            prev_w = widths[-1]
            max_dec_h = (prev_h - 5) // 2
            max_dec_w = (prev_w - 5) // 2
            max_dec_h = max(2, min(4, max_dec_h))
            max_dec_w = max(2, min(4, max_dec_w))
            dec_h = random.randint(2, max_dec_h)
            dec_w = random.randint(2, max_dec_w)
            heights.append(prev_h - 2 * dec_h)
            widths.append(prev_w - 2 * dec_w)

        if min(heights[-1], widths[-1]) < 5:
            continue

        outer_top = random.randint(1, 3)
        outer_left = random.randint(1, 3)
        extra_bottom = random.randint(2, 4)
        extra_right = random.randint(2, 4)

        H = outer_top + outer_h + extra_bottom
        W = outer_left + outer_w + extra_right
        if H > 30 or W > 30:
            continue

        # Marker must be on edge, and loops remain inward.
        if H < marker_h + 2 or W < marker_w + 2:
            continue

        # Keep loops clearly away from border.
        if outer_top <= marker_h and outer_left <= marker_w:
            pass

        palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if 0 in palette:
            palette.remove(0)
        marker_color = 9
        colors = [c for c in palette if c != marker_color]
        if len(colors) < layers:
            continue

        random.shuffle(colors)
        layer_colors = colors[:layers]

        # Construct concentric loop sizes with possibly small geometric perturbation.
        g = np.full((H, W), 0, dtype=np.int64)

        for idx in range(layers):
            i = idx
            h_i = heights[i]
            w_i = widths[i]
            top = outer_top + (outer_h - h_i) // 2
            left = outer_left + (outer_w - w_i) // 2
            bottom = top + h_i - 1
            right = left + w_i - 1
            if top < 0 or left < 0 or bottom >= H or right >= W:
                continue

            color = layer_colors[idx]
            _draw_rect_loop(g, top, left, bottom, right, color)

            # Irregularity branches: inward spike not crossing neighboring ring.
            if random.random() < 0.5 and h_i > 9 and w_i > 9:
                for _ in range(random.randint(1, 2)):
                    side = random.choice("TBLR")
                    if side == "T":
                        c = random.randint(left + 1, right - 1)
                        depth = random.randint(1, 2)
                        for d in range(1, depth + 1):
                            if top + d <= bottom - 1:
                                g[top + d, c] = color
                    elif side == "B":
                        c = random.randint(left + 1, right - 1)
                        depth = random.randint(1, 2)
                        for d in range(1, depth + 1):
                            if bottom - d >= top + 1:
                                g[bottom - d, c] = color
                    elif side == "L":
                        r = random.randint(top + 1, bottom - 1)
                        depth = random.randint(1, 2)
                        for d in range(1, depth + 1):
                            if left + d <= right - 1:
                                g[r, left + d] = color
                    else:
                        r = random.randint(top + 1, bottom - 1)
                        depth = random.randint(1, 2)
                        for d in range(1, depth + 1):
                            if right - d >= left + 1:
                                g[r, right - d] = color

        _place_marker(g, marker_color, marker_h, marker_w)

        # Ensure marker color does not collide with ring colors.
        if marker_color in layer_colors:
            continue

        # Validate non-bg component chain quickly for generator stability.
        # Each ring must be at least one-cell thick and not touching border.
        ring_pixels = g != 0
        ring_pixels = ring_pixels & (g != marker_color)
        comps = _component_mask(ring_pixels)[1]
        if len(comps) != layers:
            continue

        # Keep enough margin so detector can always find enclosed bg.
        valid = True
        for comp in comps:
            touches = False
            for rr, cc in comp["pixels"]:
                if rr == 0 or cc == 0 or rr == H - 1 or cc == W - 1:
                    touches = True
                    break
            if touches:
                valid = False
                break
        if not valid:
            continue

        return g.astype(np.int64)
