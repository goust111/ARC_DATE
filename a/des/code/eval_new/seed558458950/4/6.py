import numpy as np
from collections import deque


def _all_components(grid):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = {}
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []

            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.setdefault(color, []).append(comp)

    return comps


def _normalize(shape):
    if not shape:
        return frozenset()
    rows = [p[0] for p in shape]
    cols = [p[1] for p in shape]
    min_r = min(rows)
    min_c = min(cols)
    return frozenset((r - min_r, c - min_c) for r, c in shape)


def _shape_bbox(shape):
    if not shape:
        return 0, 0, 0, 0
    rs = [r for r, _ in shape]
    cs = [c for _, c in shape]
    return min(rs), max(rs), min(cs), max(cs)


def _rotate_shape(shape, k):
    k = k % 4
    pts = list(shape)
    if k == 0:
        rot = pts
    elif k == 1:
        rot = [(c, -r) for r, c in pts]
    elif k == 2:
        rot = [(-r, -c) for r, c in pts]
    else:
        rot = [(-c, r) for r, c in pts]
    return _normalize(rot)


def _shape_variants(shape):
    out = []
    seen = set()
    current = shape
    for k in range(4):
        current = _rotate_shape(shape, k)
        if current not in seen:
            seen.add(current)
            out.append(current)
    return out


def _detect_containers(grid, bg):
    comps = _all_components(grid)
    containers = []

    for color, color_comps in comps.items():
        if color == bg:
            continue
        for comp in color_comps:
            if len(comp) < 8:
                continue
            r_set = set(comp)
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue

            on_perimeter = True
            for r, c in comp:
                if r0 < r < r1 and c0 < c < c1:
                    on_perimeter = False
                    break
            if not on_perimeter:
                continue

            gap_side = None
            gap_indices = None
            gap_abs = None
            valid = True

            for side in ("top", "bottom", "left", "right"):
                if side == "top":
                    inds = [c for c in range(c0, c1 + 1) if (r0, c) not in r_set]
                elif side == "bottom":
                    inds = [c for c in range(c0, c1 + 1) if (r1, c) not in r_set]
                elif side == "left":
                    inds = [r for r in range(r0, r1 + 1) if (r, c0) not in r_set]
                else:
                    inds = [r for r in range(r0, r1 + 1) if (r, c1) not in r_set]

                if len(inds) == 0:
                    continue
                if len(inds) > 3:
                    valid = False
                    break
                if inds[0] + len(inds) - 1 != inds[-1]:
                    valid = False
                    break

                if gap_side is not None:
                    valid = False
                    break

                gap_side = side
                gap_indices = inds
                if side in ("top", "bottom"):
                    gap_abs = [(r0, c) for c in inds]
                else:
                    gap_abs = [(r, c0) for r in inds] if side == "left" else [(r, c1) for r in inds]

            if not valid or gap_side is None:
                continue

            containers.append({
                "color": color,
                "bbox": (r0, r1, c0, c1),
                "gap_side": gap_side,
                "gap_indices": gap_indices,
                "gap_cells": gap_abs,
            })

    return containers


def _extract_keyhole(grid, bg, container):
    r0, r1, c0, c1 = container["bbox"]
    side = container["gap_side"]
    gap_indices = container["gap_indices"]

    inside_r0, inside_r1 = r0 + 1, r1 - 1
    inside_c0, inside_c1 = c0 + 1, c1 - 1
    if inside_r0 > inside_r1 or inside_c0 > inside_c1:
        return frozenset()

    if side == "top":
        seeds = [(inside_r0, c) for c in gap_indices]
        drs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    elif side == "bottom":
        seeds = [(inside_r1, c) for c in gap_indices]
        drs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    elif side == "left":
        seeds = [(r, inside_c0) for r in gap_indices]
        drs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        seeds = [(r, inside_c1) for r in gap_indices]
        drs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    h, w = grid.shape
    q = deque()
    seen = set()
    for sr, sc in seeds:
        if inside_r0 <= sr <= inside_r1 and inside_c0 <= sc <= inside_c1 and int(grid[sr, sc]) == bg:
            q.append((sr, sc))
            seen.add((sr, sc))

    while q:
        r, c = q.popleft()
        for dr, dc in drs:
            nr, nc = r + dr, c + dc
            if nr < inside_r0 or nr > inside_r1 or nc < inside_c0 or nc > inside_c1:
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) != bg:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))

    return _normalize(seen)


def _count_ticks(grid, container, bg):
    r0, r1, c0, c1 = container["bbox"]
    gap = container["gap_indices"]
    h, w = grid.shape
    side = container["gap_side"]

    if side == "top":
        line_r = r0 - 1
        if line_r < 0:
            return 1
        scan_coords = [(line_r, c) for c in range(c0, c1 + 1)]
        anchor_start, anchor_end = min(gap), max(gap)
    elif side == "bottom":
        line_r = r1 + 1
        if line_r >= h:
            return 1
        scan_coords = [(line_r, c) for c in range(c0, c1 + 1)]
        anchor_start, anchor_end = min(gap), max(gap)
    elif side == "left":
        line_c = c0 - 1
        if line_c < 0:
            return 1
        scan_coords = [(r, line_c) for r in range(r0, r1 + 1)]
        anchor_start, anchor_end = min(gap), max(gap)
    else:
        line_c = c1 + 1
        if line_c >= w:
            return 1
        scan_coords = [(r, line_c) for r in range(r0, r1 + 1)]
        anchor_start, anchor_end = min(gap), max(gap)

    runs = []
    i = 0
    while i < len(scan_coords):
        r, c = scan_coords[i]
        color = int(grid[r, c])
        if color == bg:
            i += 1
            continue
        j = i + 1
        while j < len(scan_coords):
            rr, cc = scan_coords[j]
            if int(grid[rr, cc]) != color:
                break
            j += 1
        length = j - i
        if 1 <= length <= 4:
            pos_start = scan_coords[i][1] if side in ("top", "bottom") else scan_coords[i][0]
            pos_end = scan_coords[j - 1][1] if side in ("top", "bottom") else scan_coords[j - 1][0]
            runs.append((color, pos_start, pos_end, length))
        i = j

    if not runs:
        return 1

    best = None
    best_key = None
    for color, pos_start, pos_end, length in runs:
        if pos_end < anchor_start:
            dist = anchor_start - pos_end
        elif pos_start > anchor_end:
            dist = pos_start - anchor_end
        else:
            dist = 0

        key = (dist, -length)
        if best is None or key < best:
            best = key
            best_key = length

    return max(1, min(4, best_key))


def _candidate_keys(grid, bg, frame_color):
    comps = _all_components(grid)
    candidates = []
    for color, color_comps in comps.items():
        if color in (bg, frame_color):
            continue
        for comp in color_comps:
            if 3 <= len(comp) <= 8:
                candidates.append({"color": color, "coords": comp})
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Insert each matched key into its matching keyhole, shifted by tick depth.

    Input structure:
    Background is the most frequent color. Large one-pixel-thick gray frames define containers. Each container has one gap of 1-3 background pixels on one frame side, with background connected inside as its keyhole. A short run of marker pixels sits on the exterior side-adjacent to that side to encode insertion depth (1-4). Candidate keys are small connected monochrome components placed outside the containers.

    Rule:
    For each detected container, derive the keyhole shape by flood-filling background cells connected from the gap inward. Find the key whose shape matches the keyhole under 0/90/180/270 rotation. Read the gap side to get insertion direction and count the adjacent marker run N on that same side. Place the matched key in its matched orientation, shifted by N-1 pixels into the container along the insertion axis from its normalized keyhole position, clear the original key pixels to background, and keep frames and markers unchanged.

    Key insight:
    Identify and match shapes first (positive key shape vs negative keyhole shape), then use side-specific marker count to determine a deterministic orthogonal insertion depth.
    """
    g = np.asarray(grid)
    if g.ndim != 2:
        return g.copy()

    g = g.astype(int)
    out = g.copy()
    bg = int(np.bincount(g.ravel()).argmax())

    containers = _detect_containers(g, bg)
    if not containers:
        return out

    comp_index = {}
    for color, color_comps in _all_components(g).items():
        comp_index.setdefault(color, []).extend(color_comps)

    frame_color = containers[0]["color"]
    keys = _candidate_keys(g, bg, frame_color)
    used = [False] * len(keys)

    for container in containers:
        hole = _extract_keyhole(g, bg, container)
        if not hole:
            continue

        hole_min = (min(r for r, _ in hole), min(c for _, c in hole))
        if hole_min[0] is None:
            continue
        tick_n = _count_ticks(g, container, bg)
        side = container["gap_side"]
        if side == "top":
            dir_r, dir_c = 1, 0
        elif side == "bottom":
            dir_r, dir_c = -1, 0
        elif side == "left":
            dir_r, dir_c = 0, 1
        else:
            dir_r, dir_c = 0, -1

        matched = None
        for idx, key in enumerate(keys):
            if used[idx]:
                continue
            norm_key = _normalize(key["coords"])
            for rot in _shape_variants(norm_key):
                if rot == hole:
                    matched = (idx, rot)
                    break
            if matched is not None:
                break

        if matched is None:
            continue

        kidx, rot_shape = matched
        used[kidx] = True

        for r, c in keys[kidx]["coords"]:
            out[r, c] = bg

        shift_r = hole_min[0] + (tick_n - 1) * dir_r
        shift_c = hole_min[1] + (tick_n - 1) * dir_c
        for rr, cc in rot_shape:
            nr = shift_r + rr
            nc = shift_c + cc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = keys[kidx]["color"]

    return out


def _random_polyomino_3x3(rng, size=None):
    if size is None:
        size = int(rng.integers(3, 9))

    cells = {(1, 1)}
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while len(cells) < size:
        r, c = rng.choice(tuple(cells))
        dr, dc = moves[int(rng.integers(0, 4))]
        nr, nc = r + dr, c + dc
        if 0 <= nr <= 2 and 0 <= nc <= 2:
            cells.add((nr, nc))
    return list(cells)


def _place_shape(grid, occupied, shape, color, x_min, y_min, x_max, y_max, rng):
    h, w = grid.shape
    if x_min > x_max or y_min > y_max:
        return False

    shape = _normalize(shape)
    rs = [p[0] for p in shape]
    cs = [p[1] for p in shape]
    sh = max(rs) + 1
    sw = max(cs) + 1

    for _ in range(400):
        r0 = int(rng.integers(y_min, y_max - sh + 2))
        c0 = int(rng.integers(x_min, x_max - sw + 2))
        ok = True
        for dr, dc in shape:
            r = r0 + dr
            c = c0 + dc
            if occupied[r, c] or int(grid[r, c]) != 0:
                ok = False
                break
        if not ok:
            continue

        for dr, dc in shape:
            r = r0 + dr
            c = c0 + dc
            occupied[r, c] = True
            grid[r, c] = color
        return True

    return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    bg = 0
    frame = 7
    mark = 9
    palette = [1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15]
    rng.shuffle(palette)
    palette_ptr = 0

    h = int(rng.integers(16, 28))
    w = int(rng.integers(22, 30))
    grid = np.full((h, w), bg, dtype=int)

    num_containers = int(rng.integers(1, 4))

    occupied = np.zeros((h, w), dtype=bool)

    containers = []
    container_specs = []
    used_shapes = set()
    container_band_end = max(5, w // 2 - 3)

    for idx in range(num_containers):
        placed = False
        side = int(rng.integers(4))
        gap_side = ("top", "right", "bottom", "left")[side]

        for _ in range(350):
            cw = int(rng.integers(6, 10))
            ch = int(rng.integers(6, 10))

            y0_min = 1
            y0_max = h - ch - 2
            x0_max = min(w - cw - 2, container_band_end - cw + 1)
            x0_min = 1

            if gap_side == "top":
                y0_min = 1
            elif gap_side == "bottom":
                y0_max = h - ch - 2
            elif gap_side == "left":
                x0_min = 1
            else:
                x0_max = min(x0_max, w - cw - 2)

            if x0_max < x0_min or y0_max < y0_min:
                # relax constraints and retry in another attempt
                x0_min = 1
                x0_max = w - cw - 2
                if gap_side == "top":
                    y0_min = 1
                elif gap_side == "bottom":
                    y0_max = h - ch - 2
                else:
                    y0_max = h - ch - 2

            if x0_max < x0_min or y0_max < y0_min:
                continue

            r0 = int(rng.integers(y0_min, y0_max + 1))
            c0 = int(rng.integers(x0_min, x0_max + 1))
            r1, c1 = r0 + ch - 1, c0 + cw - 1

            if gap_side == "left" and c0 == 0:
                continue
            if gap_side == "top" and r0 == 0:
                continue
            if gap_side == "bottom" and r1 + 1 >= h:
                continue
            if gap_side == "right" and c1 + 1 >= w:
                continue

            if np.any(occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]):
                continue

            # Build a random cavity shape that is always <=3x3 so gap width is naturally <=3.
            base = _random_polyomino_3x3(rng)
            base_norm = _normalize(base)
            shape_id = min(_shape_variants(base_norm))
            if shape_id in used_shapes:
                continue
            used_shapes.add(shape_id)

            hole_rot = int(rng.integers(0, 4))
            hole_shape = _rotate_shape(base_norm, hole_rot)
            if not hole_shape:
                continue

            r_min, r_max, c_min, c_max = _shape_bbox(hole_shape)
            shape_h = r_max - r_min + 1
            shape_w = c_max - c_min + 1
            ih, iw = ch - 2, cw - 2
            if shape_h > ih or shape_w > iw:
                continue

            if gap_side == "top":
                dr0 = 0
                dc0 = rng.integers(0, iw - shape_w + 1)
            elif gap_side == "bottom":
                dr0 = ih - shape_h
                dc0 = rng.integers(0, iw - shape_w + 1)
            elif gap_side == "left":
                dr0 = rng.integers(0, ih - shape_h + 1)
                dc0 = 0
            else:
                dr0 = rng.integers(0, ih - shape_h + 1)
                dc0 = iw - shape_w

            interior_cells = [(r0 + 1 + dr0 + r, c0 + 1 + dc0 + c) for r, c in hole_shape]

            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in interior_cells):
                continue
            if any(occupied[rr, cc] for rr, cc in interior_cells):
                continue

            # Determine gap cells from hole mouth cells on the border side.
            if gap_side == "top":
                gap_abs = sorted({cc for rr, cc in interior_cells if rr == r0 + 1})
            elif gap_side == "bottom":
                gap_abs = sorted({cc for rr, cc in interior_cells if rr == r1 - 1})
            elif gap_side == "left":
                gap_abs = sorted({rr for rr, cc in interior_cells if cc == c0 + 1})
            else:
                gap_abs = sorted({rr for rr, cc in interior_cells if cc == c1 - 1})

            if not gap_abs:
                continue
            if len(gap_abs) > 3:
                continue

            # Draw frame.
            for c in range(c0, c1 + 1):
                if not (gap_side == "top" and c in gap_abs):
                    grid[r0, c] = frame
                if not (gap_side == "bottom" and c in gap_abs):
                    grid[r1, c] = frame
            for r in range(r0, r1 + 1):
                if not (gap_side == "left" and r in gap_abs):
                    grid[r, c0] = frame
                if not (gap_side == "right" and r in gap_abs):
                    grid[r, c1] = frame

            interior_fill = 2
            grid[r0 + 1 : r1, c0 + 1 : c1] = interior_fill
            for rr, cc in interior_cells:
                grid[rr, cc] = bg

            # Tick marks on outer side; keep 1..4.
            tick_n = int(rng.integers(1, 5))
            if gap_side == "top":
                row = r0 - 1
                tstart = max(c0 + 1, min(gap_abs[0], c1 - tick_n + 1))
                for cc in range(tstart, min(tstart + tick_n, c1)):
                    grid[row, cc] = mark
            elif gap_side == "bottom":
                row = r1 + 1
                tstart = max(c0 + 1, min(gap_abs[0], c1 - tick_n + 1))
                for cc in range(tstart, min(tstart + tick_n, c1)):
                    grid[row, cc] = mark
            elif gap_side == "left":
                col = c0 - 1
                tstart = max(r0 + 1, min(gap_abs[0], r1 - tick_n + 1))
                for rr in range(tstart, min(tstart + tick_n, r1)):
                    grid[rr, col] = mark
            else:
                col = c1 + 1
                tstart = max(r0 + 1, min(gap_abs[0], r1 - tick_n + 1))
                for rr in range(tstart, min(tstart + tick_n, r1)):
                    grid[rr, col] = mark

            # mark occupancy (only structural cells so key area stays available).
            occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True

            key_rot = int(rng.integers(0, 4))
            key_shape = _rotate_shape(base_norm, key_rot)

            key_color = palette[palette_ptr % len(palette)]
            palette_ptr += 1

            containers.append({
                "bbox": (r0, r1, c0, c1),
                "key_shape": key_shape,
                "key_color": key_color,
                "gap": gap_abs,
            })
            container_specs.append((key_shape, key_color))
            placed = True
            break

        if not placed:
            # Fallback weak placement: keep generating smaller container to avoid deadlock.
            cw = 6
            ch = 6
            c0 = 1 + idx * 2
            r0 = 1 + idx * 2
            r1, c1 = min(h - 2, r0 + ch - 1), min(w - 2, c0 + cw - 1)
            for c in range(c0, c1 + 1):
                grid[r0, c] = frame
                grid[r1, c] = frame
            for r in range(r0, r1 + 1):
                grid[r, c0] = frame
                grid[r, c1] = frame
            occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True

    if not containers:
        # deterministic minimal fallback
        c0, r0 = 5, 5
        cw, ch = 8, 8
        r1, c1 = r0 + ch - 1, c0 + cw - 1
        for c in range(c0, c1 + 1):
            grid[r0, c] = frame
            grid[r1, c] = frame
        for r in range(r0, r1 + 1):
            grid[r, c0] = frame
            grid[r, c1] = frame
        gap_side = "top"
        for c in (c0 + 2, c0 + 3):
            grid[r0, c] = bg
            grid[r0 - 1, c] = mark
        key_shape = frozenset({(0, 0), (1, 0), (2, 0)})
        key_color = palette[0]
        container_specs.append((key_shape, key_color))
        containers.append({"bbox": (r0, r1, c0, c1)})

    # Place matching keys and a few distractors on the right side.
    right_band = max(w // 2, max(c["bbox"][3] for c in containers) + 3)
    if right_band >= w - 4:
        right_band = max(1, w // 2)

    for key_shape, key_color in container_specs:
        _place_shape(
            grid,
            occupied,
            key_shape,
            key_color,
            max(1, right_band),
            1,
            w - 2,
            h - 2,
            rng,
        )

    extra_keys = int(rng.integers(0, 3))
    for _ in range(extra_keys):
        for _ in range(40):
            size = int(rng.integers(3, 9))
            base = _random_polyomino_3x3(rng, size=size)
            base = _normalize(base)
            shape_id = min(_shape_variants(base))
            if shape_id in used_shapes:
                continue
            used_shapes.add(shape_id)
            rot = int(rng.integers(0, 4))
            shape = _rotate_shape(base, rot)
            key_color = palette[palette_ptr % len(palette)]
            palette_ptr += 1
            placed = _place_shape(
                grid,
                occupied,
                shape,
                key_color,
                max(1, right_band),
                1,
                w - 2,
                h - 2,
                rng,
            )
            if placed:
                break

    return grid
