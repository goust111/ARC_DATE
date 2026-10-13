import numpy as np
from collections import deque


def _rotation_from_marker(local_y: int, local_x: int, h: int, w: int) -> int:
    """Map marker location in a bbox to a rotation angle.

    Returns 0, 90, 180, or -90 (counterclockwise).
    """
    if local_y * 2 < h - 1:
        return 0
    if local_y * 2 > h - 1:
        return 180
    if local_x * 2 < w - 1:
        return -90
    if local_x * 2 > w - 1:
        return 90
    return 0


def _rotate_patch(arr: np.ndarray, rot: int) -> np.ndarray:
    if rot == 0:
        return arr.copy()
    if rot == 90:
        return np.rot90(arr, -1)
    if rot == 180:
        return np.rot90(arr, 2)
    if rot == -90:
        return np.rot90(arr, 1)
    return arr.copy()


def _collect_components(grid: np.ndarray, color: int, x_min: int):
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    comps = []

    for y in range(h):
        for x in range(max(0, x_min), w):
            if visited[y, x] or grid[y, x] != color:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            coords = []
            y_min = y
            y_max = y
            x_min_c = x
            x_max_c = x

            while q:
                cy, cx = q.popleft()
                coords.append((cy, cx))
                if cy < y_min:
                    y_min = cy
                if cy > y_max:
                    y_max = cy
                if cx < x_min_c:
                    x_min_c = cx
                if cx > x_max_c:
                    x_max_c = cx

                for dy, dx in dirs:
                    ny = cy + dy
                    nx = cx + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if visited[ny, nx] or grid[ny, nx] != color:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))

            comps.append(
                {
                    "coords": coords,
                    "y0": y_min,
                    "y1": y_max,
                    "x0": x_min_c,
                    "x1": x_max_c,
                }
            )

    return comps


def _enclosed_mask(outline: np.ndarray) -> np.ndarray:
    h, w = outline.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1 : h + 1, 1 : w + 1] = outline

    outside = np.zeros_like(padded, dtype=bool)
    q = deque()
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(h + 2):
        for x in (0, w + 1):
            if not padded[y, x]:
                outside[y, x] = True
                q.append((y, x))
    for x in range(w + 2):
        for y in (0, h + 1):
            if not padded[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))

    while q:
        cy, cx = q.popleft()
        for dy, dx in dirs:
            ny = cy + dy
            nx = cx + dx
            if not (0 <= ny < h + 2 and 0 <= nx < w + 2):
                continue
            if outside[ny, nx] or padded[ny, nx]:
                continue
            outside[ny, nx] = True
            q.append((ny, nx))

    enclosed = ~outside & ~padded
    return enclosed[1 : h + 1, 1 : w + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """Map mold silhouettes to colored patches and stitch rotated pieces into a compact sheet.

    Input structure:
    - Background is black. Left part has 1-pixel-high horizontal color bands spanning the left half.
    - Right part has several closed gray outlines (3x3~6x6 bbox), each with one non-gray non-black marker pixel inside.

    Rule:
    - For every mold outline, copy its interior shape and paint interior pixels with the color of the same-band row from the left side, matching by bbox row.
    - Determine rotation from the marker: top/bottom/left/right in the bbox correspond to 0/180/ccw90/cw90.
    - Rotate each cropped patch and place all patches onto a black canvas in right-region order (top to bottom, left to right), with 1-pixel spacing and row wraps.

    Key insight:
    - The outline acts both as the patch geometry source and the container of a rotation signal, so geometry and control must be extracted in two independent steps then merged.
    """

    grid = np.asarray(grid)
    if grid.ndim != 2:
        return np.zeros((1, 1), dtype=int)

    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    h, w = grid.shape
    split = w // 2

    band_by_row = []
    for y in range(h):
        left = grid[y, :split]
        non_bg = left[left != bg]
        if non_bg.size == 0:
            band_by_row.append(bg)
            continue
        u, c = np.unique(non_bg, return_counts=True)
        band_by_row.append(int(u[np.argmax(c)]))

    # Choose the color that most looks like the mold frame color.
    right_colors = [int(c) for c in np.unique(grid[:, split:]) if c != bg]
    outline_color = None
    best_score = -1

    if right_colors:
        for color in right_colors:
            comps = _collect_components(grid, color, split)
            score = 0
            for c in comps:
                bh = c["y1"] - c["y0"] + 1
                bw = c["x1"] - c["x0"] + 1
                if 3 <= bh <= 6 and 3 <= bw <= 6 and len(c["coords"]) >= 4:
                    score += 1
            if score > best_score:
                best_score = score
                outline_color = color

    if outline_color is None:
        outline_color = right_colors[0] if right_colors else bg

    molds = []
    comps = _collect_components(grid, outline_color, split)

    for comp in comps:
        y0, y1 = comp["y0"], comp["y1"]
        x0, x1 = comp["x0"], comp["x1"]
        bh = y1 - y0 + 1
        bw = x1 - x0 + 1
        if not (3 <= bh <= 6 and 3 <= bw <= 6):
            continue

        outline_mask = grid[y0 : y1 + 1, x0 : x1 + 1] == outline_color
        interior = _enclosed_mask(outline_mask)
        inside_coords = np.argwhere(interior)
        if inside_coords.size == 0:
            continue

        slice_block = grid[y0 : y1 + 1, x0 : x1 + 1]
        marker_positions = []
        for yy, xx in inside_coords:
            c = int(slice_block[yy, xx])
            if c != bg and c != outline_color:
                marker_positions.append((yy, xx))

        if not marker_positions:
            continue

        my, mx = marker_positions[0]
        rot = _rotation_from_marker(my, mx, bh, bw)

        patch = np.full((bh, bw), bg, dtype=int)
        for yy, xx in inside_coords:
            yy_abs = y0 + yy
            patch_color = band_by_row[yy_abs]
            patch[yy, xx] = patch_color

        patch = _rotate_patch(patch, rot)
        molds.append((y0, x0, patch))

    molds.sort(key=lambda p: (p[0], p[1]))

    if not molds:
        return np.array([[bg]], dtype=int)

    placements = []
    x_cursor = 0
    y_cursor = 0
    row_height = 0
    out_w = 0
    out_h = 0
    max_w = 30

    for y0, x0, patch in molds:
        ph, pw = patch.shape
        if x_cursor > 0 and x_cursor + 1 + pw > max_w:
            y_cursor += row_height + 1
            x_cursor = 0
            row_height = 0
        placements.append((y_cursor, x_cursor, patch))
        x_cursor += pw + 1
        row_height = max(row_height, ph)
        end_x = y_cursor + 0
        out_w = max(out_w, y_cursor * 0 + x_cursor - 1)
        out_h = max(out_h, y_cursor + ph)

    out = np.full((out_h, out_w), bg, dtype=int)
    for y, x, patch in placements:
        ph, pw = patch.shape
        out[y : y + ph, x : x + pw] = patch

    return out


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    q = deque([tuple(cells[0])])
    visited[cells[0][0], cells[0][1]] = True
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    cnt = 0

    while q:
        y, x = q.popleft()
        cnt += 1
        for dy, dx in dirs:
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    return cnt == int(mask.sum())


def _build_inner_region(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    inner_h = max(1, h - 2)
    inner_w = max(1, w - 2)
    region = np.ones((inner_h, inner_w), dtype=bool)

    if inner_h <= 2 or inner_w <= 2:
        return region

    target_remove = int(rng.integers(0, max(1, inner_h * inner_w)))
    for _ in range(target_remove):
        candidates = [(yy, xx) for yy in range(1, inner_h - 1) for xx in range(1, inner_w - 1) if region[yy, xx]]
        if not candidates:
            break
        yy, xx = candidates[int(rng.integers(0, len(candidates)))]
        if np.sum(region) <= 3:
            break
        trial = region.copy()
        trial[yy, xx] = False
        if _is_connected(trial):
            region = trial
    return region


def _candidate_marker_cells(region: np.ndarray, h: int, w: int, target_rot: int):
    inner_h, inner_w = region.shape
    out = []
    for yy in range(inner_h):
        for xx in range(inner_w):
            if not region[yy, xx]:
                continue
            ly = yy + 1
            lx = xx + 1
            rot = _rotation_from_marker(ly, lx, h, w)
            if rot == target_rot:
                out.append((yy, xx))
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg = 0
    grey = 7

    height = int(rng.integers(8, 15))
    half_w = int(rng.integers(8, 13))
    width = half_w * 2
    left_w = half_w
    right_w = half_w

    band_colors = rng.choice(np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=int), size=height, replace=False)

    grid = np.full((height, width), bg, dtype=int)
    for y in range(height):
        grid[y, :left_w] = int(band_colors[y])

    # Place a random number of molds.
    max_count = min(6, right_w // 4, max(1, height // 4))
    num_molds = int(rng.integers(1, max_count + 1))

    marker_colors = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    rotations = [0, 90, 180, -90]

    occupied = np.zeros_like(grid, dtype=bool)
    for i in range(num_molds):
        rot = int(rotations[int(rng.integers(0, 4))])
        marker_color = int(marker_colors[int(rng.integers(0, len(marker_colors)))])
        placed = False

        for _ in range(120):
            h = int(rng.integers(4, 7))
            w = int(rng.integers(4, 7))
            if h > height or w > right_w:
                continue

            y0 = int(rng.integers(0, height - h + 1))
            x0 = int(rng.integers(left_w, width - w + 1))
            y1 = y0 + h
            x1 = x0 + w

            y0e = max(0, y0 - 1)
            y1e = min(height, y1 + 1)
            x0e = max(left_w, x0 - 1)
            x1e = min(width, x1 + 1)
            if occupied[y0e:y1e, x0e:x1e].any():
                continue

            region = _build_inner_region(h, w, rng)
            cand = _candidate_marker_cells(region, h, w, rot)
            if not cand:
                continue

            # Draw irregular closed contour from region.
            mold = np.full((h, w), bg, dtype=int)
            inner_h, inner_w = region.shape
            dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
            for ry in range(inner_h):
                for rx in range(inner_w):
                    if not region[ry, rx]:
                        continue
                    gy = ry + 1
                    gx = rx + 1
                    for dy, dx in dirs:
                        ny = gy + dy
                        nx = gx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if 1 <= ny < h - 1 and 1 <= nx < w - 1:
                                if not region[ny - 1, nx - 1]:
                                    mold[ny, nx] = grey
                            else:
                                mold[ny, nx] = grey

            my, mx = cand[int(rng.integers(0, len(cand)))]
            mold[my + 1, mx + 1] = marker_color

            grid[y0:y1, x0:x1] = mold
            occupied[y0:y1, x0:x1] = True
            placed = True
            break

        if not placed:
            continue

    return grid
