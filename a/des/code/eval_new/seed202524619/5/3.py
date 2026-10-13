import numpy as np
from collections import deque


def _mode_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _connected_components(mask: np.ndarray):
    """Return connected components (4-neighbor) as lists of (r, c)."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []

    for r0 in range(h):
        row = mask[r0]
        for c0 in range(w):
            if not row[c0] or visited[r0, c0]:
                continue

            q = deque([(r0, c0)])
            visited[r0, c0] = True
            comp = [(r0, c0)]

            while q:
                r, c = q.popleft()
                if r > 0:
                    nr = r - 1
                    if not visited[nr, c] and mask[nr, c]:
                        visited[nr, c] = True
                        q.append((nr, c))
                        comp.append((nr, c))
                if r + 1 < h:
                    nr = r + 1
                    if not visited[nr, c] and mask[nr, c]:
                        visited[nr, c] = True
                        q.append((nr, c))
                        comp.append((nr, c))
                if c > 0:
                    nc = c - 1
                    if not visited[r, nc] and mask[r, nc]:
                        visited[r, nc] = True
                        q.append((r, nc))
                        comp.append((r, nc))
                if c + 1 < w:
                    nc = c + 1
                    if not visited[r, nc] and mask[r, nc]:
                        visited[r, nc] = True
                        q.append((r, nc))
                        comp.append((r, nc))

            comps.append(comp)

    return comps


def _arrow_from_component(h: int, w: int, comp, color_mask: np.ndarray):
    comp_set = set(comp)
    row_map = {}
    for r, c in comp:
        row_map.setdefault(r, []).append(c)

    best = None

    for row, cols in row_map.items():
        cols = sorted(cols)
        run_start = cols[0]
        prev = cols[0]
        runs = []
        for cc in cols[1:]:
            if cc == prev + 1:
                prev = cc
            else:
                runs.append((run_start, prev))
                run_start = cc
                prev = cc
        runs.append((run_start, prev))

        if not (0 < row < h - 1):
            continue

        for run_l, run_r in runs:
            length = run_r - run_l + 1
            if length <= 0:
                continue

            # Right-pointing arrow: tip is above/below at the right of the line.
            tc = run_r + 1
            if tc < w and (row - 1, tc) in comp_set and (row + 1, tc) in comp_set:
                best = (1, length)

            # Left-pointing arrow: tip is above/below at the left of the line.
            tc = run_l - 1
            if tc >= 0 and (row - 1, tc) in comp_set and (row + 1, tc) in comp_set:
                if best is None or length > best[1]:
                    best = (-1, length)

    return best


def _find_arrow_in_band(band: np.ndarray, bg_color: int, sep_color):
    h, w = band.shape
    used_arrow = np.zeros_like(band, dtype=bool)
    arrow_info = None

    colors = set(np.unique(band).tolist())
    colors.discard(bg_color)
    if sep_color is not None:
        colors.discard(int(sep_color))

    for color in sorted(colors):
        cmask = band == color
        for comp in _connected_components(cmask):
            info = _arrow_from_component(h, w, comp, cmask)
            if info is None:
                continue
            if arrow_info is not None and info[1] <= arrow_info[0][1]:
                continue
            arrow_info = (info, comp)

    if arrow_info is None:
        return 0, 0, used_arrow

    (direction, n), comp = arrow_info
    for r, c in comp:
        used_arrow[r, c] = True

    return direction, n, used_arrow


def _band_bounds(separator_rows, h: int):
    if not separator_rows:
        return [(0, h - 1)]

    bounds = []
    prev = 0
    for sr in separator_rows:
        if prev <= sr - 1:
            bounds.append((prev, sr - 1))
        prev = sr + 1

    if prev <= h - 1:
        bounds.append((prev, h - 1))

    return bounds


def main(grid: np.ndarray) -> np.ndarray:
    """Replicate each stripe sprite along its arrow direction by an encoded distance.

    Input structure:
    - bg is the most common color.
    - Horizontal separator rows of one non-bg color split the grid into equal-height bands.
    - Each band contains one connected non-bg, non-separator sprite component and one red arrow marker.

    Rule:
    - In each band, detect the arrow direction (left or right) and line length N.
    - Keep the original sprite and generate N copies shifted horizontally in the arrow direction.
    - Copy spacing equals 2 * sprite width (one sprite-width gap), so adjacent copies are separated by the sprite width.
    - Remove arrow pixels and clip copies that go out of bounds.

    Key insight:
    - The arrow encodes both direction and count; its line length is the replication count while copy spacing is tied to sprite width.
    """
    grid = np.array(grid, dtype=np.int64).copy()
    h, w = grid.shape
    out = grid.copy()

    bg = _mode_color(grid)
    separator_rows = []
    sep_colors = []
    for r in range(h):
        row_colors = np.unique(grid[r])
        if row_colors.size == 1:
            c = int(row_colors[0])
            if c != bg:
                separator_rows.append(r)
                sep_colors.append(c)

    sep_color = None
    if sep_colors:
        vals, cnts = np.unique(sep_colors, return_counts=True)
        sep_color = int(vals[np.argmax(cnts)])

    for r0, r1 in _band_bounds(separator_rows, h):
        if r0 > r1:
            continue
        band = out[r0 : r1 + 1]

        direction, n, arrow_mask = _find_arrow_in_band(band, bg, sep_color)
        if direction == 0 or n <= 0:
            continue

        movable_mask = (band != bg)
        if sep_color is not None:
            movable_mask &= band != sep_color
        movable_mask &= ~arrow_mask
        comps = _connected_components(movable_mask)
        if not comps:
            continue

        sprite_comp = max(comps, key=len)
        rs = [p[0] for p in sprite_comp]
        cs = [p[1] for p in sprite_comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        sprite_w = max_c - min_c + 1
        if sprite_w <= 0:
            continue

        pixels = []
        for r, c in sprite_comp:
            color = band[r, c]
            pixels.append((r, c, color))

        out[r0 : r1 + 1][arrow_mask] = bg

        for step in range(1, int(n) + 1):
            dc = direction * (2 * sprite_w * step)
            for r, c, color in pixels:
                nr = r
                nc = c + dc
                if 0 <= nr < band.shape[0] and 0 <= nc < band.shape[1]:
                    out[r0 + nr, nc] = color

    return out


def _random_shape(rng: np.random.Generator, h: int, w: int, size: int):
    all_cells = [(h // 2, w // 2)]
    mask = np.zeros((h, w), dtype=bool)
    mask[all_cells[0]] = True

    frontier = [all_cells[0]]
    while len(all_cells) < size:
        base_r, base_c = frontier[rng.integers(0, len(frontier))]
        candidates = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = base_r + dr
            cc = base_c + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                candidates.append((rr, cc))

        if not candidates:
            free = np.argwhere(~mask)
            if free.size == 0:
                break
            base = free[rng.integers(0, free.shape[0])]
            all_cells.append((int(base[0]), int(base[1])))
            mask[base[0], base[1]] = True
            frontier.append((int(base[0]), int(base[1])))
            continue

        nr, nc = candidates[rng.integers(0, len(candidates))]
        all_cells.append((nr, nc))
        mask[nr, nc] = True
        frontier.append((nr, nc))

    return all_cells


def generate_input() -> np.ndarray:
    """Generate randomized ARC-compatible grids for the arrow-copying task."""
    rng = np.random.default_rng()

    band_count = int(rng.integers(1, 5))
    band_h = int(rng.integers(4, 7))
    w = int(rng.integers(16, 27))
    h = band_count * band_h + max(0, band_count - 1)

    bg = 0
    sep_color = 1
    arrow_color = 2
    palette = [3, 4, 5, 6, 7, 8, 9]

    grid = np.full((h, w), bg, dtype=np.int64)
    sprite_colors = [3, 4, 5, 6, 7, 8, 9]

    for b in range(band_count):
        top = b * (band_h + 1)
        bottom = top + band_h
        band = np.full((band_h, w), bg, dtype=np.int64)

        direction = int(rng.choice([-1, 1]))
        arrow_len = int(rng.integers(1, 5))
        arrow_row = int(rng.integers(1, band_h - 1))

        for _ in range(40):
            left_slot = int(rng.integers(0, 2))
            sprite_w = int(rng.integers(2, 6))
            sprite_h = int(rng.integers(2, min(5, band_h - 1) + 1))
            shape_size = int(rng.integers(4, min(sprite_w * sprite_h, 9) + 1))

            if left_slot:
                x_min = 0
                x_max = max(1, w // 2 - sprite_w - 1)
            else:
                x_min = max(0, w // 2)
                x_max = max(0, w - sprite_w)

            if x_max < x_min:
                continue

            if sprite_h >= band_h - 1:
                continue

            x = int(rng.integers(x_min, x_max))
            y = int(rng.integers(0, band_h - sprite_h))
            cells = _random_shape(rng, sprite_h, sprite_w, shape_size)

            c1 = int(rng.choice(sprite_colors))
            c2 = int(rng.choice([c for c in sprite_colors if c != c1]))
            for rr, cc in cells:
                band[y + rr, x + cc] = int(c1 if rng.random() < 0.75 else c2)

            used = {(rr, cc) for rr in range(band_h) for cc in range(w) if band[rr, cc] != bg}
            if (arrow_color in used) or (sep_color in used):
                continue

            if direction == 1:
                line_start_min = 2
                line_start_max = max(3, w - arrow_len - 2)
                if line_start_min > line_start_max:
                    continue
                line_start = int(rng.integers(line_start_min, line_start_max))
                if line_start + arrow_len >= w - 1:
                    continue

                for cc in range(line_start, line_start + arrow_len):
                    band[arrow_row, cc] = arrow_color
                tip = line_start + arrow_len
            else:
                line_start_min = 1
                line_start_max = max(2, w - 2)
                if line_start_min >= line_start_max:
                    continue
                line_start = int(rng.integers(line_start_min, line_start_max))
                if line_start - 1 < 0:
                    continue
                if line_start + arrow_len - 1 >= w:
                    continue

                for cc in range(line_start, line_start + arrow_len):
                    band[arrow_row, cc] = arrow_color
                tip = line_start - 1

            if not (0 <= tip < w):
                continue
            if not (0 <= arrow_row - 1 < band_h and 0 <= arrow_row + 1 < band_h):
                continue
            if band[arrow_row - 1, tip] != bg or band[arrow_row + 1, tip] != bg:
                continue

            band[arrow_row - 1, tip] = arrow_color
            band[arrow_row + 1, tip] = arrow_color
            break
        else:
            # Fallback guaranteed-safe arrow placement if loop above fails.
            direction = 1
            arrow_len = 2
            arrow_row = band_h // 2
            line_start = 6
            for cc in range(line_start, line_start + arrow_len):
                band[arrow_row, cc] = arrow_color
            band[arrow_row - 1, line_start + arrow_len] = arrow_color
            band[arrow_row + 1, line_start + arrow_len] = arrow_color

        if direction == 1:
            band[:, :] = np.where(band == arrow_color, arrow_color, band)
        
        grid[top:bottom, :] = band

        if b + 1 < band_count:
            grid[bottom, :] = sep_color

    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
