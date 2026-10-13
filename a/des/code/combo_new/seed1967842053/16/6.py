import numpy as np


def _most_common_color(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _row_runs(row, bg):
    runs = []
    c = 0
    while c < len(row):
        if int(row[c]) == bg:
            c += 1
            continue
        color = int(row[c])
        start = c
        while c < len(row) and int(row[c]) == color:
            c += 1
        runs.append((start, c, color))
    return runs


def _find_palette(grid, bg):
    h = grid.shape[0]
    for r in range(min(6, h)):
        runs = _row_runs(grid[r], bg)
        for i in range(max(0, len(runs) - 3)):
            colors = [runs[i + j][2] for j in range(4)]
            if len(set(colors)) == 4:
                return colors, r
    colors = []
    seen = set()
    for v in grid[0]:
        color = int(v)
        if color != bg and color not in seen:
            colors.append(color)
            seen.add(color)
        if len(colors) == 4:
            return colors, 0
    raise ValueError("could not find four color scale blocks")


def _expected_border(shape, direction):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if direction == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif direction == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif direction == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    return mask


def _interior_mask(shape, direction):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if direction == "up":
        mask[: h - 1, 1 : w - 1] = True
    elif direction == "down":
        mask[1:h, 1 : w - 1] = True
    elif direction == "left":
        mask[1 : h - 1, : w - 1] = True
    else:
        mask[1 : h - 1, 1:w] = True
    return mask


def _classify_direction(border_mask):
    best_direction = "up"
    best_score = None
    for direction in ("up", "down", "left", "right"):
        expected = _expected_border(border_mask.shape, direction)
        score = int(np.count_nonzero(border_mask ^ expected))
        if best_score is None or score < best_score:
            best_score = score
            best_direction = direction
    return best_direction


def _rotation_k(direction):
    if direction == "up":
        return 0
    if direction == "down":
        return 2
    if direction == "left":
        return 3
    return 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate every U-shaped container so its opening faces upward, then fill it from the top color scale by marker count.

    Input structure:
    The grid has a most-common background color, a four-color horizontal scale near the top, several same-colored U-shaped rectangular containers, and single-pixel interior marker dots. Each container is a one-pixel-thick three-sided slot whose open side may face up, down, left, or right.

    Rule:
    Read the four scale colors from left to right as the fill colors for one, two, three, and four marker dots. For each container, identify the missing side to get its opening direction, count the non-border marker pixels in its interior, rotate the whole cropped container so the opening faces upward, and replace all rotated interior cells with the scale color matching that count. Keep the border color and the scale bar unchanged.

    Key insight:
    The transformation separates three independent facts: the absent side gives the rotation, the interior dots give the count, and the top scale converts that count into the visible fill color.
    """
    grid = np.array(grid, copy=True)
    bg = _most_common_color(grid)
    palette, _ = _find_palette(grid, bg)

    non_palette = np.ones(grid.shape, dtype=bool)
    for color in palette:
        non_palette &= grid != color
    candidate_mask = (grid != bg) & non_palette
    colors, counts = np.unique(grid[candidate_mask], return_counts=True)
    if len(colors) == 0:
        return grid.copy()
    border = int(colors[np.argmax(counts)])

    objects = []
    for comp in _components(grid == border):
        if len(comp) < 7:
            continue
        pts = np.array(comp)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        local_border = np.zeros((h, w), dtype=bool)
        local_border[pts[:, 0] - r0, pts[:, 1] - c0] = True
        direction = _classify_direction(local_border)
        expected = _expected_border((h, w), direction)
        if np.count_nonzero(local_border ^ expected) > 1:
            continue

        local = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        inside = _interior_mask((h, w), direction)
        vals = local[inside]
        marker_count = 0
        for v in vals:
            color = int(v)
            if color != bg and color != border and color not in palette:
                marker_count += 1
        if not 1 <= marker_count <= 4:
            continue

        rotated = np.rot90(local, _rotation_k(direction)).copy()
        fill = palette[marker_count - 1]
        fill_mask = _interior_mask(rotated.shape, "up")
        rotated[fill_mask] = fill

        rr, cc = rotated.shape
        nr0 = min(r0, grid.shape[0] - rr)
        nc0 = min(c0, grid.shape[1] - cc)
        if nr0 < 0 or nc0 < 0:
            continue
        objects.append((r0, r1, c0, c1, nr0, nc0, rotated))

    out = grid.copy()
    for r0, r1, c0, c1, _, _, _ in objects:
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
    for _, _, _, _, r, c, obj in objects:
        h, w = obj.shape
        out[r : r + h, c : c + w] = obj
    return out


def _draw_container(bg, border, marker, height, width, direction, count, rng):
    local = np.full((height, width), bg, dtype=int)
    local[_expected_border((height, width), direction)] = border
    interior = np.argwhere(_interior_mask((height, width), direction))
    chosen = rng.choice(len(interior), size=count, replace=False)
    for idx in chosen:
        r, c = interior[idx]
        local[r, c] = marker
    return local


def _scale_colors(rng, bg, border, marker):
    choices = [c for c in range(10) if c not in (bg, border, marker)]
    return list(rng.choice(choices, size=4, replace=False))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    border = 5
    marker = 9
    bg = int(rng.choice([0, 1, 2, 3, 4, 6, 7, 8]))
    palette = _scale_colors(rng, bg, border, marker)

    for _ in range(200):
        height = int(rng.integers(18, 31))
        width = int(rng.integers(18, 31))
        grid = np.full((height, width), bg, dtype=int)

        block_w = int(rng.integers(2, 4))
        scale_h = int(rng.integers(1, 3))
        total_w = 4 * block_w
        start = int(rng.integers(0, width - total_w + 1))
        for i, color in enumerate(palette):
            grid[:scale_h, start + i * block_w : start + (i + 1) * block_w] = color

        occupied = np.zeros((height, width), dtype=bool)
        occupied[: scale_h + 2, :] = True
        n_objects = int(rng.integers(2, 6))
        placed = 0

        for _obj in range(n_objects):
            direction = str(rng.choice(["up", "down", "left", "right"]))
            obj_h = int(rng.integers(4, 6))
            obj_w = int(rng.integers(4, 6))
            count = int(rng.integers(1, 5))
            local = _draw_container(bg, border, marker, obj_h, obj_w, direction, count, rng)
            rotated_shape = np.rot90(local, _rotation_k(direction)).shape
            union_h = max(obj_h, rotated_shape[0])
            union_w = max(obj_w, rotated_shape[1])

            found = False
            for _try in range(200):
                if height - union_h - 1 <= scale_h + 2 or width - union_w - 1 <= 0:
                    break
                r = int(rng.integers(scale_h + 2, height - union_h))
                c = int(rng.integers(0, width - union_w))
                r0, r1 = max(0, r - 1), min(height, r + union_h + 2)
                c0, c1 = max(0, c - 1), min(width, c + union_w + 2)
                if np.any(occupied[r0:r1, c0:c1]):
                    continue
                grid[r : r + obj_h, c : c + obj_w] = local
                occupied[r0:r1, c0:c1] = True
                placed += 1
                found = True
                break
            if not found:
                break

        if placed >= 2:
            return grid

    return grid
