import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
VISIBLE_COLORS = np.array([1, 2, 3, 4, 6, 7, 8], dtype=int)


def _palette_from_right(grid):
    nonzero_cols = np.where(np.any(grid != BLACK, axis=0))[0]
    if len(nonzero_cols) == 0:
        return None, []
    palette_col = int(nonzero_cols[-1])
    palette = [int(grid[r, palette_col]) for r in range(grid.shape[0])
               if int(grid[r, palette_col]) != BLACK]
    return palette_col, palette


def _grey_components(grid, max_col):
    rows, cols = grid.shape
    max_col = max(0, min(max_col, cols))
    mask = grid[:, :max_col] == GREY
    seen = np.zeros(mask.shape, dtype=bool)
    components = []

    for r in range(rows):
        for c in range(max_col):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc),
                               (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < max_col:
                        if mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            components.append(comp)
    return components


def _is_rectangular_frame(grid, comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    perimeter = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(comp) != perimeter:
        return None
    if not np.all(grid[r0, c0:c1 + 1] == GREY):
        return None
    if not np.all(grid[r1, c0:c1 + 1] == GREY):
        return None
    if not np.all(grid[r0:r1 + 1, c0] == GREY):
        return None
    if not np.all(grid[r0:r1 + 1, c1] == GREY):
        return None
    return r0, r1, c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Repair mirror-damaged pixels inside framed patterns and recolor each frame.

    Input structure:
    The background is black. The rightmost non-black column is a palette column
    containing single color chips separated by black cells; the column just to
    its left is a black separator. To the left of that separator are one or more
    disjoint grey rectangular frames. Each frame encloses a pattern that should
    be vertically mirror-symmetric, except that some normal colored pixels have
    been replaced by maroon damage markers.

    Rule:
    For every grey rectangular frame, inspect only the cells inside the border.
    Each maroon cell is replaced by the color at its horizontally mirrored
    position across the content's vertical center axis. Count the repaired
    maroon cells in that frame. If the count is M, recolor the entire border
    with the M-th non-black palette chip from top to bottom. The palette column
    and separator column stay unchanged.

    Key insight:
    Maroon is not a regular pattern color; it is a damage marker whose repair
    is determined by bilateral symmetry, and the repair count indexes the
    visible palette order to determine the border color.
    """
    out = np.array(grid, copy=True)
    palette_col, palette = _palette_from_right(grid)
    if palette_col is None or palette_col < 2 or not palette:
        return out

    separator_col = palette_col - 1
    for comp in _grey_components(grid, separator_col):
        frame = _is_rectangular_frame(grid, comp)
        if frame is None:
            continue
        r0, r1, c0, c1 = frame
        inner = grid[r0 + 1:r1, c0 + 1:c1]
        damage_positions = np.argwhere(inner == MAROON)
        repaired = int(len(damage_positions))

        for dr, dc in damage_positions:
            rr = r0 + 1 + int(dr)
            cc = c0 + 1 + int(dc)
            mirror_cc = c0 + c1 - cc
            out[rr, cc] = grid[rr, mirror_cc]

        if 1 <= repaired <= len(palette):
            border_color = palette[repaired - 1]
            out[r0, c0:c1 + 1] = border_color
            out[r1, c0:c1 + 1] = border_color
            out[r0:r1 + 1, c0] = border_color
            out[r0:r1 + 1, c1] = border_color

    return out


def _make_symmetric_content(rng, height, width, min_pairs):
    content = np.zeros((height, width), dtype=int)
    pattern_colors = rng.choice(VISIBLE_COLORS, size=int(rng.integers(2, 5)),
                                replace=False)
    pair_slots = [(r, c) for r in range(height) for c in range(width // 2)]
    rng.shuffle(pair_slots)

    for r, c in pair_slots:
        if rng.random() < 0.68:
            color = int(rng.choice(pattern_colors))
            content[r, c] = color
            content[r, width - 1 - c] = color

    colored = [(r, c) for r, c in pair_slots if content[r, c] != BLACK]
    if len(colored) < min_pairs:
        for r, c in pair_slots:
            if content[r, c] == BLACK:
                color = int(rng.choice(pattern_colors))
                content[r, c] = color
                content[r, width - 1 - c] = color
                colored.append((r, c))
                if len(colored) >= min_pairs:
                    break

    if width % 2 == 1:
        center = width // 2
        for r in range(height):
            if rng.random() < 0.55:
                content[r, center] = int(rng.choice(pattern_colors))

    return content


def _draw_object(grid, rng, r0, c0, content_h, content_w, palette_count):
    r1 = r0 + content_h + 1
    c1 = c0 + content_w + 1
    grid[r0:r1 + 1, c0] = GREY
    grid[r0:r1 + 1, c1] = GREY
    grid[r0, c0:c1 + 1] = GREY
    grid[r1, c0:c1 + 1] = GREY

    content = _make_symmetric_content(rng, content_h, content_w, palette_count)
    candidates = [(r, c) for r in range(content_h)
                  for c in range(content_w // 2)
                  if content[r, c] != BLACK]
    damage_count = int(rng.integers(1, palette_count + 1))
    chosen = rng.choice(len(candidates), size=damage_count, replace=False)

    for idx in np.atleast_1d(chosen):
        rr, left_c = candidates[int(idx)]
        cc = left_c if rng.random() < 0.5 else content_w - 1 - left_c
        content[rr, cc] = MAROON

    grid[r0 + 1:r1, c0 + 1:c1] = content


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(80):
        height = int(rng.integers(14, 31))
        width = int(rng.integers(16, 31))
        palette_count = int(rng.integers(3, min(6, (height + 1) // 2) + 1))
        target_objects = int(rng.integers(1, 4))

        grid = np.zeros((height, width), dtype=int)
        palette_col = width - 1
        top = int(rng.integers(0, height - (2 * palette_count - 1) + 1))
        palette_colors = rng.choice(VISIBLE_COLORS, size=palette_count,
                                    replace=False)
        for i, color in enumerate(palette_colors):
            grid[top + 2 * i, palette_col] = int(color)

        main_width = width - 2
        occupied = np.zeros((height, main_width), dtype=bool)
        placed = 0

        for _attempt in range(240):
            if placed >= target_objects:
                break
            content_h = int(rng.integers(3, 8))
            content_w = int(rng.integers(4, 9))
            frame_h = content_h + 2
            frame_w = content_w + 2
            if frame_h > height or frame_w > main_width:
                continue

            r0 = int(rng.integers(0, height - frame_h + 1))
            c0 = int(rng.integers(0, main_width - frame_w + 1))
            r2 = r0 + frame_h
            c2 = c0 + frame_w
            guard_r0 = max(0, r0 - 1)
            guard_c0 = max(0, c0 - 1)
            guard_r1 = min(height, r2 + 1)
            guard_c1 = min(main_width, c2 + 1)
            if occupied[guard_r0:guard_r1, guard_c0:guard_c1].any():
                continue

            _draw_object(grid, rng, r0, c0, content_h, content_w,
                         palette_count)
            occupied[r0:r2, c0:c2] = True
            placed += 1

        if placed:
            return grid

    return generate_input()
