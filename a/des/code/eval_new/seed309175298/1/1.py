from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not local[0, :].all() or not local[-1, :].all():
        return False
    if not local[:, 0].all() or not local[:, -1].all():
        return False
    if local[1:-1, 1:-1].any():
        return False
    return True


def _parse_legend(row: np.ndarray, bg: int) -> dict[int, bool]:
    mapping: dict[int, bool] = {}
    c = 0
    width = int(row.shape[0])
    while c <= width - 3:
        left = int(row[c])
        mid = int(row[c + 1])
        right = int(row[c + 2])
        if left != bg and mid == bg and right != bg and left != right:
            mapping[left] = True
            mapping[right] = False
            c += 3
        else:
            c += 1
    assert mapping, "legend row must contain at least one [color,bg,color] triplet"
    return mapping


def _rotate_mask(mask: np.ndarray, clockwise: bool) -> np.ndarray:
    h, w = mask.shape
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        dr = float(r) - center_r
        dc = float(c) - center_c
        if clockwise:
            nr = center_r + dc
            nc = center_c - dr
        else:
            nr = center_r - dc
            nc = center_c + dr
        rr = int(round(nr))
        cc = int(round(nc))
        if abs(nr - rr) > 1e-6 or abs(nc - cc) > 1e-6:
            continue
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine each frame's rotation from the top legend, rotate the enclosed sprite in place, and remove the legend band.

    Input structure:
    - bg is the most common color in the full grid.
    - Row 0 is a legend row made of 1x3 triplets `[left_color, bg, right_color]`, and row 1 is an all-bg spacer row.
    - Below the spacer are several single-color closed rectangular frames with distinct border colors.
    - Each frame encloses an asymmetric sprite drawn in one shared sprite color on the bg interior.

    Rule:
    - Read the legend as a color-to-direction lookup: a color on the left means rotate clockwise by 90 degrees, and a color on the right means rotate counterclockwise by 90 degrees.
    - For every frame, use its border color to choose the direction, clear the frame interior to bg, and redraw the interior sprite rotated 90 degrees about the interior region's geometric center.
    - Keep every frame border unchanged and remove the top legend row together with the blank spacer row.

    Key insight:
    - The border color is only a lookup key; the real encoded information is whether that color sits on the left or the right side of a 1x3 legend cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3
    bg = _get_bg(g)
    legend = _parse_legend(g[0], bg)
    assert np.all(g[1] == bg), "row 1 must be the blank separator row"

    out = g[2:].copy()
    colors = [int(c) for c in np.unique(out) if int(c) != bg and int(c) in legend]
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        for comp in _components_mask(out == int(color)):
            if _is_rectangular_frame(comp):
                frames.append((int(color), _bbox(comp)))

    assert frames, "expected at least one rectangular frame in the main region"
    for frame_color, (r0, c0, r1, c1) in frames:
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        vals = [int(v) for v in np.unique(inner) if int(v) != bg]
        assert len(vals) == 1, "each frame interior must contain exactly one sprite color"
        sprite_color = vals[0]
        rotated = _rotate_mask(inner == sprite_color, clockwise=legend[frame_color])
        inner[:] = bg
        inner[rotated] = sprite_color

    return out


# --------------- generator (output-first, self-verified) ---------------

COLOR_POOL = tuple(range(10))
FRAME_COUNT_OPTIONS = (2, 3, 4)
INNER_SIZE_RANGE = (4, 7)
AREA_RANGE = (3, 6)
MAX_MAIN_SIDE = 28
MAX_ATTEMPTS = 400
LAYOUTS = {
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
    ),
}


def _split_units(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(int(total) + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [int(total) + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _sample_directions(count: int) -> list[bool]:
    cw_count = count // 2
    if count % 2 == 1 and random.random() < 0.5:
        cw_count += 1
    dirs = [True] * cw_count + [False] * (count - cw_count)
    random.shuffle(dirs)
    return dirs


def _sample_sprite_mask(size: int) -> np.ndarray:
    cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    max_area = min(AREA_RANGE[1], len(cells))
    min_area = min(AREA_RANGE[0], max_area)
    assert min_area >= 3
    for _ in range(200):
        target = random.randint(min_area, max_area)
        start = random.choice(cells)
        chosen = {start}
        while len(chosen) < target:
            frontier = []
            for r, c in chosen:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in chosen:
                        continue
                    if 1 <= nr < size - 1 and 1 <= nc < size - 1:
                        frontier.append((nr, nc))
            if not frontier:
                break
            chosen.add(random.choice(frontier))
        if len(chosen) != target:
            continue
        mask = np.zeros((size, size), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        rows, cols = np.where(mask)
        if len(set(map(int, rows))) < 2 or len(set(map(int, cols))) < 2:
            continue
        if np.array_equal(mask, _rotate_mask(mask, True)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric sprite mask")


def _plan_layout(outer_sizes: list[int], legend_width: int) -> tuple[int, int, list[tuple[int, int]]]:
    layout = list(random.choice(LAYOUTS[len(outer_sizes)]))
    random.shuffle(layout)
    row_count = 1 + max(slot[0] for slot in layout)
    col_count = 1 + max(slot[1] for slot in layout)

    row_heights = [0] * row_count
    col_widths = [0] * col_count
    for size, (rr, cc) in zip(outer_sizes, layout, strict=False):
        row_heights[rr] = max(row_heights[rr], int(size))
        col_widths[cc] = max(col_widths[cc], int(size))

    row_gaps = [random.randint(1, 3) for _ in range(row_count - 1)]
    col_gaps = [random.randint(1, 3) for _ in range(col_count - 1)]
    top = random.randint(0, 2)
    bottom = random.randint(0, 2)
    left = random.randint(0, 2)
    right = random.randint(0, 2)

    need_h = top + sum(row_heights) + sum(row_gaps) + bottom
    need_w = left + sum(col_widths) + sum(col_gaps) + right
    if need_h > MAX_MAIN_SIDE:
        raise ValueError("layout too tall")

    width_floor = max(need_w, legend_width)
    if width_floor > MAX_MAIN_SIDE:
        raise ValueError("layout too wide")

    extra_h = random.randint(0, MAX_MAIN_SIDE - need_h)
    extra_w = random.randint(0, MAX_MAIN_SIDE - width_floor)
    grow_h = _split_units(extra_h, len(row_gaps) + 2)
    grow_w = _split_units((width_floor - need_w) + extra_w, len(col_gaps) + 2)
    top += grow_h[0]
    bottom += grow_h[-1]
    left += grow_w[0]
    right += grow_w[-1]
    for i in range(len(row_gaps)):
        row_gaps[i] += grow_h[i + 1]
    for i in range(len(col_gaps)):
        col_gaps[i] += grow_w[i + 1]

    h = top + sum(row_heights) + sum(row_gaps) + bottom
    w = left + sum(col_widths) + sum(col_gaps) + right
    assert h <= MAX_MAIN_SIDE and w <= MAX_MAIN_SIDE

    row_starts = [top]
    for i in range(1, row_count):
        row_starts.append(row_starts[-1] + row_heights[i - 1] + row_gaps[i - 1])
    col_starts = [left]
    for i in range(1, col_count):
        col_starts.append(col_starts[-1] + col_widths[i - 1] + col_gaps[i - 1])

    positions: list[tuple[int, int]] = []
    for size, (rr, cc) in zip(outer_sizes, layout, strict=False):
        spare_r = row_heights[rr] - int(size)
        spare_c = col_widths[cc] - int(size)
        offset_r = random.randint(0, spare_r)
        offset_c = random.randint(0, spare_c)
        positions.append((row_starts[rr] + offset_r, col_starts[cc] + offset_c))
    return h, w, positions


def _render_expected(
    main_grid: np.ndarray,
    frames: list[dict[str, object]],
    sprite_color: int,
) -> np.ndarray:
    expected = main_grid.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        inner_size = int(frame["inner_size"])
        mask = np.array(frame["mask"], dtype=bool)
        clockwise = bool(frame["clockwise"])
        rotated = _rotate_mask(mask, clockwise=clockwise)
        inner = expected[r0 + 1 : r0 + 1 + inner_size, c0 + 1 : c0 + 1 + inner_size]
        inner[:] = _get_bg(main_grid)
        inner[rotated] = int(sprite_color)
    return expected


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    directions = _sample_directions(frame_count)
    cw_used = sum(directions)
    ccw_used = frame_count - cw_used
    filler_cw = max(0, ccw_used - cw_used)
    filler_ccw = max(0, cw_used - ccw_used)

    needed_colors = 2 + frame_count + filler_cw + filler_ccw
    if needed_colors > len(COLOR_POOL):
        return None

    bg = random.choice(COLOR_POOL)
    remaining = [c for c in COLOR_POOL if c != bg]
    chosen = random.sample(remaining, needed_colors - 1)
    sprite_color = chosen[0]
    frame_colors = chosen[1 : 1 + frame_count]
    filler_colors = chosen[1 + frame_count :]
    filler_cw_colors = filler_colors[:filler_cw]
    filler_ccw_colors = filler_colors[filler_cw:]

    inner_sizes = [random.randint(*INNER_SIZE_RANGE) for _ in range(frame_count)]
    outer_sizes = [size + 2 for size in inner_sizes]

    pair_count = max(cw_used + filler_cw, ccw_used + filler_ccw)
    legend_width = 3 * pair_count
    try:
        h, w, positions = _plan_layout(outer_sizes, legend_width)
    except ValueError:
        return None

    main_grid = np.full((h, w), int(bg), dtype=int)
    frames: list[dict[str, object]] = []
    for frame_color, inner_size, outer_size, clockwise, (r0, c0) in zip(
        frame_colors, inner_sizes, outer_sizes, directions, positions, strict=False
    ):
        mask = _sample_sprite_mask(inner_size)
        r1 = r0 + outer_size - 1
        c1 = c0 + outer_size - 1
        main_grid[r0, c0 : c1 + 1] = int(frame_color)
        main_grid[r1, c0 : c1 + 1] = int(frame_color)
        main_grid[r0 : r1 + 1, c0] = int(frame_color)
        main_grid[r0 : r1 + 1, c1] = int(frame_color)
        inner = main_grid[r0 + 1 : r1, c0 + 1 : c1]
        inner[mask] = int(sprite_color)
        frames.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "inner_size": int(inner_size),
                "mask": mask,
                "clockwise": bool(clockwise),
            }
        )

    legend_row = np.full((w,), int(bg), dtype=int)
    cw_colors = [int(color) for color, flag in zip(frame_colors, directions, strict=False) if flag] + [
        int(c) for c in filler_cw_colors
    ]
    ccw_colors = [int(color) for color, flag in zip(frame_colors, directions, strict=False) if not flag] + [
        int(c) for c in filler_ccw_colors
    ]
    random.shuffle(cw_colors)
    random.shuffle(ccw_colors)
    assert len(cw_colors) == len(ccw_colors) == pair_count
    start = random.randint(0, w - legend_width)
    for i, (cw_color, ccw_color) in enumerate(zip(cw_colors, ccw_colors, strict=False)):
        base = start + 3 * i
        legend_row[base] = int(cw_color)
        legend_row[base + 1] = int(bg)
        legend_row[base + 2] = int(ccw_color)

    separator = np.full((1, w), int(bg), dtype=int)
    inp = np.vstack([legend_row[np.newaxis, :], separator, main_grid])

    if _get_bg(inp) != int(bg):
        return None

    expected = _render_expected(main_grid, frames, sprite_color)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
