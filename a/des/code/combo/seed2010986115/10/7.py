from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GRID_H = 30
MAX_GRID_W = 30
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _segments(active: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(active) and active[j + 1]:
            j += 1
        out.append((int(i), int(j)))
        i = j + 1
    return out


def _find_separator_row(g: np.ndarray) -> int:
    h = g.shape[0]
    candidates = []
    for r in range(h):
        if int(g[r, 0]) == BLACK:
            continue
        if not np.all(g[r, :] == g[r, 0]):
            continue
        if not np.any(g[:r, :] != BLACK):
            continue
        if not np.any(g[r + 1 :, :] != BLACK):
            continue
        candidates.append(int(r))
    assert len(candidates) == 1, "expected a unique full-width separator row"
    return candidates[0]


def _extract_stamps(g: np.ndarray, sep: int) -> list[dict[str, object]]:
    upper = g[:sep, :]
    active_cols = np.any(upper != BLACK, axis=0)
    groups = _segments(active_cols)
    assert groups, "expected at least one upper stamp group"

    stamps: list[dict[str, object]] = []
    for c0, c1 in groups:
        sub = upper[:, c0 : c1 + 1]
        active_rows = np.any(sub != BLACK, axis=1)
        rows = np.flatnonzero(active_rows)
        assert rows.size >= 2, "each stamp group needs a pattern and a color bar"
        bar_row = int(rows[-1])
        bar = sub[bar_row, :]
        bar_cols = np.flatnonzero(bar != BLACK)
        assert 1 <= bar_cols.size <= 3, "bar length must be 1..3"
        assert int(bar_cols[-1] - bar_cols[0] + 1) == int(bar_cols.size), "bar must be contiguous"
        bar_colors = [int(v) for v in bar[bar_cols].tolist()]
        assert len(set(bar_colors)) == len(bar_colors), "bar colors must be distinct"

        stamp_area = sub[:bar_row, :]
        stamp_coords = np.argwhere(stamp_area != BLACK)
        assert stamp_coords.size > 0, "expected non-empty stamp above the bar"
        r0 = int(stamp_coords[:, 0].min())
        r1 = int(stamp_coords[:, 0].max())
        sc0 = int(stamp_coords[:, 1].min())
        sc1 = int(stamp_coords[:, 1].max())
        patch = stamp_area[r0 : r1 + 1, sc0 : sc1 + 1].copy()
        stamps.append({"patch": patch, "bar_colors": bar_colors})
    return stamps


def _frame_info(g: np.ndarray, points: list[tuple[int, int]]) -> tuple[int, tuple[int, int, int, int]] | None:
    colors = {int(g[r, c]) for r, c in points}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if set(points) != border:
        return None
    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != BLACK):
        return None
    return color, (r0 + 1, c0 + 1, r1 - 1, c1 - 1)


def _extract_frames(g: np.ndarray, sep: int) -> dict[int, tuple[int, int, int, int]]:
    lower = g[sep + 1 :, :]
    comps = _components(lower != BLACK)
    frames: dict[int, tuple[int, int, int, int]] = {}
    interior_shape: tuple[int, int] | None = None
    for comp in comps:
        shifted = [(r + sep + 1, c) for r, c in comp]
        info = _frame_info(g, shifted)
        if info is None:
            continue
        color, interior = info
        ir0, ic0, ir1, ic1 = interior
        shape = (ir1 - ir0 + 1, ic1 - ic0 + 1)
        if interior_shape is None:
            interior_shape = shape
        else:
            assert shape == interior_shape, "all frame interiors must have the same size"
        assert color not in frames, "frame colors must be unique"
        frames[color] = interior
    assert frames, "expected at least one lower frame"
    return frames


def _paste_centered(out: np.ndarray, patch: np.ndarray, interior: tuple[int, int, int, int]) -> None:
    ir0, ic0, ir1, ic1 = interior
    ih = ir1 - ir0 + 1
    iw = ic1 - ic0 + 1
    ph, pw = patch.shape
    assert ph <= ih and pw <= iw, "stamp must fit inside every envelope"
    top = ir0 + (ih - ph) // 2
    left = ic0 + (iw - pw) // 2
    mask = patch != BLACK
    out[top : top + ph, left : left + pw][mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat upper patterns as stamps and copy each one into the lower envelopes whose
    border colors appear in the stamp's color bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_row(g)
    stamps = _extract_stamps(g, sep)
    frames = _extract_frames(g, sep)

    out = g.copy()
    for stamp in stamps:
        patch = stamp["patch"]
        for color in stamp["bar_colors"]:
            color = int(color)
            if color in frames:
                _paste_centered(out, patch, frames[color])
    return out


def _compositions(total: int, parts: int) -> list[list[int]]:
    if parts == 1:
        return [[total]] if 1 <= total <= 3 else []
    out: list[list[int]] = []
    for first in range(1, 4):
        for rest in _compositions(total - first, parts - 1):
            out.append([first] + rest)
    return out


def _sample_group_sizes(n_env: int) -> list[int]:
    choices: list[list[int]] = []
    for parts in range(2, min(4, n_env) + 1):
        if n_env <= 3 * parts:
            choices.extend(_compositions(n_env, parts))
    assert choices
    multi = [choice for choice in choices if any(v > 1 for v in choice)]
    if multi and random.random() < 0.7:
        choices = multi
    return random.choice(choices)


def _make_stamp(h: int, w: int) -> np.ndarray:
    color_count = random.randint(2, min(4, len(COLOR_POOL)))
    colors = random.sample(COLOR_POOL, color_count)
    mode = random.choice(("rows", "cols", "checker", "random"))
    patch = np.zeros((h, w), dtype=int)

    if mode == "rows":
        row_colors = [random.choice(colors) for _ in range(h)]
        if len(set(row_colors)) == 1:
            row_colors[-1] = colors[1]
        for r, color in enumerate(row_colors):
            patch[r, :] = color
    elif mode == "cols":
        col_colors = [random.choice(colors) for _ in range(w)]
        if len(set(col_colors)) == 1:
            col_colors[-1] = colors[1]
        for c, color in enumerate(col_colors):
            patch[:, c] = color
    elif mode == "checker":
        offset = random.randint(0, color_count - 1)
        for r in range(h):
            for c in range(w):
                patch[r, c] = colors[(r + c + offset) % color_count]
    else:
        for r in range(h):
            for c in range(w):
                patch[r, c] = random.choice(colors)
        if len(set(int(v) for v in patch.ravel())) == 1:
            patch[-1, -1] = colors[1]

    if h >= 3 and w >= 3 and random.random() < 0.25:
        rr = random.randint(1, h - 2)
        cc = random.randint(1, w - 2)
        patch[rr, cc] = BLACK

    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    return patch


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_env = random.randint(3, 5)
    frame_colors = random.sample(COLOR_POOL, n_env)
    group_sizes = _sample_group_sizes(n_env)
    random.shuffle(frame_colors)

    groups: list[list[int]] = []
    cursor = 0
    for size in group_sizes:
        colors = frame_colors[cursor : cursor + size]
        cursor += size
        random.shuffle(colors)
        groups.append(colors)
    random.shuffle(groups)

    inner_h = random.randint(3, 5)
    inner_w = random.randint(max(4, max(group_sizes)), 6)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    stamps: list[dict[str, object]] = []
    for colors in groups:
        sh = random.randint(2, inner_h)
        sw = random.randint(max(2, len(colors)), inner_w)
        stamps.append({"patch": _make_stamp(sh, sw), "bar_colors": colors[:]})

    top_gap_choices = [random.randint(1, 2) for _ in range(max(0, len(stamps) - 1))]
    top_left_margin = random.randint(1, 2)
    top_right_margin = random.randint(1, 2)
    top_min_w = top_left_margin + top_right_margin + sum(int(stamp["patch"].shape[1]) for stamp in stamps) + sum(top_gap_choices)

    env_rows = random.choice([1, 2] if n_env >= 4 else [1, 2] if random.random() < 0.5 else [1])
    env_cols = (n_env + env_rows - 1) // env_rows
    lower_top_margin = random.randint(1, 2)
    lower_bottom_margin = random.randint(1, 2)
    lower_left_margin = random.randint(1, 2)
    lower_right_margin = random.randint(1, 2)
    row_gap = random.randint(1, 2)
    col_gap = random.randint(1, 2)
    lower_h = lower_top_margin + lower_bottom_margin + env_rows * outer_h + (env_rows - 1) * row_gap
    lower_w = lower_left_margin + lower_right_margin + env_cols * outer_w + (env_cols - 1) * col_gap

    width_min = max(top_min_w, lower_w)
    max_stamp_h = max(int(stamp["patch"].shape[0]) for stamp in stamps)
    top_band_h = max_stamp_h + random.randint(2, 4)
    height_min = top_band_h + 1 + lower_h
    if width_min > MAX_GRID_W or height_min > MAX_GRID_H:
        return None

    height = random.randint(height_min, MAX_GRID_H)
    width = random.randint(width_min, MAX_GRID_W)
    sep = top_band_h
    grid = np.full((height, width), BLACK, dtype=int)
    grid[sep, :] = GREY

    top_x = random.randint(0, width - top_min_w)
    xpos = top_x + top_left_margin
    for i, stamp in enumerate(stamps):
        patch = stamp["patch"]
        sh, sw = patch.shape
        max_r0 = sep - sh - 1
        if max_r0 < 1:
            return None
        r0 = random.randint(1, max_r0)
        grid[r0 : r0 + sh, xpos : xpos + sw] = patch

        colors = stamp["bar_colors"][:]
        random.shuffle(colors)
        stamp["bar_colors"] = colors
        bar_start = xpos + random.randint(0, sw - len(colors))
        grid[r0 + sh, bar_start : bar_start + len(colors)] = np.array(colors, dtype=int)

        xpos += sw
        if i + 1 < len(stamps):
            xpos += top_gap_choices[i]

    lower_x = random.randint(0, width - lower_w)
    lower_y = sep + 1
    slots = []
    for rr in range(env_rows):
        for cc in range(env_cols):
            sr = lower_y + lower_top_margin + rr * (outer_h + row_gap)
            sc = lower_x + lower_left_margin + cc * (outer_w + col_gap)
            slots.append((sr, sc))
    chosen_slots = random.sample(slots, n_env)

    frame_targets: dict[int, tuple[int, int, int, int]] = {}
    for color, (r0, c0) in zip(frame_colors, chosen_slots, strict=False):
        _draw_frame(grid, r0, c0, outer_h, outer_w, color)
        frame_targets[int(color)] = (r0 + 1, c0 + 1, r0 + inner_h, c0 + inner_w)

    out = grid.copy()
    for stamp in stamps:
        patch = stamp["patch"]
        for color in stamp["bar_colors"]:
            _paste_centered(out, patch, frame_targets[int(color)])

    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, out = built
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input for task 7")
