import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
GEN_RETRIES = 400


def _corner_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(row[0]) != bg and bool((row == row[0]).all()):
            if 0 < r < grid.shape[0] - 1:
                return int(r), int(row[0])
    raise AssertionError("separator row not found")


def _solid_rect_bbox(region: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(region == color)
    assert pts.size > 0
    r0, c0, r1, c1 = _bbox(pts)
    patch = region[r0 : r1 + 1, c0 : c1 + 1]
    assert bool((patch == color).all())
    assert int(pts.shape[0]) == int(patch.size)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each template polyline into its same-colored rectangle and repaint it grey.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _corner_bg(g)
    sep_r, grey = _find_separator(g, bg)
    top = g[:sep_r]
    bottom = g[sep_r + 1 :]

    top_colors = set(map(int, np.unique(top))) - {int(bg), int(grey)}
    bottom_colors = set(map(int, np.unique(bottom))) - {int(bg), int(grey)}
    assert top_colors
    assert top_colors == bottom_colors

    out = g.copy()
    out[: sep_r + 1, :] = int(bg)

    for color in sorted(top_colors):
        t_pts = np.argwhere(top == int(color))
        assert t_pts.size > 0
        tr0, tc0, tr1, tc1 = _bbox(t_pts)
        template = top[tr0 : tr1 + 1, tc0 : tc1 + 1] == int(color)
        th = int(tr1 - tr0 + 1)
        tw = int(tc1 - tc0 + 1)

        br0, bc0, br1, bc1 = _solid_rect_bbox(bottom, int(color))
        br0 += sep_r + 1
        br1 += sep_r + 1
        rh = int(br1 - br0 + 1)
        rw = int(bc1 - bc0 + 1)

        k_h, rem_h = divmod(rh, th)
        k_w, rem_w = divmod(rw, tw)
        assert rem_h == 0 and rem_w == 0 and k_h == k_w and k_h >= 1
        k = int(k_h)

        for lr, lc in np.argwhere(template):
            rr0 = int(br0 + lr * k)
            cc0 = int(bc0 + lc * k)
            out[rr0 : rr0 + k, cc0 : cc0 + k] = int(grey)

    assert out.shape == (h, w)
    return out


def _turn_count(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    prev = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(1, len(path) - 1):
        cur = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if cur != prev:
            turns += 1
        prev = cur
    return int(turns)


def _path_bbox_dims(path: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    return int(max(rows) - min(rows) + 1), int(max(cols) - min(cols) + 1)


def _neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((int(rr), int(cc)))
    return out


def _can_extend(path: list[tuple[int, int]], cand: tuple[int, int], visited: set[tuple[int, int]]) -> bool:
    if cand in visited:
        return False
    tail = path[-1]
    for cell in visited:
        if cell == tail:
            continue
        if abs(cell[0] - cand[0]) + abs(cell[1] - cand[1]) == 1:
            return False
    return True


def _search_path(
    h: int,
    w: int,
    length: int,
    min_turns: int,
    max_turns: int,
) -> list[tuple[int, int]] | None:
    boundary = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]

    def dfs(path: list[tuple[int, int]], visited: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        ph, pw = _path_bbox_dims(path)
        remaining = length - len(path)
        if ph + remaining < h or pw + remaining < w:
            return None
        turns = _turn_count(path)
        if turns > max_turns:
            return None
        if len(path) == length:
            if (ph, pw) == (h, w) and min_turns <= turns <= max_turns:
                return path.copy()
            return None

        nbrs = _neighbors(path[-1], h, w)
        random.shuffle(nbrs)
        nbrs.sort(key=lambda cell: (cell[0] in (0, h - 1)) + (cell[1] in (0, w - 1)), reverse=True)
        for cand in nbrs:
            if not _can_extend(path, cand, visited):
                continue
            path.append(cand)
            visited.add(cand)
            found = dfs(path, visited)
            if found is not None:
                return found
            visited.remove(cand)
            path.pop()
        return None

    starts = boundary[:]
    random.shuffle(starts)
    for start in starts:
        found = dfs([start], {start})
        if found is not None:
            return found
    return None


def _generate_l_path(h: int, w: int) -> list[tuple[int, int]]:
    orientation = random.choice(["tl", "tr", "bl", "br"])
    if orientation == "tl":
        cells = {(0, c) for c in range(w)} | {(r, 0) for r in range(h)}
    elif orientation == "tr":
        cells = {(0, c) for c in range(w)} | {(r, w - 1) for r in range(h)}
    elif orientation == "bl":
        cells = {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)}
    else:
        cells = {(h - 1, c) for c in range(w)} | {(r, w - 1) for r in range(h)}

    pts = np.array(sorted(cells), dtype=int)
    r0, c0, _, _ = _bbox(pts)
    return [(int(r - r0), int(c - c0)) for r, c in sorted(cells)]


def _generate_template(family: str) -> dict:
    if family == "hline":
        w = random.randint(2, 4)
        path = [(0, c) for c in range(w)]
        return {"family": family, "h": 1, "w": w, "path": path}

    if family == "vline":
        h = random.randint(2, 4)
        path = [(r, 0) for r in range(h)]
        return {"family": family, "h": h, "w": 1, "path": path}

    if family == "l":
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        path = _generate_l_path(h, w)
        return {"family": family, "h": h, "w": w, "path": path}

    for _ in range(120):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if h == 2 and w == 2:
            continue
        lengths = list(range(h + w - 1, min(h * w, h + w) + 1))
        random.shuffle(lengths)
        for length in lengths:
            path = _search_path(h, w, length, min_turns=2, max_turns=4)
            if path is not None:
                return {"family": family, "h": h, "w": w, "path": path}
    raise RuntimeError("failed to generate zigzag path")


def _partition_rows(count: int) -> list[int]:
    if count == 2:
        return random.choice([[2], [1, 1]])
    if count == 3:
        return random.choice([[3], [1, 2], [2, 1]])
    return random.choice([[2, 2], [1, 3], [3, 1]])


def _render_expected(
    grid: np.ndarray,
    objects: list[dict],
    sep_r: int,
    grey: int,
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    out[: sep_r + 1, :] = int(bg)
    for obj in objects:
        r0, c0 = obj["rect_pos"]
        k = obj["scale"]
        for pr, pc in obj["path"]:
            rr0 = r0 + pr * k
            cc0 = c0 + pc * k
            out[rr0 : rr0 + k, cc0 : cc0 + k] = int(grey)
    return out


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(2, 4)
    colors = random.sample(PALETTE, object_count)
    families = random.sample(["hline", "vline", "l", "zigzag"], object_count)

    scale_hi = {2: 5, 3: 4, 4: 3}[object_count]
    objects = []
    for color, family in zip(colors, families):
        spec = _generate_template(family)
        spec["color"] = int(color)
        spec["scale"] = random.randint(2, scale_hi)
        spec["rect_h"] = spec["h"] * spec["scale"]
        spec["rect_w"] = spec["w"] * spec["scale"]
        objects.append(spec)

    top_order = objects[:]
    random.shuffle(top_order)
    top_h = max(max(obj["h"] for obj in objects), 4) + random.randint(0, 1)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gaps = [random.randint(1, 3) for _ in range(max(0, object_count - 1))]
    x = left_margin
    for i, obj in enumerate(top_order):
        y = random.randint(0, top_h - obj["h"])
        obj["template_pos"] = (int(y), int(x))
        x += obj["w"]
        if i < len(gaps):
            x += gaps[i]
    top_w = x + right_margin

    bottom_order = objects[:]
    random.shuffle(bottom_order)
    row_sizes = _partition_rows(object_count)
    rows = []
    idx = 0
    for size in row_sizes:
        rows.append(bottom_order[idx : idx + size])
        idx += size

    side_margin = random.randint(1, 3)
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 3)
    row_gap = [random.randint(2, 4) for _ in range(max(0, len(rows) - 1))]
    row_widths = []
    row_heights = []
    row_inner_gaps = []
    for row in rows:
        gap = random.randint(2, 4)
        row_inner_gaps.append(gap)
        row_widths.append(sum(obj["rect_w"] for obj in row) + gap * (len(row) - 1))
        row_heights.append(max(obj["rect_h"] for obj in row))
    bottom_w = max(row_widths) + 2 * side_margin + random.randint(0, 2)
    bottom_h = top_margin + bottom_margin + sum(row_heights) + sum(row_gap)

    if top_w > 40 or bottom_w > 40 or top_h + 1 + bottom_h > 32:
        return None

    y = top_margin
    for row_idx, row in enumerate(rows):
        row_width = row_widths[row_idx]
        row_height = row_heights[row_idx]
        slack = bottom_w - 2 * side_margin - row_width
        x = side_margin + random.randint(0, max(0, slack))
        for i, obj in enumerate(row):
            dy = random.randint(0, row_height - obj["rect_h"])
            obj["rect_pos_bottom"] = (int(y + dy), int(x))
            x += obj["rect_w"]
            if i + 1 < len(row):
                x += row_inner_gaps[row_idx]
        y += row_height
        if row_idx < len(row_gap):
            y += row_gap[row_idx]

    grid_h = top_h + 1 + bottom_h
    grid_w = max(top_w, bottom_w)
    top_shift = random.randint(0, grid_w - top_w)
    bottom_shift = random.randint(0, grid_w - bottom_w)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    sep_r = top_h
    grid[sep_r, :] = GREY

    seen_colors = set()
    for obj in objects:
        assert obj["color"] not in seen_colors
        seen_colors.add(obj["color"])

        ty, tx = obj["template_pos"]
        ty += 0
        tx += top_shift
        for pr, pc in obj["path"]:
            grid[ty + pr, tx + pc] = obj["color"]

        by, bx = obj["rect_pos_bottom"]
        by += sep_r + 1
        bx += bottom_shift
        obj["rect_pos"] = (int(by), int(bx))
        grid[by : by + obj["rect_h"], bx : bx + obj["rect_w"]] = obj["color"]

    expected = _render_expected(grid, objects, sep_r, GREY, BG)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
