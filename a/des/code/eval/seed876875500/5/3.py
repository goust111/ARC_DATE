from __future__ import annotations

import random
import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
PALETTE_POOL = [1, 2, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    return int(cols[0])


def _extract_palette_colors(right: np.ndarray) -> list[int]:
    comps = _components(right != BLACK)
    blocks = []
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        patch = right[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (2, 2), "palette objects must be 2x2"
        color = int(patch[0, 0])
        assert color != BLACK and bool((patch == color).all()), "palette blocks must be monochrome"
        blocks.append((r0, c0, color))

    blocks.sort(key=lambda item: (item[0], item[1]))
    colors = [int(color) for _, _, color in blocks]
    assert colors and len(colors) == len(set(colors)), "palette colors must be distinct"
    return colors


def _ordered_path(left: np.ndarray) -> list[tuple[int, int]]:
    h, w = left.shape
    pts = {tuple(map(int, rc)) for rc in np.argwhere(left == GREEN)}
    assert pts, "left region must contain a green path"

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pts:
        nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if (nr, nc) in pts]
        assert 1 <= len(nbrs) <= 2, "path must be a non-branching 4-connected line"
        adj[(r, c)] = nbrs

    ends = [pt for pt, nbrs in adj.items() if len(nbrs) == 1]
    assert len(ends) == 2, "path must have exactly two endpoints"
    start = min(ends, key=lambda pt: (pt[0], pt[1]))

    ordered = []
    prev = None
    cur = start
    while True:
        ordered.append(cur)
        nxts = [pt for pt in adj[cur] if pt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must remain unambiguous"
        prev, cur = cur, nxts[0]

    assert len(ordered) == len(pts), "path must be a single connected component"
    return ordered


def _turn_code(prev_dir: tuple[int, int], next_dir: tuple[int, int]) -> str:
    cross = prev_dir[0] * next_dir[1] - prev_dir[1] * next_dir[0]
    assert cross != 0, "turn points must be 90-degree bends"
    return "L" if cross > 0 else "R"


def _split_segments(path: list[tuple[int, int]]) -> tuple[list[list[tuple[int, int]]], list[str]]:
    assert len(path) >= 2, "path must contain at least one step"
    dirs = []
    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        dirs.append((dr, dc))

    segments: list[list[tuple[int, int]]] = []
    turns: list[str] = []
    start = 0
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i - 1]:
            turns.append(_turn_code(dirs[i - 1], dirs[i]))
            segments.append(path[start : i + 1])
            start = i + 1
    segments.append(path[start:])

    assert all(seg for seg in segments), "segments must be non-empty"
    return segments, turns


def _segment_colors(palette_colors: list[int], turns: list[str]) -> list[int]:
    n = len(palette_colors)
    assert n % 2 == 1, "palette must have a unique middle block"
    assert len(turns) + 1 == n, "palette size must match segment count"

    mid = n // 2
    colors = [int(palette_colors[mid])]
    top_idx = 0
    bottom_idx = n - 1

    for turn in turns:
        if turn == "L":
            assert top_idx < mid, "left turns over-consumed the palette top"
            colors.append(int(palette_colors[top_idx]))
            top_idx += 1
        else:
            assert bottom_idx > mid, "right turns over-consumed the palette bottom"
            colors.append(int(palette_colors[bottom_idx]))
            bottom_idx -= 1

    assert top_idx == mid and bottom_idx == mid, "left/right turn counts must consume the palette symmetrically"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each straight segment of the left-side green path using palette colors selected
    from the middle, top-down, and bottom-up according to turn direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator(g)
    left = g[:, :sep].copy()
    right = g[:, sep + 1 :]

    palette_colors = _extract_palette_colors(right)
    path = _ordered_path(left)
    segments, turns = _split_segments(path)
    segment_colors = _segment_colors(palette_colors, turns)

    assert len(segments) == len(segment_colors)
    out = left.copy()
    out[left == GREEN] = BLACK
    for seg, color in zip(segments, segment_colors):
        for r, c in seg:
            out[r, c] = int(color)
    return out


def _random_partition(total: int, parts: int, minimum: int) -> list[int]:
    assert total >= parts * minimum
    values = [minimum] * parts
    for _ in range(total - parts * minimum):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _choose_columns(num_horizontal: int, left_w: int) -> list[int] | None:
    max_col = left_w - 2
    near_left_max = min(2, max_col)
    if max_col < 2:
        return None

    for _ in range(200):
        cols = [random.randint(0, near_left_max)]
        ok = True
        for _step in range(num_horizontal - 1):
            candidates = [c for c in range(max_col + 1) if abs(c - cols[-1]) >= 2]
            if not candidates:
                ok = False
                break
            cols.append(random.choice(candidates))
        if not ok:
            continue
        final_candidates = [c for c in range(near_left_max + 1) if abs(c - cols[-1]) >= 2]
        if not final_candidates:
            continue
        cols.append(random.choice(final_candidates))
        return cols
    return None


def _build_path(start_r: int, cols: list[int], vertical_steps: list[int]) -> list[tuple[int, int]]:
    r = int(start_r)
    c = int(cols[0])
    path = [(r, c)]

    for i in range(len(cols) - 1):
        for _ in range(vertical_steps[i]):
            r += 1
            path.append((r, c))
        step = 1 if cols[i + 1] > c else -1
        for _ in range(abs(cols[i + 1] - c)):
            c += step
            path.append((r, c))

    for _ in range(vertical_steps[-1]):
        r += 1
        path.append((r, c))
    return path


def _validate_generated(grid: np.ndarray, expected_segments: int) -> None:
    sep = _find_separator(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]

    palette_colors = _extract_palette_colors(right)
    path = _ordered_path(left)
    segments, turns = _split_segments(path)

    assert len(palette_colors) == expected_segments
    assert len(segments) == expected_segments
    assert len(turns) == expected_segments - 1
    assert len(set(path)) == len(path), "path must not self-overlap"

    out = main(grid)
    assert out.shape == left.shape
    assert int(np.sum(out != BLACK)) == len(path)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        num_segments = random.choice([3, 5, 7])
        num_horizontal = num_segments // 2

        min_h = max(12, 3 * num_segments + 1)
        grid_h = random.randint(min_h, min_h + 8)
        left_w = random.randint(7, 13)
        right_w = random.randint(3, 6)
        total_w = left_w + 1 + right_w

        start_r = random.randint(0, 2)
        bottom_margin = random.randint(0, 2)
        total_descent = grid_h - 1 - bottom_margin - start_r
        min_vertical_total = 2 * (num_horizontal + 1)
        if total_descent < min_vertical_total:
            continue

        vertical_steps = _random_partition(total_descent, num_horizontal + 1, 2)
        cols = _choose_columns(num_horizontal, left_w)
        if cols is None:
            continue

        path = _build_path(start_r, cols, vertical_steps)
        grid = np.full((grid_h, total_w), BLACK, dtype=int)
        for r, c in path:
            grid[r, c] = GREEN

        sep = left_w
        grid[:, sep] = GREY

        palette_colors = random.sample(PALETTE_POOL, num_segments)
        stack_h = 3 * num_segments - 1
        palette_top = random.randint(0, grid_h - stack_h)
        palette_c = sep + 1 + random.randint(0, right_w - 2)
        for i, color in enumerate(palette_colors):
            rr = palette_top + 3 * i
            grid[rr : rr + 2, palette_c : palette_c + 2] = int(color)

        try:
            _validate_generated(grid, num_segments)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
