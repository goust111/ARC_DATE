from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PATH_COLORS = (1, 2, 3)
FRAME_COLOR_POOL = (4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _ordered_path_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = np.isin(grid, PATH_COLORS)
    comps = _components_from_mask(mask)
    assert len(comps) == 1, "expected one 4-connected path"
    comp = comps[0]
    coord_set = {tuple(map(int, x)) for x in comp.tolist()}

    degrees: dict[tuple[int, int], int] = {}
    for r, c in coord_set:
        deg = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))
        assert deg <= 2, "expected a branch-free path"
        degrees[(r, c)] = deg

    endpoints = sorted([coord for coord, deg in degrees.items() if deg == 1])
    assert len(endpoints) == 2, "expected exactly two path endpoints"
    start = endpoints[0]

    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [p for p in _neighbors4(*cur) if p in coord_set and p != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "expected a unique next step along the path"
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt

    assert len(order) == len(coord_set), "path traversal must cover all path cells"
    return order


def _frame_component(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    cells = {tuple(map(int, x)) for x in coords.tolist()}
    border = []
    for c in range(c0, c1 + 1):
        border.append((r0, c))
        border.append((r1, c))
    for r in range(r0 + 1, r1):
        border.append((r, c0))
        border.append((r, c1))
    border = sorted(set(border))

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(cells):
        return None
    if any((r not in (r0, r1) and c not in (c0, c1)) for r, c in cells):
        return None

    missing = [p for p in border if p not in cells]
    if len(missing) not in (1, 2, 3):
        return None
    if any(p in corners for p in missing):
        return None
    same_top = all(r == r0 for r, _c in missing)
    same_bottom = all(r == r1 for r, _c in missing)
    same_left = all(c == c0 for _r, c in missing)
    same_right = all(c == c1 for _r, c in missing)
    if sum((same_top, same_bottom, same_left, same_right)) != 1:
        return None
    if same_top or same_bottom:
        vals = sorted(c for _r, c in missing)
    else:
        vals = sorted(r for r, _c in missing)
    if vals[-1] - vals[0] + 1 != len(vals):
        return None

    color = int(grid[coords[0, 0], coords[0, 1]])
    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "missing": missing,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - {BG, *PATH_COLORS}):
        comps = _components_from_mask(grid == color)
        for comp in comps:
            frame = _frame_component(grid, comp)
            if frame is not None:
                frames.append(frame)
    assert frames, "expected at least one rectangular frame"
    return frames


def _choose_fill_color(colors: list[int]) -> int:
    counts = Counter(colors)
    best = max(counts.values())
    candidates = {color for color, count in counts.items() if count == best}
    for color in colors:
        if color in candidates:
            return int(color)
    raise AssertionError("unreachable")


def _transform(grid: np.ndarray, *, reverse_colors: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    path = _ordered_path_coords(g)
    frames = _find_frames(g)

    path_colors = [int(g[r, c]) for r, c in path]
    active_colors = path_colors[::-1] if reverse_colors else path_colors
    recolored = g.copy()
    for (r, c), color in zip(path, active_colors):
        recolored[r, c] = int(color)

    out = np.full_like(g, BG)
    for frame in frames:
        rr = frame["coords"][:, 0]
        cc = frame["coords"][:, 1]
        out[rr, cc] = frame["color"]

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        indices = [
            idx for idx, (r, c) in enumerate(path)
            if r0 < r < r1 and c0 < c < c1
        ]
        assert len(indices) >= 5, "each frame must contain a substantial path segment"
        local_colors = [int(active_colors[idx]) for idx in indices]
        fill = _choose_fill_color(local_colors)
        out[r0 + 1:r1, c0 + 1:c1] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse a multicolor path, then fill each traversed frame interior from the reversed local path majority.

    Input structure:
    - Black background surrounds several different-color rectangular outline frames and one 4-connected single-pixel path.
    - The path uses exactly the three path colors 1/2/3, has exactly two endpoints, and passes through every frame.
    - Each frame is an axis-aligned rectangular outline with a short side opening used as a doorway; the path spends multiple pixels inside each frame.

    Rule:
    - Find the path endpoints and traverse the whole path from the lexicographically first endpoint.
    - Reverse the path's color sequence in place while keeping the path geometry unchanged.
    - For each frame, look only at the reversed path pixels strictly inside the frame interior, choose the most frequent color, and break ties by the earliest such color along the traversal order.
    - Fill the whole frame interior with that chosen color, covering the interior path pixels; erase all path pixels outside frame interiors to black and keep the frame outlines unchanged.

    Key insight:
    - The global color-sequence reversal moves color concentration from early path segments to late ones, so frame fill colors must be computed after reversing the path rather than from the visible input distribution.
    """
    return _transform(grid, reverse_colors=True)


def _line_exclusive(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    out: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            out.append((r0, c))
    else:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            out.append((r, c0))
    return out


def _connector(start: tuple[int, int], end: tuple[int, int], mid_col: int) -> list[tuple[int, int]]:
    cur = start
    out: list[tuple[int, int]] = []
    if cur[1] != mid_col:
        nxt = (cur[0], mid_col)
        out.extend(_line_exclusive(cur, nxt))
        cur = nxt
    if cur[0] != end[0]:
        nxt = (end[0], cur[1])
        out.extend(_line_exclusive(cur, nxt))
        cur = nxt
    if cur[1] != end[1]:
        out.extend(_line_exclusive(cur, end))
    return out


def _frame_path_local(h: int, w: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]]:
    assert h >= 7 and w >= 6
    coords: list[tuple[int, int]] = [(0, 1)]
    coords.extend((r, 1) for r in range(1, h - 1))
    coords.extend((h - 2, c) for c in range(2, w - 1))
    coords.extend((r, w - 2) for r in range(h - 3, 0, -1))
    coords.extend((1, c) for c in range(w - 3, 2, -1))
    coords.append((0, 3))
    interior = {(r, c) for r, c in coords if 0 < r < h - 1 and 0 < c < w - 1}
    opening = [(0, 1), (0, 2), (0, 3)]
    return coords, interior, opening


def _shift(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _make_majority_pattern(length: int, major: int) -> list[int]:
    others = [c for c in PATH_COLORS if c != major]
    major_count = random.randint(length // 2 + 1, length - 2)
    rest = length - major_count
    first_minor = random.randint(1, rest - 1)
    second_minor = rest - first_minor
    blocks = [
        [random.choice(others)] * first_minor,
        [major] * major_count,
        [others[0] if others[0] != major else others[1]] * second_minor,
    ]
    if random.random() < 0.5:
        blocks.reverse()
    seq: list[int] = []
    for block in blocks:
        seq.extend(block)
    return seq


def _make_tie_pattern(length: int, winner: int) -> list[int]:
    loser = random.choice([c for c in PATH_COLORS if c != winner])
    third = random.choice([c for c in PATH_COLORS if c not in (winner, loser)])
    low = max(2, length // 3 + 1)
    high = max(low, (length - 1) // 2)
    max_count = random.randint(low, high)
    third_count = length - 2 * max_count
    assert 0 < third_count < max_count
    seq = [winner] * max_count
    seq.extend([third] * third_count)
    seq.extend([loser] * max_count)
    return seq


def _random_path_runs(length: int) -> list[int]:
    seq: list[int] = []
    while len(seq) < length:
        color = random.choice(PATH_COLORS)
        run = random.randint(1, min(4, length - len(seq)))
        seq.extend([color] * run)
    return seq


def _fills_by_frame(inp: np.ndarray, reverse_colors: bool) -> list[int]:
    g = np.asarray(inp, dtype=int)
    path = _ordered_path_coords(g)
    frames = _find_frames(g)
    path_colors = [int(g[r, c]) for r, c in path]
    active = path_colors[::-1] if reverse_colors else path_colors
    ranked: list[tuple[int, int]] = []
    fills: list[int] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        indices = [i for i, (r, c) in enumerate(path) if r0 < r < r1 and c0 < c < c1]
        ranked.append((min(indices), len(fills)))
        fills.append(_choose_fill_color([active[i] for i in indices]))
    return [fills[idx] for _start, idx in sorted(ranked)]


def _semantic_checks(inp: np.ndarray, expected_tie: bool) -> bool:
    out = main(inp)
    naive = _transform(inp, reverse_colors=False)
    if np.array_equal(out, naive):
        return False

    frames = _find_frames(inp)
    if len(frames) < 3:
        return False

    good_fills = _fills_by_frame(inp, True)
    bad_fills = _fills_by_frame(inp, False)
    if len(set(good_fills)) < 2:
        return False
    if all(a == b for a, b in zip(good_fills, bad_fills)):
        return False

    if expected_tie:
        path = _ordered_path_coords(inp)
        reversed_colors = [int(inp[r, c]) for r, c in path][::-1]
        tie_seen = False
        for frame in _find_frames(inp):
            r0, c0, r1, c1 = frame["bbox"]
            indices = [i for i, (r, c) in enumerate(path) if r0 < r < r1 and c0 < c < c1]
            colors = [reversed_colors[i] for i in indices]
            counts = Counter(colors)
            top = counts.most_common()
            if len(top) >= 2 and top[0][1] == top[1][1]:
                tie_seen = True
                break
        if not tie_seen:
            return False

    return out.shape[0] <= 30 and out.shape[1] <= 30


def _build_candidate() -> tuple[np.ndarray, bool] | None:
    n_frames = random.choice([3, 4])
    widths = [random.randint(6, 7) for _ in range(n_frames)]
    heights = [random.randint(7, 8) for _ in range(n_frames)]
    gaps = [random.randint(1, 2) for _ in range(n_frames - 1)]
    frame_colors = random.sample(FRAME_COLOR_POOL, n_frames)
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    grid_w = left_margin + sum(widths) + sum(gaps) + right_margin
    if grid_w > 30:
        return None

    grid_h = random.randint(max(16, max(heights) + 4), 22)
    frames: list[dict] = []
    c0 = left_margin
    for idx in range(n_frames):
        h = heights[idx]
        w = widths[idx]
        top = random.randint(2, grid_h - h - 2)
        local_path, interior_local, opening_local = _frame_path_local(h, w)
        frames.append(
            {
                "bbox": (top, c0, top + h - 1, c0 + w - 1),
                "color": frame_colors[idx],
                "path_local": local_path,
                "interior_local": interior_local,
                "opening_local": opening_local,
            }
        )
        c0 += w
        if idx < n_frames - 1:
            c0 += gaps[idx]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0:c1 + 1] = frame["color"]
        grid[r1, c0:c1 + 1] = frame["color"]
        grid[r0:r1 + 1, c0] = frame["color"]
        grid[r0:r1 + 1, c1] = frame["color"]
        for gr, gc in frame["opening_local"]:
            grid[r0 + gr, c0 + gc] = BG

    corridor_r = 0
    path: list[tuple[int, int]] = []
    first = frames[0]
    fr0, fc0, _fr1, _fc1 = first["bbox"]
    start_gap = first["path_local"][0]
    entry = (fr0 + start_gap[0], fc0 + start_gap[1])
    path.extend([(corridor_r, c) for c in range(0, entry[1] + 1)])
    path.extend(_line_exclusive((corridor_r, entry[1]), entry))

    for idx, frame in enumerate(frames):
        r0, c0, _r1, _c1 = frame["bbox"]
        shifted = _shift(frame["path_local"], r0, c0)
        path.extend(shifted[1:])
        if idx == n_frames - 1:
            break
        next_frame = frames[idx + 1]
        nr0, nc0, _nr1, _nc1 = next_frame["bbox"]
        gap_now = shifted[-1]
        gap_next = (nr0 + next_frame["path_local"][0][0], nc0 + next_frame["path_local"][0][1])
        path.extend(_line_exclusive(gap_now, (corridor_r, gap_now[1])))
        path.extend(_line_exclusive((corridor_r, gap_now[1]), (corridor_r, gap_next[1])))
        path.extend(_line_exclusive((corridor_r, gap_next[1]), gap_next))

    tail = path[-1]
    path.extend(_line_exclusive(tail, (corridor_r, tail[1])))
    path.extend(_line_exclusive((corridor_r, tail[1]), (corridor_r, grid_w - 1)))

    if path[0] > path[-1]:
        path = path[::-1]

    if len(path) != len(set(path)):
        return None

    mask = np.zeros_like(grid, dtype=bool)
    for r, c in path:
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return None
        if grid[r, c] not in (BG,):
            start_or_end = False
            for frame in frames:
                rr0, cc0, _rr1, _cc1 = frame["bbox"]
                abs_start = (rr0 + frame["path_local"][0][0], cc0 + frame["path_local"][0][1])
                abs_end = (rr0 + frame["path_local"][-1][0], cc0 + frame["path_local"][-1][1])
                if (r, c) in (abs_start, abs_end):
                    start_or_end = True
                    break
            if not start_or_end:
                return None
        if mask[r, c]:
            return None
        mask[r, c] = True

    comp = _components_from_mask(mask)
    if len(comp) != 1 or len(comp[0]) != len(path):
        return None

    degrees = Counter()
    coord_set = set(path)
    for coord in coord_set:
        deg = sum(nei in coord_set for nei in _neighbors4(*coord))
        if deg > 2:
            return None
        degrees[deg] += 1
    if degrees[1] != 2:
        return None

    index_by_coord = {coord: idx for idx, coord in enumerate(path)}
    frame_infos: list[tuple[int, dict]] = []
    for frame in frames:
        r0, c0, _r1, _c1 = frame["bbox"]
        interior_abs = {(r0 + r, c0 + c) for r, c in frame["interior_local"]}
        indices = sorted(index_by_coord[cell] for cell in interior_abs)
        if len(indices) < 5:
            return None
        frame["indices"] = indices
        frame_infos.append((indices[0], frame))
    frames = [frame for _start, frame in sorted(frame_infos)]

    target_fills = random.choices(PATH_COLORS, k=n_frames)
    if len(set(target_fills)) < 2:
        return None
    if all(target_fills[i] == target_fills[-1 - i] for i in range(n_frames)):
        return None

    tie_frame = random.randrange(n_frames) if random.random() < 0.4 else None
    reversed_seq = _random_path_runs(len(path))
    for idx, frame in enumerate(frames):
        seg = frame["indices"]
        fill = int(target_fills[idx])
        if tie_frame == idx:
            local = _make_tie_pattern(len(seg), fill)
        else:
            local = _make_majority_pattern(len(seg), fill)
        for pos, color in zip(seg, local):
            reversed_seq[pos] = int(color)

    input_seq = reversed_seq[::-1]
    for (r, c), color in zip(path, input_seq):
        grid[r, c] = int(color)

    if not _semantic_checks(grid, expected_tie=tie_frame is not None):
        return None
    return grid, tie_frame is not None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_candidate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid reverse-fill ARC input")
