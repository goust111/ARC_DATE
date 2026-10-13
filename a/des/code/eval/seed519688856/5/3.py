from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GEN_ATTEMPTS = 400
HEIGHT_RANGE = (2, 6)
BRANCHES = ("pair", "chain", "triple")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _parse_legend(grid: np.ndarray) -> dict[tuple[int, int], int]:
    h, _ = grid.shape
    entries = []
    for pts in _components(grid != BG):
        if any(r == h - 1 for r, _ in pts):
            continue
        entries.append(pts)
    assert entries, "expected at least one floating legend entry"

    mix: dict[tuple[int, int], int] = {}
    for pts in entries:
        assert len(pts) == 3, "legend entries must be 3-cell L-shapes"
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        assert r1 - r0 == 1 and c1 - c0 == 1, "legend entry must fit in 2x2 box"

        local = {(r - r0, c - c0): (r, c) for r, c in pts}
        col_counts = {0: 0, 1: 0}
        for _, lc in local:
            col_counts[lc] += 1
        pair_col = [lc for lc, count in col_counts.items() if count == 2]
        assert len(pair_col) == 1, "legend must contain one vertical pair"
        pair_col = pair_col[0]
        other_col = 1 - pair_col

        assert (0, pair_col) in local and (1, pair_col) in local, "legend pair malformed"
        assert (0, other_col) in local, "legend result cell must align with top color"

        a = int(grid[local[(0, pair_col)]])
        b = int(grid[local[(1, pair_col)]])
        c = int(grid[local[(0, other_col)]])
        mix[(a, b)] = c
        mix[(b, a)] = c
    return mix


def _detect_towers(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    assert np.all(grid[h - 1] != BG), "bottom row must be the ground layer"

    towers: list[dict[str, int]] = []
    for c in range(w):
        color = int(grid[h - 1, c])
        r = h - 1
        while r - 1 >= 0 and int(grid[r - 1, c]) == color:
            r -= 1
        height = h - r
        if height >= 2:
            towers.append({"col": c, "color": color, "height": height, "top": r})
    assert towers, "expected at least one tower"
    towers.sort(key=lambda t: t["col"])
    return towers


def _mix_sequence(colors: list[int], mix: dict[tuple[int, int], int]) -> int:
    acc = colors[0]
    for nxt in colors[1:]:
        if acc == nxt:
            continue
        key = (acc, nxt)
        assert key in mix, f"missing legend entry for {key}"
        acc = mix[key]
    return acc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast tower colors sideways based on tower height and mix overlaps via a legend.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    mix = _parse_legend(g)
    towers = _detect_towers(g)

    covers = [[[] for _ in range(w)] for _ in range(h - 1)]
    for tower in towers:
        left = max(0, tower["col"] - tower["height"])
        right = min(w - 1, tower["col"] + tower["height"])
        for r in range(tower["top"], h - 1):
            for c in range(left, right + 1):
                if int(g[r, c]) == BG:
                    covers[r][c].append(tower["color"])

    out = g.copy()
    for r in range(h - 1):
        for c in range(w):
            if int(g[r, c]) != BG:
                continue
            if covers[r][c]:
                out[r, c] = _mix_sequence(covers[r][c], mix)
    return out


def _sample_layout(rng: random.Random, branch: str) -> tuple[list[int], list[int], int] | None:
    if branch == "triple":
        height = rng.randint(3, 6)
        heights = [height, height, height]
        d12 = rng.randint(1, height)
        d23 = rng.randint(1, height)
        span = d12 + d23
        return heights, [d12, d23], span

    for _ in range(120):
        heights = [rng.randint(*HEIGHT_RANGE) for _ in range(3)]
        if branch == "pair":
            d12 = rng.randint(1, min(heights[0] + heights[1], 5))
            min_d23 = max(heights[1] + heights[2] + 1, heights[0] + heights[2] - d12 + 1)
            max_d23 = min_d23 + 4
            d23 = rng.randint(min_d23, max_d23)
            return heights, [d12, d23], d12 + d23

        d12_cap = min(heights[0] + heights[1], 6)
        d23_cap = min(heights[1] + heights[2], 6)
        if d12_cap < 1 or d23_cap < 1:
            continue
        d12 = rng.randint(1, d12_cap)
        d23 = rng.randint(1, d23_cap)
        if d12 + d23 <= heights[0] + heights[2]:
            continue
        return heights, [d12, d23], d12 + d23
    return None


def _branch_entries(branch: str, tower_colors: list[int], mix_colors: list[int]) -> list[tuple[int, int, int]]:
    c1, c2, c3 = tower_colors
    if branch == "pair":
        return [(c1, c2, mix_colors[0])]
    if branch == "chain":
        return [(c1, c2, mix_colors[0]), (c2, c3, mix_colors[1])]
    m12 = mix_colors[0]
    return [(c1, c2, m12), (c2, c3, mix_colors[1]), (m12, c3, mix_colors[2])]


def _build_ground_row(
    rng: random.Random,
    width: int,
    forced: dict[int, int],
    palette: list[int],
) -> np.ndarray | None:
    row = np.full(width, -1, dtype=int)
    for c in range(width):
        if c in forced:
            color = forced[c]
            if c > 0 and row[c - 1] == color:
                return None
            row[c] = color
            continue

        choices = [color for color in palette if color != row[c - 1]]
        next_forced = forced.get(c + 1)
        filtered = [color for color in choices if color != next_forced]
        if filtered:
            choices = filtered
        if not choices:
            return None
        row[c] = rng.choice(choices)
    return row


def _place_legend(
    grid: np.ndarray,
    entries: list[tuple[int, int, int]],
    side: str,
    rng: random.Random,
) -> None:
    items = entries[:]
    rng.shuffle(items)
    _, w = grid.shape
    for idx, (a, b, c) in enumerate(items):
        r = idx * 3
        if side == "left":
            grid[r, 0] = a
            grid[r + 1, 0] = b
            grid[r, 1] = c
        else:
            grid[r, w - 1] = a
            grid[r + 1, w - 1] = b
            grid[r, w - 2] = c


def _coverage_patterns(grid: np.ndarray, towers: list[dict[str, int]]) -> set[tuple[int, ...]]:
    h, w = grid.shape
    patterns: set[tuple[int, ...]] = set()
    for r in range(h - 1):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            cover = []
            for idx, tower in enumerate(towers):
                if tower["top"] <= r < h - 1 and abs(c - tower["col"]) <= tower["height"]:
                    cover.append(idx)
            if cover:
                patterns.add(tuple(cover))
    return patterns


def _matches_branch(branch: str, patterns: set[tuple[int, ...]]) -> bool:
    if branch == "pair":
        return (0, 1) in patterns and (1, 2) not in patterns and (0, 2) not in patterns and all(
            len(p) <= 2 for p in patterns
        )
    if branch == "chain":
        return (0, 1) in patterns and (1, 2) in patterns and (0, 2) not in patterns and all(
            len(p) <= 2 for p in patterns
        )
    return (0, 1) in patterns and (1, 2) in patterns and (0, 1, 2) in patterns and (0, 2) not in patterns


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    branch = rng.choice(BRANCHES)
    side = rng.choice(("left", "right"))

    layout = _sample_layout(rng, branch)
    if layout is None:
        return None
    heights, gaps, span = layout

    mix_count = {"pair": 1, "chain": 2, "triple": 3}[branch]
    filler_count = rng.randint(1, 2)
    color_pool = rng.sample(list(range(1, 10)), 3 + mix_count + filler_count)
    tower_colors = color_pool[:3]
    mix_colors = color_pool[3 : 3 + mix_count]
    filler_colors = color_pool[3 + mix_count :]
    entries = _branch_entries(branch, tower_colors, mix_colors)

    legend_rows = 3 * len(entries) - 1
    max_height = max(heights)
    min_height = legend_rows + max_height + 2
    height = rng.randint(min_height, min_height + 4)

    margin = max_height
    min_width = 2 * margin + span + 1
    width_lo = max(15, min_width)
    width_hi = max(width_lo, min(width_lo + 6, 32))
    width = rng.randint(width_lo, width_hi)
    if width - 1 - margin - span < margin:
        return None

    start = rng.randint(margin, width - 1 - margin - span)
    cols = [start, start + gaps[0], start + gaps[0] + gaps[1]]
    forced = {col: color for col, color in zip(cols, tower_colors)}
    ground_palette = tower_colors + filler_colors
    ground = _build_ground_row(rng, width, forced, ground_palette)
    if ground is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[height - 1] = ground

    towers: list[dict[str, int]] = []
    for col, tower_color, tower_height in zip(cols, tower_colors, heights):
        top = height - tower_height
        grid[top:height, col] = tower_color
        towers.append({"col": col, "color": tower_color, "height": tower_height, "top": top})

    if min(tower["top"] for tower in towers) <= legend_rows:
        return None
    _place_legend(grid, entries, side, rng)

    patterns = _coverage_patterns(grid, towers)
    if not _matches_branch(branch, patterns):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
