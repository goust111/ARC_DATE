import random
from collections import Counter, deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values = Counter(map(int, grid.flatten()))
    return int(values.most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _parse_task(grid: np.ndarray) -> tuple[int, tuple[int, int], int, dict[str, list[dict]]]:
    bg = _get_bg(grid)
    comps = _find_components(grid, bg)
    seed_candidates = [comp for comp in comps if comp["area"] == 1]
    assert len(seed_candidates) == 1, "expected exactly one seed pixel"

    seed_comp = seed_candidates[0]
    sr, sc = seed_comp["coords"][0]
    seed_color = int(seed_comp["color"])
    rays = {name: [] for name in DIRS}

    for comp in comps:
        if comp is seed_comp:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == r1 and c0 <= sc <= c1 and r0 != sr:
            direction = "up" if r0 < sr else "down"
            rays[direction].append(
                {
                    "axis": r0,
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                }
            )
        elif c0 == c1 and r0 <= sr <= r1 and c0 != sc:
            direction = "left" if c0 < sc else "right"
            rays[direction].append(
                {
                    "axis": c0,
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                }
            )
        else:
            raise AssertionError("every non-seed object must be a single straight bar on one ray path")

    rays["up"].sort(key=lambda bar: bar["axis"], reverse=True)
    rays["down"].sort(key=lambda bar: bar["axis"])
    rays["left"].sort(key=lambda bar: bar["axis"], reverse=True)
    rays["right"].sort(key=lambda bar: bar["axis"])
    return bg, (sr, sc), seed_color, rays


def _simulate(grid: np.ndarray, bg: int, seed: tuple[int, int], seed_color: int, rays: dict[str, list[dict]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    sr, sc = seed
    h, w = grid.shape

    for name, (dr, dc) in DIRS.items():
        current = seed_color
        bars = rays[name]
        bar_idx = 0
        rr, cc = sr + dr, sc + dc

        while 0 <= rr < h and 0 <= cc < w:
            if bar_idx < len(bars):
                bar = bars[bar_idx]
                hit = (dr != 0 and rr == bar["axis"]) or (dc != 0 and cc == bar["axis"])
                if hit:
                    if int(bar["color"]) == current:
                        break
                    for br, bc in bar["coords"]:
                        out[br, bc] = current
                    current = int(bar["color"])
                    bar_idx += 1
                    rr += dr
                    cc += dc
                    continue

            if int(grid[rr, cc]) == bg:
                out[rr, cc] = current
            rr += dr
            cc += dc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Cast four color-swapping rays from the seed so each crossed filter bar trades colors with the ray.

    Input structure:
    - A black-background grid contains one colored seed pixel.
    - Other foreground objects are one-cell-thick straight bars that lie on exactly one axial ray from the seed.
    - Horizontal bars sit above or below the seed column; vertical bars sit left or right of the seed row.

    Rule:
    - Launch one ray upward, downward, leftward, and rightward from the seed, each starting with the seed color.
    - Along a ray, paint only black cells. When the ray reaches a bar of a different color, recolor that whole bar to the ray's current color and continue with the bar's old color.
    - If the ray reaches a bar whose color already matches the ray, it stops before the bar; bars beyond that point stay unchanged.

    Key insight:
    - Each direction is an independent one-dimensional simulation: the visible output is determined by the ordered bar colors along that axis, because passing through a bar swaps the ray color with the bar color while a same-color encounter absorbs the ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, seed, seed_color, rays = _parse_task(g)
    return _simulate(g, bg, seed, seed_color, rays)


def _sample_counts() -> dict[str, int]:
    while True:
        counts = {
            name: random.choices([0, 1, 2, 3], weights=[1, 3, 3, 2])[0]
            for name in DIRS
        }
        total = sum(counts.values())
        if total < 4 or total > 9:
            continue
        if sum(v > 0 for v in counts.values()) < 2:
            continue
        if max(counts.values()) < 2:
            continue
        return counts


def _sample_spaced_positions(lo: int, hi: int, n: int) -> list[int]:
    if n == 0:
        return []
    base = sorted(random.sample(range(lo, hi - n + 2), n))
    return [value + idx for idx, value in enumerate(base)]


def _sample_horizontal_span(seed_col: int, band: int) -> tuple[int, int]:
    length = random.randint(2, 2 * band + 1)
    lo = max(seed_col - band, seed_col - length + 1)
    hi = min(seed_col, seed_col + band - length + 1)
    start = random.randint(lo, hi)
    return start, start + length - 1


def _sample_vertical_span(seed_row: int, band: int) -> tuple[int, int]:
    length = random.randint(2, 2 * band + 1)
    lo = max(seed_row - band, seed_row - length + 1)
    hi = min(seed_row, seed_row + band - length + 1)
    start = random.randint(lo, hi)
    return start, start + length - 1


def _random_other_color(current: int, palette: list[int]) -> int:
    choices = [color for color in palette if color != current]
    return random.choice(choices)


def _build_color_sequence(n: int, seed_color: int, palette: list[int], mode: str) -> list[int]:
    assert n >= 1
    if mode == "direct_stop":
        return [seed_color] + [random.choice(palette) for _ in range(n - 1)]

    seq = []
    current = seed_color
    if mode == "pass_only":
        for _ in range(n):
            nxt = _random_other_color(current, palette)
            seq.append(nxt)
            current = nxt
        return seq

    assert mode == "late_stop"
    stop_idx = random.randint(1, n - 1)
    for _ in range(stop_idx):
        nxt = _random_other_color(current, palette)
        seq.append(nxt)
        current = nxt
    seq.append(current)
    while len(seq) < n:
        seq.append(random.choice(palette))
    return seq


def _render_bar(grid: np.ndarray, bar: dict) -> None:
    color = int(bar["color"])
    if bar["orientation"] == "h":
        grid[int(bar["axis"]), int(bar["start"]) : int(bar["end"]) + 1] = color
    else:
        grid[int(bar["start"]) : int(bar["end"]) + 1, int(bar["axis"])] = color


def _gen_attempt() -> np.ndarray | None:
    counts = _sample_counts()
    h_band = random.randint(2, 3)
    v_band = random.randint(2, 3)

    min_h = max(12, 2 * v_band + 2 * counts["up"] + 2 * counts["down"] + 3)
    min_w = max(12, 2 * h_band + 2 * counts["left"] + 2 * counts["right"] + 3)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)

    row_lo = v_band + 2 * counts["up"] + 1
    row_hi = h - (v_band + 2 * counts["down"] + 2)
    col_lo = h_band + 2 * counts["left"] + 1
    col_hi = w - (h_band + 2 * counts["right"] + 2)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    sr = random.randint(row_lo, row_hi)
    sc = random.randint(col_lo, col_hi)

    palette_size = random.randint(3, 5)
    palette = random.sample(Color.NOT_BLACK, palette_size)
    seed_color = random.choice(palette)

    directions_with_chain = [name for name, n in counts.items() if n >= 2]
    featured = random.choice(directions_with_chain)
    sequences = {}
    for name, n in counts.items():
        if n == 0:
            sequences[name] = []
            continue
        if name == featured:
            mode = "late_stop"
        elif n == 1:
            mode = random.choice(["direct_stop", "pass_only"])
        else:
            mode = random.choices(
                ["direct_stop", "late_stop", "pass_only"],
                weights=[2, 3, 3],
            )[0]
        sequences[name] = _build_color_sequence(n, seed_color, palette, mode)

    bars = []
    for row, color in zip(
        _sample_spaced_positions(1, sr - v_band - 2, counts["up"]),
        sequences["up"],
    ):
        start, end = _sample_horizontal_span(sc, h_band)
        bars.append(
            {
                "direction": "up",
                "orientation": "h",
                "axis": row,
                "start": start,
                "end": end,
                "color": color,
            }
        )
    for row, color in zip(
        _sample_spaced_positions(sr + v_band + 2, h - 2, counts["down"]),
        sequences["down"],
    ):
        start, end = _sample_horizontal_span(sc, h_band)
        bars.append(
            {
                "direction": "down",
                "orientation": "h",
                "axis": row,
                "start": start,
                "end": end,
                "color": color,
            }
        )
    for col, color in zip(
        _sample_spaced_positions(1, sc - h_band - 2, counts["left"]),
        sequences["left"],
    ):
        start, end = _sample_vertical_span(sr, v_band)
        bars.append(
            {
                "direction": "left",
                "orientation": "v",
                "axis": col,
                "start": start,
                "end": end,
                "color": color,
            }
        )
    for col, color in zip(
        _sample_spaced_positions(sc + h_band + 2, w - 2, counts["right"]),
        sequences["right"],
    ):
        start, end = _sample_vertical_span(sr, v_band)
        bars.append(
            {
                "direction": "right",
                "orientation": "v",
                "axis": col,
                "start": start,
                "end": end,
                "color": color,
            }
        )

    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid[sr, sc] = seed_color
    for bar in bars:
        _render_bar(grid, bar)

    if np.count_nonzero(grid != Color.BLACK) <= 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-swap puzzle input")
