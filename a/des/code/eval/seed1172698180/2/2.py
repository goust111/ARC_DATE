import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _clockwise_path(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    path = [(r0, c) for c in range(c0, c1 + 1)]
    path.extend((r, c1) for r in range(r0 + 1, r1))
    path.extend((r1, c) for c in range(c1, c0 - 1, -1))
    path.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return path


def _side_midpoint(
    side: str, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int]:
    if side == "top":
        return r0, (c0 + c1) // 2
    if side == "right":
        return (r0 + r1) // 2, c1
    if side == "bottom":
        return r1, (c0 + c1) // 2
    return (r0 + r1) // 2, c0


def _seed_side(seed_r: int, seed_c: int, r0: int, c0: int, r1: int, c1: int) -> str:
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    half_h = max(0.5, (r1 - r0 - 1) / 2.0)
    half_w = max(0.5, (c1 - c0 - 1) / 2.0)
    v_bias = abs(seed_r - cr) / half_h
    h_bias = abs(seed_c - cc) / half_w
    if v_bias > h_bias:
        return "top" if seed_r < cr else "bottom"
    if h_bias > v_bias:
        return "left" if seed_c < cc else "right"

    dists = {
        "top": seed_r - (r0 + 1),
        "bottom": (r1 - 1) - seed_r,
        "left": seed_c - (c0 + 1),
        "right": (c1 - 1) - seed_c,
    }
    return min(dists, key=dists.get)


def _legend_sequence(block: np.ndarray) -> list[int]:
    seq: list[int] = []
    for r in range(3):
        for c in range(3):
            if r == 0 and c == 0:
                continue
            seq.append(int(block[r, c]))
    return seq


def _extract_legends(
    grid: np.ndarray, bg: int, side: str
) -> tuple[dict[int, list[int]], np.ndarray] | None:
    h, w = grid.shape
    if side == "bottom":
        if h < 4:
            return None
        strip_mask = np.zeros((h, w), dtype=bool)
        strip_mask[h - 3 :, :] = grid[h - 3 :, :] != bg
        edge_ok = lambda r0, c0, r1, c1: r1 == h - 1  # noqa: E731
    else:
        if w < 4:
            return None
        strip_mask = np.zeros((h, w), dtype=bool)
        strip_mask[:, w - 3 :] = grid[:, w - 3 :] != bg
        edge_ok = lambda r0, c0, r1, c1: c1 == w - 1  # noqa: E731

    legends: dict[int, list[int]] = {}
    legend_mask = np.zeros((h, w), dtype=bool)
    for comp in _components(strip_mask):
        r0, c0, r1, c1 = _bbox(comp)
        if not edge_ok(r0, c0, r1, c1):
            return None
        if r1 - r0 != 2 or c1 - c0 != 2:
            return None
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(block == bg):
            return None
        key = int(block[0, 0])
        if key in legends:
            return None
        legends[key] = _legend_sequence(block)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    if not legends:
        return None
    return legends, legend_mask


def _find_rings(
    grid: np.ndarray,
    bg: int,
    active_mask: np.ndarray,
    legends: dict[int, list[int]],
) -> tuple[list[dict[str, object]], np.ndarray] | None:
    h, w = grid.shape
    explained = np.zeros((h, w), dtype=bool)
    rings: list[dict[str, object]] = []

    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        color_mask = (grid == color) & active_mask
        for comp in _components(color_mask):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = _perimeter_cells(r0, c0, r1, c1)
            if set(comp) != border:
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            nz = np.argwhere(interior != bg)
            if len(nz) != 1:
                continue
            sr = int(r0 + 1 + nz[0, 0])
            sc = int(c0 + 1 + nz[0, 1])
            seed_color = int(grid[sr, sc])
            if seed_color == color or seed_color not in legends:
                return None

            side = _seed_side(sr, sc, r0, c0, r1, c1)
            path = _clockwise_path(r0, c0, r1, c1)
            start = _side_midpoint(side, r0, c0, r1, c1)
            if start not in border:
                return None
            idx = path.index(start)
            rings.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "seed": (sr, sc),
                    "seed_color": seed_color,
                    "path": path[idx:] + path[:idx],
                }
            )
            for rr, cc in border:
                explained[rr, cc] = True
            explained[sr, sc] = True

    if not rings:
        return None
    return rings, explained


def _solve_with_side(grid: np.ndarray, side: str) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    legend_info = _extract_legends(g, bg, side)
    if legend_info is None:
        return None
    legends, legend_mask = legend_info

    active_mask = np.ones((h, w), dtype=bool)
    if side == "bottom":
        active_mask[h - 3 :, :] = False
    else:
        active_mask[:, w - 3 :] = False

    ring_info = _find_rings(g, bg, active_mask, legends)
    if ring_info is None:
        return None
    rings, ring_mask = ring_info
    seed_keys = {int(ring["seed_color"]) for ring in rings}
    if len(seed_keys) != len(rings) or seed_keys != set(legends):
        return None

    explained = legend_mask | ring_mask
    if np.any((g != bg) & ~explained):
        return None

    out = g.copy()
    for ring in rings:
        seq = legends[int(ring["seed_color"])]
        for i, (r, c) in enumerate(ring["path"]):
            out[r, c] = seq[i % len(seq)]
        sr, sc = ring["seed"]
        out[sr, sc] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular ring track with the cyclic legend pattern keyed by its seed color.

    Legends are arranged either in the bottom 3 rows or the rightmost 3 columns.
    The seed's offset inside a ring chooses the starting side; the ring is then painted
    clockwise from that side's midpoint using the legend's 8 visible pattern cells.
    """
    for side in ("bottom", "right"):
        out = _solve_with_side(grid, side)
        if out is not None:
            return out
    raise AssertionError("could not parse legends and rings")


_DIRS = ("top", "right", "bottom", "left")
_MOTIFS_2 = [
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1],
]
_MOTIFS_3 = [
    [0, 1, 2, 1, 2, 0, 2, 1],
    [0, 2, 1, 2, 1, 0, 1, 2],
    [1, 0, 2, 2, 1, 0, 2, 1],
    [2, 1, 0, 1, 0, 2, 0, 1],
]
_RETRIES = 400


def _legend_block(seed_color: int, pattern_pool: list[int]) -> np.ndarray:
    k = random.choice((2, 3))
    colors = random.sample(pattern_pool, k)
    motif = random.choice(_MOTIFS_2 if k == 2 else _MOTIFS_3)
    block = np.zeros((3, 3), dtype=int)
    block[0, 0] = seed_color
    idx = 0
    for r in range(3):
        for c in range(3):
            if r == 0 and c == 0:
                continue
            block[r, c] = colors[motif[idx]]
            idx += 1
    return block


def _ring_dim_pairs(active_h: int, active_w: int, count: int) -> list[tuple[int, int]]:
    area_cap = {1: 121, 2: 81, 3: 63}[count]
    pairs = []
    for rh in (5, 7, 9, 11):
        for rw in (5, 7, 9, 11):
            if rh <= active_h - 2 and rw <= active_w - 2 and rh * rw <= area_cap:
                pairs.append((rh, rw))
    return pairs


def _place_rings(
    grid: np.ndarray,
    active_h: int,
    active_w: int,
    specs: list[dict[str, object]],
) -> bool:
    occ = np.zeros((active_h, active_w), dtype=bool)
    for spec in sorted(specs, key=lambda x: x["dims"][0] * x["dims"][1], reverse=True):
        rh, rw = spec["dims"]
        choices: list[tuple[int, int]] = []
        for r0 in range(1, active_h - rh):
            for c0 in range(1, active_w - rw):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(active_h, r0 + rh + 1)
                cc1 = min(active_w, c0 + rw + 1)
                if occ[rr0:rr1, cc0:cc1].any():
                    continue
                choices.append((r0, c0))
        if not choices:
            return False

        r0, c0 = random.choice(choices)
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        spec["bbox"] = (r0, c0, r1, c1)
        occ[max(0, r0 - 1) : min(active_h, r1 + 2), max(0, c0 - 1) : min(active_w, c1 + 2)] = True

        color = int(spec["track_color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        side = str(spec["start_side"])
        cr = (c0 + c1) // 2
        rr = (r0 + r1) // 2
        if side == "top":
            cols = list(range(max(c0 + 1, cr - 1), min(c1 - 1, cr + 1) + 1))
            seed = (r0 + 1, random.choice(cols))
        elif side == "bottom":
            cols = list(range(max(c0 + 1, cr - 1), min(c1 - 1, cr + 1) + 1))
            seed = (r1 - 1, random.choice(cols))
        elif side == "left":
            rows = list(range(max(r0 + 1, rr - 1), min(r1 - 1, rr + 1) + 1))
            seed = (random.choice(rows), c0 + 1)
        else:
            rows = list(range(max(r0 + 1, rr - 1), min(r1 - 1, rr + 1) + 1))
            seed = (random.choice(rows), c1 - 1)
        spec["seed"] = seed
        grid[seed] = int(spec["seed_color"])
    return True


def _place_legends(
    grid: np.ndarray,
    side: str,
    specs: list[dict[str, object]],
) -> bool:
    h, w = grid.shape
    gap = random.choice((1, 2))
    span = 3 * len(specs) + gap * (len(specs) - 1)
    order = random.sample(range(len(specs)), len(specs))
    if side == "bottom":
        if span > w:
            return False
        start = random.randint(0, w - span)
        r0 = h - 3
        for i, idx in enumerate(order):
            c0 = start + i * (3 + gap)
            block = specs[idx]["legend"]
            grid[r0 : r0 + 3, c0 : c0 + 3] = block
    else:
        if span > h:
            return False
        start = random.randint(0, h - span)
        c0 = w - 3
        for i, idx in enumerate(order):
            r0 = start + i * (3 + gap)
            block = specs[idx]["legend"]
            grid[r0 : r0 + 3, c0 : c0 + 3] = block
    return True


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    bg = _get_bg(grid)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        path = _clockwise_path(r0, c0, r1, c1)
        start = _side_midpoint(str(spec["start_side"]), r0, c0, r1, c1)
        idx = path.index(start)
        seq = _legend_sequence(spec["legend"])
        for i, (r, c) in enumerate(path[idx:] + path[:idx]):
            out[r, c] = seq[i % len(seq)]
        sr, sc = spec["seed"]
        out[sr, sc] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((1, 2, 3))
    side = random.choice(("bottom", "right"))
    legend_gap = 2
    legend_span = 3 * count + legend_gap * (count - 1)

    if side == "bottom":
        active_h = random.randint(10 + 2 * (count - 1), 20)
        active_w = random.randint(max(14, legend_span), 24)
        grid = np.zeros((active_h + 3, active_w), dtype=int)
    else:
        active_h = random.randint(max(14, legend_span), 24)
        active_w = random.randint(10 + 2 * (count - 1), 20)
        grid = np.zeros((active_h, active_w + 3), dtype=int)

    dims = _ring_dim_pairs(active_h, active_w, count)
    if not dims:
        return None

    colors = list(range(1, 10))
    random.shuffle(colors)
    track_colors = colors[:count]
    seed_colors = colors[count : 2 * count]
    pattern_pool = colors[2 * count :]
    directions = random.sample(_DIRS, count)

    specs: list[dict[str, object]] = []
    for i in range(count):
        specs.append(
            {
                "track_color": track_colors[i],
                "seed_color": seed_colors[i],
                "start_side": directions[i],
                "dims": random.choice(dims),
                "legend": _legend_block(seed_colors[i], pattern_pool),
            }
        )

    if not _place_rings(grid, active_h, active_w, specs):
        return None
    if not _place_legends(grid, side, specs):
        return None

    bg = _get_bg(grid)
    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg or bg != 0:
        return None

    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-track puzzle")
