from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, in_bounds
except ModuleNotFoundError:
    ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(ARC_UTILS_DIR))
    try:
        from arc_utils import Color, in_bounds
    except ModuleNotFoundError:
        class Color:
            BLACK = 0
            BLUE = 1
            RED = 2
            GREEN = 3
            YELLOW = 4
            PINK = 6
            ORANGE = 7
            TEAL = 8
            MAROON = 9

        def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
            h, w = shape
            return 0 <= r < h and 0 <= c < w


BG = Color.BLACK
INPUT_PALETTE = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
MAX_GEN_ATTEMPTS = 500
RANK_SENTINEL = 10**9


def _mirror_col(c: int, width: int) -> int:
    return width - 1 - c


def _nearest_corner(r: int, c: int, h: int, w: int) -> tuple[int, int, int]:
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    best_r, best_c = min(corners, key=lambda corner: (abs(r - corner[0]) + abs(c - corner[1]), corner[0], corner[1]))
    return best_r, best_c, abs(r - best_r) + abs(c - best_c)


def _paired_sources(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources = []
    for r, c in np.argwhere(g != BG):
        r = int(r)
        c = int(c)
        mc = _mirror_col(c, w)
        if int(g[r, mc]) == BG:
            continue
        corner_r, corner_c, dist = _nearest_corner(r, c, h, w)
        sources.append(
            {
                "r": r,
                "c": c,
                "color": int(g[r, c]),
                "corner": (corner_r, corner_c),
                "dist": dist,
                "rect": (min(r, corner_r), min(c, corner_c), max(r, corner_r), max(c, corner_c)),
            }
        )
    sources.sort(key=lambda src: (src["dist"], src["r"], src["c"]))
    return sources


def _solve_details(grid: np.ndarray) -> tuple[list[dict], np.ndarray, list[list[list[dict]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)
    best_rank = [[(RANK_SENTINEL, RANK_SENTINEL, RANK_SENTINEL) for _ in range(w)] for _ in range(h)]
    cover_lists: list[list[list[dict]]] = [[[] for _ in range(w)] for _ in range(h)]
    sources = _paired_sources(g)

    for src in sources:
        rank = (int(src["dist"]), int(src["r"]), int(src["c"]))
        r0, c0, r1, c1 = src["rect"]
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                cover_lists[rr][cc].append(src)
                if rank < best_rank[rr][cc]:
                    best_rank[rr][cc] = rank
                    out[rr, cc] = int(src["color"])
    return sources, out, cover_lists


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only horizontally mirrored foreground pixels toward their nearest corners, erasing isolated pixels and arbitrating overlaps by corner distance.

    Input structure:
    - The grid has a black background and an even width.
    - Foreground objects are isolated colored pixels using a small 2-3 color palette.
    - Some pixels have a same-row horizontal mirror partner; others are singleton noise pixels.

    Rule:
    - Keep only foreground pixels whose reflected cell across the vertical midline is also foreground.
    - For each kept pixel, find the Manhattan-nearest grid corner, breaking ties by smaller corner row and then smaller corner column.
    - Fill the axis-aligned rectangle having the source pixel and that corner as opposite vertices.
    - When multiple rectangles cover one cell, keep the color of the source with smaller (corner distance, row, column); all unpaired pixels are otherwise erased.

    Key insight:
    - Horizontal mirror symmetry is not the visible output pattern; it is the gate that decides which pixels are allowed to spawn corner-directed rectangle fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] % 2 == 0, "expected an even-width grid"
    _, out, _ = _solve_details(g)
    return out


def _zone_candidates(h: int, w: int, zone: str) -> list[tuple[int, int]]:
    max_left_col = min(3, w // 2 - 2)
    if max_left_col < 1:
        return []
    cols = range(1, max_left_col + 1)
    if zone == "top":
        max_top_row = min(4, h // 2 - 1)
        rows = range(1, max_top_row + 1)
    else:
        min_bottom_row = max(h // 2 + 1, h - 4)
        rows = range(min_bottom_row, h - 1)
    return [(int(r), int(c)) for r in rows for c in cols]


def _zone_distance(h: int, w: int, zone: str, pos: tuple[int, int]) -> int:
    r, c = pos
    if zone == "top":
        return int(r + c)
    return int((h - 1 - r) + c)


def _not_edge_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1


def _pick_zone_positions(h: int, w: int, zone: str, count: int, force_tie: bool) -> list[tuple[int, int]] | None:
    if count == 0:
        return []

    candidates = _zone_candidates(h, w, zone)
    tie_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if force_tie and count >= 2:
        by_dist: dict[int, list[tuple[int, int]]] = {}
        for pos in candidates:
            by_dist.setdefault(_zone_distance(h, w, zone, pos), []).append(pos)
        for group in by_dist.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if a[0] != b[0] and _not_edge_adjacent(a, b):
                        tie_pairs.append((a, b))
        if not tie_pairs:
            return None

    for _ in range(300):
        chosen: list[tuple[int, int]] = []
        used_rows = set()
        if tie_pairs:
            a, b = random.choice(tie_pairs)
            chosen = [a, b]
            used_rows = {a[0], b[0]}

        pool = candidates[:]
        random.shuffle(pool)
        for pos in pool:
            if pos in chosen or pos[0] in used_rows:
                continue
            if any(not _not_edge_adjacent(pos, prev) for prev in chosen):
                continue
            chosen.append(pos)
            used_rows.add(pos[0])
            if len(chosen) == count:
                break

        if len(chosen) != count:
            continue
        if not force_tie and count >= 2:
            dists = {_zone_distance(h, w, zone, pos) for pos in chosen}
            if len(dists) == 1:
                continue
        return chosen
    return None


def _assign_zone_colors(
    positions: list[tuple[int, int]],
    h: int,
    w: int,
    zone: str,
    palette: list[int],
    winner_bias: str | None,
) -> dict[tuple[int, int], int]:
    if not positions:
        return {}

    ordered = sorted(positions, key=lambda pos: (_zone_distance(h, w, zone, pos), pos[0], pos[1]))
    mapping: dict[tuple[int, int], int] = {}

    if len(ordered) == 1:
        mapping[ordered[0]] = random.choice(palette)
        return mapping

    a, b = random.sample(palette, 2)
    low, high = min(a, b), max(a, b)
    if winner_bias == "high":
        winner_color, loser_color = high, low
    elif winner_bias == "low":
        winner_color, loser_color = low, high
    else:
        winner_color, loser_color = (a, b) if random.random() < 0.5 else (b, a)

    mapping[ordered[0]] = winner_color
    mapping[ordered[1]] = loser_color
    for pos in ordered[2:]:
        mapping[pos] = random.choice(palette)
    return mapping


def _place_pair_pixels(
    h: int,
    w: int,
    top_positions: list[tuple[int, int]],
    bottom_positions: list[tuple[int, int]],
    palette: list[int],
) -> np.ndarray:
    if len(top_positions) >= 2 and len(bottom_positions) >= 2:
        if random.random() < 0.5:
            top_bias, bottom_bias = "high", "low"
        else:
            top_bias, bottom_bias = "low", "high"
    else:
        top_bias = random.choice([None, "high", "low"])
        bottom_bias = random.choice([None, "high", "low"])

    colors = {}
    colors.update(_assign_zone_colors(top_positions, h, w, "top", palette, top_bias))
    colors.update(_assign_zone_colors(bottom_positions, h, w, "bottom", palette, bottom_bias))
    if len(set(colors.values())) < 2:
        raise ValueError("need at least two pair colors")

    grid = np.full((h, w), BG, dtype=int)
    for r, c in top_positions + bottom_positions:
        color = int(colors[(r, c)])
        grid[r, c] = color
        grid[r, _mirror_col(c, w)] = color
    return grid


def _noise_candidates(grid: np.ndarray, cover_lists: list[list[list[dict]]]) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    paired_rows = {int(r) for r, _ in np.argwhere(g[:, : w // 2] != BG)}
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    candidates = []
    for r in range(1, h - 1):
        if r in paired_rows:
            continue
        for c in range(1, w - 1):
            mc = _mirror_col(c, w)
            if int(g[r, c]) != BG or int(g[r, mc]) != BG:
                continue
            if cover_lists[r][c]:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    candidates.sort(key=lambda pos: abs(pos[0] - center_r) + abs(pos[1] - center_c))
    return candidates


def _well_spaced(candidate: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    return all(abs(candidate[0] - r) + abs(candidate[1] - c) > 1 for r, c in occupied)


def _add_noise_pixels(grid: np.ndarray, palette: list[int]) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int).copy()
    _, _, cover_lists = _solve_details(g)
    candidates = _noise_candidates(g, cover_lists)
    occupied = {tuple(map(int, coord)) for coord in np.argwhere(g != BG)}
    pair_colors = {int(g[r, c]) for r, c in occupied}
    leftover = [color for color in palette if color not in pair_colors]
    noise_palette = leftover or palette

    target_noise = random.choice([1, 2])
    chosen = []
    for pos in candidates:
        mirror = (pos[0], _mirror_col(pos[1], g.shape[1]))
        if mirror in chosen or mirror in occupied:
            continue
        if not _well_spaced(pos, occupied | set(chosen)):
            continue
        chosen.append(pos)
        if len(chosen) == target_noise:
            break
    if len(chosen) != target_noise:
        return None

    for idx, (r, c) in enumerate(chosen):
        if idx < len(leftover):
            color = int(noise_palette[idx])
        else:
            color = int(random.choice(noise_palette))
        g[r, c] = color
    return g


def _has_visible_overlap(cover_lists: list[list[list[dict]]]) -> bool:
    for row in cover_lists:
        for covers in row:
            if len(covers) >= 2 and len({int(src["color"]) for src in covers}) >= 2:
                return True
    return False


def _has_tie_overlap(cover_lists: list[list[list[dict]]]) -> bool:
    for row in cover_lists:
        for covers in row:
            by_dist: dict[int, set[int]] = {}
            for src in covers:
                by_dist.setdefault(int(src["dist"]), set()).add(int(src["color"]))
            if any(len(colors) >= 2 for colors in by_dist.values()):
                return True
    return False


def _winner_not_always_low(cover_lists: list[list[list[dict]]], out: np.ndarray) -> bool:
    for r, row in enumerate(cover_lists):
        for c, covers in enumerate(row):
            if len(covers) < 2:
                continue
            colors = {int(src["color"]) for src in covers}
            if len(colors) < 2:
                continue
            if int(out[r, c]) != min(colors):
                return True
    return False


def _validate_instance(grid: np.ndarray, require_tie: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if g.ndim != 2 or h > 30 or w > 30 or w % 2 != 0:
        return False

    fg = [tuple(map(int, coord)) for coord in np.argwhere(g != BG)]
    if not 6 <= len(fg) <= 10:
        return False

    colors = sorted({int(g[r, c]) for r, c in fg})
    if len(colors) not in {2, 3}:
        return False

    sources, out, cover_lists = _solve_details(g)
    left_sources = [src for src in sources if int(src["c"]) < w // 2]
    if len(left_sources) not in {3, 4}:
        return False

    corners = {tuple(src["corner"]) for src in sources}
    if corners != {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
        return False

    unpaired = []
    for r, c in fg:
        mc = _mirror_col(c, w)
        if int(g[r, mc]) == BG:
            unpaired.append((r, c))
        elif int(g[r, mc]) != int(g[r, c]):
            return False
    if len(unpaired) not in {1, 2}:
        return False
    if not all(int(out[r, c]) == BG and not cover_lists[r][c] for r, c in unpaired):
        return False

    if not _has_visible_overlap(cover_lists):
        return False
    if require_tie and not _has_tie_overlap(cover_lists):
        return False
    if not _winner_not_always_low(cover_lists, out):
        return False

    out_colors = {int(v) for v in out.ravel() if int(v) != BG}
    if len(out_colors) < 2:
        return False

    filled = int(np.count_nonzero(out))
    if not (12 <= filled <= int(0.55 * h * w)):
        return False

    for r, c in fg:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if in_bounds(g.shape, nr, nc) and (nr, nc) != (r, c) and int(g[nr, nc]) != BG:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 14)
    w = random.choice([10, 12, 14])
    n_pairs = 4 if random.random() < 0.6 else 3
    if n_pairs == 4:
        top_count, bottom_count = random.choice([(2, 2), (2, 2), (3, 1), (1, 3)])
    else:
        top_count, bottom_count = random.choice([(2, 1), (1, 2)])

    palette_size = random.choice([2, 3])
    palette = random.sample(INPUT_PALETTE, palette_size)
    tie_zones = [zone for zone, count in (("top", top_count), ("bottom", bottom_count)) if count >= 2]
    require_tie = bool(tie_zones) and random.random() < 0.65
    tie_zone = random.choice(tie_zones) if require_tie else None

    top_positions = _pick_zone_positions(h, w, "top", top_count, tie_zone == "top")
    bottom_positions = _pick_zone_positions(h, w, "bottom", bottom_count, tie_zone == "bottom")
    if top_positions is None or bottom_positions is None:
        return None

    try:
        pair_grid = _place_pair_pixels(h, w, top_positions, bottom_positions, palette)
    except ValueError:
        return None

    grid = _add_noise_pixels(pair_grid, palette)
    if grid is None:
        return None
    if not _validate_instance(grid, require_tie=require_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-gated nearest-corner fill input")
