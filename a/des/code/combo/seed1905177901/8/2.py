from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TOP = "top"
BOTTOM = "bottom"
LEFT = "left"
RIGHT = "right"

# Generator configuration.
GRID_H_RANGE = (24, 34)
GRID_W_RANGE = (24, 34)
OBJECT_COUNT_RANGE = (2, 4)
BUNDLE_PAD = 2
PLACEMENT_TRIES = 120
GENERATION_ATTEMPTS = 400


def _components_bool(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _strict_mode(values: list[int]) -> int | None:
    if not values:
        return None
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in values:
        if value not in counts:
            order.append(value)
            counts[value] = 0
        counts[value] += 1
    best = order[0]
    best_count = counts[best]
    tie = False
    for value in order[1:]:
        count = counts[value]
        if count > best_count:
            best = value
            best_count = count
            tie = False
        elif count == best_count:
            tie = True
    return None if tie else best


def _window_votes(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> list[int]:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(coords)
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    bg_mask = region == bg

    votes: list[int] = []
    for comp in _components_bool(bg_mask):
        boundary_cells = [
            (rr, cc)
            for rr, cc in comp
            if rr == 0 or rr == bg_mask.shape[0] - 1 or cc == 0 or cc == bg_mask.shape[1] - 1
        ]
        if len(boundary_cells) != 1:
            continue

        abs_r = r0 + boundary_cells[0][0]
        abs_c = c0 + boundary_cells[0][1]
        visible: list[int] = []
        for dr, dc in DIRS4:
            nr, nc = abs_r + dr, abs_c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                continue
            color = int(grid[nr, nc])
            if color != bg and color not in visible:
                visible.append(color)
        if len(visible) == 1:
            votes.append(visible[0])
    return votes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a wall-like component to the majority color seen through its single-cell windows.

    Each monochrome foreground component may enclose one or more azure chambers.
    A chamber counts as a windowed chamber when, inside the component's bounding
    box, its azure area touches the box boundary in exactly one cell. The color
    immediately outside that opening contributes one vote. A component with a
    unique majority vote is repainted to that color; components with no votes or
    a tie are left unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for comp in _foreground_components(g, BG):
        color = int(comp["color"])
        coords = comp["coords"]
        votes = _window_votes(g, coords, BG)
        winner = _strict_mode(votes)
        if winner is None or winner == color:
            continue
        for r, c in coords:
            out[r, c] = winner
    return out


def _outward_delta(side: str) -> tuple[int, int]:
    if side == TOP:
        return -1, 0
    if side == BOTTOM:
        return 1, 0
    if side == LEFT:
        return 0, -1
    return 0, 1


def _rand_color(rng: random.Random, exclude: set[int]) -> int:
    options = [color for color in COLORS if color not in exclude]
    return int(rng.choice(options))


def _sample_style(rng: random.Random) -> str:
    return rng.choice(["plain", "vertical", "horizontal", "cross"])


def _sample_dims(style: str, rng: random.Random) -> tuple[int, int]:
    h_min = 5 if style not in {"horizontal", "cross"} else 7
    w_min = 5 if style not in {"vertical", "cross"} else 7
    h = rng.randint(h_min, 11)
    w = rng.randint(w_min, 11)
    return h, w


def _sample_partitions(h: int, w: int, style: str, rng: random.Random) -> tuple[list[int], list[int]]:
    row_walls: list[int] = []
    col_walls: list[int] = []
    if style in {"horizontal", "cross"}:
        row_walls.append(rng.randint(3, h - 4))
    if style in {"vertical", "cross"}:
        col_walls.append(rng.randint(3, w - 4))
    return row_walls, col_walls


def _chambers(h: int, w: int, row_walls: list[int], col_walls: list[int]) -> list[dict[str, object]]:
    row_lines = [0, *sorted(row_walls), h - 1]
    col_lines = [0, *sorted(col_walls), w - 1]
    chambers: list[dict[str, object]] = []
    for i in range(len(row_lines) - 1):
        rr0 = row_lines[i] + 1
        rr1 = row_lines[i + 1] - 1
        if rr0 > rr1:
            continue
        for j in range(len(col_lines) - 1):
            cc0 = col_lines[j] + 1
            cc1 = col_lines[j + 1] - 1
            if cc0 > cc1:
                continue
            sides: list[str] = []
            if row_lines[i] == 0:
                sides.append(TOP)
            if row_lines[i + 1] == h - 1:
                sides.append(BOTTOM)
            if col_lines[j] == 0:
                sides.append(LEFT)
            if col_lines[j + 1] == w - 1:
                sides.append(RIGHT)
            chambers.append({"rows": (rr0, rr1), "cols": (cc0, cc1), "sides": sides})
    return chambers


def _window_coord(chamber: dict[str, object], side: str, rng: random.Random) -> tuple[int, int]:
    rr0, rr1 = chamber["rows"]
    cc0, cc1 = chamber["cols"]
    if side == TOP:
        return 0, rng.randint(cc0, cc1)
    if side == BOTTOM:
        return rr1 + 1, rng.randint(cc0, cc1)
    if side == LEFT:
        return rng.randint(rr0, rr1), 0
    return rng.randint(rr0, rr1), cc1 + 1


def _window_color_list(window_count: int, object_color: int, rng: random.Random) -> tuple[list[int], int]:
    majority = _rand_color(rng, {BG, object_color})
    if window_count == 1:
        return [majority], majority

    majority_count = rng.randint(window_count // 2 + 1, window_count)
    colors = [majority] * majority_count
    for _ in range(window_count - majority_count):
        colors.append(_rand_color(rng, {BG, object_color, majority}))
    rng.shuffle(colors)
    return colors, majority


def _build_bundle(
    object_color: int,
    changed: bool,
    rng: random.Random,
) -> dict[str, object]:
    style = _sample_style(rng)
    h, w = _sample_dims(style, rng)
    row_walls, col_walls = _sample_partitions(h, w, style, rng)
    chambers = _chambers(h, w, row_walls, col_walls)

    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[h - 1, :] = True
    mask[:, 0] = True
    mask[:, w - 1] = True
    for row in row_walls:
        mask[row, :] = True
    for col in col_walls:
        mask[:, col] = True

    windows: list[tuple[tuple[int, int], str]] = []
    if changed:
        max_windows = min(len(chambers), 4)
        window_count = rng.randint(1, max_windows)
        chosen = rng.sample(chambers, window_count)
        for chamber in chosen:
            side = rng.choice(chamber["sides"])
            wr, wc = _window_coord(chamber, side, rng)
            mask[wr, wc] = False
            windows.append(((wr, wc), side))

    comps = _components_bool(mask)
    assert len(comps) == 1

    window_colors, target_color = _window_color_list(len(windows), object_color, rng) if windows else ([], object_color)
    bundle = np.full((h + 2 * BUNDLE_PAD, w + 2 * BUNDLE_PAD), BG, dtype=int)
    offset = BUNDLE_PAD

    main_coords: list[tuple[int, int]] = []
    for r, c in comps[0]:
        ar, ac = offset + r, offset + c
        bundle[ar, ac] = object_color
        main_coords.append((ar, ac))

    for (window_rc, side), color in zip(windows, window_colors):
        wr, wc = window_rc
        ar, ac = offset + wr, offset + wc
        dr, dc = _outward_delta(side)
        for step in range(1, BUNDLE_PAD + 1):
            bundle[ar + dr * step, ac + dc * step] = color

    return {
        "grid": bundle,
        "main_coords": main_coords,
        "input_color": object_color,
        "target_color": target_color,
    }


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_bundles(
    grid_h: int,
    grid_w: int,
    bundles: list[dict[str, object]],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[list[tuple[int, int]], int, int]]] | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    placements: list[tuple[int, int, int, int]] = []
    metas: list[tuple[list[tuple[int, int]], int, int]] = []

    ordered = sorted(bundles, key=lambda item: item["grid"].shape[0] * item["grid"].shape[1], reverse=True)
    for bundle in ordered:
        local = bundle["grid"]
        bh, bw = local.shape
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(0, grid_h - bh)
            c0 = rng.randint(0, grid_w - bw)
            rect = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            if any(not _rects_separated(rect, other, gap=1) for other in placements):
                continue

            region = grid[r0 : r0 + bh, c0 : c0 + bw]
            if np.any((local != BG) & (region != BG)):
                continue

            region[local != BG] = local[local != BG]
            placements.append(rect)
            abs_main = [(r0 + rr, c0 + cc) for rr, cc in bundle["main_coords"]]
            metas.append((abs_main, int(bundle["input_color"]), int(bundle["target_color"])))
            placed = True
            break
        if not placed:
            return None

    return grid, metas


def _expected_output(
    grid: np.ndarray,
    metas: list[tuple[list[tuple[int, int]], int, int]],
) -> np.ndarray:
    out = grid.copy()
    for coords, _, target_color in metas:
        for r, c in coords:
            out[r, c] = target_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    changed_count = rng.randint(1, object_count)
    changed_flags = [True] * changed_count + [False] * (object_count - changed_count)
    rng.shuffle(changed_flags)

    object_colors = list(rng.sample(COLORS, object_count))
    bundles = [_build_bundle(color, changed, rng) for color, changed in zip(object_colors, changed_flags)]

    min_h = max(bundle["grid"].shape[0] for bundle in bundles) + 2
    min_w = max(bundle["grid"].shape[1] for bundle in bundles) + 2
    grid_h = rng.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    grid_w = rng.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])

    placed = _place_bundles(grid_h, grid_w, bundles, rng)
    if placed is None:
        return None
    grid, metas = placed

    expected = _expected_output(grid, metas)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
