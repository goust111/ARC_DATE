from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME_COLOR = 5
MARKER_COLORS = (1, 2, 4)
SEED_COLOR_BY_RESIDUE = {0: 8, 1: 7}

RADIUS_BOUNDS = (4, 6)
MARGIN_BOUNDS = (2, 6)
SEED_COUNT_BOUNDS = (3, 5)
GEN_ATTEMPTS = 300

_GEN_MODE_CYCLE = ("cross", "vertical_only", "horizontal_only", "mixed_no_cross")
_GEN_CALL_COUNT = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _inside_diamond(r: int, c: int, center_r: int, center_c: int, radius: int) -> bool:
    return abs(int(r) - int(center_r)) + abs(int(c) - int(center_c)) < int(radius)


def _diamond_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int], int, int, int]] = []
    for color in set(map(int, np.unique(grid))) - {int(bg)}:
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w or h % 2 == 0:
            continue

        radius = h // 2
        center_r = int(r0 + radius)
        center_c = int(c0 + radius)
        dists = np.abs(coords[:, 0] - center_r) + np.abs(coords[:, 1] - center_c)
        if coords.shape[0] == 4 * radius and np.all(dists == radius):
            candidates.append((int(color), (r0, c0, r1, c1), center_r, center_c, radius))

    assert len(candidates) == 1, "expected exactly one complete diamond frame"
    return candidates[0]


def _marker_side(
    r: int, c: int, bbox: tuple[int, int, int, int]
) -> str | None:
    r0, c0, r1, c1 = bbox
    sides = []
    if r == r0 - 1 and c0 <= c <= c1:
        sides.append("top")
    if r == r1 + 1 and c0 <= c <= c1:
        sides.append("bottom")
    if c == c0 - 1 and r0 <= r <= r1:
        sides.append("left")
    if c == c1 + 1 and r0 <= r <= r1:
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


def _side_direction(side: str) -> tuple[int, int]:
    if side == "top":
        return 1, 0
    if side == "bottom":
        return -1, 0
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, -1
    raise AssertionError(f"unknown side: {side}")


def _ray_cells(
    grid: np.ndarray,
    start_r: int,
    start_c: int,
    direction: tuple[int, int],
    center_r: int,
    center_c: int,
    radius: int,
    bg: int,
) -> list[tuple[int, int]]:
    height, width = grid.shape
    dr, dc = direction
    rr = int(start_r) + int(dr)
    cc = int(start_c) + int(dc)
    entered_interior = False
    cells: list[tuple[int, int]] = []

    while 0 <= rr < height and 0 <= cc < width:
        inside = _inside_diamond(rr, cc, center_r, center_c, radius)
        if inside:
            entered_interior = True
            if int(grid[rr, cc]) != int(bg):
                break
            cells.append((int(rr), int(cc)))
        elif entered_interior:
            break
        rr += int(dr)
        cc += int(dc)
    return cells


def _open_ray_cells(
    shape: tuple[int, int],
    start_r: int,
    start_c: int,
    direction: tuple[int, int],
    center_r: int,
    center_c: int,
    radius: int,
) -> list[tuple[int, int]]:
    height, width = shape
    dr, dc = direction
    rr = int(start_r) + int(dr)
    cc = int(start_c) + int(dc)
    entered_interior = False
    cells: list[tuple[int, int]] = []

    while 0 <= rr < height and 0 <= cc < width:
        inside = _inside_diamond(rr, cc, center_r, center_c, radius)
        if inside:
            entered_interior = True
            cells.append((int(rr), int(cc)))
        elif entered_interior:
            break
        rr += int(dr)
        cc += int(dc)
    return cells


def _interior_cells(center_r: int, center_c: int, radius: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(center_r - radius + 1, center_r + radius):
        span = radius - 1 - abs(r - center_r)
        for c in range(center_c - span, center_c + span + 1):
            cells.append((int(r), int(c)))
    return cells


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color, bbox, center_r, center_c, radius = _diamond_frame(g, bg)
    markers = []
    seeds = []

    for r_raw, c_raw in np.argwhere((g != bg) & (g != frame_color)):
        r = int(r_raw)
        c = int(c_raw)
        side = _marker_side(r, c, bbox)
        if side is not None:
            markers.append(
                {
                    "coord": (r, c),
                    "color": int(g[r, c]),
                    "side": side,
                    "direction": _side_direction(side),
                }
            )
            continue

        assert _inside_diamond(r, c, center_r, center_c, radius), "unexpected foreground outside the diamond"
        seeds.append((r, c, int(g[r, c])))

    assert markers, "expected at least one side marker"
    assert len(seeds) >= 2, "expected at least two seeds to infer the diagonal residues"

    residue_to_color: dict[int, int] = {}
    for r, c, color in seeds:
        residue = (int(r) + int(c)) % 2
        previous = residue_to_color.get(residue)
        assert previous is None or previous == int(color), "seed colors must agree within each residue class"
        residue_to_color[residue] = int(color)

    assert len(residue_to_color) == 2, "expected both diagonal residues to be visible in the seeds"
    return {
        "bg": int(bg),
        "frame_color": int(frame_color),
        "bbox": bbox,
        "center_r": int(center_r),
        "center_c": int(center_c),
        "radius": int(radius),
        "markers": markers,
        "seeds": seeds,
        "residue_to_color": residue_to_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend inward side rays through a hollow diamond and color the covered cells by diagonal residue.

    Input structure:
    - The background is black, and one complete one-pixel-thick grey diamond frame appears on the grid.
    - One or more colored marker pixels sit exactly one cell outside the diamond bbox on its top, bottom, left, or right side strips.
    - Inside the diamond, a few seed pixels in two colors reveal a parity rule: cells with the same `(row + col) mod 2` share a color.

    Rule:
    - Each outside marker emits a straight ray perpendicular to its bbox side and only affects strictly interior cells of the diamond.
    - A ray covers consecutive interior black cells until an original non-black interior pixel is reached; frame, markers, and seeds stay unchanged.
    - Every newly covered cell is colored from the seed-inferred diagonal residue map instead of from the marker color, and overlapping rays agree because the color depends only on position.

    Key insight:
    - The marker geometry decides which cells belong to the output, but the seed parity pattern decides the color of every painted cell, so both cues are necessary.
    """
    g = np.asarray(grid, dtype=int)
    info = _parse_scene(g)
    out = g.copy()

    planned: set[tuple[int, int]] = set()
    for marker in info["markers"]:
        r, c = marker["coord"]
        cells = _ray_cells(
            g,
            r,
            c,
            marker["direction"],
            info["center_r"],
            info["center_c"],
            info["radius"],
            info["bg"],
        )
        planned.update(cells)

    for r, c in planned:
        out[r, c] = int(info["residue_to_color"][(int(r) + int(c)) % 2])
    return out


def _draw_diamond(grid: np.ndarray, r0: int, c0: int, radius: int) -> None:
    center_r = int(r0 + radius)
    center_c = int(c0 + radius)
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        grid[center_r + dr, center_c - dc] = FRAME_COLOR
        grid[center_r + dr, center_c + dc] = FRAME_COLOR


def _distribute_positions(
    positions: list[int], side_a: str, side_b: str
) -> dict[str, list[int]]:
    if not positions:
        return {side_a: [], side_b: []}
    positions = positions[:]
    random.shuffle(positions)
    n = len(positions)
    splits = [(a, n - a) for a in range(min(2, n) + 1) if 0 <= n - a <= 2]
    count_a, count_b = random.choice(splits)
    return {
        side_a: positions[:count_a],
        side_b: positions[count_a : count_a + count_b],
    }


def _sample_line_sets(
    mode: str,
    v_pool: list[int],
    h_pool: list[int],
    center_r: int,
    center_c: int,
    radius: int,
) -> tuple[list[int], list[int]] | None:
    if mode == "vertical_only":
        count = random.randint(2, 3)
        return random.sample(v_pool, count), []
    if mode == "horizontal_only":
        count = random.randint(2, 3)
        return [], random.sample(h_pool, count)

    for _ in range(120):
        cols = random.sample(v_pool, random.randint(1, 2))
        rows = random.sample(h_pool, random.randint(1, 2))
        has_cross = any(
            abs(row - center_r) + abs(col - center_c) < radius
            for row in rows
            for col in cols
        )
        if mode == "cross" and has_cross:
            return cols, rows
        if mode == "mixed_no_cross" and not has_cross:
            return cols, rows
    return None


def _assign_marker_colors(mode: str, count: int) -> list[int]:
    if count == 1:
        return [random.choice(MARKER_COLORS)]
    if mode == "cross":
        colors = list(random.sample(MARKER_COLORS, 2))
        while len(colors) < count:
            colors.append(random.choice(MARKER_COLORS))
        random.shuffle(colors)
        return colors
    if mode == "vertical_only" and random.random() < 0.5:
        color = random.choice(MARKER_COLORS)
        return [color] * count
    colors = [random.choice(MARKER_COLORS) for _ in range(count)]
    if len(set(colors)) == 1 and random.random() < 0.7:
        idx = random.randrange(count)
        colors[idx] = random.choice([c for c in MARKER_COLORS if c != colors[idx]])
    return colors


def _build_marker_specs(
    mode: str,
    bbox: tuple[int, int, int, int],
    center_r: int,
    center_c: int,
    radius: int,
) -> list[dict] | None:
    r0, c0, r1, c1 = bbox
    v_pool = list(range(c0 + 1, c1))
    h_pool = list(range(r0 + 1, r1))
    sampled = _sample_line_sets(mode, v_pool, h_pool, center_r, center_c, radius)
    if sampled is None:
        return None
    cols, rows = sampled

    markers = []
    for side, positions in _distribute_positions(cols, "top", "bottom").items():
        for col in positions:
            row = r0 - 1 if side == "top" else r1 + 1
            markers.append({"coord": (int(row), int(col)), "side": side})
    for side, positions in _distribute_positions(rows, "left", "right").items():
        for row in positions:
            col = c0 - 1 if side == "left" else c1 + 1
            markers.append({"coord": (int(row), int(col)), "side": side})

    if not markers:
        return None

    colors = _assign_marker_colors(mode, len(markers))
    for marker, color in zip(markers, colors):
        marker["color"] = int(color)
    random.shuffle(markers)
    return markers


def _choose_seeds(
    eligible: list[tuple[int, int]], target_count: int
) -> list[tuple[int, int]] | None:
    even = [cell for cell in eligible if (cell[0] + cell[1]) % 2 == 0]
    odd = [cell for cell in eligible if (cell[0] + cell[1]) % 2 == 1]
    if not even or not odd:
        return None

    for _ in range(120):
        chosen = [random.choice(even), random.choice(odd)]
        blocked = set()
        for r, c in chosen:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    blocked.add((rr, cc))

        pool = eligible[:]
        random.shuffle(pool)
        for cell in pool:
            if cell in chosen or cell in blocked:
                continue
            chosen.append(cell)
            r, c = cell
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    blocked.add((rr, cc))
            if len(chosen) == target_count:
                break

        if len(chosen) < target_count:
            for cell in pool:
                if cell not in chosen:
                    chosen.append(cell)
                if len(chosen) == target_count:
                    break

        if len(chosen) == target_count and {(r + c) % 2 for r, c in chosen} == {0, 1}:
            return chosen
    return None


def _semantic_accepts(grid: np.ndarray) -> bool:
    try:
        info = _parse_scene(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    side_counts = Counter(marker["side"] for marker in info["markers"])
    if any(count > 2 for count in side_counts.values()):
        return False

    ray_cells: set[tuple[int, int]] = set()
    for marker in info["markers"]:
        r, c = marker["coord"]
        ray_cells.update(
            _ray_cells(
                grid,
                r,
                c,
                marker["direction"],
                info["center_r"],
                info["center_c"],
                info["radius"],
                info["bg"],
            )
        )

    if not ray_cells:
        return False

    seed_coords = {(r, c) for r, c, _ in info["seeds"]}
    if ray_cells & seed_coords:
        return False

    changed = {tuple(cell) for cell in np.argwhere(out != grid)}
    if changed != ray_cells:
        return False

    residue_colors = set(info["residue_to_color"].values())
    for r, c in changed:
        if int(out[r, c]) != int(info["residue_to_color"][(r + c) % 2]):
            return False
        if int(out[r, c]) not in residue_colors:
            return False

    interior = set(_interior_cells(info["center_r"], info["center_c"], info["radius"]))
    untouched = interior - ray_cells - seed_coords
    if not untouched:
        return False
    if not any(int(grid[r, c]) == int(info["bg"]) and int(out[r, c]) == int(info["bg"]) for r, c in untouched):
        return False

    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    radius = random.randint(*RADIUS_BOUNDS)
    side = 2 * radius + 1
    top = random.randint(*MARGIN_BOUNDS)
    bottom = random.randint(*MARGIN_BOUNDS)
    left = random.randint(*MARGIN_BOUNDS)
    right = random.randint(*MARGIN_BOUNDS)

    height = side + top + bottom
    width = side + left + right
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    r0 = top
    c0 = left
    _draw_diamond(grid, r0, c0, radius)
    center_r = r0 + radius
    center_c = c0 + radius
    bbox = (r0, c0, r0 + side - 1, c0 + side - 1)

    markers = _build_marker_specs(mode, bbox, center_r, center_c, radius)
    if markers is None:
        return None

    ray_cells: set[tuple[int, int]] = set()
    for marker in markers:
        r, c = marker["coord"]
        side_name = marker["side"]
        grid[r, c] = int(marker["color"])
        ray_cells.update(
            _open_ray_cells(
                grid.shape,
                r,
                c,
                _side_direction(side_name),
                center_r,
                center_c,
                radius,
            )
        )

    if len(ray_cells) < 4:
        return None

    interior = set(_interior_cells(center_r, center_c, radius))
    eligible_seeds = sorted(interior - ray_cells)
    if len(eligible_seeds) < SEED_COUNT_BOUNDS[0] + 1:
        return None

    target_seed_count = random.randint(SEED_COUNT_BOUNDS[0], min(SEED_COUNT_BOUNDS[1], len(eligible_seeds)))
    seeds = _choose_seeds(eligible_seeds, target_seed_count)
    if seeds is None:
        return None

    for r, c in seeds:
        grid[r, c] = int(SEED_COLOR_BY_RESIDUE[(r + c) % 2])

    if not _semantic_accepts(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _GEN_MODE_CYCLE[_GEN_CALL_COUNT % len(_GEN_MODE_CYCLE)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input for mode {mode}")
