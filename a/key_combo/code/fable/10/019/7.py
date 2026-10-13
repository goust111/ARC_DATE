from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
CORNER_DIRS = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
CORNER_OFFSETS = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}

MIN_SIZE = 14
MAX_SIZE = 27
RECT_H_RANGE = (3, 7)
RECT_W_RANGE = (4, 8)
MIN_RAY_LEN = 4
MAX_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _rect_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _sign_against_center(value: int, center: float) -> int:
    if value < center:
        return -1
    if value > center:
        return 1
    return 0


def _seed_kind(seed: tuple[int, int], center: tuple[float, float], direction: tuple[int, int]) -> str:
    r, c = seed
    center_r, center_c = center
    rel = (_sign_against_center(r, center_r), _sign_against_center(c, center_c))
    if rel == (-direction[0], -direction[1]):
        return "opposite"
    if rel != direction:
        return "mismatch"
    return "matching"


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    gray_coords = np.argwhere(g == GREY)
    assert gray_coords.size > 0, "expected one grey rectangle"
    r0, c0 = gray_coords.min(axis=0).astype(int).tolist()
    r1, c1 = gray_coords.max(axis=0).astype(int).tolist()
    bbox = (r0, c0, r1, c1)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert gray_coords.shape[0] == area, "grey pixels must form one solid rectangle"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey bbox must be fully filled"

    markers: dict[str, tuple[int, int]] = {}
    color_to_dir: dict[int, tuple[int, int]] = {}
    marker_colors: list[int] = []
    for name, (dr, dc) in CORNER_OFFSETS.items():
        mr = r0 - 1 if dr < 0 else r1 + 1
        mc = c0 - 1 if dc < 0 else c1 + 1
        assert _in_bounds((h, w), mr, mc), "legend markers must lie on-grid"
        color = int(g[mr, mc])
        assert color not in {BLACK, GREY}, "legend markers must be colored"
        markers[name] = (mr, mc)
        color_to_dir[color] = CORNER_DIRS[name]
        marker_colors.append(color)
    assert len(set(marker_colors)) == 4, "legend colors must be pairwise distinct"

    marker_cells = set(markers.values())
    rect = _rect_cells(bbox)
    seeds: list[tuple[int, int]] = []
    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        pos = (int(r), int(c))
        if pos not in marker_cells:
            seeds.append(pos)

    assert seeds, "expected at least one isolated seed pixel"
    for r, c in seeds:
        assert int(g[r, c]) in color_to_dir, "seed color must appear in the legend"
        for nr, nc in _neighbors8(r, c):
            if _in_bounds((h, w), nr, nc):
                assert g[nr, nc] == BLACK, "seed pixels must be isolated in the input"

    blocked = rect | marker_cells
    seen_path: set[tuple[int, int]] = set()
    for r, c in seeds:
        direction = color_to_dir[int(g[r, c])]
        path = _ray_cells((h, w), (r, c), direction)
        assert len(path) >= 2, "every seed must have a visible ray"
        assert blocked.isdisjoint(path), "seed rays may not hit the legend"
        assert seen_path.isdisjoint(path), "seed rays may not intersect each other"
        seen_path.update(path)

    return {
        "bbox": bbox,
        "rect": rect,
        "markers": markers,
        "marker_cells": marker_cells,
        "color_to_dir": color_to_dir,
        "seeds": seeds,
    }


def _oracle_output(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in scene["seeds"]:
        color = int(out[r, c])
        dr, dc = scene["color_to_dir"][color]
        rr, cc = r, c
        while _in_bounds(out.shape, rr, cc):
            out[rr, cc] = color
            rr += dr
            cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the grey legend unchanged and extend each same-colored seed along the legend's matching diagonal direction.

    Input structure:
    - The background is black.
    - Exactly one solid grey rectangle appears on the grid.
    - Four differently colored marker pixels sit one step diagonally outside that rectangle's four corners, and any other foreground pixels are isolated seeds using only those four colors.

    Rule:
    - Preserve the grey rectangle and its four corner markers exactly as they are.
    - For each isolated seed pixel, find the corner marker with the same color and read that marker's corner position as a compass diagonal: top-left, top-right, bottom-left, or bottom-right.
    - Starting from the seed itself, draw a ray of the seed's color in that diagonal direction until the grid edge.

    Key insight:
    - The rectangle is not the canvas to recolor; it is only a legend whose corner-marker positions define a global color-to-direction lookup for the other isolated pixels.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _oracle_output(grid, scene)


def _randint_inclusive(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _sample_legend() -> tuple[int, int, int, int, int, int]:
    h = _randint_inclusive(MIN_SIZE, MAX_SIZE - 1)
    w = _randint_inclusive(MIN_SIZE, MAX_SIZE - 1)
    rect_h = _randint_inclusive(RECT_H_RANGE[0], min(RECT_H_RANGE[1], h - 6))
    rect_w = _randint_inclusive(RECT_W_RANGE[0], min(RECT_W_RANGE[1], w - 6))
    r0 = _randint_inclusive(2, h - rect_h - 3)
    c0 = _randint_inclusive(2, w - rect_w - 3)
    return h, w, r0, c0, rect_h, rect_w


def _choose_seed_specs(color_to_dir: dict[int, tuple[int, int]]) -> list[tuple[int, str]]:
    colors = list(color_to_dir)
    np.random.shuffle(colors)
    n_active = 4 if np.random.rand() < 0.6 else 3
    active = colors[:n_active]
    counts = {color: 1 for color in active}
    extra_budget = _randint_inclusive(1, 2)
    while extra_budget > 0:
        color = active[int(np.random.randint(len(active)))]
        if counts[color] < 2:
            counts[color] += 1
            extra_budget -= 1
        elif all(count == 2 for count in counts.values()):
            break

    opposite_quota = 2 if len(active) == 4 else 1
    opposite_colors = set(active[:opposite_quota])
    specs: list[tuple[int, str]] = []
    for color in active:
        first_kind = "opposite" if color in opposite_colors else "mismatch"
        specs.append((color, first_kind))
        for _ in range(counts[color] - 1):
            specs.append((color, "any"))

    priority = {"opposite": 0, "mismatch": 1, "any": 2}
    specs.sort(key=lambda item: (priority[item[1]], int(np.random.randint(1000))))
    return specs


def _candidate_cells(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    color: int,
    direction: tuple[int, int],
    want_kind: str,
    occupied_input: set[tuple[int, int]],
    used_paths: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], list[tuple[int, int]], str]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
    cells = [(r, c) for r in range(h) for c in range(w)]
    np.random.shuffle(cells)
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]], str]] = []

    for r, c in cells:
        pos = (r, c)
        if pos in occupied_input:
            continue
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
            continue
        if any(_in_bounds(shape, nr, nc) and (nr, nc) in occupied_input for nr, nc in _neighbors8(r, c)):
            continue
        path = _ray_cells(shape, pos, direction)
        if len(path) < MIN_RAY_LEN:
            continue
        if not blocked.isdisjoint(path):
            continue
        if not used_paths.isdisjoint(path):
            continue
        kind = _seed_kind(pos, center, direction)
        if want_kind == "opposite" and kind != "opposite":
            continue
        if want_kind == "mismatch" and kind == "matching":
            continue
        candidates.append((pos, path, kind))

    candidates.sort(key=lambda item: (-len(item[1]), item[2] == "matching"))
    return candidates[:90]


def _place_seeds(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    color_to_dir: dict[int, tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    specs: list[tuple[int, str]],
) -> list[tuple[int, int, int]] | None:
    placements: list[tuple[int, int, int]] = []

    def rec(
        idx: int,
        current_input: set[tuple[int, int]],
        current_paths: set[tuple[int, int]],
    ) -> bool:
        if idx == len(specs):
            return True

        color, want_kind = specs[idx]
        direction = color_to_dir[color]
        candidates = _candidate_cells(
            shape,
            bbox,
            color,
            direction,
            want_kind,
            current_input,
            current_paths,
            blocked,
        )
        for pos, path, _kind in candidates:
            placements.append((pos[0], pos[1], color))
            next_input = set(current_input)
            next_input.add(pos)
            next_paths = set(current_paths)
            next_paths.update(path)
            if rec(idx + 1, next_input, next_paths):
                return True
            placements.pop()
        return False

    ok = rec(0, set(occupied_input), set())
    return placements if ok else None


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
    except AssertionError:
        return False

    bbox = scene["bbox"]
    r0, c0, r1, c1 = bbox
    center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
    seeds = scene["seeds"]
    kinds = [_seed_kind(seed, center, scene["color_to_dir"][int(grid[seed])]) for seed in seeds]
    ray_lengths = [len(_ray_cells(grid.shape, seed, scene["color_to_dir"][int(grid[seed])])) for seed in seeds]
    active_dirs = {scene["color_to_dir"][int(grid[seed])] for seed in seeds}
    used_colors = {int(grid[seed]) for seed in seeds}

    if len(seeds) < 4:
        return False
    if len(active_dirs) < 3:
        return False
    if len(used_colors) < 3:
        return False
    if kinds.count("mismatch") + kinds.count("opposite") < 2:
        return False
    if "opposite" not in kinds:
        return False
    if max(ray_lengths) < 6:
        return False

    expected = _oracle_output(grid, scene)
    if np.array_equal(expected, grid):
        return False
    if not np.all(expected[r0 : r1 + 1, c0 : c1 + 1] == GREY):
        return False
    for mr, mc in scene["marker_cells"]:
        if expected[mr, mc] != grid[mr, mc]:
            return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h, w, r0, c0, rect_h, rect_w = _sample_legend()
    grid = np.full((h, w), BLACK, dtype=int)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    bbox = (r0, c0, r1, c1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    marker_colors = np.random.choice(PALETTE, size=4, replace=False).astype(int).tolist()
    color_to_dir: dict[int, tuple[int, int]] = {}
    marker_cells: set[tuple[int, int]] = set()
    for name, color in zip(("tl", "tr", "bl", "br"), marker_colors):
        mr = r0 - 1 if name[0] == "t" else r1 + 1
        mc = c0 - 1 if name[1] == "l" else c1 + 1
        grid[mr, mc] = color
        marker_cells.add((mr, mc))
        color_to_dir[color] = CORNER_DIRS[name]

    occupied_input = _rect_cells(bbox) | marker_cells
    blocked = set(occupied_input)
    specs = _choose_seed_specs(color_to_dir)
    placements = _place_seeds((h, w), bbox, color_to_dir, occupied_input, blocked, specs)
    if placements is None:
        return None

    for r, c, color in placements:
        grid[r, c] = color

    return grid if _validate_generated_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
