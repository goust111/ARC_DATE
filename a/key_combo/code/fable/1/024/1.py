from __future__ import annotations

import numpy as np


# Lightweight helpers adapted from the shared arc_utils toolbox.
BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()

GRID_H_RANGE = (14, 25)  # upper-exclusive
GRID_W_RANGE = (14, 25)  # upper-exclusive
N_COLORS_RANGE = (3, 6)  # upper-exclusive
STEP_CAP = 6
MAX_ATTEMPTS = 400
MAX_OUTPUT_DENSITY = 0.42
MAX_CONTEST_RATIO = 0.35


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _is_corner(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _marker_side(shape: tuple[int, int], marker: tuple[int, int]) -> str:
    h, w = shape
    r, c = marker
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("marker must lie on the border")


def _inward_length(shape: tuple[int, int], marker: tuple[int, int]) -> int:
    h, w = shape
    side = _marker_side(shape, marker)
    if side in {"top", "bottom"}:
        return h - 1
    return w - 1


def _ray_cells(
    shape: tuple[int, int], marker: tuple[int, int], step: int
) -> list[tuple[int, int, int]]:
    h, w = shape
    r0, c0 = marker
    side = _marker_side(shape, marker)
    if side == "top":
        dr, dc = 1, 0
    elif side == "bottom":
        dr, dc = -1, 0
    elif side == "left":
        dr, dc = 0, 1
    else:
        dr, dc = 0, -1

    cells: list[tuple[int, int, int]] = []
    dist = int(step)
    while True:
        rr = r0 + dr * dist
        cc = c0 + dc * dist
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((int(rr), int(cc), int(dist)))
        dist += int(step)
    return cells


def _specs_from_grid(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    specs: list[dict] = []
    for color in colors_present(g, exclude={bg}):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        border = [rc for rc in coords if _is_border(g.shape, rc[0], rc[1])]
        assert len(border) == 1, "each color must have exactly one boundary marker"
        marker = border[0]
        assert not _is_corner(g.shape, marker[0], marker[1]), "markers cannot occupy corners"
        dots = [rc for rc in coords if rc != marker]
        assert dots, "each color needs at least one interior dot"
        specs.append(
            {
                "color": int(color),
                "marker": marker,
                "count": len(dots),
                "dots": dots,
                "side": _marker_side(g.shape, marker),
            }
        )
    assert specs, "expected at least one color"
    return bg, specs


def _solve_from_specs(
    shape: tuple[int, int], specs: list[dict], bg: int = BG
) -> tuple[np.ndarray, dict]:
    out = np.full(shape, int(bg), dtype=int)
    markers = {tuple(spec["marker"]): int(spec["color"]) for spec in specs}
    for marker, color in markers.items():
        out[marker] = color

    best: dict[tuple[int, int], tuple[int, int]] = {}
    claims: dict[tuple[int, int], list[tuple[int, int]]] = {}
    tie_found = False

    for spec in specs:
        color = int(spec["color"])
        step = int(spec["count"])
        for r, c, dist in _ray_cells(shape, tuple(spec["marker"]), step):
            if (r, c) in markers:
                continue
            claims.setdefault((r, c), []).append((dist, color))
            prev = best.get((r, c))
            if prev is None or dist < prev[0]:
                best[(r, c)] = (dist, color)
            elif dist == prev[0] and prev[1] != color:
                tie_found = True

    for (r, c), (_dist, color) in best.items():
        out[r, c] = color

    contested = 0
    multi_color_contested = 0
    for claim_list in claims.values():
        if len(claim_list) > 1:
            contested += 1
            if len({color for _dist, color in claim_list}) > 1:
                multi_color_contested += 1

    stats = {
        "markers": len(markers),
        "painted": len(best),
        "contested": contested,
        "multi_color_contested": multi_color_contested,
        "tie": tie_found,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert boundary source pixels into periodic inward rays whose cadence is set by same-color interior dot counts.

    Input structure:
    - A black-background grid contains several colors, each with exactly one single-pixel marker on one border but not in a corner.
    - The same colors also appear as isolated interior dots; each color's dot count may differ.
    - No other foreground structures are present.

    Rule:
    - For each color, identify its unique border marker and count the remaining same-color interior dots, then erase those dots.
    - From the marker, cast a straight ray perpendicular to its border and paint that color at distances k, 2k, 3k, ... where k is the counted dot total.
    - If multiple rays claim one cell, keep the color whose source marker is closer; border markers themselves remain unchanged.

    Key insight:
    - The interior dots are not preserved geometry: they only encode each border marker's period, and all ray conflicts resolve purely by source distance rather than by processing order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg, specs = _specs_from_grid(g)
    out, stats = _solve_from_specs(g.shape, specs, bg=bg)
    assert not stats["tie"], "equal-distance conflicts are outside this task family"
    return out


def _distance_to_intersection(
    shape: tuple[int, int], spec: dict, row: int, col: int
) -> int:
    marker = tuple(spec["marker"])
    side = spec["side"]
    if side == "top":
        return int(row - marker[0])
    if side == "bottom":
        return int(marker[0] - row)
    if side == "left":
        return int(col - marker[1])
    return int(marker[1] - col)


def _pick_mode(rng: np.random.Generator) -> str:
    return str(rng.choice(["cross", "vertical_only", "horizontal_only"], p=[0.55, 0.225, 0.225]))


def _sample_sides(mode: str, n_colors: int, rng: np.random.Generator) -> list[str]:
    if mode == "cross":
        sides = [
            str(rng.choice(["top", "bottom"])),
            str(rng.choice(["left", "right"])),
        ]
        sides.extend(str(rng.choice(["top", "bottom", "left", "right"])) for _ in range(n_colors - 2))
        rng.shuffle(sides)
        return sides
    if mode == "vertical_only":
        return [str(rng.choice(["top", "bottom"])) for _ in range(n_colors)]
    return [str(rng.choice(["left", "right"])) for _ in range(n_colors)]


def _sample_unique_positions(
    values: list[int], count: int, rng: np.random.Generator
) -> list[int]:
    assert count <= len(values)
    chosen = rng.choice(values, size=count, replace=False)
    return [int(v) for v in chosen]


def _random_step(max_step: int, rng: np.random.Generator) -> int:
    max_step = int(max_step)
    assert max_step >= 1
    capped = min(STEP_CAP, max_step)
    preferred = [k for k in range(2, capped + 1)]
    if preferred and rng.random() < 0.85:
        return int(rng.choice(preferred))
    return int(rng.integers(1, capped + 1))


def _divisors_upto(value: int, max_step: int) -> list[int]:
    return [k for k in range(1, int(max_step) + 1) if int(value) % k == 0]


def _random_divisor(distance: int, max_step: int, rng: np.random.Generator) -> int:
    divisors = _divisors_upto(distance, max_step)
    nontrivial = [k for k in divisors if k > 1]
    if nontrivial and rng.random() < 0.8:
        divisors = nontrivial
    return int(rng.choice(divisors))


def _assign_markers(
    shape: tuple[int, int], colors: list[int], sides: list[str], rng: np.random.Generator
) -> list[dict]:
    h, w = shape
    specs = [{"color": int(color), "side": side} for color, side in zip(colors, sides, strict=True)]

    vertical_indices = [i for i, spec in enumerate(specs) if spec["side"] in {"top", "bottom"}]
    horizontal_indices = [i for i, spec in enumerate(specs) if spec["side"] in {"left", "right"}]

    cols = _sample_unique_positions(list(range(1, w - 1)), len(vertical_indices), rng)
    rows = _sample_unique_positions(list(range(1, h - 1)), len(horizontal_indices), rng)

    for idx, col in zip(vertical_indices, cols, strict=True):
        side = specs[idx]["side"]
        specs[idx]["marker"] = (0, col) if side == "top" else (h - 1, col)
    for idx, row in zip(horizontal_indices, rows, strict=True):
        side = specs[idx]["side"]
        specs[idx]["marker"] = (row, 0) if side == "left" else (row, w - 1)

    return specs


def _assign_counts(
    shape: tuple[int, int], specs: list[dict], mode: str, rng: np.random.Generator
) -> bool:
    for spec in specs:
        spec["count"] = None

    if mode == "cross":
        vertical = [spec for spec in specs if spec["side"] in {"top", "bottom"}]
        horizontal = [spec for spec in specs if spec["side"] in {"left", "right"}]
        if not vertical or not horizontal:
            return False
        v_spec = vertical[int(rng.integers(len(vertical)))]
        h_spec = horizontal[int(rng.integers(len(horizontal)))]
        row = int(h_spec["marker"][0])
        col = int(v_spec["marker"][1])
        v_dist = _distance_to_intersection(shape, v_spec, row, col)
        h_dist = _distance_to_intersection(shape, h_spec, row, col)
        if v_dist <= 0 or h_dist <= 0 or v_dist == h_dist:
            return False

        v_cap = min(STEP_CAP, _inward_length(shape, tuple(v_spec["marker"])))
        h_cap = min(STEP_CAP, _inward_length(shape, tuple(h_spec["marker"])))
        v_spec["count"] = _random_divisor(v_dist, v_cap, rng)
        h_spec["count"] = _random_divisor(h_dist, h_cap, rng)

    for spec in specs:
        if spec["count"] is None:
            max_step = min(STEP_CAP, _inward_length(shape, tuple(spec["marker"])))
            spec["count"] = _random_step(max_step, rng)
    return True


def _ray_cells_exact(shape: tuple[int, int], specs: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for spec in specs:
        for r, c, _dist in _ray_cells(shape, tuple(spec["marker"]), int(spec["count"])):
            cells.add((r, c))
    return cells


def _can_place_dot(
    cell: tuple[int, int], occupied: set[tuple[int, int]], forbidden: set[tuple[int, int]]
) -> bool:
    if cell in forbidden:
        return False
    r, c = cell
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in occupied:
                return False
    return True


def _place_dots(
    grid: np.ndarray, specs: list[dict], rng: np.random.Generator
) -> bool:
    shape = grid.shape
    ray_cells = _ray_cells_exact(shape, specs)
    marker_cells = {tuple(spec["marker"]) for spec in specs}
    forbidden = set(ray_cells) | marker_cells
    occupied = set(marker_cells)
    interior = [(r, c) for r in range(1, shape[0] - 1) for c in range(1, shape[1] - 1)]

    for spec in sorted(specs, key=lambda item: int(item["count"]), reverse=True):
        spec["dots"] = []
        candidates = interior.copy()
        rng.shuffle(candidates)
        for cell in candidates:
            if len(spec["dots"]) >= int(spec["count"]):
                break
            if _can_place_dot(cell, occupied, forbidden):
                rr, cc = cell
                grid[rr, cc] = int(spec["color"])
                spec["dots"].append(cell)
                occupied.add(cell)
        if len(spec["dots"]) != int(spec["count"]):
            return False
    return True


def _validate_specs_against_grid(grid: np.ndarray, specs: list[dict]) -> bool:
    bg, parsed = _specs_from_grid(grid)
    if bg != BG:
        return False

    expected_map = {
        int(spec["color"]): (tuple(spec["marker"]), int(spec["count"]), sorted(spec["dots"]))
        for spec in specs
    }
    parsed_map = {
        int(spec["color"]): (tuple(spec["marker"]), int(spec["count"]), sorted(spec["dots"]))
        for spec in parsed
    }
    if expected_map != parsed_map:
        return False

    foreground = {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}
    markers = {tuple(spec["marker"]) for spec in specs}
    for cell in foreground - markers:
        r, c = cell
        if _is_border(grid.shape, r, c):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) == cell:
                    continue
                if (rr, cc) in foreground:
                    return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    n_colors = int(rng.integers(*N_COLORS_RANGE))
    mode = _pick_mode(rng)

    colors = [int(c) for c in rng.choice(PALETTE, size=n_colors, replace=False)]
    sides = _sample_sides(mode, n_colors, rng)
    specs = _assign_markers((h, w), colors, sides, rng)
    if not _assign_counts((h, w), specs, mode, rng):
        return None

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        grid[tuple(spec["marker"])] = int(spec["color"])

    if not _place_dots(grid, specs, rng):
        return None
    if not _validate_specs_against_grid(grid, specs):
        return None

    expected, stats = _solve_from_specs(grid.shape, specs, bg=BG)
    if stats["tie"]:
        return None
    if mode == "cross" and stats["multi_color_contested"] < 1:
        return None
    if mode != "cross" and stats["contested"] != 0:
        return None
    if stats["painted"] < len(specs):
        return None

    output_density = np.count_nonzero(expected != BG) / expected.size
    if output_density > MAX_OUTPUT_DENSITY:
        return None
    if stats["painted"] > 0 and stats["contested"] / stats["painted"] > MAX_CONTEST_RATIO:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid boundary-sowing input")
