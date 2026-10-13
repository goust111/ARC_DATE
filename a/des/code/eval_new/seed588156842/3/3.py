from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
MAROON = 9

LEGEND_POOL = (1, 2, 3, 4, 6, 8)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _find_separator_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == GREY))]
    assert rows, "expected a full grey separator row"
    sep = int(rows[0])
    assert sep > 0, "separator must have a legend row above it"
    return sep


def _legend_index(g: np.ndarray, sep: int) -> dict[int, int]:
    legend_row = g[sep - 1]
    cols = [c for c in range(g.shape[1]) if int(legend_row[c]) not in (BLACK, GREY)]
    colors = [int(legend_row[c]) for c in cols]
    assert colors, "expected a non-empty legend band"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return {color: i + 1 for i, color in enumerate(colors)}


def _detect_fences(g: np.ndarray) -> list[dict[str, object]]:
    mask = (g == MAROON) | (g == ORANGE)
    fences: list[dict[str, object]] = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _rect_border_coords(r0, c0, r1, c1)
        border_set = set(border)
        coord_set = {tuple(map(int, p)) for p in coords.tolist()}
        if coord_set != border_set:
            continue
        if any(int(g[r, c]) not in (MAROON, ORANGE) for r, c in border):
            continue

        exits = [(r, c) for r, c in border if int(g[r, c]) == ORANGE]
        assert len(exits) == 1, "each fence must have exactly one orange exit"
        exit_r, exit_c = exits[0]

        side_dirs = []
        if exit_r == r0:
            side_dirs.append((-1, 0))
        if exit_r == r1:
            side_dirs.append((1, 0))
        if exit_c == c0:
            side_dirs.append((0, -1))
        if exit_c == c1:
            side_dirs.append((0, 1))
        assert len(side_dirs) == 1, "exit marker must sit on one non-corner side"
        direction = side_dirs[0]

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere(inner != BLACK)
        assert seed_coords.shape[0] == 1, "each fence must contain exactly one seed pixel"
        seed_r = int(r0 + 1 + seed_coords[0, 0])
        seed_c = int(c0 + 1 + seed_coords[0, 1])
        seed_color = int(g[seed_r, seed_c])
        assert seed_color not in (BLACK, GREY, ORANGE, MAROON)

        dr, dc = direction
        if dr != 0:
            assert seed_c == exit_c, "vertical exit must align with the seed column"
            assert (exit_r - seed_r) * dr > 0, "vertical exit must lie ahead of the seed"
        else:
            assert seed_r == exit_r, "horizontal exit must align with the seed row"
            assert (exit_c - seed_c) * dc > 0, "horizontal exit must lie ahead of the seed"

        fences.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed": (seed_r, seed_c),
                "seed_color": seed_color,
                "exit": (exit_r, exit_c),
                "dir": direction,
            }
        )

    assert fences, "expected at least one rectangular fence"
    return fences


def _paint_ray(
    out: np.ndarray,
    seed: tuple[int, int],
    direction: tuple[int, int],
    exit_pos: tuple[int, int],
    color: int,
    steps: int,
) -> None:
    h, w = out.shape
    rr, cc = seed
    dr, dc = direction
    painted = 0
    while painted < steps:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if (rr, cc) == exit_pos:
            continue
        out[rr, cc] = color
        painted += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top legend order to choose each seed's ray length, then grow that seed along its fence's marked exit direction.

    Input structure:
    - bg = black. The row just above a full-width grey separator is a horizontal legend band of unique colors.
    - Below the separator are one or more axis-aligned maroon rectangular fences with black interiors.
    - Each fence contains exactly one colored seed pixel whose color appears in the legend.
    - Each fence border has exactly one orange replacement pixel marking the exit side.

    Rule:
    - Read the legend colors from left to right and map them to lengths 1, 2, 3, ...
    - For each fence, find its seed color, look up that legend position N, and read the exit side from the orange border marker.
    - Restore the orange marker back to maroon, then extend a straight ray of the seed color from the seed in that direction for N painted cells, skipping over the restored exit cell and stopping early only at the grid boundary.

    Key insight:
    - The puzzle combines two independent encodings: legend position gives the ray length, while the orange border marker gives only the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    legend = _legend_index(g, sep)
    fences = _detect_fences(g)

    out = g.copy()
    out[out == ORANGE] = MAROON

    for fence in fences:
        seed_color = int(fence["seed_color"])
        assert seed_color in legend, "seed color must appear in the legend"
        _paint_ray(
            out,
            seed=tuple(fence["seed"]),
            direction=tuple(fence["dir"]),
            exit_pos=tuple(fence["exit"]),
            color=seed_color,
            steps=int(legend[seed_color]),
        )

    return out


def _trace_path(
    h: int,
    w: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
    exit_pos: tuple[int, int],
) -> tuple[list[tuple[int, int]], int]:
    rr, cc = seed
    dr, dc = direction
    path: list[tuple[int, int]] = []
    paintable = 0
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        path.append((rr, cc))
        if (rr, cc) != exit_pos:
            paintable += 1
    return path, paintable


def _mark_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(mask.shape[0] - 1, r1)
    cc1 = min(mask.shape[1] - 1, c1)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _candidate_positions(limit: int, span: int, direction: tuple[int, int], clipped: bool) -> list[int]:
    if span > limit:
        return []
    hi = limit - span
    if hi < 0:
        return []
    if direction == (-1, 0):
        anchor = range(2, min(hi, 7) + 1) if clipped else range(max(2, hi // 3), hi + 1)
    elif direction == (1, 0):
        anchor = range(max(2, hi - 7), hi + 1) if clipped else range(2, max(2, (2 * hi) // 3) + 1)
    elif direction == (0, -1):
        anchor = range(0, min(hi, 7) + 1) if clipped else range(max(0, hi // 3), hi + 1)
    else:
        anchor = range(max(0, hi - 7), hi + 1) if clipped else range(0, max(0, (2 * hi) // 3) + 1)
    values = list(anchor)
    if not values:
        values = list(range(0, hi + 1))
    random.shuffle(values)
    return values


def _build_expected(inp: np.ndarray, plans: list[dict[str, object]], legend_index: dict[int, int]) -> np.ndarray:
    out = inp.copy()
    out[out == ORANGE] = MAROON
    for plan in plans:
        color = int(plan["seed_color"])
        _paint_ray(
            out,
            seed=tuple(plan["seed"]),
            direction=tuple(plan["dir"]),
            exit_pos=tuple(plan["exit"]),
            color=color,
            steps=int(legend_index[color]),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.randint(3, len(LEGEND_POOL))
    legend_colors = random.sample(list(LEGEND_POOL), legend_len)
    legend_index = {color: i + 1 for i, color in enumerate(legend_colors)}

    num_fences = random.randint(2, 5)
    min_side = max(16, 8 + 3 * num_fences)
    h = random.randint(min_side, 30)
    w = random.randint(max(min_side, legend_len + 4), 30)

    grid = np.full((h, w), BLACK, dtype=int)
    legend_start = random.randint(0, w - legend_len)
    grid[0, legend_start : legend_start + legend_len] = legend_colors
    grid[1, :] = GREY

    directions = DIRECTIONS.copy()
    random.shuffle(directions)
    while len(directions) < num_fences:
        directions.append(random.choice(DIRECTIONS))
    directions = directions[:num_fences]
    random.shuffle(directions)

    clip_flags = [True] * (num_fences // 2) + [False] * (num_fences - num_fences // 2)
    random.shuffle(clip_flags)

    frame_blocked = np.zeros((h, w), dtype=bool)
    path_blocked = np.zeros((h, w), dtype=bool)
    frame_blocked[0:2, :] = True

    plans: list[dict[str, object]] = []
    for direction, clipped in zip(directions, clip_flags):
        placed = False
        allowed_colors = legend_colors[1:] if clipped and legend_len > 1 else legend_colors
        if not allowed_colors:
            return None

        for _ in range(220):
            seed_color = random.choice(allowed_colors)
            steps = legend_index[seed_color]
            if clipped and steps == 1:
                continue

            outer_h = random.randint(4, 8)
            outer_w = random.randint(4, 8)
            if outer_h >= h - 2 or outer_w >= w:
                continue

            inner_h = outer_h - 2
            inner_w = outer_w - 2
            r_positions = _candidate_positions(h, outer_h, direction, clipped)
            c_positions = _candidate_positions(w, outer_w, direction, clipped)
            if not r_positions or not c_positions:
                continue

            candidate_budget = min(160, max(40, len(r_positions) + len(c_positions)))
            for _cand in range(candidate_budget):
                seed_ri = random.randrange(inner_h)
                seed_ci = random.randrange(inner_w)
                r0 = random.choice(r_positions)
                c0 = random.choice(c_positions)
                r1 = r0 + outer_h - 1
                c1 = c0 + outer_w - 1
                if bool(np.any(frame_blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)])):
                    continue
                if bool(np.any(path_blocked[r0 : r1 + 1, c0 : c1 + 1])):
                    continue

                seed_r = r0 + 1 + seed_ri
                seed_c = c0 + 1 + seed_ci
                dr, dc = direction
                if direction == (-1, 0):
                    exit_pos = (r0, seed_c)
                elif direction == (1, 0):
                    exit_pos = (r1, seed_c)
                elif direction == (0, -1):
                    exit_pos = (seed_r, c0)
                else:
                    exit_pos = (seed_r, c1)

                path, paintable = _trace_path(h, w, (seed_r, seed_c), direction, exit_pos)
                if clipped:
                    if not (1 <= paintable < steps):
                        continue
                else:
                    if paintable < steps:
                        continue

                border_set = set(_rect_border_coords(r0, c0, r1, c1))
                if any(bool(frame_blocked[rr, cc]) or bool(path_blocked[rr, cc]) for rr, cc in path if (rr, cc) not in border_set):
                    continue

                grid[r0, c0 : c1 + 1] = MAROON
                grid[r1, c0 : c1 + 1] = MAROON
                grid[r0 : r1 + 1, c0] = MAROON
                grid[r0 : r1 + 1, c1] = MAROON
                grid[exit_pos] = ORANGE
                grid[seed_r, seed_c] = seed_color

                _mark_box(frame_blocked, r0 - 1, c0 - 1, r1 + 1, c1 + 1)
                _mark_box(path_blocked, r0 - 1, c0 - 1, r1 + 1, c1 + 1)
                for rr, cc in path:
                    path_blocked[rr, cc] = True

                plans.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "seed": (seed_r, seed_c),
                        "seed_color": seed_color,
                        "exit": exit_pos,
                        "dir": direction,
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _build_expected(grid, plans, legend_index)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
