from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
SEED_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _connected_components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neigh = _neighbors8 if conn8 else _neighbors4
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_axis(coords: list[tuple[int, int]]) -> tuple[str, int, tuple[int, int, int, int]]:
    pts = set(coords)
    r0, c0, r1, c1 = _component_bbox(coords)
    candidates: list[tuple[str, int]] = []
    if (c0 + c1) % 2 == 0:
        axis_col = (c0 + c1) // 2
        if all((r, 2 * axis_col - c) in pts for r, c in pts):
            candidates.append(("vertical", axis_col))
    if (r0 + r1) % 2 == 0:
        axis_row = (r0 + r1) // 2
        if all((2 * axis_row - r, c) in pts for r, c in pts):
            candidates.append(("horizontal", axis_row))
    assert len(candidates) == 1, "container must have exactly one symmetry axis"
    kind, axis = candidates[0]
    return kind, axis, (r0, c0, r1, c1)


def _find_opening(
    grid: np.ndarray,
    wall_color: int,
    bbox: tuple[int, int, int, int],
    axis_kind: str,
    axis: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if axis_kind == "vertical":
        for cell, delta in [((r0, axis), (-1, 0)), ((r1, axis), (1, 0))]:
            rr, cc = cell
            if c0 <= cc <= c1 and int(grid[rr, cc]) != wall_color:
                candidates.append((cell, delta))
    else:
        for cell, delta in [((axis, c0), (0, -1)), ((axis, c1), (0, 1))]:
            rr, cc = cell
            if r0 <= rr <= r1 and int(grid[rr, cc]) != wall_color:
                candidates.append((cell, delta))
    assert len(candidates) == 1, "expected exactly one notch opening on the symmetry axis"
    return candidates[0]


def _region_from_opening(
    grid: np.ndarray,
    wall_color: int,
    bbox: tuple[int, int, int, int],
    opening: tuple[int, int],
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    q = deque([opening])
    seen = {opening}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) == wall_color:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _branch_color(seeds: list[tuple[int, int, int]]) -> int:
    counts = Counter(color for _r, _c, color in seeds)
    best_n = max(counts.values())
    best_colors = [color for color, n in counts.items() if n == best_n]
    best_pos = {
        color: min((r, c) for r, c, col in seeds if col == color)
        for color in best_colors
    }
    return min(best_colors, key=lambda color: best_pos[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the seed pixels across each symmetric grey container axis, then grow an
    outward branch from the container notch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    values, counts = np.unique(g, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    fg_counts = [(int(v), int(n)) for v, n in zip(values, counts) if int(v) != bg]
    assert fg_counts, "expected non-background content"
    wall_color = max(fg_counts, key=lambda pair: pair[1])[0]

    wall_comps = _connected_components(g == wall_color, conn8=True)
    assert wall_comps, "expected at least one container"

    out = g.copy()
    for comp in wall_comps:
        axis_kind, axis, bbox = _detect_axis(comp)
        opening, delta = _find_opening(g, wall_color, bbox, axis_kind, axis)
        region = _region_from_opening(g, wall_color, bbox, opening)

        seeds = sorted(
            [
                (r, c, int(g[r, c]))
                for r, c in region
                if int(g[r, c]) not in {bg, wall_color}
            ]
        )
        assert 1 <= len(seeds) <= 3, "each container must hold 1-3 seeds"

        for r, c, color in seeds:
            if axis_kind == "vertical":
                mr, mc = r, 2 * axis - c
            else:
                mr, mc = 2 * axis - r, c
            assert 0 <= mr < h and 0 <= mc < w
            out[mr, mc] = color

        branch_color = _branch_color(seeds)
        length = 2 * len(seeds)
        dr, dc = delta
        for step in range(1, length + 1):
            rr = opening[0] + dr * step
            cc = opening[1] + dc * step
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = branch_color

    return out


def _boundary_from_interior(interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    wall = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if interior[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and interior[nr, nc]:
                    wall[r, c] = True
                    break
    return wall


def _coord_transform(
    coord: tuple[int, int],
    shape: tuple[int, int],
    mode: str,
) -> tuple[int, int]:
    r, c = coord
    h, w = shape
    if mode == "identity":
        return r, c
    if mode == "flipud":
        return h - 1 - r, c
    if mode == "transpose":
        return c, r
    if mode == "transpose_fliplr":
        return c, h - 1 - r
    raise ValueError(mode)


def _apply_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return arr.copy()
    if mode == "flipud":
        return np.flipud(arr)
    if mode == "transpose":
        return arr.T.copy()
    if mode == "transpose_fliplr":
        return np.fliplr(arr.T)
    raise ValueError(mode)


def _sample_half_widths(seed_count: int, direction: str) -> list[int] | None:
    for _ in range(80):
        corridor_depth = random.choice((1, 2))
        chamber_rows = random.randint(max(4, seed_count + 1), 7)
        widths = [0] * corridor_depth
        hw = random.randint(1, 3)
        widths.append(hw)
        for _ in range(chamber_rows - 1):
            hw = min(4, max(1, hw + random.choice((-1, 0, 1))))
            widths.append(hw)
        if max(widths) == 1 and random.random() < 0.8:
            idx = random.randrange(corridor_depth, len(widths))
            widths[idx] = 2
        if direction in {"left", "right"} and max(widths) < seed_count:
            continue
        return widths
    return None


def _choose_seed_positions(
    widths: list[int],
    seed_colors: list[int],
    direction: str,
    axis: int,
) -> list[tuple[int, int, int]] | None:
    seed_count = len(seed_colors)
    candidate_rows = list(range(widths.index(next(hw for hw in widths if hw > 0)), len(widths)))
    for _ in range(100):
        rows = random.sample(candidate_rows, seed_count)
        used_offsets: set[int] = set()
        positions: list[tuple[int, int, int]] = []
        side = random.choice(("left", "right"))
        ok = True
        for color, row in zip(seed_colors, rows):
            choices = list(range(1, widths[row] + 1))
            if direction in {"left", "right"}:
                choices = [offset for offset in choices if offset not in used_offsets]
            if not choices:
                ok = False
                break
            offset = random.choice(choices)
            used_offsets.add(offset)
            col = axis - offset if side == "left" else axis + offset
            positions.append((row, col, color))
        if ok:
            return positions
    return None


def _build_container(seed_colors: list[int], direction: str) -> dict | None:
    widths = _sample_half_widths(len(seed_colors), direction)
    if widths is None:
        return None

    max_hw = max(widths)
    inner_h = len(widths)
    h = inner_h + 1
    w = 2 * max_hw + 3
    axis = w // 2

    interior = np.zeros((h, w), dtype=bool)
    for r, hw in enumerate(widths):
        interior[r, axis - hw : axis + hw + 1] = True
    wall_mask = _boundary_from_interior(interior)

    seeds = _choose_seed_positions(widths, seed_colors, direction, axis)
    if seeds is None:
        return None

    tile = np.full((h, w), BG, dtype=int)
    tile[wall_mask] = WALL
    for r, c, color in seeds:
        tile[r, c] = color

    mode_map = {
        "top": ("identity", "vertical", (-1, 0)),
        "bottom": ("flipud", "vertical", (1, 0)),
        "left": ("transpose", "horizontal", (0, -1)),
        "right": ("transpose_fliplr", "horizontal", (0, 1)),
    }
    mode, axis_kind, delta = mode_map[direction]

    transformed_tile = _apply_transform(tile, mode)
    transformed_seeds = [
        (*_coord_transform((r, c), tile.shape, mode), color)
        for r, c, color in seeds
    ]
    mouth = _coord_transform((0, axis), tile.shape, mode)

    output_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(transformed_tile != BG)
    }
    for r, c, _color in transformed_seeds:
        if axis_kind == "vertical":
            output_cells.add((r, 2 * axis - c))
        else:
            output_cells.add((2 * axis - r, c))
    for step in range(1, 2 * len(transformed_seeds) + 1):
        output_cells.add((mouth[0] + delta[0] * step, mouth[1] + delta[1] * step))

    rows = [r for r, _c in output_cells]
    cols = [c for _r, c in output_cells]
    bbox = (min(rows), min(cols), max(rows), max(cols))

    return {
        "tile": transformed_tile,
        "seeds": transformed_seeds,
        "axis_kind": axis_kind,
        "axis": axis,
        "mouth": mouth,
        "delta": delta,
        "out_bbox": bbox,
    }


def _sample_seed_counts(n_containers: int) -> list[int] | None:
    for _ in range(80):
        counts = [random.randint(1, 3) for _ in range(n_containers)]
        if sum(counts) <= len(SEED_PALETTE):
            return counts
    return None


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    occ = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    for spec in sorted(
        specs,
        key=lambda item: (item["out_bbox"][2] - item["out_bbox"][0] + 1)
        * (item["out_bbox"][3] - item["out_bbox"][1] + 1),
        reverse=True,
    ):
        min_r, min_c, max_r, max_c = spec["out_bbox"]
        top_lo = -min_r
        top_hi = h - 1 - max_r
        left_lo = -min_c
        left_hi = w - 1 - max_c
        if top_lo > top_hi or left_lo > left_hi:
            return None

        found = None
        for _ in range(PLACE_ATTEMPTS):
            top = random.randint(top_lo, top_hi)
            left = random.randint(left_lo, left_hi)
            rr0 = max(0, top + min_r - 1)
            rr1 = min(h - 1, top + max_r + 1)
            cc0 = max(0, left + min_c - 1)
            cc1 = min(w - 1, left + max_c + 1)
            if occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            found = dict(spec)
            found["top"] = top
            found["left"] = left
            break
        if found is None:
            return None
        placed.append(found)
    return placed


def _render_input(shape: tuple[int, int], placed_specs: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for spec in placed_specs:
        tile = spec["tile"]
        top = spec["top"]
        left = spec["left"]
        th, tw = tile.shape
        window = grid[top : top + th, left : left + tw]
        mask = tile != BG
        assert not np.any(window[mask] != BG)
        window[mask] = tile[mask]
    return grid


def _expected_output(inp: np.ndarray, placed_specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in placed_specs:
        axis_kind = spec["axis_kind"]
        axis = spec["axis"] + (spec["left"] if axis_kind == "vertical" else spec["top"])
        seeds = [
            (spec["top"] + r, spec["left"] + c, color)
            for r, c, color in spec["seeds"]
        ]
        for r, c, color in seeds:
            if axis_kind == "vertical":
                mr, mc = r, 2 * axis - c
            else:
                mr, mc = 2 * axis - r, c
            out[mr, mc] = color

        mouth_r = spec["top"] + spec["mouth"][0]
        mouth_c = spec["left"] + spec["mouth"][1]
        dr, dc = spec["delta"]
        color = _branch_color(seeds)
        for step in range(1, 2 * len(seeds) + 1):
            out[mouth_r + dr * step, mouth_c + dc * step] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    n_containers = random.randint(1, 3)
    seed_counts = _sample_seed_counts(n_containers)
    if seed_counts is None:
        return None

    all_colors = random.sample(SEED_PALETTE, sum(seed_counts))
    directions = [random.choice(("top", "bottom", "left", "right")) for _ in range(n_containers)]

    specs: list[dict] = []
    idx = 0
    for count, direction in zip(seed_counts, directions):
        spec = _build_container(all_colors[idx : idx + count], direction)
        idx += count
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_specs((h, w), specs)
    if placed is None:
        return None

    inp = _render_input((h, w), placed)
    expected = _expected_output(inp, placed)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
