from __future__ import annotations

import random

import numpy as np


BG = 0
CROSS = 2
TEMPLATE = 5
SEED_COLORS = (1, 3, 4, 6, 7, 8, 9)

GEN_HEIGHT_RANGE = (11, 20)
GEN_WIDTH_RANGE = (11, 20)
GEN_MIN_QUADRANT = 5
GEN_MAX_MARGIN_FROM_CROSS = 4
GEN_MIN_PATH_LEN = 4
GEN_RETRIES = 400


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    red_rows = np.flatnonzero(np.all(grid == CROSS, axis=1))
    red_cols = np.flatnonzero(np.all(grid == CROSS, axis=0))
    assert len(red_rows) == 1, "expected one full horizontal cross line"
    assert len(red_cols) == 1, "expected one full vertical cross line"
    return int(red_rows[0]), int(red_cols[0])


def _quadrant_bounds(h: int, w: int, cross_r: int, cross_c: int) -> dict[str, tuple[int, int, int, int]]:
    bounds = {
        "TL": (0, cross_r, 0, cross_c),
        "TR": (0, cross_r, cross_c + 1, w),
        "BL": (cross_r + 1, h, 0, cross_c),
        "BR": (cross_r + 1, h, cross_c + 1, w),
    }
    for r0, r1, c0, c1 in bounds.values():
        assert r1 > r0 and c1 > c0, "cross must split the grid into four nonempty quadrants"
    return bounds


def _template_row_index(name: str, height: int) -> int:
    return height - 1 if name[0] == "T" else 0


def _horizontal_neighbor(name: str) -> str:
    return name[0] + ("R" if name[1] == "L" else "L")


def _extract_layout(grid: np.ndarray) -> tuple[dict[str, dict[str, int | tuple[int, int]]], dict[str, np.ndarray]]:
    h, w = grid.shape
    cross_r, cross_c = _find_cross(grid)
    bounds = _quadrant_bounds(h, w, cross_r, cross_c)

    seeds: dict[str, dict[str, int | tuple[int, int]]] = {}
    templates: dict[str, np.ndarray] = {}

    for name, (r0, r1, c0, c1) in bounds.items():
        patch = grid[r0:r1, c0:c1]
        seed_mask = (patch != BG) & (patch != TEMPLATE)
        template_mask = patch == TEMPLATE

        if np.any(seed_mask):
            coords = np.argwhere(seed_mask)
            assert coords.shape[0] == 1, "each seed quadrant must contain exactly one seed"
            assert not np.any(template_mask), "seed quadrants must not also contain template pixels"
            sr, sc = map(int, coords[0])
            seeds[name] = {
                "row": r0 + sr,
                "col": c0 + sc,
                "color": int(patch[sr, sc]),
            }
            continue

        assert np.all((patch == BG) | template_mask), "template quadrants may only contain grey or background"
        boundary_row = _template_row_index(name, patch.shape[0])
        allowed = np.zeros_like(patch, dtype=bool)
        allowed[boundary_row, :] = True
        assert np.array_equal(template_mask, template_mask & allowed), "template must live on the boundary row near the cross"
        seq = template_mask[boundary_row].copy()
        assert np.any(seq), "template row must contain grey pixels"
        templates[name] = seq

    assert len(seeds) == 2 and len(templates) == 2, "expected two seeds and two template quadrants"
    seed_names = set(seeds)
    assert seed_names in ({"TL", "BR"}, {"TR", "BL"}), "seed quadrants must be diagonal"
    return seeds, templates


def _trace_growth(
    start: tuple[int, int],
    template: np.ndarray,
    vertical_step: int,
    horizontal_step: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    path: list[tuple[int, int]] = []
    budget = h * w

    while budget > 0:
        for forward in template:
            nr = r + vertical_step if bool(forward) else r
            nc = c if bool(forward) else c + horizontal_step
            if not (0 <= nr < h and 0 <= nc < w):
                return path
            path.append((nr, nc))
            r, c = nr, nc
            budget -= 1
            if budget == 0:
                break

    raise AssertionError("growth should always terminate at the grid boundary")


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    seeds, templates = _extract_layout(g)

    out = g.copy()
    paths: dict[str, list[tuple[int, int]]] = {}
    for name, seed in seeds.items():
        template = templates[_horizontal_neighbor(name)]
        vertical_step = -1 if name[0] == "T" else 1
        horizontal_step = -1 if name[1] == "L" else 1
        start = (int(seed["row"]), int(seed["col"]))
        path = _trace_growth(start, template, vertical_step, horizontal_step, g.shape)
        paths[name] = path
        for r, c in path:
            assert g[r, c] == BG, "growth path must stay in empty cells"
            out[r, c] = int(seed["color"])
    return out, paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed inside its own quadrant using the grey boundary pattern from
    the horizontally adjacent quadrant as a repeating instruction string.

    Grey template pixels mean "step one cell vertically away from the red cross".
    Background template pixels mean "step one cell sideways away from the red cross".
    """
    out, _paths = _solve(grid)
    return out


def _sample_template(width: int) -> np.ndarray:
    for _ in range(80):
        mode = random.choice(("noise", "runs", "periodic"))

        if mode == "noise":
            seq = np.array([random.random() < 0.5 for _ in range(width)], dtype=bool)
        elif mode == "runs":
            cuts = sorted(
                random.sample(
                    range(1, width),
                    random.randint(1, min(width - 1, 3)),
                )
            )
            seq = np.zeros(width, dtype=bool)
            value = random.choice((False, True))
            prev = 0
            for cut in cuts + [width]:
                seq[prev:cut] = value
                value = not value
                prev = cut
        else:
            period = random.randint(2, min(4, width))
            base = np.array([random.choice((False, True)) for _ in range(period)], dtype=bool)
            seq = np.resize(base, width)

        if np.any(seq) and np.any(~seq):
            return seq

    raise RuntimeError("failed to sample a nondegenerate template")


def _seed_local_position(height: int, width: int, top_half: bool, left_half: bool) -> tuple[int, int]:
    row_margin = random.randint(1, min(GEN_MAX_MARGIN_FROM_CROSS, height - 2))
    col_margin = random.randint(1, min(GEN_MAX_MARGIN_FROM_CROSS, width - 2))
    local_r = height - 1 - row_margin if top_half else row_margin
    local_c = width - 1 - col_margin if left_half else col_margin
    return local_r, local_c


def _place_template_row(grid: np.ndarray, bounds: tuple[int, int, int, int], name: str, seq: np.ndarray) -> None:
    r0, r1, c0, c1 = bounds
    patch = grid[r0:r1, c0:c1]
    boundary_row = _template_row_index(name, patch.shape[0])
    patch[boundary_row, :] = np.where(seq, TEMPLATE, BG)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    cross_r = random.randint(GEN_MIN_QUADRANT, h - GEN_MIN_QUADRANT - 1)
    cross_c = random.randint(GEN_MIN_QUADRANT, w - GEN_MIN_QUADRANT - 1)
    bounds = _quadrant_bounds(h, w, cross_r, cross_c)

    grid = np.full((h, w), BG, dtype=int)
    grid[cross_r, :] = CROSS
    grid[:, cross_c] = CROSS

    use_tl_br = random.choice((False, True))
    seed_quadrants = ("TL", "BR") if use_tl_br else ("TR", "BL")
    template_quadrants = [name for name in ("TL", "TR", "BL", "BR") if name not in seed_quadrants]

    for name in template_quadrants:
        r0, r1, c0, c1 = bounds[name]
        seq = _sample_template(c1 - c0)
        _place_template_row(grid, bounds[name], name, seq)

    for name, color in zip(seed_quadrants, random.sample(SEED_COLORS, 2)):
        r0, r1, c0, c1 = bounds[name]
        local_r, local_c = _seed_local_position(
            r1 - r0,
            c1 - c0,
            top_half=name[0] == "T",
            left_half=name[1] == "L",
        )
        grid[r0 + local_r, c0 + local_c] = color

    try:
        out, paths = _solve(grid)
        _seeds, templates = _extract_layout(grid)
    except AssertionError:
        return None

    if any(len(path) < GEN_MIN_PATH_LEN for path in paths.values()):
        return None

    seed_names = tuple(seed_quadrants)
    if random.choice((False, True)):
        template_lengths = {
            name: len(templates[_horizontal_neighbor(name)])
            for name in seed_names
        }
        if not any(len(paths[name]) > template_lengths[name] for name in seed_names):
            return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
