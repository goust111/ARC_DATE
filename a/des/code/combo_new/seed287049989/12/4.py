from __future__ import annotations

import numpy as np


BG_COLOR = 0
CROSS_COLOR = 2
RULER_COLOR = 5
SEED_PALETTE = np.array([1, 3, 4, 6, 7, 8, 9], dtype=int)

GRID_SIZES = tuple(range(15, 30, 2))
RULER_LENGTHS = tuple(range(2, 6))
GEN_ATTEMPTS = 400
QUADRANT_ATTEMPTS = 200

QUADRANTS = (
    ("tl", -1, -1),
    ("tr", -1, 1),
    ("bl", 1, -1),
    ("br", 1, 1),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _quadrant_slice(name: str, center_r: int, center_c: int) -> tuple[slice, slice]:
    if name == "tl":
        return slice(0, center_r), slice(0, center_c)
    if name == "tr":
        return slice(0, center_r), slice(center_c + 1, None)
    if name == "bl":
        return slice(center_r + 1, None), slice(0, center_c)
    return slice(center_r + 1, None), slice(center_c + 1, None)


def _line_length(coords: np.ndarray) -> int:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if len(rows) == 1:
        span = np.sort(coords[:, 1])
        assert int(span[-1] - span[0] + 1) == len(coords), "ruler must be contiguous"
    elif len(cols) == 1:
        span = np.sort(coords[:, 0])
        assert int(span[-1] - span[0] + 1) == len(coords), "ruler must be contiguous"
    else:
        raise AssertionError("ruler must be horizontal or vertical")
    return int(len(coords))


def _parse_quadrant(
    grid: np.ndarray, bg: int, cross_color: int, name: str, center_r: int, center_c: int
) -> dict[str, object]:
    rs, cs = _quadrant_slice(name, center_r, center_c)
    patch = grid[rs, cs]
    fg = patch[(patch != bg) & (patch != cross_color)]
    colors, counts = np.unique(fg, return_counts=True)
    assert len(colors) == 2, "each quadrant must contain one ruler and one seed"

    seed_color = None
    ruler_color = None
    for color, count in zip(colors, counts):
        if int(count) == 1:
            seed_color = int(color)
        elif 2 <= int(count) <= 5:
            ruler_color = int(color)
        else:
            raise AssertionError("unexpected quadrant color multiplicity")
    assert seed_color is not None and ruler_color is not None

    seed_coords = np.argwhere(patch == seed_color)
    ruler_coords = np.argwhere(patch == ruler_color)
    assert seed_coords.shape == (1, 2), "seed must be a singleton"
    length = _line_length(ruler_coords)

    seed_r = int(seed_coords[0, 0] + rs.start)
    seed_c = int(seed_coords[0, 1] + cs.start)
    ruler_abs = [(int(r + rs.start), int(c + cs.start)) for r, c in ruler_coords]
    return {
        "seed_color": seed_color,
        "seed": (seed_r, seed_c),
        "length": length,
        "ruler": ruler_abs,
    }


def _paint_ray(
    out: np.ndarray,
    protected: np.ndarray,
    seed: tuple[int, int],
    color: int,
    dr: int,
    dc: int,
    length: int,
) -> None:
    h, w = out.shape
    for step in range(length):
        rr = seed[0] + step * dr
        cc = seed[1] + step * dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if (rr, cc) != seed and protected[rr, cc]:
            continue
        out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each quadrant seed into an outward L using that quadrant's ruler length.

    Input structure:
    - The background is dominant, and a single-color horizontal/vertical cross
      passes through the exact grid center.
    - Each of the four quadrants contains exactly one straight ruler segment of
      length 2 to 5 and one non-cross singleton seed pixel.
    - The ruler's orientation is arbitrary and does not encode the growth
      direction; the quadrant location does.

    Rule:
    - Read the ruler length L separately inside each quadrant.
    - Keep the center cross and all ruler pixels unchanged.
    - From the seed pixel, paint one horizontal arm and one vertical arm of the
      seed color, each of total length L including the seed itself.
    - The arms always point away from the center: left/up in the top-left,
      right/up in the top-right, left/down in the bottom-left, right/down in
      the bottom-right; clip at the grid boundary if needed.

    Key insight:
    - The ruler only encodes the arm length, while the quadrant chooses the two
      outward directions and the seed is the L-corner rather than an endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 1 and g.shape[1] % 2 == 1
    h, w = g.shape
    center_r = h // 2
    center_c = w // 2

    bg = _most_common_color(g)
    cross_color = int(g[center_r, center_c])
    assert np.all(g[center_r, :] == cross_color), "expected a full horizontal cross arm"
    assert np.all(g[:, center_c] == cross_color), "expected a full vertical cross arm"
    assert cross_color != bg, "cross color must differ from the background"

    protected = np.zeros_like(g, dtype=bool)
    protected[center_r, :] = True
    protected[:, center_c] = True

    parsed: dict[str, dict[str, object]] = {}
    for name, _dr, _dc in QUADRANTS:
        info = _parse_quadrant(g, bg, cross_color, name, center_r, center_c)
        parsed[name] = info
        for rr, cc in info["ruler"]:
            protected[rr, cc] = True

    out = g.copy()
    for name, dr, dc in QUADRANTS:
        info = parsed[name]
        seed = info["seed"]
        color = int(info["seed_color"])
        length = int(info["length"])
        _paint_ray(out, protected, seed, color, dr, 0, length)
        _paint_ray(out, protected, seed, color, 0, dc, length)

    return out


def _row_room(local_r: int, qh: int, dr: int) -> int:
    return local_r if dr == -1 else qh - 1 - local_r


def _col_room(local_c: int, qw: int, dc: int) -> int:
    return local_c if dc == -1 else qw - 1 - local_c


def _arm_cells(seed: tuple[int, int], length: int, qh: int, qw: int, dr: int, dc: int) -> set[tuple[int, int]]:
    cells = {seed}
    for step in range(1, length):
        rr = seed[0] + step * dr
        if 0 <= rr < qh:
            cells.add((int(rr), int(seed[1])))
        else:
            break
    for step in range(1, length):
        cc = seed[1] + step * dc
        if 0 <= cc < qw:
            cells.add((int(seed[0]), int(cc)))
        else:
            break
    return cells


def _seed_candidates(qh: int, qw: int, length: int, dr: int, dc: int, want_clip: bool) -> list[tuple[int, int]]:
    seeds: list[tuple[int, int]] = []
    for local_r in range(qh):
        for local_c in range(qw):
            row_room = _row_room(local_r, qh, dr)
            col_room = _col_room(local_c, qw, dc)
            if row_room < 1 or col_room < 1:
                continue
            full = row_room >= length - 1 and col_room >= length - 1
            if want_clip and not full:
                seeds.append((int(local_r), int(local_c)))
            if not want_clip and full:
                seeds.append((int(local_r), int(local_c)))
    return seeds


def _ruler_candidates(
    qh: int,
    qw: int,
    length: int,
    orientation: str,
    seed: tuple[int, int],
    arm_cells: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    segments: list[list[tuple[int, int]]] = []
    if orientation == "h":
        for row in range(qh):
            if row == seed[0]:
                continue
            for start in range(qw - length + 1):
                cells = [(row, start + offset) for offset in range(length)]
                if any(cell in arm_cells for cell in cells):
                    continue
                if any(max(abs(r - seed[0]), abs(c - seed[1])) <= 1 for r, c in cells):
                    continue
                segments.append([(int(r), int(c)) for r, c in cells])
    else:
        for col in range(qw):
            if col == seed[1]:
                continue
            for start in range(qh - length + 1):
                cells = [(start + offset, col) for offset in range(length)]
                if any(cell in arm_cells for cell in cells):
                    continue
                if any(max(abs(r - seed[0]), abs(c - seed[1])) <= 1 for r, c in cells):
                    continue
                segments.append([(int(r), int(c)) for r, c in cells])
    return segments


def _local_to_global(local: tuple[int, int], name: str, center_r: int, center_c: int) -> tuple[int, int]:
    base_r = 0 if name in {"tl", "tr"} else center_r + 1
    base_c = 0 if name in {"tl", "bl"} else center_c + 1
    return int(base_r + local[0]), int(base_c + local[1])


def _place_quadrant(
    rng: np.random.Generator,
    grid: np.ndarray,
    name: str,
    center_r: int,
    center_c: int,
    seed_color: int,
    want_clip: bool,
) -> bool:
    qh = center_r
    qw = center_c
    _, dr, dc = next(spec for spec in QUADRANTS if spec[0] == name)

    for _ in range(QUADRANT_ATTEMPTS):
        min_length = 3 if want_clip else 2
        length = int(rng.integers(min_length, min(5, qh, qw) + 1))
        orientation = str(rng.choice(["h", "v"]))
        seeds = _seed_candidates(qh, qw, length, dr, dc, want_clip)
        if not seeds:
            continue
        seed = seeds[int(rng.integers(len(seeds)))]
        arm_cells = _arm_cells(seed, length, qh, qw, dr, dc)
        rulers = _ruler_candidates(qh, qw, length, orientation, seed, arm_cells)
        if not rulers:
            continue
        ruler = rulers[int(rng.integers(len(rulers)))]

        seed_r, seed_c = _local_to_global(seed, name, center_r, center_c)
        grid[seed_r, seed_c] = seed_color
        for local_r, local_c in ruler:
            rr, cc = _local_to_global((local_r, local_c), name, center_r, center_c)
            grid[rr, cc] = RULER_COLOR
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(GRID_SIZES))
    width = int(rng.choice(GRID_SIZES))
    center_r = height // 2
    center_c = width // 2
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[center_r, :] = CROSS_COLOR
    grid[:, center_c] = CROSS_COLOR

    seed_colors = rng.choice(SEED_PALETTE, size=4, replace=False)
    clip_flags = rng.random(4) < 0.5

    for (name, _dr, _dc), seed_color, want_clip in zip(QUADRANTS, seed_colors, clip_flags):
        ok = _place_quadrant(
            rng,
            grid,
            name,
            center_r,
            center_c,
            int(seed_color),
            bool(want_clip),
        )
        if not ok:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
