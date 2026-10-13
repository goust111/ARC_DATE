from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
OBJECT_COLOR = 5
PALETTE_DIRS = [(-1, 1), (1, 1), (1, -1), (-1, -1)]  # RU, RD, LD, LU
SHORT_AXES = ("v", "h")
NON_OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

EDGE_MARGIN = 1
LEFT_MARGIN = 2
RIGHT_MARGIN = 1
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (16, 30)
OBJECT_COUNT_RANGE = (2, 5)
SHORT_LEN_RANGE = (1, 4)
LONG_LEN_CAP = 6
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _analyze_l(coords: list[tuple[int, int]]) -> dict[str, object]:
    cells = set(coords)
    corner: tuple[int, int] | None = None
    endpoints: list[tuple[int, int]] = []

    for r, c in coords:
        nbs = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in cells]
        if len(nbs) == 1:
            endpoints.append((r, c))
        elif len(nbs) == 2:
            d0 = (nbs[0][0] - r, nbs[0][1] - c)
            d1 = (nbs[1][0] - r, nbs[1][1] - c)
            if d0 != (-d1[0], -d1[1]):
                assert corner is None, "expected exactly one bend pixel"
                corner = (r, c)
        else:
            assert len(nbs) <= 2, "L object must be a thin path"

    assert corner is not None, "missing L corner"
    assert len(endpoints) == 2, "L object must have exactly two endpoints"

    vert_ep = [ep for ep in endpoints if ep[1] == corner[1]]
    horiz_ep = [ep for ep in endpoints if ep[0] == corner[0]]
    assert len(vert_ep) == 1 and len(horiz_ep) == 1, "L endpoints must align to axes"
    vert_ep = vert_ep[0]
    horiz_ep = horiz_ep[0]

    v_len = abs(vert_ep[0] - corner[0])
    h_len = abs(horiz_ep[1] - corner[1])
    assert v_len >= 1 and h_len >= 1 and v_len != h_len, "expected distinct arm lengths"

    dr = 1 if vert_ep[0] > corner[0] else -1
    dc = 1 if horiz_ep[1] > corner[1] else -1
    open_dir = (dr, dc)

    if v_len < h_len:
        short_end = vert_ep
        short_len = v_len
        ray_dir = (dr, 0)
    else:
        short_end = horiz_ep
        short_len = h_len
        ray_dir = (0, dc)

    return {
        "corner": corner,
        "open_dir": open_dir,
        "short_end": short_end,
        "short_len": int(short_len),
        "ray_dir": ray_dir,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray L from a left-edge palette and extend its short arm as a matching ray.

    Input structure:
    - The background is a single dominant color.
    - Column 0 contains a vertical 4-pixel palette, with one background gap between consecutive colors.
    - Every other non-background component is a disconnected one-cell-thick L shape whose two orthogonal arms have different lengths.

    Rule:
    - Read the four palette colors from top to bottom as the colors for right-up, right-down, left-down, and left-up openings.
    - For each L, find its corner, identify the two arm directions, and use those directions to determine the opening quadrant.
    - Recolor the whole L with the palette color for that opening, then continue from the short-arm endpoint in the same direction for a number of cells equal to the short-arm length, stopping before any original non-background pixel.

    Key insight:
    - The corner and endpoints of each thin L reveal both the opening quadrant and which arm is shorter, so the palette lookup and ray direction come directly from geometry rather than color ids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _most_common_color(g)
    palette_rows = [int(r) for r in range(h) if int(g[r, 0]) != bg]
    assert len(palette_rows) == 4, "expected a 4-color left-edge palette"
    assert all(palette_rows[i + 1] - palette_rows[i] == 2 for i in range(3)), "palette rows must be evenly spaced"

    palette = {dir_key: int(g[row, 0]) for dir_key, row in zip(PALETTE_DIRS, palette_rows, strict=True)}
    object_mask = g != bg
    object_mask[palette_rows, 0] = False

    out = np.full_like(g, bg)
    for row in palette_rows:
        out[row, 0] = int(g[row, 0])

    obstacles = g != bg
    for comp in _components4(object_mask):
        info = _analyze_l(comp)
        color = palette[info["open_dir"]]  # type: ignore[index]

        for r, c in comp:
            out[r, c] = color

        rr, cc = info["short_end"]  # type: ignore[assignment]
        dr, dc = info["ray_dir"]  # type: ignore[assignment]
        for _ in range(int(info["short_len"])):
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if obstacles[rr, cc]:
                break
            out[rr, cc] = color

    return out


def _balanced_sequence(values: list[object] | tuple[object, ...], n: int) -> list[object]:
    seq: list[object] = []
    values = list(values)
    while len(seq) < n:
        batch = values[:]
        random.shuffle(batch)
        seq.extend(batch)
    return seq[:n]


def _make_spec(
    open_dir: tuple[int, int],
    short_axis: str,
    short_len: int | None = None,
) -> dict[str, object]:
    if short_len is None:
        short_len = random.randint(*SHORT_LEN_RANGE)
    long_min = short_len + 1
    long_max = min(LONG_LEN_CAP, short_len + 3)
    assert long_min <= long_max, "invalid arm-length range"
    long_len = random.randint(long_min, long_max)
    if short_axis == "v":
        v_len, h_len = short_len, long_len
    else:
        v_len, h_len = long_len, short_len
    return {
        "open_dir": open_dir,
        "short_axis": short_axis,
        "short_len": int(short_len),
        "v_len": int(v_len),
        "h_len": int(h_len),
    }


def _shape_cells(spec: dict[str, object], corner: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = corner
    dr, dc = spec["open_dir"]  # type: ignore[assignment]
    v_len = int(spec["v_len"])
    h_len = int(spec["h_len"])
    cells = {(int(r0), int(c0))}
    cells.update((int(r0 + dr * i), int(c0)) for i in range(1, v_len + 1))
    cells.update((int(r0), int(c0 + dc * i)) for i in range(1, h_len + 1))
    return cells


def _relative_cells(spec: dict[str, object]) -> list[tuple[int, int]]:
    return sorted(_shape_cells(spec, (0, 0)))


def _short_end_and_ray(spec: dict[str, object], corner: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], int]:
    r0, c0 = corner
    dr, dc = spec["open_dir"]  # type: ignore[assignment]
    if spec["short_axis"] == "v":
        short_end = (int(r0 + dr * int(spec["v_len"])), int(c0))
        ray_dir = (int(dr), 0)
    else:
        short_end = (int(r0), int(c0 + dc * int(spec["h_len"])))
        ray_dir = (0, int(dc))
    return short_end, ray_dir, int(spec["short_len"])


def _ray_cells(spec: dict[str, object], corner: tuple[int, int]) -> list[tuple[int, int]]:
    short_end, (dr, dc), short_len = _short_end_and_ray(spec, corner)
    return [
        (int(short_end[0] + dr * step), int(short_end[1] + dc * step))
        for step in range(1, short_len + 1)
    ]


def _corner_bounds(spec: dict[str, object], h: int, w: int) -> tuple[int, int, int, int] | None:
    dr, dc = spec["open_dir"]  # type: ignore[assignment]
    v_extent = int(spec["v_len"]) + (int(spec["short_len"]) if spec["short_axis"] == "v" else 0)
    h_extent = int(spec["h_len"]) + (int(spec["short_len"]) if spec["short_axis"] == "h" else 0)

    if dr > 0:
        r_lo = EDGE_MARGIN
        r_hi = h - 1 - EDGE_MARGIN - v_extent
    else:
        r_lo = EDGE_MARGIN + v_extent
        r_hi = h - 1 - EDGE_MARGIN

    if dc > 0:
        c_lo = LEFT_MARGIN
        c_hi = w - 1 - RIGHT_MARGIN - h_extent
    else:
        c_lo = LEFT_MARGIN + h_extent
        c_hi = w - 1 - RIGHT_MARGIN

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return int(r_lo), int(r_hi), int(c_lo), int(c_hi)


def _random_corner(spec: dict[str, object], h: int, w: int) -> tuple[int, int] | None:
    bounds = _corner_bounds(spec, h, w)
    if bounds is None:
        return None
    r_lo, r_hi, c_lo, c_hi = bounds
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                out.add((int(rr), int(cc)))
    return out


def _fits_object(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
    h: int,
    w: int,
    keep_gap_from_paint: bool,
) -> bool:
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if cells & occupied:
        return False
    if cells & reserved_paint:
        return False
    cell_halo = _halo(cells, h, w)
    if cell_halo & occupied:
        return False
    if keep_gap_from_paint and cell_halo & reserved_paint:
        return False
    return True


def _fits_unblocked(
    spec: dict[str, object],
    corner: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
    h: int,
    w: int,
    *,
    keep_gap_from_paint: bool,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    cells = _shape_cells(spec, corner)
    path = _ray_cells(spec, corner)
    if not _fits_object(cells, occupied, reserved_paint, h, w, keep_gap_from_paint):
        return None
    if any(not (0 <= r < h and 0 <= c < w) for r, c in path):
        return None
    if any(cell in occupied or cell in reserved_paint for cell in path):
        return None
    return cells, path


def _render_grid(
    h: int,
    w: int,
    palette_rows: list[int],
    palette_colors: list[int],
    objects: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for row, color in zip(palette_rows, palette_colors, strict=True):
        grid[int(row), 0] = int(color)
    for cells in objects:
        for r, c in cells:
            grid[r, c] = OBJECT_COLOR
    return grid


def _sample_palette_rows(h: int) -> list[int]:
    start = random.randint(EDGE_MARGIN, h - 1 - EDGE_MARGIN - 6)
    return [int(start + 2 * i) for i in range(4)]


def _place_random_object(
    spec: dict[str, object],
    occupied: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(PLACE_ATTEMPTS):
        corner = _random_corner(spec, h, w)
        if corner is None:
            return None
        placed = _fits_unblocked(spec, corner, occupied, reserved_paint, h, w, keep_gap_from_paint=True)
        if placed is not None:
            cells, path = placed
            return cells, path, corner
    return None


def _place_blocker(
    spec: dict[str, object],
    target: tuple[int, int],
    blocked_prefix: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_paint: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int]] | None:
    offsets = _relative_cells(spec)
    random.shuffle(offsets)
    for off_r, off_c in offsets:
        corner = (int(target[0] - off_r), int(target[1] - off_c))
        placed = _fits_unblocked(spec, corner, occupied, reserved_paint, h, w, keep_gap_from_paint=False)
        if placed is None:
            continue
        cells, path = placed
        if target not in cells:
            continue
        if (cells & blocked_prefix) - {target}:
            continue
        return cells, path, corner
    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    palette_rows = _sample_palette_rows(h)
    palette_colors = random.sample(NON_OBJECT_COLORS, 4)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    want_block = bool(random.getrandbits(1))

    open_dirs = _balanced_sequence(PALETTE_DIRS, n_objects)
    short_axes = _balanced_sequence(SHORT_AXES, n_objects)

    occupied = {(int(r), 0) for r in palette_rows}
    reserved_paint: set[tuple[int, int]] = set()
    objects: list[set[tuple[int, int]]] = []

    if want_block:
        source_spec = _make_spec(
            open_dirs[0],  # type: ignore[arg-type]
            short_axes[0],  # type: ignore[arg-type]
            short_len=random.randint(2, SHORT_LEN_RANGE[1]),
        )
        source_placed = _place_random_object(source_spec, occupied, reserved_paint, h, w)
        if source_placed is None:
            return None
        source_cells, source_path, _source_corner = source_placed
        source_block_at = random.randint(2, len(source_path))
        source_prefix = set(source_path[: source_block_at - 1])
        target = source_path[source_block_at - 1]

        objects.append(source_cells)
        occupied |= source_cells
        reserved_paint |= source_prefix

        blocker_spec = _make_spec(
            open_dirs[1],  # type: ignore[arg-type]
            short_axes[1],  # type: ignore[arg-type]
        )
        blocker_placed = _place_blocker(blocker_spec, target, set(source_path[:source_block_at]), occupied, reserved_paint, h, w)
        if blocker_placed is None:
            return None
        blocker_cells, blocker_path, _blocker_corner = blocker_placed
        objects.append(blocker_cells)
        occupied |= blocker_cells
        reserved_paint |= set(blocker_path)

        start_idx = 2
    else:
        start_idx = 0

    for idx in range(start_idx, n_objects):
        spec = _make_spec(
            open_dirs[idx],  # type: ignore[arg-type]
            short_axes[idx],  # type: ignore[arg-type]
        )
        placed = _place_random_object(spec, occupied, reserved_paint, h, w)
        if placed is None:
            return None
        cells, path, _corner = placed
        objects.append(cells)
        occupied |= cells
        reserved_paint |= set(path)

    grid = _render_grid(h, w, palette_rows, palette_colors, objects)
    if Counter(grid.ravel())[BG] <= grid.size // 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None

    used_object_colors = sorted(int(v) for v in np.unique(out) if int(v) not in (BG, *palette_colors))
    if used_object_colors:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
