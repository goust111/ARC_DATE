from __future__ import annotations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _corner_origins(height: int, width: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, width - 2), (height - 2, 0), (height - 2, width - 2)]


def _find_legend(
    grid: np.ndarray, bg: int
) -> tuple[tuple[int, int], dict[int, bool]]:
    height, width = grid.shape
    candidates: list[tuple[tuple[int, int], dict[int, bool]]] = []
    for r0, c0 in _corner_origins(height, width):
        block = grid[r0 : r0 + 2, c0 : c0 + 2]
        if block.shape != (2, 2):
            continue
        if (
            int(block[0, 0]) == bg
            and int(block[1, 1]) == bg
            and int(block[0, 1]) != bg
            and int(block[1, 0]) != bg
            and int(block[0, 1]) != int(block[1, 0])
        ):
            candidates.append(
                (
                    (r0, c0),
                    {
                        int(block[1, 0]): True,
                        int(block[0, 1]): False,
                    },
                )
            )
    assert len(candidates) == 1, "expected exactly one 2x2 corner legend"
    return candidates[0]


def _flood_component(
    grid: np.ndarray, start: tuple[int, int], color: int
) -> set[tuple[int, int]]:
    height, width = grid.shape
    sr, sc = start
    assert 0 <= sr < height and 0 <= sc < width
    assert int(grid[sr, sc]) == color

    stack = [(sr, sc)]
    seen = {(sr, sc)}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return seen


def _vertical_crack_rows(
    grid: np.ndarray, r: int, c: int, marker_color: int, object_color: int, bg: int
) -> list[int] | None:
    height, width = grid.shape
    if not (0 < c < width - 1):
        return None

    def valid_row(rr: int) -> bool:
        if not (0 <= rr < height):
            return False
        center = int(grid[rr, c])
        if rr == r:
            if center != marker_color:
                return False
        elif center != bg:
            return False
        return (
            int(grid[rr, c - 1]) == object_color
            and int(grid[rr, c + 1]) == object_color
        )

    rows = [r]
    rr = r - 1
    while valid_row(rr):
        rows.append(rr)
        rr -= 1
    rr = r + 1
    while valid_row(rr):
        rows.append(rr)
        rr += 1
    rows.sort()
    return rows if len(rows) >= 2 else None


def _horizontal_crack_cols(
    grid: np.ndarray, r: int, c: int, marker_color: int, object_color: int, bg: int
) -> list[int] | None:
    height, width = grid.shape
    if not (0 < r < height - 1):
        return None

    def valid_col(cc: int) -> bool:
        if not (0 <= cc < width):
            return False
        center = int(grid[r, cc])
        if cc == c:
            if center != marker_color:
                return False
        elif center != bg:
            return False
        return (
            int(grid[r - 1, cc]) == object_color
            and int(grid[r + 1, cc]) == object_color
        )

    cols = [c]
    cc = c - 1
    while valid_col(cc):
        cols.append(cc)
        cc -= 1
    cc = c + 1
    while valid_col(cc):
        cols.append(cc)
        cc += 1
    cols.sort()
    return cols if len(cols) >= 2 else None


def _marker_info(
    grid: np.ndarray, r: int, c: int, marker_color: int, bg: int
) -> tuple[str, int, list[tuple[int, int]]] | None:
    height, width = grid.shape
    assert 0 <= r < height and 0 <= c < width

    vertical: tuple[str, int, list[tuple[int, int]]] | None = None
    if 0 < c < width - 1:
        left = int(grid[r, c - 1])
        right = int(grid[r, c + 1])
        if left == right and left not in (bg, marker_color):
            rows = _vertical_crack_rows(grid, r, c, marker_color, left, bg)
            if rows is not None:
                vertical = ("vertical", left, [(rr, c) for rr in rows])

    horizontal: tuple[str, int, list[tuple[int, int]]] | None = None
    if 0 < r < height - 1:
        up = int(grid[r - 1, c])
        down = int(grid[r + 1, c])
        if up == down and up not in (bg, marker_color):
            cols = _horizontal_crack_cols(grid, r, c, marker_color, up, bg)
            if cols is not None:
                horizontal = ("horizontal", up, [(r, cc) for cc in cols])

    if (vertical is None) == (horizontal is None):
        return None
    return vertical if vertical is not None else horizontal


def main(grid: np.ndarray) -> np.ndarray:
    """Keep one half of each cracked object according to the corner legend."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    (legend_r0, legend_c0), keep_lower_left = _find_legend(g, bg)

    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[legend_r0 : legend_r0 + 2, legend_c0 : legend_c0 + 2] = True
    marker_colors = set(keep_lower_left)

    jobs: list[tuple[int, int, int, str, int, list[tuple[int, int]]]] = []
    for r, c in np.argwhere(~legend_mask):
        color = int(g[r, c])
        if color not in marker_colors:
            continue
        info = _marker_info(g, int(r), int(c), color, bg)
        if info is None:
            continue
        orientation, object_color, crack_cells = info
        jobs.append((int(r), int(c), color, orientation, object_color, crack_cells))

    assert jobs, "expected at least one crack marker"

    out = g.copy()
    for r, c, marker_color, orientation, object_color, crack_cells in jobs:
        keep_lower_or_left = keep_lower_left[marker_color]
        if orientation == "vertical":
            left_seed = (r, c - 1)
            right_seed = (r, c + 1)
            left_half = _flood_component(g, left_seed, object_color)
            right_half = _flood_component(g, right_seed, object_color)
            assert left_half.isdisjoint(right_half)
            erase = right_half if keep_lower_or_left else left_half
        else:
            above_seed = (r - 1, c)
            below_seed = (r + 1, c)
            above_half = _flood_component(g, above_seed, object_color)
            below_half = _flood_component(g, below_seed, object_color)
            assert above_half.isdisjoint(below_half)
            erase = above_half if keep_lower_or_left else below_half

        for rr, cc in erase:
            out[rr, cc] = bg
        for rr, cc in crack_cells:
            out[rr, cc] = bg

    return out


# --------------- generator ---------------

_GEN_GRID_MIN = 14
_GEN_GRID_MAX = 22
_GEN_OBJECT_COUNT = (1, 5)
_GEN_COLORS = tuple(range(10))
_GEN_ATTEMPTS = 400
_GEN_PADDING = 1
_GEN_CORNER_MARGIN = 3
_GEN_PER_OBJECT_TRIES = 80


def _reserve_box(
    occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = _GEN_PADDING
) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _reserve_corners(occupied: np.ndarray) -> None:
    h, w = occupied.shape
    for r0, c0 in ((0, 0), (0, w - _GEN_CORNER_MARGIN), (h - _GEN_CORNER_MARGIN, 0), (h - _GEN_CORNER_MARGIN, w - _GEN_CORNER_MARGIN)):
        occupied[r0 : r0 + _GEN_CORNER_MARGIN, c0 : c0 + _GEN_CORNER_MARGIN] = True


def _legend_origin(height: int, width: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - 2
    if corner == "bl":
        return height - 2, 0
    assert corner == "br"
    return height - 2, width - 2


def _render_legend(
    grid: np.ndarray, origin: tuple[int, int], bg: int, upper_right: int, lower_left: int
) -> None:
    r0, c0 = origin
    grid[r0 : r0 + 2, c0 : c0 + 2] = np.array(
        [[bg, upper_right], [lower_left, bg]], dtype=int
    )


def _render_object(grid: np.ndarray, spec: dict[str, int]) -> None:
    r0, c0 = spec["r0"], spec["c0"]
    h, w = spec["h"], spec["w"]
    object_color = spec["object_color"]
    grid[r0 : r0 + h, c0 : c0 + w] = object_color

    marker_color = spec["marker_color"]
    if spec["orientation"] == 0:
        crack_c = c0 + spec["split"]
        grid[r0 : r0 + h, crack_c] = spec["bg"]
        grid[r0 + spec["marker_pos"], crack_c] = marker_color
    else:
        crack_r = r0 + spec["split"]
        grid[crack_r, c0 : c0 + w] = spec["bg"]
        grid[crack_r, c0 + spec["marker_pos"]] = marker_color


def _expected_output(
    grid: np.ndarray, specs: list[dict[str, int]], keep_lower_left: dict[int, bool]
) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0 = spec["r0"], spec["c0"]
        h, w = spec["h"], spec["w"]
        marker_color = spec["marker_color"]
        bg = spec["bg"]
        if spec["orientation"] == 0:
            crack_c = c0 + spec["split"]
            if keep_lower_left[marker_color]:
                out[r0 : r0 + h, crack_c + 1 : c0 + w] = bg
            else:
                out[r0 : r0 + h, c0:crack_c] = bg
            out[r0 : r0 + h, crack_c] = bg
        else:
            crack_r = r0 + spec["split"]
            if keep_lower_left[marker_color]:
                out[r0:crack_r, c0 : c0 + w] = bg
            else:
                out[crack_r + 1 : r0 + h, c0 : c0 + w] = bg
            out[crack_r, c0 : c0 + w] = bg
    return out


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    by_color = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = by_color.pop(bg, 0)
    return bool(by_color) and bg_count > max(by_color.values())


def _random_object_spec(
    rng: np.random.Generator,
    occupied: np.ndarray,
    color: int,
    marker_color: int,
    bg: int,
) -> dict[str, int] | None:
    height, width = occupied.shape
    orientation = int(rng.integers(0, 2))
    for _ in range(_GEN_PER_OBJECT_TRIES):
        if orientation == 0:
            h = int(rng.integers(4, min(9, height - 3) + 1))
            w = int(rng.integers(5, min(10, width - 3) + 1))
            split = int(rng.integers(2, w - 2))
            marker_pos = int(rng.integers(0, h))
        else:
            h = int(rng.integers(5, min(10, height - 3) + 1))
            w = int(rng.integers(4, min(9, width - 3) + 1))
            split = int(rng.integers(2, h - 2))
            marker_pos = int(rng.integers(0, w))

        if h >= height or w >= width:
            continue

        r0 = int(rng.integers(1, height - h))
        c0 = int(rng.integers(1, width - w))
        if occupied[r0 : r0 + h, c0 : c0 + w].any():
            continue

        _reserve_box(occupied, r0, c0, h, w)
        return {
            "r0": r0,
            "c0": c0,
            "h": h,
            "w": w,
            "orientation": orientation,
            "split": split,
            "marker_pos": marker_pos,
            "object_color": color,
            "marker_color": marker_color,
            "bg": bg,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(_GEN_GRID_MIN, _GEN_GRID_MAX + 1))
    width = int(rng.integers(_GEN_GRID_MIN, _GEN_GRID_MAX + 1))
    bg = int(rng.choice(_GEN_COLORS))

    free_colors = [c for c in _GEN_COLORS if c != bg]
    marker_colors = list(map(int, rng.choice(free_colors, size=2, replace=False)))
    object_pool = [c for c in free_colors if c not in marker_colors]

    n_objects = int(rng.integers(*_GEN_OBJECT_COUNT))
    if len(object_pool) < n_objects:
        return None
    object_colors = list(map(int, rng.choice(object_pool, size=n_objects, replace=False)))

    corner = str(rng.choice(["tl", "tr", "bl", "br"]))
    legend_origin = _legend_origin(height, width, corner)
    keep_lower_left = {marker_colors[0]: True, marker_colors[1]: False}

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    _reserve_corners(occupied)
    _render_legend(grid, legend_origin, bg, marker_colors[1], marker_colors[0])

    specs: list[dict[str, int]] = []
    for object_color in object_colors:
        marker_color = int(marker_colors[int(rng.integers(0, 2))])
        spec = _random_object_spec(rng, occupied, object_color, marker_color, bg)
        if spec is None:
            return None
        specs.append(spec)

    for spec in specs:
        _render_object(grid, spec)

    if not _background_is_dominant(grid, bg):
        return None

    expected = _expected_output(grid, specs, keep_lower_left)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cracked-object input")
