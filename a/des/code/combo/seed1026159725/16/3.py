import random
import numpy as np


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, _w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        row = grid[r]
        if int(row[0]) == bg:
            continue
        if not np.all(row == row[0]):
            continue
        if not np.any(grid[:r] != bg):
            continue
        if not np.any(grid[r + 1 :] != bg):
            continue
        candidates.append((r, int(row[0])))
    assert len(candidates) == 1, "expected one non-background separator row"
    return candidates[0]


def _find_components(panel: np.ndarray, bg: int, row_offset: int = 0) -> list[dict]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(panel[r0, c0]) == bg:
                continue

            color = int(panel[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords = []

            while stack:
                r, c = stack.pop()
                coords.append((r + row_offset, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(panel[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.asarray(coords, dtype=int)
            rmin = int(coords_arr[:, 0].min())
            rmax = int(coords_arr[:, 0].max())
            cmin = int(coords_arr[:, 1].min())
            cmax = int(coords_arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (rmin, cmin, rmax, cmax),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a two-panel grid, the occupied columns in the upper panel select which
    lower-panel objects survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    sep_row, _sep_color = _find_separator_row(g, bg)

    upper = g[:sep_row]
    lower = g[sep_row + 1 :]
    active_cols = {int(c) for c in np.flatnonzero(np.any(upper != bg, axis=0))}
    assert active_cols, "upper panel must contain foreground"

    out = g.copy()
    for comp in _find_components(lower, bg, row_offset=sep_row + 1):
        _r0, c0, _r1, c1 = comp["bbox"]
        center_twice = c0 + c1
        keep = center_twice % 2 == 0 and (center_twice // 2) in active_cols
        if keep:
            continue
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    return out


PALETTE = tuple(range(10))
GRID_HEIGHT_RANGE = (11, 18)
GRID_WIDTH_RANGE = (12, 20)
TOP_COUNT_RANGE = (2, 5)
BOTTOM_COUNT_RANGE = (3, 6)
MIN_BG_FRACTION = 0.55
MAX_GENERATION_ATTEMPTS = 400

_RAW_SHAPES = [
    ["1"],
    ["1", "1"],
    ["1", "1", "1"],
    ["111"],
    ["010", "111"],
    ["111", "010"],
    ["100", "111"],
    ["001", "111"],
    ["110", "011"],
    ["011", "110"],
    ["010", "111", "010"],
]


def _parse_shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPES = tuple(_parse_shape(rows) for rows in _RAW_SHAPES)


def _paint(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.nonzero(shape)
    grid[top + rr, left + cc] = color


def _fits_with_gap(occupied: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    h, w = shape.shape
    H, W = occupied.shape
    rr, cc = np.nonzero(shape)
    for dr, dc in zip(rr.tolist(), cc.tolist()):
        r = top + dr
        c = left + dc
        if not (0 <= r < H and 0 <= c < W):
            return False
        r0 = max(0, r - 1)
        r1 = min(H, r + 2)
        c0 = max(0, c - 1)
        c1 = min(W, c + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _mark_occupied(occupied: np.ndarray, shape: np.ndarray, top: int, left: int) -> None:
    rr, cc = np.nonzero(shape)
    occupied[top + rr, left + cc] = True


def _all_valid_placements(
    occupied: np.ndarray,
    panel_top: int,
    panel_bottom: int,
    width: int,
    keep_centers: set[int] | None,
) -> list[tuple[np.ndarray, int, int, int]]:
    H, W = occupied.shape
    del H
    placements = []
    for shape in SHAPES:
        h, w = shape.shape
        if w != width:
            continue
        for top in range(panel_top, panel_bottom - h + 1):
            for left in range(0, W - w + 1):
                center_col = left + w // 2
                if keep_centers is not None and center_col not in keep_centers:
                    continue
                if _fits_with_gap(occupied, shape, top, left):
                    placements.append((shape, top, left, center_col))
    return placements


def _sample_branch_labels(count: int) -> list[bool]:
    for _ in range(100):
        labels = [random.random() < 0.5 for _ in range(count)]
        if any(labels) and not all(labels):
            return labels
    return [True] * (count // 2) + [False] * (count - count // 2)


def _generate_one() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    sep_row = random.randint(4, H - 5)

    bg, sep_color = random.sample(PALETTE, 2)
    remaining = [c for c in PALETTE if c not in (bg, sep_color)]

    n_top = random.randint(*TOP_COUNT_RANGE)
    n_bottom = random.randint(*BOTTOM_COUNT_RANGE)
    if n_top > len(remaining) or n_bottom > len(remaining):
        return None

    grid = np.full((H, W), bg, dtype=int)
    expected = np.full((H, W), bg, dtype=int)
    grid[sep_row, :] = sep_color
    expected[sep_row, :] = sep_color

    occupied = np.zeros((H, W), dtype=bool)
    occupied[sep_row, :] = True

    top_colors = random.sample(remaining, n_top)
    for color in top_colors:
        width = 1 if random.random() < 0.4 else 3
        placements = _all_valid_placements(occupied, 0, sep_row, width, keep_centers=None)
        if not placements:
            return None
        shape, top, left, _center = random.choice(placements)
        _paint(grid, shape, top, left, color)
        _paint(expected, shape, top, left, color)
        _mark_occupied(occupied, shape, top, left)

    active_cols = {int(c) for c in np.flatnonzero(np.any(grid[:sep_row] != bg, axis=0))}
    inactive_cols = {c for c in range(W) if c not in active_cols}
    if len(active_cols) < 2 or len(inactive_cols) < 2:
        return None

    bottom_colors = random.sample(remaining, n_bottom)
    labels = _sample_branch_labels(n_bottom)
    random.shuffle(labels)

    for color, keep in zip(bottom_colors, labels):
        width = 1 if random.random() < 0.45 else 3
        target_centers = active_cols if keep else inactive_cols
        placements = _all_valid_placements(
            occupied,
            sep_row + 1,
            H,
            width,
            keep_centers=target_centers,
        )
        if not placements:
            return None
        shape, top, left, _center = random.choice(placements)
        _paint(grid, shape, top, left, color)
        if keep:
            _paint(expected, shape, top, left, color)
        _mark_occupied(occupied, shape, top, left)

    if np.sum(grid == bg) / grid.size < MIN_BG_FRACTION:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _generate_one()
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError(f"failed to generate a valid instance after {MAX_GENERATION_ATTEMPTS} attempts")
