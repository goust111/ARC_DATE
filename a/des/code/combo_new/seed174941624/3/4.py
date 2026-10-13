import numpy as np


BLACK = 0
GREY = 5
ANCHOR_COLORS = (1, 2, 3, 4)
PATTERN_COLORS = (6, 7, 8, 9)
GRID_MIN = 20
GRID_MAX = 28
WINDOW_MIN = 7
WINDOW_MAX = 9
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use paired border anchors to recover corner crop windows and stack those crops vertically.

    Input structure:
    - The background is black, and the only non-black border pixels are 2-3 colored anchor pairs.
    - Each anchor color appears exactly twice on the outer border: once on the top/bottom edge and once on the left/right edge.
    - Small 4-connected monochrome sprites lie inside corner-oriented rectangular windows implied by those anchor pairs.

    Rule:
    - For each anchor color, cross the horizontal-edge anchor row with the vertical-edge anchor column, and cross the vertical-edge anchor row with the horizontal-edge anchor column, to obtain the two opposite corners of one crop window.
    - Extract that whole window, but clear the anchor pixels themselves so the output keeps only the interior pattern and black background.
    - Order the windows by the left-to-right position of their top/bottom anchor, pad each crop on the right to the maximum window width, and stack them from top to bottom with one grey separator row between consecutive windows.

    Key insight:
    - The border markers encode rectangle extents by combining one anchor's row with the other's column, and the horizontal-anchor columns also encode the vertical stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    windows = _detect_windows(g)
    assert 2 <= len(windows) <= 3, "expected 2-3 anchor-defined windows"

    anchor_cells = {
        window["h_anchor"] for window in windows
    } | {
        window["v_anchor"] for window in windows
    }

    ordered = sorted(
        windows,
        key=lambda window: (window["h_anchor"][1], window["h_anchor"][0]),
    )

    crops = []
    max_width = 0
    total_height = 0
    for window in ordered:
        r0, r1, c0, c1 = window["rect"]
        crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for ar, ac in anchor_cells:
            if r0 <= ar <= r1 and c0 <= ac <= c1:
                crop[ar - r0, ac - c0] = BLACK
        assert crop.shape[0] >= 2 and crop.shape[1] >= 2
        crops.append(crop)
        max_width = max(max_width, int(crop.shape[1]))
        total_height += int(crop.shape[0])

    total_height += len(crops) - 1
    out = np.full((total_height, max_width), BLACK, dtype=int)

    write_row = 0
    for idx, crop in enumerate(crops):
        height, width = crop.shape
        out[write_row : write_row + height, :width] = crop
        write_row += int(height)
        if idx + 1 < len(crops):
            out[write_row, :] = GREY
            write_row += 1

    return out


def _detect_windows(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    border_positions: dict[int, list[tuple[int, int]]] = {}

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            value = int(grid[r, c])
            if value == BLACK:
                continue
            border_positions.setdefault(value, []).append((int(r), int(c)))

    windows = []
    used_positions = 0
    for color, positions in border_positions.items():
        assert len(positions) == 2, f"anchor color {color} must appear exactly twice on the border"
        horizontal = [
            (r, c)
            for r, c in positions
            if r in (0, h - 1) and c not in (0, w - 1)
        ]
        vertical = [
            (r, c)
            for r, c in positions
            if c in (0, w - 1) and r not in (0, h - 1)
        ]
        assert len(horizontal) == 1 and len(vertical) == 1, "each anchor pair must split across horizontal and vertical edges"
        hr, hc = horizontal[0]
        vr, vc = vertical[0]
        r0, r1 = sorted((int(hr), int(vr)))
        c0, c1 = sorted((int(hc), int(vc)))
        windows.append(
            {
                "color": int(color),
                "h_anchor": (int(hr), int(hc)),
                "v_anchor": (int(vr), int(vc)),
                "rect": (int(r0), int(r1), int(c0), int(c1)),
            }
        )
        used_positions += 2

    assert used_positions == sum(len(v) for v in border_positions.values())
    return windows


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample_without_replacement(seq, k: int) -> list:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [seq[int(i)] for i in idx]


def _rect_from_corner(corner: str, height: int, width: int, win_h: int, win_w: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, win_h - 1, 0, win_w - 1
    if corner == "tr":
        return 0, win_h - 1, width - win_w, width - 1
    if corner == "bl":
        return height - win_h, height - 1, 0, win_w - 1
    if corner == "br":
        return height - win_h, height - 1, width - win_w, width - 1
    raise ValueError(f"unknown corner {corner}")


def _anchors_for_corner(corner: str, height: int, width: int, win_h: int, win_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner == "tl":
        return (0, win_w - 1), (win_h - 1, 0)
    if corner == "tr":
        return (0, width - win_w), (win_h - 1, width - 1)
    if corner == "bl":
        return (height - 1, win_w - 1), (height - win_h, 0)
    if corner == "br":
        return (height - 1, width - win_w), (height - win_h, width - 1)
    raise ValueError(f"unknown corner {corner}")


def _random_polyomino(size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < int(size):
        frontier = set()
        for r, c in cells:
            frontier.add((r - 1, c))
            frontier.add((r + 1, c))
            frontier.add((r, c - 1))
            frontier.add((r, c + 1))
        frontier -= cells
        cells.add(_choice(sorted(frontier)))

    coords = np.asarray(sorted(cells), dtype=int)
    if _randint(0, 1):
        coords[:, 1] *= -1
    for _ in range(_randint(0, 3)):
        coords = np.stack((coords[:, 1], -coords[:, 0]), axis=1)

    coords[:, 0] -= int(coords[:, 0].min())
    coords[:, 1] -= int(coords[:, 1].min())
    mask = np.zeros((int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    height, width = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    blocked[rr, cc] = True


def _make_window_pattern(win_h: int, win_w: int) -> np.ndarray | None:
    safe_h = int(win_h - 2)
    safe_w = int(win_w - 2)
    if safe_h < 3 or safe_w < 3:
        return None

    pattern = np.full((win_h, win_w), BLACK, dtype=int)
    blocked = np.zeros((win_h, win_w), dtype=bool)
    area = safe_h * safe_w
    max_shapes = 1
    if area >= 25:
        max_shapes = 2
    if area >= 42:
        max_shapes = 3
    shape_count = _randint(1, max_shapes)
    colors = _sample_without_replacement(PATTERN_COLORS, shape_count)

    placed = 0
    for color in colors:
        success = False
        for _ in range(120):
            size = _randint(3, 6)
            mask = _random_polyomino(size)
            mh, mw = map(int, mask.shape)
            if mh > safe_h or mw > safe_w:
                continue
            top = _randint(1, win_h - mh - 1)
            left = _randint(1, win_w - mw - 1)
            local = np.argwhere(mask)
            coords = local + np.array([[top, left]])
            if np.any(blocked[coords[:, 0], coords[:, 1]]):
                continue
            pattern[coords[:, 0], coords[:, 1]] = int(color)
            _mark_blocked(blocked, coords)
            placed += 1
            success = True
            break
        if not success:
            return None

    return pattern if placed >= 1 else None


def _assign_sizes(selected: list[str], side_a: str, side_b: str) -> dict[str, int]:
    side_corners = [corner for corner in selected if corner in (side_a, side_b)]
    sizes = {}
    if not side_corners:
        return sizes
    values = _sample_without_replacement(list(range(WINDOW_MIN, WINDOW_MAX + 1)), len(side_corners))
    for corner, value in zip(side_corners, values):
        sizes[corner] = int(value)
    return sizes


def _gen_attempt() -> np.ndarray | None:
    corners = ("tl", "tr", "bl", "br")
    count = _randint(2, 3)
    selected = _sample_without_replacement(list(corners), count)
    color_map = {
        corner: color
        for corner, color in zip(selected, _sample_without_replacement(ANCHOR_COLORS, count))
    }

    width_map = {}
    width_map.update(_assign_sizes(selected, "tl", "bl"))
    width_map.update(_assign_sizes(selected, "tr", "br"))

    height_map = {}
    for corner in selected:
        height_map[corner] = _randint(WINDOW_MIN, WINDOW_MAX)

    top_band = max((height_map[c] for c in selected if c in ("tl", "tr")), default=0)
    bottom_band = max((height_map[c] for c in selected if c in ("bl", "br")), default=0)
    left_band = max((width_map[c] for c in selected if c in ("tl", "bl")), default=0)
    right_band = max((width_map[c] for c in selected if c in ("tr", "br")), default=0)

    row_gap_lo = max(2, GRID_MIN - top_band - bottom_band)
    row_gap_hi = GRID_MAX - top_band - bottom_band
    col_gap_lo = max(2, GRID_MIN - left_band - right_band)
    col_gap_hi = GRID_MAX - left_band - right_band
    if row_gap_lo > row_gap_hi or col_gap_lo > col_gap_hi:
        return None

    height = top_band + bottom_band + _randint(row_gap_lo, row_gap_hi)
    width = left_band + right_band + _randint(col_gap_lo, col_gap_hi)
    if not (GRID_MIN <= height <= GRID_MAX and GRID_MIN <= width <= GRID_MAX):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for corner in selected:
        win_h = int(height_map[corner])
        win_w = int(width_map[corner])
        r0, r1, c0, c1 = _rect_from_corner(corner, height, width, win_h, win_w)
        pattern = _make_window_pattern(win_h, win_w)
        if pattern is None:
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1] = pattern
        h_anchor, v_anchor = _anchors_for_corner(corner, height, width, win_h, win_w)
        anchor_color = int(color_map[corner])
        grid[h_anchor] = anchor_color
        grid[v_anchor] = anchor_color

    windows = _detect_windows(grid)
    if len(windows) != count:
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None

    for window in windows:
        r0, r1, c0, c1 = window["rect"]
        crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        crop[window["h_anchor"][0] - r0, window["h_anchor"][1] - c0] = BLACK
        crop[window["v_anchor"][0] - r0, window["v_anchor"][1] - c0] = BLACK
        if not np.any(crop != BLACK):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-window ARC input")
