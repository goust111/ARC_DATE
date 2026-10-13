from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLOR = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500
PAIR_COUNT_OPTIONS = (2, 3)


def _mask(rows: Iterable[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SPRITE_PATTERNS = (
    _mask(["111"]),
    _mask(["1", "1", "1"]),
    _mask(["11", "11"]),
    _mask(["111", "111"]),
    _mask(["111", "010"]),
    _mask(["110", "011"]),
    _mask(["111", "110"]),
    _mask(["111", "011"]),
    _mask(["11", "11", "10"]),
    _mask(["11", "11", "01"]),
    _mask(["010", "111", "010"]),
    _mask(["111", "111", "010"]),
    _mask(["010", "111", "111"]),
)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": _bbox(arr),
                }
            )
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _is_frame_component(
    grid: np.ndarray,
    comp: dict,
    color: int,
    bg: int,
) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["coords"].shape[0]) != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and np.all(patch[~border] == bg))


def _l_path(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    out: list[tuple[int, int]] = []
    dc = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + dc, dc):
        out.append((r0, c))
    dr = 1 if r1 >= r0 else -1
    for r in range(r0 + dr, r1 + dr, dr):
        out.append((r, c1))
    return out


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors

    comps_by_color = {color: _components_of_color(g, color) for color in colors}
    frame_colors = [
        color
        for color, comps in comps_by_color.items()
        if comps and all(_is_frame_component(g, comp, color, bg) for comp in comps)
    ]
    assert len(frame_colors) == 1
    frame_color = frame_colors[0]

    interiors = np.zeros_like(g, dtype=bool)
    for comp in comps_by_color[frame_color]:
        r0, c0, r1, c1 = comp["bbox"]
        interiors[r0 + 1 : r1, c0 + 1 : c1] = True

    pairs: list[dict] = []
    for color, comps in comps_by_color.items():
        if color == frame_color:
            continue
        assert len(comps) == 2
        ordered = sorted(
            comps,
            key=lambda comp: (_center_from_bbox(comp["bbox"]), comp["bbox"]),
        )
        start = _center_from_bbox(ordered[0]["bbox"])
        end = _center_from_bbox(ordered[1]["bbox"])
        pairs.append({"color": color, "start": start, "end": end})

    assert pairs
    return g, bg, frame_color, interiors, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Reveal each same-color pair's horizontal-then-vertical path only inside grey frame interiors."""
    g, bg, _frame_color, interiors, pairs = _parse_grid(grid)
    out = g.copy()
    for pair in pairs:
        for r, c in _l_path(pair["start"], pair["end"]):
            if interiors[r, c] and g[r, c] == bg:
                out[r, c] = pair["color"]
    return out


def _cells_from_mask(
    center: tuple[int, int],
    mask: np.ndarray,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    h, w = mask.shape
    top = center[0] - (h - 1) // 2
    left = center[1] - (w - 1) // 2
    coords: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                coords.append((top + r, left + c))
    return coords, (top, left, top + h - 1, left + w - 1)


def _coords_fit(
    coords: list[tuple[int, int]],
    shape: tuple[int, int],
    col_band: tuple[int, int],
    occupied: np.ndarray,
) -> bool:
    h, w = shape
    c0, c1 = col_band
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if not (c0 <= c <= c1):
            return False
        if occupied[r, c]:
            return False
    return True


def _draw_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = outer_bbox
    h, w = grid.shape
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    br0 = max(0, r0 - 1)
    bc0 = max(0, c0 - 1)
    br1 = min(h - 1, r1 + 1)
    bc1 = min(w - 1, c1 + 1)
    blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True


def _place_horizontal_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    row: int,
    open_cols: list[int],
    col_band: tuple[int, int],
) -> bool:
    h, _w = grid.shape
    if len(open_cols) < 2:
        return False
    for _ in range(100):
        inner_w = random.randint(2, min(4, len(open_cols)))
        start_idx = random.randint(0, len(open_cols) - inner_w)
        ic0 = open_cols[start_idx]
        ic1 = ic0 + inner_w - 1
        inner_h = random.randint(2, 4)
        low = max(1, row - inner_h + 1)
        high = min(row, h - inner_h - 1)
        if low > high:
            continue
        ir0 = random.randint(low, high)
        ir1 = ir0 + inner_h - 1
        outer = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
        if outer[1] < col_band[0] or outer[3] > col_band[1]:
            continue
        if blocked[outer[0] : outer[2] + 1, outer[1] : outer[3] + 1].any():
            continue
        _draw_frame(grid, blocked, outer)
        return True
    return False


def _place_vertical_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    col: int,
    open_rows: list[int],
    col_band: tuple[int, int],
) -> bool:
    h, w = grid.shape
    if len(open_rows) < 2:
        return False
    for _ in range(100):
        inner_h = random.randint(2, min(4, len(open_rows)))
        start_idx = random.randint(0, len(open_rows) - inner_h)
        ir0 = open_rows[start_idx]
        ir1 = ir0 + inner_h - 1
        inner_w = random.randint(2, 4)
        low = max(1, col - inner_w + 1)
        high = min(col, w - inner_w - 1)
        if low > high:
            continue
        ic0 = random.randint(low, high)
        ic1 = ic0 + inner_w - 1
        outer = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
        if outer[1] < col_band[0] or outer[3] > col_band[1]:
            continue
        if blocked[outer[0] : outer[2] + 1, outer[1] : outer[3] + 1].any():
            continue
        _draw_frame(grid, blocked, outer)
        return True
    return False


def _place_decoy_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    path_mask: np.ndarray,
) -> bool:
    h, w = grid.shape
    for _ in range(100):
        inner_h = random.randint(2, 4)
        inner_w = random.randint(2, 5)
        if h - inner_h - 2 < 1 or w - inner_w - 2 < 1:
            return False
        ir0 = random.randint(1, h - inner_h - 1)
        ic0 = random.randint(1, w - inner_w - 1)
        ir1 = ir0 + inner_h - 1
        ic1 = ic0 + inner_w - 1
        outer = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
        if blocked[outer[0] : outer[2] + 1, outer[1] : outer[3] + 1].any():
            continue
        if path_mask[outer[0] : outer[2] + 1, outer[1] : outer[3] + 1].any():
            continue
        _draw_frame(grid, blocked, outer)
        return True
    return False


def _make_bands(pair_count: int) -> tuple[int, list[tuple[int, int]], int]:
    h = random.randint(17, 23)
    widths: list[int] = []
    gaps: list[int] = []
    if pair_count == 2:
        widths = [random.randint(8, 10) for _ in range(pair_count)]
        gaps = [random.randint(1, 2) for _ in range(pair_count - 1)]
    else:
        widths = [8, 8, 8]
        gaps = [1, 1]
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    w = left_margin + right_margin + sum(widths) + sum(gaps)
    if w > 31:
        raise ValueError("grid too wide")

    bands: list[tuple[int, int]] = []
    col = left_margin
    for i, width in enumerate(widths):
        bands.append((col, col + width - 1))
        col += width
        if i < len(gaps):
            col += gaps[i]
    return h, bands, w


def _generate_candidate() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    try:
        h, bands, w = _make_bands(pair_count)
    except ValueError:
        return None

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    path_mask = np.zeros((h, w), dtype=bool)

    colors = random.sample(SPRITE_COLORS, pair_count)
    mode_pool = ["h", "v", "both"]
    if pair_count == 2:
        modes = random.sample(mode_pool, 2)
    else:
        modes = mode_pool[:]
        random.shuffle(modes)

    pairs: list[dict] = []
    for band, color, mode in zip(bands, colors, modes):
        placed = False
        usable_cols = list(range(band[0] + 1, band[1]))
        if len(usable_cols) < 5:
            return None
        for _ in range(120):
            mask_a = random.choice(SPRITE_PATTERNS)
            mask_b = random.choice(SPRITE_PATTERNS)

            c2 = random.choice(usable_cols)
            c1_candidates = [c for c in usable_cols if abs(c - c2) >= 5]
            if not c1_candidates:
                continue
            c1 = random.choice(c1_candidates)

            top_limit = max(2, h // 2 - 4)
            if top_limit < 2:
                continue
            r1 = random.randint(2, top_limit)
            if r1 + 5 > h - 3:
                continue
            r2 = random.randint(r1 + 5, h - 3)

            start = (r1, c1)
            end = (r2, c2)
            coords_a, _ = _cells_from_mask(start, mask_a)
            coords_b, _ = _cells_from_mask(end, mask_b)
            if not _coords_fit(coords_a, grid.shape, band, occupied):
                continue
            if not _coords_fit(coords_b, grid.shape, band, occupied):
                continue

            sprite_cells = set(coords_a) | set(coords_b)
            path = _l_path(start, end)
            if any(path_mask[r, c] for r, c in path):
                continue
            if any(occupied[r, c] for r, c in path if (r, c) not in sprite_cells):
                continue

            open_cols = list(range(min(c1, c2) + 1, max(c1, c2)))
            open_rows = list(range(r1 + 1, r2))
            h_frame_cols = open_cols[1:-1]
            v_frame_rows = open_rows[1:-1]

            if mode in {"h", "both"} and len(h_frame_cols) < 2:
                continue
            if mode in {"v", "both"} and len(v_frame_rows) < 2:
                continue

            for r, c in coords_a:
                grid[r, c] = color
                occupied[r, c] = True
            for r, c in coords_b:
                grid[r, c] = color
                occupied[r, c] = True
            for r, c in path:
                path_mask[r, c] = True

            pairs.append(
                {
                    "color": color,
                    "start": start,
                    "end": end,
                    "h_frame_cols": h_frame_cols,
                    "v_frame_rows": v_frame_rows,
                    "mode": mode,
                    "band": band,
                }
            )
            placed = True
            break
        if not placed:
            return None

    blocked = occupied.copy()
    for pair in pairs:
        mode = pair["mode"]
        if mode in {"h", "both"} and not _place_horizontal_frame(
            grid,
            blocked,
            pair["start"][0],
            pair["h_frame_cols"],
            pair["band"],
        ):
            return None
        if mode in {"v", "both"} and not _place_vertical_frame(
            grid,
            blocked,
            pair["end"][1],
            pair["v_frame_rows"],
            pair["band"],
        ):
            return None

    for _ in range(random.randint(0, 2)):
        _place_decoy_frame(grid, blocked, path_mask)

    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        parsed_grid, bg, frame_color, interiors, pairs = _parse_grid(grid)
    except AssertionError:
        return False

    if bg != BG or frame_color != FRAME_COLOR:
        return False
    non_bg_counts = [int(np.count_nonzero(parsed_grid == color)) for color in np.unique(parsed_grid) if color != bg]
    if not non_bg_counts or int(np.count_nonzero(parsed_grid == bg)) <= max(non_bg_counts):
        return False

    out = main(parsed_grid)
    diff = out != parsed_grid
    if not np.any(diff):
        return False
    if not np.all(parsed_grid[diff] == bg):
        return False
    if not np.all(interiors[diff]):
        return False

    for pair in pairs:
        visible = 0
        for r, c in _l_path(pair["start"], pair["end"]):
            if interiors[r, c] and parsed_grid[r, c] == bg:
                visible += 1
        if visible == 0:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
