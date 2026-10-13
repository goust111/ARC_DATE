from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
CELL_SIZE = 4  # 1 marker cell + 3x3 pattern anchored at the cell's top-left corner
SEP_WIDTH = 1
PATTERN_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400

GRID_H_RANGE = (24, 31)
GRID_W_RANGE = (24, 31)
FRAME_COUNT_RANGE = (1, 5)
MAPPING_COUNT_RANGE = (2, 4)
FRAME_H_RANGE = (5, 10)
FRAME_W_RANGE = (5, 10)
PLACEMENT_ATTEMPTS = 250

SIDES = ("top", "bottom", "left", "right")
CORNERS = ("tl", "tr", "bl", "br")
LAYOUTS_BY_COUNT = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(1, 4), (2, 2), (4, 1)],
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    starts = np.argwhere(mask)
    for start_r, start_c in starts:
        start_r = int(start_r)
        start_c = int(start_c)
        if seen[start_r, start_c]:
            continue
        seen[start_r, start_c] = True
        q = deque([(start_r, start_c)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _component_from_start(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    r0, c0 = map(int, start)
    assert mask[r0, c0]
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    seen[r0, c0] = True
    q = deque([(r0, c0)])
    pts: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        pts.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return np.asarray(pts, dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _corner_candidates(g: np.ndarray) -> list[tuple[str, int, int]]:
    h, w = g.shape
    return [
        ("tl", 0, 0),
        ("tr", 0, w - 1),
        ("bl", h - 1, 0),
        ("br", h - 1, w - 1),
    ]


def _parse_palette(grid: np.ndarray) -> tuple[dict[int, np.ndarray], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    colored_corners = [(name, r, c) for name, r, c in _corner_candidates(g) if int(g[r, c]) != BG]
    assert len(colored_corners) == 1, "palette must occupy exactly one grid corner"

    _corner_name, corner_r, corner_c = colored_corners[0]
    sep = int(g[corner_r, corner_c])
    sep_coords = _component_from_start(g == sep, (corner_r, corner_c))
    r0, c0, r1, c1 = _bbox(sep_coords)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = [r for r in range(sub.shape[0]) if np.all(sub[r, :] == sep)]
    sep_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == sep)]
    assert sep_rows and sep_cols

    row_bands = _bands(sub.shape[0], sep_rows)
    col_bands = _bands(sub.shape[1], sep_cols)
    assert row_bands and col_bands

    mapping: dict[int, np.ndarray] = {}
    for rr0, rr1 in row_bands:
        for cc0, cc1 in col_bands:
            cell = sub[rr0 : rr1 + 1, cc0 : cc1 + 1]
            ch, cw = cell.shape
            assert ch >= CELL_SIZE and cw >= CELL_SIZE

            used = np.zeros((ch, cw), dtype=bool)
            used[0, 0] = True
            used[1 : 1 + PATTERN_SIZE, 1 : 1 + PATTERN_SIZE] = True
            assert np.all(cell[~used] == BG), "palette cell must contain only marker and 3x3 pattern"

            marker = int(cell[0, 0])
            assert marker not in (BG, sep)
            assert marker not in mapping

            pattern = np.array(cell[1 : 1 + PATTERN_SIZE, 1 : 1 + PATTERN_SIZE], dtype=int)
            assert pattern.shape == (PATTERN_SIZE, PATTERN_SIZE)
            assert np.any(pattern != BG)
            mapping[marker] = pattern

    return mapping, (r0, c0, r1, c1)


def _frame_side(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 5 or w < 5:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = np.argwhere(border & ~mask)
    if len(missing) != 2:
        return None

    rows = [int(v) for v in missing[:, 0]]
    cols = [int(v) for v in missing[:, 1]]

    if rows[0] == rows[1] == 0:
        cols.sort()
        if cols[1] == cols[0] + 1 and cols[0] > 0 and cols[1] < w - 1:
            return "top"
    if rows[0] == rows[1] == h - 1:
        cols.sort()
        if cols[1] == cols[0] + 1 and cols[0] > 0 and cols[1] < w - 1:
            return "bottom"
    if cols[0] == cols[1] == 0:
        rows.sort()
        if rows[1] == rows[0] + 1 and rows[0] > 0 and rows[1] < h - 1:
            return "left"
    if cols[0] == cols[1] == w - 1:
        rows.sort()
        if rows[1] == rows[0] + 1 and rows[0] > 0 and rows[1] < h - 1:
            return "right"
    return None


def _find_frames(grid: np.ndarray, palette_bbox: tuple[int, int, int, int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    pr0, pc0, pr1, pc1 = palette_bbox
    outside_palette = np.ones(g.shape, dtype=bool)
    outside_palette[pr0 : pr1 + 1, pc0 : pc1 + 1] = False

    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        mask = (g == color) & outside_palette
        for comp in _components(mask):
            if len(comp) <= 1:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            submask[comp[:, 0] - r0, comp[:, 1] - c0] = True
            side = _frame_side(submask)
            if side is None:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_pts = np.argwhere(interior != BG)
            assert len(seed_pts) == 1, "frame interior must contain exactly one seed pixel"
            seed_r, seed_c = map(int, seed_pts[0])
            seed_color = int(interior[seed_r, seed_c])
            assert seed_color != color
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "seed_color": seed_color,
                }
            )

    assert frames, "expected at least one broken frame"
    return frames


def _tiled_pattern(pattern: np.ndarray, shape: tuple[int, int], side: str) -> np.ndarray:
    h, w = map(int, shape)
    row_idx = np.arange(h, dtype=int) % PATTERN_SIZE
    col_idx = np.arange(w, dtype=int) % PATTERN_SIZE
    if side == "bottom":
        row_idx = (h - 1 - np.arange(h, dtype=int)) % PATTERN_SIZE
    elif side == "right":
        col_idx = (w - 1 - np.arange(w, dtype=int)) % PATTERN_SIZE
    elif side not in ("top", "left"):
        raise AssertionError("unknown side")
    return pattern[row_idx[:, None], col_idx[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the palette seed color to choose a 3x3 pattern, then tile it inside each broken frame.

    Input structure:
    - Black background.
    - One palette panel sits flush in a grid corner, with separator lines dividing 4x4 cells.
      Each cell contains a marker pixel at its top-left corner and a 3x3 pattern at offset (1, 1).
    - Each other object is a 1-pixel-thick rectangular frame with exactly one 2-pixel gap on one side.
      A single seed pixel inside the frame chooses a palette cell by color.

    Rule:
    - The gap side chooses the propagation axis and the tiling anchor side.
    - The seed color selects the 3x3 pattern from the palette.
    - Fill the frame interior by repeating that pattern, anchored from the gap side.
      The seed pixel is overwritten; the broken border itself remains unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, palette_bbox = _parse_palette(g)
    frames = _find_frames(g, palette_bbox)

    out = g.copy()
    for frame in frames:
        seed_color = int(frame["seed_color"])
        assert seed_color in mapping, "seed color must match a palette marker"
        r0, c0, r1, c1 = frame["bbox"]
        fill = _tiled_pattern(mapping[seed_color], (r1 - r0 - 1, c1 - c0 - 1), frame["side"])
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def _panel_size(rows: int, cols: int) -> tuple[int, int]:
    return (
        rows * CELL_SIZE + (rows + 1) * SEP_WIDTH,
        cols * CELL_SIZE + (cols + 1) * SEP_WIDTH,
    )


def _random_pattern(colors: list[int]) -> np.ndarray:
    while True:
        palette_size = random.randint(2, min(4, len(colors)))
        palette = random.sample(colors, k=palette_size)
        pattern = np.array(
            [[random.choice(palette) for _ in range(PATTERN_SIZE)] for _ in range(PATTERN_SIZE)],
            dtype=int,
        )
        if len(set(map(int, pattern.reshape(-1)))) >= 2:
            return pattern


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _central_gap_start(length: int) -> int:
    center = (length - 1) / 2.0
    candidates = []
    best = None
    for start in range(1, length - 2):
        midpoint = start + 0.5
        score = abs(midpoint - center)
        if best is None or score < best:
            best = score
            candidates = [start]
        elif score == best:
            candidates.append(start)
    return random.choice(candidates)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, side: str) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color

    if side in ("top", "bottom"):
        start = _central_gap_start(w)
        rr = r0 if side == "top" else r0 + h - 1
        grid[rr, c0 + start : c0 + start + 2] = BG
    else:
        start = _central_gap_start(h)
        cc = c0 if side == "left" else c0 + w - 1
        grid[r0 + start : r0 + start + 2, cc] = BG


def _render_palette_patch(
    rows: int,
    cols: int,
    markers: list[int],
    patterns: dict[int, np.ndarray],
    sep: int,
) -> np.ndarray:
    ph, pw = _panel_size(rows, cols)
    patch = np.full((ph, pw), sep, dtype=int)
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            marker = int(markers[idx])
            pattern = patterns[marker]
            r0 = 1 + rr * (CELL_SIZE + SEP_WIDTH)
            c0 = 1 + cc * (CELL_SIZE + SEP_WIDTH)
            patch[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = BG
            patch[r0, c0] = marker
            patch[r0 + 1 : r0 + 1 + PATTERN_SIZE, c0 + 1 : c0 + 1 + PATTERN_SIZE] = pattern
            idx += 1
    return patch


def _place_palette(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    corner: str,
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    ph, pw = patch.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    elif corner == "br":
        r0, c0 = h - ph, w - pw
    else:
        raise AssertionError("unknown corner")
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    _mark_blocked(blocked, r0, c0, ph, pw, pad=1)
    return (r0, c0, r0 + ph - 1, c0 + pw - 1)


def _expected_output(inp: np.ndarray, frames: list[dict], patterns: dict[int, np.ndarray]) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        fill = _tiled_pattern(patterns[int(frame["seed_color"])], (frame["h"] - 2, frame["w"] - 2), frame["side"])
        out[frame["r0"] + 1 : frame["r0"] + frame["h"] - 1, frame["c0"] + 1 : frame["c0"] + frame["w"] - 1] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    n_frames = random.randint(*FRAME_COUNT_RANGE)
    n_mappings = random.randint(*MAPPING_COUNT_RANGE)
    if 1 + n_frames + n_mappings > 9:
        return None

    rows, cols = random.choice(LAYOUTS_BY_COUNT[n_mappings])
    ph, pw = _panel_size(rows, cols)
    if ph >= h or pw >= w:
        return None

    colors = list(range(1, 10))
    random.shuffle(colors)
    sep = colors.pop()
    marker_colors = [int(colors.pop()) for _ in range(n_mappings)]
    frame_colors = [int(colors.pop()) for _ in range(n_frames)]

    pattern_pool = [c for c in range(1, 10) if c != sep]
    patterns: dict[int, np.ndarray] = {}
    seen_patterns: set[tuple[int, ...]] = set()
    for marker in marker_colors:
        for _ in range(50):
            pattern = _random_pattern(pattern_pool)
            sig = tuple(map(int, pattern.reshape(-1)))
            if sig not in seen_patterns:
                seen_patterns.add(sig)
                patterns[int(marker)] = pattern
                break
        else:
            return None

    patch = _render_palette_patch(rows, cols, marker_colors, patterns, sep)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    corner = random.choice(CORNERS)
    _place_palette(grid, blocked, patch, corner)

    frames: list[dict] = []
    for frame_color in frame_colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            fh = random.randint(*FRAME_H_RANGE)
            fw = random.randint(*FRAME_W_RANGE)
            if h - fh <= 1 or w - fw <= 1:
                continue

            max_r0 = h - fh - 1
            max_c0 = w - fw - 1
            if max_r0 < 1 or max_c0 < 1:
                continue

            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            if not _can_place(blocked, r0, c0, fh, fw, pad=1):
                continue

            side = random.choice(SIDES)
            seed_color = random.choice(marker_colors)
            seed_r = random.randint(r0 + 1, r0 + fh - 2)
            seed_c = random.randint(c0 + 1, c0 + fw - 2)

            _draw_frame(grid, r0, c0, fh, fw, int(frame_color), side)
            grid[seed_r, seed_c] = int(seed_color)
            _mark_blocked(blocked, r0, c0, fh, fw, pad=1)
            frames.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "h": fh,
                    "w": fw,
                    "side": side,
                    "seed_color": int(seed_color),
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, frames, patterns)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate valid input")
