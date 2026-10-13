import random
from collections import deque

import numpy as np


MAX_GEN_ATTEMPTS = 400
PAD = 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _scale_index(idx: int, src_len: int, dst_len: int) -> int:
    if src_len <= 1 or dst_len <= 1:
        return 0
    return idx * (dst_len - 1) // (src_len - 1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_window_frame(grid: np.ndarray, bg: int, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 5:
        return False
    coords = comp["coords"]
    if len(coords) != 2 * (h + w) - 4:
        return False
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub[1:-1, 1:-1] != bg):
        return False
    return True


def _find_windows(grid: np.ndarray, bg: int) -> list[dict]:
    frames = [comp for comp in _find_components(grid, bg) if _is_window_frame(grid, bg, comp)]
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _project_pixels(
    out: np.ndarray,
    pixels: list[tuple[int, int]],
    landscape_h: int,
    window: dict,
) -> None:
    r0, c0, r1, c1 = window["bbox"]
    color = int(window["color"])
    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    inner_c1 = c1 - 1
    inner_h = r1 - r0 - 1
    for sr, sc in pixels:
        if inner_c0 <= sc <= inner_c1:
            dr = inner_r0 + _scale_index(sr, landscape_h, inner_h)
            out[dr, sc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Window colors filter which landscape objects are projected into each frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    windows = _find_windows(g, bg)
    assert windows, "expected at least one hollow window frame"

    landscape_h = min(r0 for r0, _, _, _ in (window["bbox"] for window in windows))
    assert landscape_h >= 1, "expected a non-empty landscape region above the windows"

    out = g.copy()
    for window in windows:
        color = int(window["color"])
        rows, cols = np.where(g[:landscape_h] == color)
        pixels = [(int(r), int(c)) for r, c in zip(rows, cols)]
        _project_pixels(out, pixels, landscape_h, window)
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _grow_blob(rng: random.Random, h: int, w: int, area: int) -> np.ndarray:
    seed = (rng.randrange(h), rng.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _tight_crop(mask)


def _sample_object_mask(rng: random.Random) -> np.ndarray:
    kind = rng.choice(["blob", "rect", "line", "ell"])
    if kind == "rect":
        h = rng.randint(1, 4)
        w = rng.randint(1, 5)
        return np.ones((h, w), dtype=bool)
    if kind == "line":
        if rng.random() < 0.5:
            return np.ones((1, rng.randint(2, 6)), dtype=bool)
        return np.ones((rng.randint(2, 5), 1), dtype=bool)
    if kind == "ell":
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        mask = np.zeros((h, w), dtype=bool)
        arm_r = rng.randrange(h)
        arm_c = rng.randrange(w)
        mask[arm_r, :] = True
        mask[:, arm_c] = True
        if rng.random() < 0.5:
            mask[0, 0] = False
        return _tight_crop(mask)
    h = rng.randint(2, 5)
    w = rng.randint(2, 6)
    max_area = h * w
    area = rng.randint(max(3, (h + w) // 2), max(3, max_area - 1))
    return _grow_blob(rng, h, w, area)


def _free_box_positions(
    occupied: np.ndarray,
    box_h: int,
    box_w: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    padding: int = PAD,
    fixed_row: int | None = None,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    row_range = [fixed_row] if fixed_row is not None else range(row_lo, row_hi + 1)
    for r0 in row_range:
        if r0 < row_lo or r0 > row_hi:
            continue
        for c0 in range(col_lo, col_hi + 1):
            rr0 = max(0, r0 - padding)
            rr1 = min(h, r0 + box_h + padding)
            cc0 = max(0, c0 - padding)
            cc1 = min(w, c0 + box_w + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _reserve_box(occupied: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, padding: int = PAD) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - padding)
    rr1 = min(h, r0 + box_h + padding)
    cc0 = max(0, c0 - padding)
    cc1 = min(w, c0 + box_w + padding)
    occupied[rr0:rr1, cc0:cc1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _sample_windows(
    rng: random.Random,
    height: int,
    width: int,
    split: int,
    colors: list[int],
) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    windows: list[dict] = []
    order = list(colors)
    rng.shuffle(order)
    for idx, color in enumerate(order):
        frame_h = rng.randint(5, min(9, height - split))
        frame_w = rng.randint(6, min(11, width))
        fixed_row = split if idx == 0 else None
        row_hi = height - frame_h
        col_hi = width - frame_w
        positions = _free_box_positions(
            occupied,
            frame_h,
            frame_w,
            split,
            row_hi,
            0,
            col_hi,
            fixed_row=fixed_row,
        )
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _reserve_box(occupied, r0, c0, frame_h, frame_w)
        windows.append({"color": int(color), "bbox": (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)})
    windows.sort(key=lambda item: item["bbox"])
    return windows


def _sample_object_positions(
    occupied: np.ndarray,
    mask: np.ndarray,
    row_limit: int,
    overlap: tuple[int, int] | None,
    require_cross: bool | None,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = mask.shape
    positions: list[tuple[int, int]] = []
    cols_local = np.where(mask)[1]
    for r0 in range(0, row_limit - ph + 1):
        for c0 in range(0, w - pw + 1):
            rr0 = max(0, r0 - PAD)
            rr1 = min(h, r0 + ph + PAD)
            cc0 = max(0, c0 - PAD)
            cc1 = min(w, c0 + pw + PAD)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            if overlap is not None:
                lo, hi = overlap
                cols = [int(c0 + dc) for dc in cols_local]
                has_inside = any(lo <= col <= hi for col in cols)
                if not has_inside:
                    continue
                has_outside = any(col < lo or col > hi for col in cols)
                if require_cross is True and not has_outside:
                    continue
                if require_cross is False and has_outside:
                    continue
            positions.append((r0, c0))
    return positions


def _expected_output(inp: np.ndarray, split: int, windows: list[dict]) -> np.ndarray:
    out = inp.copy()
    for window in windows:
        color = int(window["color"])
        rows, cols = np.where(inp[:split] == color)
        pixels = [(int(r), int(c)) for r, c in zip(rows, cols)]
        _project_pixels(out, pixels, split, window)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(16, 24)
    width = rng.randint(16, 28)
    split_lo = max(4, height // 3)
    split_hi = min(height - 8, (2 * height) // 3)
    if split_lo > split_hi:
        return None
    split = rng.randint(split_lo, split_hi)

    bg = rng.randrange(10)
    usable = [c for c in range(10) if c != bg]

    n_windows = rng.randint(2, 4)
    if len(usable) < n_windows:
        return None
    window_colors = rng.sample(usable, n_windows)
    windows = _sample_windows(rng, height, width, split, window_colors)
    if windows is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for window in windows:
        r0, c0, r1, c1 = window["bbox"]
        _paint_frame(grid, r0, c0, r1 - r0 + 1, c1 - c0 + 1, int(window["color"]))

    occupied = np.zeros((height, width), dtype=bool)
    for window in windows:
        r0, c0, r1, c1 = window["bbox"]
        _reserve_box(occupied, r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    distractor_pool = [c for c in usable if c not in window_colors]
    n_distractors = rng.randint(0, min(2, len(distractor_pool)))
    distractor_colors = rng.sample(distractor_pool, n_distractors) if n_distractors else []

    object_plan: list[tuple[int, tuple[int, int] | None]] = []
    for window in windows:
        inner_lo = window["bbox"][1] + 1
        inner_hi = window["bbox"][3] - 1
        object_plan.append((int(window["color"]), (inner_lo, inner_hi)))
        for _ in range(rng.randint(0, 2)):
            object_plan.append((int(window["color"]), None))
    for color in distractor_colors:
        for _ in range(rng.randint(1, 2)):
            object_plan.append((int(color), None))
    rng.shuffle(object_plan)

    anchored: set[int] = set()
    for color, overlap in object_plan:
        mask = _sample_object_mask(rng)
        require_cross: bool | None = None
        if overlap is not None and color not in anchored:
            require_cross = rng.choice([True, False])
        positions = _sample_object_positions(occupied, mask, split, overlap, require_cross)
        if not positions and overlap is not None:
            positions = _sample_object_positions(occupied, mask, split, overlap, None)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_mask(grid, mask, r0, c0, color)
        _reserve_box(occupied, r0, c0, mask.shape[0], mask.shape[1])
        if overlap is not None:
            anchored.add(color)

    if anchored != set(window_colors):
        return None

    expected = _expected_output(grid, split, windows)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
