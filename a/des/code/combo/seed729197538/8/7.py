import numpy as np


FRAME_COLOR = 9
ARROW_COLOR = 8
PADDING = 2
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_frame(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 6 or w < 6:
        return False
    if coords.shape[0] != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch[0, :] == FRAME_COLOR):
        return False
    if not np.all(patch[-1, :] == FRAME_COLOR):
        return False
    if not np.all(patch[:, 0] == FRAME_COLOR):
        return False
    if not np.all(patch[:, -1] == FRAME_COLOR):
        return False
    return not bool((patch[1:-1, 1:-1] == FRAME_COLOR).any())


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for coords in _components(grid == FRAME_COLOR):
        if _is_frame(grid, coords):
            frames.append(_bbox(coords))
    frames.sort()
    return frames


def _arrow_axis(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    arrow = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == ARROW_COLOR)
    if arrow.shape[0] != 3:
        return None
    rows = arrow[:, 0]
    cols = arrow[:, 1]
    if rows.min() == rows.max() and cols.max() - cols.min() == 2:
        return "h"
    if cols.min() == cols.max() and rows.max() - rows.min() == 2:
        return "v"
    return None


def _uniform_color(arr: np.ndarray, forbidden: set[int]) -> int | None:
    flat = np.asarray(arr, dtype=int).ravel()
    if flat.size == 0:
        return None
    color = int(flat[0])
    if color in forbidden:
        return None
    if np.all(flat == color):
        return color
    return None


def _band_candidates(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    axis: str | None,
    bg: int,
) -> list[tuple[str, int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    forbidden = {bg, FRAME_COLOR, ARROW_COLOR}
    candidates: list[tuple[str, int, int]] = []

    def add_horizontal(side: str, row: int, next_row: int | None) -> None:
        if not (0 <= row < h):
            return
        color = _uniform_color(grid[row, c0 + 1 : c1], forbidden)
        if color is None:
            return
        thickness = 1
        if next_row is not None and 0 <= next_row < h:
            if _uniform_color(grid[next_row, c0 + 1 : c1], forbidden) == color:
                thickness = 2
        candidates.append((side, color, thickness))

    def add_vertical(side: str, col: int, next_col: int | None) -> None:
        if not (0 <= col < w):
            return
        color = _uniform_color(grid[r0 + 1 : r1, col], forbidden)
        if color is None:
            return
        thickness = 1
        if next_col is not None and 0 <= next_col < w:
            if _uniform_color(grid[r0 + 1 : r1, next_col], forbidden) == color:
                thickness = 2
        candidates.append((side, color, thickness))

    if axis in (None, "v"):
        add_horizontal("top", r0 - 1, r0 - 2)
        add_horizontal("bottom", r1 + 1, r1 + 2)
    if axis in (None, "h"):
        add_vertical("left", c0 - 1, c0 - 2)
        add_vertical("right", c1 + 1, c1 + 2)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _most_common_color(g)

    for bbox in _find_frames(g):
        r0, c0, r1, c1 = bbox
        axis = _arrow_axis(g, bbox)
        candidates = _band_candidates(g, bbox, axis, bg)
        if len(candidates) != 1:
            candidates = _band_candidates(g, bbox, None, bg)
        assert len(candidates) == 1, "each frame must have exactly one source band"

        side, band_color, thickness = candidates[0]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[g[r0 + 1 : r1, c0 + 1 : c1] == ARROW_COLOR] = bg

        if side == "top":
            out[r0 - thickness : r0, c0 + 1 : c1] = bg
            out[r1 - thickness : r1, c0 + 1 : c1] = band_color
        elif side == "bottom":
            out[r1 + 1 : r1 + 1 + thickness, c0 + 1 : c1] = bg
            out[r0 + 1 : r0 + 1 + thickness, c0 + 1 : c1] = band_color
        elif side == "left":
            out[r0 + 1 : r1, c0 - thickness : c0] = bg
            out[r0 + 1 : r1, c1 - thickness : c1] = band_color
        else:
            out[r0 + 1 : r1, c1 + 1 : c1 + 1 + thickness] = bg
            out[r0 + 1 : r1, c0 + 1 : c0 + 1 + thickness] = band_color

    return out


def _grid_side_range(n_frames: int) -> tuple[int, int]:
    bounds = {
        2: (18, 26),
        3: (22, 30),
        4: (26, 34),
        5: (30, 38),
    }
    return bounds[n_frames]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _draw_arrow(grid: np.ndarray, r0: int, c0: int, h: int, w: int, side: str) -> None:
    cr = r0 + h // 2
    cc = c0 + w // 2
    if side in ("top", "bottom"):
        grid[cr - 1 : cr + 2, cc] = ARROW_COLOR
    else:
        grid[cr, cc - 1 : cc + 2] = ARROW_COLOR


def _draw_band(grid: np.ndarray, r0: int, c0: int, h: int, w: int, side: str, thickness: int, color: int) -> None:
    if side == "top":
        grid[r0 - thickness : r0, c0 + 1 : c0 + w - 1] = color
    elif side == "bottom":
        grid[r0 + h : r0 + h + thickness, c0 + 1 : c0 + w - 1] = color
    elif side == "left":
        grid[r0 + 1 : r0 + h - 1, c0 - thickness : c0] = color
    else:
        grid[r0 + 1 : r0 + h - 1, c0 + w : c0 + w + thickness] = color


def _bg_is_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(n) for v, n in zip(values, counts)}
    bg_count = count_map.pop(int(bg), 0)
    return bg_count > max(count_map.values(), default=0)


def _sample_layout(rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int, int, int, str, int, int]]] | None:
    n_frames = int(rng.integers(2, 6))
    low, high = _grid_side_range(n_frames)
    height = int(rng.integers(low, high + 1))
    width = int(rng.integers(low, high + 1))

    bg_choices = [c for c in range(10) if c not in {FRAME_COLOR, ARROW_COLOR}]
    bg = int(rng.choice(bg_choices))
    band_pool = [c for c in range(10) if c not in {bg, FRAME_COLOR, ARROW_COLOR}]
    band_colors = [int(c) for c in rng.choice(np.asarray(band_pool, dtype=int), size=n_frames, replace=False)]

    grid = np.full((height, width), bg, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    specs: list[tuple[int, int, int, int, str, int, int]] = []
    sides = np.array(["top", "bottom", "left", "right"], dtype=object)

    for band_color in band_colors:
        placed = False
        for _ in range(200):
            inner_h = int(rng.integers(4, 9))
            inner_w = int(rng.integers(4, 9))
            frame_h = inner_h + 2
            frame_w = inner_w + 2
            side = str(rng.choice(sides))
            thickness = int(rng.integers(1, 3))

            max_r0 = height - frame_h - PADDING
            max_c0 = width - frame_w - PADDING
            if max_r0 < PADDING or max_c0 < PADDING:
                return None
            r0 = int(rng.integers(PADDING, max_r0 + 1))
            c0 = int(rng.integers(PADDING, max_c0 + 1))

            rr0 = r0 - PADDING
            rr1 = r0 + frame_h + PADDING
            cc0 = c0 - PADDING
            cc1 = c0 + frame_w + PADDING
            if reserved[rr0:rr1, cc0:cc1].any():
                continue

            _draw_frame(grid, r0, c0, frame_h, frame_w)
            _draw_arrow(grid, r0, c0, frame_h, frame_w, side)
            _draw_band(grid, r0, c0, frame_h, frame_w, side, thickness, band_color)
            reserved[rr0:rr1, cc0:cc1] = True
            specs.append((r0, c0, frame_h, frame_w, side, thickness, band_color))
            placed = True
            break
        if not placed:
            return None

    return grid, specs


def _valid_generated(grid: np.ndarray) -> bool:
    bg = _most_common_color(grid)
    if not _bg_is_mode(grid, bg):
        return False
    frames = _find_frames(grid)
    if len(frames) < 2:
        return False
    if len(frames) != len(_components(grid == FRAME_COLOR)):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return not np.array_equal(grid, out)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _sample_layout(rng)
        if candidate is None:
            continue
        grid, _specs = candidate
        if _valid_generated(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
