import math
from typing import Optional

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rectangle_from_component(coords: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = np.zeros((h, w), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(sub, expected):
        return None
    return r0, c0, r1, c1


def _axis_period(arr: np.ndarray, axis: int) -> int:
    size = arr.shape[axis]
    for period in range(1, size + 1):
        ok = True
        for idx in range(size):
            src = [slice(None), slice(None)]
            ref = [slice(None), slice(None)]
            src[axis] = idx
            ref[axis] = idx % period
            if not np.array_equal(arr[tuple(src)], arr[tuple(ref)]):
                ok = False
                break
        if ok:
            return period
    return size


def _minimal_tile(pattern: np.ndarray) -> np.ndarray:
    pr = _axis_period(pattern, axis=0)
    pc = _axis_period(pattern, axis=1)
    return pattern[:pr, :pc]


def _tile_to_shape(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    ph, pw = pattern.shape
    reps_r = (height + ph - 1) // ph
    reps_c = (width + pw - 1) // pw
    tiled = np.tile(pattern, (reps_r, reps_c))
    return tiled[:height, :width]


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rectangles = []
    for color in sorted(set(map(int, grid.reshape(-1))) - {bg}):
        for coords in _components(grid == color):
            box = _rectangle_from_component(coords)
            if box is None:
                continue
            r0, c0, r1, c1 = box
            inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
            rectangles.append({"color": int(color), "bbox": box, "inner": inner})
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each hollow work-frame to the palette block with the same border color,
    then tile that palette pattern into the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rectangles = _find_rectangles(g, bg)

    palettes: dict[int, np.ndarray] = {}
    frames = []
    for rect in rectangles:
        inner = rect["inner"]
        if inner.size == 0:
            continue
        if np.any(inner != bg):
            palettes[rect["color"]] = _minimal_tile(inner)
        elif np.all(inner == bg):
            frames.append(rect)

    assert palettes, "expected at least one palette block"
    assert frames, "expected at least one work frame"

    out = g.copy()
    for frame in frames:
        color = int(frame["color"])
        assert color in palettes, "every work-frame color must exist in the palette"
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_to_shape(palettes[color], inner_h, inner_w)
    return out


BG = 0
PATTERN_KINDS = (
    "checker",
    "vstripe",
    "hstripe",
    "dots",
    "plus",
    "diag",
    "blocks",
)
PALETTE_COUNT_RANGE = (2, 5)
EXTRA_FRAME_RANGE = (0, 3)
PALETTE_INNER_RANGE = (3, 6)
FRAME_INNER_MIN = 2
FRAME_INNER_MAX = 9
MAX_ATTEMPTS = 300


def _pattern_mask(kind: str, h: int, w: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    if kind == "checker":
        mask = (rr + cc) % 2 == 0
    elif kind == "vstripe":
        mask = cc % 2 == 0
    elif kind == "hstripe":
        mask = rr % 2 == 0
    elif kind == "dots":
        mask = (rr % 2 == 0) & (cc % 2 == 0)
    elif kind == "plus":
        mask = (rr == h // 2) | (cc == w // 2)
    elif kind == "diag":
        mask = (rr - cc) % 3 == 0
    elif kind == "blocks":
        mask = ((rr // 2) + (cc // 2)) % 2 == 0
    else:
        raise ValueError(f"unknown pattern {kind}")
    return mask


def _make_palette_tile(rng: np.random.Generator, fill_color: int) -> np.ndarray:
    h = int(rng.integers(*PALETTE_INNER_RANGE))
    w = int(rng.integers(*PALETTE_INNER_RANGE))
    kind = PATTERN_KINDS[int(rng.integers(len(PATTERN_KINDS)))]
    tile = np.full((h, w), BG, dtype=int)
    mask = _pattern_mask(kind, h, w)
    if not mask.any() or np.all(mask):
        return _make_palette_tile(rng, fill_color)
    tile[mask] = fill_color
    return tile


def _sample_dim(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _sample_frame_inner(rng: np.random.Generator, tile_shape: tuple[int, int]) -> tuple[int, int]:
    th, tw = tile_shape
    mode = int(rng.integers(3))

    def smaller_dim(base: int) -> int:
        return _sample_dim(rng, FRAME_INNER_MIN, max(FRAME_INNER_MIN, base))

    def larger_dim(base: int) -> int:
        lo = min(FRAME_INNER_MAX, base + 1)
        hi = min(FRAME_INNER_MAX, base + 4)
        if lo > hi:
            return smaller_dim(base)
        return _sample_dim(rng, lo, hi)

    if mode == 0:
        return smaller_dim(th), smaller_dim(tw)
    if mode == 1:
        return larger_dim(th), larger_dim(tw)
    if rng.random() < 0.5:
        return smaller_dim(th), larger_dim(tw)
    return larger_dim(th), smaller_dim(tw)


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r - pad)
    rr1 = min(occupied.shape[0], r + h + pad)
    cc0 = max(0, c - pad)
    cc1 = min(occupied.shape[1], c + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_rectangles(
    rng: np.random.Generator,
    occupied: np.ndarray,
    sizes: list[tuple[int, int]],
    row0: int,
    row1: int,
    col0: int,
    col1: int,
) -> Optional[list[tuple[int, int]]]:
    placements: list[Optional[tuple[int, int]]] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        h, w = sizes[idx]
        if row1 - row0 < h or col1 - col0 < w:
            return None
        placed = False
        for _ in range(300):
            r = int(rng.integers(row0, row1 - h + 1))
            c = int(rng.integers(col0, col1 - w + 1))
            if _can_place(occupied, r, c, h, w):
                occupied[r : r + h, c : c + w] = True
                placements[idx] = (r, c)
                placed = True
                break
        if not placed:
            return None
    return [p for p in placements if p is not None]


def _draw_border(grid: np.ndarray, r: int, c: int, h: int, w: int, color: int) -> None:
    grid[r, c : c + w] = color
    grid[r + h - 1, c : c + w] = color
    grid[r : r + h, c] = color
    grid[r : r + h, c + w - 1] = color


def _gen_attempt(rng: np.random.Generator) -> Optional[np.ndarray]:
    palette_count = int(rng.integers(*PALETTE_COUNT_RANGE))
    extra_frames = int(rng.integers(*EXTRA_FRAME_RANGE))
    work_count = palette_count + extra_frames

    border_colors = rng.choice(np.arange(1, 10), size=palette_count, replace=False).tolist()
    remaining = [c for c in range(1, 10) if c not in border_colors]
    fill_colors = rng.choice(np.asarray(remaining), size=palette_count, replace=False).tolist()

    palettes = []
    for border_color, fill_color in zip(border_colors, fill_colors):
        tile = _make_palette_tile(rng, int(fill_color))
        palettes.append(
            {
                "border": int(border_color),
                "tile": tile,
                "outer": (tile.shape[0] + 2, tile.shape[1] + 2),
            }
        )

    frame_colors = border_colors.copy()
    for _ in range(extra_frames):
        frame_colors.append(int(rng.choice(border_colors)))
    rng.shuffle(frame_colors)

    frames = []
    palette_by_color = {spec["border"]: spec["tile"] for spec in palettes}
    for color in frame_colors:
        inner_h, inner_w = _sample_frame_inner(rng, palette_by_color[int(color)].shape)
        frames.append(
            {
                "border": int(color),
                "inner": (inner_h, inner_w),
                "outer": (inner_h + 2, inner_w + 2),
            }
        )

    palette_area = sum(h * w for h, w in (spec["outer"] for spec in palettes))
    frame_area = sum(h * w for h, w in (spec["outer"] for spec in frames))
    max_obj_h = max(max(spec["outer"][0] for spec in palettes), max(spec["outer"][0] for spec in frames))
    total_area = palette_area + frame_area

    grid_h = max(max_obj_h + 4, int(math.ceil(math.sqrt(total_area * 1.9))))
    grid_h += int(rng.integers(0, 5))
    usable_h = grid_h - 2

    left_w = max(
        max(spec["outer"][1] for spec in palettes) + 2,
        int(math.ceil(palette_area / max(4, usable_h - 2))) + 5,
    )
    right_w = max(
        max(spec["outer"][1] for spec in frames) + 2,
        int(math.ceil(frame_area / max(4, usable_h - 2))) + 7,
    )
    gap_w = int(rng.integers(2, 4))
    grid_w = 1 + left_w + gap_w + right_w + 1

    if grid_h > 36 or grid_w > 44:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    left_col0 = 1
    left_col1 = left_col0 + left_w
    right_col0 = left_col1 + gap_w
    right_col1 = right_col0 + right_w

    palette_places = _place_rectangles(
        rng,
        occupied.copy(),
        [spec["outer"] for spec in palettes],
        1,
        grid_h - 1,
        left_col0,
        left_col1,
    )
    if palette_places is None:
        return None

    occupied = np.zeros_like(grid, dtype=bool)
    for (r, c), spec in zip(palette_places, palettes):
        h, w = spec["outer"]
        occupied[r : r + h, c : c + w] = True

    frame_places = _place_rectangles(
        rng,
        occupied,
        [spec["outer"] for spec in frames],
        1,
        grid_h - 1,
        right_col0,
        right_col1,
    )
    if frame_places is None:
        return None

    for (r, c), spec in zip(palette_places, palettes):
        h, w = spec["outer"]
        _draw_border(grid, r, c, h, w, spec["border"])
        grid[r + 1 : r + h - 1, c + 1 : c + w - 1] = spec["tile"]

    expected = grid.copy()
    for (r, c), spec in zip(frame_places, frames):
        inner_h, inner_w = spec["inner"]
        outer_h, outer_w = spec["outer"]
        _draw_border(grid, r, c, outer_h, outer_w, spec["border"])
        _draw_border(expected, r, c, outer_h, outer_w, spec["border"])
        expected[r + 1 : r + 1 + inner_h, c + 1 : c + 1 + inner_w] = _tile_to_shape(
            _minimal_tile(palette_by_color[spec["border"]]),
            inner_h,
            inner_w,
        )

    bg_count = int(np.sum(grid == BG))
    non_bg_counts = [int(np.sum(grid == color)) for color in range(1, 10) if np.any(grid == color)]
    if bg_count <= max(non_bg_counts, default=0):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
