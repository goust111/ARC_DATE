from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox, colors_present, draw_line, find_components, get_bg, is_contiguous, paste, random_free_location
import random


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    b = bbox(mask)
    if b is None:
        return None
    r0, c0, r1, c1 = b
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    assert src_len >= 1 and dst_len >= 1, "axis lengths must be positive"
    if src_len == 1:
        assert dst_len == 1, "a one-pixel motif axis maps to a one-pixel interior axis"
        return 0
    assert dst_len >= 2, "multi-pixel motif axis needs a multi-pixel interior axis"
    return int(round(int(i) * (dst_len - 1) / (src_len - 1)))


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke.astype(bool)
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    ph, pw = padded.shape

    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke.astype(bool) | holes


def _stretch_sprite(sprite: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = sprite.shape
    dst_h, dst_w = target_shape
    assert sprite.any(), "expected a nonempty sprite"

    stroke = np.zeros((dst_h, dst_w), dtype=int)
    coords = np.argwhere(sprite)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for sr, sc in coords:
        tr = _scaled_index(int(sr), src_h, dst_h)
        tc = _scaled_index(int(sc), src_w, dst_w)
        mapped[(int(sr), int(sc))] = (tr, tc)
        stroke[tr, tc] = 1

    for sr, sc in coords:
        sr = int(sr)
        sc = int(sc)
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) in mapped:
                r0, c0 = mapped[(sr, sc)]
                r1, c1 = mapped[(nr, nc)]
                draw_line(stroke, r0, c0, end_r=r1, end_c=c1, color=1)

    return _fill_enclosed(stroke.astype(bool))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the small colored sprite into the hollow frame by stretching its shape to the frame interior.

    Input structure:
    - The background is the dominant color.
    - There are exactly two foreground colors: one forms a complete rectangular
      one-pixel-thick frame, and the other forms one 4-connected sprite outside it.
    - The frame interior is initially background.

    Rule:
    - Clear the original sprite from its outside location.
    - Map the sprite bbox to the frame interior bbox, sending bbox extremes to
      interior extremes.
    - Draw straight horizontal or vertical segments between mapped positions of
      adjacent sprite pixels, then fill any enclosed background region inside
      those drawn segments.

    Key insight:
    The outside sprite is a miniature line/fill diagram for the frame interior,
    not a bitmap to scale by block replication.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected one frame color and one sprite color"

    frame_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        frame_bbox = _rect_frame_bbox(g == int(color))
        if frame_bbox is not None:
            frame_candidates.append((int(color), frame_bbox))
    assert len(frame_candidates) == 1, "expected a unique rectangular frame"

    frame_color, (fr0, fc0, fr1, fc1) = frame_candidates[0]
    sprite_color = next(color for color in colors if color != frame_color)
    sprite_components = find_components(g, bg=bg, colors=[sprite_color], connectivity=4)
    assert len(sprite_components) == 1, "expected a single 4-connected sprite"
    frame_region = np.zeros_like(g, dtype=bool)
    frame_region[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    assert not np.any((g == sprite_color) & frame_region), "sprite must start outside the frame"

    sr0, sc0, sr1, sc1 = sprite_components[0]["bbox"]
    sprite = g[sr0 : sr1 + 1, sc0 : sc1 + 1] == sprite_color

    ir0, ic0 = fr0 + 1, fc0 + 1
    ir1, ic1 = fr1 - 1, fc1 - 1
    assert ir0 <= ir1 and ic0 <= ic1, "frame must have an interior"
    interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
    assert np.all(interior == bg), "frame interior must start empty"

    stretched = _stretch_sprite(sprite, (ir1 - ir0 + 1, ic1 - ic0 + 1))
    out = g.copy()
    out[g == sprite_color] = bg
    out[ir0 : ir1 + 1, ic0 : ic1 + 1][stretched] = sprite_color
    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and observed fixed colors.
_GRID_SHAPE = (10, 10)
_COLOR_COUNT = 10
_BG = 7
_FRAME_COLOR = 6

# Frame size bounds and retry budget.
_FRAME_SIZE_MIN = 3
_FRAME_SIZE_MAX = 8
_SINGLE_PIXEL_FRAME_SIZE = 3
_MAX_GENERATION_ATTEMPTS = 300

# Sprite color pool and source masks.
_SPRITE_COLORS = tuple(c for c in range(_COLOR_COUNT) if c not in {_BG, _FRAME_COLOR})
_SPRITE_SINGLE = np.ones((1, 1), dtype=bool)
_SPRITE_PATTERNS_2X2 = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
]

def _outside_locations(
    frame_bbox: tuple[int, int, int, int], patch_shape: tuple[int, int]
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    ph, pw = patch_shape
    locations: list[tuple[int, int]] = []
    for r in range(_GRID_SHAPE[0] - ph + 1):
        for c in range(_GRID_SHAPE[1] - pw + 1):
            pr0, pc0, pr1, pc1 = r, c, r + ph - 1, c + pw - 1
            separated = pr1 < fr0 or pr0 > fr1 or pc1 < fc0 or pc0 > fc1
            if separated:
                locations.append((r, c))
    return locations

def _frame_positions(size: int, patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r in range(_GRID_SHAPE[0] - size + 1):
        for c in range(_GRID_SHAPE[1] - size + 1):
            frame_bbox = (r, c, r + size - 1, c + size - 1)
            if _outside_locations(frame_bbox, patch_shape):
                positions.append((r, c))
    return positions

def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    r1, c1 = r0 + size - 1, c0 + size - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color

def _sample_sprite(size: int) -> np.ndarray:
    return _SPRITE_SINGLE.copy() if size == _SINGLE_PIXEL_FRAME_SIZE else random.choice(_SPRITE_PATTERNS_2X2).copy()

def _sample_case() -> tuple[int, int, np.ndarray]:
    sprite_color = random.choice(_SPRITE_COLORS)
    size = random.randint(_FRAME_SIZE_MIN, _FRAME_SIZE_MAX)
    return sprite_color, size, _sample_sprite(size)

def _build_candidate_grids(
    frame_bbox: tuple[int, int, int, int], sprite: np.ndarray, sprite_color: int
) -> tuple[np.ndarray, np.ndarray] | None:
    fr0, fc0, fr1, fc1 = frame_bbox
    stretched = _stretch_sprite(sprite, (fr1 - fr0 - 1, fc1 - fc0 - 1))
    if not np.any(stretched):
        return None

    out = np.full(_GRID_SHAPE, _BG, dtype=int)
    _draw_frame(out, fr0, fc0, fr1 - fr0 + 1, _FRAME_COLOR)
    out[fr0 + 1 : fr1, fc0 + 1 : fc1][stretched] = sprite_color

    inp = out.copy()
    inp[fr0 + 1 : fr1, fc0 + 1 : fc1] = _BG

    locations = _outside_locations(frame_bbox, sprite.shape)
    if not locations:
        return None
    sr, sc = random.choice(locations)
    patch = np.where(sprite, sprite_color, _BG)
    paste(inp, patch, sr, sc, transparent=_BG)
    return inp, out

def _gen_attempt() -> np.ndarray | None:
    sprite_color, size, sprite = _sample_case()

    frame_positions = _frame_positions(size, sprite.shape)
    if not frame_positions:
        return None
    fr0, fc0 = random.choice(frame_positions)
    frame_bbox = (fr0, fc0, fr0 + size - 1, fc0 + size - 1)

    candidate = _build_candidate_grids(frame_bbox, sprite, sprite_color)
    if candidate is None:
        return None
    inp, out = candidate

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None

def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 465b7d93 input")
