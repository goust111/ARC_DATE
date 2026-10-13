import random
from dataclasses import dataclass

import numpy as np


BG = 0
YELLOW = 4
BORDER_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
PATTERN_COLORS = tuple(c for c in range(1, 10) if c != YELLOW)


@dataclass(frozen=True)
class Rect:
    r0: int
    c0: int
    r1: int
    c1: int
    color: int
    marker_corner: int | None = None

    @property
    def area(self) -> int:
        return (self.r1 - self.r0 + 1) * (self.c1 - self.c0 + 1)


def _boundary_points(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield r0, c
    for r in range(r0 + 1, r1):
        yield r, c1
    for c in range(c1, c0 - 1, -1):
        yield r1, c
    for r in range(r1 - 1, r0, -1):
        yield r, c0


def _solid_rect(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> Rect | None:
    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    vals = [int(grid[r, c]) for r, c in corners]
    if vals.count(YELLOW) != 1:
        return None

    frame_vals = [v for v in vals if v != YELLOW]
    if len(set(frame_vals)) != 1:
        return None
    color = frame_vals[0]
    if color in (BG, YELLOW):
        return None

    marker_pos = corners[vals.index(YELLOW)]
    for r, c in _boundary_points(r0, c0, r1, c1):
        expected = YELLOW if (r, c) == marker_pos else color
        if int(grid[r, c]) != expected:
            return None

    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG):
        return None

    return Rect(r0, c0, r1, c1, color, vals.index(YELLOW))


def _dashed_rect(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> Rect | None:
    corners = [int(grid[r0, c0]), int(grid[r0, c1]), int(grid[r1, c1]), int(grid[r1, c0])]
    if len(set(corners)) != 1:
        return None
    color = corners[0]
    if color in (BG, YELLOW):
        return None

    # The generated dashed frames use the checkerboard phase with all four
    # corners present, so both side lengths between corners are even.
    if (r1 - r0) % 2 or (c1 - c0) % 2:
        return None

    for r, c in _boundary_points(r0, c0, r1, c1):
        expected = color if ((r - r0) + (c - c0)) % 2 == 0 else BG
        if int(grid[r, c]) != expected:
            return None

    if not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
        return None

    return Rect(r0, c0, r1, c1, color)


def _find_frames(grid: np.ndarray) -> tuple[list[Rect], list[Rect]]:
    h, w = grid.shape
    templates: list[Rect] = []
    targets: list[Rect] = []

    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    target = _solid_rect(grid, r0, c0, r1, c1)
                    if target is not None:
                        targets.append(target)

                    template = _dashed_rect(grid, r0, c0, r1, c1)
                    if template is not None:
                        templates.append(template)

    return templates, targets


def _largest_by_color(rects: list[Rect]) -> dict[int, Rect]:
    chosen: dict[int, Rect] = {}
    for rect in rects:
        old = chosen.get(rect.color)
        if old is None or rect.area > old.area:
            chosen[rect.color] = rect
    return chosen


def _rotated(pattern: np.ndarray, corner: int) -> np.ndarray:
    if corner == 0:       # upper-left: no rotation
        return pattern.copy()
    if corner == 1:       # upper-right: clockwise 90
        return np.rot90(pattern, -1)
    if corner == 2:       # lower-right: 180
        return np.rot90(pattern, 2)
    if corner == 3:       # lower-left: clockwise 270
        return np.rot90(pattern, 1)
    raise AssertionError("target marker corner must be one of the four frame corners")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each dashed-frame template into its same-colored solid target after marker-directed rotation.

    Input structure:
    - The background is black, with several one-pixel rectangular frames.
    - Dashed checkerboard frames mark template regions; their interiors contain colored patterns.
    - Solid frames mark target regions; their interiors are black, and exactly one target corner is yellow.
    - A dashed frame and a solid frame with the same non-black border color form one pair.

    Rule:
    - Detect frames by border style, then pair each solid target frame with the dashed template frame of the same border color.
    - Extract the template interior, excluding the dashed border itself.
    - Use the target's yellow corner to choose the rotation: upper-left = 0, upper-right = clockwise 90, lower-right = 180, lower-left = clockwise 270.
    - Paint the rotated template pattern into the target interior while preserving all frames and the original template.

    Key insight:
    The transformation is decoded in three independent steps: border dash pattern selects the source, border color links source to target, and the yellow corner selects the rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates, targets = _find_frames(g)
    template_by_color = _largest_by_color(templates)
    target_by_color = _largest_by_color(targets)
    assert target_by_color, "expected at least one solid target frame"

    out = g.copy()
    for color, target in target_by_color.items():
        assert color in template_by_color, "every target color must have a matching template"
        template = template_by_color[color]

        pattern = g[template.r0 + 1 : template.r1, template.c0 + 1 : template.c1]
        rotated = _rotated(pattern, int(target.marker_corner))
        tr = slice(target.r0 + 1, target.r1)
        tc = slice(target.c0 + 1, target.c1)
        assert rotated.shape == out[tr, tc].shape, "rotated pattern must fit the target interior"
        out[tr, tc] = rotated

    return out


def _make_pattern(size: int, palette: list[int], rng: random.Random) -> np.ndarray:
    coords = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(200):
        pattern = np.zeros((size, size), dtype=int)
        count = rng.randint(max(3, size), min(size * size - 1, size * 2 + 2))
        for r, c in rng.sample(coords, count):
            pattern[r, c] = rng.choice(palette)

        if np.any(pattern != BG) and all(not np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            return pattern

    pattern = np.zeros((size, size), dtype=int)
    pattern[0, 0] = palette[0]
    pattern[1, 0] = palette[1 % len(palette)]
    pattern[1, 1] = palette[2 % len(palette)]
    pattern[-1, -2] = palette[3 % len(palette)]
    return pattern


def _can_place(occupied: np.ndarray, r0: int, c0: int, side: int, gap: int = 1) -> bool:
    h, w = occupied.shape
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h, r1 + gap + 1)
    cc1 = min(w, c1 + gap + 1)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _reserve(occupied: np.ndarray, r0: int, c0: int, side: int, gap: int = 1) -> None:
    h, w = occupied.shape
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h, r1 + gap + 1)
    cc1 = min(w, c1 + gap + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_rect(occupied: np.ndarray, side: int, rng: random.Random) -> tuple[int, int] | None:
    h, w = occupied.shape
    if h < side + 2 or w < side + 2:
        return None
    for _ in range(200):
        r0 = rng.randint(1, h - side - 1)
        c0 = rng.randint(1, w - side - 1)
        if _can_place(occupied, r0, c0, side):
            _reserve(occupied, r0, c0, side)
            return r0, c0
    return None


def _draw_dashed(grid: np.ndarray, r0: int, c0: int, size: int, color: int, pattern: np.ndarray) -> None:
    r1 = r0 + size + 1
    c1 = c0 + size + 1
    for r, c in _boundary_points(r0, c0, r1, c1):
        if ((r - r0) + (c - c0)) % 2 == 0:
            grid[r, c] = color
    grid[r0 + 1 : r1, c0 + 1 : c1] = pattern


def _draw_solid(grid: np.ndarray, r0: int, c0: int, size: int, color: int, marker_corner: int) -> None:
    r1 = r0 + size + 1
    c1 = c0 + size + 1
    for r, c in _boundary_points(r0, c0, r1, c1):
        grid[r, c] = color
    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    mr, mc = corners[marker_corner]
    grid[mr, mc] = YELLOW


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    pair_count = rng.randint(2, 4)
    sizes = [rng.choice((3, 5, 7)) for _ in range(pair_count)]

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    border_colors = rng.sample(list(BORDER_COLORS), pair_count)
    pattern_palette = [c for c in PATTERN_COLORS if c not in border_colors]
    if len(pattern_palette) < 3:
        return None

    corners = list(range(4))
    rng.shuffle(corners)

    for i, (color, size) in enumerate(zip(border_colors, sizes)):
        side = size + 2
        pattern = _make_pattern(size, pattern_palette, rng)

        tpos = _place_rect(occupied, side, rng)
        gpos = _place_rect(occupied, side, rng)
        if tpos is None or gpos is None:
            return None

        _draw_dashed(grid, tpos[0], tpos[1], size, color, pattern)
        _draw_solid(grid, gpos[0], gpos[1], size, color, corners[i % 4])

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to place non-overlapping template and target frames")
