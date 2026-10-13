from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
BODY_COLORS = tuple(range(1, 10))
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_ATTEMPTS = 300
GEOMETRY_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 80


@dataclass
class Square:
    top: int
    left: int
    size: int
    color: int
    marker_dir: tuple[int, int] | None

    @property
    def bottom(self) -> int:
        return self.top + self.size - 1

    @property
    def right(self) -> int:
        return self.left + self.size - 1

    def shift(self, direction: tuple[int, int], steps: int) -> None:
        dr, dc = direction
        self.top += dr * steps
        self.left += dc * steps


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_squares(grid: np.ndarray) -> tuple[int, list[Square]]:
    bg = _get_bg(grid)
    comps = _components(grid != bg)
    assert comps, "expected at least one foreground square"

    squares: list[Square] = []
    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        size = int(r1 - r0 + 1)
        assert size == int(c1 - c0 + 1), "foreground objects must be axis-aligned squares"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (size, size)
        assert coords.shape[0] == size * size, "each object must fill its square bbox"
        assert np.all(patch != bg), "square interiors must stay non-background"

        colors, counts = np.unique(patch, return_counts=True)
        order = np.argsort(counts)[::-1]
        colors = colors[order]
        counts = counts[order]
        assert len(colors) == 1 or int(counts[0]) > int(counts[1]), "square body color must be unique"

        body_color = int(colors[0])
        marker_cells = np.argwhere(patch != body_color)
        marker_dir = None
        if len(marker_cells) == 1:
            assert size >= 3 and size % 2 == 1, "rudder squares need an odd side length"
            center = np.array([size // 2, size // 2], dtype=int)
            delta = marker_cells[0] - center
            assert abs(int(delta[0])) + abs(int(delta[1])) == 1, "marker must touch the center cell"
            marker_dir = (int(delta[0]), int(delta[1]))
        else:
            assert len(marker_cells) == 0, "expected either a pure square or one embedded rudder"

        squares.append(Square(int(r0), int(c0), size, body_color, marker_dir))

    return bg, squares


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _wall_gap(square: Square, direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    dr, dc = direction
    if dr < 0:
        return square.top
    if dr > 0:
        return h - 1 - square.bottom
    if dc < 0:
        return square.left
    return w - 1 - square.right


def _collision_gap(mover: Square, other: Square, direction: tuple[int, int]) -> int | None:
    dr, dc = direction
    if dr:
        if not _overlap(mover.left, mover.right, other.left, other.right):
            return None
        if dr > 0:
            if other.top <= mover.bottom:
                return None
            return other.top - mover.bottom - 1
        if other.bottom >= mover.top:
            return None
        return mover.top - other.bottom - 1

    if not _overlap(mover.top, mover.bottom, other.top, other.bottom):
        return None
    if dc > 0:
        if other.left <= mover.right:
            return None
        return other.left - mover.right - 1
    if other.right >= mover.left:
        return None
    return mover.left - other.right - 1


def _next_event(
    squares: list[Square], active_idx: int, direction: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int | None]:
    wall_gap = _wall_gap(squares[active_idx], direction, shape)
    best_gap: int | None = None
    blockers: list[int] = []

    for idx, other in enumerate(squares):
        if idx == active_idx:
            continue
        gap = _collision_gap(squares[active_idx], other, direction)
        if gap is None:
            continue
        if best_gap is None or gap < best_gap:
            best_gap = gap
            blockers = [idx]
        elif gap == best_gap:
            blockers.append(idx)

    if best_gap is None or wall_gap < best_gap:
        return wall_gap, None

    assert best_gap <= wall_gap, "collision must happen before the wall"
    assert len(blockers) == 1, "relay target must be unambiguous"
    return best_gap, blockers[0]


def _simulate_relays(squares: list[Square], shape: tuple[int, int]) -> None:
    consumed: set[int] = set()
    for idx, square in enumerate(squares):
        if square.marker_dir is None or idx in consumed:
            continue
        direction = square.marker_dir
        active_idx = idx
        consumed.add(idx)
        while True:
            steps, blocker = _next_event(squares, active_idx, direction, shape)
            squares[active_idx].shift(direction, steps)
            if blocker is None:
                break
            assert blocker not in consumed, "relay chains must stay independent"
            consumed.add(blocker)
            active_idx = blocker


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each marked square in its encoded direction, relaying the motion through any square it hits until the front square reaches the border.

    Input structure:
    - The dominant background is black.
    - Foreground objects are separated filled squares, each monochrome except that some squares contain exactly one off-color interior pixel.
    - Whenever present, that off-color pixel sits one step above, below, left, or right of the square center and acts as a rudder.

    Rule:
    - Treat the majority color of each square as its body color and the offset interior pixel as a one-time travel direction.
    - A marked square slides along that direction until the next step would leave the grid or overlap another square; on collision it stops flush against the struck square and transfers the same motion to that square.
    - After every relay finishes, repaint each square as a solid block in its body color, removing all rudder pixels.

    Key insight:
    - The marker only identifies the starter and direction of a collision chain; the final output is obtained by simulating whole-square relay motion, not by preserving the marker colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, squares = _parse_squares(g)
    _simulate_relays(squares, g.shape)

    out = np.full_like(g, bg)
    for square in squares:
        patch = out[square.top : square.top + square.size, square.left : square.left + square.size]
        assert patch.shape == (square.size, square.size), "final square left the grid"
        assert np.all(patch == bg), "final squares must not overlap"
        patch[:] = square.color
    return out


def _sample_chain_lengths(chain_count: int) -> list[int] | None:
    for _ in range(GEOMETRY_ATTEMPTS):
        lengths = [random.randint(1, 3) for _ in range(chain_count)]
        relay_idx = random.randrange(chain_count)
        lengths[relay_idx] = random.randint(2, 4)
        if chain_count > 1 and random.random() < 0.75:
            singleton_candidates = [idx for idx in range(chain_count) if idx != relay_idx]
            lengths[random.choice(singleton_candidates)] = 1
        if sum(lengths) <= len(BODY_COLORS):
            return lengths
    return None


def _sample_geometry(
    height: int, width: int, direction: tuple[int, int], length: int
) -> tuple[int, list[int], int] | None:
    axis_cap = width if direction[1] else height
    cross_cap = height if direction[1] else width
    size_choices = [
        size
        for size in (3, 5, 7)
        if size <= cross_cap - 2 and length * size + (length - 1) + 1 <= axis_cap
    ]
    if not size_choices:
        return None

    for _ in range(GEOMETRY_ATTEMPTS):
        size = random.choice(size_choices)
        gaps = [random.randint(1, 4) for _ in range(length - 1)]
        wall_gap = random.randint(1, 4)
        if length * size + sum(gaps) + wall_gap <= axis_cap:
            return size, gaps, wall_gap
    return None


def _axis_positions(axis_cap: int, direction_sign: int, size: int, gaps: list[int], wall_gap: int) -> list[int]:
    length = len(gaps) + 1
    positions = [0] * length
    positions[-1] = axis_cap - wall_gap - size if direction_sign > 0 else wall_gap
    for idx in range(length - 2, -1, -1):
        if direction_sign > 0:
            positions[idx] = positions[idx + 1] - size - gaps[idx]
        else:
            positions[idx] = positions[idx + 1] + size + gaps[idx]
    return positions


def _corridor_area(height: int, width: int, spec: dict) -> int:
    direction = spec["direction"]
    size = spec["size"]
    axis_positions = spec["axis_positions"]

    if direction[1]:
        c0 = min(axis_positions) if direction[1] > 0 else 0
        c1 = width - 1 if direction[1] > 0 else max(pos + size - 1 for pos in axis_positions)
        return (size + 2) * (c1 - c0 + 1)

    r0 = min(axis_positions) if direction[0] > 0 else 0
    r1 = height - 1 if direction[0] > 0 else max(pos + size - 1 for pos in axis_positions)
    return (size + 2) * (r1 - r0 + 1)


def _place_chain(reserved: np.ndarray, spec: dict) -> list[tuple[int, int]] | None:
    height, width = reserved.shape
    direction = spec["direction"]
    size = spec["size"]
    axis_positions = spec["axis_positions"]

    if direction[1]:
        c0 = min(axis_positions) if direction[1] > 0 else 0
        c1 = width - 1 if direction[1] > 0 else max(pos + size - 1 for pos in axis_positions)
        top_lo = 1 if height - size >= 3 else 0
        top_hi = height - size - 1 if height - size >= 3 else height - size
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(top_lo, top_hi)
            zr0 = max(0, top - 1)
            zr1 = min(height - 1, top + size)
            zc0 = max(0, c0 - 1)
            zc1 = min(width - 1, c1 + 1)
            if reserved[zr0 : zr1 + 1, zc0 : zc1 + 1].any():
                continue
            reserved[zr0 : zr1 + 1, zc0 : zc1 + 1] = True
            return [(top, left) for left in axis_positions]
        return None

    r0 = min(axis_positions) if direction[0] > 0 else 0
    r1 = height - 1 if direction[0] > 0 else max(pos + size - 1 for pos in axis_positions)
    left_lo = 1 if width - size >= 3 else 0
    left_hi = width - size - 1 if width - size >= 3 else width - size
    for _ in range(PLACEMENT_ATTEMPTS):
        left = random.randint(left_lo, left_hi)
        zr0 = max(0, r0 - 1)
        zr1 = min(height - 1, r1 + 1)
        zc0 = max(0, left - 1)
        zc1 = min(width - 1, left + size)
        if reserved[zr0 : zr1 + 1, zc0 : zc1 + 1].any():
            continue
        reserved[zr0 : zr1 + 1, zc0 : zc1 + 1] = True
        return [(top, left) for top in axis_positions]
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    chain_count = random.choices([1, 2, 3], weights=[1, 3, 2])[0]
    lengths = _sample_chain_lengths(chain_count)
    if lengths is None:
        return None

    directions = random.sample(DIRS, chain_count)
    specs: list[dict] = []
    for direction, length in zip(directions, lengths):
        geometry = _sample_geometry(height, width, direction, length)
        if geometry is None:
            return None
        size, gaps, wall_gap = geometry
        axis_cap = width if direction[1] else height
        axis_positions = _axis_positions(axis_cap, direction[0] + direction[1], size, gaps, wall_gap)
        if min(axis_positions) < 0 or max(axis_positions) + size > axis_cap:
            return None
        specs.append(
            {
                "direction": direction,
                "size": size,
                "gaps": gaps,
                "wall_gap": wall_gap,
                "axis_positions": axis_positions,
            }
        )

    specs.sort(key=lambda spec: _corridor_area(height, width, spec), reverse=True)
    reserved = np.zeros((height, width), dtype=bool)
    for spec in specs:
        positions = _place_chain(reserved, spec)
        if positions is None:
            return None
        spec["positions"] = positions

    total_objects = sum(len(spec["positions"]) for spec in specs)
    body_colors = random.sample(BODY_COLORS, total_objects)
    marker_palette = [color for color in BODY_COLORS if color not in body_colors]
    if not marker_palette:
        marker_palette = list(BODY_COLORS)

    grid = np.full((height, width), BG, dtype=int)
    color_idx = 0
    for spec in specs:
        direction = spec["direction"]
        size = spec["size"]
        positions = spec["positions"]
        for obj_idx, (top, left) in enumerate(positions):
            body_color = body_colors[color_idx]
            color_idx += 1
            grid[top : top + size, left : left + size] = body_color
            if obj_idx == 0:
                center_r = top + size // 2
                center_c = left + size // 2
                marker_r = center_r + direction[0]
                marker_c = center_c + direction[1]
                marker_choices = [color for color in marker_palette if color != body_color]
                if not marker_choices:
                    marker_choices = [color for color in BODY_COLORS if color != body_color]
                grid[marker_r, marker_c] = random.choice(marker_choices)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
