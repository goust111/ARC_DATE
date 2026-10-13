import random

import numpy as np


BLACK = 0
GREY = 5
SPRITE_COLORS = tuple(color for color in range(1, 10) if color != GREY)
DIRECTIONS = ("up", "right", "down", "left")

GRID_HEIGHT_RANGE = (14, 30)
GRID_WIDTH_RANGE = (14, 30)
FRAME_COUNT_OPTIONS = (2, 3, 4)
INTERIOR_SIZES = (3, 4)
GENERATION_RETRIES = 400

# Canonical tip-up sprites. The projection along the tip axis narrows
# monotonically toward the single-cell end, which keeps the direction readable.
BASE_SPRITES = (
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0]], dtype=bool),
    np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == GREY):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h == inner_w and inner_h in INTERIOR_SIZES, "unexpected frame size"

        expected = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in {r0, r1} or c in {c0, c1}
        }
        assert set(coords) == expected, "grey components must be one-pixel rectangular frames"
        frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one grey frame"
    return sorted(frames)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _direction_score(profile: np.ndarray, perp: np.ndarray, tip_line: np.ndarray) -> tuple | None:
    if len(profile) < 2 or profile[0] >= profile[-1]:
        return None
    diffs = np.diff(profile)
    if np.any(diffs < 0):
        return None

    tip_positions = np.flatnonzero(tip_line)
    center = (len(tip_line) - 1) / 2.0
    centeredness = -float(np.mean(np.abs(tip_positions - center)))
    weighted_mass = float(np.dot(np.arange(len(profile)), profile))
    return (
        int(profile[-1] - profile[0]),
        int(np.count_nonzero(diffs > 0)),
        weighted_mass,
        -abs(int(perp[0] - perp[-1])),
        centeredness,
        len(profile),
    )


def _tip_direction(mask: np.ndarray) -> str:
    sprite = _tight_mask(np.asarray(mask, dtype=bool))
    row_counts = sprite.sum(axis=1)
    col_counts = sprite.sum(axis=0)
    candidates: list[tuple[tuple, str]] = []

    for direction, profile, perp, tip_line in (
        ("up", row_counts, col_counts, sprite[0]),
        ("down", row_counts[::-1], col_counts, sprite[-1]),
        ("left", col_counts, row_counts, sprite[:, 0]),
        ("right", col_counts[::-1], row_counts, sprite[:, -1]),
    ):
        score = _direction_score(profile, perp, tip_line)
        if score is not None:
            candidates.append((score, direction))

    assert candidates, "failed to infer a tip direction"
    candidates.sort(reverse=True)
    if len(candidates) > 1:
        assert candidates[0][0] != candidates[1][0], "ambiguous sprite direction"
    return candidates[0][1]


def _marker_cell(frame_bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "up":
        return r0 - 1, mid_c
    if direction == "down":
        return r1 + 1, mid_c
    if direction == "left":
        return mid_r, c0 - 1
    if direction == "right":
        return mid_r, c1 + 1
    raise ValueError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """Place one outside marker on the side pointed to by each fenced sprite's narrow tip.

    Input structure:
    - The background is black.
    - Several grey one-pixel rectangular frames appear, separated from each other.
    - Each frame has a 3x3 or 4x4 interior containing one single-color non-grey sprite.

    Rule:
    - For each frame, inspect the sprite's row and column projection widths.
    - The direction whose projection narrows monotonically toward one end is the tip direction.
    - Paint one pixel of the sprite's color one cell outside the matching side midpoint of the frame.
    - Keep the frame, the sprite, and all other pixels unchanged.

    Key insight:
    The sprite is not copied or moved; only its asymmetric taper matters, because the
    narrow end encodes a cardinal direction that selects where the external marker goes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for frame_bbox in _frame_bboxes(g):
        r0, c0, r1, c1 = frame_bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = sorted(set(map(int, np.unique(inner))) - {BLACK, GREY})
        assert len(colors) == 1, "each frame interior must contain exactly one sprite color"
        sprite_color = colors[0]
        sprite = inner == sprite_color
        pixel_count = int(sprite.sum())
        assert 3 <= pixel_count <= 5, "sprite size must stay within the described range"

        direction = _tip_direction(sprite)
        mr, mc = _marker_cell(frame_bbox, direction)
        assert 0 <= mr < out.shape[0] and 0 <= mc < out.shape[1], "marker must stay on-grid"
        out[mr, mc] = sprite_color

    return out


def _rotation_steps(direction: str) -> int:
    return {"up": 0, "left": 1, "down": 2, "right": 3}[direction]


def _oriented_sprite(base: np.ndarray, direction: str) -> np.ndarray:
    sprite = np.rot90(base, _rotation_steps(direction))
    assert _tip_direction(sprite) == direction
    return sprite


def _interior_choices(count: int) -> list[int]:
    sizes = [3, 4]
    while len(sizes) < count:
        sizes.append(random.choice(INTERIOR_SIZES))
    random.shuffle(sizes)
    return sizes[:count]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, outer_size: int) -> None:
    r1, c1 = r0 + outer_size - 1, c0 + outer_size - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY


def _candidate_positions(blocked: np.ndarray, outer_size: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - outer_size):
        for c0 in range(1, w - outer_size):
            r1, c1 = r0 + outer_size - 1, c0 + outer_size - 1
            if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue
            positions.append((r0, c0))
    return positions


def _shape_options(interior_size: int, direction: str) -> list[np.ndarray]:
    options: list[np.ndarray] = []
    for base in BASE_SPRITES:
        sprite = _oriented_sprite(base, direction)
        if sprite.shape[0] <= interior_size and sprite.shape[1] <= interior_size:
            options.append(sprite)
    assert options, "every interior size should admit at least one sprite"
    return options


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    directions = random.sample(DIRECTIONS, k=frame_count)
    interior_sizes = _interior_choices(frame_count)
    colors = random.sample(SPRITE_COLORS, k=frame_count)

    specs = []
    for direction, interior_size, color in zip(directions, interior_sizes, colors):
        sprite = random.choice(_shape_options(interior_size, direction))
        specs.append(
            {
                "direction": direction,
                "interior_size": interior_size,
                "outer_size": interior_size + 2,
                "color": color,
                "sprite": sprite,
            }
        )

    specs.sort(key=lambda spec: spec["outer_size"], reverse=True)
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[tuple[dict, int, int]] = []

    for spec in specs:
        positions = _candidate_positions(blocked, int(spec["outer_size"]))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        outer_size = int(spec["outer_size"])
        r1, c1 = r0 + outer_size - 1, c0 + outer_size - 1
        blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        placed.append((spec, r0, c0))

    grid = np.full((h, w), BLACK, dtype=int)
    for spec, r0, c0 in placed:
        outer_size = int(spec["outer_size"])
        interior_size = int(spec["interior_size"])
        color = int(spec["color"])
        sprite = np.asarray(spec["sprite"], dtype=bool)

        _draw_frame(grid, r0, c0, outer_size)
        top = r0 + 1 + random.randint(0, interior_size - sprite.shape[0])
        left = c0 + 1 + random.randint(0, interior_size - sprite.shape[1])
        patch = grid[top : top + sprite.shape[0], left : left + sprite.shape[1]]
        patch[sprite] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
