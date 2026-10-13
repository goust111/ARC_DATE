from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 5
ARROW = 9

TILE = 5
INNER = 3
SLOT = 7
MIN_H_GAP = 1
GEN_ATTEMPTS = 400

BORDER_COLORS = (1, 2, 3, 4, 6, 7, 8)
PATTERN_MASKS = (
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
)

ROT_K = {
    "right": 0,
    "up": 1,
    "left": 2,
    "down": 3,
}
BLOCK_CANDIDATES = (
    ("right", (-2, -5)),
    ("left", (-2, 1)),
    ("up", (1, -2)),
    ("down", (-5, -2)),
)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _divider_row(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        color = int(row[0])
        if color == bg or color == ARROW:
            continue
        if bool(np.all(row == color)):
            candidates.append(r)
    if len(candidates) != 1:
        raise ValueError("expected exactly one full-width divider row")
    return int(candidates[0])


def _ring_mask() -> np.ndarray:
    mask = np.zeros((TILE, TILE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


RING_MASK = _ring_mask()


def _valid_template_tile(tile: np.ndarray, bg: int) -> bool:
    if tile.shape != (TILE, TILE):
        return False
    border = int(tile[0, 0])
    if border in (bg, DIVIDER, ARROW):
        return False
    if not bool(np.all(tile[RING_MASK] == border)):
        return False
    if bool(np.any(tile[1:4, 1:4] == border)):
        return False
    inner = tile[1:4, 1:4]
    vals = {int(v) for v in np.unique(inner)}
    if bg not in vals:
        return False
    fg = vals - {bg}
    return len(fg) == 1


def _parse_templates(region: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for r, c in np.argwhere(region == ARROW):
        top = int(r) - 2
        left = int(c) - 5
        if top < 0 or left < 0 or top + TILE > region.shape[0] or left + TILE > region.shape[1]:
            continue
        tile = region[top : top + TILE, left : left + TILE]
        if not _valid_template_tile(tile, bg):
            continue
        border = int(tile[0, 0])
        templates[border] = np.array(tile[1:4, 1:4], copy=True)
    if not templates:
        raise ValueError("no templates found")
    return templates


def _solid_block_tile(tile: np.ndarray, bg: int) -> bool:
    if tile.shape != (TILE, TILE):
        return False
    color = int(tile[0, 0])
    if color in (bg, DIVIDER, ARROW):
        return False
    return bool(np.all(tile == color))


def _parse_blocks(region: np.ndarray, bg: int) -> list[tuple[int, int, int, str]]:
    blocks: list[tuple[int, int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    h, w = region.shape
    for r, c in np.argwhere(region == ARROW):
        matches: list[tuple[int, int, int, str]] = []
        for orient, (dr, dc) in BLOCK_CANDIDATES:
            top = int(r) + dr
            left = int(c) + dc
            if top < 0 or left < 0 or top + TILE > h or left + TILE > w:
                continue
            tile = region[top : top + TILE, left : left + TILE]
            if _solid_block_tile(tile, bg):
                matches.append((top, left, int(tile[0, 0]), orient))
        if len(matches) != 1:
            continue
        top, left, color, orient = matches[0]
        if (top, left) in seen:
            continue
        seen.add((top, left))
        blocks.append((top, left, color, orient))
    if not blocks:
        raise ValueError("no solid blocks found")
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid color block with the matching template pattern, rotated to the block's arrow direction.

    Input structure:
    - bg = black. A single grey horizontal row splits the grid into a template area above and a main canvas below.
    - Each template is a 5x5 framed tile: a colored 1-pixel border surrounds a 3x3 black-and-color motif, and a maroon marker sits immediately to the right of the tile's middle row to define the default orientation as right.
    - The lower canvas contains several solid 5x5 color blocks. Each block uses one template border color and has one adjacent maroon marker on its top, bottom, left, or right side.

    Rule:
    - Read every upper template and map its border color to its 3x3 interior motif.
    - For each lower solid block, use the shared color to select the template, then infer the target orientation from the side containing the maroon marker.
    - Rotate the template motif from right-facing into that target orientation and write it into the block's inner 3x3 area, leaving the surrounding 1-pixel border and the arrow marker unchanged.
    - Return only the lower canvas, i.e. everything below the grey divider row.

    Key insight:
    - The color tells you which motif to use, while the marker position independently tells you how much to rotate that motif before insertion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    divider = _divider_row(g, bg)

    upper = g[:divider, :]
    lower = np.array(g[divider + 1 :, :], copy=True)

    templates = _parse_templates(upper, bg)
    blocks = _parse_blocks(lower, bg)

    for top, left, color, orient in blocks:
        if color not in templates:
            raise ValueError(f"missing template for color {color}")
        rotated = np.rot90(templates[color], ROT_K[orient])
        lower[top + 1 : top + 4, left + 1 : left + 4] = rotated

    return lower


def _mask_to_pattern(mask: np.ndarray, color: int) -> np.ndarray:
    pattern = np.full((INNER, INNER), BG, dtype=int)
    pattern[mask] = int(color)
    return pattern


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[rng.randrange(parts)] += 1
    return values


def _row_origins(count: int, width: int, rng: random.Random) -> list[int]:
    min_width = count * SLOT + max(0, count - 1) * MIN_H_GAP
    if min_width > width:
        raise ValueError("row does not fit")
    extras = _random_partition(width - min_width, count + 1, rng)
    origins: list[int] = []
    x = extras[0]
    for idx in range(count):
        origins.append(int(x))
        x += SLOT
        if idx < count - 1:
            x += MIN_H_GAP + extras[idx + 1]
    return origins


def _lower_row_counts(n_blocks: int, rng: random.Random) -> list[int]:
    if n_blocks <= 3 and rng.random() < 0.45:
        return [n_blocks]
    if n_blocks == 3:
        return list(rng.choice(([2, 1], [1, 2])))
    if n_blocks == 4:
        return list(rng.choice(([2, 2], [3, 1], [1, 3])))
    return list(rng.choice(([3, 2], [2, 3])))


def _render_template(
    canvas: np.ndarray,
    slot_top: int,
    slot_left: int,
    border_color: int,
    pattern_color: int,
    pattern_mask: np.ndarray,
) -> None:
    top = slot_top + 1
    left = slot_left + 1
    canvas[top : top + TILE, left : left + TILE] = int(border_color)
    canvas[top + 1 : top + 4, left + 1 : left + 4] = _mask_to_pattern(pattern_mask, pattern_color)
    canvas[slot_top + 3, slot_left + 6] = ARROW


def _render_block(
    canvas: np.ndarray,
    slot_top: int,
    slot_left: int,
    color: int,
    orient: str,
) -> None:
    top = slot_top + 1
    left = slot_left + 1
    canvas[top : top + TILE, left : left + TILE] = int(color)
    if orient == "right":
        canvas[slot_top + 3, slot_left + 6] = ARROW
    elif orient == "left":
        canvas[slot_top + 3, slot_left + 0] = ARROW
    elif orient == "up":
        canvas[slot_top + 0, slot_left + 3] = ARROW
    elif orient == "down":
        canvas[slot_top + 6, slot_left + 3] = ARROW
    else:
        raise ValueError("unknown orientation")


def _balanced_directions(n: int, rng: random.Random) -> list[str]:
    base = ["right", "down", "left", "up"] * 3
    start = rng.randrange(4)
    dirs = base[start : start + n]
    rng.shuffle(dirs)
    return dirs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.randint(2, 3)
    n_blocks = rng.randint(max(3, n_templates), 5)

    border_colors = rng.sample(BORDER_COLORS, n_templates)
    mask_choices = rng.sample(range(len(PATTERN_MASKS)), n_templates)
    pattern_colors: list[int] = []
    for border in border_colors:
        choices = [c for c in BORDER_COLORS if c != border]
        pattern_colors.append(int(rng.choice(choices)))

    block_colors = border_colors + [int(rng.choice(border_colors)) for _ in range(n_blocks - n_templates)]
    rng.shuffle(block_colors)
    directions = _balanced_directions(n_blocks, rng)

    lower_counts = _lower_row_counts(n_blocks, rng)
    max_row = max(n_templates, max(lower_counts))
    width = max_row * SLOT + (max_row - 1) * MIN_H_GAP + rng.randint(0, 5)
    if width > 30:
        return None

    upper_h = SLOT + rng.randint(0, 3)
    lower_base_h = SLOT * len(lower_counts)
    lower_h = lower_base_h + rng.randint(0, 3)
    total_h = upper_h + 1 + lower_h
    if total_h > 30:
        return None

    upper = np.full((upper_h, width), BG, dtype=int)
    lower = np.full((lower_h, width), BG, dtype=int)

    upper_row_top = rng.randint(0, upper_h - SLOT)
    upper_xs = _row_origins(n_templates, width, rng)
    for idx, slot_left in enumerate(upper_xs):
        _render_template(
            upper,
            upper_row_top,
            slot_left,
            border_colors[idx],
            pattern_colors[idx],
            PATTERN_MASKS[mask_choices[idx]],
        )

    if len(lower_counts) == 1:
        lower_tops = [rng.randint(0, lower_h - SLOT)]
    else:
        extra = lower_h - lower_base_h
        top_pad, between_pad, _bottom_pad = _random_partition(extra, 3, rng)
        lower_tops = [top_pad, top_pad + SLOT + between_pad]

    cursor = 0
    for row_idx, count in enumerate(lower_counts):
        xs = _row_origins(count, width, rng)
        for slot_left in xs:
            _render_block(
                lower,
                lower_tops[row_idx],
                slot_left,
                block_colors[cursor],
                directions[cursor],
            )
            cursor += 1

    grid = np.vstack(
        [
            upper,
            np.full((1, width), DIVIDER, dtype=int),
            lower,
        ]
    )

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != lower.shape:
        return None
    if len({tuple(np.rot90(PATTERN_MASKS[m], k).ravel()) for m in mask_choices for k in range(4)}) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
