from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple
from numpy.random import default_rng

import numpy as np

from arc_utils import colors_present, get_bg


class Sprite(NamedTuple):
    body: int
    marker: int
    top: int
    left: int
    marker_rc: tuple[int, int]
    mask: np.ndarray
    body_mask: np.ndarray


def _corners(h: int, w: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _corner_labels(g: np.ndarray, bg: int) -> tuple[set[int], dict[int, tuple[int, int]]]:
    labels: set[int] = set()
    label_corner: dict[int, tuple[int, int]] = {}
    h, w = g.shape
    for r, c in _corners(h, w):
        color = int(g[r, c])
        if color == bg:
            continue
        assert color not in label_corner, "each marker color must appear in exactly one corner"
        labels.add(color)
        label_corner[color] = (r, c)
    assert labels, "expected at least one corner marker"
    return labels, label_corner


def _find_sprites(g: np.ndarray, bg: int, labels: set[int]) -> list[Sprite]:
    h, w = g.shape
    sprites: list[Sprite] = []
    body_colors = set(colors_present(g, exclude={bg})) - labels
    assert body_colors, "expected non-corner body colors"

    for body in sorted(body_colors):
        body_pts = np.argwhere(g == body)
        assert body_pts.size, "body color must be present"
        r0, c0 = body_pts.min(axis=0)
        r1, c1 = body_pts.max(axis=0)
        assert r1 - r0 < 5 and c1 - c0 < 5, "each body color must fit in one 5x5 window"

        candidates: list[Sprite] = []
        top_min = max(0, int(r1) - 4)
        top_max = min(int(r0), h - 5)
        left_min = max(0, int(c1) - 4)
        left_max = min(int(c0), w - 5)
        for top in range(top_min, top_max + 1):
            for left in range(left_min, left_max + 1):
                window = g[top : top + 5, left : left + 5]
                other_colors = set(colors_present(window, exclude={bg, body}))
                if len(other_colors) != 1:
                    continue
                marker = other_colors.pop()
                if marker not in labels:
                    continue
                if int(np.count_nonzero(window == marker)) != 1:
                    continue
                if int(np.count_nonzero(window == body)) != len(body_pts):
                    continue

                marker_pos = np.argwhere(window == marker)
                assert len(marker_pos) == 1
                mr, mc = map(int, marker_pos[0])
                candidates.append(
                    Sprite(
                        body=int(body),
                        marker=int(marker),
                        top=int(top),
                        left=int(left),
                        marker_rc=(mr, mc),
                        mask=window != bg,
                        body_mask=window == body,
                    )
                )

        assert len(candidates) == 1, "each body color must define a unique marked 5x5 sprite"
        sprites.append(candidates[0])

    return sprites


def _owner_sets(group: list[Sprite]) -> list[list[set[int]]]:
    owners = [[set() for _c in range(5)] for _r in range(5)]
    for i, sprite in enumerate(group):
        for r, c in np.argwhere(sprite.mask):
            owners[int(r)][int(c)].add(i)
    return owners


def _run_reaches_target_edge(
    owners: list[list[set[int]]], owner: int, r: int, c: int, *, top_corner: bool
) -> bool:
    if top_corner:
        rr = r
        while rr >= 0 and owner in owners[rr][c]:
            rr -= 1
        return rr == -1

    rr = r
    while rr < 5 and owner in owners[rr][c]:
        rr += 1
    return rr == 5


def _corner_composite_mask(group: list[Sprite], *, top_corner: bool) -> np.ndarray:
    owners = _owner_sets(group)
    keep = np.array([[bool(owners[r][c]) for c in range(5)] for r in range(5)], dtype=bool)

    # A one-sprite vertical tail hidden behind horizontal neighbors from other sprites
    # is not part of the visible composite unless that run reaches the target edge.
    if top_corner:
        rows = range(1, 4)
        away_step = 1
        toward = lambda r: r - 1
        in_bounds = lambda r: r < 5
    else:
        rows = range(3, 0, -1)
        away_step = -1
        toward = lambda r: r + 1
        in_bounds = lambda r: r >= 0

    for r in rows:
        for c in range(1, 4):
            if not keep[r, c] or len(owners[r][c]) != 1:
                continue
            owner = next(iter(owners[r][c]))
            if _run_reaches_target_edge(owners, owner, r, c, top_corner=top_corner):
                continue
            if owner not in owners[toward(r)][c]:
                continue
            if not owners[r][c - 1] or not owners[r][c + 1]:
                continue
            if owner in owners[r][c - 1] or owner in owners[r][c + 1]:
                continue

            rr = r
            while in_bounds(rr) and keep[rr, c] and owners[rr][c] == {owner}:
                keep[rr, c] = False
                rr += away_step

    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each 5x5 sprite with its corner marker color and stamp same-marker composite masks into the matching corner.

    Input structure:
    - A 30x30 grid has one dominant background color.
    - Non-background corner pixels are marker labels.
    - Every other foreground color is the body of one 5x5 sprite containing exactly one pixel of a corner marker color.

    Rule:
    - For each sprite, recolor body pixels to the sprite marker color and recolor the marker pixel back to its body color.
    - If a marker lies on the side facing its matching corner and another same-marker sprite has a body pixel at that same local cell, use that visible body color for the marker pixel.
    - For each corner marker, overlay the masks of all sprites with that marker color into the corner 5x5 block and paint the visible composite with the marker color.

    Key insight:
    The corner pixels are labels that group 5x5 sprites by their embedded marker color; each labeled corner shows the grouped sprite silhouette.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    labels, label_corner = _corner_labels(g, bg)
    sprites = _find_sprites(g, bg, labels)

    by_marker: dict[int, list[Sprite]] = defaultdict(list)
    for sprite in sprites:
        by_marker[sprite.marker].append(sprite)

    out = g.copy()
    for sprite in sprites:
        window = g[sprite.top : sprite.top + 5, sprite.left : sprite.left + 5]
        patch = out[sprite.top : sprite.top + 5, sprite.left : sprite.left + 5]
        patch[window == sprite.body] = sprite.marker

        mr, mc = sprite.marker_rc
        replacement = sprite.body
        _corner_r, corner_c = label_corner[sprite.marker]
        side_col = 0 if corner_c == 0 else 4
        if mc == side_col:
            overlaps = [
                other.body
                for other in by_marker[sprite.marker]
                if other is not sprite and bool(other.body_mask[mr, mc])
            ]
            if overlaps:
                assert len(overlaps) == 1, "side marker overlap must identify one visible body color"
                replacement = overlaps[0]
        patch[mr, mc] = replacement

    for r, c in _corners(h, w):
        label = int(g[r, c])
        if label == bg:
            continue
        group = by_marker[label]
        assert group, "each corner marker must have at least one matching sprite"
        mask = _corner_composite_mask(group, top_corner=(r == 0))
        rows = slice(0, 5) if r == 0 else slice(h - 5, h)
        cols = slice(0, 5) if c == 0 else slice(w - 5, w)
        patch = out[rows, cols]
        patch[mask] = label

    return out

# --------------- generator (att1) ---------------

# Canvas and sprite geometry.
GRID_H = 30
GRID_W = 30
SPRITE_SIZE = 5
PLACEMENT_GAP = 1

# Marker and sprite counts.
NUM_CORNERS_OPTIONS = (2, 3)
NUM_SPRITES_OPTIONS = (4, 5)
MIN_GROUP_SPRITES = 1
MAX_GROUP_SPRITES = 3

# Sprite mask sampling: foreground includes the one marker pixel.
MIN_BODY_PIXELS = 4
MAX_BODY_PIXELS = 15
MIN_FOREGROUND_PIXELS = MIN_BODY_PIXELS + 1
MAX_FOREGROUND_PIXELS_EXCLUSIVE = MAX_BODY_PIXELS + 2

# Retry budgets.
DISTRIBUTION_ATTEMPTS = 100
SPRITE_MASK_ATTEMPTS = 200
GENERATE_ATTEMPTS = 300

ARC_COLORS = list(range(10))
CORNER_POSITIONS = (
    (0, 0),
    (0, GRID_W - 1),
    (GRID_H - 1, 0),
    (GRID_H - 1, GRID_W - 1),
)
SPRITE_CELLS = [(r, c) for r in range(SPRITE_SIZE) for c in range(SPRITE_SIZE)]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    """Generate a single valid input grid for the b74ca5d1 task."""
    all_colors = ARC_COLORS.copy()
    rng.shuffle(all_colors)
    bg = all_colors[0]

    num_corners = rng.choice(NUM_CORNERS_OPTIONS)
    chosen_corner_indices = rng.choice(4, size=num_corners, replace=False)
    chosen_corners = [CORNER_POSITIONS[i] for i in chosen_corner_indices]
    marker_colors = all_colors[1:1 + num_corners]

    num_sprites = rng.choice(NUM_SPRITES_OPTIONS)
    distribution = _distribute_sprites(rng, num_sprites, num_corners)
    if distribution is None:
        return None

    body_colors = all_colors[1 + num_corners: 1 + num_corners + num_sprites]
    if len(body_colors) < num_sprites:
        return None

    sprites = _build_sprites(rng, distribution, marker_colors, body_colors, chosen_corners)
    positions = _place_sprites(rng, len(sprites))
    if positions is None:
        return None

    grid = _render_grid(bg, sprites, positions, marker_colors, chosen_corners)
    return grid if _solver_accepts(grid) else None


def _build_sprites(rng, distribution, marker_colors, body_colors, chosen_corners):
    """Return sprite dictionaries grouped by the sampled marker distribution."""
    sprites = []
    color_idx = 0
    for marker_idx, count in enumerate(distribution):
        marker_color = marker_colors[marker_idx]
        corner_r, corner_c = chosen_corners[marker_idx]
        for _ in range(count):
            body_color = body_colors[color_idx]
            color_idx += 1
            mask, marker_rc = _gen_sprite_mask(rng)
            sprites.append({
                'body_color': body_color,
                'marker_color': marker_color,
                'mask': mask,
                'marker_rc': marker_rc,
                'corner': (corner_r, corner_c),
            })
    return sprites


def _corner_window_origin(cr, cc):
    return (
        0 if cr == 0 else GRID_H - SPRITE_SIZE,
        0 if cc == 0 else GRID_W - SPRITE_SIZE,
    )


def _forbidden_corner_mask():
    """Reserve every 5x5 corner window for corner composites."""
    forbidden = np.zeros((GRID_H, GRID_W), dtype=bool)
    for cr, cc in CORNER_POSITIONS:
        top, left = _corner_window_origin(cr, cc)
        forbidden[top:top + SPRITE_SIZE, left:left + SPRITE_SIZE] = True
    return forbidden


def _place_sprites(rng, sprite_count):
    """Find non-overlapping 5x5 placements with a one-cell reserved gap."""
    occupied = _forbidden_corner_mask()
    positions = []
    for _ in range(sprite_count):
        pos = _find_position(rng, occupied)
        if pos is None:
            return None
        top, left = pos
        positions.append((top, left))
        r_lo = max(0, top - PLACEMENT_GAP)
        r_hi = min(GRID_H, top + SPRITE_SIZE + PLACEMENT_GAP)
        c_lo = max(0, left - PLACEMENT_GAP)
        c_hi = min(GRID_W, left + SPRITE_SIZE + PLACEMENT_GAP)
        occupied[r_lo:r_hi, c_lo:c_hi] = True
    return positions


def _render_grid(bg, sprites, positions, marker_colors, chosen_corners):
    """Paint sampled sprites and their corner marker pixels onto the canvas."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    for sprite_info, (top, left) in zip(sprites, positions):
        _paint_sprite(grid, sprite_info, top, left)

    for idx, (cr, cc) in enumerate(chosen_corners):
        grid[cr, cc] = marker_colors[idx]
    return grid


def _paint_sprite(grid, sprite_info, top, left):
    window = grid[top:top + SPRITE_SIZE, left:left + SPRITE_SIZE]
    window[sprite_info['mask']] = sprite_info['body_color']
    mr, mc = sprite_info['marker_rc']
    window[mr, mc] = sprite_info['marker_color']


def _solver_accepts(grid):
    try:
        main(grid)
    except (AssertionError, Exception):
        return False
    return True


def _distribute_sprites(rng, num_sprites, num_markers):
    """Distribute num_sprites among num_markers groups, each getting 1-3."""
    if num_markers == 1:
        if MIN_GROUP_SPRITES <= num_sprites <= MAX_GROUP_SPRITES:
            return [num_sprites]
        return None

    for _ in range(DISTRIBUTION_ATTEMPTS):
        parts = []
        remaining = num_sprites
        for i in range(num_markers - 1):
            later_markers = num_markers - 1 - i
            max_val = min(MAX_GROUP_SPRITES, remaining - later_markers)
            min_val = max(MIN_GROUP_SPRITES, remaining - MAX_GROUP_SPRITES * later_markers)
            if min_val > max_val:
                break
            val = rng.integers(min_val, max_val + 1)
            parts.append(val)
            remaining -= val
        else:
            if MIN_GROUP_SPRITES <= remaining <= MAX_GROUP_SPRITES:
                parts.append(remaining)
                return parts
    return None


def _gen_sprite_mask(rng):
    """Generate a 5x5 foreground mask plus one marker pixel position."""
    for _ in range(SPRITE_MASK_ATTEMPTS):
        num_pixels = rng.integers(MIN_FOREGROUND_PIXELS, MAX_FOREGROUND_PIXELS_EXCLUSIVE)
        all_positions = SPRITE_CELLS.copy()
        rng.shuffle(all_positions)
        chosen = all_positions[:num_pixels]

        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True

        marker_idx = rng.integers(0, num_pixels)
        marker_rc = chosen[marker_idx]
        body_count = num_pixels - 1
        if MIN_BODY_PIXELS <= body_count <= MAX_BODY_PIXELS:
            return mask, marker_rc

    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    mask[0:2, 0:3] = True
    return mask, (0, 0)


def _find_position(rng, occupied):
    """Find a random free 5x5 position on the grid."""
    valid = []
    for top in range(GRID_H - SPRITE_SIZE + 1):
        for left in range(GRID_W - SPRITE_SIZE + 1):
            if not occupied[top:top + SPRITE_SIZE, left:left + SPRITE_SIZE].any():
                valid.append((top, left))

    if not valid:
        return None

    idx = rng.integers(0, len(valid))
    return valid[idx]


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GENERATE_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATE_ATTEMPTS} attempts")
