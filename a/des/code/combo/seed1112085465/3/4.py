from __future__ import annotations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
_RNG = np.random.default_rng()

# Generation limits.
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX_EXCLUSIVE = 6
MAX_GRID_SIDE = 30
MAX_OUTPUT_HEIGHT = 28
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _sprite_from_color(upper: np.ndarray, color: int) -> np.ndarray:
    mask = upper == color
    comps = _components(mask)
    assert len(comps) == 1, "each control color must map to exactly one object"
    coords = np.asarray(comps[0], dtype=int)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return sprite


def _alignment_col(out_w: int, sprite_w: int, sprite_h: int) -> int:
    if sprite_w > sprite_h:
        return 0
    if sprite_h > sprite_w:
        return out_w - sprite_w
    return (out_w - sprite_w) // 2


def _stack_sprites(sprites: list[np.ndarray]) -> np.ndarray:
    assert sprites
    out_h = sum(sprite.shape[0] for sprite in sprites) + len(sprites) - 1
    out_w = max(sprite.shape[1] for sprite in sprites)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for sprite in sprites:
        h, w = sprite.shape
        col = _alignment_col(out_w, w, h)
        window = out[row : row + h, col : col + w]
        mask = sprite != BG
        window[mask] = sprite[mask]
        row += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder objects by the bottom control-strip color sequence, then align them by bbox aspect ratio.

    Input structure:
    - Background is black.
    - The bottom row contains a control strip of distinct non-black pixels.
    - Above the strip, each control color appears in exactly one connected monochrome object.

    Rule:
    - Read the control-strip colors from left to right.
    - Extract the matching object for each color and erase original placements.
    - Stack the objects from top to bottom in control-strip order.
    - If an object's bbox is wider than tall, left-align it.
      If taller than wide, right-align it.
      If square, center it.
    - Leave one empty row between consecutive objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 1

    control_row = g[-1]
    control_cols = [c for c in range(g.shape[1]) if int(control_row[c]) != BG]
    assert control_cols, "control strip must contain at least one non-background color"
    control_colors = [int(control_row[c]) for c in control_cols]
    assert len(control_colors) == len(set(control_colors)), "control colors must be unique"

    upper = g[:-1]
    upper_colors = {int(v) for v in np.unique(upper) if int(v) != BG}
    assert upper_colors == set(control_colors), "upper objects must match the control-strip palette"

    sprites = [_sprite_from_color(upper, color) for color in control_colors]
    return _stack_sprites(sprites)


def _make_rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _make_l(h: int, w: int, corner: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if corner == "tl":
        mask[0, :] = True
        mask[:, 0] = True
    elif corner == "tr":
        mask[0, :] = True
        mask[:, -1] = True
    elif corner == "bl":
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        mask[-1, :] = True
        mask[:, -1] = True
    return mask


def _make_t(h: int, w: int, side: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if side == "up":
        mask[0, :] = True
        mask[:, w // 2] = True
    elif side == "down":
        mask[-1, :] = True
        mask[:, w // 2] = True
    elif side == "left":
        mask[:, 0] = True
        mask[h // 2, :] = True
    else:
        mask[:, -1] = True
        mask[h // 2, :] = True
    return mask


def _make_frame(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _sample_dims(branch: str) -> tuple[int, int]:
    if branch == "wide":
        h = int(_RNG.integers(1, 4))
        w = int(_RNG.integers(h + 1, min(h + 5, 8)))
        return h, w
    if branch == "tall":
        w = int(_RNG.integers(1, 4))
        h = int(_RNG.integers(w + 1, min(w + 5, 8)))
        return h, w
    side = int(_RNG.integers(2, 6))
    return side, side


def _sample_shape_mask(branch: str, h: int, w: int) -> np.ndarray:
    families = ["rect", "l"]
    if h >= 2 and w >= 2:
        families.append("t")
    if branch == "square" and h >= 3:
        families.append("frame")

    family = str(_RNG.choice(families))
    if family == "rect":
        return _make_rect(h, w)
    if family == "l":
        return _make_l(h, w, str(_RNG.choice(["tl", "tr", "bl", "br"])))
    if family == "t":
        return _make_t(h, w, str(_RNG.choice(["up", "down", "left", "right"])))
    return _make_frame(h, w)


def _sample_objects() -> list[dict[str, object]] | None:
    count = int(_RNG.integers(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX_EXCLUSIVE))
    branches = ["wide", "tall", "square"]
    for _ in range(count - 3):
        branches.append(str(_RNG.choice(["wide", "tall", "square"])))
    _RNG.shuffle(branches)

    colors = list(map(int, _RNG.choice(PALETTE, size=count, replace=False)))
    _RNG.shuffle(colors)

    objects: list[dict[str, object]] = []
    for branch, color in zip(branches, colors, strict=False):
        h, w = _sample_dims(branch)
        mask = _sample_shape_mask(branch, h, w)
        sprite = np.full((h, w), BG, dtype=int)
        sprite[mask] = color
        objects.append({"branch": branch, "color": color, "sprite": sprite})

    out_height = sum(int(obj["sprite"].shape[0]) for obj in objects) + count - 1
    max_width = max(int(obj["sprite"].shape[1]) for obj in objects)
    if out_height > MAX_OUTPUT_HEIGHT or max_width > MAX_GRID_SIDE:
        return None
    return objects


def _pick_canvas(objects: list[dict[str, object]]) -> tuple[int, int] | None:
    total_bbox_area = sum(int(obj["sprite"].shape[0] * obj["sprite"].shape[1]) for obj in objects)
    max_h = max(int(obj["sprite"].shape[0]) for obj in objects)
    max_w = max(int(obj["sprite"].shape[1]) for obj in objects)
    count = len(objects)

    side_floor = int(np.ceil(np.sqrt(total_bbox_area * 3.0)))
    h_lo = max(10, max_h + 3, side_floor)
    w_lo = max(12, max_w + 3, count + 3, side_floor + 1)
    h_hi = min(MAX_GRID_SIDE, h_lo + 8)
    w_hi = min(MAX_GRID_SIDE, w_lo + 10)
    if h_lo > h_hi or w_lo > w_hi:
        return None
    return int(_RNG.integers(h_lo, h_hi + 1)), int(_RNG.integers(w_lo, w_hi + 1))


def _place_objects(
    height: int, width: int, objects: list[dict[str, object]]
) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros((height - 1, width), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(
        range(len(objects)),
        key=lambda idx: int(objects[idx]["sprite"].shape[0] * objects[idx]["sprite"].shape[1]),
        reverse=True,
    )

    for idx in order:
        sprite = objects[idx]["sprite"]
        h, w = sprite.shape
        max_top = height - h - 2
        if max_top < 0 or w > width:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(_RNG.integers(0, max_top + 1))
            left = int(_RNG.integers(0, width - w + 1))
            r0 = max(0, top - 1)
            r1 = min(height - 1, top + h + 1)
            c0 = max(0, left - 1)
            c1 = min(width, left + w + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[top : top + h, left : left + w] = True
            placements[idx] = (top, left)
            placed = True
            break

        if not placed:
            return None
    return placements


def _render_input(
    height: int,
    width: int,
    objects: list[dict[str, object]],
    placements: dict[int, tuple[int, int]],
    control_colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for idx, obj in enumerate(objects):
        top, left = placements[idx]
        sprite = obj["sprite"]
        window = grid[top : top + sprite.shape[0], left : left + sprite.shape[1]]
        mask = sprite != BG
        window[mask] = sprite[mask]

    start = int(_RNG.integers(0, width - len(control_colors) + 1))
    grid[-1, start : start + len(control_colors)] = np.asarray(control_colors, dtype=int)
    return grid


def _expected_from_objects(objects: list[dict[str, object]], control_colors: list[int]) -> np.ndarray:
    sprite_by_color = {int(obj["color"]): obj["sprite"] for obj in objects}
    ordered = [sprite_by_color[color] for color in control_colors]
    return _stack_sprites(ordered)


def _background_dominant(grid: np.ndarray) -> bool:
    bg_cells = int(np.sum(grid == BG))
    fg_cells = grid.size - bg_cells
    return bg_cells > fg_cells


def _gen_attempt() -> np.ndarray | None:
    objects = _sample_objects()
    if objects is None:
        return None

    canvas = _pick_canvas(objects)
    if canvas is None:
        return None
    height, width = canvas

    placements = _place_objects(height, width, objects)
    if placements is None:
        return None

    control_colors = list(map(int, _RNG.permutation([int(obj["color"]) for obj in objects])))
    grid = _render_input(height, width, objects, placements, control_colors)
    if not _background_dominant(grid):
        return None

    expected = _expected_from_objects(objects, control_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
