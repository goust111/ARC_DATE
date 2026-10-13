from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste
import random

BBox = tuple[int, int, int, int]


def _intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _solid_rectangles(g: np.ndarray, bg: int, color: int) -> list[dict]:
    rectangles: list[dict] = []
    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    ):
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        if int(comp["area"]) > 1 and int(comp["area"]) == h * w:
            rectangles.append(comp)
    return rectangles


def _find_sprites(g: np.ndarray, bg: int) -> list[tuple[BBox, np.ndarray, int]]:
    sprites: list[tuple[BBox, np.ndarray, int]] = []
    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    ):
        b = comp["bbox"]
        if bbox_h(b) != 3 or bbox_w(b) != 3 or int(comp["area"]) != 9:
            continue

        patch = crop_bbox(g, b)
        border = np.ones((3, 3), dtype=bool)
        border[1, 1] = False
        border_colors = set(map(int, patch[border]))
        if len(border_colors) != 1:
            continue

        body = border_colors.pop()
        center = int(patch[1, 1])
        if body == bg or center == bg or body == center:
            continue
        sprites.append((b, patch, center))

    return sprites


def _direction_to_target(sprite: BBox, target: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = sprite
    tr0, tc0, tr1, tc1 = target
    row_overlap = not (r1 < tr0 or tr1 < r0)
    col_overlap = not (c1 < tc0 or tc1 < c0)

    directions: list[tuple[int, int]] = []
    if row_overlap and tc0 > c1:
        directions.append((0, 1))
    if row_overlap and tc1 < c0:
        directions.append((0, -1))
    if col_overlap and tr0 > r1:
        directions.append((1, 0))
    if col_overlap and tr1 < r0:
        directions.append((-1, 0))

    assert len(directions) == 1, "expected exactly one cardinal target direction"
    return directions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each 3x3 control-center sprite toward the matching solid rectangle until a copy first overlaps that rectangle.

    Input structure
    - The background is the dominant color.
    - Each sprite is a 3x3 foreground object whose eight border pixels have one color and whose center has a different control color.
    - For every control color, there is exactly one separate solid rectangular object of that color.

    Rule
    - Match each sprite to the unique solid rectangle whose color equals the sprite center.
    - Copy the whole 3x3 sprite in the cardinal direction of that rectangle, stepping by three cells each time.
    - Stop after pasting the first copy whose 3x3 bbox overlaps the rectangle; copied pixels overwrite the rectangle there.

    Key insight
    The center pixel is not the paint color: it is a destination key, while the entire 3x3 sprite is the motif tiled along the straight path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    bg = int(get_bg(g))
    sprites = _find_sprites(g, bg)
    assert sprites, "expected at least one 3x3 control-center sprite"

    out = g.copy()
    for sprite_bbox, patch, center in sprites:
        targets = _solid_rectangles(g, bg, center)
        assert len(targets) == 1, "each control color must have one solid rectangular target"
        target_bbox = targets[0]["bbox"]
        assert not _intersects(sprite_bbox, target_bbox), "sprite and target must start separate"

        dr, dc = _direction_to_target(sprite_bbox, target_bbox)
        r, c = sprite_bbox[0], sprite_bbox[1]

        while True:
            current_bbox = (r, c, r + 2, c + 2)
            assert 0 <= r and r + 3 <= h and 0 <= c and c + 3 <= w
            paste(out, patch, r, c)

            if _intersects(current_bbox, target_bbox):
                break

            r += dr * 3
            c += dc * 3

    return out

# --------------- generator (att1) ---------------
# Canvas and colors.
BG_COLOR, COLOR_COUNT = 8, 10
GRID_MIN_SIZE, GRID_MAX_SIZE = 20, 25

# Rectangles.
RECT_COUNT_MIN, RECT_COUNT_MAX = 2, 4
RECT_SIDE_MIN, RECT_SIDE_MAX = 2, 14
RECT_MARGIN, RECT_SPACING_PAD = 1, 5
RECT_PLACEMENT_ATTEMPTS = 50

# Sprites and paths.
SPRITE_SIZE, SPRITE_TOTAL_MAX = 3, 5
SPRITE_GAPS = [1, 4, 7, 10]
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Retry limits.
GENERATION_ATTEMPTS = 300


def _intersects_check(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _has_bg_border(grid, sr, sc, bg):
    H, W = grid.shape
    for r in range(sr, sr + SPRITE_SIZE):
        for c in range(sc, sc + SPRITE_SIZE):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not (sr <= nr < sr + SPRITE_SIZE and sc <= nc < sc + SPRITE_SIZE):
                        if grid[nr, nc] != bg:
                            return False
    return True


def _sample_palette(bg):
    all_colors = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(all_colors)
    num_rects = random.randint(RECT_COUNT_MIN, RECT_COUNT_MAX)
    body_color = all_colors[0]
    target_colors = all_colors[1:1 + num_rects]
    return body_color, target_colors


def _rect_has_spacing(rect, rects, pad):
    r0, c0, r1, c1 = rect
    for rr0, rc0, rr1, rc1, _ in rects:
        if not (r1 + pad < rr0 or rr1 + pad < r0 or c1 + pad < rc0 or rc1 + pad < c0):
            return False
    return True


def _place_rectangles(grid, target_colors):
    H, W = grid.shape
    rects = []
    for color in target_colors:
        for _ in range(RECT_PLACEMENT_ATTEMPTS):
            rh = random.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, H - 4))
            rw = random.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, W - 4))
            r0 = random.randint(RECT_MARGIN, H - rh - RECT_MARGIN)
            c0 = random.randint(RECT_MARGIN, W - rw - RECT_MARGIN)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            if _rect_has_spacing((r0, c0, r1, c1), rects, RECT_SPACING_PAD):
                rects.append((r0, c0, r1, c1, color))
                break
        else:
            return None
    for (r0, c0, r1, c1, color) in rects:
        grid[r0:r1 + 1, c0:c1 + 1] = color
    return rects


def _assign_sprites(rects):
    num_rects = len(rects)
    sprite_assignments = []
    for i, (_, _, _, _, color) in enumerate(rects):
        sprite_assignments.append((i, color))
    max_extra = min(SPRITE_TOTAL_MAX - len(sprite_assignments), num_rects)
    if max_extra > 0:
        extra = random.randint(0, max_extra)
        for _ in range(extra):
            i = random.randint(0, num_rects - 1)
            sprite_assignments.append((i, rects[i][4]))
    if len(sprite_assignments) > SPRITE_TOTAL_MAX:
        sprite_assignments = sprite_assignments[:SPRITE_TOTAL_MAX]
    return sprite_assignments


def _candidate_sprite_position(rect, direction, gap, H, W):
    r0, c0, r1, c1, _ = rect
    dr, dc = direction
    if dr == 0 and dc == 1:
        sc = c0 - gap - SPRITE_SIZE
        valid_rows = list(range(max(0, r0), min(H - SPRITE_SIZE, r1) + 1))
        if not valid_rows:
            return None
        sr = random.choice(valid_rows)
    elif dr == 0 and dc == -1:
        sc = c1 + gap + 1
        valid_rows = list(range(max(0, r0), min(H - SPRITE_SIZE, r1) + 1))
        if not valid_rows:
            return None
        sr = random.choice(valid_rows)
    elif dr == 1 and dc == 0:
        sr = r0 - gap - SPRITE_SIZE
        valid_cols = list(range(max(0, c0), min(W - SPRITE_SIZE, c1) + 1))
        if not valid_cols:
            return None
        sc = random.choice(valid_cols)
    else:
        sr = r1 + gap + 1
        valid_cols = list(range(max(0, c0), min(W - SPRITE_SIZE, c1) + 1))
        if not valid_cols:
            return None
        sc = random.choice(valid_cols)
    return sr, sc


def _sprite_in_bounds(sr, sc, H, W):
    return 0 <= sr and sr + SPRITE_SIZE <= H and 0 <= sc and sc + SPRITE_SIZE <= W


def _copy_positions_in_bounds(sr, sc, direction, copies, H, W):
    dr, dc = direction
    for step in range(1, copies + 1):
        nr = sr + dr * SPRITE_SIZE * step
        nc = sc + dc * SPRITE_SIZE * step
        if not _sprite_in_bounds(nr, nc, H, W):
            return False
    return True


def _corridor_clear(grid, sr, sc, direction, copies, bg):
    dr, dc = direction
    for step in range(1, copies):
        nr = sr + dr * SPRITE_SIZE * step
        nc = sc + dc * SPRITE_SIZE * step
        if not np.all(grid[nr:nr + SPRITE_SIZE, nc:nc + SPRITE_SIZE] == bg):
            return False
    return True


def _last_copy_is_valid(grid, sr, sc, direction, copies, rect, bg):
    r0, c0, r1, c1, _ = rect
    dr, dc = direction
    last_r = sr + dr * SPRITE_SIZE * copies
    last_c = sc + dc * SPRITE_SIZE * copies
    last_bbox = (last_r, last_c, last_r + SPRITE_SIZE - 1, last_c + SPRITE_SIZE - 1)
    if not _intersects_check(last_bbox, (r0, c0, r1, c1)):
        return False
    for rr in range(last_r, last_r + SPRITE_SIZE):
        for cc in range(last_c, last_c + SPRITE_SIZE):
            if not (r0 <= rr <= r1 and c0 <= cc <= c1) and grid[rr, cc] != bg:
                return False
    return True


def _overlaps_placed_sprite(sr, sc, placed_sprites):
    for ps_r, ps_c in placed_sprites:
        if not (sr + SPRITE_SIZE < ps_r or ps_r + SPRITE_SIZE < sr or sc + SPRITE_SIZE < ps_c or ps_c + SPRITE_SIZE < sc):
            return True
    return False


def _place_sprite(grid, sr, sc, body_color, center_color):
    patch = np.full((SPRITE_SIZE, SPRITE_SIZE), body_color, dtype=int)
    patch[SPRITE_SIZE // 2, SPRITE_SIZE // 2] = center_color
    grid[sr:sr + SPRITE_SIZE, sc:sc + SPRITE_SIZE] = patch


def _try_place_assigned_sprite(grid, rect, center_color, body_color, placed_sprites, bg):
    H, W = grid.shape
    directions = DIRECTIONS.copy()
    random.shuffle(directions)
    for direction in directions:
        gaps = SPRITE_GAPS.copy()
        random.shuffle(gaps)
        for gap in gaps:
            copies = gap // SPRITE_SIZE + 1
            pos = _candidate_sprite_position(rect, direction, gap, H, W)
            if pos is None:
                continue
            sr, sc = pos
            if not _sprite_in_bounds(sr, sc, H, W):
                continue
            if not _copy_positions_in_bounds(sr, sc, direction, copies, H, W):
                continue
            if not np.all(grid[sr:sr + SPRITE_SIZE, sc:sc + SPRITE_SIZE] == bg):
                continue
            if not _has_bg_border(grid, sr, sc, bg):
                continue
            if not _corridor_clear(grid, sr, sc, direction, copies, bg):
                continue
            if not _last_copy_is_valid(grid, sr, sc, direction, copies, rect, bg):
                continue
            if _overlaps_placed_sprite(sr, sc, placed_sprites):
                continue
            _place_sprite(grid, sr, sc, body_color, center_color)
            placed_sprites.append((sr, sc))
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    """Generate a single input grid for task 985ae207."""
    H = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    W = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    bg = BG_COLOR
    grid = np.full((H, W), bg, dtype=int)

    body_color, target_colors = _sample_palette(bg)
    rects = _place_rectangles(grid, target_colors)
    if rects is None:
        return None

    sprite_assignments = _assign_sprites(rects)
    placed_sprites = []
    for rect_idx, center_color in sprite_assignments:
        if not _try_place_assigned_sprite(grid, rects[rect_idx], center_color, body_color, placed_sprites, bg):
            return None

    try:
        main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
