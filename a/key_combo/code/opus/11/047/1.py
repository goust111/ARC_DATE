import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SPRITE_COLORS = (BLUE, RED, GREEN, MAGENTA, ORANGE, TEAL, MAROON)
RECOLOR_BY_COUNT = {1: RED, 2: BLUE, 3: GREEN}

SHAPE_LIBRARY = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 2), (1, 2), (1, 1), (2, 1), (2, 0)),
)
SPECIAL_SHAPES = (
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)

COUNT_PATTERNS = {
    3: ([1, 2, 4], [1, 3, 4], [2, 3, 4]),
    4: ([1, 2, 3, 4], [1, 2, 3, 5]),
    5: ([1, 1, 2, 3, 4], [1, 2, 2, 3, 4]),
}

GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords):
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _transform_shape(coords, *, flip_h: bool = False, flip_v: bool = False):
    pts = list(coords)
    if flip_h:
        max_c = max(c for _, c in pts)
        pts = [(r, max_c - c) for r, c in pts]
    if flip_v:
        max_r = max(r for r, _ in pts)
        pts = [(max_r - r, c) for r, c in pts]
    return _normalize(pts)


def _find_components(grid: np.ndarray, allowed_colors):
    allowed = set(int(c) for c in allowed_colors)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in allowed:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": tuple(sorted(coords)), "bbox": _bbox(coords)})
    return comps


def _analyze_sprites(grid: np.ndarray):
    comps = _find_components(grid, SPRITE_COLORS)
    sprites = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        local = tuple(sorted((r - r0, c - c0) for r, c in comp["coords"]))
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "local": local,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "area": len(comp["coords"]),
            }
        )
    sprites.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["color"]))
    return sprites


def _count_to_color(n: int) -> int:
    return RECOLOR_BY_COUNT.get(int(n), MAGENTA)


def _trajectory_union(sprite):
    top, left, _bottom, _right = sprite["bbox"]
    delta = top - 1
    cells = set()
    for r_rel, c_rel in sprite["local"]:
        abs_r = top + r_rel
        abs_c = left + c_rel
        for shift in range(delta + 1):
            cells.add((abs_r - shift, abs_c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each interior sprite upward to the grey ceiling, consume yellow markers in its swept bbox-strip, and recolor the sprite by the number consumed.

    Input structure:
    - The background is black, and the top row is a full grey bar that serves as the movement wall.
    - Several monochrome 4-connected sprites of distinct non-grey, non-yellow colors sit below the bar.
    - Isolated yellow singleton markers are scattered in the interior, some inside the vertical bbox-strips above each sprite.

    Rule:
    - Treat the grey full row as the cue that every sprite slides straight upward until its bbox top row becomes row 1.
    - For each sprite, consider the full rectangle swept by its bbox during that vertical slide: rows 1 through the sprite's original bbox bottom, across its original bbox columns.
    - Remove every yellow marker inside that swept rectangle, count how many were removed, and recolor the translated sprite with count 1->red, 2->blue, 3->green, 4 or more->magenta.
    - Keep the grey bar unchanged and preserve only yellow markers that lie outside every sprite's swept rectangle.

    Key insight:
    - The sprite's final color is determined neither by its original color nor by its final resting place alone; it depends on counting yellow resources inside the bbox-strip revealed by the completed upward slide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 4
    assert np.all(g[0] == GREY), "expected a full grey top bar"

    sprites = _analyze_sprites(g)
    assert 3 <= len(sprites) <= 5, "expected 3-5 sprites"

    intervals = []
    strip_union = np.zeros_like(g, dtype=bool)
    out = np.zeros_like(g)
    out[0, :] = GREY

    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        for pc0, pc1 in intervals:
            assert c1 < pc0 or c0 > pc1, "sprite column ranges must be disjoint"
        intervals.append((c0, c1))

        strip_union[1 : r1 + 1, c0 : c1 + 1] = True
        eaten = int(np.count_nonzero(g[1 : r1 + 1, c0 : c1 + 1] == YELLOW))
        final_color = _count_to_color(eaten)
        dr = 1 - r0
        for r, c in sprite["coords"]:
            out[r + dr, c] = final_color

    remaining_yellow = (g == YELLOW) & (~strip_union)
    out[remaining_yellow] = YELLOW
    return out


def _sample_counts(n_sprites: int):
    counts = list(random.choice(COUNT_PATTERNS[n_sprites]))
    random.shuffle(counts)
    return counts


def _sample_shape(special: bool, count: int):
    if special:
        pool = SPECIAL_SHAPES
    elif count >= 4:
        pool = tuple(shape for shape in SHAPE_LIBRARY if _shape_dims(shape)[1] >= 2)
    elif count == 3:
        pool = tuple(shape for shape in SHAPE_LIBRARY if _shape_dims(shape)[1] >= 1 and len(shape) >= 4)
    else:
        pool = SHAPE_LIBRARY
    base = random.choice(pool)
    flip_h = bool(random.getrandbits(1))
    flip_v = False if special else bool(random.getrandbits(1))
    return _transform_shape(base, flip_h=flip_h, flip_v=flip_v)


def _shape_dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _sample_lefts(widths, total_width):
    min_gap = 1
    required = sum(widths) + min_gap * (len(widths) - 1)
    if required > total_width:
        return None
    slack = total_width - required
    buckets = [0] * (len(widths) + 1)
    for _ in range(slack):
        buckets[random.randrange(len(buckets))] += 1
    lefts = []
    cursor = buckets[0]
    for idx, width in enumerate(widths):
        lefts.append(cursor)
        cursor += width
        if idx != len(widths) - 1:
            cursor += min_gap + buckets[idx + 1]
    return lefts


def _cell_touches_sprite(cell, sprite_cells):
    r, c = cell
    if (r, c) in sprite_cells:
        return True
    return any((nr, nc) in sprite_cells for nr, nc in _neighbors4(r, c))


def _cell_touches_any_yellow(cell, yellow_cells):
    r, c = cell
    if (r, c) in yellow_cells:
        return True
    return any((nr, nc) in yellow_cells for nr, nc in _neighbors4(r, c))


def _sprite_candidates(sprite):
    r0, c0, r1, c1 = sprite["bbox"]
    sprite_cells = set(sprite["coords"])
    traj = _trajectory_union(sprite)
    candidates = []
    above = []
    above_bbox_only = []
    interior = []
    for r in range(1, r1 + 1):
        for c in range(c0, c1 + 1):
            cell = (r, c)
            if _cell_touches_sprite(cell, sprite_cells):
                continue
            candidates.append(cell)
            if r < r0:
                above.append(cell)
                if cell not in traj:
                    above_bbox_only.append(cell)
            else:
                interior.append(cell)
    return candidates, above, above_bbox_only, interior


def _pick_cells(pool, amount, used_yellows):
    choices = list(pool)
    random.shuffle(choices)
    picked = []
    for cell in choices:
        if _cell_touches_any_yellow(cell, used_yellows):
            continue
        picked.append(cell)
        used_yellows.add(cell)
        if len(picked) == amount:
            return picked
    return None


def _sample_sprite_layout(shape, left: int, count: int, color: int, h: int, special: bool):
    shape_h, _shape_w = _shape_dims(shape)
    top_min = 3 if count <= 3 else count + 1
    top_max = min(12, h - shape_h)
    if top_min > top_max:
        return None

    tops = list(range(top_min, top_max + 1))
    random.shuffle(tops)
    best_sprite = None
    best_score = None
    for top in tops[:12]:
        coords = tuple(sorted((top + r, left + c) for r, c in shape))
        sprite = {"color": int(color), "coords": coords, "local": shape, "bbox": _bbox(coords), "target_count": int(count)}
        candidates, above, above_bbox_only, interior = _sprite_candidates(sprite)
        trial_used = set()
        if special:
            picked = _pick_cells(above_bbox_only, 1, trial_used)
            if picked is None:
                continue
            picked = _pick_cells(interior, 1, trial_used)
            if picked is None:
                continue
        else:
            picked = _pick_cells(above, 1, trial_used)
            if picked is None:
                continue
        picked = _pick_cells(candidates, count - len(trial_used), trial_used)
        if picked is None:
            continue
        score = (len(candidates), len(interior), top)
        if best_score is None or score > best_score:
            best_score = score
            best_sprite = sprite
    return best_sprite


def _build_attempt():
    h = random.randint(14, 18)
    w = random.randint(15, 20)
    n_sprites = random.randint(3, 5)
    counts = _sample_counts(n_sprites)
    special_options = [idx for idx, count in enumerate(counts) if count >= 2]
    if not special_options:
        return None
    special_idx = random.choice(special_options)

    shapes = [_sample_shape(idx == special_idx, counts[idx]) for idx in range(n_sprites)]
    widths = [_shape_dims(shape)[1] for shape in shapes]
    lefts = _sample_lefts(widths, w)
    if lefts is None:
        return None

    colors = random.sample(SPRITE_COLORS, n_sprites)
    sprites = []
    for idx, (shape, left, count, color) in enumerate(zip(shapes, lefts, counts, colors)):
        sprite = _sample_sprite_layout(shape, left, count, color, h, idx == special_idx)
        if sprite is None:
            return None
        sprites.append(sprite)

    consumed = set()
    strip_union = np.zeros((h, w), dtype=bool)
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        strip_union[1 : r1 + 1, c0 : c1 + 1] = True

    for idx, sprite in enumerate(sprites):
        target = sprite["target_count"]
        candidates, above, above_bbox_only, interior = _sprite_candidates(sprite)
        if len(candidates) < target or not above:
            return None

        sprite_used = set()
        if idx == special_idx:
            chosen = _pick_cells(above_bbox_only, 1, consumed)
            if chosen is None:
                return None
            sprite_used.update(chosen)
            chosen = _pick_cells(interior, 1, consumed)
            if chosen is None:
                return None
            sprite_used.update(chosen)
        else:
            chosen = _pick_cells(above, 1, consumed)
            if chosen is None:
                return None
            sprite_used.update(chosen)

        remaining = target - len(sprite_used)
        available = [cell for cell in candidates if cell not in sprite_used]
        extra = _pick_cells(available, remaining, consumed)
        if extra is None:
            return None
        sprite_used.update(extra)
        sprite["yellow_cells"] = tuple(sorted(sprite_used))

    total_consumed = sum(sprite["target_count"] for sprite in sprites)
    extra_min = max(1, 8 - total_consumed)
    extra_max = min(3, 15 - total_consumed)
    if extra_min > extra_max:
        return None
    extra_needed = random.randint(extra_min, extra_max)

    all_sprite_cells = {cell for sprite in sprites for cell in sprite["coords"]}
    outside_candidates = []
    for r in range(1, h):
        for c in range(w):
            cell = (r, c)
            if strip_union[r, c]:
                continue
            if _cell_touches_sprite(cell, all_sprite_cells):
                continue
            outside_candidates.append(cell)
    outside = _pick_cells(outside_candidates, extra_needed, consumed)
    if outside is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = GREY
    for sprite in sprites:
        for r, c in sprite["coords"]:
            grid[r, c] = sprite["color"]
        for r, c in sprite["yellow_cells"]:
            grid[r, c] = YELLOW
    for r, c in outside:
        grid[r, c] = YELLOW

    return grid


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if h > 30 or w > 30 or h < 4 or w < 4:
        return False
    if not np.all(grid[0] == GREY):
        return False
    if np.any(grid[1:] == GREY):
        return False

    sprites = _analyze_sprites(grid)
    if not (3 <= len(sprites) <= 5):
        return False
    if len({sprite["color"] for sprite in sprites}) != len(sprites):
        return False

    intervals = []
    strip_union = np.zeros_like(grid, dtype=bool)
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        if not (3 <= r0 <= 12):
            return False
        if not (3 <= sprite["area"] <= 6):
            return False
        for pc0, pc1 in intervals:
            if not (c1 < pc0 or c0 > pc1):
                return False
        intervals.append((c0, c1))
        strip_union[1 : r1 + 1, c0 : c1 + 1] = True

    yellow_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == YELLOW)]
    if not (8 <= len(yellow_cells) <= 15):
        return False

    sprite_cell_set = {cell for sprite in sprites for cell in sprite["coords"]}
    for r, c in yellow_cells:
        if any((nr, nc) in sprite_cell_set for nr, nc in _neighbors4(r, c)):
            return False
    yellow_set = set(yellow_cells)
    for r, c in yellow_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in yellow_set:
                return False

    for sprite in sprites:
        r0, c0, _r1, c1 = sprite["bbox"]
        if not np.any(grid[1:r0, c0 : c1 + 1] == YELLOW):
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.all(out[0] == GREY):
        return False
    remaining_yellow = int(np.count_nonzero(out == YELLOW))
    if remaining_yellow < 1:
        return False

    final_sprites = _analyze_sprites(out)
    if len(final_sprites) != len(sprites):
        return False

    color_changed = False
    has_bbox_only_marker = False
    has_interior_consumed_marker = False
    consumed_yellow = (grid == YELLOW) & (out != YELLOW)
    for before, after in zip(sprites, final_sprites):
        c0 = before["bbox"][1]
        c1 = before["bbox"][3]
        if after["bbox"][1] != c0 or after["bbox"][3] != c1:
            return False
        if after["bbox"][0] != 1:
            return False
        if before["local"] != after["local"]:
            return False

        r0, c0, r1, c1 = before["bbox"]
        count = int(np.count_nonzero(grid[1 : r1 + 1, c0 : c1 + 1] == YELLOW))
        if after["color"] != _count_to_color(count):
            return False
        if before["color"] != after["color"]:
            color_changed = True

        traj = _trajectory_union(before)
        for r, c in np.argwhere(consumed_yellow):
            rr, cc = int(r), int(c)
            if c0 <= cc <= c1 and 1 <= rr <= r1:
                if rr >= r0:
                    has_interior_consumed_marker = True
                if (rr, cc) not in traj:
                    has_bbox_only_marker = True

    if not color_changed:
        return False
    if not has_bbox_only_marker:
        return False
    if not has_interior_consumed_marker:
        return False

    preserved = (grid == YELLOW) & (~strip_union)
    if int(np.count_nonzero(preserved)) != remaining_yellow:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is None:
            continue
        try:
            if _validate_generated_input(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
