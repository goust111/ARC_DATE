import numpy as np

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg, paste, random_free_location
import itertools
import random


def _hole_count(coords: np.ndarray) -> int:
    assert coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] > 0

    r0, c0, r1, c1 = bbox_coords(coords)
    obj = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    obj[coords[:, 0] - r0, coords[:, 1] - c0] = True

    padded = np.pad(obj, 1, constant_values=False)
    h, w = padded.shape
    holes = 0
    for bg_coords in components_mask(~padded, connectivity=4):
        rr = bg_coords[:, 0]
        cc = bg_coords[:, 1]
        touches_outer_edge = np.any((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1))
        if not touches_outer_edge:
            holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each teal object by how many enclosed black holes it contains.

    Input structure
    - The grid has a black background and one or more 4-connected teal objects.
    - Each teal object encloses one to four 4-connected black regions inside its bbox.

    Rule
    - Count the enclosed black regions for each teal object.
    - Recolor the object's pixels: one hole -> blue, two holes -> green, three holes -> red, four holes -> yellow.
    - Leave the black background unchanged.

    Key insight
    - The varied object shapes and sizes are distractions; the color class is determined by topological hole count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.BLACK
    fg_colors = colors_present(g, exclude={bg})
    assert fg_colors == [Color.TEAL]

    hole_palette = {
        1: Color.BLUE,
        2: Color.GREEN,
        3: Color.RED,
        4: Color.YELLOW,
    }

    out = g.copy()
    comps = components_mask(g == Color.TEAL, connectivity=4)
    assert len(comps) > 0
    for coords in comps:
        holes = _hole_count(coords)
        assert holes in hole_palette
        out[coords[:, 0], coords[:, 1]] = hole_palette[holes]

    return out

# --------------- generator (o5x) ---------------

# CFG: canvas size and scene complexity.
_GEN_GRID_H_MIN, _GEN_GRID_H_MAX = 9, 15
_GEN_GRID_W_MIN, _GEN_GRID_W_MAX = 11, 17
_GEN_OBJECT_COUNT_MIN, _GEN_OBJECT_COUNT_MAX = 3, 4
_GEN_TARGET_HOLES_MIN, _GEN_TARGET_HOLES_MAX = 1, 4

# CFG: object and enclosed-hole bounds.
_GEN_SPRITE_H_MIN, _GEN_SPRITE_H_MAX = 3, 8
_GEN_SPRITE_W_MIN, _GEN_SPRITE_W_MAX = 3, 10
_GEN_OBJECT_AREA_MIN, _GEN_OBJECT_AREA_MAX = 8, 28
_GEN_HOLE_AREA_MIN, _GEN_HOLE_AREA_MAX = 1, 8

# CFG: layout enumeration bounds.
_GEN_LAYOUT_ROWS_MIN, _GEN_LAYOUT_ROWS_MAX = 1, 3
_GEN_LAYOUT_COLS_MIN, _GEN_LAYOUT_COLS_MAX = 1, 4
_GEN_EXTRA_LAYOUT_CELLS_MAX = 2
_GEN_CELL_H_MIN, _GEN_CELL_H_MAX = 1, 3
_GEN_CELL_W_MIN, _GEN_CELL_W_MAX = 1, 4

# CFG: shape mutation probabilities and compact filtering.
_GEN_COMPACT_AREA_MAX = 34
_GEN_CORNER_TRIM_ATTEMPT_PROB, _GEN_CORNER_TRIM_KEEP_PROB = 0.65, 0.55
_GEN_EDGE_NOTCH_MIN, _GEN_EDGE_NOTCH_MAX = 0, 3

# CFG: placement and retry limits.
_GEN_COMPACT_OBJECT_COUNT, _GEN_COMPACT_GRID_H_MAX, _GEN_COMPACT_GRID_W_MAX = 4, 11, 13
_GEN_PLACEMENT_BORDER, _GEN_PLACEMENT_PADDING, _GEN_PLACEMENT_CONNECTIVITY = 0, 2, 8
_GEN_MAX_ATTEMPTS = 300

_GEN_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_HOLE_PALETTE = {1: Color.BLUE, 2: Color.GREEN, 3: Color.RED, 4: Color.YELLOW}
_GEN_LAYOUT_CACHE = {}
_GEN_SIZE_LAYOUT_CACHE = {}


def _gen_axis_starts(lengths):
    starts = []
    pos = 1
    for length in lengths:
        starts.append(pos)
        pos += length + 1
    return starts


def _gen_layout_shape(row_heights, col_widths):
    return sum(row_heights) + len(row_heights) + 1, sum(col_widths) + len(col_widths) + 1


def _gen_crop(sprite):
    coords = np.argwhere(sprite == Color.TEAL)
    if coords.size == 0:
        return sprite[:0, :0].copy()
    r0, c0, r1, c1 = bbox_coords(coords)
    return sprite[r0 : r1 + 1, c0 : c1 + 1].copy()


def _gen_hole_sizes_from_coords(coords):
    if coords.size == 0:
        return []
    r0, c0, r1, c1 = bbox_coords(coords)
    obj = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    obj[coords[:, 0] - r0, coords[:, 1] - c0] = True

    padded = np.pad(obj, 1, constant_values=False)
    h, w = padded.shape
    sizes = []
    for bg_coords in components_mask(~padded, connectivity=4):
        rr = bg_coords[:, 0]
        cc = bg_coords[:, 1]
        touches_edge = np.any((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1))
        if not touches_edge:
            sizes.append(int(bg_coords.shape[0]))
    return sizes


def _gen_valid_sprite(sprite, target_holes):
    sprite = _gen_crop(sprite)
    if sprite.size == 0:
        return False
    h, w = sprite.shape
    if not (_GEN_SPRITE_H_MIN <= h <= _GEN_SPRITE_H_MAX and _GEN_SPRITE_W_MIN <= w <= _GEN_SPRITE_W_MAX):
        return False
    area = int(np.sum(sprite == Color.TEAL))
    if not (_GEN_OBJECT_AREA_MIN <= area <= _GEN_OBJECT_AREA_MAX):
        return False
    comps = components_mask(sprite == Color.TEAL, connectivity=4)
    if len(comps) != 1:
        return False
    hole_sizes = _gen_hole_sizes_from_coords(comps[0])
    return len(hole_sizes) == target_holes and all(
        _GEN_HOLE_AREA_MIN <= s <= _GEN_HOLE_AREA_MAX for s in hole_sizes
    )


def _gen_layout_options(target_holes):
    if target_holes in _GEN_LAYOUT_CACHE:
        return _GEN_LAYOUT_CACHE[target_holes]

    options = []
    for rows in range(_GEN_LAYOUT_ROWS_MIN, _GEN_LAYOUT_ROWS_MAX + 1):
        for cols in range(_GEN_LAYOUT_COLS_MIN, _GEN_LAYOUT_COLS_MAX + 1):
            if rows * cols < target_holes or rows * cols > target_holes + _GEN_EXTRA_LAYOUT_CELLS_MAX:
                continue
            min_layout_h = rows * (_GEN_CELL_H_MIN + 1) + 1
            min_layout_w = cols * (_GEN_CELL_W_MIN + 1) + 1
            if min_layout_h > _GEN_SPRITE_H_MAX or min_layout_w > _GEN_SPRITE_W_MAX:
                continue

            cells = tuple(range(rows * cols))
            for selected in itertools.combinations(cells, target_holes):
                selected_set = set(selected)
                for row_heights in itertools.product(range(_GEN_CELL_H_MIN, _GEN_CELL_H_MAX + 1), repeat=rows):
                    sprite_h = sum(row_heights) + rows + 1
                    if not (_GEN_SPRITE_H_MIN <= sprite_h <= _GEN_SPRITE_H_MAX):
                        continue
                    for col_widths in itertools.product(range(_GEN_CELL_W_MIN, _GEN_CELL_W_MAX + 1), repeat=cols):
                        sprite_w = sum(col_widths) + cols + 1
                        if not (_GEN_SPRITE_W_MIN <= sprite_w <= _GEN_SPRITE_W_MAX):
                            continue

                        min_hole_area = 0
                        max_hole_area = 0
                        ok = True
                        for idx in selected_set:
                            r = idx // cols
                            c = idx % cols
                            cell_area = row_heights[r] * col_widths[c]
                            if cell_area < _GEN_HOLE_AREA_MIN:
                                ok = False
                                break
                            min_hole_area += _GEN_HOLE_AREA_MIN
                            max_hole_area += min(_GEN_HOLE_AREA_MAX, cell_area)
                        min_teal_area = sprite_h * sprite_w - max_hole_area
                        max_teal_area = sprite_h * sprite_w - min_hole_area
                        if ok and min_teal_area <= _GEN_OBJECT_AREA_MAX and max_teal_area >= _GEN_OBJECT_AREA_MIN:
                            options.append((rows, cols, row_heights, col_widths, tuple(selected)))

    _GEN_LAYOUT_CACHE[target_holes] = options
    return options


def _gen_connected_hole_mask(cell_h, cell_w, area):
    cells = {(random.randrange(cell_h), random.randrange(cell_w))}
    frontier = set()
    for r, c in cells:
        for dr, dc in _GEN_NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w:
                frontier.add((nr, nc))

    while len(cells) < area:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        for dr, dc in _GEN_NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((cell_h, cell_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_size_layout_choices(target_holes):
    if target_holes in _GEN_SIZE_LAYOUT_CACHE:
        return _GEN_SIZE_LAYOUT_CACHE[target_holes]

    by_sizes = {}
    for option in _gen_layout_options(target_holes):
        _rows, cols, row_heights, col_widths, selected = option
        sprite_h, sprite_w = _gen_layout_shape(row_heights, col_widths)
        sprite_area = sprite_h * sprite_w
        caps = []
        for idx in selected:
            r = idx // cols
            c = idx % cols
            caps.append(min(_GEN_HOLE_AREA_MAX, row_heights[r] * col_widths[c]))

        for sizes in itertools.product(*(range(_GEN_HOLE_AREA_MIN, cap + 1) for cap in caps)):
            teal_area = sprite_area - sum(sizes)
            if _GEN_OBJECT_AREA_MIN <= teal_area <= _GEN_OBJECT_AREA_MAX:
                by_sizes.setdefault(sizes, []).append(option)

    counts = {}
    for sizes in by_sizes:
        for size in sizes:
            counts[size] = counts.get(size, 0) + 1

    choices = []
    weights = []
    for sizes, options in by_sizes.items():
        choices.append((sizes, options))
        weights.append(sum(1.0 / counts[size] for size in sizes) / len(sizes))

    _GEN_SIZE_LAYOUT_CACHE[target_holes] = (choices, weights)
    return choices, weights


def _gen_render_layout(_rows, cols, row_heights, col_widths, selected, hole_sizes):
    sprite_h, sprite_w = _gen_layout_shape(row_heights, col_widths)
    sprite = np.full((sprite_h, sprite_w), Color.TEAL, dtype=int)
    row_starts = _gen_axis_starts(row_heights)
    col_starts = _gen_axis_starts(col_widths)

    for idx, hole_area in zip(selected, hole_sizes):
        r = idx // cols
        c = idx % cols
        rr = row_starts[r]
        cc = col_starts[c]
        hole_mask = _gen_connected_hole_mask(row_heights[r], col_widths[c], hole_area)
        view = sprite[rr : rr + row_heights[r], cc : cc + col_widths[c]]
        view[hole_mask] = Color.BLACK
    return sprite


def _gen_try_corner_trims(sprite, rows, cols, row_heights, col_widths, selected, target_holes):
    selected_set = set(selected)
    row_starts = _gen_axis_starts(row_heights)
    col_starts = _gen_axis_starts(col_widths)

    h, w = sprite.shape
    corner_specs = [
        (0, 0, slice(0, row_starts[0] + row_heights[0]), slice(0, col_starts[0] + col_widths[0])),
        (
            0,
            cols - 1,
            slice(0, row_starts[0] + row_heights[0]),
            slice(col_starts[-1], w),
        ),
        (
            rows - 1,
            0,
            slice(row_starts[-1], h),
            slice(0, col_starts[0] + col_widths[0]),
        ),
        (
            rows - 1,
            cols - 1,
            slice(row_starts[-1], h),
            slice(col_starts[-1], w),
        ),
    ]
    random.shuffle(corner_specs)

    out = sprite.copy()
    for r, c, rs, cs in corner_specs:
        if r * cols + c in selected_set or random.random() > _GEN_CORNER_TRIM_KEEP_PROB:
            continue
        trial = out.copy()
        trial[rs, cs] = Color.BLACK
        trial = _gen_crop(trial)
        if _gen_valid_sprite(trial, target_holes):
            out = trial
    return out


def _gen_try_edge_notches(sprite, target_holes):
    out = _gen_crop(sprite)
    h, w = out.shape
    boundary = []
    for r in range(h):
        for c in range(w):
            if out[r, c] == Color.TEAL and (r in (0, h - 1) or c in (0, w - 1)):
                boundary.append((r, c))
    random.shuffle(boundary)

    target_notches = random.randint(_GEN_EDGE_NOTCH_MIN, _GEN_EDGE_NOTCH_MAX)
    made = 0
    for r, c in boundary:
        if made >= target_notches:
            break
        if r >= out.shape[0] or c >= out.shape[1] or out[r, c] != Color.TEAL:
            continue
        trial = out.copy()
        trial[r, c] = Color.BLACK
        trial = _gen_crop(trial)
        if _gen_valid_sprite(trial, target_holes):
            out = trial
            made += 1
    return out


def _gen_orient(sprite, target_holes):
    variants = []
    base = _gen_crop(sprite)
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
        variants.append(np.flipud(rot))

    valid = []
    seen = set()
    for variant in variants:
        candidate = _gen_crop(variant)
        key = candidate.tobytes()
        if key in seen:
            continue
        seen.add(key)
        if _gen_valid_sprite(candidate, target_holes):
            valid.append(candidate)
    return random.choice(valid) if valid else base


def _gen_option_area(option):
    _rows, _cols, row_heights, col_widths, _selected = option
    sprite_h, sprite_w = _gen_layout_shape(row_heights, col_widths)
    return sprite_h * sprite_w


def _gen_make_sprite(target_holes, compact=False):
    choices, weights = _gen_size_layout_choices(target_holes)
    if not choices:
        return None

    if compact:
        compact_choices = []
        compact_weights = []
        for (sizes, options), weight in zip(choices, weights):
            compact_options = [option for option in options if _gen_option_area(option) <= _GEN_COMPACT_AREA_MAX]
            if compact_options:
                compact_choices.append((sizes, compact_options))
                compact_weights.append(weight)
        if compact_choices:
            choices = compact_choices
            weights = compact_weights

    hole_sizes, options = random.choices(choices, weights=weights, k=1)[0]
    rows, cols, row_heights, col_widths, selected = random.choice(options)
    sprite = _gen_render_layout(rows, cols, row_heights, col_widths, selected, hole_sizes)
    if random.random() < _GEN_CORNER_TRIM_ATTEMPT_PROB:
        sprite = _gen_try_corner_trims(sprite, rows, cols, row_heights, col_widths, selected, target_holes)
    sprite = _gen_try_edge_notches(sprite, target_holes)
    sprite = _gen_orient(sprite, target_holes)
    sprite = _gen_crop(sprite)
    return sprite if _gen_valid_sprite(sprite, target_holes) else None


def _gen_expected_output(grid):
    out = grid.copy()
    for coords in components_mask(grid == Color.TEAL, connectivity=4):
        hole_count = len(_gen_hole_sizes_from_coords(coords))
        if hole_count not in _GEN_HOLE_PALETTE:
            return None
        out[coords[:, 0], coords[:, 1]] = _GEN_HOLE_PALETTE[hole_count]
    return out


def _gen_sample_scene_shape():
    """Sample grid dimensions and the requested object count."""
    return (
        random.randint(_GEN_GRID_H_MIN, _GEN_GRID_H_MAX),
        random.randint(_GEN_GRID_W_MIN, _GEN_GRID_W_MAX),
        random.randint(_GEN_OBJECT_COUNT_MIN, _GEN_OBJECT_COUNT_MAX),
    )


def _gen_sample_targets(n_objects):
    """Sample per-object hole targets, avoiding all-equal target lists."""
    targets = [random.randint(_GEN_TARGET_HOLES_MIN, _GEN_TARGET_HOLES_MAX) for _ in range(n_objects)]
    if len(set(targets)) == 1:
        choices = [x for x in range(_GEN_TARGET_HOLES_MIN, _GEN_TARGET_HOLES_MAX + 1) if x != targets[0]]
        targets[random.randrange(n_objects)] = random.choice(choices)
    return targets


def _gen_uses_compact_sprites(h, w, n_objects):
    return n_objects == _GEN_COMPACT_OBJECT_COUNT or h <= _GEN_COMPACT_GRID_H_MAX or w <= _GEN_COMPACT_GRID_W_MAX


def _gen_make_scene_sprites(targets, compact_scene):
    """Build one valid sprite for each target hole count, or fail the attempt."""
    sprites = []
    for target in targets:
        sprite = _gen_make_sprite(target, compact=compact_scene)
        if sprite is None:
            return None
        sprites.append(sprite)
    return sprites


def _gen_place_sprites(h, w, sprites):
    """Place sprites on a black grid with the configured separation padding."""
    grid = np.zeros((h, w), dtype=int)
    for sprite in sorted(sprites, key=lambda s: int(np.sum(s == Color.TEAL)), reverse=True):
        try:
            r0, c0 = random_free_location(
                grid,
                sprite,
                transparent=Color.BLACK,
                border=_GEN_PLACEMENT_BORDER,
                padding=_GEN_PLACEMENT_PADDING,
                padding_connectivity=_GEN_PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(grid, sprite, r0, c0, transparent=Color.BLACK)
    return grid


def _gen_valid_scene(grid, n_objects):
    """Check post-placement object count, bbox, area, and enclosed-hole bounds."""
    comps = components_mask(grid == Color.TEAL, connectivity=4)
    if len(comps) != n_objects:
        return False
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        area = int(coords.shape[0])
        holes = _gen_hole_sizes_from_coords(coords)
        if not (
            _GEN_SPRITE_H_MIN <= obj_h <= _GEN_SPRITE_H_MAX
            and _GEN_SPRITE_W_MIN <= obj_w <= _GEN_SPRITE_W_MAX
            and _GEN_OBJECT_AREA_MIN <= area <= _GEN_OBJECT_AREA_MAX
        ):
            return False
        if not (
            _GEN_TARGET_HOLES_MIN <= len(holes) <= _GEN_TARGET_HOLES_MAX
            and all(_GEN_HOLE_AREA_MIN <= s <= _GEN_HOLE_AREA_MAX for s in holes)
        ):
            return False
    return True


def _gen_matches_solver(grid):
    """Confirm the integrated solver applies the generator's expected recoloring."""
    expected = _gen_expected_output(grid)
    if expected is None:
        return False
    try:
        if not np.array_equal(main(grid), expected):
            return False
    except Exception:
        return False
    return True


def _gen_attempt():
    h, w, n_objects = _gen_sample_scene_shape()
    targets = _gen_sample_targets(n_objects)
    compact_scene = _gen_uses_compact_sprites(h, w, n_objects)
    sprites = _gen_make_scene_sprites(targets, compact_scene)
    if sprites is None:
        return None

    grid = _gen_place_sprites(h, w, sprites)
    if grid is None or not _gen_valid_scene(grid, n_objects):
        return None
    if not _gen_matches_solver(grid):
        return None
    return grid


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0a2355a6 input")
