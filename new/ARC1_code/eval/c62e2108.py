import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste
import itertools


def _nearest_unique(candidates, distance):
    assert candidates
    distances = [distance(candidate) for candidate in candidates]
    best = min(distances)
    assert distances.count(best) == 1
    return candidates[distances.index(best)]


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the control bars and tile each matched sprite toward its bar.

    Input structure
    - The background contains colored non-line sprites plus one-cell-thick control
      bars.
    - A horizontal bar has the same column span as a sprite; a vertical bar has
      the same row span as a sprite.

    Rule
    - Remove every control bar.
    - For each bar, find the nearest sprite aligned with the bar's span on the
      perpendicular axis.
    - Repeat that sprite periodically from its original position toward the bar,
      clipping at the grid boundary and leaving sprite-background cells empty.

    Key insight
    - The bar is not copied itself; it gives the direction and terminal line for
      repeating the aligned sprite motif.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    sprites = []
    bars = []
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if (height == 1) ^ (width == 1):
            assert component["area"] == height * width
            bars.append(component)
        else:
            assert height > 1 and width > 1
            sprites.append(component)

    assert sprites
    assert bars

    out = g.copy()
    for bar in bars:
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for bar in bars:
        br0, bc0, br1, bc1 = bar["bbox"]
        bar_h = br1 - br0 + 1
        bar_w = bc1 - bc0 + 1

        if bar_h == 1:
            aligned = [
                sprite
                for sprite in sprites
                if sprite["bbox"][1] == bc0 and sprite["bbox"][3] == bc1
            ]
            above = [sprite for sprite in aligned if br0 < sprite["bbox"][0]]
            below = [sprite for sprite in aligned if br0 > sprite["bbox"][2]]
            assert bool(above) ^ bool(below)
            if above:
                sprite = _nearest_unique(above, lambda s: s["bbox"][0] - br0)
            else:
                sprite = _nearest_unique(below, lambda s: br0 - s["bbox"][2])

            sr0, sc0, _sr1, _sc1 = sprite["bbox"]
            motif = crop_bbox(g, sprite["bbox"])
            motif_h, motif_w = motif.shape
            assert motif_w == bar_w

            if br0 < sr0:
                r = sr0
                while r + motif_h - 1 >= br0:
                    paste(out, motif, r, sc0, transparent=bg, clip=True)
                    r -= motif_h
            else:
                r = sr0
                while r <= br0:
                    paste(out, motif, r, sc0, transparent=bg, clip=True)
                    r += motif_h
        else:
            aligned = [
                sprite
                for sprite in sprites
                if sprite["bbox"][0] == br0 and sprite["bbox"][2] == br1
            ]
            left = [sprite for sprite in aligned if bc0 < sprite["bbox"][1]]
            right = [sprite for sprite in aligned if bc0 > sprite["bbox"][3]]
            assert bool(left) ^ bool(right)
            if left:
                sprite = _nearest_unique(left, lambda s: s["bbox"][1] - bc0)
            else:
                sprite = _nearest_unique(right, lambda s: bc0 - s["bbox"][3])

            sr0, sc0, _sr1, _sc1 = sprite["bbox"]
            motif = crop_bbox(g, sprite["bbox"])
            motif_h, motif_w = motif.shape
            assert motif_h == bar_h

            if bc0 < sc0:
                c = sc0
                while c + motif_w - 1 >= bc0:
                    paste(out, motif, sr0, c, transparent=bg, clip=True)
                    c -= motif_w
            else:
                c = sc0
                while c <= bc0:
                    paste(out, motif, sr0, c, transparent=bg, clip=True)
                    c += motif_w

    return out

# --------------- generator (o5x) ---------------

# Canvas and motif geometry.
_GEN_BACKGROUND, _GEN_BAR_COLOR = 0, 1
_GEN_GRID_SIZE_MIN, _GEN_GRID_SIZE_MAX = 17, 24
_GEN_MOTIF_SIZE, _GEN_EDGE_MARGIN = 4, 1

# Palette and direction primitives.
_GEN_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
_GEN_DIRECTIONS = ("up", "down", "left", "right")

# Structural counts and tiling bounds.
_GEN_DIRECTION_COUNT_MIN, _GEN_DIRECTION_COUNT_MAX = 1, 3
_GEN_SINGLE_SPRITE_DIRECTION_COUNTS = (2, 3)
_GEN_PAIR_DIRECTION_COUNTS = (
    (1, 1), (1, 2), (2, 1), (1, 3),
    (3, 1), (2, 2), (2, 3), (3, 2),
)
_GEN_TILE_COUNT_MIN, _GEN_TILE_COUNT_MAX = 2, 5

# Sampling branch and retry limits.
_GEN_SINGLE_SPRITE_ROLL_VALUES, _GEN_SINGLE_SPRITE_ROLL_PICK = 2, 0
_GEN_RETRY_LIMIT = 300
_GEN_RNG = np.random.default_rng()


def _gen_choice(items):
    return items[int(_GEN_RNG.integers(len(items)))]


def _gen_hollow_motif(color):
    motif = np.zeros((_GEN_MOTIF_SIZE, _GEN_MOTIF_SIZE), dtype=int)
    motif[0, :] = color
    motif[-1, :] = color
    motif[:, 0] = color
    motif[:, -1] = color
    return motif


def _gen_axis_starts(axis_size, motif_size, low_tiles, high_tiles):
    """Return sprite starts that realize the requested edge tile counts."""
    lo = _GEN_EDGE_MARGIN
    hi = axis_size - motif_size - _GEN_EDGE_MARGIN
    if low_tiles is not None:
        lo = max(lo, (low_tiles - 2) * motif_size + _GEN_EDGE_MARGIN, motif_size)
        hi = min(hi, (low_tiles - 1) * motif_size)
    if high_tiles is not None:
        lo = max(lo, axis_size - high_tiles * motif_size)
        hi = min(
            hi,
            axis_size - _GEN_EDGE_MARGIN - (high_tiles - 1) * motif_size,
            axis_size - 2 * motif_size,
        )
    if lo > hi:
        return ()
    return tuple(range(lo, hi + 1))


def _gen_direction_origins(plan, direction, height, width):
    r0, c0 = plan["r0"], plan["c0"]
    origins = []
    if direction == "up":
        r = r0
        while r + _GEN_MOTIF_SIZE - 1 >= 0:
            origins.append((r, c0))
            r -= _GEN_MOTIF_SIZE
    elif direction == "down":
        r = r0
        while r <= height - 1:
            origins.append((r, c0))
            r += _GEN_MOTIF_SIZE
    elif direction == "left":
        c = c0
        while c + _GEN_MOTIF_SIZE - 1 >= 0:
            origins.append((r0, c))
            c -= _GEN_MOTIF_SIZE
    else:
        c = c0
        while c <= width - 1:
            origins.append((r0, c))
            c += _GEN_MOTIF_SIZE
    return origins


def _gen_plan_mask(plan, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for direction in plan["dirs"]:
        for r, c in _gen_direction_origins(plan, direction, height, width):
            r0, c0 = max(0, r), max(0, c)
            r1 = min(height, r + _GEN_MOTIF_SIZE)
            c1 = min(width, c + _GEN_MOTIF_SIZE)
            mask[r0:r1, c0:c1] = True
    return mask


def _gen_plan_has_requested_tiles(plan, height, width):
    return all(
        len(_gen_direction_origins(plan, direction, height, width))
        == plan["tiles"][direction]
        for direction in plan["dirs"]
    )


def _gen_all_plans(height, width):
    """Enumerate all valid sprite layouts for one grid size."""
    plans = []
    tile_range = range(_GEN_TILE_COUNT_MIN, _GEN_TILE_COUNT_MAX + 1)
    for count in range(_GEN_DIRECTION_COUNT_MIN, _GEN_DIRECTION_COUNT_MAX + 1):
        for dirs in itertools.combinations(_GEN_DIRECTIONS, count):
            for tile_values in itertools.product(tile_range, repeat=count):
                tiles = dict(zip(dirs, tile_values))
                rows = _gen_axis_starts(
                    height, _GEN_MOTIF_SIZE, tiles.get("up"), tiles.get("down")
                )
                cols = _gen_axis_starts(
                    width, _GEN_MOTIF_SIZE, tiles.get("left"), tiles.get("right")
                )
                for r0 in rows:
                    for c0 in cols:
                        plan = {
                            "r0": r0,
                            "c0": c0,
                            "dirs": dirs,
                            "tiles": tiles,
                        }
                        if not _gen_plan_has_requested_tiles(plan, height, width):
                            continue
                        plan["mask"] = _gen_plan_mask(plan, height, width)
                        plans.append(plan)
    return plans


def _gen_compatible(first, second):
    if abs(first["r0"] - second["r0"]) < _GEN_MOTIF_SIZE + _GEN_EDGE_MARGIN:
        return False
    if abs(first["c0"] - second["c0"]) < _GEN_MOTIF_SIZE + _GEN_EDGE_MARGIN:
        return False
    return not np.any(first["mask"] & second["mask"])


def _gen_pick_pair(plans_by_count, counts):
    firsts = plans_by_count[counts[0]]
    seconds = plans_by_count[counts[1]]
    if not firsts or not seconds:
        return None

    for i in _GEN_RNG.permutation(len(firsts)):
        first = firsts[int(i)]
        compatible = [
            second
            for second in seconds
            if second is not first and _gen_compatible(first, second)
        ]
        if compatible:
            return [first, _gen_choice(compatible)]
    return None


def _gen_plans_by_direction_count(plans):
    return {
        count: [plan for plan in plans if len(plan["dirs"]) == count]
        for count in range(_GEN_DIRECTION_COUNT_MIN, _GEN_DIRECTION_COUNT_MAX + 1)
    }


def _gen_pick_single(plans_by_count):
    possible = [
        count for count in _GEN_SINGLE_SPRITE_DIRECTION_COUNTS if plans_by_count[count]
    ]
    if not possible:
        return None
    return [_gen_choice(plans_by_count[_gen_choice(possible)])]


def _gen_select_plans(plans):
    plans_by_count = _gen_plans_by_direction_count(plans)

    if (
        int(_GEN_RNG.integers(_GEN_SINGLE_SPRITE_ROLL_VALUES))
        == _GEN_SINGLE_SPRITE_ROLL_PICK
    ):
        return _gen_pick_single(plans_by_count)

    for i in _GEN_RNG.permutation(len(_GEN_PAIR_DIRECTION_COUNTS)):
        picked = _gen_pick_pair(plans_by_count, _GEN_PAIR_DIRECTION_COUNTS[int(i)])
        if picked is not None:
            return picked
    return None


def _gen_edge_bar_index(plan, direction, height, width):
    r0, c0 = plan["r0"], plan["c0"]
    if direction == "up":
        return 0, slice(c0, c0 + _GEN_MOTIF_SIZE)
    if direction == "down":
        return height - 1, slice(c0, c0 + _GEN_MOTIF_SIZE)
    if direction == "left":
        return slice(r0, r0 + _GEN_MOTIF_SIZE), 0
    return slice(r0, r0 + _GEN_MOTIF_SIZE), width - 1


def _gen_build_grids(height, width, motif, plans):
    """Return the generated input grid plus its expected tiled output."""
    out = np.zeros((height, width), dtype=int)
    for plan in plans:
        for direction in plan["dirs"]:
            for r, c in _gen_direction_origins(plan, direction, height, width):
                paste(out, motif, r, c, transparent=_GEN_BACKGROUND, clip=True)

    keep = np.zeros((height, width), dtype=bool)
    for plan in plans:
        r0, c0 = plan["r0"], plan["c0"]
        keep[r0 : r0 + _GEN_MOTIF_SIZE, c0 : c0 + _GEN_MOTIF_SIZE] = True

    inp = out.copy()
    inp[~keep] = _GEN_BACKGROUND
    for plan in plans:
        for direction in plan["dirs"]:
            inp[_gen_edge_bar_index(plan, direction, height, width)] = _GEN_BAR_COLOR
    return inp, out


def _gen_attempt():
    height = int(_GEN_RNG.integers(_GEN_GRID_SIZE_MIN, _GEN_GRID_SIZE_MAX + 1))
    width = int(_GEN_RNG.integers(_GEN_GRID_SIZE_MIN, _GEN_GRID_SIZE_MAX + 1))
    plans = _gen_select_plans(_gen_all_plans(height, width))
    if plans is None:
        return None

    motif = _gen_hollow_motif(_gen_choice(_GEN_COLORS))
    inp, expected = _gen_build_grids(height, width, motif, plans)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c62e2108 input")
