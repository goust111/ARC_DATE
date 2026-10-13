import numpy as np

from arc_utils import find_components, get_bg, paste as _gen_paste, random_free_location as _gen_random_free_location
import itertools as _gen_itertools
import random as _gen_random


def _platform_runs(row: np.ndarray, platform: int, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    W = int(row.shape[0])
    while c < W:
        if int(row[c]) == platform:
            c0 = c
            while c < W and int(row[c]) == platform:
                c += 1
            runs.append((c0, c - 1))
        else:
            assert int(row[c]) == bg
            c += 1
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """Move each rectangular object into the matching platform gap, rotating it so its two-cell side is vertical.

    Input structure
    - The last row is a solid non-background platform color.
    - The row above the platform contains only background and platform-color separator runs; background gaps between consecutive separator runs are slots.
    - Every non-background, non-platform object above the shelf is a solid monochrome rectangle with one side of length two.

    Rule
    - Clear those source rectangles from their original locations.
    - Rotate a rectangle if needed so it is two rows tall.
    - Place each normalized rectangle bottom-aligned with the shelf row inside the unique slot whose width matches the rectangle's normalized width.

    Key insight
    - The platform separators encode target slot widths, while each source rectangle's non-two dimension encodes the slot it belongs in.
    """
    g = np.asarray(grid, dtype=int)
    H = int(g.shape[0])
    bg = get_bg(g)

    bottom_colors = set(map(int, g[H - 1]))
    assert len(bottom_colors) == 1
    platform = next(iter(bottom_colors))
    assert platform != bg

    shelf = H - 2
    assert shelf >= 1
    assert set(map(int, g[shelf])) <= {bg, platform}

    runs = _platform_runs(g[shelf], platform, bg)
    assert len(runs) >= 2

    slots: list[tuple[int, int, int]] = []
    for (_left_start, left_end), (right_start, _right_end) in zip(runs, runs[1:]):
        start = left_end + 1
        end = right_start - 1
        assert start <= end
        assert np.all(g[shelf, start : end + 1] == bg)
        slots.append((start, end, end - start + 1))

    colors = sorted(set(map(int, g.ravel())) - {bg, platform})
    components = find_components(g, bg=bg, colors=colors, connectivity=4, monochromatic=True)

    objects: list[dict] = []
    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 < shelf

        sprite = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sprite == color)
        h, w = sprite.shape
        assert h == 2 or w == 2

        if h == 2:
            normalized = sprite.copy()
        else:
            normalized = np.rot90(sprite, k=1)
            assert normalized.shape[0] == 2

        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "width": int(normalized.shape[1]),
                "sprite": normalized,
            }
        )

    assert len(objects) == len(slots)
    slot_by_width = {width: (start, end) for start, end, width in slots}
    object_by_width = {obj["width"]: obj for obj in objects}
    assert len(slot_by_width) == len(slots)
    assert len(object_by_width) == len(objects)
    assert set(slot_by_width) == set(object_by_width)

    out = g.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for width, (start, end) in slot_by_width.items():
        sprite = object_by_width[width]["sprite"]
        h, w = sprite.shape
        assert h == 2
        assert w == width
        top = shelf - h + 1
        assert top >= 0
        assert np.all(out[top : shelf + 1, start : end + 1] == bg)
        out[top : shelf + 1, start : end + 1] = sprite

    return out

# --------------- generator (o5x) ---------------

# Canvas/platform geometry.
BG = 0
GRID_HEIGHTS = tuple(range(4, 10))
GRID_WIDTH_MIN, GRID_WIDTH_MAX = 5, 12
SHELF_FROM_BOTTOM, OPEN_ROWS_RESERVED = 2, 3

# Object, separator, palette, and retry choices.
OBJECT_COUNTS = (1, 2, 3)
NORMALIZED_WIDTHS, SEPARATOR_WIDTHS = (1, 2, 3, 4), (1, 2)
OBJECT_SHORT_SIDE = 2
VERTICAL_ORIENTATION, HORIZONTAL_ORIENTATION = "v", "h"
PALETTE = tuple(range(1, 9))
MAX_ATTEMPTS = 300

def _gen_configurations():
    """Enumerate deterministic layouts that fit the open area above the shelf."""
    configs = []
    for h in GRID_HEIGHTS:
        top_h = h - OPEN_ROWS_RESERVED
        for n in OBJECT_COUNTS:
            for widths in _gen_itertools.permutations(NORMALIZED_WIDTHS, n):
                for seps in _gen_itertools.product(SEPARATOR_WIDTHS, repeat=n + 1):
                    w = sum(widths) + sum(seps)
                    if not GRID_WIDTH_MIN <= w <= GRID_WIDTH_MAX:
                        continue
                    orientation_options = []
                    for width in widths:
                        opts = []
                        if OBJECT_SHORT_SIDE <= top_h:
                            opts.append(VERTICAL_ORIENTATION)
                        if width <= top_h:
                            opts.append(HORIZONTAL_ORIENTATION)
                        if not opts:
                            break
                        orientation_options.append(opts)
                    else:
                        for orientations in _gen_itertools.product(*orientation_options):
                            configs.append((h, w, n, widths, seps, orientations))
    return configs


_GEN_CONFIGURATIONS = _gen_configurations()

def _gen_choose_configuration():
    configs = _GEN_CONFIGURATIONS
    h = _gen_random.choice(sorted({cfg[0] for cfg in configs}))
    configs = [cfg for cfg in configs if cfg[0] == h]
    n = _gen_random.choice(sorted({cfg[2] for cfg in configs}))
    configs = [cfg for cfg in configs if cfg[2] == n]
    w = _gen_random.choice(sorted({cfg[1] for cfg in configs}))
    configs = [cfg for cfg in configs if cfg[1] == w]
    return _gen_random.choice(configs)


def _gen_build_grids(H, W, widths, seps, colors, platform):
    """Render the target, then clear target slots to form the source grid."""
    shelf = H - SHELF_FROM_BOTTOM
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)
    out[H - 1, :] = inp[H - 1, :] = platform

    c = 0
    fill_rows = slice(shelf - OBJECT_SHORT_SIDE + 1, shelf + 1)
    for i, width in enumerate(widths):
        out[shelf, c : c + seps[i]] = inp[shelf, c : c + seps[i]] = platform
        c += seps[i]
        start = c
        c += width
        out[fill_rows, start:c] = colors[i]
    out[shelf, c : c + seps[-1]] = inp[shelf, c : c + seps[-1]] = platform
    return inp, out, shelf

def _gen_scatter_sources(grid, shelf, widths, colors, orientations):
    """Place source rectangles in the open area; return False if crowded."""
    order = list(range(len(widths)))
    _gen_random.shuffle(order)
    top_area = grid[: shelf - 1, :]
    for i in order:
        width, color = int(widths[i]), int(colors[i])
        if orientations[i] == VERTICAL_ORIENTATION:
            sprite = np.full((OBJECT_SHORT_SIDE, width), color, dtype=int)
        else:
            sprite = np.full((width, OBJECT_SHORT_SIDE), color, dtype=int)
        try:
            r, c = _gen_random_free_location(top_area, sprite, transparent=BG)
        except ValueError:
            return False
        _gen_paste(grid, sprite, r, c)
    return True


def _gen_attempt() -> np.ndarray | None:
    H, W, n, widths, seps, orientations = _gen_choose_configuration()
    platform = _gen_random.choice(PALETTE)
    colors = _gen_random.sample([c for c in PALETTE if c != platform], n)
    inp, out, shelf = _gen_build_grids(H, W, widths, seps, colors, platform)
    if not _gen_scatter_sources(inp, shelf, widths, colors, orientations):
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
