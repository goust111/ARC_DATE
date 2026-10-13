import random
from itertools import product

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_legend_start(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    for c0 in range(w - 3, -1, -1):
        block = grid[:, c0 : c0 + 3]
        if block.shape[1] != 3:
            continue
        if c0 + 3 < w and np.any(grid[:, c0 + 3 :] != bg):
            continue
        if c0 > 0 and not np.all(grid[:, c0 - 1] == bg):
            continue
        row_counts = [int(np.count_nonzero(block[r] != bg)) for r in range(h)]
        if not any(count == 3 for count in row_counts):
            continue
        if all(count in (0, 3) for count in row_counts):
            return c0
    raise AssertionError("could not locate a right-side 1x3 legend block")


def _read_legend(block: np.ndarray, bg: int) -> tuple[dict[frozenset[int], int], dict[int, frozenset[int]]]:
    direct: dict[frozenset[int], int] = {}
    inverse: dict[int, frozenset[int]] = {}
    for row in block:
        if np.all(row == bg):
            continue
        assert np.all(row != bg), "legend rows must be full 1x3 color triples"
        a, b, mix = map(int, row)
        pair = frozenset((a, b))
        assert len(pair) == 2, "legend operands must be two distinct stripe colors"
        assert mix not in pair, "mix color must differ from both stripe colors"
        assert pair not in direct, "each color pair should appear once in the legend"
        assert mix not in inverse, "each mix color should decode to one pair"
        direct[pair] = mix
        inverse[mix] = pair
    assert direct, "legend must contain at least one mixing row"
    return direct, inverse


def _stripe_rows(scene: np.ndarray, bg: int) -> list[int]:
    rows = [r for r in range(scene.shape[0]) if np.all(scene[r] != bg)]
    assert rows, "expected at least one horizontal stripe row"
    return rows


def _stripe_cols(scene: np.ndarray, bg: int) -> list[int]:
    cols = [c for c in range(scene.shape[1]) if np.all(scene[:, c] != bg)]
    assert cols, "expected at least one vertical stripe column"
    return cols


def _row_colors(scene: np.ndarray, rows: list[int], cols: list[int], bg: int) -> dict[int, int]:
    col_set = set(cols)
    solved: dict[int, int] = {}
    for r in rows:
        pure = {int(scene[r, c]) for c in range(scene.shape[1]) if c not in col_set and int(scene[r, c]) != bg}
        assert len(pure) == 1, "each horizontal stripe row must have one pure stripe color"
        solved[r] = pure.pop()
    return solved


def _col_colors(scene: np.ndarray, rows: list[int], cols: list[int], bg: int) -> dict[int, int]:
    row_set = set(rows)
    solved: dict[int, int] = {}
    for c in cols:
        pure = {int(scene[r, c]) for r in range(scene.shape[0]) if r not in row_set and int(scene[r, c]) != bg}
        assert len(pure) == 1, "each vertical stripe column must have one pure stripe color"
        solved[c] = pure.pop()
    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a striped mixing grid into separate single-color stripe layers using the legend on the right.

    Input structure:
    - The dominant color is the background.
    - The left part of the grid is a stripe canvas with several full-width horizontal stripes and full-height vertical stripes, each stripe having a single base color.
    - Whenever a horizontal and vertical stripe cross, that cell shows a third mix color.
    - The right edge contains a small legend made of 1×3 color rows; each row means color A + color B = color C.

    Rule:
    - Locate the right-side legend and read every 1×3 row as one unordered mixing rule.
    - On the stripe canvas, identify the horizontal stripe rows and vertical stripe columns, then recover each stripe's base color from its non-intersection cells.
    - Use the legend to verify the mixed intersection colors and collect the distinct base stripe colors.
    - Output one full-size layer per base color, arranged left-to-right by the earliest stripe position of that color in the original canvas, with one background separator column between layers.
    - In each layer, keep only the stripes of that color and fill their crossings with that same color; leave the legend area blank.

    Key insight:
    - Mixed intersection cells are not standalone colors: the right-side legend teaches how each crossing decomposes back into one horizontal stripe color and one vertical stripe color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    legend_c0 = _find_legend_start(g, bg)
    direct_mix, inverse_mix = _read_legend(g[:, legend_c0 : legend_c0 + 3], bg)

    scene_w = legend_c0 - 1 if legend_c0 > 0 and np.all(g[:, legend_c0 - 1] == bg) else legend_c0
    assert scene_w > 0, "stripe canvas must be non-empty"
    scene = g[:, :scene_w]

    rows = _stripe_rows(scene, bg)
    cols = _stripe_cols(scene, bg)
    row_color = _row_colors(scene, rows, cols, bg)
    col_color = _col_colors(scene, rows, cols, bg)

    for r in rows:
        for c in cols:
            a = int(row_color[r])
            b = int(col_color[c])
            val = int(scene[r, c])
            if a == b:
                assert val == a, "same-color crossings should stay that base color"
            else:
                pair = frozenset((a, b))
                assert pair in direct_mix, "missing legend entry for a visible stripe crossing"
                assert val == direct_mix[pair], "intersection color disagrees with legend"
                assert val in inverse_mix and inverse_mix[val] == pair

    anchors: dict[int, tuple[int, int]] = {}
    for r, color in row_color.items():
        pos = (int(r), 0)
        if color not in anchors or pos < anchors[color]:
            anchors[color] = pos
    for c, color in col_color.items():
        pos = (0, int(c))
        if color not in anchors or pos < anchors[color]:
            anchors[color] = pos

    ordered_colors = [color for color, _pos in sorted(anchors.items(), key=lambda kv: kv[1])]
    assert ordered_colors, "expected at least one stripe color"

    layers = []
    for color in ordered_colors:
        layer = np.full_like(g, bg)
        for r, stripe_color in row_color.items():
            if stripe_color == color:
                layer[r, :scene_w] = color
        for c, stripe_color in col_color.items():
            if stripe_color == color:
                layer[:, c] = color
        layers.append(layer)

    sep = np.full((g.shape[0], 1), bg, dtype=int)
    out = layers[0]
    for layer in layers[1:]:
        out = np.concatenate((out, sep, layer), axis=1)
    return out


# --------------- generator ---------------

MAX_RETRIES = 400
HEIGHT_RANGE = (6, 10)
SCENE_WIDTH_K2 = (4, 10)
SCENE_WIDTH_K3 = (5, 5)
EXTRA_PER_COLOR = 1


def _allocate_counts(color_count: int, capacity: int) -> list[int]:
    counts = [1] * color_count
    max_extra = min(EXTRA_PER_COLOR * color_count, capacity - color_count - 2)
    if max_extra <= 0:
        return counts
    slots = [i for i in range(color_count) for _ in range(EXTRA_PER_COLOR)]
    chosen = random.sample(slots, random.randint(0, max_extra))
    for idx in chosen:
        counts[idx] += 1
    return counts


def _render_scene(
    height: int,
    scene_w: int,
    bg: int,
    row_assignments: list[tuple[int, int]],
    col_assignments: list[tuple[int, int]],
    mix_lookup: dict[tuple[int, int], int],
) -> np.ndarray:
    scene = np.full((height, scene_w), bg, dtype=int)
    row_color = {r: color for r, color in row_assignments}
    for r, color in row_assignments:
        scene[r, :] = color
    for c, color in col_assignments:
        scene[:, c] = color
        for r, h_color in row_color.items():
            pair = tuple(sorted((int(h_color), int(color))))
            scene[r, c] = mix_lookup[pair]
    return scene


def _ordered_base_colors(
    row_assignments: list[tuple[int, int]], col_assignments: list[tuple[int, int]]
) -> list[int]:
    anchors: dict[int, tuple[int, int]] = {}
    for r, color in row_assignments:
        pos = (int(r), 0)
        if color not in anchors or pos < anchors[color]:
            anchors[color] = pos
    for c, color in col_assignments:
        pos = (0, int(c))
        if color not in anchors or pos < anchors[color]:
            anchors[color] = pos
    return [color for color, _pos in sorted(anchors.items(), key=lambda kv: kv[1])]


def _expected_output(
    inp_shape: tuple[int, int],
    scene_w: int,
    bg: int,
    row_assignments: list[tuple[int, int]],
    col_assignments: list[tuple[int, int]],
) -> np.ndarray:
    height, width = inp_shape
    ordered_colors = _ordered_base_colors(row_assignments, col_assignments)
    row_groups: dict[int, list[int]] = {}
    col_groups: dict[int, list[int]] = {}
    for r, color in row_assignments:
        row_groups.setdefault(color, []).append(r)
    for c, color in col_assignments:
        col_groups.setdefault(color, []).append(c)

    layers = []
    for color in ordered_colors:
        layer = np.full((height, width), bg, dtype=int)
        for r in row_groups.get(color, []):
            layer[r, :scene_w] = color
        for c in col_groups.get(color, []):
            layer[:, c] = color
        layers.append(layer)

    sep = np.full((height, 1), bg, dtype=int)
    out = layers[0]
    for layer in layers[1:]:
        out = np.concatenate((out, sep, layer), axis=1)
    return out


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    tally = {int(v): int(c) for v, c in zip(vals, counts)}
    bg_count = tally.get(int(bg), 0)
    other_max = max((count for color, count in tally.items() if color != int(bg)), default=0)
    return bg_count > other_max


def _gen_attempt() -> np.ndarray | None:
    k = random.choice((2, 3))
    h_color_count, v_color_count = random.choice(((1, 1),)) if k == 2 else random.choice(((1, 2), (2, 1)))

    scene_low, scene_high = SCENE_WIDTH_K2 if k == 2 else SCENE_WIDTH_K3
    max_input_w = (30 - (k - 1)) // k
    scene_w = random.randint(scene_low, min(scene_high, max_input_w - 4))
    input_w = scene_w + 4

    min_h = max(HEIGHT_RANGE[0], h_color_count + 3)
    height = random.randint(min_h, HEIGHT_RANGE[1])

    h_counts = _allocate_counts(h_color_count, height)
    v_counts = _allocate_counts(v_color_count, scene_w)
    if sum(h_counts) > height - 2 or sum(v_counts) > scene_w - 2:
        return None

    top_free = random.choice((True, False))
    row_pool = list(range(1, height)) if top_free else list(range(height))
    col_pool = list(range(scene_w)) if top_free else list(range(1, scene_w))
    if len(row_pool) < sum(h_counts) or len(col_pool) < sum(v_counts):
        return None

    pair_count = h_color_count * v_color_count
    needed_colors = 1 + k + pair_count
    palette = random.sample(range(10), needed_colors)
    bg = palette[0]
    base = palette[1 : 1 + k]
    mixes = palette[1 + k :]
    random.shuffle(base)
    h_colors = base[:h_color_count]
    v_colors = base[h_color_count:]

    h_rows = random.sample(row_pool, sum(h_counts))
    v_cols = random.sample(col_pool, sum(v_counts))
    row_labels = [color for color, count in zip(h_colors, h_counts) for _ in range(count)]
    col_labels = [color for color, count in zip(v_colors, v_counts) for _ in range(count)]
    random.shuffle(row_labels)
    random.shuffle(col_labels)
    row_assignments = sorted(zip(h_rows, row_labels))
    col_assignments = sorted(zip(v_cols, col_labels))

    pair_rows = random.sample(range(height), pair_count)
    legend_entries = list(product(h_colors, v_colors))
    random.shuffle(legend_entries)
    mix_lookup: dict[tuple[int, int], int] = {}
    legend_map: list[tuple[int, int, int, int]] = []
    for (a, b), mix_color, row in zip(legend_entries, mixes, pair_rows):
        mix_lookup[tuple(sorted((int(a), int(b))))] = int(mix_color)
        legend_map.append((int(row), int(a), int(b), int(mix_color)))

    scene = _render_scene(height, scene_w, bg, row_assignments, col_assignments, mix_lookup)
    inp = np.full((height, input_w), bg, dtype=int)
    inp[:, :scene_w] = scene
    for row, a, b, mix_color in legend_map:
        inp[row, scene_w + 1 : scene_w + 4] = (a, b, mix_color)

    if not _bg_is_dominant(inp, bg):
        return None

    expected = _expected_output(inp.shape, scene_w, bg, row_assignments, col_assignments)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
