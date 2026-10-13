import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components


GEN_BG = 0
GEN_TEMPLATE = 5
GEN_SEPARATOR = 8
GEN_COLORS = [1, 2, 3, 4, 6, 7, 9]
GEN_ATTEMPTS = 400


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]
    assert len(rows) == 1, "expected exactly one solid separator row"
    return int(rows[0])


def _hole_count(template: np.ndarray, bg: int) -> int:
    holes = np.where(template == bg, 1, 0)
    count = 0
    for comp in find_components(holes, bg=0, colors=[1], connectivity=4):
        r0, c0, r1, c1 = comp["bbox"]
        if r0 > 0 and c0 > 0 and r1 < template.shape[0] - 1 and c1 < template.shape[1] - 1:
            count += 1
    return int(count)


def _infer_top_colors(top: np.ndarray) -> tuple[int, int]:
    colors = sorted(set(map(int, np.unique(top))))
    assert len(colors) == 2, "top legend must contain exactly background and template color"
    template_color = None
    for color in colors:
        mask_grid = np.where(top == color, 1, 0)
        comps = find_components(mask_grid, bg=0, colors=[1], connectivity=4)
        if len(comps) < 2:
            continue
        boxes = [comp["bbox"] for comp in comps]
        side = boxes[0][2] - boxes[0][0] + 1
        if side < 3:
            continue
        if all(
            box[2] - box[0] + 1 == side and box[3] - box[1] + 1 == side
            for box in boxes
        ):
            template_color = color
            break
    assert template_color is not None, "could not infer template color"
    bg = next(color for color in colors if color != template_color)
    return int(bg), int(template_color)


def _extract_templates(top: np.ndarray, template_color: int) -> list[np.ndarray]:
    mask_grid = np.where(top == template_color, 1, 0)
    comps = find_components(mask_grid, bg=0, colors=[1], connectivity=4)
    assert len(comps) >= 2, "expected multiple top templates"
    comps.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    templates = [crop_bbox(top, comp["bbox"]) for comp in comps]
    side = templates[0].shape[0]
    assert side == templates[0].shape[1], "templates must be square"
    for template in templates:
        assert template.shape == (side, side), "templates must share one size"
    return templates


def _extract_index_grid(lower: np.ndarray, bg: int) -> np.ndarray:
    box = bbox(lower != bg)
    assert box is not None, "expected a lower index grid"
    grid = crop_bbox(lower, box)
    assert np.all(grid != bg), "index grid must be dense"
    return grid


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Expand each lower-grid pixel into the top template whose hole count matches that pixel color's frequency.

    Input structure:
    - The top region contains several separated gray square templates of equal size on a background.
    - Each template is a solid square mask with one or more background holes, and the templates have distinct hole counts.
    - A single full-width separator row sits below the templates; below it is a dense rectangular color index grid.

    Rule:
    - Count the holes inside each top template using the template's own square bbox as the reference frame.
    - Count how many pixels each color occupies in the lower index grid, then pair each color with the unique template whose hole count equals that color count.
    - Discard the legend and separator, and replace every lower-grid pixel by the whole paired template block: solid template cells recolored to that pixel's color while holes stay background.

    Key insight:
    - The left-to-right order of the templates is a decoy; the real legend is hidden in the lower grid's color counts, which uniquely match the templates' hole counts.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_row(g)
    assert 0 < sep < g.shape[0] - 1, "separator must split top legend and lower index grid"

    bg, template_color = _infer_top_colors(g[:sep])
    templates = _extract_templates(g[:sep], template_color)
    side = int(templates[0].shape[0])
    hole_to_template = {}
    for template in templates:
        holes = _hole_count(template, bg)
        assert holes > 0, "template must contain at least one hole"
        assert holes not in hole_to_template, "template hole counts must be unique"
        hole_to_template[holes] = template

    index_grid = _extract_index_grid(g[sep + 1 :], bg)
    color_counts = Counter(map(int, index_grid.ravel()))
    counts = list(color_counts.values())
    assert len(counts) == len(set(counts)), "index-grid color counts must be unique"
    assert set(counts) == set(hole_to_template), "hole counts must match lower-grid color counts"

    out = np.full((index_grid.shape[0] * side, index_grid.shape[1] * side), bg, dtype=int)
    for r in range(index_grid.shape[0]):
        for c in range(index_grid.shape[1]):
            color = int(index_grid[r, c])
            template = hole_to_template[color_counts[color]]
            block = out[r * side : (r + 1) * side, c * side : (c + 1) * side]
            block[template != bg] = color
    return out


def _max_holes_for_size(size: int) -> int:
    return {5: 4, 6: 6}[size]


def _sample_counts_and_shape(size: int) -> tuple[list[int], tuple[int, int]]:
    max_side = 30 // size
    max_holes = _max_holes_for_size(size)
    candidates = []
    for n_templates in range(2, min(4, max_holes) + 1):
        for counts in combinations(range(1, max_holes + 1), n_templates):
            area = sum(counts)
            factor_pairs = [
                (h, w)
                for h in range(1, max_side + 1)
                for w in range(1, max_side + 1)
                if h * w == area
            ]
            if factor_pairs:
                candidates.append((list(counts), factor_pairs))
    counts, factor_pairs = random.choice(candidates)
    return counts, random.choice(factor_pairs)


def _make_template(size: int, hole_count: int) -> np.ndarray:
    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(400):
        template = np.full((size, size), GEN_TEMPLATE, dtype=int)
        holes = []
        remaining = interior[:]
        random.shuffle(remaining)
        while len(holes) < hole_count:
            options = []
            for r, c in remaining:
                if any(abs(r - hr) + abs(c - hc) == 1 for hr, hc in holes):
                    continue
                trial = template.copy()
                trial[r, c] = GEN_BG
                body = np.where(trial == GEN_TEMPLATE, 1, 0)
                comps = find_components(body, bg=0, colors=[1], connectivity=4)
                if len(comps) == 1 and _hole_count(trial, GEN_BG) == len(holes) + 1:
                    options.append((r, c))
            if not options:
                break
            pick = random.choice(options)
            template[pick] = GEN_BG
            holes.append(pick)
            remaining.remove(pick)
        if len(holes) == hole_count:
            return template
    raise RuntimeError("failed to build a connected template")


def _choose_template_order(counts: list[int]) -> list[int]:
    mode = random.choice(["sorted", "reverse", "shuffle"])
    if mode == "sorted":
        return sorted(counts)
    if mode == "reverse":
        return sorted(counts, reverse=True)
    order = counts[:]
    random.shuffle(order)
    return order


def _pattern_from_counts(colors: list[int], counts: dict[int, int], shape: tuple[int, int]) -> np.ndarray:
    total = shape[0] * shape[1]
    flat = []
    for color in colors:
        flat.extend([color] * counts[color])
    assert len(flat) == total

    mode = random.choice(["shuffle", "blocks", "round_robin", "snake"])
    if mode == "shuffle":
        random.shuffle(flat)
    elif mode == "round_robin":
        pools = {color: [color] * counts[color] for color in colors}
        flat = []
        while len(flat) < total:
            progressed = False
            for color in random.sample(colors, len(colors)):
                if pools[color]:
                    flat.append(pools[color].pop())
                    progressed = True
            assert progressed
    elif mode == "snake":
        ranked = sorted(colors, key=lambda color: counts[color], reverse=random.random() < 0.5)
        flat = []
        remaining = counts.copy()
        i = 0
        while len(flat) < total:
            color = ranked[i % len(ranked)]
            if remaining[color] > 0:
                flat.append(color)
                remaining[color] -= 1
            i += 1
    else:
        order = colors[:]
        random.shuffle(order)
        flat = []
        for color in order:
            flat.extend([color] * counts[color])

    grid = np.array(flat, dtype=int).reshape(shape)
    if mode == "snake" and shape[0] > 1:
        for r in range(1, shape[0], 2):
            grid[r] = grid[r, ::-1]
    return grid


def _expected_output(index_grid: np.ndarray, hole_to_template: dict[int, np.ndarray], bg: int) -> np.ndarray:
    side = next(iter(hole_to_template.values())).shape[0]
    color_counts = Counter(map(int, index_grid.ravel()))
    out = np.full((index_grid.shape[0] * side, index_grid.shape[1] * side), bg, dtype=int)
    for r in range(index_grid.shape[0]):
        for c in range(index_grid.shape[1]):
            color = int(index_grid[r, c])
            template = hole_to_template[color_counts[color]]
            block = out[r * side : (r + 1) * side, c * side : (c + 1) * side]
            block[template != bg] = color
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    size = random.choice([5, 6])
    counts, lower_shape = _sample_counts_and_shape(size)
    n_templates = len(counts)
    legend_gap = 1 if size == 6 and n_templates >= 4 else random.randint(1, 2)
    legend_width = n_templates * size + (n_templates - 1) * legend_gap
    max_width = 30
    slack = max_width - legend_width
    assert slack >= 0
    left_margin = random.randint(0, min(2, slack))
    right_margin = random.randint(0, min(2, slack - left_margin))
    width = max(legend_width + left_margin + right_margin, lower_shape[1])

    colors = random.sample(GEN_COLORS, n_templates)
    color_to_count = dict(zip(colors, random.sample(counts, len(counts))))
    ordered_counts = _choose_template_order(counts)
    templates = {count: _make_template(size, count) for count in counts}

    grid = np.full((size + 1 + lower_shape[0], width), GEN_BG, dtype=int)
    x = left_margin
    for count in ordered_counts:
        grid[:size, x : x + size] = templates[count]
        x += size + legend_gap
    grid[size, :] = GEN_SEPARATOR

    lower = _pattern_from_counts(colors, color_to_count, lower_shape)
    lower_left = random.randint(0, width - lower_shape[1])
    grid[size + 1 :, lower_left : lower_left + lower_shape[1]] = lower

    expected = _expected_output(lower, templates, GEN_BG)
    meta = {
        "counts": counts,
        "ordered_counts": ordered_counts,
        "lower_shape": lower_shape,
        "color_to_count": color_to_count,
        "size": size,
        "lower_left": lower_left,
    }
    return grid, expected, meta


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    assert isinstance(inp, np.ndarray) and inp.ndim == 2
    assert isinstance(expected, np.ndarray) and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30

    sep = _separator_row(inp)
    bg, template_color = _infer_top_colors(inp[:sep])
    templates = _extract_templates(inp[:sep], template_color)
    found_counts = sorted(_hole_count(template, bg) for template in templates)
    assert found_counts == sorted(meta["counts"])

    lower = _extract_index_grid(inp[sep + 1 :], bg)
    assert tuple(lower.shape) == tuple(meta["lower_shape"])
    actual_counts = Counter(map(int, lower.ravel()))
    assert sorted(actual_counts.values()) == sorted(meta["counts"])
    assert len(set(actual_counts.values())) == len(actual_counts)

    solved = main(inp)
    assert np.array_equal(solved, expected)
    assert not np.array_equal(lower, solved), "task must not degenerate to identity"

    side = meta["size"]
    for r in range(lower.shape[0]):
        for c in range(lower.shape[1]):
            block = solved[r * side : (r + 1) * side, c * side : (c + 1) * side]
            assert block.shape == (side, side)
            assert np.any(block == bg), "each output block must preserve at least one hole"
            assert np.any(block != bg), "each output block must contain template body"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected, meta = _build_case()
        try:
            _validate_generated_case(inp, expected, meta)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 2")
