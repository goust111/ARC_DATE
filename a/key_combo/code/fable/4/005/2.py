from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[Coord] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and bool(mask[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_vertical_pair_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 2:
        return False
    cells = sorted((int(r), int(c)) for r, c in coords)
    (r0, c0), (r1, c1) = cells
    return c0 == c1 and r1 == r0 + 1 and int(grid[r0, c0]) != int(grid[r1, c1])


def _parse_input(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    components = _components_mask(g != bg)
    legends: list[tuple[int, int, int, int]] = []
    block_components: list[np.ndarray] = []
    for coords in components:
        if _is_vertical_pair_component(g, coords):
            cells = sorted((int(r), int(c)) for r, c in coords)
            (r0, c0), (r1, c1) = cells
            legends.append((r0, c0, int(g[r0, c0]), int(g[r1, c1])))
        else:
            block_components.append(coords)

    assert len(legends) == 1, "expected one isolated vertical sign legend"
    lr, lc, pos_color, neg_color = legends[0]
    assert pos_color != bg and neg_color != bg and pos_color != neg_color

    sign_pairs = []
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1]):
            top = int(g[r, c])
            bottom = int(g[r + 1, c])
            if {top, bottom} == {pos_color, neg_color}:
                sign_pairs.append((r, c))
    assert sign_pairs == [(lr, lc)], "the legend must be the only vertical positive/negative pair"
    assert int(g[lr, lc]) == pos_color and int(g[lr + 1, lc]) == neg_color

    blocks: list[dict[str, int]] = []
    for coords in block_components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(region != bg)), "each block component must fill its bbox"

        colors = set(map(int, np.unique(region)))
        block_colors = colors - {pos_color, neg_color}
        assert len(block_colors) == 1, "each rectangle must have one own color plus the two control colors"
        block_color = int(next(iter(block_colors)))
        assert block_color != bg

        pos_count = int(np.count_nonzero(region == pos_color))
        neg_count = int(np.count_nonzero(region == neg_color))
        assert pos_count > 0 and neg_count > 0, "each block must contain both signed control colors"
        assert int(np.count_nonzero(region == block_color)) > 0
        assert colors <= {block_color, pos_color, neg_color}

        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        net = pos_count - neg_count
        assert abs(net) * 2 < min(height, width), "net magnitude must leave a nonempty core"

        blocks.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "height": height,
                "width": width,
                "color": block_color,
                "pos_count": pos_count,
                "neg_count": neg_count,
                "net": net,
            }
        )

    assert blocks, "expected at least one rectangular block"
    return bg, (lr, lc, pos_color, neg_color), blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Grow or erode each rectangle by the signed net count of its embedded controls.

    Input structure:
    - The background contains several separated solid rectangular blocks, each with one own color.
    - Inside every block are pixels of the same two control colors.
    - A single isolated vertical two-pixel legend gives the sign convention: top color is positive, bottom color is negative.

    Rule:
    - For each block, compute net n = count(positive control color) - count(negative control color).
    - If n > 0, paint an outward rectangular border of thickness n in the block's own color, then restore the original block and controls.
    - If n < 0, paint the outer |n| layers of that block back to background, including any controls in those layers.
    - If n = 0, leave the block unchanged.

    Key insight:
    - The two control colors are signed operands read from the legend; the net sign chooses growth versus erosion and the magnitude gives the layer count.
    """
    g = np.asarray(grid, dtype=int)
    bg, _legend, blocks = _parse_input(g)
    h, w = g.shape
    out = g.copy()
    grown = np.zeros((h, w), dtype=bool)

    for block in blocks:
        r0, c0, r1, c1 = block["r0"], block["c0"], block["r1"], block["c1"]
        net = int(block["net"])
        color = int(block["color"])

        if net > 0:
            pad = net
            er0, ec0 = r0 - pad, c0 - pad
            er1, ec1 = r1 + pad, c1 + pad
            assert 0 <= er0 <= er1 < h and 0 <= ec0 <= ec1 < w, "grown block must stay in bounds"

            expanded = np.zeros((h, w), dtype=bool)
            expanded[er0 : er1 + 1, ec0 : ec1 + 1] = True
            original = np.zeros((h, w), dtype=bool)
            original[r0 : r1 + 1, c0 : c1 + 1] = True
            border = expanded & ~original
            assert not bool(np.any(g[border] != bg)), "growth border must not cover another input object"
            assert not bool(np.any(grown[expanded])), "grown rectangles must not overlap"

            out[er0 : er1 + 1, ec0 : ec1 + 1] = color
            out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
            grown[expanded] = True

        elif net < 0:
            cut = -net
            out[r0 : r1 + 1, c0 : c1 + 1] = bg
            out[r0 + cut : r1 - cut + 1, c0 + cut : c1 - cut + 1] = g[
                r0 + cut : r1 - cut + 1, c0 + cut : c1 - cut + 1
            ]

    return out


_COLORS = list(range(10))
_MAX_ATTEMPTS = 500


def _sample_counts(net: int, area: int) -> tuple[int, int]:
    max_total = max(4, area // 3)
    if net == 0:
        base = random.randint(1, max(1, min(4, max_total // 2)))
        return base, base

    mag = abs(int(net))
    max_base = max(1, min(4, (max_total - mag) // 2))
    base = random.randint(1, max_base)
    if net > 0:
        return base + mag, base
    return base, base + mag


def _sample_specs(block_colors: list[int]) -> list[dict[str, int]]:
    signs = ["pos", "neg", "zero"]
    if len(block_colors) > 3:
        signs.append(random.choice(["pos", "neg", "zero"]))
    random.shuffle(signs)

    specs: list[dict[str, int]] = []
    for color, sign in zip(block_colors, signs):
        for _ in range(100):
            height = random.randint(5, 9)
            width = random.randint(5, 10)
            max_abs = (min(height, width) - 1) // 2
            if sign == "zero":
                net = 0
            elif sign == "pos":
                net = random.randint(1, min(3, max_abs))
            else:
                net = -random.randint(1, min(3, max_abs))
            pos_count, neg_count = _sample_counts(net, height * width)
            if pos_count + neg_count < height * width:
                specs.append(
                    {
                        "height": height,
                        "width": width,
                        "net": net,
                        "color": int(color),
                        "pos_count": pos_count,
                        "neg_count": neg_count,
                    }
                )
                break
        else:
            raise RuntimeError("failed to sample a block spec")
    return specs


def _is_outer_ring(r: int, c: int, height: int, width: int, thickness: int) -> bool:
    return r < thickness or c < thickness or r >= height - thickness or c >= width - thickness


def _place_control_cells(height: int, width: int, pos_count: int, neg_count: int, erode: int) -> dict[Coord, int]:
    tokens = [1] * int(pos_count) + [-1] * int(neg_count)
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    outer_cells = [(r, c) for r, c in all_cells if erode > 0 and _is_outer_ring(r, c, height, width, erode)]
    core_cells = [(r, c) for r, c in all_cells if erode > 0 and not _is_outer_ring(r, c, height, width, erode)]

    def can_place(placed: dict[Coord, int], r: int, c: int, label: int) -> bool:
        if (r, c) in placed:
            return False
        opposite = -int(label)
        return placed.get((r - 1, c)) != opposite and placed.get((r + 1, c)) != opposite

    def place_one(placed: dict[Coord, int], label: int, pool: list[Coord]) -> bool:
        choices = list(pool)
        random.shuffle(choices)
        for r, c in choices:
            if can_place(placed, r, c, label):
                placed[(r, c)] = int(label)
                return True
        return False

    for _ in range(300):
        remaining = list(tokens)
        random.shuffle(remaining)
        placed: dict[Coord, int] = {}

        if erode > 0:
            outer_label = remaining.pop()
            if not place_one(placed, outer_label, outer_cells):
                continue
            core_label = remaining.pop()
            if not place_one(placed, core_label, core_cells):
                continue

        random.shuffle(remaining)
        ok = True
        for label in remaining:
            if not place_one(placed, label, all_cells):
                ok = False
                break
        if ok:
            return placed

    raise RuntimeError("failed to place control cells")


def _make_patch(spec: dict[str, int], pos_color: int, neg_color: int) -> np.ndarray:
    height = int(spec["height"])
    width = int(spec["width"])
    patch = np.full((height, width), int(spec["color"]), dtype=int)
    erode = max(0, -int(spec["net"]))
    cells = _place_control_cells(height, width, int(spec["pos_count"]), int(spec["neg_count"]), erode)
    for (r, c), label in cells.items():
        patch[r, c] = int(pos_color if label == 1 else neg_color)
    return patch


def _free(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, gap: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h - 1, r1 + gap)
    cc1 = min(w - 1, c1 + gap)
    return not bool(np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _mark(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, gap: int = 1) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h - 1, r1 + gap)
    cc1 = min(w - 1, c1 + gap)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_layout(height: int, width: int, specs: list[dict[str, int]]) -> tuple[int, int] | None:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: (specs[i]["height"] + 2 * max(0, specs[i]["net"])) * (specs[i]["width"] + 2 * max(0, specs[i]["net"])), reverse=True)

    for idx in order:
        spec = specs[idx]
        pad = max(0, int(spec["net"]))
        eff_h = int(spec["height"]) + 2 * pad
        eff_w = int(spec["width"]) + 2 * pad
        candidates = [
            (r, c)
            for r in range(height - eff_h + 1)
            for c in range(width - eff_w + 1)
            if _free(occupied, r, c, r + eff_h - 1, c + eff_w - 1, gap=1)
        ]
        if not candidates:
            return None
        er0, ec0 = random.choice(candidates)
        spec["r0"] = er0 + pad
        spec["c0"] = ec0 + pad
        spec["er0"] = er0
        spec["ec0"] = ec0
        spec["er1"] = er0 + eff_h - 1
        spec["ec1"] = ec0 + eff_w - 1
        _mark(occupied, er0, ec0, er0 + eff_h - 1, ec0 + eff_w - 1, gap=1)

    legend_candidates = [
        (r, c)
        for r in range(height - 1)
        for c in range(width)
        if _free(occupied, r, c, r + 1, c, gap=1)
    ]
    if not legend_candidates:
        return None
    lr, lc = random.choice(legend_candidates)
    _mark(occupied, lr, lc, lr + 1, lc, gap=1)
    return lr, lc


def _render_case(
    height: int,
    width: int,
    bg: int,
    pos_color: int,
    neg_color: int,
    specs: list[dict[str, int]],
    legend: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), int(bg), dtype=int)
    lr, lc = legend
    inp[lr, lc] = int(pos_color)
    inp[lr + 1, lc] = int(neg_color)

    for spec in specs:
        patch = _make_patch(spec, pos_color, neg_color)
        r0, c0 = int(spec["r0"]), int(spec["c0"])
        inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        spec["r1"] = r0 + patch.shape[0] - 1
        spec["c1"] = c0 + patch.shape[1] - 1

    out = inp.copy()
    for spec in specs:
        r0, c0, r1, c1 = int(spec["r0"]), int(spec["c0"]), int(spec["r1"]), int(spec["c1"])
        net = int(spec["net"])
        if net > 0:
            pad = net
            out[r0 - pad : r1 + pad + 1, c0 - pad : c1 + pad + 1] = int(spec["color"])
            out[r0 : r1 + 1, c0 : c1 + 1] = inp[r0 : r1 + 1, c0 : c1 + 1]
        elif net < 0:
            cut = -net
            out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
            out[r0 + cut : r1 - cut + 1, c0 + cut : c1 - cut + 1] = inp[
                r0 + cut : r1 - cut + 1, c0 + cut : c1 - cut + 1
            ]
    return inp, out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
        bg, legend, blocks = _parse_input(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if _get_bg(inp) != bg or inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    nets = [int(block["net"]) for block in blocks]
    if not (any(n > 0 for n in nets) and any(n < 0 for n in nets) and any(n == 0 for n in nets)):
        return False
    if np.array_equal(inp, expected):
        return False
    if len({int(block["color"]) for block in blocks}) != len(blocks):
        return False

    _lr, _lc, pos_color, neg_color = legend
    for block in blocks:
        r0, c0, r1, c1 = block["r0"], block["c0"], block["r1"], block["c1"]
        net = int(block["net"])
        total_controls = int(block["pos_count"] + block["neg_count"])
        if net != 0 and total_controls <= abs(net):
            return False
        region = inp[r0 : r1 + 1, c0 : c1 + 1]
        if net < 0:
            cut = -net
            ring = np.ones(region.shape, dtype=bool)
            ring[cut : region.shape[0] - cut, cut : region.shape[1] - cut] = False
            if not bool(np.any((region == pos_color) | (region == neg_color)) and np.any(((region == pos_color) | (region == neg_color)) & ring)):
                return False
        elif net == 0:
            if not np.array_equal(expected[r0 : r1 + 1, c0 : c1 + 1], region):
                return False
        else:
            pad = net
            border = expected[r0 - pad : r1 + pad + 1, c0 - pad : c1 + pad + 1].copy()
            border[pad : pad + region.shape[0], pad : pad + region.shape[1]] = int(block["color"])
            if not bool(np.all(border == int(block["color"]))):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(24, 30)
    width = random.randint(24, 30)
    block_count = random.choice([3, 3, 4])

    palette = list(_COLORS)
    random.shuffle(palette)
    bg = int(palette[0])
    pos_color = int(palette[1])
    neg_color = int(palette[2])
    block_colors = [int(c) for c in palette[3 : 3 + block_count]]

    specs = _sample_specs(block_colors)
    legend = _place_layout(height, width, specs)
    if legend is None:
        return None

    try:
        inp, expected = _render_case(height, width, bg, pos_color, neg_color, specs, legend)
    except Exception:
        return None

    if _get_bg(inp) != bg:
        return None
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signed grow/erode input")
