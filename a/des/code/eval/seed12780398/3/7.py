import random
from itertools import combinations

import numpy as np


BG = 0
AXIS = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
LEGEND_WIDTH = 6
LEGEND_RULE_CHOICES = (3, 4)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_axis(grid: np.ndarray) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []
    for r in range(h):
        color = int(grid[r, 0])
        if color != BG and np.all(grid[r, :] == color) and 2 * r + 1 == h:
            candidates.append(("horizontal", r, color))
    for c in range(w):
        color = int(grid[0, c])
        if color != BG and np.all(grid[:, c] == color) and 2 * c + 1 == w:
            candidates.append(("vertical", c, color))
    assert len(candidates) == 1, "expected exactly one centered full axis line"
    return candidates[0]


def _corner_rect(grid: np.ndarray, corner: str, height: int, width: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - width
    elif corner == "bl":
        r0, c0 = h - height, 0
    else:
        r0, c0 = h - height, w - width
    rect = grid[r0 : r0 + height, c0 : c0 + width]
    return rect, (r0, c0)


def _parse_legend(grid: np.ndarray, axis_color: int) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[dict[tuple[int, int], int], tuple[int, int, int, int]]] = []

    for corner in ("tl", "tr", "bl", "br"):
        for rules in LEGEND_RULE_CHOICES:
            height = rules + 2
            width = LEGEND_WIDTH
            if height > h or width > w:
                continue
            rect, (r0, c0) = _corner_rect(grid, corner, height, width)
            if not np.all(rect[0, :] == axis_color):
                continue
            if not np.all(rect[-1, :] == axis_color):
                continue
            if not np.all(rect[:, 0] == axis_color):
                continue
            if not np.all(rect[:, -1] == axis_color):
                continue
            interior = rect[1:-1, 1:-1]
            if interior.size == 0 or np.any(interior == axis_color):
                continue

            mapping: dict[tuple[int, int], int] = {}
            ok = True
            for rr in range(interior.shape[0]):
                row = interior[rr]
                cols = [cc for cc in range(interior.shape[1]) if int(row[cc]) != BG]
                if len(cols) != 3:
                    ok = False
                    break
                colors = [int(row[cc]) for cc in cols]
                if colors[0] == colors[1] or any(color == BG for color in colors):
                    ok = False
                    break
                key = tuple(sorted((colors[0], colors[1])))
                if key in mapping and mapping[key] != colors[2]:
                    ok = False
                    break
                mapping[key] = colors[2]
            if not ok or len(mapping) != rules:
                continue
            candidates.append((mapping, (r0, c0, height, width)))

    assert len(candidates) == 1, "expected exactly one legend"
    mapping, (r0, c0, height, width) = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + height, c0 : c0 + width] = True
    return mapping, mask


def _mix_colors(a: int, b: int, mapping: dict[tuple[int, int], int]) -> int:
    if a == BG:
        return b
    if b == BG:
        return a
    if a == b:
        return a
    key = tuple(sorted((a, b)))
    assert key in mapping, "missing legend rule for overlapping colors"
    return mapping[key]


def _fold_halves(left_or_top: np.ndarray, right_or_bottom: np.ndarray, mapping: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.zeros_like(left_or_top)
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            out[r, c] = _mix_colors(int(left_or_top[r, c]), int(right_or_bottom[r, c]), mapping)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the grid across the full grey axis and mix overlapping colors via the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    orientation, axis_idx, axis_color = _find_axis(g)
    mapping, legend_mask = _parse_legend(g, axis_color)

    work = g.copy()
    work[legend_mask] = BG
    if orientation == "vertical":
        work[:, axis_idx] = BG
        left = work[:, :axis_idx]
        right = np.fliplr(work[:, axis_idx + 1 :])
        assert left.shape == right.shape
        return _fold_halves(left, right, mapping)

    work[axis_idx, :] = BG
    top = work[:axis_idx, :]
    bottom = np.flipud(work[axis_idx + 1 :, :])
    assert top.shape == bottom.shape
    return _fold_halves(top, bottom, mapping)


# --------------- generator ---------------


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _random_shape(max_height: int, max_width: int) -> np.ndarray:
    for _ in range(100):
        height = random.randint(1, min(4, max_height))
        width = random.randint(1, min(4, max_width))
        area = height * width
        target = random.randint(1 if area == 1 else 2, min(area, 6))
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return _trim_mask(mask)
    raise RuntimeError("failed to build shape")


def _place_shape(mask: np.ndarray, blocked: np.ndarray) -> np.ndarray | None:
    out_h, out_w = blocked.shape
    mask_h, mask_w = mask.shape
    for _ in range(200):
        r0 = random.randint(0, out_h - mask_h)
        c0 = random.randint(0, out_w - mask_w)
        placed = np.zeros_like(blocked)
        placed[r0 : r0 + mask_h, c0 : c0 + mask_w] = mask
        if np.any(placed & blocked):
            continue
        return placed
    return None


def _legend_origin(full_shape: tuple[int, int], legend_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    full_h, full_w = full_shape
    legend_h, legend_w = legend_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, full_w - legend_w
    if corner == "bl":
        return full_h - legend_h, 0
    return full_h - legend_h, full_w - legend_w


def _project_legend_reservation(
    orientation: str,
    out_shape: tuple[int, int],
    corner: str,
    legend_shape: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = out_shape
    legend_h, legend_w = legend_shape
    if orientation == "vertical":
        full_mask = np.zeros((out_h, 2 * out_w + 1), dtype=bool)
        r0, c0 = _legend_origin(full_mask.shape, legend_shape, corner)
        full_mask[r0 : r0 + legend_h, c0 : c0 + legend_w] = True
        left = full_mask[:, :out_w]
        right = np.fliplr(full_mask[:, out_w + 1 :])
        return left | right

    full_mask = np.zeros((2 * out_h + 1, out_w), dtype=bool)
    r0, c0 = _legend_origin(full_mask.shape, legend_shape, corner)
    full_mask[r0 : r0 + legend_h, c0 : c0 + legend_w] = True
    top = full_mask[:out_h, :]
    bottom = np.flipud(full_mask[out_h + 1 :, :])
    return top | bottom


def _draw_legend(
    grid: np.ndarray,
    corner: str,
    rules: list[tuple[int, int, int]],
) -> None:
    legend_h = len(rules) + 2
    legend_w = LEGEND_WIDTH
    r0, c0 = _legend_origin(grid.shape, (legend_h, legend_w), corner)
    grid[r0 : r0 + legend_h, c0 : c0 + legend_w] = BG
    grid[r0, c0 : c0 + legend_w] = AXIS
    grid[r0 + legend_h - 1, c0 : c0 + legend_w] = AXIS
    grid[r0 : r0 + legend_h, c0] = AXIS
    grid[r0 : r0 + legend_h, c0 + legend_w - 1] = AXIS
    for idx, (a, b, out) in enumerate(rules):
        rr = r0 + 1 + idx
        grid[rr, c0 + 1] = a
        grid[rr, c0 + 2] = b
        grid[rr, c0 + 4] = out


def _choose_rules() -> list[tuple[int, int, int]]:
    rules = random.choice(LEGEND_RULE_CHOICES)
    palette = COLOR_POOL[:]
    random.shuffle(palette)
    input_colors = palette[:4] if rules == 4 else palette[: random.choice((4, 5))]
    output_pool = [color for color in palette if color not in input_colors]
    output_colors = output_pool[:rules]
    pairs = random.sample(list(combinations(input_colors, 2)), rules)
    return [(a, b, out) for (a, b), out in zip(pairs, output_colors, strict=False)]


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        out_h = random.randint(10, 16)
        out_w = random.randint(8, 13)
    else:
        out_h = random.randint(8, 13)
        out_w = random.randint(10, 16)

    rules = _choose_rules()
    legend_shape = (len(rules) + 2, LEGEND_WIDTH)
    corner = random.choice(("tl", "tr", "bl", "br"))
    reserved = _project_legend_reservation(orientation, (out_h, out_w), corner, legend_shape)

    layer_a = np.zeros((out_h, out_w), dtype=int)
    layer_b = np.zeros((out_h, out_w), dtype=int)
    blocked = _expand_mask(reserved)

    scene_colors = sorted({color for rule in rules for color in rule})
    items: list[tuple[str, tuple[int, int, int] | None]] = [("pair_diff", rule) for rule in rules]
    items.extend([("pair_same", None), ("left_only", None), ("right_only", None)])
    extra_count = random.randint(1, 3)
    for _ in range(extra_count):
        kind = random.choice(("pair_same", "left_only", "right_only", "pair_diff"))
        items.append((kind, random.choice(rules) if kind == "pair_diff" else None))
    random.shuffle(items)

    for kind, payload in items:
        shape = _random_shape(out_h, out_w)
        placed = _place_shape(shape, blocked)
        if placed is None:
            return None
        blocked |= _expand_mask(placed)
        coords = placed

        if kind == "pair_diff":
            assert payload is not None
            a, b, _out = payload
            if random.random() < 0.5:
                a, b = b, a
            layer_a[coords] = a
            layer_b[coords] = b
        elif kind == "pair_same":
            color = random.choice(scene_colors)
            layer_a[coords] = color
            layer_b[coords] = color
        elif kind == "left_only":
            color = random.choice(scene_colors)
            layer_a[coords] = color
        else:
            color = random.choice(scene_colors)
            layer_b[coords] = color

    mapping = {tuple(sorted((a, b))): out for a, b, out in rules}
    assert np.any((layer_a != BG) & (layer_b != BG) & (layer_a == layer_b))
    assert np.any((layer_a != BG) & (layer_b != BG) & (layer_a != layer_b))
    assert np.any((layer_a != BG) ^ (layer_b != BG))

    expected = _fold_halves(layer_a, layer_b, mapping)

    if orientation == "vertical":
        full = np.zeros((out_h, 2 * out_w + 1), dtype=int)
        full[:, :out_w] = layer_a
        full[:, out_w] = AXIS
        full[:, out_w + 1 :] = np.fliplr(layer_b)
    else:
        full = np.zeros((2 * out_h + 1, out_w), dtype=int)
        full[:out_h, :] = layer_a
        full[out_h, :] = AXIS
        full[out_h + 1 :, :] = np.flipud(layer_b)

    _draw_legend(full, corner, rules)
    if not np.array_equal(main(full), expected):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fold-and-mix input")
