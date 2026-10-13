from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox(np.asarray(arr, dtype=int) != int(bg))
    assert box is not None, "pattern must contain foreground"
    r0, c0, r1, c1 = box
    return np.asarray(arr[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not m[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _is_rect_ring(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    patch = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(patch, border):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    top = np.asarray(grid[0], dtype=int)
    colors = [int(top[c]) for c in range(top.shape[0]) if int(top[c]) != int(bg)]
    assert 3 <= len(colors) <= 4, "legend must contain 3 or 4 entries"
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return {color: idx + 1 for idx, color in enumerate(colors)}


def _frame_chain(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict[str, object]] = []
    for color in map(int, np.unique(g)):
        if color == int(bg):
            continue
        for comp in _components(g == color):
            box = _is_rect_ring(comp)
            if box is None:
                continue
            r0, c0, r1, c1 = box
            if r0 == 0:
                continue
            candidates.append({"color": color, "bbox": box})

    assert candidates, "expected rectangular frame components"
    bbox_to_frame = {tuple(item["bbox"]): item for item in candidates}

    def area(item: dict[str, object]) -> int:
        r0, c0, r1, c1 = item["bbox"]  # type: ignore[index]
        return int((r1 - r0 + 1) * (c1 - c0 + 1))

    best: list[dict[str, object]] = []
    for item in sorted(candidates, key=area, reverse=True):
        chain = [item]
        r0, c0, r1, c1 = item["bbox"]  # type: ignore[index]
        while True:
            next_bbox = (int(r0 + 2), int(c0 + 2), int(r1 - 2), int(c1 - 2))
            nxt = bbox_to_frame.get(next_bbox)
            if nxt is None:
                break
            chain.append(nxt)
            r0, c0, r1, c1 = next_bbox
        if len(chain) > len(best) or (len(chain) == len(best) and area(chain[0]) > area(best[0])):
            best = chain

    assert 2 <= len(best) <= 3, "expected a 2- or 3-layer nested frame chain"
    return best


def _apply_operation(pattern: np.ndarray, op_index: int, color: int, bg: int) -> np.ndarray:
    out = np.asarray(pattern, dtype=int)
    if op_index == 1:
        out = np.rot90(out, -1)
    elif op_index == 2:
        out = np.fliplr(out)
    elif op_index == 3:
        out = np.flipud(out)
    elif op_index == 4:
        out = out.copy()
        out[out != int(bg)] = int(color)
    else:
        raise AssertionError(f"unsupported legend position: {op_index}")
    return _crop_non_bg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply legend-coded outer-to-inner frame transforms to the central core pattern and return the cropped result.

    Input structure:
    - The background is black and the top row contains a sparse left-to-right color legend made of isolated pixels.
    - Below the legend, a small multicolor core pattern sits inside 2 or 3 nested 1-pixel rectangular frames.
    - Adjacent frames are separated by a 1-pixel black gap, and each frame color appears in the legend.

    Rule:
    - Read the top-row legend from left to right: position 1 means rotate the core 90 degrees clockwise, position 2
      means horizontal flip, position 3 means vertical flip, and position 4 means recolor every non-background core
      pixel to that frame's color.
    - Detect the nested rectangular frames and sort them from outermost to innermost.
    - Starting with the original core crop, apply the transform encoded by each frame color in outer-to-inner order.
    - Remove the legend and frames, and output the transformed core as a tight crop.

    Key insight:
    - The frame colors do not directly alter geometry; they first index into the top legend, and those legend
      positions encode cumulative transforms whose order is the nesting order from outside to inside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    legend = _parse_legend(g, bg)
    frames = _frame_chain(g, bg)

    r0, c0, r1, c1 = frames[-1]["bbox"]  # type: ignore[index]
    inner_region = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    core = _crop_non_bg(inner_region, bg)

    out = core
    for frame in frames:
        color = int(frame["color"])  # type: ignore[index]
        assert color in legend, "every frame color must appear in the legend"
        out = _apply_operation(out, legend[color], color, bg)
    return out.astype(int)


# --------------- generator ---------------

_BG = 0
_NON_BG = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 400


def _random_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < target:
        if not frontier:
            break
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        add_frontier(*cell)
    return mask


def _dihedral_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _has_nontrivial_symmetry(arr: np.ndarray) -> bool:
    base = np.asarray(arr, dtype=int)
    variants = _dihedral_variants(base)
    return any(np.array_equal(base, var) for var in variants[1:])


def _has_ring_component(arr: np.ndarray, bg: int) -> bool:
    g = np.asarray(arr, dtype=int)
    for color in map(int, np.unique(g)):
        if color == int(bg):
            continue
        for comp in _components(g == color):
            if _is_rect_ring(comp) is not None:
                return True
    return False


def _generate_core(colors: list[int]) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = h * w
        min_area = max(h + w - 2, (area + 1) // 2)
        max_area = area - 1
        if min_area > max_area:
            continue
        target = random.randint(min_area, max_area)
        mask = _random_connected_mask(h, w, target)
        if int(mask.sum()) != target:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        n_colors = 2 if len(colors) == 2 else random.choice((2, 3))
        chosen = random.sample(colors, n_colors)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)

        core = np.full((h, w), _BG, dtype=int)
        for color, (r, c) in zip(chosen, coords):
            core[r, c] = int(color)
        for r, c in coords[n_colors:]:
            core[r, c] = int(random.choice(chosen))

        if len(set(map(int, core[mask]))) != n_colors:
            continue
        if _has_nontrivial_symmetry(core):
            continue
        if _has_ring_component(core, _BG):
            continue
        return core
    return None


def _sample_operation_sequence(legend_len: int, layer_count: int) -> list[int]:
    if legend_len == 3:
        if layer_count == 2:
            ops = [1, random.choice((2, 3))]
        else:
            ops = [1, 2, 3]
    else:
        if layer_count == 2:
            ops = [1, random.choice((2, 3))]
        else:
            ops = random.choice(([1, 2, 3], [1, 2, 4], [1, 3, 4])).copy()
    random.shuffle(ops)
    return ops


def _expected_output(core: np.ndarray, legend_colors: list[int], frame_ops: list[int]) -> np.ndarray:
    out = np.asarray(core, dtype=int)
    for op_index in frame_ops:
        frame_color = int(legend_colors[op_index - 1])
        out = _apply_operation(out, op_index, frame_color, _BG)
    return out


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    legend_len = random.choice((3, 4))
    layer_count = random.choice((2, 3))
    if layer_count > legend_len:
        return None

    legend_colors = random.sample(list(_NON_BG), legend_len)
    remaining = [c for c in _NON_BG if c not in legend_colors]
    core_color_count = random.choice((2, 3))
    if len(remaining) < core_color_count:
        return None
    core = _generate_core(random.sample(remaining, core_color_count))
    if core is None:
        return None

    frame_ops = _sample_operation_sequence(legend_len, layer_count)
    outer_h = int(core.shape[0] + 4 * layer_count)
    outer_w = int(core.shape[1] + 4 * layer_count)
    legend_span = 2 * legend_len - 1

    min_h = outer_h + 3
    min_w = max(outer_w + 2, legend_span + 1)
    if min_h > 30 or min_w > 30:
        return None
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)

    max_top = height - outer_h - 1
    if max_top < 2:
        return None
    obj_r0 = random.randint(2, max_top)
    obj_c0 = random.randint(0, width - outer_w)
    legend_c0 = random.randint(0, width - legend_span)

    grid = np.full((height, width), _BG, dtype=int)
    for idx, color in enumerate(legend_colors):
        grid[0, legend_c0 + 2 * idx] = int(color)

    for layer_idx, op_index in enumerate(frame_ops):
        color = int(legend_colors[op_index - 1])
        top = obj_r0 + 2 * layer_idx
        left = obj_c0 + 2 * layer_idx
        bottom = obj_r0 + outer_h - 1 - 2 * layer_idx
        right = obj_c0 + outer_w - 1 - 2 * layer_idx
        _paint_frame(grid, (top, left, bottom, right), color)

    core_r0 = obj_r0 + 2 * layer_count
    core_c0 = obj_c0 + 2 * layer_count
    core_mask = core != _BG
    grid[core_r0 : core_r0 + core.shape[0], core_c0 : core_c0 + core.shape[1]][core_mask] = core[core_mask]

    expected = _expected_output(core, legend_colors, frame_ops)
    if expected.ndim != 2 or max(expected.shape) > 30:
        return None
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    built = _build_input()
    if built is None:
        return None
    grid, expected = built
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid.astype(int)
    raise RuntimeError("failed to generate a valid input for puzzle 5")
