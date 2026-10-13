from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    assert h >= 3 and w >= 3

    specs = [
        ("tl", grid[:3, :3], {(0, 0), (0, 1), (1, 0), (1, 1)}, (0, 0)),
        ("tr", grid[:3, -3:], {(0, 1), (0, 2), (1, 1), (1, 2)}, (0, w - 3)),
        ("bl", grid[-3:, :3], {(1, 0), (1, 1), (2, 0), (2, 1)}, (h - 3, 0)),
        ("br", grid[-3:, -3:], {(1, 1), (1, 2), (2, 1), (2, 2)}, (h - 3, w - 3)),
    ]

    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for _name, zone, local_cells, (base_r, base_c) in specs:
        marker_vals = [int(zone[r, c]) for r, c in local_cells]
        color = marker_vals[0]
        if color == bg or any(v != color for v in marker_vals):
            continue

        ok = True
        for r in range(zone.shape[0]):
            for c in range(zone.shape[1]):
                if (r, c) in local_cells:
                    continue
                if int(zone[r, c]) != bg:
                    ok = False
        if not ok:
            continue

        cells = [(base_r + r, base_c + c) for r, c in local_cells]
        candidates.append((color, cells))

    assert len(candidates) == 1, "expected exactly one isolated 2x2 corner marker"
    return candidates[0]


def _extract_layers(grid: np.ndarray, bg: int, marker_cells: list[tuple[int, int]]) -> dict[int, dict[str, np.ndarray | tuple[int, int, int, int]]]:
    work = np.array(grid, dtype=int, copy=True)
    for r, c in marker_cells:
        work[r, c] = bg

    layers: dict[int, dict[str, np.ndarray | tuple[int, int, int, int]]] = {}
    for color in sorted(set(map(int, work.ravel())) - {bg}):
        coords = np.argwhere(work == color)
        assert coords.size > 0, "every layer color must remain at least partially visible"
        layers[color] = {
            "coords": coords,
            "bbox": _bbox(coords),
        }

    assert 2 <= len(layers) <= 4, "expected 2 to 4 colored layers"
    sizes = {
        (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
        for bbox in (info["bbox"] for info in layers.values())
    }
    assert len(sizes) == 1, "all inferred layers must share one rectangle size"
    return layers


def _inside_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (rows >= r0) & (rows <= r1) & (cols >= c0) & (cols <= c1)


def _layer_order(layers: dict[int, dict[str, np.ndarray | tuple[int, int, int, int]]]) -> list[int]:
    colors = list(layers)
    edges = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}

    for upper in colors:
        coords = np.asarray(layers[upper]["coords"], dtype=int)
        for lower in colors:
            if upper == lower:
                continue
            bbox = layers[lower]["bbox"]
            assert isinstance(bbox, tuple)
            if np.any(_inside_bbox(coords, bbox)):
                edges[upper].add(lower)

    for upper, lowers in edges.items():
        for lower in lowers:
            indeg[lower] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        ready = sorted(color for color in remaining if indeg[color] == 0)
        assert len(ready) == 1, "visible occlusions must imply one total stacking order"
        color = ready[0]
        order.append(color)
        remaining.remove(color)
        for lower in edges[color]:
            indeg[lower] -= 1
    return order


def _restore_target(
    grid: np.ndarray,
    bg: int,
    target_color: int,
    layers: dict[int, dict[str, np.ndarray | tuple[int, int, int, int]]],
    order: list[int],
) -> np.ndarray:
    bbox = layers[target_color]["bbox"]
    assert isinstance(bbox, tuple)
    r0, c0, r1, c1 = bbox
    patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=False)

    idx = order.index(target_color)
    above = set(order[:idx])
    below = set(order[idx + 1 :])

    state = np.full(patch.shape, -1, dtype=int)
    state[patch == bg] = 0
    state[patch == target_color] = 1
    for color in below:
        state[patch == color] = 0

    unexpected = set(map(int, np.unique(patch))) - {bg, target_color} - above - below
    assert not unexpected, "target crop should only contain layer colors"

    h, w = patch.shape
    for c in range(w // 2):
        mc = w - 1 - c
        left = state[:, c].copy()
        right = state[:, mc].copy()
        both_known = (left != -1) & (right != -1)
        assert np.all(left[both_known] == right[both_known]), "known symmetric cells must agree"

        left_unknown = (left == -1) & (right != -1)
        right_unknown = (right == -1) & (left != -1)
        left[left_unknown] = right[left_unknown]
        right[right_unknown] = left[right_unknown]
        assert not np.any((left == -1) & (right == -1)), "every occluded cell needs one visible mirror"

        state[:, c] = left
        state[:, mc] = right

    assert np.all(state != -1), "target symmetry should resolve every hidden cell"

    out = np.full((h, w), bg, dtype=int)
    out[state == 1] = target_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full symmetric target layer from a stack of transparent monochrome rectangles.

    Input structure:
    - bg is the most common color.
    - 2 to 4 same-size rectangular layers are partially overlapped.
    - Each layer uses one non-bg color with bg holes acting as transparency.
    - A separate corner 2x2 marker uses the target layer's color.

    Rule:
    - Infer each layer rectangle from the visible pixels of its color.
    - Infer the top-to-bottom order from which colors visibly appear inside other layer rectangles.
    - Crop the target rectangle and classify each cell as known target, known background, or hidden by an upper layer.
    - Use the target layer's left-right symmetry to fill the hidden cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    target_color, marker_cells = _find_marker(g, bg)
    layers = _extract_layers(g, bg, marker_cells)
    assert target_color in layers, "marker color must match one visible layer"
    order = _layer_order(layers)
    return _restore_target(g, bg, target_color, layers, order)


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _mandatory_overlap(h: int, w: int, step_r: int, step_c: int, sign: int, has_lower: bool) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if not has_lower:
        return mask
    if sign > 0:
        mask[step_r:, step_c:] = True
    else:
        mask[step_r:, : w - step_c] = True
    return mask


def _random_non_target_mask(h: int, w: int, mandatory: np.ndarray) -> np.ndarray:
    mask = _frame_mask(h, w) | mandatory
    density = random.uniform(0.20, 0.55)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c]:
                continue
            if random.random() < density:
                mask[r, c] = True
    if np.all(mask):
        return np.zeros((0, 0), dtype=bool)
    return mask


def _random_target_mask(h: int, w: int, mandatory: np.ndarray) -> np.ndarray:
    mask = _frame_mask(h, w) | mandatory
    mask |= np.fliplr(mask)
    density = random.uniform(0.25, 0.65)
    for r in range(1, h - 1):
        for c in range(w // 2):
            if mask[r, c] or mask[r, w - 1 - c]:
                continue
            if random.random() < density:
                mask[r, c] = True
                mask[r, w - 1 - c] = True
    if np.all(mask):
        return np.zeros((0, 0), dtype=bool)
    return mask


def _render_scene(
    shape: tuple[int, int],
    bg: int,
    layers: list[dict[str, int | np.ndarray]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for layer in reversed(layers):
        r0 = int(layer["r0"])
        c0 = int(layer["c0"])
        color = int(layer["color"])
        mask = np.asarray(layer["mask"], dtype=bool)
        sub = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        sub[mask] = color
    return grid


def _place_marker(grid: np.ndarray, color: int, corner: str) -> None:
    if corner == "tl":
        grid[:2, :2] = color
    elif corner == "tr":
        grid[:2, -2:] = color
    elif corner == "bl":
        grid[-2:, :2] = color
    elif corner == "br":
        grid[-2:, -2:] = color
    else:
        raise ValueError(corner)


def _true_bbox(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0, c0, r0 + h - 1, c0 + w - 1


def _expected_output(layer: dict[str, int | np.ndarray], bg: int) -> np.ndarray:
    mask = np.asarray(layer["mask"], dtype=bool)
    color = int(layer["color"])
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.randint(2, 4)
    h = random.randint(6, 10)
    w = random.choice([6, 8, 10, 12])

    max_step_r = max(1, (h - 2) // (n_layers - 1))
    max_step_c = max(1, (w - 2) // (n_layers - 1))
    step_r = random.randint(1, max_step_r)
    step_c = random.randint(1, max_step_c)
    sign = random.choice([-1, 1])

    pad_top = random.randint(3, 6)
    pad_bottom = random.randint(3, 6)
    pad_left = random.randint(3, 6)
    pad_right = random.randint(3, 6)

    grid_h = pad_top + h + (n_layers - 1) * step_r + pad_bottom
    grid_w = pad_left + w + (n_layers - 1) * step_c + pad_right

    if sign > 0:
        base_c = pad_left
    else:
        base_c = pad_left + (n_layers - 1) * step_c
    base_r = pad_top

    colors = random.sample(range(10), n_layers + 1)
    bg = colors[0]
    layer_colors = colors[1:]
    target_idx = random.randint(1, n_layers - 1)
    marker_corner = random.choice(["tl", "tr", "bl", "br"])

    layers: list[dict[str, int | np.ndarray]] = []
    for idx in range(n_layers):
        r0 = base_r + idx * step_r
        c0 = base_c + sign * idx * step_c
        mandatory = _mandatory_overlap(h, w, step_r, step_c, sign, has_lower=(idx < n_layers - 1))
        if idx == target_idx:
            mask = _random_target_mask(h, w, mandatory)
        else:
            mask = _random_non_target_mask(h, w, mandatory)
        if mask.shape != (h, w):
            return None
        layers.append(
            {
                "color": layer_colors[idx],
                "r0": r0,
                "c0": c0,
                "mask": mask,
            }
        )

    target_layer = layers[target_idx]
    target_color = int(target_layer["color"])

    grid = _render_scene((grid_h, grid_w), bg, layers)
    _place_marker(grid, target_color, marker_corner)

    if _get_bg(grid) != bg:
        return None

    expected = _expected_output(target_layer, bg)
    if len(np.unique(expected)) < 2:
        return None

    try:
        marker_color, marker_cells = _find_marker(grid, bg)
        if marker_color != target_color:
            return None
        inferred_layers = _extract_layers(grid, bg, marker_cells)
        if set(inferred_layers) != set(layer_colors):
            return None
        for layer in layers:
            color = int(layer["color"])
            r0 = int(layer["r0"])
            c0 = int(layer["c0"])
            true_bbox = _true_bbox(r0, c0, h, w)
            if inferred_layers[color]["bbox"] != true_bbox:
                return None
        order = _layer_order(inferred_layers)
        if order != layer_colors:
            return None
        restored = _restore_target(grid, bg, target_color, inferred_layers, order)
        if not np.array_equal(restored, expected):
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    tr0 = int(target_layer["r0"])
    tc0 = int(target_layer["c0"])
    target_patch = grid[tr0 : tr0 + h, tc0 : tc0 + w]
    above_colors = {int(layer_colors[i]) for i in range(target_idx)}
    hidden = np.isin(target_patch, list(above_colors))
    if not np.any(hidden):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
