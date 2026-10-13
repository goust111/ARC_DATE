import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5]
GEN_ATTEMPTS = 400


def bbox(mask: np.ndarray):
    """arc_utils-style inclusive bbox for a boolean mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_coords(coords: np.ndarray):
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """arc_utils-style connected components for a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _mirror_patch(patch: np.ndarray, axis: str) -> np.ndarray:
    if axis == "vertical":
        return patch[:, ::-1].copy()
    if axis == "horizontal":
        return patch[::-1, :].copy()
    raise ValueError(f"unknown axis {axis}")


def _extract_layers(local: np.ndarray) -> list[dict]:
    layers = []
    for color in sorted(int(v) for v in np.unique(local) if int(v) != BG):
        mask = local == color
        box = bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        assert side_h == side_w and side_h % 2 == 1
        patch = np.where(crop_bbox(local, box) == color, color, BG)
        layers.append(
            {
                "color": color,
                "bbox": box,
                "side": side_h,
                "patch": patch,
            }
        )
    layers.sort(key=lambda item: item["side"], reverse=True)
    assert len({layer["side"] for layer in layers}) == len(layers)
    return layers


def _indicator_corner(coords: np.ndarray, shape: tuple[int, int]) -> str:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    if rows.max() <= 1 and cols.max() <= 1:
        return "tl"
    if rows.max() <= 1 and cols.min() >= w - 2:
        return "tr"
    if rows.min() >= h - 2 and cols.max() <= 1:
        return "bl"
    if rows.min() >= h - 2 and cols.min() >= w - 2:
        return "br"
    raise AssertionError("indicator is not in a corner")


def _decode_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1

    fg_comps = components_mask(g != BG, connectivity=4)
    assert len(fg_comps) >= 2

    nested_coords = max(fg_comps, key=len)
    nested_bbox = bbox_coords(nested_coords)
    nr0, nc0, nr1, nc1 = nested_bbox
    assert nr1 - nr0 == nc1 - nc0
    local = crop_bbox(g, nested_bbox)
    layers = _extract_layers(local)
    assert len(layers) in (3, 4)
    outer_color = layers[0]["color"]

    indicator_coords = None
    for comp in fg_comps:
        if len(comp) != 2:
            continue
        box = bbox_coords(comp)
        if (box[2] - box[0] + 1, box[3] - box[1] + 1) not in {(1, 2), (2, 1)}:
            continue
        corner = _indicator_corner(comp, g.shape)
        indicator_coords = comp
        break
    assert indicator_coords is not None
    corner = _indicator_corner(indicator_coords, g.shape)
    axis = "vertical" if corner in {"tl", "br"} else "horizontal"

    center_r = (nr0 + nr1) / 2.0
    center_c = (nc0 + nc1) / 2.0

    cells = [tuple(map(int, rc)) for rc in indicator_coords]
    colors = [int(g[r, c]) for r, c in cells]
    assert colors[0] != colors[1]
    assert sum(color == outer_color for color in colors) == 1
    distances = [abs(r - center_r) + abs(c - center_c) for r, c in cells]
    assert distances[0] != distances[1]
    inward_idx = 0 if distances[0] < distances[1] else 1
    inward_color = colors[inward_idx]
    parity = "odd" if inward_color == outer_color else "even"

    return {
        "grid": g,
        "nested_bbox": nested_bbox,
        "local": local,
        "layers": layers,
        "outer_color": outer_color,
        "indicator_coords": cells,
        "corner": corner,
        "axis": axis,
        "parity": parity,
    }


def _apply_scene(scene: dict) -> np.ndarray:
    g = scene["grid"]
    out = np.zeros_like(g)
    local_out = np.zeros_like(scene["local"])
    want_odd = scene["parity"] == "odd"

    for index, layer in enumerate(scene["layers"], start=1):
        patch = layer["patch"]
        selected = (index % 2 == 1) if want_odd else (index % 2 == 0)
        if selected:
            patch = _mirror_patch(patch, scene["axis"])
        r0, c0, r1, c1 = layer["bbox"]
        target = local_out[r0 : r1 + 1, c0 : c1 + 1]
        target[patch != BG] = patch[patch != BG]

    nr0, nc0, nr1, nc1 = scene["nested_bbox"]
    out[nr0 : nr1 + 1, nc0 : nc1 + 1] = local_out
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Selectively mirror odd or even layers of a nested square-shell object using a two-cell corner indicator.

    Input structure:
    - The background is black.
    - A large centered foreground component is made of 3-4 concentric one-cell-thick square layers, with the innermost layer possibly being a single center pixel.
    - Each layer has its own color and may contain an asymmetric missing cell so that mirroring a layer is visible.
    - A separate two-cell indicator sits in one corner; its two colors differ, and exactly one of them matches the outermost layer color.

    Rule:
    - Detect the centered nested component and order its layers from outer to inner by their nested square bboxes.
    - Decode the indicator corner: top-left or bottom-right means mirror selected layers left-right, while top-right or bottom-left means mirror them up-down.
    - Compare the two indicator cells by which one is closer to the nested component. If the closer cell uses the outer-layer color, mirror odd-numbered layers; otherwise mirror even-numbered layers.
    - Mirror only the selected layers inside their own square bboxes, reassemble the nested object at the original center, and remove the indicator.

    Key insight:
    - The indicator carries two independent control channels at once: corner location chooses the mirror axis, and color order toward the center chooses which layer parity is transformed.
    """
    return _apply_scene(_decode_scene(np.asarray(grid, dtype=int)))


def _ring_coords(side: int) -> list[tuple[int, int]]:
    coords = []
    for r in range(side):
        for c in range(side):
            if r in {0, side - 1} or c in {0, side - 1}:
                coords.append((r, c))
    return coords


def _fixed_under_axis(cell: tuple[int, int], side: int, axis: str) -> bool:
    r, c = cell
    mid = side // 2
    if axis == "vertical":
        return c == mid
    if axis == "horizontal":
        return r == mid
    raise ValueError(axis)


def _pick_gap(side: int, axis: str, must_move: bool) -> tuple[int, int]:
    candidates = _ring_coords(side)
    if must_move:
        candidates = [cell for cell in candidates if not _fixed_under_axis(cell, side, axis)]
    return random.choice(candidates)


def _indicator_layout(corner: str, side: int):
    if corner == "tl":
        return [
            ([(0, 0), (0, 1)], (0, 1), "h"),
            ([(0, 0), (1, 0)], (1, 0), "v"),
        ]
    if corner == "tr":
        return [
            ([(0, side - 2), (0, side - 1)], (0, side - 2), "h"),
            ([(0, side - 1), (1, side - 1)], (1, side - 1), "v"),
        ]
    if corner == "bl":
        return [
            ([(side - 1, 0), (side - 1, 1)], (side - 1, 1), "h"),
            ([(side - 2, 0), (side - 1, 0)], (side - 2, 0), "v"),
        ]
    if corner == "br":
        return [
            ([(side - 1, side - 2), (side - 1, side - 1)], (side - 1, side - 2), "h"),
            ([(side - 2, side - 1), (side - 1, side - 1)], (side - 2, side - 1), "v"),
        ]
    raise ValueError(corner)


def _build_case() -> np.ndarray:
    layer_count = random.choice([3, 4])
    outer_side = 2 * layer_count - 1
    grid_side = 11 if layer_count == 4 else random.choice([9, 11])
    offset0 = (grid_side - outer_side) // 2

    corner = random.choice(["tl", "tr", "bl", "br"])
    axis = "vertical" if corner in {"tl", "br"} else "horizontal"
    select_odd = bool(random.getrandbits(1))

    layer_colors = random.sample(PALETTE, layer_count)
    outer_color = layer_colors[0]
    command_color = random.choice([c for c in PALETTE if c not in layer_colors])

    grid = np.zeros((grid_side, grid_side), dtype=int)
    for index, color in enumerate(layer_colors, start=1):
        side = outer_side - 2 * (index - 1)
        offset = offset0 + (index - 1)
        if side == 1:
            grid[offset, offset] = color
            continue
        selected = (index % 2 == 1) if select_odd else (index % 2 == 0)
        gap = _pick_gap(side, axis=axis, must_move=selected)
        for r, c in _ring_coords(side):
            if (r, c) != gap:
                grid[offset + r, offset + c] = color

    cells, inward_cell, _orientation = random.choice(_indicator_layout(corner, grid_side))
    inward_color = outer_color if select_odd else command_color
    outward_color = command_color if select_odd else outer_color
    for r, c in cells:
        grid[r, c] = outward_color
    grid[inward_cell] = inward_color

    return grid


def _validate_case(grid: np.ndarray) -> dict:
    scene = _decode_scene(grid)
    out = main(grid)
    expected = _apply_scene(scene)
    assert np.array_equal(out, expected)
    assert out.shape == grid.shape and max(out.shape) <= 30
    assert not np.any(out[np.array(scene["indicator_coords"])[:, 0], np.array(scene["indicator_coords"])[:, 1]])

    local_out = crop_bbox(out, scene["nested_bbox"])
    want_odd = scene["parity"] == "odd"
    visible_selected = 0
    for index, layer in enumerate(scene["layers"], start=1):
        r0, c0, r1, c1 = layer["bbox"]
        actual = local_out[r0 : r1 + 1, c0 : c1 + 1]
        actual_layer = np.where(actual == layer["color"], layer["color"], BG)
        selected = (index % 2 == 1) if want_odd else (index % 2 == 0)
        if selected:
            mirrored = _mirror_patch(layer["patch"], scene["axis"])
            assert np.array_equal(actual_layer, mirrored)
            if layer["side"] > 1 and not np.array_equal(layer["patch"], mirrored):
                visible_selected += 1
        else:
            assert np.array_equal(actual_layer, layer["patch"])
    assert visible_selected >= 1
    return scene


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        try:
            _validate_case(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid nested selective-mirror input")
