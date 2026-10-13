from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
BACKGROUND = 0
RING_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r1, c) for c in range(c0, c1 + 1)]
    coords += [(r, c0) for r in range(r0 + 1, r1)]
    coords += [(r, c1) for r in range(r0 + 1, r1)]
    return coords


def _side_gap_coord(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    if side == "top":
        return r0, center_c
    if side == "bottom":
        return r1, center_c
    if side == "left":
        return center_r, c0
    if side == "right":
        return center_r, c1
    raise ValueError(f"unknown side: {side}")


def _paint_border(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def _marker_side(shape: tuple[int, int], r: int, c: int, center_r: int, center_c: int) -> str:
    h, w = shape
    if r == 0 and c == center_c:
        return "top"
    if r == h - 1 and c == center_c:
        return "bottom"
    if c == 0 and r == center_r:
        return "left"
    if c == w - 1 and r == center_r:
        return "right"
    raise AssertionError("marker must lie on the boundary and align with the ring center")


def _infer_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) >= 2, "expected grey rings plus colored markers"

    ring_color = max(colors, key=lambda color: int(np.count_nonzero(g == color)))
    ring_mask = g == ring_color
    outer_bbox = _bbox(ring_mask)
    assert outer_bbox is not None, "expected at least one ring pixel"

    r0, c0, r1, c1 = outer_bbox
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    assert outer_h >= 3 and outer_w >= 3
    assert outer_h % 2 == 1 and outer_w % 2 == 1, "ring rectangle must have a single center row and column"
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    layers: list[dict] = []
    reconstructed = np.zeros_like(ring_mask, dtype=bool)
    inset = 0
    while True:
        bbox = (r0 + 2 * inset, c0 + 2 * inset, r1 - 2 * inset, c1 - 2 * inset)
        rr0, cc0, rr1, cc1 = bbox
        if rr1 - rr0 + 1 < 3 or cc1 - cc0 + 1 < 3:
            break

        gap_coords = {side: _side_gap_coord(bbox, side) for side in SIDES}
        border = _border_coords(bbox)
        if not any(int(g[cell]) == ring_color for cell in border):
            break
        gaps = set()
        for side, cell in gap_coords.items():
            value = int(g[cell])
            assert value in (bg, ring_color), "ring cells must be either grey or a one-cell gap"
            if value == bg:
                gaps.add(side)

        gap_cells = set(gap_coords.values())
        for cell in border:
            value = int(g[cell])
            if cell in gap_cells:
                assert value in (bg, ring_color)
                if value == ring_color:
                    reconstructed[cell] = True
            else:
                assert value == ring_color, "only side-center cells may be missing"
                reconstructed[cell] = True

        layers.append({"bbox": bbox, "gaps": gaps})
        inset += 1

    assert layers, "expected at least one ring layer"
    assert np.array_equal(reconstructed, ring_mask), "grey pixels must match concentric layer borders exactly"

    marker_coords = np.argwhere((g != bg) & (g != ring_color))
    assert len(marker_coords) >= 1, "expected boundary markers"
    markers: list[dict] = []
    seen_sides: set[str] = set()
    for r, c in marker_coords:
        rr, cc = int(r), int(c)
        side = _marker_side(g.shape, rr, cc, center_r, center_c)
        assert side not in seen_sides, "expected at most one marker per side"
        seen_sides.add(side)
        markers.append({"side": side, "color": int(g[rr, cc]), "coord": (rr, cc)})

    targets: dict[str, int] = {}
    for marker in markers:
        target = None
        for depth, layer in enumerate(layers):
            if marker["side"] in layer["gaps"]:
                continue
            target = depth
            break
        assert target is not None, "each marker must hit some ring"
        targets[marker["side"]] = target
    assert len(set(targets.values())) == len(targets), "different markers must stop on different layers"

    return {
        "shape": tuple(map(int, g.shape)),
        "bg": int(bg),
        "ring_color": int(ring_color),
        "outer_bbox": outer_bbox,
        "center": (center_r, center_c),
        "layers": layers,
        "markers": markers,
        "targets": targets,
    }


def _render_output_from_structure(structure: dict) -> np.ndarray:
    h, w = structure["shape"]
    out = np.full((h, w), int(structure["bg"]), dtype=int)

    layer_colors = [int(structure["ring_color"]) for _ in structure["layers"]]
    marker_color_by_side = {marker["side"]: int(marker["color"]) for marker in structure["markers"]}
    for side, depth in structure["targets"].items():
        layer_colors[int(depth)] = marker_color_by_side[side]

    for color, layer in zip(layer_colors, structure["layers"]):
        _paint_border(out, layer["bbox"], color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each boundary marker inward through aligned ring gaps, recolor the first blocking ring, and seal every gap.

    Input structure:
    - A background canvas contains several one-cell-thick concentric rectangular rings, all initially grey.
    - Some ring sides have a one-cell gap exactly on the shared center row or center column, forming straight corridors.
    - Colored single-pixel markers sit on the grid boundary, each aligned with the ring center row or center column.

    Rule:
    - For each marker, move straight toward the ring center along its row or column.
    - The marker passes through every outer ring whose touched side has an aligned gap and stops at the first ring whose touched side is complete.
    - Recolor that entire blocking ring to the marker color, erase all markers, and fill every gap with the final color of its ring.

    Key insight:
    - A marker's destination is determined by corridor depth: the count of aligned outer-ring gaps on its side, not by marker color, distance, or processing order.
    """
    structure = _infer_structure(np.asarray(grid, dtype=int))
    return _render_output_from_structure(structure)


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [value for value in range(int(lo), int(hi) + 1) if value % 2 == 1]


def _sample_spec() -> dict | None:
    layer_count = random.randint(2, 5)
    inner_h = random.choice((3, 5, 7))
    inner_w = random.choice((3, 5, 7))
    outer_h = inner_h + 4 * (layer_count - 1)
    outer_w = inner_w + 4 * (layer_count - 1)

    top = random.randint(2, 4)
    bottom = random.randint(2, 4)
    left = random.randint(2, 4)
    right = random.randint(2, 4)
    height = outer_h + top + bottom
    width = outer_w + left + right
    if height > 30 or width > 30:
        return None

    marker_count = random.randint(2, min(4, layer_count))
    sides = random.sample(SIDES, marker_count)
    depths = random.sample(range(layer_count), marker_count)
    colors = random.sample(MARKER_COLORS, marker_count)

    side_to_depth = dict(zip(sides, depths))
    side_to_color = dict(zip(sides, colors))
    gaps_by_layer = [set() for _ in range(layer_count)]
    for side, depth in side_to_depth.items():
        for layer in range(depth):
            gaps_by_layer[layer].add(side)

    outer_bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
    center_r = top + outer_h // 2
    center_c = left + outer_w // 2

    layers = []
    for depth in range(layer_count):
        bbox = (
            outer_bbox[0] + 2 * depth,
            outer_bbox[1] + 2 * depth,
            outer_bbox[2] - 2 * depth,
            outer_bbox[3] - 2 * depth,
        )
        layers.append({"bbox": bbox, "gaps": set(gaps_by_layer[depth])})

    marker_coords = {
        "top": (0, center_c),
        "bottom": (height - 1, center_c),
        "left": (center_r, 0),
        "right": (center_r, width - 1),
    }
    markers = [
        {"side": side, "color": int(side_to_color[side]), "coord": marker_coords[side]}
        for side in sides
    ]

    return {
        "shape": (height, width),
        "bg": BACKGROUND,
        "ring_color": RING_COLOR,
        "outer_bbox": outer_bbox,
        "center": (center_r, center_c),
        "layers": layers,
        "markers": markers,
        "targets": {side: int(depth) for side, depth in side_to_depth.items()},
    }


def _render_input_from_spec(spec: dict) -> np.ndarray:
    h, w = spec["shape"]
    grid = np.full((h, w), int(spec["bg"]), dtype=int)
    for layer in spec["layers"]:
        _paint_border(grid, layer["bbox"], int(spec["ring_color"]))
        for side in layer["gaps"]:
            rr, cc = _side_gap_coord(layer["bbox"], side)
            grid[rr, cc] = int(spec["bg"])
    for marker in spec["markers"]:
        rr, cc = marker["coord"]
        grid[rr, cc] = int(marker["color"])
    return grid


def _validate_spec(spec: dict, inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if inp.shape != spec["shape"] or out.shape != spec["shape"]:
        return False
    if max(inp.shape) > 30:
        return False
    if len(spec["markers"]) < 2:
        return False
    if len(set(spec["targets"].values())) != len(spec["targets"]):
        return False
    if not any(depth > 0 for depth in spec["targets"].values()):
        return False

    gap_cells: set[tuple[int, int]] = set()
    for marker in spec["markers"]:
        side = marker["side"]
        depth = int(spec["targets"][side])
        for layer_index, layer in enumerate(spec["layers"]):
            rr, cc = _side_gap_coord(layer["bbox"], side)
            if layer_index < depth:
                if int(inp[rr, cc]) != int(spec["bg"]):
                    return False
                gap_cells.add((rr, cc))
            else:
                if int(inp[rr, cc]) != int(spec["ring_color"]):
                    return False

    actual_gap_cells = set()
    for layer in spec["layers"]:
        for side in layer["gaps"]:
            actual_gap_cells.add(_side_gap_coord(layer["bbox"], side))
    if gap_cells != actual_gap_cells:
        return False

    layer_colors = [int(spec["ring_color"]) for _ in spec["layers"]]
    for marker in spec["markers"]:
        layer_colors[int(spec["targets"][marker["side"]])] = int(marker["color"])
    for color, layer in zip(layer_colors, spec["layers"]):
        for rr, cc in _border_coords(layer["bbox"]):
            if int(out[rr, cc]) != color:
                return False

    solved = main(inp)
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        spec = _sample_spec()
        if spec is None:
            continue
        inp = _render_input_from_spec(spec)
        out = _render_output_from_structure(spec)
        if _validate_spec(spec, inp, out):
            return inp
    raise RuntimeError("failed to generate a valid corridor-ring projection input")
