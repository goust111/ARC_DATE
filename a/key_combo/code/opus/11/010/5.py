from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (2, 3, 4, 7)
GEN_RETRIES = 400
RNG = np.random.default_rng()


SHAPE_LIBRARY: dict[str, np.ndarray] = {
    "domino": np.array([[1, 1]], dtype=bool),
    "line3": np.array([[1, 1, 1]], dtype=bool),
    "el3": np.array([[1, 0], [1, 1]], dtype=bool),
    "square": np.array([[1, 1], [1, 1]], dtype=bool),
    "tee": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "el4": np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    "zig4": np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
}

SLOT_RULES = {
    "NW": {"row_band": "top", "col_band": "left", "top_dir": "up", "bottom_dir": "left"},
    "NE": {"row_band": "top", "col_band": "right", "top_dir": "up", "bottom_dir": "right"},
    "SW": {"row_band": "bottom", "col_band": "left", "top_dir": "left", "bottom_dir": "down"},
    "SE": {"row_band": "bottom", "col_band": "right", "top_dir": "right", "bottom_dir": "down"},
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    bbox = _bbox_from_mask(mask)
    assert bbox is not None
    return _crop_bbox(mask.astype(bool), bbox).astype(bool)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    mins = coords.min(axis=0)
    rel = coords - mins
    h = int(rel[:, 0].max()) + 1
    w = int(rel[:, 1].max()) + 1
    out = np.zeros((h, w), dtype=bool)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _shape_variants(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    base = _trim_mask(mask)
    variants: list[tuple[tuple[int, int], ...]] = []
    for k in range(4):
        rot = np.rot90(base, k=k)
        for candidate in (rot, np.fliplr(rot)):
            trimmed = _trim_mask(candidate)
            coords = tuple(sorted(map(tuple, np.argwhere(trimmed))))
            variants.append(coords)
    return variants


def _canonical_shape_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_shape_variants(_mask_from_coords(coords)))


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _overlay_panels(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    conflict = (top != BLACK) & (bottom != BLACK) & (top != bottom)
    assert not np.any(conflict), "panel overlay color conflict"
    return np.where(top != BLACK, top, bottom)


def _extract_holes(object_mask: np.ndarray) -> list[np.ndarray]:
    h, w = object_mask.shape
    holes: list[np.ndarray] = []
    for comp in _components(~object_mask):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue
        holes.append(comp)
    return holes


def _marker_components(grid: np.ndarray, object_bbox: tuple[int, int, int, int]) -> list[dict]:
    markers: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        for coords in _components(grid == color):
            box = _bbox_from_coords(coords)
            if _bboxes_overlap(box, object_bbox):
                continue
            markers.append({"color": color, "coords": coords, "bbox": box})
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the two panels, reveal the object's hidden holes, fill each hole by marker-shape match, then crop the object.

    Input structure:
    - The grid has black background and even height, split into equal top and bottom panels.
    - Each panel contains part of the same grey object at the same position, plus a few small colored marker shapes outside that object.
    - Neither panel alone fully closes every internal black hole of the object, but the pixelwise union does.

    Rule:
    - Overlay the two panels cell by cell, taking the non-black color wherever either panel has foreground.
    - In the merged panel, find the grey object and its internal black 4-connected hole components.
    - Match each hole's normalized shape against the external marker shapes, allowing rotations and reflections.
    - Fill each hole with the color of its matching marker, then crop the grey object's bounding box.

    Key insight:
    - The markers identify colors by shape, but those hole shapes are only complete after the two incomplete object panels are merged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0
    half = g.shape[0] // 2
    top = g[:half, :]
    bottom = g[half:, :]
    merged = _overlay_panels(top, bottom)

    grey_components = _components(merged == GREY)
    assert len(grey_components) == 1, "expected exactly one merged grey object"
    object_bbox = _bbox_from_coords(grey_components[0])
    cropped = _crop_bbox(merged, object_bbox)
    object_mask = cropped == GREY
    holes = _extract_holes(object_mask)
    assert holes, "expected internal holes"

    marker_map: dict[tuple[tuple[int, int], ...], int] = {}
    for marker in _marker_components(merged, object_bbox):
        key = _canonical_shape_from_coords(marker["coords"])
        assert key not in marker_map, "marker shapes must be unique"
        marker_map[key] = int(marker["color"])

    used_keys: set[tuple[tuple[int, int], ...]] = set()
    for hole in holes:
        key = _canonical_shape_from_coords(hole)
        assert key in marker_map, "unmatched hole shape"
        used_keys.add(key)
        cropped[hole[:, 0], hole[:, 1]] = marker_map[key]

    assert used_keys == set(marker_map), "every marker must be used exactly once"
    return cropped


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if int(RNG.integers(2)):
        out = np.fliplr(out)
    return np.rot90(out, k=int(RNG.integers(4)))


def _hole_slot_ranges(
    slot: str, object_h: int, object_w: int, shape_h: int, shape_w: int
) -> tuple[list[int], list[int]]:
    row_mid = object_h // 2
    col_mid = object_w // 2
    if SLOT_RULES[slot]["row_band"] == "top":
        row_values = list(range(1, row_mid - shape_h + 1))
    else:
        row_values = list(range(row_mid, object_h - shape_h))
    if SLOT_RULES[slot]["col_band"] == "left":
        col_values = list(range(1, col_mid - shape_w + 1))
    else:
        col_values = list(range(col_mid, object_w - shape_w))
    return row_values, col_values


def _fits_with_padding(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(canvas.shape[0], r0 + h + pad)
    cc1 = min(canvas.shape[1], c0 + w + pad)
    if np.any(canvas[rr0:rr1, cc0:cc1]):
        return False
    return True


def _edge_anchor(hole_mask: np.ndarray, direction: str) -> tuple[int, int]:
    coords = np.argwhere(hole_mask)
    if direction == "left":
        edge = coords[:, 1].min()
        edge_cells = coords[coords[:, 1] == edge]
        pick = edge_cells[len(edge_cells) // 2]
        return int(pick[0]), int(edge)
    if direction == "right":
        edge = coords[:, 1].max()
        edge_cells = coords[coords[:, 1] == edge]
        pick = edge_cells[len(edge_cells) // 2]
        return int(pick[0]), int(edge)
    if direction == "up":
        edge = coords[:, 0].min()
        edge_cells = coords[coords[:, 0] == edge]
        pick = edge_cells[len(edge_cells) // 2]
        return int(edge), int(pick[1])
    edge = coords[:, 0].max()
    edge_cells = coords[coords[:, 0] == edge]
    pick = edge_cells[len(edge_cells) // 2]
    return int(edge), int(pick[1])


def _channel_cells(hole_mask: np.ndarray, direction: str, object_h: int, object_w: int) -> np.ndarray:
    out = np.zeros((object_h, object_w), dtype=bool)
    rr, cc = _edge_anchor(hole_mask, direction)
    if direction == "left":
        out[rr, :cc] = True
    elif direction == "right":
        out[rr, cc + 1 :] = True
    elif direction == "up":
        out[:rr, cc] = True
    else:
        out[rr + 1 :, cc] = True
    return out


def _closed_hole_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    return {_canonical_shape_from_coords(hole) for hole in _extract_holes(mask)}


def _place_markers(
    top: np.ndarray,
    bottom: np.ndarray,
    markers: list[tuple[int, np.ndarray]],
) -> bool:
    panel_h, panel_w = top.shape
    corners = ["TL", "TR", "BL", "BR"]
    RNG.shuffle(corners)
    if len(markers) > len(corners):
        return False

    for (color, shape), corner in zip(markers, corners):
        ph, pw = shape.shape
        if corner == "TL":
            r0, c0 = 0, 0
        elif corner == "TR":
            r0, c0 = 0, panel_w - pw
        elif corner == "BL":
            r0, c0 = panel_h - ph, 0
        else:
            r0, c0 = panel_h - ph, panel_w - pw
        panel_name = "top" if int(RNG.integers(2)) == 0 else "bottom"
        panel = top if panel_name == "top" else bottom
        if np.any(panel[r0 : r0 + ph, c0 : c0 + pw] != BLACK):
            return False
        panel[r0 : r0 + ph, c0 : c0 + pw][shape] = color
    return True


def _build_expected_output(
    object_mask: np.ndarray,
    hole_specs: list[dict],
    colors_by_key: dict[tuple[tuple[int, int], ...], int],
) -> np.ndarray:
    out = np.full(object_mask.shape, GREY, dtype=int)
    for spec in hole_specs:
        key = _canonical_shape_from_coords(spec["coords"])
        coords = spec["coords"]
        out[coords[:, 0], coords[:, 1]] = colors_by_key[key]
    return out


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = int(RNG.integers(13, 15))
    panel_w = int(RNG.integers(13, 16))

    margins = [int(RNG.integers(3, 5)) for _ in range(4)]
    top_margin, bottom_margin, left_margin, right_margin = margins
    object_h = panel_h - top_margin - bottom_margin
    object_w = panel_w - left_margin - right_margin
    if object_h < 7 or object_w < 7:
        return None

    hole_count = 2 + int(RNG.integers(2))
    slots = list(RNG.choice(np.array(list(SLOT_RULES)), size=hole_count, replace=False))
    shape_names = list(RNG.choice(np.array(list(SHAPE_LIBRARY)), size=hole_count, replace=False))

    hole_canvas = np.zeros((object_h, object_w), dtype=bool)
    hole_specs: list[dict] = []
    for slot, shape_name in zip(slots, shape_names):
        shape = _random_transform(SHAPE_LIBRARY[shape_name])
        sh, sw = shape.shape
        row_values, col_values = _hole_slot_ranges(slot, object_h, object_w, sh, sw)
        if not row_values or not col_values:
            return None

        placed = False
        for _ in range(20):
            r0 = int(RNG.choice(row_values))
            c0 = int(RNG.choice(col_values))
            if not _fits_with_padding(hole_canvas, shape, r0, c0, pad=1):
                continue
            hole_canvas[r0 : r0 + sh, c0 : c0 + sw][shape] = True
            coords = np.argwhere(hole_canvas[r0 : r0 + sh, c0 : c0 + sw]) + np.array([r0, c0])
            hole_specs.append({"slot": slot, "coords": coords.copy(), "shape": shape.copy()})
            placed = True
            break
        if not placed:
            return None

    if len(_components(hole_canvas)) != hole_count:
        return None

    object_mask = ~hole_canvas
    if len(_components(object_mask)) != 1:
        return None

    top_obj = object_mask.copy()
    bottom_obj = object_mask.copy()
    top_removed = np.zeros_like(object_mask, dtype=bool)
    bottom_removed = np.zeros_like(object_mask, dtype=bool)
    for spec in hole_specs:
        hole_mask = np.zeros_like(object_mask, dtype=bool)
        coords = spec["coords"]
        hole_mask[coords[:, 0], coords[:, 1]] = True
        top_channel = _channel_cells(hole_mask, SLOT_RULES[spec["slot"]]["top_dir"], object_h, object_w)
        bottom_channel = _channel_cells(hole_mask, SLOT_RULES[spec["slot"]]["bottom_dir"], object_h, object_w)
        if np.any(top_channel & bottom_channel):
            return None
        top_removed |= top_channel
        bottom_removed |= bottom_channel

    if np.any(top_removed & bottom_removed):
        return None

    top_obj[top_removed] = False
    bottom_obj[bottom_removed] = False
    if not np.array_equal(top_obj | bottom_obj, object_mask):
        return None
    if np.array_equal(top_obj, object_mask) or np.array_equal(bottom_obj, object_mask):
        return None
    if len(_components(top_obj)) != 1 or len(_components(bottom_obj)) != 1:
        return None

    final_keys = _closed_hole_keys(object_mask)
    if len(final_keys) != hole_count:
        return None
    if not _closed_hole_keys(top_obj) < final_keys:
        return None
    if not _closed_hole_keys(bottom_obj) < final_keys:
        return None

    top_panel = np.zeros((panel_h, panel_w), dtype=int)
    bottom_panel = np.zeros((panel_h, panel_w), dtype=int)
    obj_r0, obj_c0 = top_margin, left_margin
    top_panel[obj_r0 : obj_r0 + object_h, obj_c0 : obj_c0 + object_w][top_obj] = GREY
    bottom_panel[obj_r0 : obj_r0 + object_h, obj_c0 : obj_c0 + object_w][bottom_obj] = GREY

    colors = list(map(int, RNG.choice(np.array(MARKER_COLORS), size=hole_count, replace=False)))
    colors_by_key: dict[tuple[tuple[int, int], ...], int] = {}
    marker_specs: list[tuple[int, np.ndarray]] = []
    for color, spec in zip(colors, hole_specs):
        key = _canonical_shape_from_coords(spec["coords"])
        if key in colors_by_key:
            return None
        colors_by_key[key] = color
        marker_specs.append((color, _random_transform(_mask_from_coords(spec["coords"]))))

    if not _place_markers(top_panel, bottom_panel, marker_specs):
        return None

    inp = np.vstack([top_panel, bottom_panel])
    merged = _overlay_panels(top_panel, bottom_panel)
    merged_object = merged[obj_r0 : obj_r0 + object_h, obj_c0 : obj_c0 + object_w] == GREY
    if not np.array_equal(merged_object, object_mask):
        return None

    expected = _build_expected_output(object_mask, hole_specs, colors_by_key)
    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None

    grey_components = _components(merged == GREY)
    if len(grey_components) != 1:
        return None
    object_bbox = _bbox_from_coords(grey_components[0])
    if object_bbox != (obj_r0, obj_c0, obj_r0 + object_h - 1, obj_c0 + object_w - 1):
        return None

    marker_keys = {_canonical_shape_from_coords(marker["coords"]) for marker in _marker_components(merged, object_bbox)}
    if marker_keys != final_keys:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _attempt_generate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid overlay-hole ARC input")
