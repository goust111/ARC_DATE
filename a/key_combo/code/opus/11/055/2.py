from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
OBJECT_COLORS = (1, 3, 5, 8)
MARKER_COLORS = (2, 4, 6, 7)
MARKER_BY_OBJECT = {1: 2, 3: 4, 5: 6, 8: 7}

GRID_MIN = 17
GRID_MAX = 23
OBJECT_COUNT_CHOICES = (3, 4)
PADDING = 2
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 250


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, mask.shape, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_coords(coords: np.ndarray, axis: str, box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    mirrored: set[tuple[int, int]] = set()
    if axis == "vertical":
        for r, c in coords:
            mirrored.add((int(r), int(c0 + c1 - c)))
    else:
        for r, c in coords:
            mirrored.add((int(r0 + r1 - r), int(c)))
    return mirrored


def _hole_components(completed: set[tuple[int, int]], box: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    local_h = r1 - r0 + 1
    local_w = c1 - c0 + 1
    mask = np.zeros((local_h, local_w), dtype=bool)
    for r, c in completed:
        mask[r - r0, c - c0] = True

    holes: list[list[tuple[int, int]]] = []
    for comp in _components(~mask, connectivity=4):
        touches_edge = bool(
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == local_h - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == local_w - 1)
        )
        if not touches_edge:
            holes.append([(int(r0 + rr), int(c0 + cc)) for rr, cc in comp])
    return holes


def _object_components(grid: np.ndarray) -> list[dict]:
    objs: list[dict] = []
    for color in OBJECT_COLORS:
        for coords in _components(grid == color, connectivity=4):
            objs.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    objs.sort(key=lambda item: item["bbox"])
    return objs


def _adjacent_markers(grid: np.ndarray, coords: np.ndarray) -> set[tuple[int, int]]:
    marker_cells: set[tuple[int, int]] = set()
    marker_palette = set(MARKER_COLORS)
    h, w = grid.shape
    for r, c in coords:
        for nr, nc in _neighbors(int(r), int(c), (h, w), connectivity=8):
            if int(grid[nr, nc]) in marker_palette:
                marker_cells.add((nr, nc))
    return marker_cells


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    infos: list[dict] = []
    for obj in _object_components(g):
        color = obj["color"]
        coords = obj["coords"]
        r0, c0, r1, c1 = obj["bbox"]
        markers = _adjacent_markers(g, coords)
        assert len(markers) == 1, "each object must have exactly one adjacent marker"
        marker = next(iter(markers))
        mr, mc = marker
        marker_color = int(g[mr, mc])
        assert marker_color in MARKER_COLORS, "marker must use marker palette"
        assert marker_color != color, "marker color must differ from object color"

        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        is_vertical = (r1 - r0 + 1) % 2 == 1 and mr == mid_r and mc in {c0 - 1, c1 + 1}
        is_horizontal = (c1 - c0 + 1) % 2 == 1 and mc == mid_c and mr in {r0 - 1, r1 + 1}
        assert is_vertical != is_horizontal, "marker must encode exactly one mirror axis"
        axis = "vertical" if is_vertical else "horizontal"

        original = {tuple(map(int, rc)) for rc in coords}
        mirrored = _mirror_coords(coords, axis, obj["bbox"])
        completed = original | mirrored
        holes = _hole_components(completed, obj["bbox"])
        infos.append(
            {
                "color": color,
                "coords": coords,
                "bbox": obj["bbox"],
                "axis": axis,
                "marker": marker,
                "marker_color": marker_color,
                "original": original,
                "completed": completed,
                "new_cells": completed - original,
                "holes": holes,
                "has_hole": bool(holes),
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each marked half-object, then fill any newly enclosed holes with the marker color.

    Input structure:
    - A black background contains several separated 4-connected monochrome objects in object colors.
    - Each object has exactly one 8-neighbor marker pixel in a different marker color.
    - The marker sits at the midpoint just outside one bbox edge, encoding a horizontal or vertical mirror axis.

    Rule:
    - For each object, use the marker position to decide whether to mirror across the bbox's horizontal or vertical midline.
    - Add the mirrored pixels in the object's own color to complete the shape inside the same bbox.
    - If the completed shape encloses any 4-connected black holes, fill all those holes with the marker color and keep the marker; otherwise erase the marker to black.

    Key insight:
    - The single marker pixel simultaneously chooses the mirror axis, supplies the hole-fill color, and is only preserved when the mirror completion creates enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for info in _analyze_objects(g):
        for r, c in info["completed"]:
            out[r, c] = info["color"]
        if info["has_hole"]:
            for hole in info["holes"]:
                for r, c in hole:
                    out[r, c] = info["marker_color"]
        else:
            mr, mc = info["marker"]
            out[mr, mc] = BACKGROUND
    return out


def _build_half_frame(height: int, width: int, hole: bool) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mid = width // 2
    mask[:, 0] = True
    if hole or random.random() < 0.5:
        mask[0, : mid + 1] = True
    if hole or not mask[0].any() or random.random() < 0.5:
        mask[-1, : mid + 1] = True
    if not hole and mask[0, : mid + 1].all() and mask[-1, : mid + 1].all():
        if random.random() < 0.5:
            mask[0, 1 : mid + 1] = False
        else:
            mask[-1, 1 : mid + 1] = False
    return mask


def _make_local_patch(
    object_color: int,
    orientation: str,
    hole: bool,
) -> np.ndarray:
    height = random.choice((5, 7))
    width = random.choice((5, 7))
    base = _build_half_frame(height, width, hole)
    if orientation == "horizontal":
        base = np.rot90(base)

    h, w = base.shape
    side = random.choice(("neg", "pos"))
    if orientation == "vertical":
        patch = np.zeros((h, w + 1), dtype=int)
        if side == "neg":
            patch[:, 1:] = np.where(base, object_color, BACKGROUND)
            patch[h // 2, 0] = MARKER_BY_OBJECT[object_color]
        else:
            patch[:, :w] = np.where(base, object_color, BACKGROUND)
            patch[h // 2, w] = MARKER_BY_OBJECT[object_color]
    else:
        patch = np.zeros((h + 1, w), dtype=int)
        if side == "neg":
            patch[1:, :] = np.where(base, object_color, BACKGROUND)
            patch[0, w // 2] = MARKER_BY_OBJECT[object_color]
        else:
            patch[:h, :] = np.where(base, object_color, BACKGROUND)
            patch[h, w // 2] = MARKER_BY_OBJECT[object_color]
    return patch


def _reserve_cells(cells: list[tuple[int, int]], shape: tuple[int, int], padding: int) -> set[tuple[int, int]]:
    h, w = shape
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    reserved.add((rr, cc))
    return reserved


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    reserved: set[tuple[int, int]],
    *,
    attempts: int = MAX_PLACEMENT_ATTEMPTS,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    gh, gw = grid.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != BACKGROUND)]
    for _ in range(attempts):
        r0 = random.randint(0, gh - ph)
        c0 = random.randint(0, gw - pw)
        placed = [(r0 + r, c0 + c) for r, c in coords]
        if any(cell in reserved for cell in placed):
            continue
        return r0, c0
    return None


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BACKGROUND
    h, w = patch.shape
    dst_slice = dst[r0 : r0 + h, c0 : c0 + w]
    dst_slice[mask] = patch[mask]


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    infos = _analyze_objects(grid)
    assert 3 <= len(infos) <= 4, "expected 3-4 objects"
    assert any(info["axis"] == "vertical" for info in infos), "need vertical-axis coverage"
    assert any(info["axis"] == "horizontal" for info in infos), "need horizontal-axis coverage"
    assert any(info["has_hole"] for info in infos), "need at least one hole-producing object"
    assert any(not info["has_hole"] for info in infos), "need at least one non-hole object"
    assert int(np.count_nonzero(np.isin(grid, MARKER_COLORS))) == len(infos), "marker count mismatch"

    seen_completed: set[tuple[int, int]] = set()
    for info in infos:
        coords = info["coords"]
        assert 6 <= len(coords) <= 15, "input object size out of range"
        assert info["new_cells"], "mirror must add pixels"
        original_holes = _hole_components(info["original"], info["bbox"])
        assert not original_holes, "input object should not already contain holes"
        assert not (seen_completed & info["completed"]), "completed objects must not overlap"
        seen_completed |= info["completed"]

    out = main(grid)
    assert out.shape == grid.shape
    for info in infos:
        mr, mc = info["marker"]
        if info["has_hole"]:
            assert out[mr, mc] == info["marker_color"], "hole case must keep marker"
            filled = {cell for hole in info["holes"] for cell in hole}
            assert filled, "hole case must fill at least one cell"
            assert all(out[r, c] == info["marker_color"] for r, c in filled), "holes must use marker color"
        else:
            assert out[mr, mc] == BACKGROUND, "non-hole case must erase marker"


def _sample_layout() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    object_count = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(list(OBJECT_COLORS), object_count)

    descriptors: list[tuple[int, str, bool]] = []
    while True:
        descriptors = []
        for color in colors:
            orientation = random.choice(("vertical", "horizontal"))
            hole = random.choice((False, True))
            descriptors.append((color, orientation, hole))
        if any(o == "vertical" for _c, o, _h in descriptors) and any(o == "horizontal" for _c, o, _h in descriptors):
            if any(hole for _c, _o, hole in descriptors) and any(not hole for _c, _o, hole in descriptors):
                break

    patches = []
    for color, orientation, hole in descriptors:
        patch = _make_local_patch(color, orientation, hole)
        patches.append((patch.shape[0] * patch.shape[1], patch, color, orientation, hole))
    patches.sort(reverse=True, key=lambda item: item[0])

    reserved: set[tuple[int, int]] = set()
    for _area, patch, _color, _orientation, _hole in patches:
        loc = _place_patch(grid, patch, reserved)
        if loc is None:
            return None
        r0, c0 = loc
        _paste(grid, patch, r0, c0)
        cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(patch != BACKGROUND)]
        reserved |= _reserve_cells(cells, grid.shape, PADDING)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_layout()
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid mirror-hole ARC input")
