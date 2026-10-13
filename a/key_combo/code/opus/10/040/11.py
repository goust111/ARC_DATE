from __future__ import annotations

import random
from itertools import permutations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120
INPUT_SIZE_RANGE = (13, 17)
OBJECT_COLORS = (BLUE, GREEN)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_color_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox_from_coords(coords)})
    return comps


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _rect_distance(bbox: tuple[int, int, int, int], point: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    r, c = point
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _associate_markers(
    components: list[dict], markers: list[tuple[int, int]]
) -> list[tuple[dict, tuple[int, int]]]:
    assert len(components) == len(markers) == 2, "expected exactly two objects and two markers"
    best = None
    best_score = None
    for perm in permutations(markers):
        score = sum(_rect_distance(comp["bbox"], marker) for comp, marker in zip(components, perm))
        if best_score is None or score < best_score:
            best_score = score
            best = list(zip(components, perm))
    assert best is not None
    return best


def _recover_coords(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    strip_depth: int,
    axis: str,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    recovered: set[tuple[int, int]] = set()
    if axis == "lr":
        cutoff = c1 - strip_depth
        kept = [(int(r), int(c)) for r, c in coords if int(c) <= cutoff]
        mirror_sum = c0 + c1
        for r, c in kept:
            recovered.add((r, c))
            recovered.add((r, mirror_sum - c))
        return recovered

    cutoff = r1 - strip_depth
    kept = [(int(r), int(c)) for r, c in coords if int(r) <= cutoff]
    mirror_sum = r0 + r1
    for r, c in kept:
        recovered.add((r, c))
        recovered.add((mirror_sum - r, c))
    return recovered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase a marker-coded local strip and restore each damaged object by the marker-selected symmetry axis.

    Input structure:
    - A black background grid contains exactly two monochromatic 4-connected foreground objects, one blue and one green.
    - Each object has a nearby red marker pixel placed just outside its own bounding box on the right.
    - The marker's column offset encodes a strip depth of 1 or 2, while the marker row sits in either the object's upper or lower half.

    Rule:
    - Pair each object with its nearest red marker outside the object's bounding box.
    - Let `dc` be the marker's horizontal offset from the object's right bbox edge.
    - If the marker row is in the upper half (including the middle row), erase the object's rightmost `dc` local columns and mirror-complete it left-right.
    - Otherwise, erase the object's bottom `dc` local rows and mirror-complete it up-down.
    - Remove all red markers in the final output.

    Key insight:
    - One red control pixel carries two independent parameters at once: horizontal offset gives strip depth, and vertical placement chooses which symmetry axis repairs the object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = sorted(_find_color_components(g, OBJECT_COLORS), key=lambda comp: comp["color"])
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    pairs = _associate_markers(components, markers)

    out = np.zeros_like(g)
    for comp, marker in pairs:
        color = int(comp["color"])
        bbox = comp["bbox"]
        r0, _c0, r1, c1 = bbox
        marker_r, marker_c = marker
        dc = int(marker_c - c1)
        assert dc >= 1, "marker must be to the right of its object"
        mid_r = (r0 + r1) // 2
        axis = "lr" if marker_r <= mid_r else "ud"
        repaired = _recover_coords(comp["coords"], bbox, dc, axis)
        for r, c in repaired:
            out[r, c] = color
    return out


def _grow_connected(seed_mask: np.ndarray, target_area: int) -> np.ndarray:
    mask = seed_mask.copy()
    h, w = mask.shape
    frontier = set()
    for r, c in zip(*np.where(mask)):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    while int(mask.sum()) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _choose_off_center_segment(length: int, min_len: int) -> tuple[int, int]:
    choices = []
    for seg_len in range(min_len, length):
        for start in range(0, length - seg_len + 1):
            end = start + seg_len - 1
            if start != length - 1 - end:
                choices.append((start, end))
    if not choices:
        raise RuntimeError("no off-center segment available")
    return random.choice(choices)


def _make_lr_shape(dc: int) -> np.ndarray:
    for _ in range(80):
        if dc == 1:
            h = random.randint(5, 7)
            s, e = _choose_off_center_segment(h, 2)
            seg = np.zeros(h, dtype=bool)
            seg[s : e + 1] = True
            full = np.zeros((h, 3), dtype=bool)
            full[:, 1] = True
            full[:, 0] = seg
            full[:, 2] = seg
        else:
            h = random.randint(4, 6)
            s, e = _choose_off_center_segment(h, 1)
            seg = np.zeros(h, dtype=bool)
            seg[s : e + 1] = True
            full = np.zeros((h, 4), dtype=bool)
            full[:, 0] = True
            full[:, 3] = True
            full[:, 1] = seg
            full[:, 2] = seg
        if 10 <= int(full.sum()) <= 18 and _mask_connected(full):
            return full
    raise RuntimeError("failed to make lr shape")


def _make_ud_shape(dc: int) -> np.ndarray:
    for _ in range(80):
        if dc == 1:
            w = random.randint(5, 7)
            s, e = _choose_off_center_segment(w, 2)
            seg = np.zeros(w, dtype=bool)
            seg[s : e + 1] = True
            full = np.zeros((3, w), dtype=bool)
            full[1, :] = True
            full[0, :] = seg
            full[2, :] = seg
        else:
            w = random.randint(4, 6)
            s, e = _choose_off_center_segment(w, 1)
            seg = np.zeros(w, dtype=bool)
            seg[s : e + 1] = True
            full = np.zeros((4, w), dtype=bool)
            full[0, :] = True
            full[3, :] = True
            full[1, :] = seg
            full[2, :] = seg
        if 10 <= int(full.sum()) <= 18 and _mask_connected(full):
            return full
    raise RuntimeError("failed to make ud shape")


def _damage_shape(full_mask: np.ndarray, axis: str, dc: int) -> np.ndarray:
    h, w = full_mask.shape
    if axis == "lr":
        visible = full_mask.copy()
        if dc == 1:
            zero_rows = [r for r in range(h) if not full_mask[r, 2]]
        else:
            zero_rows = [r for r in range(h) if not full_mask[r, 2]]
        if not zero_rows:
            raise RuntimeError("no rows available for right-strip corruption")
        cut = random.randint(1, len(zero_rows))
        col = 2 if dc == 2 else w - 1
        for r in random.sample(zero_rows, cut):
            visible[r, col] = True
        if not _mask_connected(visible):
            raise RuntimeError("damaged lr shape disconnected")
        return visible
    else:
        visible = full_mask.copy()
        if dc == 1:
            zero_cols = [c for c in range(w) if not full_mask[2, c]]
        else:
            zero_cols = [c for c in range(w) if not full_mask[2, c]]
        if not zero_cols:
            raise RuntimeError("no cols available for bottom-strip corruption")
        cut = random.randint(1, len(zero_cols))
        row = 2 if dc == 2 else h - 1
        for c in random.sample(zero_cols, cut):
            visible[row, c] = True
        if not _mask_connected(visible):
            raise RuntimeError("damaged ud shape disconnected")
        return visible


def _stamp(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        for nr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for nc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                out[nr, nc] = True
    return out


def _place_objects(specs: list[dict], height: int, width: int) -> np.ndarray | None:
    for _ in range(PLACE_ATTEMPTS):
        grid = np.zeros((height, width), dtype=int)
        forbidden = np.zeros((height, width), dtype=bool)
        placed: list[dict] = []
        ok = True
        order = list(range(len(specs)))
        random.shuffle(order)
        for idx in order:
            spec = specs[idx]
            vis = spec["visible"]
            full = spec["full"]
            color = spec["color"]
            marker_r_local = spec["marker_r_local"]
            dc = spec["dc"]
            h, w = vis.shape
            max_r = height - h
            max_c = width - (w + dc)
            if max_r < 0 or max_c < 0:
                return None
            choices = [(r, c) for r in range(max_r + 1) for c in range(max_c + 1)]
            random.shuffle(choices)
            picked = None
            for top, left in choices:
                shape_cells = {(top + r, left + c) for r, c in zip(*np.where(vis))}
                marker = (top + marker_r_local, left + w - 1 + dc)
                if forbidden[marker]:
                    continue
                if any(forbidden[r, c] for r, c in shape_cells):
                    continue
                full_bbox = (top, left, top + full.shape[0] - 1, left + full.shape[1] - 1)
                picked = {"top": top, "left": left, "shape_cells": shape_cells, "marker": marker, "bbox": full_bbox}
                break
            if picked is None:
                ok = False
                break
            _stamp(grid, vis, picked["top"], picked["left"], color)
            grid[picked["marker"]] = RED
            occupancy = np.zeros_like(grid, dtype=bool)
            for r, c in picked["shape_cells"]:
                occupancy[r, c] = True
            occupancy[picked["marker"]] = True
            forbidden |= _dilate(occupancy)
            placed.append({**spec, **picked})
        if not ok:
            continue
        if _placement_semantics_ok(grid, placed):
            return grid
    return None


def _expected_output(height: int, width: int, placed: list[dict]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for spec in placed:
        _stamp(out, spec["full"], spec["top"], spec["left"], spec["color"])
    return out


def _recover_local(mask: np.ndarray, dc: int, axis: str) -> np.ndarray:
    coords = np.argwhere(mask)
    bbox = _bbox_from_coords(coords)
    recovered = _recover_coords(coords, bbox, dc, axis)
    out = np.zeros_like(mask, dtype=bool)
    for r, c in recovered:
        out[r, c] = True
    return out


def _placement_semantics_ok(grid: np.ndarray, placed: list[dict]) -> bool:
    if len(placed) != 2:
        return False

    expected = _expected_output(*grid.shape, placed)
    try:
        produced = main(grid)
    except Exception:
        return False
    if not np.array_equal(produced, expected):
        return False

    comps = sorted(_find_color_components(grid, OBJECT_COLORS), key=lambda comp: comp["color"])
    if len(comps) != 2 or len(np.argwhere(grid == RED)) != 2:
        return False

    for comp, spec in zip(comps, sorted(placed, key=lambda item: item["color"])):
        visible_local = comp["coords"].copy()
        visible_local[:, 0] -= spec["top"]
        visible_local[:, 1] -= spec["left"]
        local_mask = np.zeros_like(spec["visible"], dtype=bool)
        local_mask[visible_local[:, 0], visible_local[:, 1]] = True
        if not np.array_equal(local_mask, spec["visible"]):
            return False
        true_axis = spec["axis"]
        true_dc = spec["dc"]
        recovered = _recover_local(local_mask, true_dc, true_axis)
        if not np.array_equal(recovered, spec["full"]):
            return False
        for alt_axis in ("lr", "ud"):
            for alt_dc in (1, 2):
                if alt_axis == true_axis and alt_dc == true_dc:
                    continue
                alt = _recover_local(local_mask, alt_dc, alt_axis)
                if np.array_equal(alt, spec["full"]):
                    return False
    return True


def _one_spec(color: int, axis: str, dc: int) -> dict:
    for _ in range(120):
        full = _make_lr_shape(dc) if axis == "lr" else _make_ud_shape(dc)
        try:
            visible = _damage_shape(full, axis, dc)
        except RuntimeError:
            continue
        h, _w = full.shape
        if axis == "lr":
            marker_r_local = random.randint(0, (h - 1) // 2)
        else:
            marker_r_local = random.randint((h // 2) + 1, h - 1)
        return {
            "color": color,
            "axis": axis,
            "dc": dc,
            "full": full,
            "visible": visible,
            "marker_r_local": marker_r_local,
        }
    raise RuntimeError("failed to sample object spec")


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(*INPUT_SIZE_RANGE)
        width = random.randint(*INPUT_SIZE_RANGE)
        dcs = [1, 2]
        random.shuffle(dcs)
        colors = list(OBJECT_COLORS)
        random.shuffle(colors)
        specs = [
            _one_spec(colors[0], "lr", dcs[0]),
            _one_spec(colors[1], "ud", dcs[1]),
        ]
        grid = _place_objects(specs, height, width)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
