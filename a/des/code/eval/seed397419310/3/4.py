from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "I3": ((0, 0), (0, 1), (0, 2)),
    "L3": ((0, 0), (1, 0), (1, 1)),
    "T4": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "Z4": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "V5": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    "P5": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
}
CHIRAL_SHAPES = ("Z4", "P5")
GEN_RETRIES = 400
PLACE_RETRIES = 160


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_size(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    norm = _normalize(coords)
    return max(r for r, _ in norm) + 1, max(c for _, c in norm) + 1


def _rotate90(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    norm = _normalize(coords)
    h = max(r for r, _ in norm) + 1
    return _normalize([(c, h - 1 - r) for r, c in norm])


def _flip_h(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    norm = _normalize(coords)
    w = max(c for _, c in norm) + 1
    return _normalize([(r, w - 1 - c) for r, c in norm])


def _dihedral_orientations(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], bool, int]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], bool, int]] = []

    cur = _normalize(coords)
    for rot_k in range(4):
        norm = _normalize(cur)
        if norm not in seen:
            seen.add(norm)
            out.append((norm, False, rot_k))
        cur = _rotate90(cur)

    cur = _flip_h(coords)
    for rot_k in range(4):
        norm = _normalize(cur)
        if norm not in seen:
            seen.add(norm)
            out.append((norm, True, rot_k))
        cur = _rotate90(cur)

    return out


def _canonical(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(ori for ori, _, _ in _dihedral_orientations(coords))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend_frame(g: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for cc in range(c0, c1 + 1):
            border.add((r0, cc))
            border.add((r1, cc))
        for rr in range(r0, r1 + 1):
            border.add((rr, c0))
            border.add((rr, c1))
        if set(comp) == border:
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, f"expected one grey legend frame, found {len(candidates)}"
    return candidates[0]


def _parse_legend(g: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[tuple[tuple[int, int], ...], int]:
    r0, c0, r1, c1 = frame_bbox
    sub = g[r0 + 1 : r1, c0 + 1 : c1]
    entries = _components(sub != BG)
    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    seen_target_colors: set[int] = set()

    for comp in entries:
        colors = {int(sub[r, c]) for r, c in comp}
        assert GREY in colors, "legend entry must contain a grey template"
        target_colors = colors - {GREY}
        assert len(target_colors) == 1, "legend entry must contain exactly one target color"
        target = target_colors.pop()
        assert target not in seen_target_colors, "legend target colors must be unique"
        seen_target_colors.add(target)

        grey_cells = [(r, c) for r, c in comp if int(sub[r, c]) == GREY]
        target_cells = [(r, c) for r, c in comp if int(sub[r, c]) == target]
        assert len(target_cells) == 1, "legend color marker must be a singleton"
        assert 3 <= len(grey_cells) <= 5, "template size must be in [3, 5]"

        key = _canonical(grey_cells)
        assert key not in mapping, "legend shapes must be unique up to rotation/flip"
        mapping[key] = target

    assert mapping, "legend must contain at least one entry"
    return mapping


def _parse_objects_and_markers(
    g: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    outside = (g != BG).copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    objects: list[dict] = []
    markers: dict[int, tuple[int, int]] = {}
    for comp in _components(outside):
        colors = {int(g[r, c]) for r, c in comp}
        assert GREY not in colors, "only the legend may use grey"
        assert len(colors) == 1, "outside components must be monochrome"
        color = next(iter(colors))
        if len(comp) == 1:
            assert color not in markers, "marker colors must be unique"
            markers[color] = comp[0]
            continue

        assert 3 <= len(comp) <= 5, "object size must be in [3, 5]"
        br0, bc0, _, _ = _bbox(comp)
        rel = tuple(sorted((r - br0, c - bc0) for r, c in comp))
        objects.append(
            {
                "cells": comp,
                "rel": rel,
                "shape_key": _canonical(rel),
            }
        )

    assert objects, "expected at least one movable object"
    assert markers, "expected at least one color marker"
    return objects, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each right-side object to a grey legend template by shape, recolor it,
    and move it so its top-left aligns with the matching target-color marker.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox = _find_legend_frame(g)
    legend_map = _parse_legend(g, frame_bbox)
    objects, markers = _parse_objects_and_markers(g, frame_bbox)

    out = g.copy()
    for obj in objects:
        for r, c in obj["cells"]:
            out[r, c] = BG
    for r, c in markers.values():
        out[r, c] = BG

    occupied: set[tuple[int, int]] = set()
    for obj in objects:
        assert obj["shape_key"] in legend_map, "object shape missing from legend"
        target_color = legend_map[obj["shape_key"]]
        assert target_color in markers, "missing destination marker for target color"
        top, left = markers[target_color]
        for dr, dc in obj["rel"]:
            rr = top + dr
            cc = left + dc
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "translated object went out of bounds"
            assert (rr, cc) not in occupied, "translated objects must not overlap"
            occupied.add((rr, cc))
            out[rr, cc] = target_color

    return out


def _all_anchor_orientations(name: str) -> list[tuple[tuple[tuple[int, int], ...], bool, int]]:
    return [rec for rec in _dihedral_orientations(SHAPES[name]) if (0, 0) in rec[0]]


def _make_entry_patch(shape_coords: tuple[tuple[int, int], ...], target_color: int) -> np.ndarray:
    h, w = _shape_size(shape_coords)
    patch = np.zeros((h, w + 1), dtype=int)
    for r, c in shape_coords:
        patch[r, c] = GREY
    edge_rows = sorted({r for r, c in shape_coords if c == w - 1})
    patch[random.choice(edge_rows), w] = target_color
    return patch


def _make_legend_patch(cases: list[dict]) -> np.ndarray:
    entry_patches = [_make_entry_patch(case["legend_shape"], case["target_color"]) for case in cases]
    max_w = max(p.shape[1] for p in entry_patches)
    total_h = 4 + sum(p.shape[0] for p in entry_patches) + (len(entry_patches) - 1)
    total_w = 4 + max_w

    legend = np.zeros((total_h, total_w), dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    legend[:, 0] = GREY
    legend[:, -1] = GREY

    row = 2
    for patch in entry_patches:
        ph, pw = patch.shape
        legend[row : row + ph, 2 : 2 + pw] = patch
        row += ph + 1
    return legend


def _can_reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int, min_col: int) -> bool:
    H, W = reserved.shape
    if top < 1 or left < min_col or top + h > H - 1 or left + w > W - 1:
        return False
    return not reserved[top - 1 : top + h + 1, left - 1 : left + w + 1].any()


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    reserved[top - 1 : top + h + 1, left - 1 : left + w + 1] = True


def _place_boxes(
    boxes: list[tuple[str, int, int, int]],
    H: int,
    W: int,
    min_col: int,
) -> dict[tuple[str, int], tuple[int, int]] | None:
    reserved = np.zeros((H, W), dtype=bool)
    order = boxes[:]
    random.shuffle(order)
    order.sort(key=lambda item: (item[2] * item[3], item[2] + item[3]), reverse=True)

    positions: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, h, w in order:
        if min_col > W - w - 1:
            return None
        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(1, H - h - 1)
            left = random.randint(min_col, W - w - 1)
            if not _can_reserve(reserved, top, left, h, w, min_col):
                continue
            _reserve(reserved, top, left, h, w)
            positions[(kind, idx)] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return positions


def _rotation_capable_shapes() -> list[str]:
    out = []
    for name in SHAPES:
        non_flipped = [rec for rec in _all_anchor_orientations(name) if not rec[1]]
        if len(non_flipped) >= 2:
            out.append(name)
    return out


ROTATE_SHAPES = tuple(_rotation_capable_shapes())
assert len({_canonical(shape) for shape in SHAPES.values()}) == len(SHAPES)


def _gen_attempt() -> np.ndarray | None:
    n_entries = random.choice((3, 4))
    relation_mode = random.choice(("rotate", "flip"))
    if relation_mode == "flip":
        special_shape = random.choice(CHIRAL_SHAPES)
    else:
        special_shape = random.choice(ROTATE_SHAPES)

    other_shapes = [name for name in SHAPES if name != special_shape]
    chosen_shapes = [special_shape] + random.sample(other_shapes, n_entries - 1)
    random.shuffle(chosen_shapes)
    special_index = chosen_shapes.index(special_shape)

    colors = list(PALETTE)
    random.shuffle(colors)
    target_colors = colors[:n_entries]
    object_colors = colors[n_entries : 2 * n_entries]

    cases: list[dict] = []
    for idx, shape_name in enumerate(chosen_shapes):
        all_oris = _dihedral_orientations(SHAPES[shape_name])
        anchor_oris = _all_anchor_orientations(shape_name)
        assert anchor_oris

        if idx == special_index and relation_mode == "flip":
            legend_choices = [rec for rec in anchor_oris if not rec[1]]
            object_choices = [rec for rec in anchor_oris if rec[1]]
            if not legend_choices or not object_choices:
                return None
            legend_rec = random.choice(legend_choices)
            object_rec = random.choice(object_choices)
        elif idx == special_index and relation_mode == "rotate":
            rotate_choices = [rec for rec in anchor_oris if not rec[1]]
            if len(rotate_choices) < 2:
                return None
            legend_rec, object_rec = random.sample(rotate_choices, 2)
        else:
            legend_rec = random.choice(all_oris)
            object_rec = random.choice(anchor_oris)

        cases.append(
            {
                "shape_name": shape_name,
                "legend_shape": legend_rec[0],
                "object_shape": object_rec[0],
                "target_color": target_colors[idx],
                "object_color": object_colors[idx],
            }
        )

    legend = _make_legend_patch(cases)
    lh, lw = legend.shape
    H_min = max(lh + 2, 20)
    W_min = max(lw + 15, 24)
    H_max = max(H_min, 28)
    W_max = max(W_min, 32)
    if H_min > H_max or W_min > W_max:
        return None

    H = random.randint(H_min, H_max)
    W = random.randint(W_min, W_max)
    legend_top = random.randint(1, H - lh - 1)
    legend_left = random.randint(1, 2)
    right_min = legend_left + lw + 2
    if right_min >= W - 3:
        return None

    boxes: list[tuple[str, int, int, int]] = []
    for idx, case in enumerate(cases):
        h, w = _shape_size(case["object_shape"])
        boxes.append(("object", idx, h, w))
        boxes.append(("dest", idx, h, w))
    positions = _place_boxes(boxes, H, W, right_min)
    if positions is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    grid[legend_top : legend_top + lh, legend_left : legend_left + lw] = legend

    for idx, case in enumerate(cases):
        obj_top, obj_left = positions[("object", idx)]
        for dr, dc in case["object_shape"]:
            grid[obj_top + dr, obj_left + dc] = case["object_color"]

        dst_top, dst_left = positions[("dest", idx)]
        grid[dst_top, dst_left] = case["target_color"]

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "generated puzzle must induce a change"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except AssertionError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
