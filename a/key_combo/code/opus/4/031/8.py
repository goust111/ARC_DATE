from __future__ import annotations

import random
from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_task_031_8",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4


LEFT_COLOR = 1
RIGHT_COLOR = 3
BG = 0
ZERO_HOLE_COLOR = 2
ONE_HOLE_COLOR = 4
MANY_HOLE_COLOR = 7
PANEL_SIZE = 7
GRID_SHAPE = (PANEL_SIZE, PANEL_SIZE * 2)
GEN_ATTEMPTS = 400


def _transform(mask: np.ndarray, rot: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, k=int(rot))
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def _embed(mask: np.ndarray, top: int, left: int, shape: tuple[int, int] = (PANEL_SIZE, PANEL_SIZE)) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    out[top : top + h, left : left + w] = mask
    return out


def _foreground_components(mask: np.ndarray) -> list[np.ndarray]:
    return components_mask(mask.astype(bool), connectivity=4)


def _label_background(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = mask.shape
    bg = ~mask
    labels = np.zeros((h, w), dtype=int)
    infos: list[dict] = []
    label_id = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or labels[r, c] != 0:
                continue
            label_id += 1
            q = deque([(r, c)])
            labels[r, c] = label_id
            coords = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and bg[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label_id
                        q.append((nr, nc))
            infos.append(
                {
                    "id": label_id,
                    "coords": np.array(coords, dtype=int),
                    "touches_edge": touches_edge,
                }
            )
    return labels, infos


def _hole_counts(mask: np.ndarray) -> dict[int, int]:
    fg_comps = _foreground_components(mask)
    comp_labels = np.zeros(mask.shape, dtype=int)
    for comp_id, coords in enumerate(fg_comps, start=1):
        comp_labels[coords[:, 0], coords[:, 1]] = comp_id

    bg_labels, bg_infos = _label_background(mask)
    counts = {comp_id: 0 for comp_id in range(1, len(fg_comps) + 1)}
    for info in bg_infos:
        if info["touches_edge"]:
            continue
        neighbor_ids: set[int] = set()
        for r, c in info["coords"]:
            for nr, nc in neighbors4(int(r), int(c)):
                if in_bounds(mask.shape, nr, nc) and comp_labels[nr, nc] != 0:
                    neighbor_ids.add(int(comp_labels[nr, nc]))
        if len(neighbor_ids) == 1:
            counts[next(iter(neighbor_ids))] += 1
    return counts


def _recolor_from_holes(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    counts = _hole_counts(mask)
    for comp_id, coords in enumerate(_foreground_components(mask), start=1):
        hole_count = counts.get(comp_id, 0)
        color = ZERO_HOLE_COLOR if hole_count == 0 else ONE_HOLE_COLOR if hole_count == 1 else MANY_HOLE_COLOR
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two side-by-side panels, count emergent holes in each union component, and recolor by hole count.

    Input structure:
    - The input is a 7x14 grid made of two 7x7 panels placed side by side with no separator.
    - The left panel contains blue foreground shapes on black, and the right panel contains green foreground shapes on black.
    - Shapes can overlap by coordinate across the two panels when the panels are transparently overlaid.

    Rule:
    - Split the grid into its left and right 7x7 panels.
    - Form the 7x7 boolean union: a cell is foreground iff either panel has a non-black pixel there.
    - Find every 4-connected foreground component in that union and count how many enclosed 4-connected black holes it has.
    - Recolor each whole component by its hole count: 0 holes -> red, 1 hole -> yellow, 2 or more holes -> orange.

    Key insight:
    - The deciding property is not any single panel's shape, but the hole topology that appears only after the transparent overlay is unioned.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected shape {GRID_SHAPE}, got {g.shape}"
    left = g[:, :PANEL_SIZE]
    right = g[:, PANEL_SIZE:]
    union_mask = (left != BG) | (right != BG)
    return _recolor_from_holes(union_mask)


def _one_hole_motif() -> tuple[np.ndarray, np.ndarray]:
    left = np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    )
    right = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    return left, right


def _two_hole_motif() -> tuple[np.ndarray, np.ndarray]:
    left = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    right = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    return left, right


_ZERO_HOLE_LIBRARY = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]


def _random_zero_hole_shape() -> np.ndarray:
    shape = random.choice(_ZERO_HOLE_LIBRARY)
    return _transform(shape, rot=random.randrange(4), flip_lr=bool(random.randrange(2)))


def _variants(left: np.ndarray, right: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    seen: set[bytes] = set()
    for rot in range(4):
        for flip_lr in (False, True):
            l2 = _transform(left, rot, flip_lr)
            r2 = _transform(right, rot, flip_lr)
            key = l2.tobytes() + b"|" + r2.tobytes() + bytes(l2.shape) + bytes(r2.shape)
            if key not in seen:
                seen.add(key)
                out.append((l2, r2))
    random.shuffle(out)
    return out


def _can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    for r, c in _mask_to_coords(patch):
        rr = top + int(r)
        cc = left + int(c)
        if occupied[rr, cc]:
            return False
        for nr, nc in neighbors4(rr, cc):
            if in_bounds(occupied.shape, nr, nc) and occupied[nr, nc]:
                return False
    return True


def _place_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = patch.shape
    region = dst[top : top + h, left : left + w]
    region[patch] = color


def _try_place_motifs(motifs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    left_panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    right_panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    occupied = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)

    motifs = sorted(motifs, key=lambda pair: int((pair[0] | pair[1]).sum()), reverse=True)
    for base_left, base_right in motifs:
        placed = False
        for left_mask, right_mask in _variants(base_left, base_right):
            union = left_mask | right_mask
            h, w = union.shape
            anchors = [(r, c) for r in range(PANEL_SIZE - h + 1) for c in range(PANEL_SIZE - w + 1)]
            random.shuffle(anchors)
            for top, left in anchors:
                if not _can_place(occupied, union, top, left):
                    continue
                if left_mask.any():
                    _place_patch(left_panel, left_mask, top, left, LEFT_COLOR)
                if right_mask.any():
                    _place_patch(right_panel, right_mask, top, left, RIGHT_COLOR)
                occupied |= _embed(union, top, left)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return left_panel, right_panel


def _component_sizes(panel: np.ndarray, color: int) -> list[int]:
    return [int(coords.shape[0]) for coords in _foreground_components(panel == color)]


def _panel_hole_counts(panel: np.ndarray) -> list[int]:
    mask = panel != BG
    counts = _hole_counts(mask)
    return [counts[i] for i in range(1, len(_foreground_components(mask)) + 1)]


def _has_cross_panel_contact(comp_mask: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray) -> bool:
    overlap = comp_mask & left_mask & right_mask
    if overlap.any():
        return True
    left_cells = np.argwhere(comp_mask & left_mask)
    right_cells = np.argwhere(comp_mask & right_mask)
    right_set = {(int(r), int(c)) for r, c in right_cells}
    for r, c in left_cells:
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) in right_set:
                return True
    return False


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.shape != GRID_SHAPE:
        return False
    if not np.all(np.isin(grid, [BG, LEFT_COLOR, RIGHT_COLOR])):
        return False

    left = grid[:, :PANEL_SIZE]
    right = grid[:, PANEL_SIZE:]
    if not np.all(np.isin(left, [BG, LEFT_COLOR])) or not np.all(np.isin(right, [BG, RIGHT_COLOR])):
        return False

    left_sizes = _component_sizes(left, LEFT_COLOR)
    right_sizes = _component_sizes(right, RIGHT_COLOR)
    if not (2 <= len(left_sizes) <= 3 and 2 <= len(right_sizes) <= 3):
        return False
    if any(size < 3 or size > 7 for size in left_sizes + right_sizes):
        return False
    if any(count != 0 for count in _panel_hole_counts(left) + _panel_hole_counts(right)):
        return False

    union_mask = (left != BG) | (right != BG)
    union_components = _foreground_components(union_mask)
    counts = _hole_counts(union_mask)
    hole_values = [counts[i] for i in range(1, len(union_components) + 1)]
    if 0 not in hole_values or 1 not in hole_values:
        return False

    emergent = False
    left_mask = left != BG
    right_mask = right != BG
    for comp_id, coords in enumerate(union_components, start=1):
        if counts[comp_id] <= 0:
            continue
        comp_mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        if (comp_mask & left_mask).any() and (comp_mask & right_mask).any() and _has_cross_panel_contact(comp_mask, left_mask, right_mask):
            emergent = True
            break
    if not emergent:
        return False

    out = main(grid)
    if out.shape != (PANEL_SIZE, PANEL_SIZE):
        return False
    out_colors = set(map(int, out.ravel())) - {BG}
    if not ({ZERO_HOLE_COLOR, ONE_HOLE_COLOR} <= out_colors):
        return False
    return True


def _build_scene() -> np.ndarray | None:
    motifs: list[tuple[np.ndarray, np.ndarray]] = [_one_hole_motif()]
    include_two_holes = random.random() < 0.55
    if include_two_holes:
        motifs.append(_two_hole_motif())
        extra = _random_zero_hole_shape()
        if random.random() < 0.5:
            motifs.append((extra, np.zeros_like(extra, dtype=bool)))
        else:
            motifs.append((np.zeros_like(extra, dtype=bool), extra))
    else:
        left_extra = _random_zero_hole_shape()
        right_extra = _random_zero_hole_shape()
        motifs.append((left_extra, np.zeros_like(left_extra, dtype=bool)))
        motifs.append((np.zeros_like(right_extra, dtype=bool), right_extra))

    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    for left_mask, right_mask in motifs:
        left_mask = np.asarray(left_mask, dtype=bool)
        right_mask = np.asarray(right_mask, dtype=bool)
        union = left_mask | right_mask
        coords = _mask_to_coords(union)
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        normalized.append((left_mask[r0 : r1 + 1, c0 : c1 + 1], right_mask[r0 : r1 + 1, c0 : c1 + 1]))

    placed = _try_place_motifs(normalized)
    if placed is None:
        return None
    left_panel, right_panel = placed
    grid = np.hstack([left_panel, right_panel])
    return grid if _validate_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlay-hole ARC input")
