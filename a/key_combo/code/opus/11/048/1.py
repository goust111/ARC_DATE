from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4


@dataclass(frozen=True)
class BBox:
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


@dataclass
class Structure:
    bbox: BBox
    orient: str
    axis: int
    source_sign: int
    cavities: list[tuple[np.ndarray, BBox]]


@dataclass
class RedComponent:
    coords: np.ndarray
    bbox: BBox


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return BBox(int(r0), int(c0), int(r1), int(c1))


def components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def detect_structures(grid: np.ndarray) -> list[Structure]:
    structs = []
    for coords in components(np.isin(grid, [BLUE, YELLOW])):
        bbox = bbox_coords(coords)
        ys = coords[grid[coords[:, 0], coords[:, 1]] == YELLOW]
        if ys.shape[0] < 3:
            continue

        row_counts = {}
        col_counts = {}
        for r, c in ys:
            row_counts[int(r)] = row_counts.get(int(r), 0) + 1
            col_counts[int(c)] = col_counts.get(int(c), 0) + 1
        best_row, best_row_count = max(row_counts.items(), key=lambda x: x[1])
        best_col, best_col_count = max(col_counts.items(), key=lambda x: x[1])
        if best_row_count >= best_col_count:
            orient = "h"
            axis = int(best_row)
            off_axis = ys[ys[:, 0] != axis]
            if off_axis.shape[0] != 1:
                raise ValueError("expected exactly one horizontal control marker")
            control = off_axis[0]
            source_sign = -1 if int(control[0]) < axis else 1
        else:
            orient = "v"
            axis = int(best_col)
            off_axis = ys[ys[:, 1] != axis]
            if off_axis.shape[0] != 1:
                raise ValueError("expected exactly one vertical control marker")
            control = off_axis[0]
            source_sign = -1 if int(control[1]) < axis else 1

        cavities = []
        view = grid[bbox.r0 : bbox.r1 + 1, bbox.c0 : bbox.c1 + 1]
        for local in components(view == BLACK):
            if np.any(local[:, 0] == 0) or np.any(local[:, 0] == view.shape[0] - 1):
                continue
            if np.any(local[:, 1] == 0) or np.any(local[:, 1] == view.shape[1] - 1):
                continue
            global_coords = local + np.array([[bbox.r0, bbox.c0]])
            cavities.append((global_coords, bbox_coords(global_coords)))
        cavities.sort(key=lambda item: (item[1].r0, item[1].c0, item[1].r1, item[1].c1))
        structs.append(Structure(bbox=bbox, orient=orient, axis=axis, source_sign=source_sign, cavities=cavities))
    structs.sort(key=lambda s: (s.bbox.r0, s.bbox.c0))
    return structs


def detect_red_components(grid: np.ndarray) -> list[RedComponent]:
    reds = []
    for coords in components(grid == RED):
        reds.append(RedComponent(coords=coords, bbox=bbox_coords(coords)))
    reds.sort(key=lambda r: (r.bbox.r0, r.bbox.c0))
    return reds


def red_candidates(red: RedComponent, structures: list[Structure]) -> list[tuple[int, int, tuple[int, int]]]:
    candidates = []
    for sid, struct in enumerate(structures):
        bbox = struct.bbox
        if struct.orient == "v":
            move = (0, -struct.source_sign)
            if struct.source_sign < 0 and red.bbox.c1 >= bbox.c0:
                continue
            if struct.source_sign > 0 and red.bbox.c0 <= bbox.c1:
                continue
        else:
            move = (-struct.source_sign, 0)
            if struct.source_sign < 0 and red.bbox.r1 >= bbox.r0:
                continue
            if struct.source_sign > 0 and red.bbox.r0 <= bbox.r1:
                continue

        for cid, (cavity_coords, cavity_bbox) in enumerate(struct.cavities):
            if struct.orient == "v":
                if not interval_overlap(red.bbox.r0, red.bbox.r1, cavity_bbox.r0, cavity_bbox.r1):
                    continue
                shift_c = cavity_bbox.c1 - red.bbox.c1 if move[1] > 0 else cavity_bbox.c0 - red.bbox.c0
                if shift_c * move[1] <= 0:
                    continue
                shifted = red.coords + np.array([[0, shift_c]])
            else:
                if not interval_overlap(red.bbox.c0, red.bbox.c1, cavity_bbox.c0, cavity_bbox.c1):
                    continue
                shift_r = cavity_bbox.r1 - red.bbox.r1 if move[0] > 0 else cavity_bbox.r0 - red.bbox.r0
                if shift_r * move[0] <= 0:
                    continue
                shifted = red.coords + np.array([[shift_r, 0]])

            cavity_set = {tuple(map(int, xy)) for xy in cavity_coords}
            if all(tuple(map(int, xy)) in cavity_set for xy in shifted):
                candidates.append((sid, cid, (int(shifted[0, 0] - red.coords[0, 0]), int(shifted[0, 1] - red.coords[0, 1]))))
    return candidates


def compute_assignments(grid: np.ndarray, structures: list[Structure] | None = None):
    structures = detect_structures(grid) if structures is None else structures
    reds = detect_red_components(grid)
    assignments = []
    matched_reds = set()
    used_slots = set()
    for rid, red in enumerate(reds):
        candidates = red_candidates(red, structures)
        if len(candidates) == 1:
            sid, cid, shift = candidates[0]
            if (sid, cid) in used_slots:
                continue
            assignments.append((rid, sid, cid, shift))
            matched_reds.add(rid)
            used_slots.add((sid, cid))
    return reds, assignments, matched_reds


def transform(grid: np.ndarray, *, do_dock: bool = True, do_mirror: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    structures = detect_structures(g)
    reds, assignments, _ = compute_assignments(g, structures)

    out = g.copy()
    out[out == RED] = BLACK

    if do_dock:
        for rid, _sid, _cid, shift in assignments:
            red = reds[rid]
            shifted = red.coords + np.array([[shift[0], shift[1]]])
            out[shifted[:, 0], shifted[:, 1]] = RED

    if do_mirror:
        for struct in structures:
            b = struct.bbox
            if struct.orient == "v":
                if struct.source_sign < 0:
                    target_cols = range(struct.axis + 1, b.c1 + 1)
                else:
                    target_cols = range(b.c0, struct.axis)
                for r in range(b.r0, b.r1 + 1):
                    for c in target_cols:
                        src_c = 2 * struct.axis - c
                        src = int(out[r, src_c])
                        if src != YELLOW and out[r, c] != YELLOW:
                            out[r, c] = src
            else:
                if struct.source_sign < 0:
                    target_rows = range(struct.axis + 1, b.r1 + 1)
                else:
                    target_rows = range(b.r0, struct.axis)
                for r in target_rows:
                    src_r = 2 * struct.axis - r
                    for c in range(b.c0, b.c1 + 1):
                        src = int(out[src_r, c])
                        if src != YELLOW and out[r, c] != YELLOW:
                            out[r, c] = src
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock source-side red components into source-side cavities, then mirror the trusted half across the yellow axis.

    Input structure:
    - The background is black.
    - Each large structure is made of blue pixels plus one yellow straight axis and one adjacent off-axis yellow control pixel.
    - Each structure's black rectangular cavities lie on the control-marked source side, while small red components sit outside the structures.

    Rule:
    - For each structure, read the yellow line as a horizontal or vertical symmetry axis and read the single off-axis yellow pixel as the source side.
    - A red component qualifies only if it lies outside that structure on the source side and, by moving perpendicular to the axis toward the center, can end with its leading edge flush against one cavity wall while remaining fully inside that cavity.
    - Qualified red components are translated into their matched cavities; unqualified red components disappear.
    - After docking, copy the source half of each structure's bounding box onto the opposite half by reflection across the axis, mirroring blue, red, and black cells while leaving yellow pixels unchanged.

    Key insight:
    - The same yellow axis controls both the docking direction and the final reflection, while the lone off-axis yellow pixel identifies which half is trustworthy enough to drive both steps.
    """
    return transform(grid, do_dock=True, do_mirror=True)


VERTICAL_ONE = ((1, 2), (2, 2), (3, 2))
HORIZONTAL_ONE = ((2, 1), (2, 2), (2, 3))
DISTRACTOR_SHAPES = (
    np.array([(0, 0), (0, 1)], dtype=int),
    np.array([(0, 0), (1, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2), (1, 1)], dtype=int),
)


def connected_target_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    # Each row is a contiguous prefix from the axis-adjacent side, so the
    # target half stays connected and cannot create accidental enclosed holes.
    widths = np.ones(h, dtype=int)
    for i in range(h):
        widths[i] = int(rng.integers(1, w + 1))
    if np.all(widths == 1):
        widths[int(rng.integers(h))] = int(rng.integers(2, w + 1))
    if np.all(widths == w):
        widths[int(rng.integers(h))] = int(rng.integers(1, w))

    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        mask[r, : widths[r]] = True
    return mask


def build_vertical_structure(rng: np.random.Generator) -> np.ndarray:
    for _ in range(80):
        cavity_count = 1 if rng.random() < 0.6 else 2
        h = 5 if cavity_count == 1 else 7
        w = 9
        g = np.zeros((h, w), dtype=int)
        g[:, :4] = BLUE
        if cavity_count == 1:
            ch, cw = VERTICAL_ONE[int(rng.integers(len(VERTICAL_ONE)))]
            top = 1 if ch == 3 else int(rng.integers(1, 3))
            g[top : top + ch, 1 : 1 + cw] = BLACK
        else:
            g[1:3, 1:3] = BLACK
            g[4:6, 1:3] = BLACK
        g[:, 4] = YELLOW
        control_row = int(rng.integers(h))
        g[control_row, 3] = YELLOW
        target = connected_target_mask(h, 4, rng)
        g[:, 5:9][target] = BLUE
        if rng.random() < 0.5:
            g = np.fliplr(g)
        structs = detect_structures(g)
        if len(structs) != 1:
            continue
        if len(structs[0].cavities) != cavity_count:
            continue
        return g
    raise RuntimeError("failed to build vertical structure")


def build_horizontal_structure(rng: np.random.Generator) -> np.ndarray:
    for _ in range(80):
        cavity_count = 1 if rng.random() < 0.6 else 2
        h = 9
        w = 5 if cavity_count == 1 else 7
        g = np.zeros((h, w), dtype=int)
        g[:4, :] = BLUE
        if cavity_count == 1:
            ch, cw = HORIZONTAL_ONE[int(rng.integers(len(HORIZONTAL_ONE)))]
            left = 1 if cw == 3 else int(rng.integers(1, 3))
            g[1 : 1 + ch, left : left + cw] = BLACK
        else:
            g[1:3, 1:3] = BLACK
            g[1:3, 4:6] = BLACK
        g[4, :] = YELLOW
        control_col = int(rng.integers(w))
        g[3, control_col] = YELLOW
        target = connected_target_mask(w, 4, rng).T
        g[5:9, :][target] = BLUE
        if rng.random() < 0.5:
            g = np.flipud(g)
        structs = detect_structures(g)
        if len(structs) != 1:
            continue
        if len(structs[0].cavities) != cavity_count:
            continue
        return g
    raise RuntimeError("failed to build horizontal structure")


def rotate_shape(shape: np.ndarray, k: int) -> np.ndarray:
    pts = shape.copy()
    for _ in range(k % 4):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    pts -= pts.min(axis=0)
    return pts


def choose_matched_shape(cavity_bbox: BBox, move: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    rows = np.arange(cavity_bbox.r0, cavity_bbox.r1 + 1)
    cols = np.arange(cavity_bbox.c0, cavity_bbox.c1 + 1)
    if move[1] != 0:
        width = 1 if cavity_bbox.width > 1 and rng.random() < 0.5 else cavity_bbox.width
        use_cols = cols[-width:] if move[1] > 0 else cols[:width]
        coords = np.array([(r, c) for r in rows for c in use_cols], dtype=int)
    else:
        height = 1 if cavity_bbox.height > 1 and rng.random() < 0.5 else cavity_bbox.height
        use_rows = rows[-height:] if move[0] > 0 else rows[:height]
        coords = np.array([(r, c) for r in use_rows for c in cols], dtype=int)
    return coords


def paste_shape(grid: np.ndarray, coords: np.ndarray, color: int, *, separate_red: bool = False) -> bool:
    h, w = grid.shape
    if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= h) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= w):
        return False
    if np.any(grid[coords[:, 0], coords[:, 1]] != BLACK):
        return False
    if separate_red and color == RED:
        coord_set = {tuple(map(int, xy)) for xy in coords}
        for r, c in coord_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coord_set and grid[nr, nc] == RED:
                    return False
    grid[coords[:, 0], coords[:, 1]] = color
    return True


def local_to_global(local_coords: np.ndarray, top: int, left: int) -> np.ndarray:
    return local_coords + np.array([[top, left]])


def place_structures(canvas: np.ndarray, locals_: list[np.ndarray], rng: np.random.Generator):
    placements = []
    h, w = canvas.shape
    occupied = np.zeros_like(canvas, dtype=bool)
    for local in locals_:
        sh, sw = local.shape
        success = False
        for _ in range(200):
            top = int(rng.integers(0, h - sh + 1))
            left = int(rng.integers(0, w - sw + 1))
            r0 = max(0, top - 2)
            c0 = max(0, left - 2)
            r1 = min(h, top + sh + 2)
            c1 = min(w, left + sw + 2)
            if occupied[r0:r1, c0:c1].any():
                continue
            patch = local != BLACK
            canvas[top : top + sh, left : left + sw][patch] = local[patch]
            occupied[top : top + sh, left : left + sw] |= patch
            placements.append((top, left, local))
            success = True
            break
        if not success:
            return None
    return placements


def structure_move(struct: Structure) -> tuple[int, int]:
    if struct.orient == "v":
        return (0, -struct.source_sign)
    return (-struct.source_sign, 0)


def place_matched_reds(canvas: np.ndarray, matched_sids: set[int], rng: np.random.Generator) -> bool:
    structures = detect_structures(canvas)
    for sid in sorted(matched_sids):
        struct = structures[sid]
        move = structure_move(struct)
        cid = int(rng.integers(len(struct.cavities)))
        cavity_coords, cavity_bbox = struct.cavities[cid]
        final_coords = choose_matched_shape(cavity_bbox, move, rng)
        distance = int(rng.integers(4, 8))
        initial = final_coords - np.array([[move[0] * distance, move[1] * distance]])
        if not paste_shape(canvas, initial, RED, separate_red=True):
            return False
    return True


def place_distractors(canvas: np.ndarray, count: int, rng: np.random.Generator) -> bool:
    h, w = canvas.shape
    for _ in range(count):
        placed = False
        for _ in range(200):
            shape = DISTRACTOR_SHAPES[int(rng.integers(len(DISTRACTOR_SHAPES)))]
            shape = rotate_shape(shape, int(rng.integers(4)))
            max_r = h - int(shape[:, 0].max()) - 1
            max_c = w - int(shape[:, 1].max()) - 1
            top = int(rng.integers(0, max_r + 1))
            left = int(rng.integers(0, max_c + 1))
            coords = shape + np.array([[top, left]])
            if paste_shape(canvas, coords, RED, separate_red=True):
                placed = True
                break
        if not placed:
            return False
    return True


def mirror_changes(grid: np.ndarray, output: np.ndarray, struct: Structure) -> bool:
    b = struct.bbox
    if struct.orient == "v":
        if struct.source_sign < 0:
            region_in = grid[b.r0 : b.r1 + 1, struct.axis + 1 : b.c1 + 1]
            region_out = output[b.r0 : b.r1 + 1, struct.axis + 1 : b.c1 + 1]
        else:
            region_in = grid[b.r0 : b.r1 + 1, b.c0 : struct.axis]
            region_out = output[b.r0 : b.r1 + 1, b.c0 : struct.axis]
    else:
        if struct.source_sign < 0:
            region_in = grid[struct.axis + 1 : b.r1 + 1, b.c0 : b.c1 + 1]
            region_out = output[struct.axis + 1 : b.r1 + 1, b.c0 : b.c1 + 1]
        else:
            region_in = grid[b.r0 : struct.axis, b.c0 : b.c1 + 1]
            region_out = output[b.r0 : struct.axis, b.c0 : b.c1 + 1]
    return not np.array_equal(region_in, region_out)


def validate_input(grid: np.ndarray, expected_structures: int, expected_matches: int) -> bool:
    structures = detect_structures(grid)
    if len(structures) != expected_structures:
        return False
    for struct in structures:
        if len(struct.cavities) not in (1, 2):
            return False
        for _, cavity_bbox in struct.cavities:
            if struct.orient == "v":
                if struct.source_sign < 0 and cavity_bbox.c1 >= struct.axis:
                    return False
                if struct.source_sign > 0 and cavity_bbox.c0 <= struct.axis:
                    return False
            else:
                if struct.source_sign < 0 and cavity_bbox.r1 >= struct.axis:
                    return False
                if struct.source_sign > 0 and cavity_bbox.r0 <= struct.axis:
                    return False
    reds, assignments, matched = compute_assignments(grid, structures)
    if not (3 <= len(reds) <= 5):
        return False
    if len(assignments) != expected_matches:
        return False
    for red in reds:
        if len(red_candidates(red, structures)) > 1:
            return False
    final = main(grid)
    mirror_only = transform(grid, do_dock=False, do_mirror=True)
    dock_only = transform(grid, do_dock=True, do_mirror=False)
    if np.array_equal(final, mirror_only):
        return False
    if np.array_equal(final, dock_only):
        return False
    if any(not mirror_changes(grid, final, struct) for struct in structures):
        return False
    final_reds = detect_red_components(final)
    if len(final_reds) < expected_matches:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        h = int(rng.integers(20, 27))
        w = int(rng.integers(20, 27))
        n_struct = 2 if rng.random() < 0.55 else 3
        locals_ = []
        for _ in range(n_struct):
            if rng.random() < 0.5:
                locals_.append(build_vertical_structure(rng))
            else:
                locals_.append(build_horizontal_structure(rng))
        order = np.argsort([-local.size for local in locals_])
        locals_ = [locals_[int(i)] for i in order]
        canvas = np.zeros((h, w), dtype=int)
        placements = place_structures(canvas, locals_, rng)
        if placements is None:
            continue

        parsed = detect_structures(canvas)
        n_match = n_struct if rng.random() < 0.65 else n_struct - 1
        matched_sids = set(int(x) for x in rng.choice(len(parsed), size=n_match, replace=False))
        if not place_matched_reds(canvas, matched_sids, rng):
            continue

        total_red = int(rng.integers(max(3, n_match + 1), 6))
        distractors = total_red - n_match
        if distractors < 1:
            distractors = 1
        if not place_distractors(canvas, distractors, rng):
            continue

        if validate_input(canvas, expected_structures=n_struct, expected_matches=n_match):
            return canvas
    raise RuntimeError("failed to generate valid input")
