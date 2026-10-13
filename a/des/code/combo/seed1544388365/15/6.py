from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
ORTHO4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARC_COLORS = tuple(range(1, 10))

# --------------- solver ---------------


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ORTHO4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append(np.array(pts, dtype=int))

    return comps


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each almost point-symmetric colored region, flip defect-color
    markers through the region center, and restore their original cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    out = g.copy()
    for coords in _components(g != BLACK):
        values = g[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        main_color = int(colors[int(np.argmax(counts))])

        main_coords = coords[values == main_color]
        assert len(main_coords) > 0
        center_r = _round_half_up(float(main_coords[:, 0].mean()))
        center_c = _round_half_up(float(main_coords[:, 1].mean()))

        defect_coords = coords[values != main_color]
        for r, c in defect_coords:
            defect_color = int(g[r, c])
            out[r, c] = main_color

            rr = 2 * center_r - int(r)
            cc = 2 * center_c - int(c)
            if 0 <= rr < height and 0 <= cc < width and int(g[rr, cc]) == BLACK:
                out[rr, cc] = defect_color

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_SIZE_CHOICES = (5, 7, 9, 11)
PAIR_COUNT_RANGE = (4, 12)
PLACEMENT_BORDER = 1
OBJECT_ATTEMPTS = 120
DEFECT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _grow_symmetric_mask(height: int, width: int, pair_count: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    cr, cc = height // 2, width // 2
    mask[cr, cc] = True

    for _ in range(pair_count):
        candidates: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            r = int(r)
            c = int(c)
            for dr, dc in ORTHO4:
                nr, nc = r + dr, c + dc
                rr, rc = height - 1 - nr, width - 1 - nc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if not (0 <= rr < height and 0 <= rc < width):
                    continue
                if mask[nr, nc] or mask[rr, rc]:
                    continue
                candidates.add((nr, nc))

        if not candidates:
            return None

        nr, nc = random.choice(tuple(candidates))
        rr, rc = height - 1 - nr, width - 1 - nc
        mask[nr, nc] = True
        mask[rr, rc] = True

    used_rows = np.flatnonzero(mask.any(axis=1))
    used_cols = np.flatnonzero(mask.any(axis=0))
    if len(used_rows) < 3 or len(used_cols) < 3:
        return None
    return mask


def _is_boundary_cell(mask: np.ndarray, r: int, c: int) -> bool:
    height, width = mask.shape
    for nr, nc in _neighbors4(r, c, height, width):
        if not mask[nr, nc]:
            return True
    return r in (0, height - 1) or c in (0, width - 1)


def _boundary_pairs(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    height, width = mask.shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        rr, rc = height - 1 - r, width - 1 - c
        if (r, c) >= (rr, rc):
            continue
        if not _is_boundary_cell(mask, r, c) or not _is_boundary_cell(mask, rr, rc):
            continue
        pairs.append(((r, c), (rr, rc)))
    return pairs


def _choose_defects(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    pairs = _boundary_pairs(mask)
    area = int(mask.sum())
    max_defects = min(3, len(pairs), (area - 1) // 3)
    if max_defects < 1:
        return None

    for _ in range(DEFECT_ATTEMPTS):
        defect_count = random.randint(1, max_defects)
        chosen = random.sample(pairs, defect_count)

        source_cells: list[tuple[int, int]] = []
        target_cells: list[tuple[int, int]] = []
        occupied = mask.copy()
        main_mask = mask.copy()

        for a, b in chosen:
            source, target = (a, b) if random.random() < 0.5 else (b, a)
            source_cells.append(source)
            target_cells.append(target)
            occupied[target] = False
            main_mask[source] = False
            main_mask[target] = False

        if not _is_connected(occupied):
            continue
        if not _is_connected(main_mask):
            continue
        return source_cells, target_cells

    return None


def _crop_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _build_object(main_color: int, defect_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(OBJECT_ATTEMPTS):
        height = random.choice(OBJECT_SIZE_CHOICES)
        width = random.choice(OBJECT_SIZE_CHOICES)
        max_pairs = min(PAIR_COUNT_RANGE[1], (height * width - 1) // 2)
        if max_pairs < PAIR_COUNT_RANGE[0]:
            continue
        pair_count = random.randint(PAIR_COUNT_RANGE[0], max_pairs)

        mask = _grow_symmetric_mask(height, width, pair_count)
        if mask is None:
            continue

        defects = _choose_defects(mask)
        if defects is None:
            continue
        source_cells, target_cells = defects

        r0, r1, c0, c1 = _crop_to_bbox(mask)
        cropped_mask = mask[r0:r1, c0:c1]
        inp = np.zeros(cropped_mask.shape, dtype=int)
        out = np.zeros(cropped_mask.shape, dtype=int)
        inp[cropped_mask] = main_color
        out[cropped_mask] = main_color

        for r, c in source_cells:
            inp[r - r0, c - c0] = defect_color
        for r, c in target_cells:
            inp[r - r0, c - c0] = BLACK
            out[r - r0, c - c0] = defect_color

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp, out

    return None


def _can_place(reserved: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    total_h, total_w = reserved.shape
    rb0 = max(0, r0 - PLACEMENT_BORDER)
    cb0 = max(0, c0 - PLACEMENT_BORDER)
    rb1 = min(total_h, r0 + height + PLACEMENT_BORDER)
    cb1 = min(total_w, c0 + width + PLACEMENT_BORDER)
    return not reserved[rb0:rb1, cb0:cb1].any()


def _stamp(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    height, width = patch.shape
    view = dst[r0 : r0 + height, c0 : c0 + width]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)

    defect_colors = random.sample(ARC_COLORS, object_count)
    main_choices = [color for color in ARC_COLORS if color not in defect_colors]
    main_pool_size = random.randint(1, min(3, len(main_choices)))
    main_pool = random.sample(main_choices, main_pool_size)

    objects: list[tuple[np.ndarray, np.ndarray]] = []
    for defect_color in defect_colors:
        obj = _build_object(random.choice(main_pool), defect_color)
        if obj is None:
            return None
        objects.append(obj)

    objects.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for inp_patch, out_patch in objects:
        ph, pw = inp_patch.shape
        if ph > height or pw > width:
            return None

        positions = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            if not _can_place(reserved, r0, c0, ph, pw):
                continue
            _stamp(inp, inp_patch, r0, c0)
            _stamp(out, out_patch, r0, c0)
            reserved[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break

        if not placed:
            return None

    try:
        solved = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
