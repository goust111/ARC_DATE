import random
from collections import Counter, deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay each carrier object's left-right mirror residuals to reconstruct the hidden pattern.

    Input structure:
    - A black background contains several separated single-color carrier objects.
    - Every object has a bounding box of the same size and is almost left-right symmetric
      inside that box, except for a few extra foreground cells whose mirrored positions are black.
    - Different objects use different colors, and their residual cells occupy different local
      coordinates inside the shared bounding-box frame.

    Rule:
    - Find every monochrome object and crop it to its own bounding box.
    - Compare the crop with its left-right mirror. Any cell that is foreground while its mirrored
      partner is background is that object's residual cell.
    - Create a black canvas with the shared bounding-box size and paint each object's residual
      cells back at the same local coordinates using that object's original color.

    Key insight:
    - The asymmetric cells are not noise to erase. They are colored fragments hidden inside
      otherwise symmetric carriers, and only the overlay of all local residuals reveals the answer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    objects = _find_components(g, bg=bg)
    assert len(objects) >= 2

    box_shapes = {(r1 - r0 + 1, c1 - c0 + 1) for _color, (r0, c0, r1, c1), _coords in objects}
    assert len(box_shapes) == 1
    h, w = next(iter(box_shapes))

    out = np.full((h, w), bg, dtype=int)
    for color, bbox, _coords in objects:
        crop = _crop(g, bbox)
        mask = crop == color
        residual = mask & (~np.fliplr(mask))
        assert residual.any()
        assert not np.any(out[residual] != bg)
        out[residual] = color
    return out


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, _bbox_from_coords(coords), coords))
    return comps


def _mirror_col(width: int, col: int) -> int:
    return width - 1 - col


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _carrier_base(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if w % 2 == 1:
        center = w // 2
        mask[:, center] = True
        core_cols = [center]
    else:
        left = w // 2 - 1
        right = w // 2
        mask[:, left] = True
        mask[:, right] = True
        core_cols = [left, right]

    edge_row = rng.randrange(h)
    mask[edge_row, :] = True

    span_cap = max(1, w // 2 - 1)
    row_count = rng.randint(2, min(h, 5))
    rows = rng.sample(range(h), row_count)
    for r in rows:
        span = rng.randint(1, span_cap)
        if w % 2 == 1:
            c0 = max(0, core_cols[0] - span)
            c1 = min(w - 1, core_cols[0] + span)
        else:
            c0 = max(0, core_cols[0] - span)
            c1 = min(w - 1, core_cols[-1] + span)
        mask[r, c0 : c1 + 1] = True

    pair_count = rng.randint(1, max(2, h // 2))
    for _ in range(pair_count):
        r = rng.randrange(h)
        c = rng.randrange(w)
        mc = _mirror_col(w, c)
        if c == mc:
            continue
        mask[r, c] = True
        mask[r, mc] = True

    return mask


def _admissible_residual_cells(base: np.ndarray) -> list[tuple[int, int]]:
    h, w = base.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            mc = _mirror_col(w, c)
            if c == mc or base[r, c] or base[r, mc]:
                continue
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if _in_bounds(base.shape, nr, nc) and base[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _choose_scattered_subset(
    rng: random.Random,
    cells: list[tuple[int, int]],
    count: int,
    used: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    choices = [cell for cell in cells if cell not in used]
    rng.shuffle(choices)
    if len(choices) < count:
        return None

    picked: list[tuple[int, int]] = []
    for cell in choices:
        if all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in picked):
            picked.append(cell)
            if len(picked) == count:
                return picked

    if len(choices) >= count:
        return choices[:count]
    return None


def _make_object_mask(
    rng: random.Random,
    h: int,
    w: int,
    residual_cells: list[tuple[int, int]],
) -> np.ndarray | None:
    for _ in range(120):
        base = _carrier_base(rng, h, w)
        if any(base[r, c] or base[r, _mirror_col(w, c)] for r, c in residual_cells):
            continue
        if not all(any(_in_bounds(base.shape, r + dr, c + dc) and base[r + dr, c + dc] for dr, dc in DIRS4) for r, c in residual_cells):
            continue
        obj = base.copy()
        for r, c in residual_cells:
            obj[r, c] = True
        if len(_component_coords(obj)) != 1:
            continue
        if np.array_equal(obj, np.fliplr(obj)):
            continue
        residual = obj & (~np.fliplr(obj))
        if set(map(tuple, np.argwhere(residual))) != set(residual_cells):
            continue
        if np.argwhere(obj).min(axis=0).tolist() != [0, 0]:
            continue
        if np.argwhere(obj).max(axis=0).tolist() != [h - 1, w - 1]:
            continue
        if int(obj.sum()) < len(residual_cells) + 7:
            continue
        if int(obj.sum()) > (h * w * 3) // 4:
            continue
        return obj
    return None


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _place_bboxes(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    count: int,
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for _ in range(count):
        candidates: list[tuple[int, int]] = []
        for r0 in range(grid_h - box_h + 1):
            for c0 in range(grid_w - box_w + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(grid_h, r0 + box_h + 1)
                cc1 = min(grid_w, c0 + box_w + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = candidates[rng.randrange(len(candidates))]
        placements.append(pos)
        r0, c0 = pos
        occupied[r0 : r0 + box_h, c0 : c0 + box_w] = True
    rng.shuffle(placements)
    return placements


def _naive_global_overlay(grid: np.ndarray, objects: list[tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]], bg: int) -> np.ndarray:
    out = np.full_like(grid, bg)
    for color, bbox, _coords in objects:
        r0, c0, _r1, _c1 = bbox
        crop = _crop(grid, bbox)
        mask = crop == color
        residual = mask & (~np.fliplr(mask))
        for rr, cc in np.argwhere(residual):
            out[r0 + rr, c0 + cc] = color
    coords = np.argwhere(out != bg)
    if coords.size == 0:
        return out
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(5, 9)
    w = rng.randint(5, 10)
    object_count = rng.randint(3, 5)
    total_residual = rng.randint(max(5, object_count + 1), min(object_count * 3, max(6, (h * w) // 3)))

    counts = [1] * object_count
    for _ in range(total_residual - object_count):
        expandable = [idx for idx, val in enumerate(counts) if val < 3]
        counts[rng.choice(expandable)] += 1
    rng.shuffle(counts)

    local_cells = [(r, c) for r in range(h) for c in range(w) if c != _mirror_col(w, c)]
    used: set[tuple[int, int]] = set()
    object_masks: list[np.ndarray] = []
    residual_sets: list[list[tuple[int, int]]] = []

    for count in counts:
        for _ in range(160):
            probe_base = _carrier_base(rng, h, w)
            admissible = _admissible_residual_cells(probe_base)
            admissible = [cell for cell in admissible if cell in local_cells and cell not in used]
            if len(admissible) < count:
                continue
            chosen = _choose_scattered_subset(rng, admissible, count, used)
            if chosen is None:
                continue
            obj_mask = _make_object_mask(rng, h, w, chosen)
            if obj_mask is None:
                continue
            used.update(chosen)
            residual_sets.append(chosen)
            object_masks.append(obj_mask)
            break
        else:
            return None

    output = np.full((h, w), BG, dtype=int)
    colors = rng.sample(list(range(1, 10)), object_count)
    for color, cells in zip(colors, residual_sets):
        for r, c in cells:
            output[r, c] = color

    nonzero = np.argwhere(output != BG)
    rows = {int(r) for r, _c in nonzero}
    cols = {int(c) for _r, c in nonzero}
    left = any(c < _mirror_col(w, c) for _r, c in nonzero)
    right = any(c > _mirror_col(w, c) for _r, c in nonzero)
    if len(rows) < 2 or len(cols) < 3 or not left or not right:
        return None

    grid_h = rng.randint(max(h + 6, 16), 30)
    grid_w = rng.randint(max(w + 6, 16), 30)
    placements = _place_bboxes(rng, grid_h, grid_w, h, w, object_count)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for color, obj_mask, (r0, c0) in zip(colors, object_masks, placements):
        patch = np.where(obj_mask, color, BG)
        grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = color

    objects = _find_components(grid, bg=BG)
    if len(objects) != object_count:
        return None
    if {(r1 - r0 + 1, c1 - c0 + 1) for _color, (r0, c0, r1, c1), _coords in objects} != {(h, w)}:
        return None

    solved = main(grid)
    if not np.array_equal(solved, output):
        return None

    for color, bbox, _coords in objects:
        crop = _crop(grid, bbox)
        mask = crop == color
        residual = mask & (~np.fliplr(mask))
        if not residual.any():
            return None

    naive = _naive_global_overlay(grid, objects, BG)
    if np.array_equal(naive, output):
        return None

    if int(np.count_nonzero(output)) <= max(counts):
        return None

    if len({tuple(mask.astype(int).ravel()) for mask in object_masks}) < 2:
        return None

    return grid, output


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
