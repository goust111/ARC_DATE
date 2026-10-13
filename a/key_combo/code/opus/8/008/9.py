from __future__ import annotations

import random

import numpy as np


BLUE = 1
BLACK = 0
ORANGE = 7
FOREGROUND = (BLUE, ORANGE)
MAX_GEN_ATTEMPTS = 400


def _count_blue_neighbors4(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    count = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BLUE:
            count += 1
    return count


def _count_black_neighbors8(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BLACK:
                count += 1
    return count


def _erase_spurs(grid: np.ndarray) -> np.ndarray:
    base = np.asarray(grid, dtype=int).copy()
    out = base.copy()
    erase: list[tuple[int, int]] = []
    h, w = base.shape
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLUE:
                continue
            if _count_blue_neighbors4(base, r, c) <= 1 and _count_black_neighbors8(base, r, c) >= 5:
                erase.append((r, c))
    for r, c in erase:
        out[r, c] = BLACK
    return out


def _find_defects(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    defects: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != BLACK:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) not in FOREGROUND:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                defects.append((r, c))
    return defects


def _mark_defects(grid: np.ndarray) -> np.ndarray:
    base = np.asarray(grid, dtype=int)
    out = base.copy()
    defects = _find_defects(base)
    defect_set = set(defects)
    h, w = base.shape

    for r, c in defects:
        ring: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    ring.append((rr, cc))
                    if int(out[rr, cc]) in FOREGROUND:
                        out[rr, cc] = ORANGE

        # Extend the orange glow one more step into the surrounding background.
        for rr, cc in ring:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < h and 0 <= c2 < w and (r2, c2) not in defect_set and int(out[r2, c2]) == BLACK:
                        out[r2, c2] = ORANGE

    for r, c in defects:
        out[r, c] = BLACK
    return out


def _has_southeast_support(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if r + 1 >= h or c + 1 >= w:
        return False
    return bool(np.any(np.isin(grid[r + 1 :, c + 1 :], FOREGROUND)))


def _slide_foreground(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            if int(out[r, c]) not in FOREGROUND:
                continue
            if c + 1 >= w:
                continue
            if _has_southeast_support(out, r, c):
                out[r, c + 1] = out[r, c]
                out[r, c] = BLACK
    return out


def _slide_with_labels(grid: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    lab = np.asarray(labels, dtype=int).copy()
    h, w = out.shape
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            if int(out[r, c]) not in FOREGROUND:
                continue
            if c + 1 >= w:
                continue
            if _has_southeast_support(out, r, c):
                out[r, c + 1] = out[r, c]
                out[r, c] = BLACK
                lab[r, c + 1] = lab[r, c]
                lab[r, c] = 0
    return lab


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase thin blue spurs, surround enclosed black defects with orange, then slide the modified foreground rightward.

    Input structure:
    - Black (0) background with one or two blue (1) 4-connected foreground regions.
    - Regions may contain one-pixel black holes fully enclosed by blue, plus occasional thin blue spur pixels.
    - No other colors are required in the input; orange is introduced by the transformation.

    Rule:
    - First erase every blue spur whose blue 4-neighbor count is at most one and whose black 8-neighbor count is at least five.
    - Then detect black defect pixels whose in-bounds 8-neighbors are all foreground, recolor their surrounding blue ring orange, and extend that orange halo one step into adjacent black background while keeping defect pixels black.
    - Finally treat blue and orange as foreground and, scanning from bottom-right to top-left, move each foreground pixel one cell right iff some foreground pixel currently exists strictly southeast of it.

    Key insight:
    - The orange defect halo is not cosmetic: it changes the foreground support geometry that controls which pixels slide.
    """
    g = np.asarray(grid, dtype=int)
    erased = _erase_spurs(g)
    marked = _mark_defects(erased)
    return _slide_foreground(marked)


def _crop_nonzero(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _count_components(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            size = 0
            while q:
                rr, cc = q.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < h and 0 <= c2 < w and mask[r2, c2] and not seen[r2, c2]:
                        seen[r2, c2] = True
                        q.append((r2, c2))
            sizes.append(size)
    return sizes


def _choose_holes(height: int, width: int, hole_count: int) -> list[tuple[int, int]] | None:
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    preferred = [(r, c) for r, c in interior if r == height - 2 or c == width - 2]
    if not preferred:
        return None
    first = random.choice(preferred)
    holes = [first]
    if hole_count == 2:
        others = [
            (r, c)
            for r, c in interior
            if max(abs(r - first[0]), abs(c - first[1])) >= 2
        ]
        if not others:
            return None
        holes.append(random.choice(others))
    return holes


def _add_spurs(patch: np.ndarray, height: int, width: int) -> None:
    spur_count = random.choices([0, 1, 2], weights=[2, 4, 3], k=1)[0]
    options: list[tuple[int, int]] = []
    for c in range(2, width):
        options.append((0, c))
        options.append((height + 1, c))
    for r in range(2, height):
        options.append((r, 0))
        options.append((r, width + 1))
    random.shuffle(options)
    used_sides: set[str] = set()
    added = 0
    for r, c in options:
        side = "top" if r == 0 else "bottom" if r == height + 1 else "left" if c == 0 else "right"
        if side in used_sides:
            continue
        rr = 1 if r == 0 else height if r == height + 1 else r
        cc = 1 if c == 0 else width if c == width + 1 else c
        if int(patch[rr, cc]) != BLUE:
            continue
        patch[r, c] = BLUE
        used_sides.add(side)
        added += 1
        if added >= spur_count:
            break


def _build_region_patch() -> np.ndarray | None:
    for _ in range(80):
        height = random.randint(4, 7)
        width = random.randint(4, 7)
        hole_count = random.choices([1, 2], weights=[3, 2], k=1)[0]
        holes = _choose_holes(height, width, hole_count)
        if holes is None:
            continue

        patch = np.zeros((height + 2, width + 2), dtype=int)
        patch[1 : height + 1, 1 : width + 1] = BLUE
        for r, c in holes:
            patch[r + 1, c + 1] = BLACK
        _add_spurs(patch, height, width)
        patch = _crop_nonzero(patch)

        blue_mask = patch == BLUE
        area = int(np.sum(blue_mask))
        if not (15 <= area <= 40):
            continue
        if _count_components(blue_mask) != [area]:
            continue
        if len(_find_defects(patch)) != hole_count:
            continue
        return patch
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h - 4 or pw > w - 4:
        return False
    rows = list(range(2, h - ph - 1))
    cols = list(range(2, w - pw - 1))
    random.shuffle(rows)
    random.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            separated = True
            for br0, bc0, br1, bc1 in boxes:
                if not (box[2] + 2 < br0 or br1 + 2 < box[0] or box[3] + 2 < bc0 or bc1 + 2 < box[1]):
                    separated = False
                    break
            if not separated:
                continue
            if np.any(grid[r0 : r0 + ph, c0 : c0 + pw] != BLACK):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(grid[r0 : r0 + ph, c0 : c0 + pw], patch)
            boxes.append(box)
            return True
    return False


def _original_label_grid(grid: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(grid, dtype=int)
    idx = 1
    for r, c in np.argwhere(grid == BLUE):
        labels[int(r), int(c)] = idx
        idx += 1
    return labels


def _marking_changes_original_motion(grid: np.ndarray) -> bool:
    erased = _erase_spurs(grid)
    marked = _mark_defects(erased)
    if np.array_equal(erased, marked):
        return False
    labels = _original_label_grid(erased)
    base_labels = _slide_with_labels(erased, labels)
    marked_labels = _slide_with_labels(marked, labels)
    return not np.array_equal(base_labels, marked_labels)


def _component_sizes(grid: np.ndarray) -> list[int]:
    return sorted(_count_components(grid == BLUE))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    grid = np.zeros((h, w), dtype=int)
    object_count = random.choice([1, 2])
    boxes: list[tuple[int, int, int, int]] = []

    patches: list[np.ndarray] = []
    for _ in range(object_count):
        patch = _build_region_patch()
        if patch is None:
            return None
        patches.append(patch)

    for patch in sorted(patches, key=lambda x: x.shape[0] * x.shape[1], reverse=True):
        if not _place_patch(grid, patch, boxes):
            return None

    sizes = _component_sizes(grid)
    if len(sizes) != object_count or any(size < 15 or size > 40 for size in sizes):
        return None

    defect_count = len(_find_defects(grid))
    if defect_count < object_count or defect_count > 2 * object_count:
        return None

    if not _marking_changes_original_motion(grid):
        return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
