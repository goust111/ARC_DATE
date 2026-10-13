from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
SIDE_CHOICES = (5, 7, 9)
GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (1, 4)
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 400
PATCH_ATTEMPTS = 120
FEATURE_COUNT_RANGE = (2, 6)


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _mismatch_pair(patch: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = patch.shape
    if h != w or h % 2 == 0:
        return None

    size = h
    mismatches: list[tuple[int, int]] = []
    for r in range(size):
        for c in range(size):
            rr = size - 1 - r
            cc = size - 1 - c
            if int(patch[r, c]) != int(patch[rr, cc]):
                mismatches.append((r, c))

    if len(mismatches) != 2:
        return None

    p = mismatches[0]
    q = (size - 1 - p[0], size - 1 - p[1])
    if p == q or set(mismatches) != {p, q}:
        return None
    return p, q


def _touches_all_sides(patch: np.ndarray) -> bool:
    mask = patch != BG
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _has_bg_ring(grid: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = grid.shape
    r1 = r0 + size - 1
    c1 = c0 + size - 1

    if r0 > 0 and bool((grid[r0 - 1, c0 : c1 + 1] != BG).any()):
        return False
    if r1 + 1 < h and bool((grid[r1 + 1, c0 : c1 + 1] != BG).any()):
        return False
    if c0 > 0 and bool((grid[r0 : r1 + 1, c0 - 1] != BG).any()):
        return False
    if c1 + 1 < w and bool((grid[r0 : r1 + 1, c1 + 1] != BG).any()):
        return False
    return True


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_object_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    boxes: list[tuple[int, int, int, int]] = []

    for size in range(3, min(h, w) + 1, 2):
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                patch = g[r0 : r0 + size, c0 : c0 + size]
                if not bool((patch != BG).any()):
                    continue
                if not _touches_all_sides(patch):
                    continue
                if not _has_bg_ring(g, r0, c0, size):
                    continue
                if _mismatch_pair(patch) is None:
                    continue
                boxes.append((r0, c0, r0 + size - 1, c0 + size - 1))

    boxes.sort()
    assert boxes, "expected at least one defective square motif"

    for i, box in enumerate(boxes):
        for other in boxes[i + 1 :]:
            assert not _boxes_overlap(box, other), "motif boxes must be disjoint"

    covered = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        covered[r0 : r1 + 1, c0 : c1 + 1] |= g[r0 : r1 + 1, c0 : c1 + 1] != BG
    assert np.array_equal(covered, g != BG), "every foreground pixel must belong to exactly one motif box"
    return boxes


def _majority_info(grid: np.ndarray, r: int, c: int) -> tuple[bool | None, int]:
    neighbors = [int(grid[nr, nc]) for nr, nc in _neighbors4(grid.shape, r, c)]
    counts = Counter(neighbors)
    support = counts.get(int(grid[r, c]), 0)
    if not counts:
        return None, support
    best = max(counts.values())
    majors = [color for color, count in counts.items() if count == best]
    if len(majors) != 1:
        return None, support
    return bool(int(grid[r, c]) == int(majors[0])), support


def _choose_defect(
    grid: np.ndarray, p: tuple[int, int], q: tuple[int, int]
) -> tuple[int, int]:
    p_match, p_support = _majority_info(grid, *p)
    q_match, q_support = _majority_info(grid, *q)

    if p_match is False and q_match is True:
        return p
    if q_match is False and p_match is True:
        return q

    if p_support < q_support:
        return p
    if q_support < p_support:
        return q

    p_fix_support = sum(int(grid[nr, nc]) == int(grid[q]) for nr, nc in _neighbors4(grid.shape, *p))
    q_fix_support = sum(int(grid[nr, nc]) == int(grid[p]) for nr, nc in _neighbors4(grid.shape, *q))
    if p_fix_support > q_fix_support:
        return p
    if q_fix_support > p_fix_support:
        return q

    return p if p <= q else q


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each square motif by replacing its single tampered pixel with the color required for 180-degree rotational symmetry.

    Input structure:
    - The background is black.
    - Several disjoint colored motifs appear on the canvas, each occupying a tight odd-sided square bounding box.
    - Inside each box, the motif is almost 180-degree rotationally symmetric around its center, except for one non-center pixel whose color was changed.

    Rule:
    - For every motif box, compare the patch with its 180-degree rotation about the box center.
    - This reveals one mismatching pixel pair P and P' related by the rotation.
    - Use 4-neighborhood consistency to decide which member of the pair is the defect: the defect is the one whose current color disagrees with the local majority more strongly.
    - Recolor only that defect pixel with the color seen at its rotational counterpart, and leave all other pixels unchanged.

    Key insight:
    - The hidden structure is near-perfect point symmetry inside each odd square, while the local neighborhood identifies which side of the unique mismatching pair is the corrupted one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r0, c0, r1, c1 in _find_object_boxes(g):
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        pair = _mismatch_pair(patch)
        assert pair is not None, "each motif must have exactly one mismatching rotation pair"
        (pr, pc), (qr, qc) = pair

        p = (r0 + pr, c0 + pc)
        q = (r0 + qr, c0 + qc)
        defect = _choose_defect(g, p, q)
        source = q if defect == p else p
        out[defect] = int(g[source])

    return out


def _paint_rot_rect(patch: np.ndarray, color: int) -> None:
    size = patch.shape[0]
    height = random.randint(1, max(1, size // 2))
    width = random.randint(1, max(1, size // 2))
    r0 = random.randint(0, size - height)
    c0 = random.randint(0, size - width)
    r1 = size - height - r0
    c1 = size - width - c0

    patch[r0 : r0 + height, c0 : c0 + width] = color
    patch[r1 : r1 + height, c1 : c1 + width] = color


def _defect_pairs(patch: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    size = patch.shape[0]
    center = size // 2
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if r == center and c == center:
                continue
            q = (size - 1 - r, size - 1 - c)
            if (r, c) > q:
                continue
            color = int(patch[r, c])
            if int(patch[q]) != color:
                continue
            if all(int(patch[nr, nc]) == color for nr, nc in _neighbors4(patch.shape, r, c)) and all(
                int(patch[nr, nc]) == color for nr, nc in _neighbors4(patch.shape, *q)
            ):
                pairs.append(((r, c), q))
    return pairs


def _build_solved_patch(size: int) -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        palette = random.sample(ALL_COLORS, random.randint(2, min(4, len(ALL_COLORS))))
        patch = np.full((size, size), palette[0], dtype=int)
        for _ in range(random.randint(*FEATURE_COUNT_RANGE)):
            _paint_rot_rect(patch, random.choice(palette))
        if np.unique(patch).size < 2:
            continue
        if _defect_pairs(patch):
            return patch
    return None


def _make_object(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    solved = _build_solved_patch(size)
    if solved is None:
        return None

    pairs = _defect_pairs(solved)
    if not pairs:
        return None

    inp = solved.copy()
    defect, _mate = random.choice(pairs)
    wrong_choices = [color for color in ALL_COLORS if color != int(solved[defect])]
    inp[defect] = random.choice(wrong_choices)

    if _mismatch_pair(inp) is None:
        return None
    return inp, solved


def _can_place(used: np.ndarray, r0: int, c0: int, size: int, pad: int = PLACEMENT_PADDING) -> bool:
    h, w = used.shape
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_used(used: np.ndarray, r0: int, c0: int, size: int, pad: int = PLACEMENT_PADDING) -> None:
    h, w = used.shape
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid_in = np.full((h, w), BG, dtype=int)
    grid_out = np.full((h, w), BG, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    sizes = [random.choice(SIDE_CHOICES) for _ in range(object_count)]
    sizes.sort(reverse=True)

    placed = 0
    for size in sizes:
        obj = _make_object(size)
        if obj is None:
            return None
        inp_patch, out_patch = obj

        positions = [(r, c) for r in range(1, h - size) for c in range(1, w - size)]
        random.shuffle(positions)

        for r0, c0 in positions:
            if not _can_place(used, r0, c0, size):
                continue
            grid_in[r0 : r0 + size, c0 : c0 + size] = inp_patch
            grid_out[r0 : r0 + size, c0 : c0 + size] = out_patch
            _mark_used(used, r0, c0, size)
            placed += 1
            break

    if placed == 0 or placed != object_count:
        return None

    if not np.array_equal(main(grid_in), grid_out):
        return None
    return grid_in


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-defect puzzle")
