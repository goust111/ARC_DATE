import random
from typing import List, Sequence, Tuple

import numpy as np


BLACK = 0
GREY = 5
PATTERN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
INVERSE_TRANSFORM = [0, 3, 2, 1, 4, 5, 6, 7]
LAYOUT_CHOICES = [(1, 4), (1, 5), (2, 2), (2, 3), (3, 2)]
MAX_GENERATION_ATTEMPTS = 500


def _apply_transform(arr: np.ndarray, t: int) -> np.ndarray:
    t = int(t)
    if t < 4:
        return np.rot90(arr, k=t)
    return np.rot90(np.fliplr(arr), k=t - 4)


def _inverse_transform(arr: np.ndarray, t: int) -> np.ndarray:
    return _apply_transform(arr, INVERSE_TRANSFORM[int(t)])


def _segments(length: int, dividers: Sequence[int]) -> List[Tuple[int, int]]:
    cuts = [-1] + sorted(int(x) for x in dividers) + [int(length)]
    spans = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        if a < b:
            spans.append((int(a), int(b)))
    return spans


def _extract_slots(grid: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[np.ndarray]]:
    row_divs = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_divs = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_spans = _segments(grid.shape[0], row_divs)
    col_spans = _segments(grid.shape[1], col_divs)
    assert row_spans and col_spans

    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1
    size = next(iter(heights))
    assert size == next(iter(widths))

    slots = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            slots.append(grid[r0:r1, c0:c1].copy())
    return row_spans, col_spans, slots


def _compatible(a: np.ndarray, b: np.ndarray) -> bool:
    overlap = (a != BLACK) & (b != BLACK)
    return bool(np.array_equal(a[overlap], b[overlap]))


def _merge_partials(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    if not _compatible(a, b):
        return None
    out = a.copy()
    fill = (out == BLACK) & (b != BLACK)
    out[fill] = b[fill]
    return out


def _solve_slots(slots: Sequence[np.ndarray], max_solutions: int = 2) -> List[Tuple[Tuple[int, ...], np.ndarray]]:
    partials = [[_inverse_transform(slot, t) for t in range(8)] for slot in slots]
    n = len(slots)
    assignments: List[int | None] = [None] * n
    assignments[0] = 0
    solutions: List[Tuple[Tuple[int, ...], np.ndarray]] = []

    def rec(current: np.ndarray) -> None:
        if len(solutions) >= max_solutions:
            return
        if all(a is not None for a in assignments):
            solutions.append((tuple(int(x) for x in assignments), current.copy()))
            return

        best_idx = None
        best_options = None
        for i in range(1, n):
            if assignments[i] is not None:
                continue
            options = []
            for t in range(8):
                merged = _merge_partials(current, partials[i][t])
                if merged is not None:
                    options.append((t, merged))
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_idx = i
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_idx is not None and best_options is not None
        for t, merged in best_options:
            assignments[best_idx] = int(t)
            rec(merged)
            assignments[best_idx] = None

    rec(partials[0][0])
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each scrambled slot orientation, rebuild the complete master pattern, and refill every black occlusion.

    Input structure:
    - Gray divider lines split the canvas into same-sized square slots, arranged in one or two dimensions.
    - Every slot contains the same fully colored square pattern under one of the eight dihedral orientations.
    - Black cells are missing pixels inside the slots; the pattern itself never uses black, and divider lines stay gray.

    Rule:
    - Enumerate slot orientations with the first slot as the reference orientation.
    - For each candidate assignment, inverse-transform the visible pixels into the reference frame and reject any color conflict.
    - Use the unique conflict-free assignment to merge all visible pixels into the complete master pattern, then reapply each slot's orientation and fill its black cells.

    Key insight:
    - No slot is complete and no orientation labels are given, so the orientations and the recovered pattern must be inferred together by cross-slot consistency.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, slots = _extract_slots(g)
    solutions = _solve_slots(slots, max_solutions=2)
    assert len(solutions) == 1, "expected a unique consistent orientation assignment"
    assignments, mother = solutions[0]
    assert np.all(mother != BLACK), "every master-pattern pixel should be visible in at least one slot"

    out = g.copy()
    idx = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            out[r0:r1, c0:c1] = _apply_transform(mother, assignments[idx])
            idx += 1
    return out


def _assemble_grid(slot_rows: int, slot_cols: int, slot_size: int, slot_patches: Sequence[np.ndarray]) -> np.ndarray:
    h = slot_rows * slot_size + (slot_rows - 1)
    w = slot_cols * slot_size + (slot_cols - 1)
    grid = np.full((h, w), GREY, dtype=int)
    idx = 0
    for rr in range(slot_rows):
        for cc in range(slot_cols):
            r0 = rr * (slot_size + 1)
            c0 = cc * (slot_size + 1)
            grid[r0 : r0 + slot_size, c0 : c0 + slot_size] = slot_patches[idx]
            idx += 1
    return grid


def _all_transforms_unique(arr: np.ndarray) -> bool:
    return len({_apply_transform(arr, t).tobytes() for t in range(8)}) == 8


def _random_partition_pattern(size: int) -> np.ndarray:
    for _ in range(200):
        n_colors = random.randint(3, min(5, len(PATTERN_COLORS)))
        palette = random.sample(PATTERN_COLORS, n_colors)
        rects = [(0, 0, size, size)]
        target_rects = random.randint(4, min(7, size + 2))
        while len(rects) < target_rects:
            candidates = []
            for idx, (r0, c0, r1, c1) in enumerate(rects):
                h = r1 - r0
                w = c1 - c0
                if h >= 2:
                    candidates.append((idx, "h"))
                if w >= 2:
                    candidates.append((idx, "v"))
            if not candidates:
                break
            idx, axis = random.choice(candidates)
            r0, c0, r1, c1 = rects.pop(idx)
            if axis == "h":
                cut = random.randint(r0 + 1, r1 - 1)
                rects.extend([(r0, c0, cut, c1), (cut, c0, r1, c1)])
            else:
                cut = random.randint(c0 + 1, c1 - 1)
                rects.extend([(r0, c0, r1, cut), (r0, cut, r1, c1)])

        pattern = np.zeros((size, size), dtype=int)
        for rect in rects:
            color = random.choice(palette)
            r0, c0, r1, c1 = rect
            pattern[r0:r1, c0:c1] = color

        for _ in range(random.randint(1, 3)):
            color = random.choice(palette)
            kind = random.choice(["block", "bar", "corner"])
            if kind == "block":
                h = random.randint(1, max(1, size // 2))
                w = random.randint(1, max(1, size // 2))
                r0 = random.randint(0, size - h)
                c0 = random.randint(0, size - w)
                pattern[r0 : r0 + h, c0 : c0 + w] = color
            elif kind == "bar":
                if random.choice([True, False]):
                    r = random.randint(0, size - 1)
                    c0 = random.randint(0, size - 2)
                    c1 = random.randint(c0 + 1, size - 1)
                    pattern[r, c0 : c1 + 1] = color
                else:
                    c = random.randint(0, size - 1)
                    r0 = random.randint(0, size - 2)
                    r1 = random.randint(r0 + 1, size - 1)
                    pattern[r0 : r1 + 1, c] = color
            else:
                arm = random.randint(2, size)
                r0 = random.randint(0, size - arm)
                c0 = random.randint(0, size - arm)
                pattern[r0 : r0 + arm, c0] = color
                pattern[r0, c0 : c0 + arm] = color

        counts = [int(np.sum(pattern == color)) for color in np.unique(pattern)]
        corner_colors = {int(pattern[0, 0]), int(pattern[0, -1]), int(pattern[-1, 0]), int(pattern[-1, -1])}
        if len(np.unique(pattern)) < 3:
            continue
        if min(counts) < 2:
            continue
        if max(counts) > int(0.72 * size * size):
            continue
        if len(corner_colors) < 3:
            continue
        if not _all_transforms_unique(pattern):
            continue
        return pattern
    raise RuntimeError("failed to generate asymmetric master pattern")


def _random_blob(size: int) -> np.ndarray:
    blob = np.zeros((size, size), dtype=bool)
    kind = random.choice(["rect", "rect", "lshape"])
    h = random.randint(1, max(1, size - 1))
    w = random.randint(1, max(1, size - 1))
    r0 = random.randint(0, size - h)
    c0 = random.randint(0, size - w)
    if kind == "rect" or h == 1 or w == 1:
        blob[r0 : r0 + h, c0 : c0 + w] = True
        return blob

    arm_h = random.randint(1, h)
    arm_w = random.randint(1, w)
    if random.choice([True, False]):
        blob[r0 : r0 + h, c0 : c0 + arm_w] = True
        blob[r0 : r0 + arm_h, c0 : c0 + w] = True
    else:
        blob[r0 : r0 + arm_h, c0 : c0 + w] = True
        blob[r0 : r0 + h, c0 + w - arm_w : c0 + w] = True
    return blob


def _random_visible_mask(size: int) -> np.ndarray:
    total = size * size
    min_visible = int(np.ceil(total * 0.45))
    max_visible = int(np.floor(total * 0.78))
    for _ in range(200):
        visible = np.ones((size, size), dtype=bool)
        for _ in range(random.randint(2, 4)):
            visible[_random_blob(size)] = False
        visible_count = int(np.sum(visible))
        if min_visible <= visible_count <= max_visible:
            return visible
    raise RuntimeError("failed to generate a non-degenerate missing mask")


def _choose_layout() -> Tuple[int, int, int]:
    slot_rows, slot_cols = random.choice(LAYOUT_CHOICES)
    max_size = min((30 - (slot_rows - 1)) // slot_rows, (30 - (slot_cols - 1)) // slot_cols)
    slot_size = random.randint(5, min(7, max_size))
    return int(slot_rows), int(slot_cols), int(slot_size)


def _choose_orientations(n_slots: int) -> List[int]:
    while True:
        orientations = [random.randrange(8) for _ in range(n_slots)]
        if len(set(orientations)) < 3:
            continue
        if not any(t < 4 for t in orientations):
            continue
        if not any(t >= 4 for t in orientations):
            continue
        return orientations


def _coverage_masks_for_mother(orientations: Sequence[int], visible_masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [_inverse_transform(mask.astype(int), t).astype(bool) for mask, t in zip(visible_masks, orientations)]


def _forward_coord(size: int, r: int, c: int, t: int) -> Tuple[int, int]:
    marker = np.zeros((size, size), dtype=int)
    marker[int(r), int(c)] = 1
    rr, cc = np.argwhere(_apply_transform(marker, t) == 1)[0]
    return int(rr), int(cc)


def _has_anchor_ambiguity(slots: Sequence[np.ndarray]) -> bool:
    partials = [[_inverse_transform(slot, t) for t in range(8)] for slot in slots]
    anchor = partials[0][0]
    for i in range(1, len(slots)):
        ok = sum(1 for t in range(8) if _compatible(anchor, partials[i][t]))
        if ok >= 2:
            return True
    return False


def _multi_slot_coverage_needed(assignments: Sequence[int], slots: Sequence[np.ndarray]) -> bool:
    partials = [_inverse_transform(slot, t) != BLACK for slot, t in zip(slots, assignments)]
    cover_count = np.sum(np.stack(partials, axis=0), axis=0)
    unique_owner_slots = set()
    for idx, mask in enumerate(partials):
        if np.any(mask & (cover_count == 1)):
            unique_owner_slots.add(int(idx))
    return len(unique_owner_slots) >= 2


def _build_candidate_case() -> Tuple[np.ndarray, np.ndarray] | None:
    slot_rows, slot_cols, slot_size = _choose_layout()
    n_slots = slot_rows * slot_cols
    mother = _random_partition_pattern(slot_size)
    orientations = _choose_orientations(n_slots)
    full_slots = [_apply_transform(mother, t) for t in orientations]

    visible_masks = [_random_visible_mask(slot_size) for _ in range(n_slots)]
    coverage_masks = _coverage_masks_for_mother(orientations, visible_masks)
    covered = np.any(np.stack(coverage_masks, axis=0), axis=0)
    if not np.all(covered):
        missing_cells = np.argwhere(~covered)
        for r, c in missing_cells:
            candidates = list(range(n_slots))
            if not candidates:
                return None
            chosen = random.choice(candidates)
            rr, cc = _forward_coord(slot_size, int(r), int(c), orientations[chosen])
            visible_masks[chosen][rr, cc] = True

    if any(np.all(mask) or not np.any(~mask) for mask in visible_masks):
        return None
    coverage_masks = _coverage_masks_for_mother(orientations, visible_masks)
    if not np.all(np.any(np.stack(coverage_masks, axis=0), axis=0)):
        return None

    partial_slots = []
    for slot, visible in zip(full_slots, visible_masks):
        patch = np.full_like(slot, BLACK)
        patch[visible] = slot[visible]
        partial_slots.append(patch)

    inp = _assemble_grid(slot_rows, slot_cols, slot_size, partial_slots)
    expected = _assemble_grid(slot_rows, slot_cols, slot_size, full_slots)

    solved = _solve_slots(partial_slots, max_solutions=2)
    if len(solved) != 1:
        return None
    assignments, recovered = solved[0]
    if not np.array_equal(main(inp), expected):
        return None
    if not np.array_equal(recovered, full_slots[0]):
        return None
    if not _has_anchor_ambiguity(partial_slots):
        return None
    if not _multi_slot_coverage_needed(assignments, partial_slots):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _build_candidate_case()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid complementary-slot puzzle")
