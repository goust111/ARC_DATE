from __future__ import annotations

import random
from typing import Iterable

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _connected_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    steps = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    if not bool(mask.any()):
        return False
    return len(_connected_components(mask, connectivity=connectivity)) == 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _cluster_components(mask: np.ndarray, pad: int = 1) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    comps = _connected_components(mask, connectivity=8)
    assert comps
    n = len(comps)
    parents = list(range(n))
    boxes = [_bbox_from_coords(coords) for coords in comps]

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_overlap(boxes[i], boxes[j], pad=pad):
                union(i, j)

    grouped: dict[int, list[np.ndarray]] = {}
    for i, coords in enumerate(comps):
        grouped.setdefault(find(i), []).append(coords)

    result: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords_list in grouped.values():
        merged = np.concatenate(coords_list, axis=0)
        bbox = _bbox_from_coords(merged)
        result.append((bbox, merged))
    result.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]))
    return result


def _extract_copies(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray], int]:
    fg_mask = grid != bg
    colors = [int(color) for color in np.unique(grid[fg_mask])]
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    groups = _cluster_components(fg_mask, pad=1)
    assert len(groups) >= 3, "expected at least three sprite copies"

    bboxes: list[tuple[int, int, int, int]] = []
    patches: list[np.ndarray] = []
    for bbox, _coords in groups:
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1] == fg
        assert bool(patch.any())
        bboxes.append(bbox)
        patches.append(patch)
    return bboxes, patches, fg


def _has_nontrivial_rot_symmetry(mask: np.ndarray) -> bool:
    for k in (1, 2, 3):
        rotated = np.rot90(mask, k)
        if rotated.shape == mask.shape and np.array_equal(mask, rotated):
            return True
    return False


def _alignment_options(mask: np.ndarray, target_shape: tuple[int, int]) -> list[tuple[int, np.ndarray]]:
    options: list[tuple[int, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        if rotated.shape == target_shape:
            options.append((k, rotated))
    return options


def _union_score(coverage: np.ndarray) -> tuple[int, int, int, bytes]:
    union = coverage > 0
    connected_penalty = 0 if _is_connected(union, connectivity=8) else 1
    union_area = int(union.sum())
    overlap_score = int((coverage * coverage).sum())
    return connected_penalty, union_area, -overlap_score, union.tobytes()


def _infer_union_and_rotations(patches: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    target_shapes = {
        np.rot90(patch, k).shape
        for patch in patches
        for k in range(4)
    }
    best_key: tuple[int, int, int, bytes] | None = None
    best_union: np.ndarray | None = None
    best_assignment: list[int] | None = None

    for target_shape in sorted(target_shapes):
        options_per_patch = [_alignment_options(patch, target_shape) for patch in patches]
        if any(not options for options in options_per_patch):
            continue

        coverage = np.zeros(target_shape, dtype=int)
        assignment = [0] * len(patches)

        def dfs(index: int) -> None:
            nonlocal best_key, best_union, best_assignment
            if index == len(patches):
                score = _union_score(coverage)
                if best_key is None or score < best_key:
                    best_key = score
                    best_union = (coverage > 0).copy()
                    best_assignment = assignment.copy()
                return

            for k, rotated in options_per_patch[index]:
                coverage[rotated] += 1
                assignment[index] = k
                dfs(index + 1)
                coverage[rotated] -= 1

        dfs(0)

    assert best_union is not None and best_assignment is not None
    return best_union, best_assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a full sprite by overlaying multiple incomplete rotated copies, then
    write the reconstructed sprite back into every copy location.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    bboxes, patches, fg = _extract_copies(g, bg)
    full_mask, rotations_to_standard = _infer_union_and_rotations(patches)

    out = g.copy()
    for bbox, to_standard in zip(bboxes, rotations_to_standard):
        r0, c0, r1, c1 = bbox
        restored = np.rot90(full_mask, -to_standard)
        assert restored.shape == (r1 - r0 + 1, c1 - c0 + 1)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[:] = bg
        patch[restored] = fg
    return out


# --------------- generator ---------------

GRID_ROWS_RANGE = (18, 28)
GRID_COLS_RANGE = (18, 28)
COPY_COUNT_RANGE = (3, 5)
SHAPE_SIZE_RANGE = (4, 7)
SHAPE_FILL_MIN = 0.45
SHAPE_FILL_MAX = 0.72
FRAGMENT_SIZE_RANGE = (1, 4)
PLACEMENT_GAP = 3
PLACEMENT_RETRIES = 120
SHAPE_RETRIES = 120
FRAGMENT_RETRIES = 200
GENERATION_RETRIES = 400
PALETTE = list(range(10))


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = start, end
    steps: list[tuple[int, int]] = []
    steps.extend([(1 if r1 > r0 else -1, 0)] * abs(r1 - r0))
    steps.extend([(0, 1 if c1 > c0 else -1)] * abs(c1 - c0))
    random.shuffle(steps)

    r, c = r0, c0
    cells = {(r, c)}
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))
    return cells


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    frontier: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    filled = np.argwhere(mask)
    for r, c in filled:
        for dr, dc in NEIGHBORS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                frontier.append((nr, nc))
    return frontier


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_base_shape(copy_count: int) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        if random.random() < 0.5:
            size = random.randint(*SHAPE_SIZE_RANGE)
            h = w = size
        else:
            h = random.randint(*SHAPE_SIZE_RANGE)
            w = random.randint(*SHAPE_SIZE_RANGE)
            if h == w:
                w = SHAPE_SIZE_RANGE[0] if w != SHAPE_SIZE_RANGE[0] else SHAPE_SIZE_RANGE[1]

        anchors = [
            (0, random.randrange(w)),
            (random.randrange(h), 0),
            (h - 1, random.randrange(w)),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(anchors)
        cells = set(anchors[:1])
        for a, b in zip(anchors, anchors[1:]):
            cells |= _random_path(a, b)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        area_min = max(int(np.ceil(h * w * SHAPE_FILL_MIN)), len(cells) + copy_count)
        area_max = int(np.floor(h * w * SHAPE_FILL_MAX))
        if area_min > area_max:
            continue

        target_area = random.randint(area_min, area_max)
        while int(mask.sum()) < target_area:
            frontier = _frontier_cells(mask)
            if not frontier:
                break
            r, c = random.choice(frontier)
            mask[r, c] = True

        if not _is_connected(mask, connectivity=4):
            continue
        if not _spans_bbox(mask):
            continue
        if _has_nontrivial_rot_symmetry(mask):
            continue
        return mask
    return None


def _protected_edge_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()
    top = [(0, c) for c in np.where(mask[0])[0]]
    bottom = [(mask.shape[0] - 1, c) for c in np.where(mask[-1])[0]]
    left = [(r, 0) for r in np.where(mask[:, 0])[0]]
    right = [(r, mask.shape[1] - 1) for r in np.where(mask[:, -1])[0]]
    protected.add(random.choice(top))
    protected.add(random.choice(bottom))
    protected.add(random.choice(left))
    protected.add(random.choice(right))
    return protected


def _grow_fragment(mask: np.ndarray, protected: set[tuple[int, int]]) -> frozenset[tuple[int, int]] | None:
    candidates = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if (int(r), int(c)) not in protected
    ]
    if not candidates:
        return None

    target = random.randint(*FRAGMENT_SIZE_RANGE)
    start = random.choice(candidates)
    fragment = {start}

    while len(fragment) < target:
        frontier = []
        for r, c in fragment:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < mask.shape[0]
                    and 0 <= nc < mask.shape[1]
                    and mask[nr, nc]
                    and (nr, nc) not in protected
                    and (nr, nc) not in fragment
                ):
                    frontier.append((nr, nc))
        if not frontier:
            break
        fragment.add(random.choice(frontier))

    damaged = mask.copy()
    for r, c in fragment:
        damaged[r, c] = False

    if np.array_equal(damaged, mask):
        return None
    if not _spans_bbox(damaged):
        return None
    if not _is_connected(damaged, connectivity=4):
        return None
    if _has_nontrivial_rot_symmetry(damaged):
        return None
    return frozenset(fragment)


def _select_fragments(mask: np.ndarray, copy_count: int) -> list[frozenset[tuple[int, int]]] | None:
    protected = _protected_edge_cells(mask)
    candidates: dict[frozenset[tuple[int, int]], bytes] = {}
    for _ in range(FRAGMENT_RETRIES):
        fragment = _grow_fragment(mask, protected)
        if fragment is None:
            continue
        damaged = mask.copy()
        for r, c in fragment:
            damaged[r, c] = False
        candidates[fragment] = damaged.tobytes()
        if len(candidates) >= copy_count * 6:
            break

    items = list(candidates.items())
    if len(items) < copy_count:
        return None

    for _ in range(FRAGMENT_RETRIES):
        chosen = random.sample(items, copy_count)
        fragments = [fragment for fragment, _key in chosen]
        damaged_keys = [key for _fragment, key in chosen]
        if len(set(damaged_keys)) != copy_count:
            continue

        common = set(fragments[0])
        for fragment in fragments[1:]:
            common &= set(fragment)
        if common:
            continue
        return fragments
    return None


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_boxes(sizes: list[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed: dict[int, tuple[int, int, int, int]] = {}
    existing: list[tuple[int, int, int, int]] = []

    for idx in order:
        h, w = sizes[idx]
        if rows - h - 1 < 1 or cols - w - 1 < 1:
            return None
        box = None
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, rows - h - 1)
            c0 = random.randint(1, cols - w - 1)
            trial = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_separated(trial, prev, PLACEMENT_GAP) for prev in existing):
                box = trial
                break
        if box is None:
            return None
        placed[idx] = box
        existing.append(box)

    return [placed[i] for i in range(len(sizes))]


def _render_mask(grid: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, _r1, _c1 = box
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = color


def _build_expected(
    rows: int,
    cols: int,
    bg: int,
    fg: int,
    full_masks: list[np.ndarray],
    damaged_masks: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), bg, dtype=int)
    out = np.full((rows, cols), bg, dtype=int)
    for full_mask, damaged_mask, box in zip(full_masks, damaged_masks, boxes):
        _render_mask(inp, damaged_mask, box, fg)
        _render_mask(out, full_mask, box, fg)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    copy_count = random.randint(*COPY_COUNT_RANGE)
    full_mask = _make_base_shape(copy_count)
    if full_mask is None:
        return None

    fragments = _select_fragments(full_mask, copy_count)
    if fragments is None:
        return None

    bg, fg = random.sample(PALETTE, 2)
    full_masks: list[np.ndarray] = []
    damaged_masks: list[np.ndarray] = []
    for fragment in fragments:
        damaged = full_mask.copy()
        for r, c in fragment:
            damaged[r, c] = False
        rotation = random.randrange(4)
        full_masks.append(np.rot90(full_mask, rotation))
        damaged_masks.append(np.rot90(damaged, rotation))

    sizes = [mask.shape for mask in full_masks]
    row_min = max(GRID_ROWS_RANGE[0], max(h for h, _w in sizes) + 2)
    col_min = max(GRID_COLS_RANGE[0], max(w for _h, w in sizes) + 2)
    if row_min > GRID_ROWS_RANGE[1] or col_min > GRID_COLS_RANGE[1]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        rows = random.randint(row_min, GRID_ROWS_RANGE[1])
        cols = random.randint(col_min, GRID_COLS_RANGE[1])
        boxes = _place_boxes(sizes, rows, cols)
        if boxes is None:
            continue
        inp, out = _build_expected(rows, cols, bg, fg, full_masks, damaged_masks, boxes)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
