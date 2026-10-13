import hashlib
import itertools
import random
from collections import Counter

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 600
SEEN_INPUT_SIGS: set[str] = set()
SEEN_OUTPUT_SIGS: set[str] = set()


def _grid_sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _rotate_coord(r: int, c: int, n: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return n - 1 - c, r
    if k == 2:
        return n - 1 - r, n - 1 - c
    return c, n - 1 - r


def _anchor_positions(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
    }


def _extract_patches_and_anchor(grid: np.ndarray) -> tuple[list[np.ndarray], int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    anchor_coords = np.argwhere(g == ANCHOR_COLOR)
    assert anchor_coords.shape[0] == 1, "expected exactly one gray anchor pixel"
    anchor = tuple(map(int, anchor_coords[0]))

    mask = (g != bg) & (g != ANCHOR_COLOR)
    comps = _components_mask(mask)
    assert len(comps) == 4, "expected four square copies"

    items = []
    for coords in comps:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        n = r1 - r0 + 1
        assert n == c1 - c0 + 1, "copies must be square"
        patch = _crop(g, box)
        assert np.all((patch != bg) & (patch != ANCHOR_COLOR)), "copy bbox must be fully occupied"
        items.append((r0, c0, box, patch))

    items.sort(key=lambda item: (item[0], item[1]))
    boxes = [item[2] for item in items]
    patches = [item[3] for item in items]

    anchor_matches = [i for i, box in enumerate(boxes) if anchor in _anchor_positions(box)]
    assert len(anchor_matches) == 1, "anchor must identify exactly one copy"
    return patches, anchor_matches[0], boxes


def _assignment_score_and_tile(
    patches: list[np.ndarray], assignment: tuple[int, int, int, int]
) -> tuple[int, np.ndarray]:
    aligned = [np.rot90(patch, -rot) for patch, rot in zip(patches, assignment)]
    n = aligned[0].shape[0]
    out = np.zeros((n, n), dtype=int)
    score = 0
    for r in range(n):
        for c in range(n):
            counts = Counter(int(tile[r, c]) for tile in aligned)
            best = max(counts.values())
            score += best
            winners = [color for color, cnt in counts.items() if cnt == best]
            out[r, c] = winners[0]
    return score, out


def _all_assignments() -> list[tuple[int, int, int, int]]:
    return [(0, *perm) for perm in itertools.permutations((1, 2, 3))]


def _best_assignment(patches: list[np.ndarray]) -> tuple[tuple[int, int, int, int], np.ndarray, int]:
    scored = []
    for assignment in _all_assignments():
        score, tile = _assignment_score_and_tile(patches, assignment)
        scored.append((score, assignment, tile))
    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "rotation alignment must be uniquely determined"
    return scored[0][1], scored[0][2], scored[0][0]


def _majority_tile(patches: list[np.ndarray], assignment: tuple[int, int, int, int]) -> np.ndarray:
    aligned = [np.rot90(patch, -rot) for patch, rot in zip(patches, assignment)]
    n = aligned[0].shape[0]
    out = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            counts = Counter(int(tile[r, c]) for tile in aligned)
            best = max(counts.values())
            winners = [color for color, cnt in counts.items() if cnt == best]
            assert best >= 3 and len(winners) == 1, "each rotational orbit must have a unique majority"
            out[r, c] = winners[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the scattered noisy rotated copies by orbit majority vote and return the tile in the anchored copy's orientation.

    Input structure:
    - A mostly background grid contains four disjoint square copies of the same N×N multicolor tile.
    - The four copies are the four quarter-turn rotations of that tile, but their positions are arbitrary and unlabeled.
    - Some cells are corrupted to wrong colors, with at most one corruption inside each four-way rotational orbit.
    - Exactly one copy has a single gray anchor pixel placed diagonally just outside one corner of its bbox.

    Rule:
    - Detect the four square copies and the anchored one.
    - Find the unique rotation assignment that makes the four copies align with the highest cellwise color agreement.
    - Rotate the copies into that common frame, recover each cell by majority vote across the four witnesses, then rotate the reconstructed tile back to the anchored copy's displayed orientation.

    Key insight:
    - The gray marker identifies only which copy's orientation should be output; the actual content must be reconstructed from all four copies because the anchored copy can itself be corrupted.
    """
    patches, anchor_idx, _ = _extract_patches_and_anchor(grid)
    assignment, _, _ = _best_assignment(patches)
    tile = _majority_tile(patches, assignment)
    return np.rot90(tile, assignment[anchor_idx]).copy()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _paint_random_rect(tile: np.ndarray, color: int) -> None:
    n = tile.shape[0]
    h = random.randint(1, max(1, n - 1))
    w = random.randint(1, max(1, n - 1))
    r0 = random.randint(0, n - h)
    c0 = random.randint(0, n - w)
    tile[r0 : r0 + h, c0 : c0 + w] = color


def _paint_random_l(tile: np.ndarray, color: int) -> None:
    n = tile.shape[0]
    corner = random.choice([(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)])
    arm_a = random.randint(2, n)
    arm_b = random.randint(2, n)
    rr, cc = corner
    if rr == 0:
        rows = slice(0, arm_a)
    else:
        rows = slice(n - arm_a, n)
    if cc == 0:
        cols = slice(0, arm_b)
    else:
        cols = slice(n - arm_b, n)
    tile[rows, cc] = color
    tile[rr, cols] = color


def _paint_random_band(tile: np.ndarray, color: int) -> None:
    n = tile.shape[0]
    if random.random() < 0.5:
        r = random.randrange(n)
        c0 = random.randint(0, n - 2)
        c1 = random.randint(c0 + 1, n - 1)
        tile[r, c0 : c1 + 1] = color
    else:
        c = random.randrange(n)
        r0 = random.randint(0, n - 2)
        r1 = random.randint(r0 + 1, n - 1)
        tile[r0 : r1 + 1, c] = color


def _paint_random_block(tile: np.ndarray, color: int) -> None:
    n = tile.shape[0]
    h = random.randint(1, min(2, n))
    w = random.randint(1, min(2, n))
    r0 = random.randint(0, n - h)
    c0 = random.randint(0, n - w)
    tile[r0 : r0 + h, c0 : c0 + w] = color


def _make_tile(n: int) -> np.ndarray:
    for _ in range(300):
        palette = random.sample(PALETTE, random.randint(3, min(5, len(PALETTE))))
        tile = np.full((n, n), palette[0], dtype=int)

        ops = [_paint_random_rect, _paint_random_l, _paint_random_band, _paint_random_block]
        for color in palette[1:]:
            random.choice(ops)(tile, color)
        for _ in range(random.randint(2, 5)):
            random.choice(ops)(tile, random.choice(palette))

        used = sorted(set(map(int, tile.flat)))
        if len(used) < 3:
            continue
        if ANCHOR_COLOR in used or BACKGROUND in used:
            continue
        if any(np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3)):
            continue
        if min(_hamming(tile, np.rot90(tile, k)) for k in (1, 2, 3)) < max(4, (n * n) // 3):
            continue
        corner_colors = {int(tile[0, 0]), int(tile[0, -1]), int(tile[-1, 0]), int(tile[-1, -1])}
        if len(corner_colors) < 2:
            continue
        return tile
    raise RuntimeError("failed to generate a sufficiently distinctive tile")


def _copy_metadata(tile: np.ndarray) -> tuple[list[np.ndarray], list[int], int]:
    rotations = random.sample([0, 1, 2, 3], 4)
    patches = [np.rot90(tile, k).copy() for k in rotations]
    anchor_idx = random.randrange(4)
    return patches, rotations, anchor_idx


def _apply_corruption(
    tile: np.ndarray, patches: list[np.ndarray], rotations: list[int], anchor_idx: int
) -> tuple[list[np.ndarray], list[list[tuple[int, int]]]]:
    n = tile.shape[0]
    palette = sorted(set(map(int, tile.flat)))
    wrong_pool = [color for color in PALETTE if color not in {ANCHOR_COLOR, BACKGROUND}]
    orbit_cells = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(orbit_cells)

    total_corrupt = random.randint(4, min(n * n - 1, 2 * n + 2))
    assignments = [anchor_idx]
    remaining_copies = [idx for idx in range(4) if idx != anchor_idx]
    random.shuffle(remaining_copies)
    assignments.extend(remaining_copies)
    while len(assignments) < total_corrupt:
        assignments.append(random.randrange(4))

    corrupted = [patch.copy() for patch in patches]
    by_copy = [[] for _ in range(4)]
    for (r, c), copy_idx in zip(orbit_cells[:total_corrupt], assignments):
        rr, cc = _rotate_coord(r, c, n, rotations[copy_idx])
        truth = int(corrupted[copy_idx][rr, cc])
        choices = [color for color in wrong_pool if color != truth]
        if random.random() < 0.7:
            local_choices = [color for color in palette if color != truth]
            if local_choices:
                choices = local_choices
        corrupted[copy_idx][rr, cc] = random.choice(choices)
        by_copy[copy_idx].append((rr, cc))

    assert all(by_copy[idx] for idx in range(4)), "every copy must carry at least one corruption"
    return corrupted, by_copy


def _boxes_far_enough(
    boxes: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = candidate
    for or0, oc0, or1, oc1 in boxes:
        if not (r1 + 1 < or0 - 1 or or1 + 1 < r0 - 1 or c1 + 1 < oc0 - 1 or oc1 + 1 < c0 - 1):
            return False
    return True


def _place_patches(
    patches: list[np.ndarray], anchor_idx: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], tuple[int, int]]:
    n = patches[0].shape[0]
    for _ in range(300):
        h = random.randint(max(14, 3 * n), min(30, 5 * n + 4))
        w = random.randint(max(14, 3 * n), min(30, 5 * n + 4))
        boxes = []
        for _copy_idx in range(4):
            placed = False
            for _ in range(300):
                r0 = random.randint(1, h - n - 2)
                c0 = random.randint(1, w - n - 2)
                candidate = (r0, c0, r0 + n - 1, c0 + n - 1)
                if _boxes_far_enough(boxes, candidate):
                    boxes.append(candidate)
                    placed = True
                    break
            if not placed:
                boxes = []
                break
        if len(boxes) != 4:
            continue

        grid = np.full((h, w), BACKGROUND, dtype=int)
        for patch, box in zip(patches, boxes):
            r0, c0, r1, c1 = box
            grid[r0 : r1 + 1, c0 : c1 + 1] = patch

        ar0, ac0, ar1, ac1 = boxes[anchor_idx]
        anchor = random.choice(list(_anchor_positions(boxes[anchor_idx])))
        if not (0 <= anchor[0] < h and 0 <= anchor[1] < w):
            continue
        if grid[anchor] != BACKGROUND:
            continue
        grid[anchor] = ANCHOR_COLOR
        return grid, boxes, anchor
    raise RuntimeError("failed to place the four copies")


def _expected_assignment(rotations: list[int]) -> tuple[int, int, int, int]:
    shift = rotations[0]
    return tuple((rot - shift) % 4 for rot in rotations)


def _validate_candidate(
    inp: np.ndarray,
    true_tile: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    rotations: list[int],
    anchor_idx: int,
    corruptions: list[list[tuple[int, int]]],
) -> bool:
    expected = np.rot90(true_tile, rotations[anchor_idx])
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    patches, found_anchor_idx, found_boxes = _extract_patches_and_anchor(inp)
    if found_anchor_idx < 0 or len(found_boxes) != 4:
        return False
    if found_anchor_idx >= 4:
        return False

    order = sorted(range(4), key=lambda idx: (boxes[idx][0], boxes[idx][1]))
    sorted_rotations = [rotations[idx] for idx in order]
    sorted_corruptions = [corruptions[idx] for idx in order]
    expected_assign = _expected_assignment(sorted_rotations)

    scores = []
    for assignment in _all_assignments():
        score, _ = _assignment_score_and_tile(patches, assignment)
        scores.append((score, assignment))
    scores.sort(key=lambda item: item[0], reverse=True)
    if scores[0][1] != expected_assign:
        return False
    if scores[0][0] <= scores[1][0]:
        return False

    if not all(sorted_corruptions):
        return False

    anchor_patch = patches[found_anchor_idx]
    if np.array_equal(anchor_patch, expected):
        return False

    for patch, rot in zip(patches, sorted_rotations):
        if np.array_equal(np.rot90(patch, -rot), true_tile):
            return False

    output_sig = _grid_sig(expected)
    if output_sig in SEEN_OUTPUT_SIGS:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(4, 6)
    tile = _make_tile(n)
    clean_patches, rotations, anchor_idx = _copy_metadata(tile)
    corrupted_patches, corruptions = _apply_corruption(tile, clean_patches, rotations, anchor_idx)
    inp, boxes, _anchor = _place_patches(corrupted_patches, anchor_idx)

    if not _validate_candidate(inp, tile, boxes, rotations, anchor_idx, corruptions):
        return None

    input_sig = _grid_sig(inp)
    output_sig = _grid_sig(main(inp))
    if input_sig in SEEN_INPUT_SIGS or output_sig in SEEN_OUTPUT_SIGS:
        return None
    SEEN_INPUT_SIGS.add(input_sig)
    SEEN_OUTPUT_SIGS.add(output_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scattered-rotation majority puzzle")
