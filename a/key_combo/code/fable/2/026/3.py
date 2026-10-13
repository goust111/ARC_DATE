import random
import numpy as np


ALL_COLORS = tuple(range(10))
PATTERN_COLORS = tuple(range(1, 10))
GEN_MAX_ATTEMPTS = 400
MASK_MODES = ("vertical_blocked", "horizontal_blocked")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    return (
        (int(r), int(c)),
        (int(n - 1 - r), int(c)),
        (int(r), int(n - 1 - c)),
        (int(n - 1 - r), int(n - 1 - c)),
    )


def _swap_color(color: int, pair: tuple[int, int]) -> int:
    a, b = pair
    if int(color) == int(a):
        return int(b)
    if int(color) == int(b):
        return int(a)
    return int(color)


def _swap_array(arr: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    a, b = pair
    out = np.array(arr, dtype=int, copy=True)
    mask_a = out == int(a)
    mask_b = out == int(b)
    out[mask_a] = int(b)
    out[mask_b] = int(a)
    return out


def _solid_rect_color_candidates(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, np.unique(grid)))):
        mask = grid == int(color)
        if not bool(mask.any()):
            continue
        box = _bbox(mask)
        r0, c0, r1, c1 = box
        if int(mask.sum()) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)):
            candidates.append((int(color), box))
    return candidates


def _infer_swap_pair(grid: np.ndarray, mask_color: int) -> tuple[int, int]:
    n = int(grid.shape[0])
    mismatch_pairs: set[tuple[int, int]] = set()
    mismatch_count = 0
    for r in range(n):
        for c in range(n // 2):
            left = int(grid[r, c])
            right = int(grid[r, n - 1 - c])
            if left == int(mask_color) or right == int(mask_color):
                continue
            if left == right:
                continue
            mismatch_pairs.add(tuple(sorted((left, right))))
            mismatch_count += 1
    assert mismatch_count > 0
    assert len(mismatch_pairs) == 1
    pair = next(iter(mismatch_pairs))
    assert pair[0] != pair[1]
    return int(pair[0]), int(pair[1])


def _validate_visible_symmetry(
    grid: np.ndarray,
    mask_color: int,
    swap_pair: tuple[int, int],
) -> None:
    n = int(grid.shape[0])
    for r in range(n // 2):
        for c in range(n):
            top = int(grid[r, c])
            bottom = int(grid[n - 1 - r, c])
            if top == int(mask_color) or bottom == int(mask_color):
                continue
            assert top == bottom

    saw_swap = False
    for r in range(n):
        for c in range(n // 2):
            left = int(grid[r, c])
            right = int(grid[r, n - 1 - c])
            if left == int(mask_color) or right == int(mask_color):
                continue
            want = _swap_color(left, swap_pair)
            assert right == want
            if left != right:
                saw_swap = True
    assert saw_swap


def _reconstruct_mask(
    grid: np.ndarray,
    mask_box: tuple[int, int, int, int],
    mask_color: int,
    swap_pair: tuple[int, int],
) -> np.ndarray:
    n = int(grid.shape[0])
    r0, c0, r1, c1 = mask_box
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            inferred: list[int] = []
            for rr, cc in _orbit(r, c, n):
                color = int(grid[rr, cc])
                if color == int(mask_color):
                    continue
                if int(cc) != int(c):
                    color = _swap_color(color, swap_pair)
                inferred.append(int(color))
            assert inferred
            assert len(set(inferred)) == 1
            out[r - r0, c - c0] = inferred[0]
    return out


def _valid_mask_candidates(
    grid: np.ndarray,
) -> list[tuple[int, tuple[int, int, int, int], tuple[int, int], np.ndarray]]:
    valid: list[tuple[int, tuple[int, int, int, int], tuple[int, int], np.ndarray]] = []
    for mask_color, box in _solid_rect_color_candidates(grid):
        try:
            swap_pair = _infer_swap_pair(grid, mask_color)
            _validate_visible_symmetry(grid, mask_color, swap_pair)
            out = _reconstruct_mask(grid, box, mask_color, swap_pair)
        except AssertionError:
            continue
        valid.append((int(mask_color), box, swap_pair, out))
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the masked rectangle from a pattern with horizontal symmetry and vertical color-swap symmetry.

    Input structure:
    - The input is an even-sized square grid filled by a multicolor pattern.
    - Reflecting across the horizontal midline preserves every visible color exactly.
    - Reflecting across the vertical midline preserves stable colors but swaps one fixed color pair.
    - One solid rectangular mask of a new color hides part of the pattern and may span one of the two midlines.

    Rule:
    - Detect the solid mask rectangle and infer the swapped color pair from visible vertical mirror pairs.
    - For each masked cell, inspect its four-cell orbit under horizontal and vertical reflection.
    - Any visible representative in the same column contributes the target color directly; any visible representative in the mirrored column contributes the target color after one swap of the inferred pair.
    - Return the reconstructed mask contents cropped to the mask rectangle.

    Key insight:
    - The two axes are not equivalent: horizontal reflection is color-preserving, while vertical reflection toggles exactly one color pair, so reconstruction depends on orbit membership and vertical-reflection parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h == w
    assert h % 2 == 0

    candidates = _valid_mask_candidates(g)
    assert len(candidates) == 1
    return candidates[0][3]


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= int(parts) <= int(total)
    if int(parts) == 1:
        return [0, int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    return [0, *cuts, int(total)]


def _make_quadrant(
    half: int,
    swap_pair: tuple[int, int],
    stable_colors: tuple[int, ...],
) -> np.ndarray:
    row_blocks = random.randint(2, min(4, half))
    col_blocks = random.randint(2, min(4, half))
    row_edges = _random_partition(half, row_blocks)
    col_edges = _random_partition(half, col_blocks)
    palette = list(stable_colors) + [int(swap_pair[0]), int(swap_pair[1])]
    block_colors = np.empty((row_blocks, col_blocks), dtype=int)
    for i in range(row_blocks):
        for j in range(col_blocks):
            block_colors[i, j] = random.choice(palette)

    slots = [(i, j) for i in range(row_blocks) for j in range(col_blocks)]
    random.shuffle(slots)
    block_colors[slots[0]] = int(swap_pair[0])
    block_colors[slots[1]] = int(swap_pair[1])
    block_colors[slots[2]] = int(stable_colors[0])

    quadrant = np.empty((half, half), dtype=int)
    for i in range(row_blocks):
        for j in range(col_blocks):
            r0, r1 = row_edges[i], row_edges[i + 1]
            c0, c1 = col_edges[j], col_edges[j + 1]
            quadrant[r0:r1, c0:c1] = int(block_colors[i, j])
    return quadrant


def _build_full_pattern(quadrant: np.ndarray, swap_pair: tuple[int, int]) -> np.ndarray:
    top = np.concatenate([quadrant, _swap_array(np.fliplr(quadrant), swap_pair)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)


def _count_hidden_vertical(box: tuple[int, int, int, int], n: int) -> int:
    r0, c0, r1, c1 = box
    count = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            mc = n - 1 - c
            if c0 <= mc <= c1:
                count += 1
    return int(count)


def _count_hidden_horizontal(box: tuple[int, int, int, int], n: int) -> int:
    r0, c0, r1, c1 = box
    count = 0
    for r in range(r0, r1 + 1):
        mr = n - 1 - r
        if r0 <= mr <= r1:
            count += c1 - c0 + 1
    return int(count)


def _visible_vertical_mismatches(grid: np.ndarray, mask_color: int) -> int:
    n = int(grid.shape[0])
    count = 0
    for r in range(n):
        for c in range(n // 2):
            left = int(grid[r, c])
            right = int(grid[r, n - 1 - c])
            if left == int(mask_color) or right == int(mask_color):
                continue
            if left != right:
                count += 1
    return int(count)


def _shortcut_horizontal(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    mask_color: int,
) -> np.ndarray | None:
    n = int(grid.shape[0])
    r0, c0, r1, c1 = box
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rr, cc = n - 1 - r, c
            color = int(grid[rr, cc])
            if color == int(mask_color):
                return None
            out[r - r0, c - c0] = color
    return out


def _shortcut_vertical(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    mask_color: int,
    swap_pair: tuple[int, int],
    *,
    apply_swap: bool,
) -> np.ndarray | None:
    n = int(grid.shape[0])
    r0, c0, r1, c1 = box
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rr, cc = r, n - 1 - c
            color = int(grid[rr, cc])
            if color == int(mask_color):
                return None
            if bool(apply_swap):
                color = _swap_color(color, swap_pair)
            out[r - r0, c - c0] = color
    return out


def _has_parity_witness(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    mask_color: int,
    swap_pair: tuple[int, int],
) -> bool:
    n = int(grid.shape[0])
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            h_color = int(grid[n - 1 - r, c])
            p_color = int(grid[n - 1 - r, n - 1 - c])
            if h_color == int(mask_color) or p_color == int(mask_color):
                continue
            if h_color != p_color and _swap_color(p_color, swap_pair) == h_color:
                return True
    return False


def _mode_boxes(n: int, mode: str) -> list[tuple[int, int, int, int]]:
    half = n // 2
    boxes: list[tuple[int, int, int, int]] = []
    max_h = min(8, n - 2)
    max_w = min(8, n - 2)
    for h in range(2, max_h + 1):
        for w in range(2, max_w + 1):
            if mode == "vertical_blocked":
                c_start_lo = max(0, half - w + 1)
                c_start_hi = min(half - 1, n - w)
                if c_start_lo > c_start_hi:
                    continue
                for c0 in range(c_start_lo, c_start_hi + 1):
                    c1 = c0 + w - 1
                    if not (c0 < half and c1 >= half):
                        continue
                    for r0 in range(0, half - h + 1):
                        boxes.append((r0, c0, r0 + h - 1, c1))
                    for r0 in range(half, n - h + 1):
                        boxes.append((r0, c0, r0 + h - 1, c1))
            else:
                r_start_lo = max(0, half - h + 1)
                r_start_hi = min(half - 1, n - h)
                if r_start_lo > r_start_hi:
                    continue
                for r0 in range(r_start_lo, r_start_hi + 1):
                    r1 = r0 + h - 1
                    if not (r0 < half and r1 >= half):
                        continue
                    for c0 in range(0, half - w + 1):
                        boxes.append((r0, c0, r1, c0 + w - 1))
                    for c0 in range(half, n - w + 1):
                        boxes.append((r0, c0, r1, c0 + w - 1))
    random.shuffle(boxes)
    return boxes


def _validate_instance(
    masked: np.ndarray,
    full: np.ndarray,
    box: tuple[int, int, int, int],
    mask_color: int,
    mode: str,
    swap_pair: tuple[int, int],
) -> bool:
    expected = _crop_bbox(full, box)
    if len(set(map(int, expected.ravel()))) < 2:
        return False
    if not any(int(x) in swap_pair for x in expected.ravel()):
        return False
    if _visible_vertical_mismatches(masked, mask_color) < 4:
        return False

    candidates = _valid_mask_candidates(masked)
    if len(candidates) != 1:
        return False
    solved = candidates[0][3]
    if not np.array_equal(solved, expected):
        return False

    if mode == "horizontal_blocked":
        if _count_hidden_horizontal(box, masked.shape[0]) == 0:
            return False
        if _shortcut_horizontal(masked, box, mask_color) is not None:
            return False
        vertical = _shortcut_vertical(masked, box, mask_color, swap_pair, apply_swap=True)
        if vertical is None or not np.array_equal(vertical, expected):
            return False
        wrong = _shortcut_vertical(masked, box, mask_color, swap_pair, apply_swap=False)
        if wrong is None or np.array_equal(wrong, expected):
            return False
    else:
        if _count_hidden_vertical(box, masked.shape[0]) == 0:
            return False
        if _shortcut_vertical(masked, box, mask_color, swap_pair, apply_swap=True) is not None:
            return False
        horizontal = _shortcut_horizontal(masked, box, mask_color)
        if horizontal is None or not np.array_equal(horizontal, expected):
            return False
        if not _has_parity_witness(masked, box, mask_color, swap_pair):
            return False

    try:
        via_main = main(masked)
    except AssertionError:
        return False
    return np.array_equal(via_main, expected)


def _generate_one() -> np.ndarray | None:
    half = random.randint(6, 12)
    n = half * 2
    swap_pair = tuple(sorted(random.sample(PATTERN_COLORS, 2)))
    stable_pool = [c for c in PATTERN_COLORS if c not in swap_pair]
    stable_count = random.randint(2, 3)
    stable_colors = tuple(random.sample(stable_pool, stable_count))
    quadrant = _make_quadrant(half, swap_pair, stable_colors)
    full = _build_full_pattern(quadrant, swap_pair)

    mode = random.choice(MASK_MODES)
    available_mask_colors = [c for c in ALL_COLORS if c not in set(map(int, np.unique(full)))]
    if not available_mask_colors:
        return None
    mask_color = random.choice(available_mask_colors)

    for box in _mode_boxes(n, mode):
        masked = full.copy()
        r0, c0, r1, c1 = box
        masked[r0 : r1 + 1, c0 : c1 + 1] = int(mask_color)
        if _validate_instance(masked, full, box, mask_color, mode, swap_pair):
            return masked
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid half-color-swap symmetry puzzle")
