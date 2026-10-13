import random
from collections import deque, defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
MAGENTA = 6

ATTEMPT_LIMIT = 600
HEIGHT_RANGE = (11, 18)
PANEL_WIDTH_RANGE = (5, 9)
DAMAGE_COUNT_RANGE = (4, 7)
AREA_RATIO_RANGE = (0.30, 0.50)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == YELLOW)]
    assert len(cols) == 1, f"expected one full-height yellow separator, got {cols}"
    return cols[0]


def _split_panels(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    sep = _find_separator(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    assert left.shape[1] == right.shape[1] and left.shape[1] > 0, "panels must have equal positive width"
    return sep, left, right


def _ideal_right_from_left(left: np.ndarray) -> np.ndarray:
    mirror = np.fliplr(left)
    return np.where(mirror == RED, BLACK, BLUE).astype(int)


def _filled_output(grid: np.ndarray) -> np.ndarray:
    sep, left, right = _split_panels(grid)
    expected_right = _ideal_right_from_left(left)
    out = grid.copy()
    out[:, sep + 1 :] = np.where(right == MAGENTA, expected_right, right)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair magenta damage in a right panel that is the mirrored negative of the left panel.

    Input structure:
    - The grid has black background and exactly one full-height yellow separator column.
    - The left panel contains a single 4-connected red shape on black.
    - The right panel has the same size as the left panel and is mostly complete, but some cells are replaced by magenta damage markers.

    Rule:
    - Split the grid at the yellow separator into left and right panels.
    - Match each right-panel cell with the horizontally mirrored cell in the left panel.
    - If the mirrored left cell is red, the correct right value is black; if the mirrored left cell is black, the correct right value is blue.
    - Replace every magenta right-panel cell with that mirrored inverted value and leave all other cells unchanged.

    Key insight:
    - The right panel is not a direct mirror copy: position comes from horizontal mirroring, while color comes from occupancy inversion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _filled_output(g)


def _shape_signature(mask: np.ndarray) -> tuple:
    coords = sorted(map(tuple, np.argwhere(mask)))
    return tuple(coords)


def _grow_connected_shape(h: int, w: int, target_area: int) -> np.ndarray:
    seed_r = random.randrange(h)
    seed_c = random.randrange(w)
    cells = {(seed_r, seed_c)}
    frontier = set()
    for nr, nc in _neighbors4(seed_r, seed_c):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < target_area and frontier:
        frontier_list = list(frontier)
        weights = []
        for r, c in frontier_list:
            touch = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
            center_bias = 1 + min(r, h - 1 - r, c, w - 1 - c)
            weights.append(4 * touch + center_bias)
        chosen = random.choices(frontier_list, weights=weights, k=1)[0]
        frontier.remove(chosen)
        cells.add(chosen)
        for nr, nc in _neighbors4(*chosen):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _valid_left_shape(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    ratio = area / float(h * w)
    if not (AREA_RATIO_RANGE[0] <= ratio <= AREA_RATIO_RANGE[1]):
        return False
    comps = _components4(mask)
    if len(comps) != 1:
        return False
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.sum() < max(3, h // 2):
        return False
    if cols.sum() < max(3, w // 2):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if area == h * w or area == 0:
        return False
    return True


def _neighbor_signature(right: np.ndarray, r: int, c: int) -> tuple[int, int, int]:
    n_black = 0
    n_blue = 0
    degree = 0
    h, w = right.shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w:
            degree += 1
            n_black += int(right[nr, nc] == BLACK)
            n_blue += int(right[nr, nc] == BLUE)
    return n_black, n_blue, degree


def _choose_damage_positions(left: np.ndarray, expected_right: np.ndarray) -> list[tuple[int, int]]:
    h, w = left.shape
    by_target = {
        BLACK: [(r, c) for r in range(h) for c in range(w) if expected_right[r, c] == BLACK],
        BLUE: [(r, c) for r in range(h) for c in range(w) if expected_right[r, c] == BLUE],
    }
    if not by_target[BLACK] or not by_target[BLUE]:
        return []

    sig_groups: dict[tuple[int, int, int], dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: {BLACK: [], BLUE: []}
    )
    for target, positions in by_target.items():
        for r, c in positions:
            sig_groups[_neighbor_signature(expected_right, r, c)][target].append((r, c))

    paired_signatures = [sig for sig, groups in sig_groups.items() if groups[BLACK] and groups[BLUE]]
    if not paired_signatures:
        return []

    total_damage = random.randrange(*DAMAGE_COUNT_RANGE)
    chosen: list[tuple[int, int]] = []

    def can_add(pos: tuple[int, int]) -> bool:
        return all(max(abs(pos[0] - rr), abs(pos[1] - cc)) > 1 for rr, cc in chosen)

    forced_sig = random.choice(paired_signatures)
    black_pair = random.choice(sig_groups[forced_sig][BLACK])
    blue_pair = random.choice(sig_groups[forced_sig][BLUE])
    if not can_add(black_pair):
        return []
    chosen.append(black_pair)
    if not can_add(blue_pair):
        return []
    chosen.append(blue_pair)

    asymmetric = [(r, c) for r in range(h) for c in range(w) if left[r, c] != left[r, w - 1 - c]]
    random.shuffle(asymmetric)
    for pos in asymmetric:
        if pos in chosen or not can_add(pos):
            continue
        if len(chosen) >= 3:
            break
        chosen.append(pos)

    remaining = [pos for pos in by_target[BLACK] + by_target[BLUE] if pos not in chosen]
    random.shuffle(remaining)
    for pos in remaining:
        if len(chosen) >= total_damage:
            break
        if can_add(pos):
            chosen.append(pos)

    if len(chosen) != total_damage:
        return []
    random.shuffle(chosen)
    return chosen


def _wrong_all_constant(grid: np.ndarray, color: int) -> np.ndarray:
    sep, _left, right = _split_panels(grid)
    out = grid.copy()
    out[:, sep + 1 :] = np.where(right == MAGENTA, color, right)
    return out


def _wrong_same_column_inversion(grid: np.ndarray) -> np.ndarray:
    sep, left, right = _split_panels(grid)
    out = grid.copy()
    same_col = np.where(left == RED, BLACK, BLUE)
    out[:, sep + 1 :] = np.where(right == MAGENTA, same_col, right)
    return out


def _wrong_mirror_without_inversion(grid: np.ndarray) -> np.ndarray:
    sep, left, right = _split_panels(grid)
    out = grid.copy()
    mirror_direct = np.where(np.fliplr(left) == RED, BLUE, BLACK)
    out[:, sep + 1 :] = np.where(right == MAGENTA, mirror_direct, right)
    return out


def _wrong_local_majority(grid: np.ndarray) -> np.ndarray:
    sep, _left, right = _split_panels(grid)
    out = grid.copy()
    filled = right.copy()
    visible = right[right != MAGENTA]
    default_color = BLUE if np.count_nonzero(visible == BLUE) >= np.count_nonzero(visible == BLACK) else BLACK
    for r, c in map(tuple, np.argwhere(right == MAGENTA)):
        votes = []
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < right.shape[0] and 0 <= nc < right.shape[1] and right[nr, nc] != MAGENTA:
                votes.append(int(right[nr, nc]))
        if votes:
            blue_votes = votes.count(BLUE)
            black_votes = votes.count(BLACK)
            filled[r, c] = BLUE if blue_votes >= black_votes else BLACK
        else:
            filled[r, c] = default_color
    out[:, sep + 1 :] = filled
    return out


def _validate_example(grid: np.ndarray) -> bool:
    try:
        sep, left, right = _split_panels(grid)
        if sep <= 0 or sep >= grid.shape[1] - 1:
            return False
        if set(np.unique(left)) - {BLACK, RED}:
            return False
        if set(np.unique(right)) - {BLACK, BLUE, MAGENTA}:
            return False
        if np.any(grid[:, sep] != YELLOW):
            return False
        if not _valid_left_shape(left == RED):
            return False

        expected_right = _ideal_right_from_left(left)
        damage_mask = right == MAGENTA
        damage_positions = list(map(tuple, np.argwhere(damage_mask)))
        if not (DAMAGE_COUNT_RANGE[0] <= len(damage_positions) < DAMAGE_COUNT_RANGE[1]):
            return False
        if np.any(right[~damage_mask] != expected_right[~damage_mask]):
            return False

        targets = {int(expected_right[r, c]) for r, c in damage_positions}
        if targets != {BLACK, BLUE}:
            return False

        if not any(left[r, c] != left[r, left.shape[1] - 1 - c] for r, c in damage_positions):
            return False

        sig_targets: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        for r, c in damage_positions:
            sig_targets[_neighbor_signature(right, r, c)].add(int(expected_right[r, c]))
        if not any(values == {BLACK, BLUE} for values in sig_targets.values()):
            return False

        expected = _filled_output(grid)
        wrong_solvers = [
            _wrong_all_constant(grid, BLACK),
            _wrong_all_constant(grid, BLUE),
            _wrong_same_column_inversion(grid),
            _wrong_mirror_without_inversion(grid),
            _wrong_local_majority(grid),
        ]
        if any(np.array_equal(wrong, expected) for wrong in wrong_solvers):
            return False

        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randrange(*HEIGHT_RANGE)
    w = random.randrange(*PANEL_WIDTH_RANGE)
    area_min = int(np.ceil(AREA_RATIO_RANGE[0] * h * w))
    area_max = int(np.floor(AREA_RATIO_RANGE[1] * h * w))
    target_area = random.randint(area_min, area_max)

    left_mask = _grow_connected_shape(h, w, target_area)
    if not _valid_left_shape(left_mask):
        return None

    left = np.where(left_mask, RED, BLACK).astype(int)
    expected_right = _ideal_right_from_left(left)
    damage_positions = _choose_damage_positions(left, expected_right)
    if not damage_positions:
        return None

    right = expected_right.copy()
    for r, c in damage_positions:
        right[r, c] = MAGENTA

    grid = np.zeros((h, 2 * w + 1), dtype=int)
    grid[:, :w] = left
    grid[:, w] = YELLOW
    grid[:, w + 1 :] = right

    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    seen = set()
    for _ in range(ATTEMPT_LIMIT):
        grid = _generate_attempt()
        if grid is None:
            continue
        signature = _shape_signature(grid == RED), tuple(grid.shape)
        if signature in seen and len(seen) < 8:
            continue
        seen.add(signature)
        return grid
    raise RuntimeError("failed to generate a valid reverse-mirror-completion input")
