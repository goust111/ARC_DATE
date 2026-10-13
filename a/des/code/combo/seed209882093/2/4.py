import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _rot_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, -int(quarter_turns) % 4)


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sequence_from_row(row: np.ndarray, sep_color: int) -> List[int]:
    cols = np.flatnonzero((row != BLACK) & (row != sep_color))
    return [int(row[c]) for c in cols]


def _template_candidates(upper: np.ndarray, color: int, size: int) -> List[Tuple[int, int]]:
    coords = np.argwhere(upper == color)
    if coords.size == 0:
        return []

    r0, c0, r1, c1 = _bbox(coords)
    top_lo = max(0, r1 - size + 1)
    top_hi = min(r0, upper.shape[0] - size)
    left_lo = max(0, c1 - size + 1)
    left_hi = min(c0, upper.shape[1] - size)
    if top_lo > top_hi or left_lo > left_hi:
        return []

    total = int(coords.shape[0])
    other_mask = (upper != BLACK) & (upper != color)
    candidates: List[Tuple[int, int]] = []
    for top in range(top_lo, top_hi + 1):
        for left in range(left_lo, left_hi + 1):
            window = upper[top : top + size, left : left + size]
            if window.shape != (size, size):
                continue
            if int(np.count_nonzero(window == color)) != total:
                continue
            if bool(other_mask[top : top + size, left : left + size].any()):
                continue
            candidates.append((top, left))
    return candidates


def _assign_non_overlapping_lefts(
    left_options: Dict[int, List[int]], size: int
) -> Optional[Dict[int, int]]:
    colors = sorted(left_options, key=lambda color: len(left_options[color]))
    assigned: Dict[int, int] = {}

    def backtrack(i: int) -> bool:
        if i == len(colors):
            return True
        color = colors[i]
        for left in left_options[color]:
            if any(abs(left - prev_left) <= size for prev_left in assigned.values()):
                continue
            assigned[color] = left
            if backtrack(i + 1):
                return True
            del assigned[color]
        return False

    if not backtrack(0):
        return None
    return assigned


def _extract_templates(upper: np.ndarray, colors: Iterable[int], size: int) -> Dict[int, np.ndarray]:
    colors = list(dict.fromkeys(int(color) for color in colors))
    assert colors

    per_color: Dict[int, List[Tuple[int, int]]] = {}
    common_tops: Optional[set[int]] = None
    for color in colors:
        candidates = _template_candidates(upper, color, size)
        assert candidates
        per_color[color] = candidates
        tops = {top for top, _left in candidates}
        common_tops = tops if common_tops is None else (common_tops & tops)

    assert common_tops

    for top in sorted(common_tops, reverse=True):
        left_options = {
            color: sorted(left for cand_top, left in per_color[color] if cand_top == top)
            for color in colors
        }
        if any(not options for options in left_options.values()):
            continue
        assigned = _assign_non_overlapping_lefts(left_options, size)
        if assigned is None:
            continue
        return {
            color: upper[top : top + size, left : left + size].copy()
            for color, left in assigned.items()
        }

    raise AssertionError("template windows are ambiguous")


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert len(rows) == 1
    return int(rows[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Print rotated template squares onto the lower canvas according to the color strip.

    The grid is split by a full-width grey row. Above it are same-sized square
    templates plus a one-row color strip; below it is a blank black canvas whose
    height equals the template side length. Each strip color selects a template,
    which is rotated clockwise by (non-black pixel count mod 4) quarter-turns
    before being pasted into the next output slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    height, width = map(int, g.shape)
    sep = _find_separator_row(g)
    size = height - sep - 1
    assert size > 0
    assert width % size == 0

    seq_len = width // size
    candidates: List[Tuple[int, List[int], Dict[int, np.ndarray]]] = []
    for row in range(sep):
        seq = _sequence_from_row(g[row], GREY)
        if len(seq) != seq_len:
            continue
        if not seq:
            continue
        try:
            templates = _extract_templates(g[:row, :], seq, size)
        except AssertionError:
            continue
        candidates.append((row, seq, templates))

    assert candidates
    seq_row, seq, templates = max(candidates, key=lambda item: item[0])
    assert seq_row < sep

    out = g.copy()
    out[sep + 1 :, :] = BLACK
    for i, color in enumerate(seq):
        template = templates[int(color)]
        turns = int(np.count_nonzero(template != BLACK) % 4)
        out[sep + 1 : sep + 1 + size, i * size : (i + 1) * size] = _rot_cw(template, turns)
    return out


def _connect_path(
    cells: set[Tuple[int, int]], start: Tuple[int, int], end: Tuple[int, int], rng: random.Random
) -> None:
    r, c = start
    er, ec = end
    cells.add((r, c))
    while (r, c) != (er, ec):
        moves = []
        if r < er:
            moves.append((1, 0))
        elif r > er:
            moves.append((-1, 0))
        if c < ec:
            moves.append((0, 1))
        elif c > ec:
            moves.append((0, -1))
        dr, dc = rng.choice(moves)
        r += dr
        c += dc
        cells.add((r, c))


def _has_dihedral_symmetry(mask: np.ndarray) -> bool:
    for k in range(4):
        rot = np.rot90(mask, k)
        if k and np.array_equal(mask, rot):
            return True
        if np.array_equal(mask, np.fliplr(rot)):
            return True
    return False


def _grow_frontier(cells: set[Tuple[int, int]], size: int) -> List[Tuple[int, int]]:
    frontier = set()
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return list(frontier)


def _random_template_mask(size: int, target_mod: int, rng: random.Random) -> Optional[np.ndarray]:
    for _ in range(300):
        center = (rng.randrange(size), rng.randrange(size))
        anchors = [
            (0, rng.randrange(size)),
            (size - 1, rng.randrange(size)),
            (rng.randrange(size), 0),
            (rng.randrange(size), size - 1),
        ]
        cells = {center}
        for anchor in anchors:
            _connect_path(cells, anchor, center, rng)

        min_count = max(len(cells), size + 1)
        counts = [n for n in range(min_count, size * size) if n % 4 == target_mod]
        if not counts:
            continue
        target_count = rng.choice(counts)

        while len(cells) < target_count:
            frontier = _grow_frontier(cells, size)
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != target_count:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not mask[0].any() or not mask[-1].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if _has_dihedral_symmetry(mask):
            continue
        return mask
    return None


def _build_sequence(colors: Sequence[int], length: int, rng: random.Random) -> List[int]:
    seq = list(colors)
    rng.shuffle(seq)
    while len(seq) < length:
        seq.insert(rng.randrange(len(seq) + 1), rng.choice(colors))
    return seq


def _render_expected(
    inp: np.ndarray, sep: int, size: int, seq: Sequence[int], templates: Dict[int, np.ndarray]
) -> np.ndarray:
    out = inp.copy()
    out[sep + 1 :, :] = BLACK
    for i, color in enumerate(seq):
        template = templates[int(color)]
        turns = int(np.count_nonzero(template != BLACK) % 4)
        out[sep + 1 : sep + 1 + size, i * size : (i + 1) * size] = _rot_cw(template, turns)
    return out


def _slot_columns(size: int, count: int, rng: random.Random) -> List[int]:
    return [i * size + rng.randrange(size) for i in range(count)]


def _layout_attempt(rng: random.Random) -> Optional[np.ndarray]:
    size = rng.randint(3, 6)
    max_templates = min(5, len(COLOR_POOL))
    n_templates = rng.randint(2, max_templates)

    base_left = rng.randint(0, max(0, size - 1))
    base_right = rng.randint(0, max(0, size - 1))
    base_gaps = [rng.randint(1, max(1, size // 2 + 1)) for _ in range(n_templates - 1)]
    min_width = base_left + base_right + n_templates * size + sum(base_gaps)

    min_seq_len = max(n_templates, (min_width + size - 1) // size)
    max_seq_len = 30 // size
    if min_seq_len > max_seq_len:
        return None

    duplicate_low = max(min_seq_len, n_templates + 1)
    if duplicate_low <= max_seq_len and rng.random() < 0.7:
        low = duplicate_low
    else:
        low = min_seq_len
    high = min(max_seq_len, low + 3)
    seq_len = rng.randint(low, high)
    width = size * seq_len

    extras = [0] * (n_templates + 1)
    for _ in range(width - min_width):
        extras[rng.randrange(len(extras))] += 1

    left_margin = base_left + extras[0]
    gaps = [base_gaps[i] + extras[i + 1] for i in range(n_templates - 1)]
    top_margin = rng.randint(0, 3)
    gap_above_seq = rng.randint(1, 2)
    seq_row = top_margin + size + gap_above_seq
    sep = seq_row + 1
    height = sep + 1 + size

    colors = rng.sample(COLOR_POOL, n_templates)
    mods = [rng.randrange(4) for _ in range(n_templates)]
    if all(mod == 0 for mod in mods):
        mods[rng.randrange(n_templates)] = rng.randint(1, 3)
    if n_templates >= 4 and rng.random() < 0.5:
        forced = [0, 1, 2, 3]
        rng.shuffle(forced)
        mods[:4] = forced

    masks = []
    for mod in mods:
        mask = _random_template_mask(size, mod, rng)
        if mask is None:
            return None
        masks.append(mask)

    grid = np.zeros((height, width), dtype=int)
    grid[sep, :] = GREY

    templates: Dict[int, np.ndarray] = {}
    left = left_margin
    for i, color in enumerate(colors):
        patch = np.zeros((size, size), dtype=int)
        patch[masks[i]] = int(color)
        grid[top_margin : top_margin + size, left : left + size] = patch
        templates[int(color)] = patch
        left += size
        if i + 1 < n_templates:
            left += gaps[i]

    seq = _build_sequence(colors, seq_len, rng)
    for col, color in zip(_slot_columns(size, seq_len, rng), seq):
        grid[seq_row, col] = int(color)

    colored_counts = [
        int(np.count_nonzero((grid[r] != BLACK) & (grid[r] != GREY))) for r in range(sep)
    ]
    if sum(count == seq_len for count in colored_counts) != 1:
        return None

    expected = _render_expected(grid, sep, size, seq, templates)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _layout_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
