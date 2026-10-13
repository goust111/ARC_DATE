import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

GEN_RETRIES = 400
QUADRANT_SIZES = (7, 8)
COUNT_VALUES = (3, 4, 4, 5, 5, 6)
FRAGMENT_COUNT_VALUES = (1, 2)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_separator_index(vecs: np.ndarray) -> int:
    hits = [i for i in range(vecs.shape[0]) if np.all(vecs[i] == GRAY)]
    assert len(hits) == 1
    return int(hits[0])


def _split_quadrants(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sep_r = _find_separator_index(grid)
    sep_c = _find_separator_index(grid.T)
    assert sep_r * 2 + 1 == grid.shape[0]
    assert sep_c * 2 + 1 == grid.shape[1]
    ul = grid[:sep_r, :sep_c]
    ur = grid[:sep_r, sep_c + 1 :]
    ll = grid[sep_r + 1 :, :sep_c]
    lr = grid[sep_r + 1 :, sep_c + 1 :]
    assert ul.shape == ur.shape == ll.shape == lr.shape
    return ul, ur, ll, lr


def _template_mask_and_offsets(region: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    comps = _components(region == GREEN)
    assert len(comps) == 1
    mask = _tight_mask(region == GREEN)
    assert mask.shape == (3, 3)
    offsets = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    assert len(offsets) >= 4
    return mask, offsets


def _matching_offsets(
    fragment_cells: set[tuple[int, int]],
    template_offsets: set[tuple[int, int]],
    region_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    qh, qw = map(int, region_shape)
    template_list = sorted(template_offsets)
    candidates = set()
    for fr, fc in fragment_cells:
        for tr, tc in template_list:
            r0 = int(fr - tr)
            c0 = int(fc - tc)
            placed = {(r0 + rr, c0 + cc) for rr, cc in template_list}
            if any(rr < 0 or rr >= qh or cc < 0 or cc >= qw for rr, cc in placed):
                continue
            if fragment_cells.issubset(placed):
                candidates.add((r0, c0))
    return sorted(candidates)


def _reconstruct_fragment_copies(lr: np.ndarray, template_offsets: set[tuple[int, int]]) -> dict[int, set[tuple[int, int]]]:
    out: dict[int, set[tuple[int, int]]] = {}
    colors = [int(v) for v in np.unique(lr) if int(v) != BLACK]
    assert colors
    for color in colors:
        cells = {(int(r), int(c)) for r, c in np.argwhere(lr == color)}
        assert len(cells) >= 2
        offsets = _matching_offsets(cells, template_offsets, lr.shape)
        assert len(offsets) == 1
        r0, c0 = offsets[0]
        full_copy = {(r0 + rr, c0 + cc) for rr, cc in template_offsets}
        assert cells != full_copy
        out[color] = full_copy
    return out


def _tile_template(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    th, tw = mask.shape
    out = np.zeros((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            if mask[r % th, c % tw]:
                out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the upper control pixels to size an output, validate lower-right fragments against the lower-left template, and tile that template.

    Input structure:
    - A gray one-pixel cross splits the grid into four equal quadrants on a black background.
    - The upper-left quadrant contains scattered blue pixels whose count gives the output height, and the upper-right contains scattered red pixels whose count gives the output width.
    - The lower-left quadrant contains one green 4-connected irregular 3x3 template component.
    - The lower-right quadrant contains one yellow fragment and sometimes one orange fragment, each being a partial translated copy of the green template.

    Rule:
    - Count blue pixels in the upper-left to get output height and red pixels in the upper-right to get output width.
    - Crop the lower-left green component to its tight 3x3 bounding box to obtain the template mask.
    - For each non-black fragment color in the lower-right, find the unique translation where the template covers all observed fragment pixels; this only validates the template shape and placement logic.
    - Produce a black output of the counted size and tile the cropped green template mask from the top-left, clipping at the output boundary.

    Key insight:
    - The top quadrants decide only the output rectangle, while the bottom quadrants show that the reusable tile is the completed green template itself rather than the colored fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    ul, ur, ll, lr = _split_quadrants(g)
    out_h = int(np.count_nonzero(ul == BLUE))
    out_w = int(np.count_nonzero(ur == RED))
    assert out_h > 0 and out_w > 0

    template_mask, template_offsets = _template_mask_and_offsets(ll)
    _reconstruct_fragment_copies(lr, template_offsets)
    return _tile_template(template_mask, out_h, out_w)


def _is_asymmetric(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    return True


def _template_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    cells = [(r, c) for r in range(3) for c in range(3)]
    for area in range(4, 7):
        for subset in itertools.combinations(cells, area):
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in subset:
                mask[r, c] = True
            if not mask[0].any() or not mask[2].any() or not mask[:, 0].any() or not mask[:, 2].any():
                continue
            if len(_components(mask)) != 1:
                continue
            if mask.all():
                continue
            if not _is_asymmetric(mask):
                continue
            shapes.append(mask)
    assert shapes
    return shapes


TEMPLATE_LIBRARY = _template_library()


def _weighted_count() -> int:
    return int(random.choice(COUNT_VALUES))


def _sample_control_positions(q: int, count: int) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(q) for c in range(q)]
    mode = random.choice(["scatter", "compact", "rows"])
    if mode == "scatter":
        random.shuffle(coords)
        return coords[:count]
    if mode == "rows":
        rows = list(range(q))
        cols = list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        ordered = [(r, c) for r in rows for c in cols]
        return ordered[:count]
    ar = random.randrange(q)
    ac = random.randrange(q)
    coords.sort(key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    return coords[:count]


def _paste_mask(region: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    for r, c in coords:
        region[top + int(r), left + int(c)] = color


def _non_overlapping_starts(q: int, count: int) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(q - 2) for c in range(q - 2)]
    random.shuffle(starts)
    chosen: list[tuple[int, int]] = []
    for r, c in starts:
        ok = True
        for rr, cc in chosen:
            if not (r + 2 < rr or rr + 2 < r or c + 2 < cc or cc + 2 < c):
                ok = False
                break
        if ok:
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _fragment_subset_options(
    template_offsets: set[tuple[int, int]],
    start: tuple[int, int],
    q: int,
) -> list[set[tuple[int, int]]]:
    offsets = sorted(template_offsets)
    options: list[set[tuple[int, int]]] = []
    r0, c0 = start
    for size in range(2, len(offsets)):
        for subset in itertools.combinations(offsets, size):
            fragment = {(r0 + r, c0 + c) for r, c in subset}
            matches = _matching_offsets(fragment, template_offsets, (q, q))
            if len(matches) == 1:
                options.append(set(subset))
    return options


def _choose_fragment_subset(
    template_mask: np.ndarray,
    template_offsets: set[tuple[int, int]],
    start: tuple[int, int],
    q: int,
) -> set[tuple[int, int]] | None:
    options = _fragment_subset_options(template_offsets, start, q)
    if not options:
        return None

    def score(subset: set[tuple[int, int]]) -> tuple[int, int, float]:
        missing = len(template_offsets) - len(subset)
        local = np.zeros_like(template_mask, dtype=bool)
        for r, c in subset:
            local[r, c] = True
        connected = len(_components(local)) == 1
        return (missing, 0 if connected else 1, random.random())

    options.sort(key=score, reverse=True)
    top_band = options[: min(8, len(options))]
    return random.choice(top_band)


def _expected_output(template_mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    return _tile_template(template_mask, out_h, out_w)


def _anti_shortcut_checks(
    template_mask: np.ndarray,
    fragment_subsets: list[set[tuple[int, int]]],
    out_h: int,
    out_w: int,
    expected: np.ndarray,
) -> bool:
    if template_mask.all():
        return False
    solid = np.full((out_h, out_w), GREEN, dtype=int)
    if np.array_equal(solid, expected):
        return False
    first_subset = fragment_subsets[0]
    subset_mask = np.zeros_like(template_mask, dtype=bool)
    for r, c in first_subset:
        subset_mask[r, c] = True
    if np.array_equal(_tile_template(subset_mask, out_h, out_w), expected):
        return False
    bbox_frame = np.zeros_like(template_mask, dtype=bool)
    bbox_frame[:, :] = True
    if np.array_equal(_tile_template(bbox_frame, out_h, out_w), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    q = int(random.choice(QUADRANT_SIZES))
    side = 2 * q + 1
    grid = np.zeros((side, side), dtype=int)
    grid[q, :] = GRAY
    grid[:, q] = GRAY

    ul = grid[:q, :q]
    ur = grid[:q, q + 1 :]
    ll = grid[q + 1 :, :q]
    lr = grid[q + 1 :, q + 1 :]

    out_h = _weighted_count()
    out_w = _weighted_count()
    for r, c in _sample_control_positions(q, out_h):
        ul[r, c] = BLUE
    for r, c in _sample_control_positions(q, out_w):
        ur[r, c] = RED

    template_mask = random.choice(TEMPLATE_LIBRARY)
    ll_top = random.randint(1, q - 4)
    ll_left = random.randint(1, q - 4)
    _paste_mask(ll, template_mask, ll_top, ll_left, GREEN)
    template_offsets = {(int(r), int(c)) for r, c in np.argwhere(template_mask)}

    n_fragments = int(random.choice(FRAGMENT_COUNT_VALUES))
    starts = _non_overlapping_starts(q, n_fragments)
    if starts is None:
        return None

    colors = [YELLOW] if n_fragments == 1 else [YELLOW, ORANGE]
    random.shuffle(colors)
    fragment_subsets: list[set[tuple[int, int]]] = []
    for color, start in zip(colors, starts):
        subset = _choose_fragment_subset(template_mask, template_offsets, start, q)
        if subset is None:
            return None
        fragment_subsets.append(subset)
        for r, c in subset:
            lr[start[0] + int(r), start[1] + int(c)] = color

    expected = _expected_output(template_mask, out_h, out_w)
    if not _anti_shortcut_checks(template_mask, fragment_subsets, out_h, out_w, expected):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    _, _, _, lr_check = _split_quadrants(grid)
    full_copies = _reconstruct_fragment_copies(lr_check, template_offsets)
    if len(full_copies) != n_fragments:
        return None
    if any(len(full_copies[color]) != len(template_offsets) for color in full_copies):
        return None
    if any((grid == color).sum() >= len(template_offsets) for color in full_copies):
        return None
    if any(color not in (YELLOW, ORANGE) for color in full_copies):
        return None
    if np.count_nonzero(expected == GREEN) == 0 or np.count_nonzero(expected == BLACK) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
