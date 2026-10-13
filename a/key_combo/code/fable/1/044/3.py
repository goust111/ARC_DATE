from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

try:
    from arc_utils import get_bg as _shared_get_bg
except Exception:
    _shared_get_bg = None


_QUADRANTS = ("tl", "tr", "bl", "br")
_MAX_GENERATION_ATTEMPTS = 400
_MIN_QUADRANT_SIDE = 4
_MAX_QUADRANT_SIDE = 14
_PATTERN_MODES = ("blob", "split")


def _get_bg(grid: np.ndarray) -> int:
    if _shared_get_bg is not None:
        return int(_shared_get_bg(grid))
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        coords = np.argwhere(grid == color)
        if int(coords.shape[0]) != 4:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r1 - r0), int(c1 - c0)) != (1, 1):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((int(color), (int(r0), int(c0), int(r1), int(c1))))

    assert len(candidates) == 1, "expected exactly one solid 2x2 marker block"
    return candidates[0]


def _quadrant_masks(grid: np.ndarray, marker_box: tuple[int, int, int, int], fg_color: int) -> list[np.ndarray]:
    r0, c0, r1, c1 = marker_box
    h, w = map(int, grid.shape)

    top_h = int(r0)
    bottom_h = int(h - r1 - 1)
    left_w = int(c0)
    right_w = int(w - c1 - 1)
    assert top_h == bottom_h and left_w == right_w, "marker block must sit at the grid center"
    assert top_h > 0 and left_w > 0, "quadrants must have positive size"

    tl = grid[:r0, :c0] == int(fg_color)
    tr = np.fliplr(grid[:r0, c1 + 1 :] == int(fg_color))
    bl = np.flipud(grid[r1 + 1 :, :c0] == int(fg_color))
    br = np.flipud(np.fliplr(grid[r1 + 1 :, c1 + 1 :] == int(fg_color)))

    assert tl.shape == tr.shape == bl.shape == br.shape
    return [tl, tr, bl, br]


def _paint_quadrants(
    out: np.ndarray,
    normalized_masks: list[np.ndarray],
    marker_box: tuple[int, int, int, int],
    fg_color: int,
) -> None:
    r0, c0, r1, c1 = marker_box
    out[:r0, :c0][normalized_masks[0]] = int(fg_color)
    out[:r0, c1 + 1 :][np.fliplr(normalized_masks[1])] = int(fg_color)
    out[r1 + 1 :, :c0][np.flipud(normalized_masks[2])] = int(fg_color)
    out[r1 + 1 :, c1 + 1 :][np.flipud(np.fliplr(normalized_masks[3]))] = int(fg_color)


def _build_output(
    clean_mask: np.ndarray,
    shape: tuple[int, int],
    marker_box: tuple[int, int, int, int],
    bg: int,
    marker_color: int,
    fg_color: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r0, c0, r1, c1 = marker_box
    out[r0 : r1 + 1, c0 : c1 + 1] = int(marker_color)
    _paint_quadrants(out, [clean_mask] * 4, marker_box, fg_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct the clean four-way pattern by majority-voting the three noisy quadrants.

    Input structure
    - The grid has one dominant background color and one solid 2x2 marker block at the center.
    - Around that marker, exactly three quadrants contain a noisy foreground pattern and one quadrant is fully blank.
    - The three visible patterns use one foreground color and are intended to be mirror-equivalent across the marker's horizontal and vertical axes.

    Rule
    - Detect the unique 2x2 marker block and treat its center as the horizontal and vertical reflection axes.
    - Fold the three non-empty quadrants into one shared quadrant coordinate system, then take a per-cell majority vote: foreground wins iff at least two quadrants mark that cell.
    - Paint that voted clean mask back into all four quadrants by reflection, while preserving the marker block unchanged.

    Key insight
    - No single visible quadrant is trustworthy: the only deterministic reconstruction is to align the three redundant noisy copies and use per-cell majority voting before mirroring the result back out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    marker_color, marker_box = _find_marker(g, bg)
    other_colors = sorted(set(map(int, np.unique(g))) - {int(bg), int(marker_color)})
    assert len(other_colors) == 1, "expected exactly one foreground pattern color"
    fg_color = other_colors[0]

    normalized = _quadrant_masks(g, marker_box, fg_color)
    known = [mask for mask in normalized if bool(mask.any())]
    assert len(known) == 3, "expected exactly three non-empty quadrants"

    votes = np.sum(np.stack(known, axis=0), axis=0)
    clean = votes >= 2
    return _build_output(clean, g.shape, marker_box, bg, marker_color, fg_color)


def _neighbors4(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _random_rect_cells(h: int, w: int, *, max_h: int, max_w: int) -> set[tuple[int, int]]:
    rh = random.randint(1, max_h)
    rw = random.randint(1, max_w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    return {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}


def _seed_cells(h: int, w: int, mode: str) -> set[tuple[int, int]]:
    max_seed_h = max(1, min(3, h // 2))
    max_seed_w = max(1, min(4, w // 2))
    cells = _random_rect_cells(h, w, max_h=max_seed_h, max_w=max_seed_w)

    if mode != "split":
        return cells

    for _ in range(40):
        other = _random_rect_cells(h, w, max_h=max_seed_h, max_w=max_seed_w)
        if cells & other:
            continue
        if any(nb in cells for r, c in other for nb in _neighbors4(h, w, r, c)):
            continue
        return cells | other
    return cells


def _grow_mask(h: int, w: int, target_area: int, seed_cells: set[tuple[int, int]]) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    frontier: set[tuple[int, int]] = set()

    for r, c in seed_cells:
        mask[r, c] = True
    for r, c in seed_cells:
        frontier.update(nb for nb in _neighbors4(h, w, r, c) if not mask[nb])

    while int(mask.sum()) < int(target_area):
        if not frontier:
            return None
        candidates = list(frontier)
        # Prefer cells that keep the shape locally clustered while staying irregular.
        candidates.sort(
            key=lambda cell: (
                -sum(1 for nb in _neighbors4(h, w, cell[0], cell[1]) if mask[nb]),
                random.random(),
            )
        )
        r, c = candidates[0]
        frontier.remove((r, c))
        if mask[r, c]:
            continue
        mask[r, c] = True
        frontier.update(nb for nb in _neighbors4(h, w, r, c) if not mask[nb])

    return mask


def _bbox_area(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _sample_clean_mask(qh: int, qw: int) -> np.ndarray:
    total = qh * qw
    min_area = max(6, total // 4)
    max_area = min(total - 6, (11 * total) // 20)

    for _ in range(120):
        target_area = random.randint(min_area, max_area)
        mode = random.choice(_PATTERN_MODES)
        seeds = _seed_cells(qh, qw, mode)
        if len(seeds) > target_area:
            continue
        mask = _grow_mask(qh, qw, target_area, seeds)
        if mask is None:
            continue
        if int(mask.sum()) < 6 or int((~mask).sum()) < 6:
            continue
        if _bbox_area(mask) == int(mask.sum()):
            continue
        return mask

    raise RuntimeError("failed to sample a non-degenerate canonical quadrant mask")


def _partition_unique_coords(
    coords: list[tuple[int, int]],
    group_count: int,
    max_each: int,
) -> list[list[tuple[int, int]]] | None:
    if len(coords) < group_count:
        return None

    pool = coords[:]
    random.shuffle(pool)
    groups = [[pool.pop()] for _ in range(group_count)]
    extra_budget = random.randint(0, min(len(pool), group_count * (max_each - 1)))

    while extra_budget > 0 and pool:
        choices = [idx for idx in range(group_count) if len(groups[idx]) < max_each]
        if not choices:
            break
        groups[random.choice(choices)].append(pool.pop())
        extra_budget -= 1

    return groups


def _noisy_quadrants(clean_mask: np.ndarray, blank_index: int) -> list[np.ndarray] | None:
    known_indices = [idx for idx in range(4) if idx != blank_index]
    quadrants = [np.zeros_like(clean_mask, dtype=bool) for _ in range(4)]
    for idx in known_indices:
        quadrants[idx] = clean_mask.copy()

    fg_coords = [tuple(map(int, coord)) for coord in np.argwhere(clean_mask)]
    bg_coords = [tuple(map(int, coord)) for coord in np.argwhere(~clean_mask)]
    max_each = 3 if clean_mask.size >= 45 else 2

    deletions = _partition_unique_coords(fg_coords, len(known_indices), max_each)
    additions = _partition_unique_coords(bg_coords, len(known_indices), max_each)
    if deletions is None or additions is None:
        return None

    for idx, del_cells, add_cells in zip(known_indices, deletions, additions):
        view = quadrants[idx]
        for r, c in del_cells:
            view[r, c] = False
        for r, c in add_cells:
            view[r, c] = True

    return quadrants


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    clean_mask: np.ndarray,
    blank_index: int,
) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    bg = _get_bg(inp)
    marker_color, marker_box = _find_marker(inp, bg)
    fg_colors = sorted(set(map(int, np.unique(inp))) - {int(bg), int(marker_color)})
    if len(fg_colors) != 1:
        return False

    normalized = _quadrant_masks(inp, marker_box, fg_colors[0])
    if sum(1 for mask in normalized if not mask.any()) != 1:
        return False
    if normalized[blank_index].any():
        return False

    known = [normalized[idx] for idx in range(4) if idx != blank_index]
    votes = np.sum(np.stack(known, axis=0), axis=0)
    if not np.array_equal(votes >= 2, clean_mask):
        return False
    if not np.any(clean_mask & (votes == 2)):
        return False
    if not np.any((~clean_mask) & (votes == 1)):
        return False

    for mask in known:
        if np.array_equal(mask, clean_mask):
            return False
        if not np.any(clean_mask & ~mask):
            return False
        if not np.any((~clean_mask) & mask):
            return False

    out_normalized = _quadrant_masks(expected, marker_box, fg_colors[0])
    if not all(np.array_equal(mask, clean_mask) for mask in out_normalized):
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    qh = random.randint(_MIN_QUADRANT_SIDE, _MAX_QUADRANT_SIDE)
    qw = random.randint(_MIN_QUADRANT_SIDE, _MAX_QUADRANT_SIDE)
    height = 2 * qh + 2
    width = 2 * qw + 2
    assert height <= 30 and width <= 30

    bg, fg_color, marker_color = random.sample(list(range(10)), 3)
    marker_box = (qh, qw, qh + 1, qw + 1)
    blank_index = random.randrange(4)

    clean_mask = _sample_clean_mask(qh, qw)
    noisy = _noisy_quadrants(clean_mask, blank_index)
    if noisy is None:
        raise RuntimeError("failed to sample quadrant noise")

    inp = np.full((height, width), int(bg), dtype=int)
    r0, c0, r1, c1 = marker_box
    inp[r0 : r1 + 1, c0 : c1 + 1] = int(marker_color)
    _paint_quadrants(inp, noisy, marker_box, fg_color)

    expected = _build_output(clean_mask, inp.shape, marker_box, bg, marker_color, fg_color)
    return inp, expected, clean_mask, blank_index


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected, clean_mask, blank_index = _generate_case()
        except RuntimeError:
            continue
        if _semantic_check(inp, expected, clean_mask, blank_index):
            return inp
    raise RuntimeError("failed to generate a valid three-quadrant-voting ARC input")
