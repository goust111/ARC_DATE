from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg, paste
from itertools import combinations


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[bool, ...]]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for transformed in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, transformed.shape)), tuple(map(bool, transformed.ravel())))
            if key in seen:
                continue
            seen.add(key)
            transforms.append(transformed.copy())
    return transforms


def _find_source(
    g: np.ndarray, bg: int
) -> tuple[int, int, np.ndarray]:
    source_candidates: list[tuple[int, int, np.ndarray]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        fill_color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(region)))
        marker_colors = colors - {bg, fill_color}

        if bg in colors or len(marker_colors) != 1:
            continue
        if int(np.count_nonzero(region == fill_color)) != int(comp["area"]):
            continue

        marker_color = marker_colors.pop()
        marker_mask = region == marker_color
        source_candidates.append((fill_color, marker_color, marker_mask))

    assert len(source_candidates) == 1, "expected one filled rectangle with embedded markers"
    return source_candidates[0]


def _matching_fill_cells(
    g: np.ndarray, bg: int, marker_color: int, marker_mask: np.ndarray
) -> list[frozenset[tuple[int, int]]]:
    H, W = map(int, g.shape)
    matches: list[frozenset[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()

    for transformed in _unique_transforms(marker_mask):
        h, w = map(int, transformed.shape)
        marker_total = int(np.count_nonzero(transformed))
        assert marker_total > 0, "source marker mask must be nonempty"

        for r0 in range(1 - h, H):
            for c0 in range(1 - w, W):
                marker_hits = 0
                fill_cells: list[tuple[int, int]] = []
                ok = True

                for rr in range(h):
                    for cc in range(w):
                        r = r0 + rr
                        c = c0 + cc
                        if not (0 <= r < H and 0 <= c < W):
                            continue

                        if transformed[rr, cc]:
                            if int(g[r, c]) != marker_color:
                                ok = False
                                break
                            marker_hits += 1
                        else:
                            if int(g[r, c]) != bg:
                                ok = False
                                break
                            fill_cells.append((r, c))
                    if not ok:
                        break

                if ok and marker_hits == marker_total and fill_cells:
                    frozen = frozenset(fill_cells)
                    if frozen not in seen:
                        seen.add(frozen)
                        matches.append(frozen)

    assert matches, "expected at least one empty rectangle matching the marker mask"
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each empty rectangle whose marker pixels match a transformed copy of the source sprite's marker mask.

    Input structure
    - A dominant background contains one solid rectangular source sprite made from a fill color and embedded marker-color pixels.
    - Additional marker-color pixels are scattered elsewhere, and some groups form transformed copies of the source marker mask.

    Rule
    - Find the source rectangle as the unique fill-color component whose bbox is fully occupied by fill and marker colors.
    - Consider all unique rotations and reflections of the source rectangle's marker mask.
    - Wherever a transformed marker mask exactly overlays marker-color pixels and every other in-bounds cell of that rectangle is background, paint those background cells with the fill color.

    Key insight
    The scattered marker pixels identify empty stamp locations by reproducing the source sprite's internal marker pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fill_color, marker_color, marker_mask = _find_source(g, bg)

    out = g.copy()
    for fill_cells in _matching_fill_cells(g, bg, marker_color, marker_mask):
        for r, c in fill_cells:
            out[r, c] = fill_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and fixed colors.
GRID_SIZE = 23
BG_COLOR = 0
FILL_COLOR = 1
MARKER_COLOR = 4

# Source rectangle and interior marker constraints.
SOURCE_H_MIN = 3
SOURCE_H_MAX_EXCL = 6
SOURCE_W_MIN = 4
SOURCE_W_MAX_EXCL = 6
SOURCE_MARK_MIN = 2
SOURCE_MARK_MAX = 3
SOURCE_MIN_UNIQUE_TRANSFORMS = 2

# Generated match slots, distractors, and retry budget.
MATCH_COUNT_MIN = 1
MATCH_COUNT_MAX_EXCL = 6
DISTRACTOR_COUNT_MIN = 11
DISTRACTOR_COUNT_MAX_EXCL = 36
GEN_RETRY_LIMIT = 300


def _gen_mask_from_cells(shape: tuple[int, int], cells) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_source_mask() -> np.ndarray:
    h = int(np.random.randint(SOURCE_H_MIN, SOURCE_H_MAX_EXCL))
    w = int(np.random.randint(SOURCE_W_MIN, SOURCE_W_MAX_EXCL))
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]

    by_count = {}
    for n_marks in range(SOURCE_MARK_MIN, min(SOURCE_MARK_MAX, len(interior)) + 1):
        choices = []
        for combo in combinations(interior, n_marks):
            mask = _gen_mask_from_cells((h, w), combo)
            if _gen_marker_components_rectangular(mask) and len(
                _unique_transforms(mask)
            ) >= SOURCE_MIN_UNIQUE_TRANSFORMS:
                choices.append(combo)
        if choices:
            by_count[n_marks] = choices

    counts = sorted(by_count)
    n_marks = counts[int(np.random.randint(0, len(counts)))]
    combo = by_count[n_marks][int(np.random.randint(0, len(by_count[n_marks])))]
    return _gen_mask_from_cells((h, w), combo)


def _gen_marker_components_rectangular(mask: np.ndarray) -> bool:
    for coords in components_mask(mask, connectivity=4):
        r0, c0 = np.min(coords, axis=0)
        r1, c1 = np.max(coords, axis=0)
        if int(np.count_nonzero(mask[r0 : r1 + 1, c0 : c1 + 1])) != int(
            (r1 - r0 + 1) * (c1 - c0 + 1)
        ):
            return False
    return True


def _gen_match_index(
    marker_mask: np.ndarray,
):
    placements = []
    by_cell = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    for transformed in _unique_transforms(marker_mask):
        h, w = map(int, transformed.shape)
        for r0 in range(1 - h, GRID_SIZE):
            for c0 in range(1 - w, GRID_SIZE):
                marker_cells = []
                fill_cells = []
                ok = True
                for rr in range(h):
                    for cc in range(w):
                        r = r0 + rr
                        c = c0 + cc
                        in_bounds = 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE
                        if transformed[rr, cc]:
                            if not in_bounds:
                                ok = False
                                break
                            marker_cells.append((r, c))
                        elif in_bounds:
                            fill_cells.append((r, c))
                    if not ok:
                        break
                if not ok or not fill_cells:
                    continue

                marker_tuple = tuple(marker_cells)
                fill_set = frozenset(fill_cells)
                footprint = frozenset(marker_cells + fill_cells)
                placements.append((transformed, r0, c0, marker_tuple, fill_set, footprint))
                for r, c in marker_tuple:
                    by_cell[r][c].append((marker_tuple, fill_set))

    return placements, by_cell


def _gen_candidate_safe(grid, r, c, marker_color, allowed_matches, by_cell) -> bool:
    for marker_cells, fill_set in by_cell[r][c]:
        for mr, mc in marker_cells:
            if (mr, mc) != (r, c) and int(grid[mr, mc]) != marker_color:
                break
        else:
            for fr, fc in fill_set:
                if int(grid[fr, fc]) != BG_COLOR:
                    break
            else:
                key = (frozenset(marker_cells), fill_set)
                if key not in allowed_matches:
                    return False
    return True


def _gen_valid_match_keys(grid: np.ndarray, marker_color: int, placements):
    keys = set()
    for _, _, _, marker_cells, fill_set, _ in placements:
        if all(int(grid[r, c]) == marker_color for r, c in marker_cells) and all(
            int(grid[r, c]) == 0 for r, c in fill_set
        ):
            keys.add((frozenset(marker_cells), fill_set))
    return keys


def _gen_place_matches(out: np.ndarray, placements, occupied):
    chosen = []
    match_count = int(np.random.randint(MATCH_COUNT_MIN, MATCH_COUNT_MAX_EXCL))
    for _ in range(match_count):
        options = [p for p in placements if not (p[5] & occupied)]
        if not options:
            return None
        placement = options[int(np.random.randint(0, len(options)))]
        chosen.append(placement)
        occupied = occupied | placement[5]

        transformed, r0, c0, _, _, _ = placement
        patch = np.where(transformed, MARKER_COLOR, FILL_COLOR)
        paste(out, patch, r0, c0, clip=True)
    return chosen


def _gen_remove_match_fills(out: np.ndarray, chosen):
    inp = out.copy()
    expected_fills = set()
    expected_fill_cells = set()
    for _, _, _, _, fill_set, _ in chosen:
        expected_fills.add(fill_set)
        expected_fill_cells |= set(fill_set)
        for r, c in fill_set:
            inp[r, c] = BG_COLOR
    return inp, expected_fills, expected_fill_cells


def _gen_solver_matches(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        return np.array_equal(main(inp), out)
    except Exception:
        return False


def _gen_add_distractors(inp, out, source_bbox, expected_fill_cells, allowed_matches, by_cell) -> bool:
    target_distractors = int(np.random.randint(DISTRACTOR_COUNT_MIN, DISTRACTOR_COUNT_MAX_EXCL))
    sr, sc, source_r1, source_c1 = source_bbox
    for _ in range(target_distractors):
        candidates = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if int(inp[r, c]) != BG_COLOR:
                    continue
                if source_r1 >= r >= sr and source_c1 >= c >= sc:
                    continue
                if (r, c) in expected_fill_cells:
                    continue
                if _gen_candidate_safe(inp, r, c, MARKER_COLOR, allowed_matches, by_cell):
                    candidates.append((r, c))
        if not candidates:
            return False
        r, c = candidates[int(np.random.randint(0, len(candidates)))]
        inp[r, c] = MARKER_COLOR
        out[r, c] = MARKER_COLOR
    return True


def _gen_attempt() -> np.ndarray | None:
    marker_mask = _gen_source_mask()
    placements, by_cell = _gen_match_index(marker_mask)

    out = np.full((GRID_SIZE, GRID_SIZE), BG_COLOR, dtype=int)
    h, w = map(int, marker_mask.shape)
    source_patch = np.where(marker_mask, MARKER_COLOR, FILL_COLOR)
    sr = int(np.random.randint(0, GRID_SIZE + 1 - h))
    sc = int(np.random.randint(0, GRID_SIZE + 1 - w))
    paste(out, source_patch, sr, sc)
    occupied = frozenset((sr + rr, sc + cc) for rr in range(h) for cc in range(w))

    chosen = _gen_place_matches(out, placements, occupied)
    if chosen is None:
        return None

    inp, expected_fills, expected_fill_cells = _gen_remove_match_fills(out, chosen)
    if not _gen_solver_matches(inp, out):
        return None

    allowed_matches = _gen_valid_match_keys(inp, MARKER_COLOR, placements)
    if {fill_set for _, fill_set in allowed_matches} != expected_fills:
        return None
    if len(allowed_matches) != len(expected_fills):
        return None

    source_bbox = (sr, sc, sr + h - 1, sc + w - 1)
    if not _gen_add_distractors(inp, out, source_bbox, expected_fill_cells, allowed_matches, by_cell):
        return None
    if not _gen_solver_matches(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 7df24a62 input")
