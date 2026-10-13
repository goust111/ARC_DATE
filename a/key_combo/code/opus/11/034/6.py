import itertools
import random

import numpy as np


BG = 0
MARKER_SET = (1, 2, 3, 4)
EXTRA_CHOICES = (5, 6, 7, 8, 9)
QUADRANT_SIZES = (4, 5)
GEN_RETRIES = 180
QUADRANT_RETRIES = 48
SIZE5_PROB = 0.28

_INDEX_GRID_CACHE: dict[int, list[np.ndarray]] = {}


def _candidate_index_grids(n: int) -> list[np.ndarray]:
    cached = _INDEX_GRID_CACHE.get(n)
    if cached is not None:
        return cached

    seen: set[bytes] = set()
    out: list[np.ndarray] = []
    perms = list(itertools.permutations(range(n)))
    cols = np.arange(n, dtype=int)
    for row_perm in perms:
        row_arr = np.array(row_perm, dtype=int)
        base = (row_arr[:, None] + cols[None, :]) % n
        for transpose in (False, True):
            grid = base.T.copy() if transpose else base.copy()
            key = grid.tobytes()
            if key not in seen:
                seen.add(key)
                out.append(grid)

    _INDEX_GRID_CACHE[n] = out
    return out


def _center_and_size(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    assert h == w, "expected a square grid"
    assert h >= 10 and h <= 30, "grid side out of ARC bounds"
    assert (h - 2) % 2 == 0, "expected four equal quadrants around a 2x2 center block"
    n = (h - 2) // 2
    assert n in QUADRANT_SIZES, "unsupported quadrant size"
    return n, n, n


def _marker_palette(marker: np.ndarray, n: int, grid: np.ndarray) -> tuple[int, ...]:
    flat_marker = tuple(int(v) for v in marker.flatten())
    assert len(set(flat_marker)) == 4, "marker block must use four distinct colors"
    assert set(flat_marker) == set(MARKER_SET), "marker block must be {1,2,3,4}"
    if n == 4:
        return flat_marker

    extras = sorted(set(int(v) for v in np.unique(grid)) - {BG, *MARKER_SET})
    assert len(extras) == 1, "5x5 instances must use exactly one extra non-marker color"
    return flat_marker + (extras[0],)


def _quadrant_slices(n: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(0, n), slice(0, n)),
        "tr": (slice(0, n), slice(n + 2, 2 * n + 2)),
        "bl": (slice(n + 2, 2 * n + 2), slice(0, n)),
        "br": (slice(n + 2, 2 * n + 2), slice(n + 2, 2 * n + 2)),
    }


def _anchor_by_quadrant(marker: np.ndarray) -> dict[str, int]:
    return {
        "br": int(marker[0, 0]),
        "bl": int(marker[0, 1]),
        "tr": int(marker[1, 0]),
        "tl": int(marker[1, 1]),
    }


def _apply_color_map(grid: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    lut = np.arange(10, dtype=int)
    for src, dst in color_map.items():
        lut[int(src)] = int(dst)
    return lut[grid]


def _candidate_from_indices(indices: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    arr = np.array(palette, dtype=int)
    return arr[indices]


def _partial_symbol_grid(partial: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    index_of = {int(color): idx for idx, color in enumerate(palette)}
    out = np.full(partial.shape, -1, dtype=int)
    for r, c in np.argwhere(partial != BG):
        out[r, c] = index_of[int(partial[r, c])]
    return out


def _compatible_with_partial_indices(
    partial_symbols: np.ndarray, candidate_indices: np.ndarray, anchor_index: int | None
) -> bool:
    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    if anchor_index is not None:
        mapping[anchor_index] = anchor_index
        inverse[anchor_index] = anchor_index

    for r, c in np.argwhere(partial_symbols >= 0):
        src = int(candidate_indices[r, c])
        dst = int(partial_symbols[r, c])
        mapped = mapping.get(src)
        if mapped is not None and mapped != dst:
            return False
        back = inverse.get(dst)
        if back is not None and back != src:
            return False
        mapping[src] = dst
        inverse[dst] = src
    return True


def _count_matching_candidates(
    partial: np.ndarray, palette: tuple[int, ...], anchor: int | None, stop_after: int
) -> tuple[int, np.ndarray | None]:
    partial_symbols = _partial_symbol_grid(partial, palette)
    anchor_index = None if anchor is None else palette.index(anchor)
    count = 0
    last: np.ndarray | None = None
    for indices in _candidate_index_grids(len(palette)):
        if _compatible_with_partial_indices(partial_symbols, indices, anchor_index):
            count += 1
            last = indices
            if count >= stop_after:
                return count, last
    return count, last


def _require_unique_candidate(partial: np.ndarray, palette: tuple[int, ...], anchor: int) -> np.ndarray:
    count, indices = _count_matching_candidates(partial, palette, anchor, stop_after=2)
    assert count == 1 and indices is not None, f"expected a unique anchored completion, got {count}"
    return _candidate_from_indices(indices, palette)


def _row_col_nondup(partial: np.ndarray) -> bool:
    n = partial.shape[0]
    for i in range(n):
        row = [int(v) for v in partial[i] if v != BG]
        col = [int(v) for v in partial[:, i] if v != BG]
        if len(row) != len(set(row)) or len(col) != len(set(col)):
            return False
    return True


def _visible_per_line(partial: np.ndarray) -> bool:
    return np.all(np.count_nonzero(partial != BG, axis=0) >= 1) and np.all(
        np.count_nonzero(partial != BG, axis=1) >= 1
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each quadrant's anchored color relabeling and complete the unique Latin square.

    Input structure:
    - A square grid is split into four equal corner quadrants by a central 2x2 marker block.
    - The marker block uses the four distinct colors {1,2,3,4}; each quadrant is a partially filled n x n color table with n in {4,5}.
    - Within a quadrant, visible non-black cells may be a locally relabeled version of the final Latin square, but the anchor color named by the diagonally opposite marker cell already uses its correct label.

    Rule:
    - Keep the central 2x2 marker block unchanged.
    - Map marker colors diagonally: marker TL -> quadrant BR, TR -> BL, BL -> TR, BR -> TL.
    - For each quadrant, find the unique Latin-square completion over that quadrant's palette such that every visible cell can be explained by one consistent within-quadrant color permutation, and the mapped anchor color is fixed by that permutation.
    - Replace the whole quadrant by that decoded Latin square.

    Key insight:
    - The center block does not directly say what to paint; it tells which color in each opposite quadrant is already trustworthy, and that anchor breaks the otherwise ambiguous local color relabeling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    n, r0, c0 = _center_and_size(g)
    marker = g[r0 : r0 + 2, c0 : c0 + 2]
    palette = _marker_palette(marker, n, g)
    anchors = _anchor_by_quadrant(marker)
    slices = _quadrant_slices(n)

    out = g.copy()
    for name, (rs, cs) in slices.items():
        partial = g[rs, cs]
        solved = _require_unique_candidate(partial, palette, anchors[name])
        out[rs, cs] = solved

    return out


# --------------- generator ---------------


def _random_marker() -> np.ndarray:
    vals = list(MARKER_SET)
    random.shuffle(vals)
    return np.array(vals, dtype=int).reshape(2, 2)


def _random_output_quadrant(palette: tuple[int, ...]) -> np.ndarray:
    indices = random.choice(_candidate_index_grids(len(palette)))
    return _candidate_from_indices(indices, palette)


def _random_color_map(palette: tuple[int, ...], anchor: int) -> dict[int, int]:
    others = [c for c in palette if c != anchor]
    shuffled = others[:]
    random.shuffle(shuffled)
    use_identity = random.random() < 0.2
    if not use_identity and len(others) > 1 and shuffled == others:
        shuffled = others[1:] + others[:1]
    mapping = {anchor: anchor}
    for src, dst in zip(others, shuffled):
        mapping[int(src)] = int(dst)
    return mapping


def _blanking_order(
    encoded: np.ndarray, output: np.ndarray, anchor: int, must_keep_color: int | None
) -> list[tuple[int, int]]:
    anchor_cells = [
        (int(r), int(c))
        for r, c in np.argwhere(output == anchor)
    ]
    special_cells = []
    if must_keep_color is not None:
        special_cells = [
            (int(r), int(c))
            for r, c in np.argwhere(encoded == must_keep_color)
        ]
    other_cells = [
        (r, c)
        for r in range(encoded.shape[0])
        for c in range(encoded.shape[1])
        if (r, c) not in anchor_cells and (r, c) not in special_cells
    ]
    random.shuffle(anchor_cells)
    random.shuffle(special_cells)
    random.shuffle(other_cells)
    return other_cells + special_cells + anchor_cells


def _anchorless_ambiguous(partial: np.ndarray, palette: tuple[int, ...]) -> bool:
    count, _ = _count_matching_candidates(partial, palette, None, stop_after=2)
    return count > 1


def _valid_partial(
    partial: np.ndarray,
    palette: tuple[int, ...],
    anchor: int,
    min_anchor_visible: int,
    must_keep_color: int | None,
) -> bool:
    if np.count_nonzero(partial == anchor) < min_anchor_visible:
        return False
    if must_keep_color is not None and np.count_nonzero(partial == must_keep_color) == 0:
        return False
    if not _row_col_nondup(partial):
        return False
    if not _visible_per_line(partial):
        return False
    anchored_count, _ = _count_matching_candidates(partial, palette, anchor, stop_after=2)
    if anchored_count != 1:
        return False
    if not _anchorless_ambiguous(partial, palette):
        return False
    return True


def _make_quadrant_input(
    output: np.ndarray,
    palette: tuple[int, ...],
    anchor: int,
    must_keep_color: int | None,
) -> np.ndarray | None:
    n = output.shape[0]
    min_anchor_visible = 1 if n == 4 else 2

    for _ in range(QUADRANT_RETRIES):
        encoded = _apply_color_map(output, _random_color_map(palette, anchor))
        if must_keep_color is not None and np.count_nonzero(encoded == must_keep_color) == 0:
            continue

        partial = encoded.copy()
        removed = 0
        for r, c in _blanking_order(encoded, output, anchor, must_keep_color):
            trial = partial.copy()
            trial[r, c] = BG
            if _valid_partial(trial, palette, anchor, min_anchor_visible, must_keep_color):
                partial = trial
                removed += 1

        if removed >= max(3, n - 1) and _valid_partial(
            partial, palette, anchor, min_anchor_visible, must_keep_color
        ):
            return partial
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n = 5 if random.random() < SIZE5_PROB else 4
    marker = _random_marker()
    extra = None
    if n == 5:
        extra = random.choice(EXTRA_CHOICES)
    palette = tuple(int(v) for v in marker.flatten()) + (() if extra is None else (extra,))
    anchors = _anchor_by_quadrant(marker)
    slices = _quadrant_slices(n)
    side = 2 * n + 2

    output = np.zeros((side, side), dtype=int)
    output[n : n + 2, n : n + 2] = marker

    first_extra_quadrant = None
    if extra is not None:
        first_extra_quadrant = random.choice(tuple(slices.keys()))

    quadrant_inputs: dict[str, np.ndarray] = {}
    for name, (rs, cs) in slices.items():
        quadrant_output = _random_output_quadrant(palette)
        must_keep_color = extra if name == first_extra_quadrant else None
        quadrant_input = _make_quadrant_input(
            quadrant_output, palette, anchors[name], must_keep_color
        )
        if quadrant_input is None:
            return None
        output[rs, cs] = quadrant_output
        quadrant_inputs[name] = quadrant_input

    inp = np.zeros_like(output)
    inp[n : n + 2, n : n + 2] = marker

    for name, (rs, cs) in slices.items():
        inp[rs, cs] = quadrant_inputs[name]

    if extra is not None and np.count_nonzero(inp == extra) == 0:
        return None

    if not np.array_equal(main(inp), output):
        return None

    for name, (rs, cs) in slices.items():
        partial = inp[rs, cs]
        anchor = anchors[name]
        anchored_count, _ = _count_matching_candidates(partial, palette, anchor, stop_after=2)
        if anchored_count != 1:
            return None
        if not _anchorless_ambiguous(partial, palette):
            return None

    return inp, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, _ = built
        return inp
    raise RuntimeError("failed to generate a non-degenerate anchored Latin-square input")
