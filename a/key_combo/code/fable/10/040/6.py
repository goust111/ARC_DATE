from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
components_mask = _ARC_UTILS.components_mask

BLACK = Color.BLACK
RED = Color.RED
YELLOW = Color.YELLOW
TEAL = Color.TEAL

GRID_HEIGHT_RANGE = (12, 23)
GRID_WIDTH_RANGE = (12, 23)
BAR_LENGTH_RANGE = (4, 9)
MAX_MARKERS_PER_SIDE = 3
GENERATION_RETRIES = 400


def _contiguous_run(indices: np.ndarray) -> tuple[int, int]:
    idx = np.asarray(indices, dtype=int)
    assert idx.size > 0, "expected a non-empty run"
    assert np.all(np.diff(idx) == 1), "edge bar must be contiguous"
    return int(idx[0]), int(idx[-1])


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)
    assert H <= 30 and W <= 30, "ARC grids must fit inside 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, RED, TEAL}, "unexpected input colors"

    top_red = np.flatnonzero(g[0] == RED)
    left_red = np.flatnonzero(g[:, 0] == RED)
    assert top_red.size >= 2 and left_red.size >= 2, "expected one top bar and one left bar"

    h_c0, h_c1 = _contiguous_run(top_red)
    v_r0, v_r1 = _contiguous_run(left_red)
    bar_len = h_c1 - h_c0 + 1

    assert h_c0 > 0 and h_c1 < W - 1, "top bar must stay away from corners"
    assert v_r0 > 0 and v_r1 < H - 1, "left bar must stay away from corners"
    assert bar_len == v_r1 - v_r0 + 1, "bars must have equal length"

    red_mask = g == RED
    expected_red = np.zeros_like(red_mask, dtype=bool)
    expected_red[0, h_c0 : h_c1 + 1] = True
    expected_red[v_r0 : v_r1 + 1, 0] = True
    assert np.array_equal(red_mask, expected_red), "red cells must form exactly the two edge bars"

    marker_comps = components_mask(g == TEAL, connectivity=4)
    assert marker_comps, "expected at least one teal marker"
    markers = []
    used_top_offsets: set[int] = set()
    used_left_offsets: set[int] = set()

    for comp in marker_comps:
        assert comp.shape == (1, 2), "markers must be isolated single pixels"
        r, c = map(int, comp[0])
        top_aligned = h_c0 <= c <= h_c1 and r > 0
        left_aligned = v_r0 <= r <= v_r1 and c > 0
        assert top_aligned ^ left_aligned, "each marker must align with exactly one source bar"

        if top_aligned:
            assert not (v_r0 <= r <= v_r1), "top-aligned markers cannot also overlap the left-bar rows"
            offset = c - h_c0
            assert r >= 2, "top marker must leave room for a visible source segment"
            path = g[1:r, c]
            assert np.all(path == BLACK), "top marker path to the bar must be clear"
            used_top_offsets.add(offset)
            markers.append({"kind": "top", "row": r, "col": c, "offset": offset})
        else:
            assert not (h_c0 <= c <= h_c1), "left-aligned markers cannot also overlap the top-bar columns"
            offset = r - v_r0
            assert c >= 2, "left marker must leave room for a visible source segment"
            path = g[r, 1:c]
            assert np.all(path == BLACK), "left marker path to the bar must be clear"
            used_left_offsets.add(offset)
            markers.append({"kind": "left", "row": r, "col": c, "offset": offset})

    assert len(used_top_offsets) == sum(m["kind"] == "top" for m in markers), "top offsets must be unique"
    assert len(used_left_offsets) == sum(m["kind"] == "left" for m in markers), "left offsets must be unique"

    # Cross-orientation offset collisions would make emitted rays overwrite other markers.
    assert used_top_offsets.isdisjoint(used_left_offsets), "top and left offsets must be disjoint"

    for marker in markers:
        offset = marker["offset"]
        if marker["kind"] == "top":
            target_row = v_r0 + offset
            assert np.all(g[target_row, 1:] == BLACK), "paired horizontal ray row must be empty"
        else:
            target_col = h_c0 + offset
            assert np.all(g[1:, target_col] == BLACK), "paired vertical ray column must be empty"

    return {
        "grid": g,
        "shape": (H, W),
        "bar_len": bar_len,
        "top_bar": (h_c0, h_c1),
        "left_bar": (v_r0, v_r1),
        "markers": markers,
    }


def _spec_output(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_input(grid)
    g = parsed["grid"]
    H, W = parsed["shape"]
    h_c0, h_c1 = parsed["top_bar"]
    v_r0, v_r1 = parsed["left_bar"]
    out = g.copy()

    for marker in parsed["markers"]:
        r = marker["row"]
        c = marker["col"]
        offset = marker["offset"]
        out[r, c] = YELLOW

        if marker["kind"] == "top":
            if r > 1:
                out[1:r, c] = TEAL
            target_row = v_r0 + offset
            out[target_row, 1:W] = TEAL
            out[target_row, 0] = RED
        else:
            if c > 1:
                out[r, 1:c] = TEAL
            target_col = h_c0 + offset
            out[1:H, target_col] = TEAL
            out[0, target_col] = RED

    out[0, h_c0 : h_c1 + 1] = RED
    out[v_r0 : v_r1 + 1, 0] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each teal marker into a right-angle key press that selects the matching index on the perpendicular red edge bar.

    Input structure:
    - The background is black.
    - One solid red bar lies on the top edge and one equal-length solid red bar lies on the left edge; neither touches a corner.
    - Teal marker pixels are isolated singles. Each marker is aligned with exactly one bar: either under a top-bar column or to the right of a left-bar row.

    Rule:
    - A top-aligned marker points upward along its column to the top red bar; recolor that marker yellow and paint the black cells between it and the bar teal.
    - A left-aligned marker points leftward along its row to the left red bar; recolor that marker yellow and paint the black cells between it and the bar teal.
    - Read the hit index within the source bar: left-to-right on the top bar, top-to-bottom on the left bar.
    - The other red bar emits a full teal ray at the same index, away from its edge: the left bar shoots rightward across its row, and the top bar shoots downward along its column. Red bar cells stay red.

    Key insight:
    - The marker does not choose an absolute row or column in the output. It chooses a bar-local index, and that index turns a right angle when transferred to the perpendicular bar.
    """
    return _spec_output(grid)


def _sample_bar_geometry(rng: random.Random) -> tuple[int, int, int, int, int, int, int]:
    H = rng.randint(*GRID_HEIGHT_RANGE)
    W = rng.randint(*GRID_WIDTH_RANGE)
    max_len = min(BAR_LENGTH_RANGE[1], H - 3, W - 3)
    min_len = min(BAR_LENGTH_RANGE[0], max_len)
    assert max_len >= 4
    bar_len = rng.randint(min_len, max_len)
    h_c0 = rng.randint(1, W - bar_len - 2)
    v_r0 = rng.randint(1, H - bar_len - 2)
    h_c1 = h_c0 + bar_len - 1
    v_r1 = v_r0 + bar_len - 1
    return H, W, bar_len, h_c0, h_c1, v_r0, v_r1


def _choose_offsets(
    rng: random.Random,
    bar_len: int,
    mode: str,
) -> tuple[list[int], list[int]]:
    offsets = list(range(bar_len))
    rng.shuffle(offsets)

    if mode == "top":
        k_top = rng.randint(2, min(MAX_MARKERS_PER_SIDE, bar_len))
        return sorted(offsets[:k_top]), []
    if mode == "left":
        k_left = rng.randint(2, min(MAX_MARKERS_PER_SIDE, bar_len))
        return [], sorted(offsets[:k_left])

    total = rng.randint(2, min(MAX_MARKERS_PER_SIDE * 2, bar_len))
    k_top = rng.randint(1, min(MAX_MARKERS_PER_SIDE, total - 1, bar_len - 1))
    k_left = total - k_top
    if k_left < 1 or k_left > MAX_MARKERS_PER_SIDE or k_left > bar_len - k_top:
        return _choose_offsets(rng, bar_len, mode)
    return sorted(offsets[:k_top]), sorted(offsets[k_top : k_top + k_left])


def _place_top_markers(
    grid: np.ndarray,
    rng: random.Random,
    h_c0: int,
    top_offsets: list[int],
    left_span: tuple[int, int],
) -> None:
    H, _W = grid.shape
    v_r0, v_r1 = left_span
    rows = [r for r in range(2, H - 1) if not (v_r0 <= r <= v_r1)]
    assert len(rows) >= len(top_offsets)
    chosen_rows = rng.sample(rows, len(top_offsets))
    rng.shuffle(chosen_rows)
    for offset, row in zip(top_offsets, chosen_rows):
        col = h_c0 + offset
        grid[row, col] = TEAL


def _place_left_markers(
    grid: np.ndarray,
    rng: random.Random,
    v_r0: int,
    left_offsets: list[int],
    top_span: tuple[int, int],
) -> None:
    _H, W = grid.shape
    h_c0, h_c1 = top_span
    cols = [c for c in range(2, W - 1) if not (h_c0 <= c <= h_c1)]
    assert len(cols) >= len(left_offsets)
    chosen_cols = rng.sample(cols, len(left_offsets))
    rng.shuffle(chosen_cols)
    for offset, col in zip(left_offsets, chosen_cols):
        row = v_r0 + offset
        grid[row, col] = TEAL


def _build_input_from_plan(
    H: int,
    W: int,
    bar_len: int,
    h_c0: int,
    h_c1: int,
    v_r0: int,
    v_r1: int,
    top_offsets: list[int],
    left_offsets: list[int],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((H, W), BLACK, dtype=int)
    grid[0, h_c0 : h_c1 + 1] = RED
    grid[v_r0 : v_r1 + 1, 0] = RED
    _place_top_markers(grid, rng, h_c0, top_offsets, (v_r0, v_r1))
    _place_left_markers(grid, rng, v_r0, left_offsets, (h_c0, h_c1))
    return grid


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    parsed = _parse_input(inp)
    expected = _spec_output(inp)
    assert np.array_equal(out, expected), "main() must implement the puzzle rule exactly"

    g = parsed["grid"]
    h_c0, h_c1 = parsed["top_bar"]
    v_r0, v_r1 = parsed["left_bar"]
    changed = out != g
    assert np.any(changed), "output must differ from input"
    assert np.all(out[0, h_c0 : h_c1 + 1] == RED), "top bar must stay red"
    assert np.all(out[v_r0 : v_r1 + 1, 0] == RED), "left bar must stay red"

    for marker in parsed["markers"]:
        r = marker["row"]
        c = marker["col"]
        offset = marker["offset"]
        assert out[r, c] == YELLOW, "every marker must turn yellow"
        if marker["kind"] == "top":
            assert np.all(out[1:r, c] == TEAL), "top marker must connect upward to the bar"
            target_row = v_r0 + offset
            assert np.all(out[target_row, 1:] == TEAL), "left bar must emit a full horizontal teal ray"
        else:
            assert np.all(out[r, 1:c] == TEAL), "left marker must connect leftward to the bar"
            target_col = h_c0 + offset
            assert np.all(out[1:, target_col] == TEAL), "top bar must emit a full vertical teal ray"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H, W, bar_len, h_c0, h_c1, v_r0, v_r1 = _sample_bar_geometry(rng)
    mode = rng.choice(["top", "left", "mixed"])
    top_offsets, left_offsets = _choose_offsets(rng, bar_len, mode)

    try:
        inp = _build_input_from_plan(
            H,
            W,
            bar_len,
            h_c0,
            h_c1,
            v_r0,
            v_r1,
            top_offsets,
            left_offsets,
            rng,
        )
        parsed = _parse_input(inp)
        if mode == "mixed":
            if not any(m["kind"] == "top" for m in parsed["markers"]):
                return None
            if not any(m["kind"] == "left" for m in parsed["markers"]):
                return None
        out = main(inp)
        _validate_semantics(inp, out)
        return inp
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")


if __name__ == "__main__":
    path = Path(__file__)
    grid = generate_input()
    print(path.name, grid.shape)
    print(grid)
