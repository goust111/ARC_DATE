import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

INPUT_SHAPE = (8, 8)
PATTERN_SIZE = 3
MARKER_COL = INPUT_SHAPE[1] - 1
VALID_MARKER_ROWS = (0, 1, 2, 3)
MIN_FOREGROUND = 2
MAX_FOREGROUND = 7

_RNG = random.Random()
_GEN_COUNTER = 0
_CASE_CACHE = None


def _resize_nn(mask: np.ndarray, size: int) -> np.ndarray:
    src_h, src_w = mask.shape
    row_idx = np.clip(
        np.rint((np.arange(size) + 0.5) * src_h / size - 0.5).astype(int), 0, src_h - 1
    )
    col_idx = np.clip(
        np.rint((np.arange(size) + 0.5) * src_w / size - 0.5).astype(int), 0, src_w - 1
    )
    return mask[np.ix_(row_idx, col_idx)]


def _connected_components_4(mask: np.ndarray) -> int:
    seen = set()
    count = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or (r, c) in seen:
                continue
            count += 1
            stack = [(r, c)]
            seen.add((r, c))
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                        if mask[nr, nc] and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
    return count


def _output_from_mask(mask: np.ndarray, marker_row: int) -> np.ndarray:
    scale = marker_row + 1
    tile = _resize_nn(1 - mask, scale).astype(int)
    out = np.zeros((PATTERN_SIZE * scale, PATTERN_SIZE * scale), dtype=int)
    for r in range(PATTERN_SIZE):
        for c in range(PATTERN_SIZE):
            if mask[r, c]:
                r0 = r * scale
                c0 = c * scale
                out[r0 : r0 + scale, c0 : c0 + scale] = tile * BLUE
    return out


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.ravel())


def _output_signature(out: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, out.shape)), out.tobytes()


def _build_case_cache():
    all_cases = []
    outputs = {}
    for bits in itertools.product((0, 1), repeat=PATTERN_SIZE * PATTERN_SIZE):
        mask = np.array(bits, dtype=int).reshape(PATTERN_SIZE, PATTERN_SIZE)
        fg_count = int(mask.sum())
        if not (MIN_FOREGROUND <= fg_count <= MAX_FOREGROUND):
            continue
        connected = _connected_components_4(mask.astype(bool)) == 1
        for marker_row in VALID_MARKER_ROWS:
            out = _output_from_mask(mask, marker_row)
            signature = _output_signature(out)
            case = {
                "marker_row": marker_row,
                "mask": mask,
                "mask_key": _mask_key(mask),
                "fg_count": fg_count,
                "connected": connected,
                "signature": signature,
            }
            all_cases.append(case)
            outputs.setdefault(signature, []).append(case)

    unique_cases = [case for case in all_cases if len(outputs[case["signature"]]) == 1]
    pools = {}
    for case in unique_cases:
        key = (case["marker_row"], case["fg_count"], case["connected"])
        pools.setdefault(key, []).append(case)
    return {"unique_cases": unique_cases, "pools": pools, "outputs": outputs}


def _get_case_cache():
    global _CASE_CACHE
    if _CASE_CACHE is None:
        _CASE_CACHE = _build_case_cache()
    return _CASE_CACHE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge a complement tile at foreground positions using a scale encoded by a red marker row.

    Input structure:
    - The input is an 8x8 black background grid.
    - The top-left 3x3 area contains a blue (1) foreground mask on black (0).
    - The rightmost column contains exactly one red (2) marker pixel whose row encodes the scale.

    Rule:
    - Read the red marker row r and set the scale S = r + 1.
    - Take the 3x3 blue mask M from the top-left corner and build its complement mask 1-M.
    - Resize that complement mask to SxS by nearest-neighbor sampling, then use it as the tile for every blue source cell.
    - Produce a (3*S) x (3*S) output where each blue cell in M stamps that SxS tile into its corresponding block and each black cell leaves its block black.

    Key insight:
    - The red marker does not mark a destination; its row index is a control code that changes both the output size and the local tile painted at every foreground position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE, f"expected {INPUT_SHAPE} grid, got {g.shape}"

    allowed_colors = {BLACK, BLUE, RED}
    assert set(map(int, np.unique(g))) <= allowed_colors, "unexpected colors present"
    assert np.all(g[:, :-1][3:, :] == BLACK), "only the top-left 3x3 may contain blue pixels"
    assert np.all(g[:3, 3:-1] == BLACK), "cells outside the top-left 3x3 must stay black"
    assert np.all(g[3:, 0:-1] == BLACK), "cells outside the pattern and marker must stay black"
    assert np.count_nonzero(g == RED) == 1, "expected exactly one red marker"
    assert np.count_nonzero(g[:, -1] == RED) == 1, "marker must be on the right edge"
    assert np.all(g[:, -1][g[:, -1] != RED] == BLACK), "right edge may contain only the marker"

    marker_row = int(np.flatnonzero(g[:, -1] == RED)[0])
    assert marker_row in VALID_MARKER_ROWS, "marker row must encode a scale in rows 0..3"

    mask = (g[:PATTERN_SIZE, :PATTERN_SIZE] == BLUE).astype(int)
    assert np.all(g[:PATTERN_SIZE, :PATTERN_SIZE][mask == 0] == BLACK), "pattern uses only blue on black"
    fg_count = int(mask.sum())
    assert MIN_FOREGROUND <= fg_count <= MAX_FOREGROUND, "foreground count must stay within 2..7"

    return _output_from_mask(mask, marker_row)


def _pick_case():
    global _GEN_COUNTER
    cache = _get_case_cache()

    marker_row = VALID_MARKER_ROWS[_GEN_COUNTER % len(VALID_MARKER_ROWS)]
    fg_count_cycle = (2, 4, 6, 7, 3, 5)
    fg_count = fg_count_cycle[(_GEN_COUNTER // len(VALID_MARKER_ROWS)) % len(fg_count_cycle)]
    connected = bool((_GEN_COUNTER // (len(VALID_MARKER_ROWS) * len(fg_count_cycle))) % 2)
    _GEN_COUNTER += 1

    preferred = cache["pools"].get((marker_row, fg_count, connected), [])
    if preferred:
        return _RNG.choice(preferred)

    fallback = [
        case
        for case in cache["unique_cases"]
        if case["marker_row"] == marker_row and case["fg_count"] == fg_count
    ]
    if fallback:
        return _RNG.choice(fallback)

    same_scale = [case for case in cache["unique_cases"] if case["marker_row"] == marker_row]
    assert same_scale, "each scale branch must have valid unique examples"
    return _RNG.choice(same_scale)


def _build_input(mask: np.ndarray, marker_row: int) -> np.ndarray:
    grid = np.zeros(INPUT_SHAPE, dtype=int)
    grid[:PATTERN_SIZE, :PATTERN_SIZE] = mask * BLUE
    grid[marker_row, MARKER_COL] = RED
    return grid


def _semantic_oracle(grid: np.ndarray) -> np.ndarray:
    marker_row = int(np.flatnonzero(grid[:, -1] == RED)[0])
    mask = (grid[:PATTERN_SIZE, :PATTERN_SIZE] == BLUE).astype(int)
    out = _output_from_mask(mask, marker_row)
    scale = marker_row + 1
    tile = _resize_nn(1 - mask, scale).astype(int)
    for r in range(PATTERN_SIZE):
        for c in range(PATTERN_SIZE):
            block = out[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            if mask[r, c]:
                assert np.array_equal(block, tile), "foreground block must equal the resized complement tile"
            else:
                assert np.count_nonzero(block) == 0, "background block must remain black"
    return out


def generate_input() -> np.ndarray:
    for _ in range(200):
        case = _pick_case()
        grid = _build_input(case["mask"], case["marker_row"])
        expected = _semantic_oracle(grid)
        actual = main(grid)
        assert np.array_equal(actual, expected), "solver output must match the semantic oracle"

        cache = _get_case_cache()
        signature = _output_signature(actual)
        matches = cache["outputs"][signature]
        assert len(matches) == 1, "generated example must be uniquely decodable from its output"
        only = matches[0]
        assert only["marker_row"] == case["marker_row"]
        assert only["mask_key"] == case["mask_key"]
        return grid
    raise RuntimeError("failed to generate a valid unique input")
