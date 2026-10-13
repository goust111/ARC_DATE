from __future__ import annotations

import itertools
import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_task_018_4",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components


BLACK = Color.BLACK
GREY = Color.GREY
SEGMENT_COLORS = (Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

MIN_SEGMENTS = 3
MAX_SEGMENTS = 5
MIN_WIDTH = 15
MAX_WIDTH = 24
MIN_BOTTOM_H = 5
MAX_BOTTOM_H = 8
MAX_GENERATION_ATTEMPTS = 400

TARGET_PATH = Path("/home/lixinyu/snap/a/key_combo/code/opus/8/018/4.py")


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected one full-width grey separator row"
    return int(rows[0])


def _extract_segments(top: np.ndarray) -> list[dict]:
    segments = find_components(
        top,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
    )
    assert MIN_SEGMENTS <= len(segments) <= MAX_SEGMENTS, "expected 3-5 top segments"

    seen_colors = set()
    lengths = []
    for comp in segments:
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        area = int(comp["area"])

        assert color not in {BLACK, GREY}, "top segments must use non-black non-grey colors"
        assert r0 == r1, "top component must be one cell high"
        assert area == c1 - c0 + 1, "top component must be a contiguous horizontal segment"
        assert coords.shape[0] == area
        assert color not in seen_colors, "segment colors must be unique"
        seen_colors.add(color)
        lengths.append(area)

    assert len(set(lengths)) == len(lengths), "segment lengths must be unique"
    return segments


def _sorted_color_sequence(segments: list[dict]) -> list[int]:
    ordered = sorted((int(comp["area"]), int(comp["color"])) for comp in segments)
    return [color for _length, color in ordered]


def _periodic_row(colors: list[int], width: int) -> np.ndarray:
    period = len(colors)
    return np.array([colors[c % period] for c in range(width)], dtype=int)


def _compatible_permutation_count(bottom: np.ndarray, colors: list[int]) -> int:
    count = 0
    for perm in itertools.permutations(colors):
        ok = True
        for c in range(bottom.shape[1]):
            visible = bottom[:, c]
            visible = visible[visible != BLACK]
            if visible.size and not np.all(visible == perm[c % len(perm)]):
                ok = False
                break
        if ok:
            count += 1
    return count


def _max_black_run(row: np.ndarray) -> int:
    best = cur = 0
    for v in row:
        if int(v) == BLACK:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the occluded periodic lower panel using the color order implied by sorted top segment lengths.

    Input structure:
    - A black background contains 3-5 scattered horizontal single-color segments in the upper area.
    - A full-width grey row separates those segments from a lower rectangular panel.
    - The lower panel is a horizontally repeating color pattern built from the same segment colors, with some cells occluded to black.

    Rule:
    - Extract every upper horizontal segment and record its color and length.
    - Sort the segments by increasing length to obtain the tile color sequence.
    - Use that sequence as a period-N horizontal tile indexed by the global column: color at column c is sequence[c mod N].
    - In the lower panel, keep visible non-black cells unchanged and replace each black occluded cell with the tile color for its column.
    - Preserve the upper area and the grey separator row.

    Key insight:
    - The visible lower pixels do not uniquely determine the color order; the missing tile order must be reconstructed from the top segments by sorting them by length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    assert 1 <= sep <= g.shape[0] - 2, "separator must leave content above and below"

    segments = _extract_segments(g[:sep])
    colors = _sorted_color_sequence(segments)
    period = len(colors)

    out = g.copy()
    expected = _periodic_row(colors, g.shape[1])
    bottom = g[sep + 1 :]
    assert bottom.size > 0, "expected a lower panel"
    assert np.any(bottom == BLACK), "expected at least one occluded lower-panel cell"

    for r in range(sep + 1, g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] == BLACK:
                out[r, c] = expected[c]
            else:
                assert int(g[r, c]) == int(expected[c]), "visible lower pixels must already match the inferred tile"

    assert period == len(set(colors)), "tile colors must be unique"
    return out


def _sample_segments(rng: random.Random, top_h: int, width: int, n_segments: int) -> list[tuple[int, int, int, int]]:
    max_len = min(5, width - 4)
    lengths = rng.sample(range(1, max_len + 1), n_segments)
    colors = rng.sample(list(SEGMENT_COLORS), n_segments)
    rows = sorted(rng.sample(range(top_h), n_segments))

    starts = []
    for length in lengths:
        starts.append(rng.randint(0, width - length))

    segs = [(rows[i], starts[i], lengths[i], colors[i]) for i in range(n_segments)]
    by_len = [color for _r, _c, _l, color in sorted(segs, key=lambda x: x[2])]
    by_row = [color for _r, _c, _l, color in sorted(segs, key=lambda x: x[0])]
    by_col = [color for _r, _c, _l, color in sorted(segs, key=lambda x: x[1])]

    if by_len == by_row or by_len == by_row[::-1]:
        raise ValueError("row order leaked the length order")
    if by_len == by_col or by_len == by_col[::-1]:
        raise ValueError("column order leaked the length order")
    return segs


def _build_mask(
    rng: random.Random,
    bottom_h: int,
    width: int,
    period: int,
    hidden_mods: set[int],
) -> np.ndarray:
    mask = np.zeros((bottom_h, width), dtype=bool)
    for c in range(width):
        if c % period in hidden_mods:
            mask[:, c] = True

    for r in range(bottom_h):
        run_len = rng.randint(period, min(width - 1, period + max(1, period)))
        start = rng.randint(0, width - run_len)
        mask[r, start : start + run_len] = True

        for _ in range(rng.randint(0, 2)):
            c = rng.randrange(width)
            mask[r, c] = True

        if np.all(mask[r]):
            keep = rng.choice([c for c in range(width) if c % period not in hidden_mods])
            mask[r, keep] = False

    return mask


def _segment_sequence_from_input(grid: np.ndarray) -> list[int]:
    sep = _separator_row(grid)
    return _sorted_color_sequence(_extract_segments(grid[:sep]))


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    sep = _separator_row(inp)
    colors = _segment_sequence_from_input(inp)
    period = len(colors)
    expected_row = _periodic_row(colors, inp.shape[1])
    bottom_in = inp[sep + 1 :]
    bottom_out = out[sep + 1 :]

    assert np.array_equal(bottom_out, np.tile(expected_row, (bottom_out.shape[0], 1)))
    assert _compatible_permutation_count(bottom_in, colors) >= 2
    assert np.count_nonzero(bottom_in == BLACK) > 0
    assert np.count_nonzero(bottom_in != BLACK) >= bottom_in.size // 6
    assert np.count_nonzero(bottom_in != BLACK) <= bottom_in.size * 3 // 4

    visible_residues = {c % period for c in range(inp.shape[1]) if np.any(bottom_in[:, c] != BLACK)}
    assert len(visible_residues) <= period - 2, "lower panel alone must leave at least two residue classes unresolved"

    for row in bottom_in:
        assert _max_black_run(row) >= period


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_segments = rng.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    width = rng.randint(max(MIN_WIDTH, n_segments * 4), min(MAX_WIDTH, 24))
    top_h = rng.randint(n_segments, n_segments + 2)
    bottom_h = rng.randint(MIN_BOTTOM_H, MAX_BOTTOM_H)
    height = top_h + 1 + bottom_h
    if height > 30 or width > 30:
        return None

    try:
        segments = _sample_segments(rng, top_h, width, n_segments)
    except ValueError:
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, c0, length, color in segments:
        grid[r, c0 : c0 + length] = color
    grid[top_h, :] = GREY

    seq = [color for _r, _c, _length, color in sorted(segments, key=lambda x: x[2])]
    base_row = _periodic_row(seq, width)
    panel = np.tile(base_row, (bottom_h, 1))

    hidden_count = 2 if n_segments == 3 else rng.randint(2, min(3, n_segments - 1))
    hidden_mods = set(rng.sample(range(n_segments), hidden_count))

    for _ in range(80):
        mask = _build_mask(rng, bottom_h, width, n_segments, hidden_mods)
        visible = panel.copy()
        visible[mask] = BLACK

        if not np.any(visible != BLACK):
            continue
        if _compatible_permutation_count(visible, seq) < 2:
            continue

        grid[top_h + 1 :] = visible
        try:
            out = main(grid)
            _semantic_checks(grid, out)
        except AssertionError:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 018/4")


if __name__ == "__main__":
    sample = generate_input()
    result = main(sample)
    print(TARGET_PATH)
    print(sample)
    print()
    print(result)
