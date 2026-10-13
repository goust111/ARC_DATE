from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
PALETTE = (2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATE_ATTEMPTS = 400


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if not indices:
        return runs
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == int(color), dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _unique_majority(block: np.ndarray) -> int:
    counts = Counter(map(int, block.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "block must be non-empty"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "block majority must be unique"
    return int(ranked[0][0])


def _find_run_index(runs: list[tuple[int, int]], value: int) -> int:
    for idx, (start, end) in enumerate(runs):
        if start <= int(value) <= end:
            return idx
    raise AssertionError(f"value {value} not inside any run")


def _detect_lattice(grid: np.ndarray) -> dict[str, Any]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size, "expected grey lattice separators"
    r0, c0 = map(int, grey_coords.min(axis=0))
    r1, c1 = map(int, grey_coords.max(axis=0))

    sep_rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == GREY)]
    sep_cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == GREY)]
    assert sep_rows[0] == r0 and sep_rows[-1] == r1
    assert sep_cols[0] == c0 and sep_cols[-1] == c1

    row_runs = _runs([r for r in range(r0, r1 + 1) if r not in set(sep_rows)])
    col_runs = _runs([c for c in range(c0, c1 + 1) if c not in set(sep_cols)])
    assert 2 <= len(row_runs) <= 3
    assert 3 <= len(col_runs) <= 4

    row_heights = {end - start + 1 for start, end in row_runs}
    col_widths = {end - start + 1 for start, end in col_runs}
    assert len(row_heights) == 1 and len(col_widths) == 1

    return {
        "bbox": (r0, c0, r1, c1),
        "row_runs": row_runs,
        "col_runs": col_runs,
    }


def _frame_infos(grid: np.ndarray) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for comp in _components_of_color(grid, BLUE):
        r0, c0, r1, c1 = _bbox_coords(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4

        gap_cols = []
        for c in range(c0 + 1, c1):
            if int(grid[r0, c]) != BLUE:
                gap_cols.append(c)
        assert len(gap_cols) == 1, "each frame needs exactly one top-edge gap"
        gap_c = gap_cols[0]

        for c in range(c0, c1 + 1):
            expected = BLACK if c == gap_c and c0 < c < c1 else BLUE
            assert int(grid[r0, c]) == expected
            assert int(grid[r1, c]) == BLUE
        for r in range(r0 + 1, r1):
            assert int(grid[r, c0]) == BLUE
            assert int(grid[r, c1]) == BLUE
            assert np.all(grid[r, c0 + 1 : c1] == BLACK)

        frames.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "gap_r": r0, "gap_c": gap_c})

    assert 2 <= len(frames) <= 3
    frames.sort(key=lambda item: (item["c0"], item["r0"]))
    return frames


def _trace_hit(grid: np.ndarray, gap_r: int, gap_c: int) -> tuple[int, int]:
    for r in range(gap_r - 1, -1, -1):
        value = int(grid[r, gap_c])
        if value in {BLACK, BLUE, GREY}:
            continue
        return r, gap_c
    raise AssertionError("ray failed to hit a colored block pixel")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped blue frame with the unique majority color of the lattice block selected by its upward gap ray.

    Input structure:
    - A black background contains a grey-separated lattice of equal-sized colored blocks in the upper region and 2-3 blue hollow frames in the lower region.
    - Each lattice block contains only non-black, non-grey colors and has one unique majority color despite 1-2 noise pixels.
    - Each blue frame is a 1-pixel-thick rectangle whose top edge has exactly one missing non-corner pixel; frame interiors are black.

    Rule:
    - For each frame, locate its top-edge gap and trace a vertical ray upward from the cell immediately above it.
    - Ignore black, grey, and blue cells until the ray reaches the first colored block pixel, then use the surrounding grey separator rows and columns to recover that whole lattice block.
    - Fill the frame interior with that block's unique majority color and recolor the missing top-edge gap blue, while leaving the lattice, separators, and existing blue borders unchanged.

    Key insight:
    - The gap is not a place where color escapes; it is a vertical pointer that selects one noisy lattice block whose true color must be decoded by majority vote before filling the frame and repairing the gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    lattice = _detect_lattice(g)
    row_runs = lattice["row_runs"]
    col_runs = lattice["col_runs"]
    out = g.copy()

    for frame in _frame_infos(g):
        hit_r, hit_c = _trace_hit(g, frame["gap_r"], frame["gap_c"])
        row_idx = _find_run_index(row_runs, hit_r)
        col_idx = _find_run_index(col_runs, hit_c)
        br0, br1 = row_runs[row_idx]
        bc0, bc1 = col_runs[col_idx]
        block = g[br0 : br1 + 1, bc0 : bc1 + 1]
        fill_color = _unique_majority(block)

        out[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]] = fill_color
        out[frame["gap_r"], frame["gap_c"]] = BLUE

    return out


def _band_ranges(lattice_c0: int, col_count: int, block_w: int) -> list[tuple[int, int]]:
    ranges = []
    for idx in range(col_count):
        start = lattice_c0 + 1 + idx * (block_w + 1)
        ranges.append((start, start + block_w - 1))
    return ranges


def _candidate_frames(width: int, band_range: tuple[int, int]) -> list[dict[str, int]]:
    start, end = band_range
    candidates: list[dict[str, int]] = []
    for frame_w in range(4, 7):
        for gap_offset in range(1, frame_w - 1):
            for gap_c in range(start, end + 1):
                left = gap_c - gap_offset
                right = left + frame_w - 1
                if 0 <= left and right < width:
                    candidates.append(
                        {
                            "c0": left,
                            "c1": right,
                            "gap_c": gap_c,
                            "gap_offset": gap_offset,
                            "w": frame_w,
                        }
                    )
    return candidates


def _search_frame_layout(
    width: int,
    band_ranges: list[tuple[int, int]],
    selected_bands: list[int],
    rng: np.random.Generator,
) -> list[dict[str, int]] | None:
    ordered_bands = sorted(selected_bands)
    candidate_lists = []
    for band in ordered_bands:
        candidates = _candidate_frames(width, band_ranges[band])
        rng.shuffle(candidates)
        candidate_lists.append(candidates)

    chosen: list[dict[str, int]] = []

    def backtrack(idx: int) -> bool:
        if idx == len(ordered_bands):
            return True
        for candidate in candidate_lists[idx]:
            if chosen and candidate["c0"] < chosen[-1]["c1"] + 3:
                continue
            chosen.append({"band": ordered_bands[idx], **candidate})
            if backtrack(idx + 1):
                return True
            chosen.pop()
        return False

    if not backtrack(0):
        return None
    return chosen


def _other_color(rng: np.random.Generator, forbidden: int) -> int:
    options = [color for color in PALETTE if color != int(forbidden)]
    return int(rng.choice(options))


def _expected_from_meta(grid: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    out = grid.copy()
    for frame in meta["frames"]:
        out[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]] = frame["fill_color"]
        out[frame["r0"], frame["gap_c"]] = BLUE
    return out


def _validate_generated(grid: np.ndarray, meta: dict[str, Any]) -> None:
    expected = _expected_from_meta(grid, meta)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output disagrees with construction metadata"

    lattice = _detect_lattice(grid)
    row_runs = lattice["row_runs"]
    col_runs = lattice["col_runs"]
    frames = _frame_infos(grid)
    assert len(frames) == len(meta["frames"])

    hit_bands = []
    misleading = 0
    for frame in frames:
        hit_r, hit_c = _trace_hit(grid, frame["gap_r"], frame["gap_c"])
        row_idx = _find_run_index(row_runs, hit_r)
        col_idx = _find_run_index(col_runs, hit_c)
        hit_bands.append(col_idx)
        br0, br1 = row_runs[row_idx]
        bc0, bc1 = col_runs[col_idx]
        block = grid[br0 : br1 + 1, bc0 : bc1 + 1]
        majority = _unique_majority(block)
        if int(grid[hit_r, hit_c]) != majority:
            misleading += 1

        assert row_idx == len(row_runs) - 1, "upward ray should first enter the bottom lattice row band"
        assert np.all(grid[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]] == BLACK)
        assert np.all(actual[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]] == majority)
        assert int(actual[frame["r0"], frame["gap_c"]]) == BLUE

    assert len(set(hit_bands)) == len(frames), "different frames must target different column bands"
    assert set(hit_bands) == {frame["band"] for frame in meta["frames"]}
    assert misleading >= 1, "generator must include counterexamples to first-hit-color shortcut"


def _attempt_generate(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]] | None:
    row_count = int(rng.integers(2, 4))
    col_count = int(rng.integers(3, 5))
    frame_count = int(rng.integers(2, min(3, col_count) + 1))
    block_h = int(rng.integers(3, 5))
    block_w = int(rng.integers(4, 7))

    lattice_h = row_count * block_h + row_count + 1
    lattice_w = col_count * block_w + col_count + 1
    if lattice_w > 30:
        return None

    width = lattice_w + int(rng.integers(0, min(30 - lattice_w, 4) + 1))
    lattice_c0 = int(rng.integers(0, width - lattice_w + 1))
    lattice_r0 = 1
    band_ranges = _band_ranges(lattice_c0, col_count, block_w)

    selected_bands = sorted(map(int, rng.choice(col_count, size=frame_count, replace=False)))
    layout = _search_frame_layout(width, band_ranges, selected_bands, rng)
    if layout is None:
        return None

    heights = {frame["band"]: int(rng.integers(4, 7)) for frame in layout}
    vertical_gap = int(rng.integers(2, 5))
    frame_top = lattice_r0 + lattice_h + vertical_gap
    height = frame_top + max(heights.values()) + int(rng.integers(1, 3))
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    lattice_r1 = lattice_r0 + lattice_h - 1
    lattice_c1 = lattice_c0 + lattice_w - 1
    grid[lattice_r0 : lattice_r1 + 1, lattice_c0 : lattice_c1 + 1] = GREY

    row_runs = []
    for row_idx in range(row_count):
        start = lattice_r0 + 1 + row_idx * (block_h + 1)
        row_runs.append((start, start + block_h - 1))
    col_runs = []
    for col_idx in range(col_count):
        start = lattice_c0 + 1 + col_idx * (block_w + 1)
        col_runs.append((start, start + block_w - 1))

    targeted_colors = list(map(int, rng.choice(PALETTE, size=frame_count, replace=False)))
    majorities = [[0 for _ in range(col_count)] for _ in range(row_count)]
    for row_idx in range(row_count):
        for col_idx in range(col_count):
            majorities[row_idx][col_idx] = int(rng.choice(PALETTE))
    for band, color in zip(selected_bands, targeted_colors):
        majorities[row_count - 1][band] = int(color)

    mislead_count = int(rng.integers(1, frame_count))
    misleading_bands = set(map(int, rng.choice(selected_bands, size=mislead_count, replace=False)))
    frame_by_band = {frame["band"]: frame for frame in layout}

    for row_idx, (r0, r1) in enumerate(row_runs):
        for col_idx, (c0, c1) in enumerate(col_runs):
            majority = int(majorities[row_idx][col_idx])
            patch = np.full((block_h, block_w), majority, dtype=int)
            forced_noise: dict[tuple[int, int], int] = {}

            if row_idx == row_count - 1 and col_idx in frame_by_band and col_idx in misleading_bands:
                gap_c = frame_by_band[col_idx]["gap_c"]
                forced_pos = (block_h - 1, gap_c - c0)
                forced_noise[forced_pos] = _other_color(rng, majority)

            noise_total = max(len(forced_noise), int(rng.integers(1, 3)))
            while len(forced_noise) < noise_total:
                pos = (int(rng.integers(0, block_h)), int(rng.integers(0, block_w)))
                if pos in forced_noise:
                    continue
                forced_noise[pos] = _other_color(rng, majority)

            for (rr, cc), color in forced_noise.items():
                patch[rr, cc] = color

            grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    meta_frames = []
    for frame in layout:
        h = heights[frame["band"]]
        r0 = frame_top
        r1 = r0 + h - 1
        c0 = frame["c0"]
        c1 = frame["c1"]
        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
        grid[r0, frame["gap_c"]] = BLACK

        fill_color = int(majorities[row_count - 1][frame["band"]])
        meta_frames.append(
            {
                "band": frame["band"],
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "gap_c": frame["gap_c"],
                "fill_color": fill_color,
            }
        )

    if len({frame["fill_color"] for frame in meta_frames}) < min(2, len(meta_frames)):
        return None

    meta = {
        "frames": meta_frames,
        "lattice": {"row_runs": row_runs, "col_runs": col_runs},
    }
    _validate_generated(grid, meta)
    return grid, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        attempt = _attempt_generate(rng)
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input for puzzle 2")
