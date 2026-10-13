from __future__ import annotations

import random

import numpy as np


BG = 0
TRACK = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

TRACK_COUNT_OPTIONS = [(1, 1), (2, 1), (1, 2), (2, 2)]
TRACK_LENGTH_RANGE = (5, 10)
SEED_COUNT_RANGE = (1, 4)
GRID_MIN = 12
GRID_MAX = 26
PLACEMENT_PADDING = 1
PLACEMENT_TRIES = 120
GENERATION_TRIES = 400


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _scan_tracks(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    tracks: list[dict[str, int]] = []

    for r in range(h):
        for c0, c1 in _runs(grid[r] != BG):
            if c1 - c0 + 1 < 5:
                continue
            segment = grid[r, c0 : c1 + 1]
            if np.any(segment == TRACK):
                tracks.append({"orientation": "h", "r": int(r), "c0": int(c0), "c1": int(c1)})

    for c in range(w):
        for r0, r1 in _runs(grid[:, c] != BG):
            if r1 - r0 + 1 < 5:
                continue
            segment = grid[r0 : r1 + 1, c]
            if np.any(segment == TRACK):
                tracks.append({"orientation": "v", "c": int(c), "r0": int(r0), "r1": int(r1)})

    assert tracks, "expected at least one track"
    return tracks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed away from its grey guide track, then erase the guide track.

    Input structure:
    - The background is black.
    - Grey pixels form one-cell-thick horizontal or vertical straight tracks of
      length at least five, with colored seed pixels replacing some track cells.
    - Each track contains one or more non-black, non-grey seed pixels, and the
      seeds on the same track have distinct colors.

    Rule:
    - For each horizontal track, read its seeds from left to right; for each
      vertical track, read its seeds from top to bottom.
    - The seed with order k grows a same-colored segment of length k away from
      the track: upward for horizontal tracks and leftward for vertical tracks.
    - The seed pixel itself stays in place, and every remaining grey track pixel
      is erased to black after all growth is drawn.

    Key insight:
    - The grey line is only a temporary scaffold: its orientation sets the growth
      direction, and the seed order along that scaffold encodes each segment's
      length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g <= 9))
    assert int(g[0, 0]) in range(10)

    out = g.copy()
    for track in _scan_tracks(g):
        if track["orientation"] == "h":
            r = track["r"]
            c0 = track["c0"]
            c1 = track["c1"]
            segment = g[r, c0 : c1 + 1]
            seed_offsets = [idx for idx, value in enumerate(segment.tolist()) if value not in (BG, TRACK)]
            assert seed_offsets, "each track needs at least one seed"
            seed_colors = [int(segment[idx]) for idx in seed_offsets]
            assert len(set(seed_colors)) == len(seed_colors), "seed colors on a track must be distinct"
            for rank, offset in enumerate(seed_offsets, start=1):
                col = c0 + offset
                color = int(segment[offset])
                assert r - rank >= 0, "horizontal growth must fit above the track"
                out[r - rank : r, col] = color
            continue

        c = track["c"]
        r0 = track["r0"]
        r1 = track["r1"]
        segment = g[r0 : r1 + 1, c]
        seed_offsets = [idx for idx, value in enumerate(segment.tolist()) if value not in (BG, TRACK)]
        assert seed_offsets, "each track needs at least one seed"
        seed_colors = [int(segment[idx]) for idx in seed_offsets]
        assert len(set(seed_colors)) == len(seed_colors), "seed colors on a track must be distinct"
        for rank, offset in enumerate(seed_offsets, start=1):
            row = r0 + offset
            color = int(segment[offset])
            assert c - rank >= 0, "vertical growth must fit left of the track"
            out[row, c - rank : c] = color

    out[out == TRACK] = BG
    return out


def _dilate(coords: set[tuple[int, int]], h: int, w: int, padding: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _sample_track_shape(orientation: str) -> dict[str, object]:
    length = random.randint(*TRACK_LENGTH_RANGE)
    max_seeds = min(SEED_COUNT_RANGE[1], length - 1)
    seed_count = random.randint(SEED_COUNT_RANGE[0], max_seeds)
    seed_offsets = sorted(random.sample(range(length), seed_count))
    colors = random.sample(COLOR_POOL, seed_count)
    return {
        "orientation": orientation,
        "length": length,
        "seed_offsets": seed_offsets,
        "colors": colors,
    }


def _important_cells(track: dict[str, object], anchor_a: int, anchor_b: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    orientation = str(track["orientation"])
    length = int(track["length"])
    seed_offsets = list(track["seed_offsets"])

    inp: set[tuple[int, int]] = set()
    out: set[tuple[int, int]] = set()

    if orientation == "h":
        r = anchor_a
        c0 = anchor_b
        inp.update((r, c0 + offset) for offset in range(length))
        for rank, offset in enumerate(seed_offsets, start=1):
            c = c0 + offset
            out.add((r, c))
            out.update((r - step, c) for step in range(1, rank + 1))
        return inp, out

    r0 = anchor_a
    c = anchor_b
    inp.update((r0 + offset, c) for offset in range(length))
    for rank, offset in enumerate(seed_offsets, start=1):
        r = r0 + offset
        out.add((r, c))
        out.update((r, c - step) for step in range(1, rank + 1))
    return inp, out


def _paint_track(
    inp: np.ndarray,
    expected: np.ndarray,
    track: dict[str, object],
    anchor_a: int,
    anchor_b: int,
) -> None:
    orientation = str(track["orientation"])
    length = int(track["length"])
    seed_offsets = list(track["seed_offsets"])
    colors = list(track["colors"])

    if orientation == "h":
        r = anchor_a
        c0 = anchor_b
        inp[r, c0 : c0 + length] = TRACK
        for rank, (offset, color) in enumerate(zip(seed_offsets, colors), start=1):
            c = c0 + offset
            inp[r, c] = int(color)
            expected[r, c] = int(color)
            expected[r - rank : r, c] = int(color)
        return

    r0 = anchor_a
    c = anchor_b
    inp[r0 : r0 + length, c] = TRACK
    for rank, (offset, color) in enumerate(zip(seed_offsets, colors), start=1):
        r = r0 + offset
        inp[r, c] = int(color)
        expected[r, c] = int(color)
        expected[r, c - rank : c] = int(color)


def _place_track(
    inp: np.ndarray,
    expected: np.ndarray,
    reserved: set[tuple[int, int]],
    track: dict[str, object],
) -> bool:
    h, w = inp.shape
    orientation = str(track["orientation"])
    length = int(track["length"])
    seed_count = len(track["seed_offsets"])

    for _ in range(PLACEMENT_TRIES):
        if orientation == "h":
            row = random.randint(seed_count, h - 1)
            col = random.randint(0, w - length)
            inp_cells, out_cells = _important_cells(track, row, col)
        else:
            row = random.randint(0, h - length)
            col = random.randint(seed_count, w - 1)
            inp_cells, out_cells = _important_cells(track, row, col)

        all_cells = inp_cells | out_cells
        if any(cell in reserved for cell in all_cells):
            continue

        _paint_track(inp, expected, track, row, col)
        reserved.update(_dilate(all_cells, h, w, PLACEMENT_PADDING))
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h_tracks, v_tracks = random.choice(TRACK_COUNT_OPTIONS)
    h = random.randint(max(GRID_MIN, 7 + 4 * h_tracks), GRID_MAX)
    w = random.randint(max(GRID_MIN, 7 + 4 * v_tracks), GRID_MAX)

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()

    orientations = ["h"] * h_tracks + ["v"] * v_tracks
    random.shuffle(orientations)
    for orientation in orientations:
        track = _sample_track_shape(orientation)
        if not _place_track(inp, expected, reserved, track):
            return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guide-track growth input")
