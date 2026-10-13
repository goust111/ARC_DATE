from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
RANK_COLORS = (1, 2, 3, 4)
SAMPLE_SIZE = 3
FRAME_PAD = 1
BG = 0
SAMPLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_signature(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    expected = {
        *((r, c0 + 3) for r in range(r0, r0 + height)),
        *((r0 + 3, c) for c in range(c0, c0 + width)),
    }
    got = {tuple(map(int, cell)) for cell in coords}
    assert got == expected, "grey component must be an L extending the sample's right and bottom sides"
    assert height >= 4 and width >= 4, "marker must reach both the sample's right side and bottom side"
    return r0, c0, height, width


def _sample_matches(grid: np.ndarray, sample: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(h - SAMPLE_SIZE + 1):
        for c in range(w - SAMPLE_SIZE + 1):
            if np.array_equal(grid[r : r + SAMPLE_SIZE, c : c + SAMPLE_SIZE], sample):
                out.append((r, c))
    return out


def _frame_ring_is_clear(grid: np.ndarray, top: int, left: int) -> bool:
    r0 = top - FRAME_PAD
    c0 = left - FRAME_PAD
    r1 = top + SAMPLE_SIZE
    c1 = left + SAMPLE_SIZE
    if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
        return False
    ring = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    ring[1:-1, 1:-1] = BG
    return bool(np.all(ring == BG))


def _marker_infos(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for coords in _components_of_color(grid, GREY):
        sr, sc, height, width = _marker_signature(coords)
        assert sr + SAMPLE_SIZE < grid.shape[0] and sc + SAMPLE_SIZE < grid.shape[1], "sample must fit in bounds"
        sample = grid[sr : sr + SAMPLE_SIZE, sc : sc + SAMPLE_SIZE]
        assert GREY not in sample, "sample may not contain grey marker pixels"
        matches = _sample_matches(grid, sample)
        assert matches.count((sr, sc)) == 1, "sample source must be present exactly once"
        assert len(matches) == 2, "sample must have exactly one other exact copy"
        duplicate = matches[0] if matches[1] == (sr, sc) else matches[1]
        assert _frame_ring_is_clear(grid, duplicate[0], duplicate[1]), "duplicate needs a blank 1-cell frame margin"
        infos.append(
            {
                "source": (sr, sc),
                "sample": sample.copy(),
                "duplicate": duplicate,
                "area": int(coords.shape[0]),
                "height": int(height),
                "width": int(width),
            }
        )
    assert 2 <= len(infos) <= 4, "expected 2 to 4 grey L markers"
    areas = [info["area"] for info in infos]
    assert len(set(areas)) == len(areas), "marker pixel counts must be distinct"
    samples = [info["sample"].tobytes() for info in infos]
    assert len(set(samples)) == len(samples), "all marked samples must be distinct"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each grey L marker both to locate a 3x3 sample and to rank the frame color for that sample's duplicate.

    Input structure:
    - A black background contains several distinct 3x3 colored sample regions and one exact duplicate of each sample elsewhere.
    - Every source sample has a grey L-shaped marker wrapped along its right side and bottom side, with different markers using different arm lengths.
    - Each duplicate sample has a blank one-cell margin around it, and no duplicate touches any grey marker.

    Rule:
    - Find every grey connected component and read the marked 3x3 sample immediately at that component's upper-left.
    - Search the grid for the unique other occurrence of that exact 3x3 pattern.
    - Rank the grey markers by pixel count from largest to smallest and map ranks to frame colors: first -> blue(1), second -> red(2), third -> green(3), fourth -> yellow(4).
    - Erase all grey marker pixels, then draw a one-cell-thick rectangular frame around each duplicate in its rank color while keeping the duplicate interior unchanged.

    Key insight:
    - The same grey L carries two independent signals: its position selects which 3x3 sample to copy-match, and its size determines the color of the frame drawn around that sample's duplicate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    infos = _marker_infos(g)
    ranked = sorted(infos, key=lambda info: info["area"], reverse=True)

    out = g.copy()
    out[out == GREY] = BG

    for rank, info in enumerate(ranked):
        color = RANK_COLORS[rank]
        tr, tc = info["duplicate"]
        r0 = tr - FRAME_PAD
        c0 = tc - FRAME_PAD
        r1 = tr + SAMPLE_SIZE
        c1 = tc + SAMPLE_SIZE
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color

    return out


AREA_TO_SHAPES = {
    7: ((4, 4),),
    8: ((4, 5), (5, 4)),
    9: ((4, 6), (5, 5), (6, 4)),
    10: ((5, 6), (6, 5)),
    11: ((6, 6),),
}


def _make_sample() -> np.ndarray:
    for _ in range(200):
        patch = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=int)
        count = random.randint(4, 6)
        cells = random.sample(range(SAMPLE_SIZE * SAMPLE_SIZE), count)
        color_count = random.randint(2, min(4, count))
        palette = random.sample(SAMPLE_COLORS, color_count)
        values = palette + random.choices(palette, k=count - color_count)
        random.shuffle(values)
        for idx, value in zip(cells, values):
            patch[idx // SAMPLE_SIZE, idx % SAMPLE_SIZE] = value
        if len(set(map(int, patch.flatten())) - {BG}) >= 2:
            return patch
    raise RuntimeError("failed to create a non-degenerate 3x3 sample")


def _build_source_patch(sample: np.ndarray, height: int, width: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[:SAMPLE_SIZE, :SAMPLE_SIZE] = sample
    patch[:height, 3] = GREY
    patch[3, :width] = GREY
    return patch


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = np.where(patch != BG, patch, grid[top : top + h, left : left + w])


def _find_slot(occupied: np.ndarray, height: int, width: int, *, pad: int = 1) -> tuple[int, int] | None:
    h, w = occupied.shape
    candidates = [(r, c) for r in range(h - height + 1) for c in range(w - width + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if _can_place(occupied, r, c, height, width, pad=pad):
            return r, c
    return None


def _semantic_check(inp: np.ndarray) -> None:
    infos = _marker_infos(inp)
    rank_by_area = [info["area"] for info in sorted(infos, key=lambda info: info["area"], reverse=True)]
    assert rank_by_area == sorted(rank_by_area, reverse=True)

    for info in infos:
        dr, dc = info["duplicate"]
        sr, sc = info["source"]
        assert (dr, dc) != (sr, sc)
        assert abs(dr - sr) + abs(dc - sc) >= 4, "duplicate must be spatially separated from its source"

    source_order = [info["area"] for info in sorted(infos, key=lambda info: info["source"])]
    dup_order = [info["area"] for info in sorted(infos, key=lambda info: info["duplicate"])]
    if len(infos) >= 3:
        assert source_order != rank_by_area or dup_order != rank_by_area, "size ranking should not collapse to a simple scan order"


def _attempt_generate() -> np.ndarray | None:
    group_count = random.randint(2, 4)
    grid_h = random.randint(18, 28)
    grid_w = random.randint(18, 28)

    candidate_areas = [7, 8, 9, 10, 11]
    random.shuffle(candidate_areas)
    chosen_areas = sorted(candidate_areas[:group_count], reverse=True)

    samples: list[np.ndarray] = []
    while len(samples) < group_count:
        sample = _make_sample()
        if any(np.array_equal(sample, other) for other in samples):
            continue
        samples.append(sample)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    random.shuffle(samples)
    for sample, area in zip(samples, chosen_areas):
        height, width = random.choice(AREA_TO_SHAPES[area])
        source_patch = _build_source_patch(sample, height, width)

        source_pos = _find_slot(occupied, height, width, pad=1)
        if source_pos is None:
            return None
        _paste(grid, source_patch, source_pos[0], source_pos[1])
        _mark_occupied(occupied, source_pos[0], source_pos[1], height, width, pad=1)

        dup_box_h = SAMPLE_SIZE + 2 * FRAME_PAD
        dup_box_w = SAMPLE_SIZE + 2 * FRAME_PAD
        dup_pos = _find_slot(occupied, dup_box_h, dup_box_w, pad=1)
        if dup_pos is None:
            return None
        grid[
            dup_pos[0] + FRAME_PAD : dup_pos[0] + FRAME_PAD + SAMPLE_SIZE,
            dup_pos[1] + FRAME_PAD : dup_pos[1] + FRAME_PAD + SAMPLE_SIZE,
        ] = sample
        _mark_occupied(occupied, dup_pos[0], dup_pos[1], dup_box_h, dup_box_w, pad=1)

    try:
        _semantic_check(grid)
        expected = main(grid)
        assert np.array_equal(main(grid.copy()), expected)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
