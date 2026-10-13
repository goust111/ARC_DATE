from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

WAVE_COLORS = (BLUE, RED, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


@dataclass(frozen=True)
class Source:
    sid: int
    color: int
    alt_color: int
    frame_r0: int
    frame_c0: int
    gap: tuple[int, int]
    seed: tuple[int, int]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.frame_r0, self.frame_c0, self.frame_r0 + 3, self.frame_c0 + 3)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _frame_border_cells(frame_r0: int, frame_c0: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(frame_r0, frame_r0 + 4):
        for c in range(frame_c0, frame_c0 + 4):
            if r in (frame_r0, frame_r0 + 3) or c in (frame_c0, frame_c0 + 3):
                cells.append((r, c))
    return cells


def _structure_mask_from_sources(shape: tuple[int, int], sources: list[Source]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for source in sources:
        r0, c0 = source.frame_r0, source.frame_c0
        for r, c in _frame_border_cells(r0, c0):
            if (r, c) != source.gap:
                mask[r, c] = True
        mask[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = True
    return mask


def _parse_sources(grid: np.ndarray) -> list[Source]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)
    found: list[Source] = []
    seen_blocks: set[tuple[int, int]] = set()

    for br in range(1, h - 2):
        for bc in range(1, w - 2):
            block = g[br : br + 2, bc : bc + 2]
            color = int(block[0, 0])
            if color in (bg, GREEN, GREY):
                continue
            if not np.all(block == color):
                continue

            fr0, fc0 = br - 1, bc - 1
            border = _frame_border_cells(fr0, fc0)
            green_cells: list[tuple[int, int]] = []
            gap_cells: list[tuple[int, int]] = []
            valid = True
            for r, c in border:
                value = int(g[r, c])
                if value == GREEN:
                    green_cells.append((r, c))
                elif value == bg:
                    gap_cells.append((r, c))
                else:
                    valid = False
                    break
            if not valid or len(green_cells) != 11 or len(gap_cells) != 1:
                continue

            gap_r, gap_c = gap_cells[0]
            dr = -1 if gap_r == fr0 else (1 if gap_r == fr0 + 3 else 0)
            dc = -1 if gap_c == fc0 else (1 if gap_c == fc0 + 3 else 0)
            if dr == 0 and dc == 0:
                continue
            seed = (gap_r + dr, gap_c + dc)
            if not _in_bounds(g.shape, seed[0], seed[1]):
                continue
            if int(g[seed]) != bg:
                continue

            if (br, bc) in seen_blocks:
                continue
            seen_blocks.add((br, bc))
            found.append(
                Source(
                    sid=-1,
                    color=color,
                    alt_color=BLUE if color == RED else RED,
                    frame_r0=fr0,
                    frame_c0=fc0,
                    gap=(gap_r, gap_c),
                    seed=seed,
                )
            )

    found.sort(key=lambda s: (s.frame_r0, s.frame_c0))
    assert 2 <= len(found) <= 3, "expected 2-3 framed wave sources"
    colors = [source.color for source in found]
    assert len(colors) == len(set(colors)), "each source block color must be unique"
    seeds = [source.seed for source in found]
    assert len(seeds) == len(set(seeds)), "source seeds must be distinct"

    structure_mask = _structure_mask_from_sources(g.shape, found)
    assert np.array_equal(structure_mask, g != bg), "input must contain only framed source structures on background"

    out: list[Source] = []
    for sid, source in enumerate(found):
        out.append(
            Source(
                sid=sid,
                color=source.color,
                alt_color=source.alt_color,
                frame_r0=source.frame_r0,
                frame_c0=source.frame_c0,
                gap=source.gap,
                seed=source.seed,
            )
        )
    return out


def _simulate_waves(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Source], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sources = _parse_sources(g)
    structure_mask = _structure_mask_from_sources(g.shape, sources)

    out = g.copy()
    owner = np.full(g.shape, -1, dtype=int)
    dist = np.full(g.shape, -1, dtype=int)
    walls = np.zeros(g.shape, dtype=bool)

    frontiers: list[set[tuple[int, int]]] = [set() for _ in sources]
    for source in sources:
        sr, sc = source.seed
        assert not structure_mask[sr, sc], "seed must be outside all source structures"
        owner[sr, sc] = source.sid
        dist[sr, sc] = 0
        out[sr, sc] = source.color
        frontiers[source.sid].add(source.seed)

    round_idx = 1
    while True:
        attacked: set[tuple[int, int]] = set()
        for sid, frontier in enumerate(frontiers):
            for r, c in frontier:
                if owner[r, c] != sid or walls[r, c]:
                    continue
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if structure_mask[nr, nc] or walls[nr, nc]:
                        continue
                    other = int(owner[nr, nc])
                    if other != -1 and other != sid:
                        attacked.add((nr, nc))

        for r, c in attacked:
            walls[r, c] = True
            owner[r, c] = -1
            dist[r, c] = -1
            out[r, c] = GREY

        proposals: dict[tuple[int, int], set[int]] = {}
        for sid, frontier in enumerate(frontiers):
            for r, c in frontier:
                if owner[r, c] != sid or walls[r, c]:
                    continue
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if structure_mask[nr, nc] or walls[nr, nc]:
                        continue
                    if owner[nr, nc] != -1:
                        continue
                    proposals.setdefault((nr, nc), set()).add(sid)

        new_frontiers: list[set[tuple[int, int]]] = [set() for _ in sources]
        tie_cells: list[tuple[int, int]] = []
        for cell, sids in proposals.items():
            if len(sids) >= 2:
                tie_cells.append(cell)
                continue
            sid = next(iter(sids))
            r, c = cell
            owner[r, c] = sid
            dist[r, c] = round_idx
            color = sources[sid].color if round_idx % 2 == 0 else sources[sid].alt_color
            out[r, c] = color
            new_frontiers[sid].add(cell)

        for r, c in tie_cells:
            walls[r, c] = True
            owner[r, c] = -1
            dist[r, c] = -1
            out[r, c] = GREY

        if not attacked and not proposals:
            break
        frontiers = new_frontiers
        round_idx += 1

    return out, owner, dist, sources, walls


def main(grid: np.ndarray) -> np.ndarray:
    """Expand concentric Manhattan wavefronts from framed block gaps and freeze multi-source collision boundaries as grey walls.

    Input structure:
    - The background is black and the only non-background objects are 2-3 source structures.
    - Each source structure is a unique-colored 2x2 block inside a green 4x4 frame with exactly one missing border cell.
    - The background cell immediately outside that gap is the wave seed, and source structures themselves are impassable.

    Rule:
    - Detect every framed 2x2 block, read its inner color as the source wave color, and start a wave from the seed outside its frame gap.
    - Expand all waves synchronously by 4-neighbor steps; distance 0 and every even round use the source color, while odd rounds use red unless the source itself is red, in which case odd rounds use blue.
    - If multiple sources try to claim the same new cell in one round, or if a source reaches a cell already claimed by another source in a later round, that cell becomes a grey wall and blocks all further expansion through it; untouched background stays black and the original framed sources stay unchanged.

    Key insight:
    - The output is a competitive multi-source flood fill: the colored rings come from each seed's expansion round parity, and the emergent grey collision walls carve the grid into Voronoi-like territories.
    """
    out, _owner, _dist, _sources, _walls = _simulate_waves(np.asarray(grid, dtype=int))
    return out


SECTOR_LAYOUTS = {
    2: [
        ("top", "bottom"),
        ("left", "right"),
        ("tl", "br"),
        ("tr", "bl"),
        ("top", "left"),
        ("bottom", "right"),
    ],
    3: [
        ("top", "bl", "br"),
        ("bottom", "tl", "tr"),
        ("left", "top", "bottom"),
        ("right", "top", "bottom"),
        ("top", "left", "right"),
        ("bottom", "left", "right"),
    ],
}

SECTOR_GAPS = {
    "top": [(3, 1), (3, 2)],
    "bottom": [(0, 1), (0, 2)],
    "left": [(1, 3), (2, 3)],
    "right": [(1, 0), (2, 0)],
    "tl": [(3, 3)],
    "tr": [(3, 0)],
    "bl": [(0, 3)],
    "br": [(0, 0)],
}


def _range_values(lo: int, hi: int) -> list[int]:
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _sector_positions(h: int, w: int, sector: str) -> list[tuple[int, int]]:
    if sector == "top":
        rows = _range_values(1, 2)
        cols = _range_values(3, w - 7)
    elif sector == "bottom":
        rows = _range_values(h - 6, h - 5)
        cols = _range_values(3, w - 7)
    elif sector == "left":
        rows = _range_values(3, h - 7)
        cols = _range_values(1, 2)
    elif sector == "right":
        rows = _range_values(3, h - 7)
        cols = _range_values(w - 6, w - 5)
    elif sector == "tl":
        rows = _range_values(1, 2)
        cols = _range_values(1, 2)
    elif sector == "tr":
        rows = _range_values(1, 2)
        cols = _range_values(w - 6, w - 5)
    elif sector == "bl":
        rows = _range_values(h - 6, h - 5)
        cols = _range_values(1, 2)
    elif sector == "br":
        rows = _range_values(h - 6, h - 5)
        cols = _range_values(w - 6, w - 5)
    else:
        raise ValueError(f"unknown sector {sector}")

    return [(r, c) for r in rows for c in cols]


def _frame_gap_to_seed(frame_r0: int, frame_c0: int, gap_rel: tuple[int, int]) -> tuple[int, int]:
    gap_r = frame_r0 + gap_rel[0]
    gap_c = frame_c0 + gap_rel[1]
    dr = -1 if gap_rel[0] == 0 else (1 if gap_rel[0] == 3 else 0)
    dc = -1 if gap_rel[1] == 0 else (1 if gap_rel[1] == 3 else 0)
    return (gap_r + dr, gap_c + dc)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _seed_spacing_ok(seeds: list[tuple[int, int]]) -> bool:
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            sr0, sc0 = seeds[i]
            sr1, sc1 = seeds[j]
            if abs(sr0 - sr1) + abs(sc0 - sc1) < 8:
                return False
    return True


def _choose_colors(n: int) -> list[int]:
    palette = list(WAVE_COLORS)
    if n >= 2 and random.random() < 0.55:
        palette.remove(RED)
        chosen = [RED] + random.sample(palette, n - 1)
        random.shuffle(chosen)
        return chosen
    return random.sample(palette, n)


def _build_candidate_grid() -> np.ndarray | None:
    h = random.randint(15, 21)
    w = random.randint(15, 21)
    source_count = random.choice((2, 3))
    sectors = random.choice(SECTOR_LAYOUTS[source_count])
    colors = _choose_colors(source_count)

    placements: list[tuple[int, int, tuple[int, int], int]] = []
    bboxes: list[tuple[int, int, int, int]] = []
    seeds: list[tuple[int, int]] = []

    for sector, color in zip(sectors, colors, strict=False):
        candidates = _sector_positions(h, w, sector)
        random.shuffle(candidates)
        placed = False
        for frame_r0, frame_c0 in candidates:
            gap_rel = random.choice(SECTOR_GAPS[sector])
            seed = _frame_gap_to_seed(frame_r0, frame_c0, gap_rel)
            bbox = (frame_r0, frame_c0, frame_r0 + 3, frame_c0 + 3)
            if not _in_bounds((h, w), seed[0], seed[1]):
                continue
            if any(_boxes_too_close(bbox, prev, margin=2) for prev in bboxes):
                continue
            if any(abs(seed[0] - rs) + abs(seed[1] - cs) < 8 for rs, cs in seeds):
                continue
            placements.append((frame_r0, frame_c0, gap_rel, color))
            bboxes.append(bbox)
            seeds.append(seed)
            placed = True
            break
        if not placed:
            return None

    if not _seed_spacing_ok(seeds):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for frame_r0, frame_c0, gap_rel, color in placements:
        for r, c in _frame_border_cells(frame_r0, frame_c0):
            grid[r, c] = GREEN
        gap = (frame_r0 + gap_rel[0], frame_c0 + gap_rel[1])
        grid[gap] = BLACK
        grid[frame_r0 + 1 : frame_r0 + 3, frame_c0 + 1 : frame_c0 + 3] = color

    return grid


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        out, owner, dist, sources, walls = _simulate_waves(grid)
    except AssertionError:
        return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    if np.count_nonzero(walls) < 2:
        return False
    if np.count_nonzero(out != grid) < 18:
        return False

    structure_mask = _structure_mask_from_sources(grid.shape, sources)
    if not np.array_equal(structure_mask, grid != BLACK):
        return False

    wall_adjacent_sources: set[int] = set()
    for r, c in np.argwhere(walls):
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc) and owner[nr, nc] != -1:
                wall_adjacent_sources.add(int(owner[nr, nc]))
    if len(wall_adjacent_sources) < 2:
        return False

    farthest = 0
    for source in sources:
        owned = owner == source.sid
        if np.count_nonzero(owned) < 6:
            return False
        if not np.any(dist[owned] >= 3):
            return False
        farthest = max(farthest, int(dist[owned].max(initial=0)))
        expected_odd = source.alt_color
        expected_even = source.color
        odd_cells = owned & (dist % 2 == 1)
        even_cells = owned & (dist % 2 == 0)
        if np.any(out[odd_cells] != expected_odd):
            return False
        if np.any(out[even_cells] != expected_even):
            return False
    if farthest < 5:
        return False

    changed_wave_colors = set(map(int, np.unique(out[(owner != -1) & (~structure_mask)])))
    if len(changed_wave_colors) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate_grid()
        if grid is not None and _validate_generated_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid concentric-wave collision puzzle")
