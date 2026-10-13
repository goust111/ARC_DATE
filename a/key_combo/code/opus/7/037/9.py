from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


GREY = 5
BLACK = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
RNG = random.Random()


@dataclass(frozen=True)
class ChainInfo:
    size: int
    bbox: tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = neighbors4(0, 0) if connectivity == 4 else neighbors8(0, 0)
    steps = [(dr, dc) for dr, dc in steps]
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_of(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def intersects(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def contiguous_segments(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        segs.append((start, prev))
        start = prev = value
    segs.append((start, prev))
    return segs


def find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_mask = grid == GREY
    candidates: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for comp in components(grey_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_of(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        border_size = 2 * h + 2 * w - 4
        if len(comp) != border_size:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros_like(box, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not np.all(box[border_mask] == GREY):
            continue
        interior = box[1:-1, 1:-1]
        if interior.size == 0:
            continue
        color_mask = (interior != BLACK) & (interior != GREY)
        if not np.any(color_mask):
            continue
        key = (r0, -c1, -(h * w), -len(comp))
        candidates.append(((r0, c0, r1, c1), key))
    assert candidates, "legend frame not found"
    candidates.sort(key=lambda item: item[1])
    return candidates[0][0]


def legend_mapping(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> dict[int, int]:
    r0, c0, r1, c1 = legend_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != BLACK) & (interior != GREY)
    mapping: dict[int, int] = {}
    for comp in components(mask, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_of(comp)
        color = int(interior[comp[0, 0], comp[0, 1]])
        assert np.all(interior[comp[:, 0], comp[:, 1]] == color)
        assert rr0 == rr1, "legend bars must be horizontal"
        assert len(comp) == cc1 - cc0 + 1, "legend bars must be contiguous"
        mapping[cc1 - cc0 + 1] = color
    assert len(mapping) >= 3
    return mapping


def diagonal_chains(grey_mask: np.ndarray) -> tuple[np.ndarray, list[ChainInfo]]:
    height, width = grey_mask.shape
    diag_mask = np.zeros_like(grey_mask, dtype=bool)
    for r in range(height):
        for c in range(width):
            if not grey_mask[r, c]:
                continue
            orth = 0
            diag = 0
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and grey_mask[nr, nc]:
                    orth += 1
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < height and 0 <= nc < width and grey_mask[nr, nc] and nr != r and nc != c:
                    diag += 1
            if orth == 0 and diag > 0:
                diag_mask[r, c] = True
    infos: list[ChainInfo] = []
    for comp in components(diag_mask, connectivity=8):
        if len(comp) < 2:
            continue
        infos.append(ChainInfo(size=len(comp), bbox=bbox_of(comp)))
    return diag_mask, infos


def fill_short_gaps(work: np.ndarray, chain_infos: list[ChainInfo]) -> None:
    height, width = work.shape
    base = work == GREY

    def chain_near(box: tuple[int, int, int, int], gap_len: int) -> bool:
        for info in chain_infos:
            if info.size >= gap_len and intersects(info.bbox, box):
                return True
        return False

    for r in range(height):
        cols = np.flatnonzero(base[r])
        segs = contiguous_segments(cols)
        for (a0, b0), (a1, _b1) in zip(segs, segs[1:]):
            gap0 = b0 + 1
            gap1 = a1 - 1
            gap_len = gap1 - gap0 + 1
            if gap_len <= 0:
                continue
            near_box = (max(0, r - 1), max(0, gap0 - 1), min(height - 1, r + 1), min(width - 1, gap1 + 1))
            if chain_near(near_box, gap_len):
                work[r, gap0 : gap1 + 1] = GREY

    base = work == GREY
    for c in range(width):
        rows = np.flatnonzero(base[:, c])
        segs = contiguous_segments(rows)
        for (a0, b0), (a1, _b1) in zip(segs, segs[1:]):
            gap0 = b0 + 1
            gap1 = a1 - 1
            gap_len = gap1 - gap0 + 1
            if gap_len <= 0:
                continue
            near_box = (max(0, gap0 - 1), max(0, c - 1), min(height - 1, gap1 + 1), min(width - 1, c + 1))
            if chain_near(near_box, gap_len):
                work[gap0 : gap1 + 1, c] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal grey hint chains, complete the nearby broken straight lines, then recolor each finished line by its legend-matched length.

    Input structure:
    - The grid has a black background, a left main region with grey horizontal or vertical line segments, and a grey-framed legend in the upper-right.
    - Some main-region grey lines are broken by short black gaps, and each such gap has a nearby grey 45-degree diagonal hint chain.
    - The legend interior contains several colored horizontal bars whose lengths encode the recolor mapping.

    Rule:
    - Detect the grey legend frame and read each colored legend bar as a length-to-color entry.
    - In the main region, remove grey pixels that belong only to diagonal chains, then fill horizontal or vertical gaps whose nearby removed chain is at least as long as the gap.
    - Measure every resulting grey straight line segment and recolor the whole segment when its length appears in the legend; leave unmatched grey segments unchanged.
    - Keep the legend region unchanged.

    Key insight:
    - The diagonal noise is not just distractor material: removing it reveals which short gaps may be repaired, and the repaired line length is the key used to look up the final recolor in the legend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_bbox = find_legend_bbox(g)
    mapping = legend_mapping(g, legend_bbox)

    main_mask = np.ones_like(g, dtype=bool)
    r0, c0, r1, c1 = legend_bbox
    main_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    grey_main = (g == GREY) & main_mask
    diag_mask, chain_infos = diagonal_chains(grey_main)
    out[diag_mask] = BLACK

    work = np.where(main_mask, out, BLACK)
    fill_short_gaps(work, chain_infos)
    out[main_mask] = work[main_mask]

    final_grey = (out == GREY) & main_mask
    for comp in components(final_grey, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_of(comp)
        if rr0 != rr1 and cc0 != cc1:
            continue
        length = len(comp)
        if length in mapping:
            out[comp[:, 0], comp[:, 1]] = mapping[length]

    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    return out


def _expanded(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    out.add((nr, nc))
    return out


def _place_legend(grid: np.ndarray, lengths: list[int], colors: list[int], top: int, left: int) -> None:
    legend_h = 2 * len(lengths) + 1
    legend_w = max(lengths) + 2
    grid[top : top + legend_h, left : left + legend_w] = BLACK
    grid[top, left : left + legend_w] = GREY
    grid[top + legend_h - 1, left : left + legend_w] = GREY
    grid[top : top + legend_h, left] = GREY
    grid[top : top + legend_h, left + legend_w - 1] = GREY
    inner_w = legend_w - 2
    for idx, (length, color) in enumerate(zip(lengths, colors)):
        row = top + 1 + 2 * idx
        start = left + 1 + RNG.randint(0, inner_w - length)
        grid[row, start : start + length] = color


def _full_line_cells(orientation: str, row: int, col: int, length: int) -> set[tuple[int, int]]:
    if orientation == "h":
        return {(row, col + offset) for offset in range(length)}
    return {(row + offset, col) for offset in range(length)}


def _build_broken_input(
    orientation: str,
    row: int,
    col: int,
    length: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    full_cells = _full_line_cells(orientation, row, col, length)
    max_gap = min(3, length - 2)
    gap_choices = [gap for gap in range(1, max_gap + 1) if length - gap >= 4]
    if not gap_choices:
        return None
    RNG.shuffle(gap_choices)
    for gap_len in gap_choices:
        preferred = list(range(2, length - gap_len - 1))
        fallback = list(range(1, length - gap_len))
        starts = preferred if preferred else fallback
        RNG.shuffle(starts)
        for gap_start in starts:
            gap_cells: set[tuple[int, int]] = set()
            if orientation == "h":
                gap_cells = {(row, col + gap_start + offset) for offset in range(gap_len)}
                left_end = col + gap_start - 1
                right_start = col + gap_start + gap_len
                attach_options = []
                for dr in (-1, 1):
                    attach_options.append([(row + dr * step, left_end + step) for step in range(1, 5)])
                    attach_options.append([(row + dr * step, right_start - step) for step in range(1, 5)])
            else:
                gap_cells = {(row + gap_start + offset, col) for offset in range(gap_len)}
                top_end = row + gap_start - 1
                bottom_start = row + gap_start + gap_len
                attach_options = []
                for dc in (-1, 1):
                    attach_options.append([(top_end + step, col + dc * step) for step in range(1, 5)])
                    attach_options.append([(bottom_start - step, col + dc * step) for step in range(1, 5)])
            RNG.shuffle(attach_options)
            for raw_chain in attach_options:
                min_chain = max(2, gap_len)
                max_chain = min(4, len(raw_chain))
                lengths = list(range(min_chain, max_chain + 1))
                RNG.shuffle(lengths)
                for chain_len in lengths:
                    chain_cells = set(raw_chain[:chain_len])
                    if chain_cells & full_cells:
                        continue
                    return full_cells - gap_cells, chain_cells, full_cells
    return None


def _line_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _gen_attempt() -> np.ndarray | None:
    line_count = RNG.choice((3, 4))
    lengths = sorted(RNG.sample(list(range(5, 9)), line_count))
    colors = RNG.sample(list(PALETTE), line_count)
    legend_h = 2 * line_count + 1
    legend_w = max(lengths) + 2

    height = RNG.randint(max(14, legend_h + 2), 18)
    main_w = RNG.randint(max(max(lengths) + 4, 12), 15)
    width = main_w + 3 + legend_w
    if width > 30:
        return None

    legend_top = RNG.randint(1, 2)
    legend_left = width - legend_w - 1
    main_limit = legend_left - 2
    if main_limit < max(lengths) + 2:
        return None

    grid = np.zeros((height, width), dtype=int)
    _place_legend(grid, lengths, colors, legend_top, legend_left)
    expected = grid.copy()

    broken_target = RNG.randint(1, line_count - 1)
    broken_indices = set(RNG.sample(range(line_count), broken_target))
    orientations = ["h", "v"]
    while len(orientations) < line_count:
        orientations.append(RNG.choice(("h", "v")))
    RNG.shuffle(orientations)

    occupied_input: set[tuple[int, int]] = set()
    occupied_future: set[tuple[int, int]] = set()
    row_used: set[int] = set()
    col_used: set[int] = set()
    legend_zone = {(r, c) for r in range(legend_top - 1, legend_top + legend_h + 1) for c in range(legend_left - 1, width)}

    for idx, length in enumerate(lengths):
        orientation = orientations[idx]
        broken = idx in broken_indices
        placed = False
        for _ in range(120):
            if orientation == "h":
                row = RNG.randint(2, height - 3)
                if row in row_used:
                    continue
                col = RNG.randint(1, main_limit - length)
            else:
                row = RNG.randint(1, height - length - 1)
                col = RNG.randint(2, main_limit - 1)
                if col in col_used:
                    continue

            if broken:
                built = _build_broken_input(orientation, row, col, length)
                if built is None:
                    continue
                line_cells, chain_cells, full_cells = built
            else:
                full_cells = _full_line_cells(orientation, row, col, length)
                line_cells = full_cells
                chain_cells = set()

            struct_cells = set(full_cells) | set(chain_cells)
            if any(r < 0 or r >= height or c < 0 or c >= width for r, c in struct_cells):
                continue
            if struct_cells & legend_zone:
                continue
            if _expanded(struct_cells, height, width) & occupied_input:
                continue
            if _expanded(full_cells, height, width) & occupied_future:
                continue

            bbox = _line_bbox(struct_cells)
            if bbox[3] >= main_limit:
                continue

            grid[tuple(zip(*line_cells))] = GREY
            if chain_cells:
                grid[tuple(zip(*chain_cells))] = GREY
            color = colors[lengths.index(length)]
            for r, c in full_cells:
                expected[r, c] = color
            for r, c in chain_cells:
                expected[r, c] = BLACK

            occupied_input |= set(line_cells) | set(chain_cells)
            occupied_future |= set(full_cells)
            if orientation == "h":
                row_used.add(row)
            else:
                col_used.add(col)
            placed = True
            break
        if not placed:
            return None

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
