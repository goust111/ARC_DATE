from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components


GREY = Color.GREY
BLUE = Color.BLUE
RED = Color.RED
BLACK = Color.BLACK

MIN_GRID = 14
MAX_GRID = 19
MIN_SEG = 3
MAX_SEG = 7
GEN_RETRIES = 600
GROW_RETRIES = 120


def _control_rows(grid: np.ndarray) -> list[int]:
    return sorted(int(r) for r in np.where(grid[1:, 0] == GREY)[0] + 1)


def _control_cols(grid: np.ndarray) -> list[int]:
    return sorted(int(c) for c in np.where(grid[0, 1:] == GREY)[0] + 1)


def _spans(size: int, controls: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 1
    for line in controls:
        if not (start <= line < size - 1):
            raise ValueError("invalid control line placement")
        spans.append((start, int(line)))
        start = int(line) + 1
    if start >= size:
        raise ValueError("missing trailing region")
    spans.append((start, size - 1))
    return spans


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    out[patch == BLUE] = BLACK
    out[patch == BLACK] = RED
    return out


def _transform_interior(interior: np.ndarray, height: int, width: int, h_ctrl: list[int], v_ctrl: list[int]) -> np.ndarray:
    out = np.array(interior, dtype=int, copy=True)
    row_spans = _spans(height, h_ctrl)
    col_spans = _spans(width, v_ctrl)
    for ri, (r0, r1) in enumerate(row_spans):
        rr0 = r0 - 1
        rr1 = r1
        for ci, (c0, c1) in enumerate(col_spans):
            if (ri + ci) % 2 == 0:
                continue
            cc0 = c0 - 1
            cc1 = c1
            out[rr0:rr1, cc0:cc1] = _transform_patch(out[rr0:rr1, cc0:cc1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use invisible rays from grey controls to partition the inner board and invert every odd-parity region.

    Input structure:
    - The background is black, with grey control pixels on the top row and left column.
    - The mutable interior is the subgrid rows 1..end and cols 1..end, containing black cells and several blue 4-connected blobs.
    - Each grey pixel on the left border defines a horizontal split, and each grey pixel on the top border defines a vertical split.

    Rule:
    - Treat every left-border grey pixel as an invisible horizontal separator drawn rightward after that row, and every top-border grey pixel as an invisible vertical separator drawn downward after that column.
    - These separators partition the interior into rectangular regions ordered from the interior top-left corner.
    - For each region, count how many horizontal separators and vertical separators lie before it; if the sum is odd, change blue to black and black to red inside that region.
    - Keep even-parity regions unchanged, and leave the top row and left column exactly as they were.

    Key insight:
    - The rays never get painted, but their row and column positions create a hidden checkerboard of parity regions that decides where local occupancy inversion happens.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    h_ctrl = _control_rows(g)
    v_ctrl = _control_cols(g)

    out = g.copy()
    out[1:, 1:] = _transform_interior(g[1:, 1:], h, w, h_ctrl, v_ctrl)
    return out


def _sample_segments(total: int, parts: int) -> list[int] | None:
    base = MIN_SEG * parts
    if total < base:
        return None
    extra = total - base
    for _ in range(80):
        bins = [MIN_SEG] * parts
        for _ in range(extra):
            idx = random.randrange(parts)
            if bins[idx] < MAX_SEG:
                bins[idx] += 1
            else:
                movable = [i for i, val in enumerate(bins) if val < MAX_SEG]
                if not movable:
                    break
                bins[random.choice(movable)] += 1
        if sum(bins) == total and max(bins) <= MAX_SEG:
            random.shuffle(bins)
            return bins
    return None


def _controls_from_segments(segments: list[int]) -> list[int]:
    pos = 0
    out = []
    for size in segments[:-1]:
        pos += int(size)
        out.append(pos)
    return out


def _control_adjacent_mask(shape: tuple[int, int], h_ctrl: list[int], v_ctrl: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    for r in h_ctrl:
        for dr in (-1, 0, 1):
            rr = r + dr
            cc = 1
            if 1 <= rr < h and 1 <= cc < w:
                mask[rr, cc] = True
    for c in v_ctrl:
        for dc in (-1, 0, 1):
            rr = 1
            cc = c + dc
            if 1 <= rr < h and 1 <= cc < w:
                mask[rr, cc] = True
    return mask


def _region_index(pos: int, controls: list[int]) -> int:
    return sum(int(line) < int(pos) for line in controls)


def _shape_signature(coords: np.ndarray) -> tuple[int, int, bytes]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
    mask[rows - r0, cols - c0] = 1
    return int(mask.shape[0]), int(mask.shape[1]), mask.tobytes()


def _is_irregular(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    h = int(rows.max() - rows.min() + 1)
    w = int(cols.max() - cols.min() + 1)
    return h >= 2 and w >= 2 and int(coords.shape[0]) < h * w


def _halo_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, shape)
    halo = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 1 <= rr < h and 1 <= cc < w:
                    halo[rr, cc] = True
    return halo


def _candidate_mask(shape: tuple[int, int], row_range: tuple[int, int], col_range: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    r0, r1 = row_range
    c0, c1 = col_range
    r0 = max(1, int(r0))
    c0 = max(1, int(c0))
    r1 = min(h - 1, int(r1))
    c1 = min(w - 1, int(c1))
    if r0 <= r1 and c0 <= c1:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _choose_frontier(
    frontier: list[tuple[int, int]],
    comp: set[tuple[int, int]],
    req_h: int | None,
    req_v: int | None,
) -> tuple[int, int]:
    random.shuffle(frontier)
    if req_h is not None:
        has_up = any(r <= req_h for r, _ in comp)
        has_down = any(r >= req_h + 1 for r, _ in comp)
        if not has_up:
            for cell in frontier:
                if cell[0] <= req_h:
                    return cell
        if not has_down:
            for cell in frontier:
                if cell[0] >= req_h + 1:
                    return cell
    if req_v is not None:
        has_left = any(c <= req_v for _, c in comp)
        has_right = any(c >= req_v + 1 for _, c in comp)
        if not has_left:
            for cell in frontier:
                if cell[1] <= req_v:
                    return cell
        if not has_right:
            for cell in frontier:
                if cell[1] >= req_v + 1:
                    return cell
    return frontier[0]


def _grow_component(mask: np.ndarray, size: int, req_h: int | None = None, req_v: int | None = None) -> list[tuple[int, int]] | None:
    allowed = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    if len(allowed) < size:
        return None

    seeds = list(allowed)
    near = [
        cell
        for cell in seeds
        if (req_h is None or abs(cell[0] - req_h) <= 1 or abs(cell[0] - (req_h + 1)) <= 1)
        and (req_v is None or abs(cell[1] - req_v) <= 1 or abs(cell[1] - (req_v + 1)) <= 1)
    ]
    if near:
        seeds = near

    for _ in range(GROW_RETRIES):
        seed = random.choice(seeds)
        comp = {seed}
        frontier = {
            nb
            for nb in ((seed[0] - 1, seed[1]), (seed[0] + 1, seed[1]), (seed[0], seed[1] - 1), (seed[0], seed[1] + 1))
            if nb in allowed
        }
        while len(comp) < size and frontier:
            cell = _choose_frontier(list(frontier), comp, req_h, req_v)
            frontier.remove(cell)
            comp.add(cell)
            r, c = cell
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in allowed and nb not in comp:
                    frontier.add(nb)
        if len(comp) != size:
            continue
        coords = np.array(sorted(comp), dtype=int)
        if req_h is not None:
            if not (np.any(coords[:, 0] <= req_h) and np.any(coords[:, 0] >= req_h + 1)):
                continue
        if req_v is not None:
            if not (np.any(coords[:, 1] <= req_v) and np.any(coords[:, 1] >= req_v + 1)):
                continue
        if not _is_irregular(coords):
            continue
        return [tuple(map(int, cell)) for cell in sorted(comp)]
    return None


def _paint_component(grid: np.ndarray, blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        grid[r, c] = BLUE
    blocked |= _halo_mask(grid.shape, coords)


def _place_horizontal_cross(
    grid: np.ndarray,
    blocked: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    h_ctrl: list[int],
    signatures: set[tuple[int, int, bytes]],
) -> bool:
    split_choices = list(enumerate(h_ctrl))
    random.shuffle(split_choices)
    col_choices = list(enumerate(col_spans))
    random.shuffle(col_choices)
    for ri, split_r in split_choices:
        upper = row_spans[ri]
        lower = row_spans[ri + 1]
        for cj, (c0, c1) in col_choices:
            mask = _candidate_mask(
                grid.shape,
                (max(upper[0], split_r - 2), min(lower[1], split_r + 2)),
                (c0, c1),
            )
            mask &= ~blocked
            if np.count_nonzero(mask) < 4:
                continue
            sizes = list(range(4, 8))
            random.shuffle(sizes)
            for size in sizes:
                coords = _grow_component(mask, size=size, req_h=split_r)
                if coords is None:
                    continue
                sig = _shape_signature(np.array(coords, dtype=int))
                if sig in signatures:
                    continue
                signatures.add(sig)
                _paint_component(grid, blocked, coords)
                return True
    return False


def _place_vertical_cross(
    grid: np.ndarray,
    blocked: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    v_ctrl: list[int],
    signatures: set[tuple[int, int, bytes]],
) -> bool:
    split_choices = list(enumerate(v_ctrl))
    random.shuffle(split_choices)
    row_choices = list(enumerate(row_spans))
    random.shuffle(row_choices)
    for ci, split_c in split_choices:
        left = col_spans[ci]
        right = col_spans[ci + 1]
        for ri, (r0, r1) in row_choices:
            mask = _candidate_mask(
                grid.shape,
                (r0, r1),
                (max(left[0], split_c - 2), min(right[1], split_c + 2)),
            )
            mask &= ~blocked
            if np.count_nonzero(mask) < 4:
                continue
            sizes = list(range(4, 8))
            random.shuffle(sizes)
            for size in sizes:
                coords = _grow_component(mask, size=size, req_v=split_c)
                if coords is None:
                    continue
                sig = _shape_signature(np.array(coords, dtype=int))
                if sig in signatures:
                    continue
                signatures.add(sig)
                _paint_component(grid, blocked, coords)
                return True
    return False


def _place_contained(
    grid: np.ndarray,
    blocked: np.ndarray,
    regions: list[tuple[int, int, int, int, int]],
    wanted_parity: int,
    signatures: set[tuple[int, int, bytes]],
) -> bool:
    candidates = [region for region in regions if region[4] == wanted_parity]
    random.shuffle(candidates)
    for r0, r1, c0, c1, _parity in candidates:
        mask = _candidate_mask(grid.shape, (r0, r1), (c0, c1))
        mask &= ~blocked
        if np.count_nonzero(mask) < 4:
            continue
        sizes = list(range(3, 7))
        random.shuffle(sizes)
        for size in sizes:
            coords = _grow_component(mask, size=size)
            if coords is None:
                continue
            sig = _shape_signature(np.array(coords, dtype=int))
            if sig in signatures:
                continue
            signatures.add(sig)
            _paint_component(grid, blocked, coords)
            return True
    return False


def _place_optional(
    grid: np.ndarray,
    blocked: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    h_ctrl: list[int],
    v_ctrl: list[int],
    regions: list[tuple[int, int, int, int, int]],
    signatures: set[tuple[int, int, bytes]],
) -> bool:
    modes = ["contained_even", "contained_odd", "both_cross", "horizontal_cross", "vertical_cross"]
    random.shuffle(modes)
    for mode in modes:
        if mode == "contained_even" and _place_contained(grid, blocked, regions, 0, signatures):
            return True
        if mode == "contained_odd" and _place_contained(grid, blocked, regions, 1, signatures):
            return True
        if mode == "horizontal_cross" and _place_horizontal_cross(grid, blocked, row_spans, col_spans, h_ctrl, signatures):
            return True
        if mode == "vertical_cross" and _place_vertical_cross(grid, blocked, row_spans, col_spans, v_ctrl, signatures):
            return True
        if mode == "both_cross":
            pairs = [(r, c) for r in h_ctrl for c in v_ctrl]
            random.shuffle(pairs)
            for split_r, split_c in pairs:
                mask = _candidate_mask(grid.shape, (split_r - 2, split_r + 2), (split_c - 2, split_c + 2))
                mask &= ~blocked
                if np.count_nonzero(mask) < 5:
                    continue
                sizes = [5, 6, 7]
                random.shuffle(sizes)
                for size in sizes:
                    coords = _grow_component(mask, size=size, req_h=split_r, req_v=split_c)
                    if coords is None:
                        continue
                    sig = _shape_signature(np.array(coords, dtype=int))
                    if sig in signatures:
                        continue
                    signatures.add(sig)
                    _paint_component(grid, blocked, coords)
                    return True
    return False


def _build_regions(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> list[tuple[int, int, int, int, int]]:
    regions = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            regions.append((r0, r1, c0, c1, (ri + ci) % 2))
    return regions


def _generate_layout() -> tuple[int, int, list[int], list[int]] | None:
    height = random.randint(MIN_GRID, MAX_GRID)
    width = random.randint(MIN_GRID, MAX_GRID)
    n_h = random.randint(2, 3)
    n_v = random.randint(2, 3)
    row_segments = _sample_segments(height - 1, n_h + 1)
    col_segments = _sample_segments(width - 1, n_v + 1)
    if row_segments is None or col_segments is None:
        return None
    return height, width, _controls_from_segments(row_segments), _controls_from_segments(col_segments)


def _component_region_parities(coords: np.ndarray, h_ctrl: list[int], v_ctrl: list[int]) -> set[int]:
    return {
        (_region_index(int(r), h_ctrl) + _region_index(int(c), v_ctrl)) % 2
        for r, c in coords
    }


def _crosses_horizontal(coords: np.ndarray, h_ctrl: list[int]) -> bool:
    row_regions = {_region_index(int(r), h_ctrl) for r, _ in coords}
    return len(row_regions) > 1


def _crosses_vertical(coords: np.ndarray, v_ctrl: list[int]) -> bool:
    col_regions = {_region_index(int(c), v_ctrl) for _, c in coords}
    return len(col_regions) > 1


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert np.all(g[0, 0] == BLACK)

    h_ctrl = _control_rows(g)
    v_ctrl = _control_cols(g)
    assert 2 <= len(h_ctrl) <= 3
    assert 2 <= len(v_ctrl) <= 3
    row_spans = _spans(h, h_ctrl)
    col_spans = _spans(w, v_ctrl)
    assert all(r1 - r0 + 1 >= 3 for r0, r1 in row_spans)
    assert all(c1 - c0 + 1 >= 3 for c0, c1 in col_spans)
    assert set(map(int, np.unique(g[1:, 1:]))) <= {BLACK, BLUE}

    forbidden = _control_adjacent_mask(g.shape, h_ctrl, v_ctrl)
    assert not np.any((g == BLUE) & forbidden)
    assert np.all(g[1:, 0][np.isin(np.arange(1, h), h_ctrl, invert=True)] == BLACK)
    assert np.all(g[0, 1:][np.isin(np.arange(1, w), v_ctrl, invert=True)] == BLACK)

    comps4 = find_components(g, bg=BLACK, connectivity=4, monochromatic=True, colors=[BLUE], return_format="dict")
    comps8 = find_components(g, bg=BLACK, connectivity=8, monochromatic=True, colors=[BLUE], return_format="dict")
    assert 4 <= len(comps4) <= 5
    assert len(comps4) == len(comps8)

    seen: set[tuple[int, int, bytes]] = set()
    crosses_h = False
    crosses_v = False
    contained_odd = False
    contained_even = False
    odd_has_blue = False
    even_has_blue = False
    for comp in comps4:
        coords = np.array(comp["coords"], dtype=int)
        assert 3 <= int(comp["area"]) <= 8
        sig = _shape_signature(coords)
        assert sig not in seen
        seen.add(sig)
        assert _is_irregular(coords)

        parities = _component_region_parities(coords, h_ctrl, v_ctrl)
        if 1 in parities:
            odd_has_blue = True
        if 0 in parities:
            even_has_blue = True
        if parities == {1}:
            contained_odd = True
        if parities == {0}:
            contained_even = True
        crosses_h |= _crosses_horizontal(coords, h_ctrl)
        crosses_v |= _crosses_vertical(coords, v_ctrl)

    assert crosses_h
    assert crosses_v
    assert contained_odd
    assert contained_even
    assert odd_has_blue
    assert even_has_blue

    out = main(g)
    assert out.shape == g.shape
    assert np.array_equal(out[0, :], g[0, :])
    assert np.array_equal(out[:, 0], g[:, 0])
    expected = g.copy()
    expected[1:, 1:] = _transform_interior(g[1:, 1:], h, w, h_ctrl, v_ctrl)
    assert np.array_equal(out, expected)
    assert not np.array_equal(out, g)
    assert np.any(out[1:, 1:] == RED)
    assert np.any((g[1:, 1:] == BLUE) & (out[1:, 1:] == BLUE))
    assert np.any((g[1:, 1:] == BLUE) & (out[1:, 1:] == BLACK))

    base = out[1:, 1:]
    for line in h_ctrl:
        alt = _transform_interior(g[1:, 1:], h, w, [x for x in h_ctrl if x != line], v_ctrl)
        assert not np.array_equal(base, alt)
    for line in v_ctrl:
        alt = _transform_interior(g[1:, 1:], h, w, h_ctrl, [x for x in v_ctrl if x != line])
        assert not np.array_equal(base, alt)


def _attempt() -> np.ndarray | None:
    layout = _generate_layout()
    if layout is None:
        return None
    height, width, h_ctrl, v_ctrl = layout
    grid = np.zeros((height, width), dtype=int)
    for r in h_ctrl:
        grid[r, 0] = GREY
    for c in v_ctrl:
        grid[0, c] = GREY

    row_spans = _spans(height, h_ctrl)
    col_spans = _spans(width, v_ctrl)
    regions = _build_regions(row_spans, col_spans)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[0, :] = True
    blocked[:, 0] = True
    blocked |= _control_adjacent_mask(grid.shape, h_ctrl, v_ctrl)

    signatures: set[tuple[int, int, bytes]] = set()
    if not _place_horizontal_cross(grid, blocked, row_spans, col_spans, h_ctrl, signatures):
        return None
    if not _place_vertical_cross(grid, blocked, row_spans, col_spans, v_ctrl, signatures):
        return None
    if not _place_contained(grid, blocked, regions, 1, signatures):
        return None
    if not _place_contained(grid, blocked, regions, 0, signatures):
        return None

    if random.random() < 0.6:
        if not _place_optional(grid, blocked, row_spans, col_spans, h_ctrl, v_ctrl, regions, signatures):
            return None

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition inversion input")
