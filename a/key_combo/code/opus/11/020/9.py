import random
from collections import deque
from functools import lru_cache

import numpy as np


ORANGE = 7
TEAL = 8
OUTLINE_COLORS = (ORANGE, TEAL)
TOP_DIFFS_BY_TURNS = {
    1: [
        (-1, -1, 1, 1),
        (-1, -1, -1, 1),
        (-1, -1, -1, 1, 1),
        (-1, 1, 1, 1),
    ],
    2: [
        (-1, 1, -1, 1),
        (-1, -1, 1, -1, 1),
        (-1, 1, -1, -1, 1),
        (-1, 1, 1, -1, 1),
        (-1, -1, 1, -1, 1, 1),
    ],
    3: [
        (-1, 1, -1, 1, -1, 1),
        (-1, -1, 1, -1, 1, -1, 1),
        (-1, 1, -1, -1, 1, -1, 1),
        (-1, 1, -1, 1, -1, -1, 1),
    ],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _turn_count(coords: np.ndarray) -> int:
    rows_by_col: dict[int, list[int]] = {}
    for r, c in coords:
        rows_by_col.setdefault(int(c), []).append(int(r))
    cols = sorted(rows_by_col)
    assert cols == list(range(cols[0], cols[-1] + 1)), "outline must span every column in its bbox"
    top_rows = [min(rows_by_col[c]) for c in cols]
    diffs = [b - a for a, b in zip(top_rows, top_rows[1:]) if b != a]
    assert diffs, "outline must move diagonally across columns"
    peaks = sum(1 for prev, cur in zip(diffs, diffs[1:]) if prev < 0 and cur > 0)
    valleys = sum(1 for prev, cur in zip(diffs, diffs[1:]) if prev > 0 and cur < 0)
    turns = max(peaks, valleys)
    assert 1 <= turns <= 3, f"expected turn count in [1, 3], got {turns}"
    return turns


def _enclosed_local_cells(local_outline: np.ndarray) -> list[tuple[int, int]]:
    h, w = local_outline.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not local_outline[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not local_outline[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not local_outline[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    enclosed = []
    for r in range(h):
        for c in range(w):
            if not local_outline[r, c] and not outside[r, c]:
                enclosed.append((r, c))
    assert enclosed, "each outline must enclose a black interior"
    return enclosed


def _extract_regions(grid: np.ndarray) -> list[dict]:
    regions = []
    for color in sorted(set(map(int, np.unique(grid))) & set(OUTLINE_COLORS)):
        for coords in _components_of_color(grid, color):
            turns = _turn_count(coords)
            r0, c0, r1, c1 = _bbox(coords)
            local_outline = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in coords:
                local_outline[int(r) - r0, int(c) - c0] = True
            enclosed_local = _enclosed_local_cells(local_outline)
            filled_mask = np.zeros_like(grid, dtype=bool)
            for r, c in coords:
                filled_mask[int(r), int(c)] = True
            for lr, lc in enclosed_local:
                filled_mask[r0 + lr, c0 + lc] = True
            regions.append(
                {
                    "color": color,
                    "turns": turns,
                    "coords": coords,
                    "filled_mask": filled_mask,
                }
            )
    assert 2 <= len(regions) <= 3, "expected two or three closed outlines"
    turns_seen = [region["turns"] for region in regions]
    assert len(set(turns_seen)) == len(turns_seen), "turn counts must be distinct across outlines"
    return regions


def _compose_output(grid: np.ndarray, regions: list[dict]) -> tuple[np.ndarray, bool]:
    out = np.array(grid, dtype=int, copy=True)
    base_union = np.zeros(grid.shape, dtype=bool)
    for region in regions:
        base_union |= region["filled_mask"]
        out[region["filled_mask"]] = region["color"]

    h, w = grid.shape
    best_turn = np.full((h, w), -1, dtype=int)
    best_color = np.zeros((h, w), dtype=int)
    cover_count = np.zeros((h, w), dtype=int)
    for region in regions:
        turns = int(region["turns"])
        cand = set()
        for r, c in np.argwhere(region["filled_mask"]):
            for dr in range(-turns, turns + 1):
                span = turns - abs(dr)
                for dc in range(-span, span + 1):
                    rr, cc = int(r) + dr, int(c) + dc
                    if _in_bounds(grid.shape, rr, cc) and not base_union[rr, cc]:
                        cand.add((rr, cc))
        for rr, cc in cand:
            cover_count[rr, cc] += 1
            if turns > best_turn[rr, cc]:
                best_turn[rr, cc] = turns
                best_color[rr, cc] = region["color"]

    out[best_turn >= 0] = best_color[best_turn >= 0]
    return out, bool(np.any(cover_count > 1))


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each diagonal closed outline, then dilate it by a distance encoded by its own scanline turning complexity.

    Input structure:
    - The grid is mostly black background with 2-3 disjoint closed diagonal outlines.
    - Outline pixels use orange (7) and teal (8); components are identified per connected outline, not by a fixed template.
    - Each outline encloses a 4-connected black interior and spans every column of its own bounding box.

    Rule:
    - Find every 8-connected outline component in colors 7 and 8.
    - For each component, scan its columns left to right, track the topmost outline row, and count its vertical turning complexity.
    - Fill the black interior enclosed by the outline with that outline color.
    - Dilate the filled region by that turn count using Manhattan distance; if dilation claims overlap, the larger turn count wins on those added pixels.

    Key insight:
    - The outline geometry itself encodes the dilation radius: more zigzag turning in the contour causes a larger outward expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    regions = _extract_regions(g)
    out, _ = _compose_output(g, regions)
    return out


def _rows_from_diffs(diffs: tuple[int, ...]) -> list[int]:
    rows = [0]
    for diff in diffs:
        rows.append(rows[-1] + diff)
    low = min(rows)
    return [row - low for row in rows]


def _find_bottom_rows(top_rows: tuple[int, ...]) -> list[int] | None:
    width = len(top_rows)
    end_row = top_rows[-1]

    @lru_cache(maxsize=None)
    def dfs(i: int, row: int):
        if i == width - 1:
            return () if row == end_row else None
        remaining = width - i - 2
        for diff in (-1, 1):
            next_row = row + diff
            gap = next_row - top_rows[i + 1]
            if i + 1 < width - 1:
                if gap < 2:
                    continue
            elif gap != 0:
                continue
            if abs(end_row - next_row) > remaining:
                continue
            if (remaining - abs(end_row - next_row)) % 2:
                continue
            rest = dfs(i + 1, next_row)
            if rest is not None:
                return (next_row,) + rest
        return None

    result = dfs(0, top_rows[0])
    if result is None:
        return None
    return [top_rows[0], *result]


def _build_patch(turns: int, color: int) -> np.ndarray:
    diffs = random.choice(TOP_DIFFS_BY_TURNS[turns])
    top = _rows_from_diffs(diffs)
    bottom = _find_bottom_rows(tuple(top))
    assert bottom is not None, f"no lower chain for turn count {turns}"
    low = min(min(top), min(bottom))
    top = [row - low for row in top]
    bottom = [row - low for row in bottom]
    patch = np.zeros((max(max(top), max(bottom)) + 1, len(top)), dtype=int)
    for c, row in enumerate(top):
        patch[row, c] = color
    for c, row in enumerate(bottom):
        patch[row, c] = color
    return patch


def _boxes_clear(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _place_patches(shape: tuple[int, int], patches: list[np.ndarray], gap: int = 1) -> list[tuple[int, int]] | None:
    h, w = shape
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    placed_boxes: list[tuple[int, int, int, int]] = []

    def rec(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        ph, pw = patches[idx].shape
        candidates = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _boxes_clear(box, placed_boxes, gap):
                    candidates.append((r0, c0, box))
        random.shuffle(candidates)
        for r0, c0, box in candidates:
            positions[idx] = (r0, c0)
            placed_boxes.append(box)
            if rec(k + 1):
                return True
            placed_boxes.pop()
            positions[idx] = None
        return False

    return [pos for pos in positions] if rec(0) else None


def _color_assignment(count: int) -> list[int]:
    if count == 2:
        colors = [ORANGE, TEAL]
        random.shuffle(colors)
        return colors
    repeat = random.choice(OUTLINE_COLORS)
    colors = [ORANGE, TEAL, repeat]
    random.shuffle(colors)
    return colors


def _gen_attempt() -> np.ndarray | None:
    count = 3 if random.random() < 0.7 else 2
    want_overlap = random.random() < 0.5
    if count == 3:
        h = random.randint(15, 17)
        w = random.randint(15, 17)
        turns = [1, 2, 3]
    else:
        base = [1, 2, 3]
        turns = random.sample(base, 2)
        if want_overlap:
            h = random.randint(13, 15)
            w = random.randint(13, 15)
        else:
            h = random.randint(13, 17)
            w = random.randint(13, 17)

    colors = _color_assignment(count)
    patches = [_build_patch(turn, color) for turn, color in zip(turns, colors)]
    positions = _place_patches((h, w), patches, gap=1)
    if positions is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw][patch != 0] = patch[patch != 0]

    try:
        regions = _extract_regions(grid)
    except AssertionError:
        return None
    if sorted(region["turns"] for region in regions) != sorted(turns):
        return None

    out, has_overlap = _compose_output(grid, regions)
    if has_overlap != want_overlap:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < 8:
        return None
    return grid if np.array_equal(main(grid), out) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
