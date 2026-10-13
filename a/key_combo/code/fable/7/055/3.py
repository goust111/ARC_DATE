from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = tuple(c for c in range(10) if c not in (BLACK, GRAY))
MAX_GENERATION_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array, copied from the common ARC utility style."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _gray_patch_mask(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = grid == GRAY
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a gray occlusion rectangle"
    r0, c0, r1, c1 = bbox_coords(coords)
    box = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(box)), "gray cells must form one solid rectangle"
    assert int(box.size) == int(mask.sum()), "gray patch must be rectangular with no stray cells"
    return mask, (r0, c0, r1, c1)


def _mono_lines(tile: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(tile.shape[0]) if bool(np.all(tile[r, :] == tile[r, 0]))]
    cols = [c for c in range(tile.shape[1]) if bool(np.all(tile[:, c] == tile[0, c]))]
    return rows, cols


def _repair_row_candidate(grid: np.ndarray, patch: np.ndarray, guide_r: int) -> dict:
    n = grid.shape[0]
    visible = [int(grid[guide_r, c]) for c in range(n) if not patch[guide_r, c]]
    assert visible, "guide row cannot be fully occluded"
    assert len(set(visible)) == 1, "visible guide-row cells must agree on one color"
    guide_color = visible[0]

    repaired = np.array(grid, copy=True)
    for r0, c0 in np.argwhere(patch):
        r = int(r0)
        c = int(c0)
        if r == guide_r:
            repaired[r, c] = guide_color
            continue
        mirror_c = n - 1 - c
        assert not patch[r, mirror_c], "every occluded non-guide cell needs a visible mirror partner"
        repaired[r, c] = int(grid[r, mirror_c])

    assert not bool(np.any(repaired == GRAY)), "repair must remove the occlusion color"
    assert bool(np.all(repaired[guide_r, :] == guide_color)), "guide row must be fully restored"
    for r in range(n):
        if r == guide_r:
            continue
        assert np.array_equal(repaired[r, :], repaired[r, ::-1]), "non-guide rows must be vertically symmetric"

    full_rows, full_cols = _mono_lines(repaired)
    assert full_rows == [guide_r], "guide row must be the only full monochrome row"
    assert full_cols == [], "row-guided tile must not also contain a full monochrome column"
    return {"orient": "row", "index": int(guide_r), "guide_color": int(guide_color), "tile": repaired}


def _repair_col_candidate(grid: np.ndarray, patch: np.ndarray, guide_c: int) -> dict:
    n = grid.shape[0]
    visible = [int(grid[r, guide_c]) for r in range(n) if not patch[r, guide_c]]
    assert visible, "guide column cannot be fully occluded"
    assert len(set(visible)) == 1, "visible guide-column cells must agree on one color"
    guide_color = visible[0]

    repaired = np.array(grid, copy=True)
    for r0, c0 in np.argwhere(patch):
        r = int(r0)
        c = int(c0)
        if c == guide_c:
            repaired[r, c] = guide_color
            continue
        mirror_r = n - 1 - r
        assert not patch[mirror_r, c], "every occluded non-guide cell needs a visible mirror partner"
        repaired[r, c] = int(grid[mirror_r, c])

    assert not bool(np.any(repaired == GRAY)), "repair must remove the occlusion color"
    assert bool(np.all(repaired[:, guide_c] == guide_color)), "guide column must be fully restored"
    for c in range(n):
        if c == guide_c:
            continue
        assert np.array_equal(repaired[:, c], repaired[::-1, c]), "non-guide columns must be horizontally symmetric"

    full_rows, full_cols = _mono_lines(repaired)
    assert full_rows == [], "column-guided tile must not also contain a full monochrome row"
    assert full_cols == [guide_c], "guide column must be the only full monochrome column"
    return {"orient": "col", "index": int(guide_c), "guide_color": int(guide_color), "tile": repaired}


def _candidate_repairs(grid: np.ndarray) -> list[dict]:
    patch, _bbox = _gray_patch_mask(grid)
    candidates: list[dict] = []
    for r in range(grid.shape[0]):
        try:
            candidates.append(_repair_row_candidate(grid, patch, r))
        except AssertionError:
            pass
    for c in range(grid.shape[1]):
        try:
            candidates.append(_repair_col_candidate(grid, patch, c))
        except AssertionError:
            pass
    return candidates


def _tile_band(tile: np.ndarray, orient: str, index: int) -> np.ndarray:
    n = tile.shape[0]
    out = np.full((n * n, n * n), BLACK, dtype=int)
    if orient == "row":
        r0 = index * n
        r1 = r0 + n
        for block_c in range(n):
            c0 = block_c * n
            out[r0:r1, c0 : c0 + n] = tile
    else:
        c0 = index * n
        c1 = c0 + n
        for block_r in range(n):
            r0 = block_r * n
            out[r0 : r0 + n, c0:c1] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the occluded symmetric tile, then tile it only across the guide-selected block band.

    Input structure:
    - The input is one square tile containing exactly one gray rectangular occlusion patch.
    - Beneath the occlusion, the true tile has exactly one full monochrome guide row or guide column.
    - Aside from that guide line, the tile is mirror-symmetric across the axis perpendicular to the guide line.

    Rule:
    - First remove the gray patch: cells on the guide line become the guide color, and every other occluded cell copies the color of its mirror partner on the opposite side of the tile.
    - Then build an n-by-n block canvas whose blocks are each n-by-n, initialized to black.
    - If the guide is row r, paste the repaired tile into every block of block-row r; if the guide is column c, paste the repaired tile into every block of block-column c.

    Key insight:
    - The gray rectangle hides part of the generating tile, so the tile must be repaired from its own mirror symmetry before the guide line can select the single output band to keep.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected one square tile"

    candidates = _candidate_repairs(g)
    assert len(candidates) == 1, "input must determine exactly one repaired guide line"
    solved = candidates[0]
    return _tile_band(solved["tile"], solved["orient"], solved["index"])


def _valid_row_tile(tile: np.ndarray, guide_r: int, guide_color: int) -> bool:
    n = tile.shape[0]
    if not bool(np.all(tile[guide_r, :] == guide_color)):
        return False
    if bool(np.any(tile == GRAY)):
        return False
    if len(set(map(int, tile.flatten()))) < 3:
        return False
    for r in range(n):
        if r == guide_r:
            continue
        if not np.array_equal(tile[r, :], tile[r, ::-1]):
            return False
        if bool(np.all(tile[r, :] == tile[r, 0])):
            return False
    full_rows, full_cols = _mono_lines(tile)
    return full_rows == [guide_r] and full_cols == []


def _make_row_tile(n: int, guide_r: int, guide_color: int) -> np.ndarray | None:
    half = n // 2
    center_choices = [c for c in PALETTE if c != guide_color]
    for _ in range(100):
        palette = random.sample(center_choices, random.randint(2, min(5, len(center_choices))))
        tile = np.empty((n, n), dtype=int)
        tile[guide_r, :] = guide_color
        for r in range(n):
            if r == guide_r:
                continue
            for _ in range(40):
                left = [random.choice(palette) for _ in range(half)]
                middle = [random.choice(palette)] if n % 2 else []
                row = np.asarray(left + middle + left[::-1], dtype=int)
                if len(set(map(int, row))) >= 2:
                    tile[r, :] = row
                    break
            else:
                break
        else:
            if _valid_row_tile(tile, guide_r, guide_color):
                return tile
    return None


def _row_patch_bbox(n: int, guide_r: int, overlap_guide: bool) -> tuple[int, int, int, int] | None:
    half = n // 2
    side_width = half
    assert side_width >= 1
    width = random.randint(1, side_width)
    side = random.choice(["left", "right"])
    if side == "left":
        c0 = random.randint(0, side_width - width)
    else:
        start = n - side_width
        c0 = random.randint(start, n - width)

    options: list[tuple[int, int]] = []
    max_h = min(3, n)
    for h in range(1, max_h + 1):
        for r0 in range(0, n - h + 1):
            rows = range(r0, r0 + h)
            includes_guide = guide_r in rows
            has_non_guide = any(r != guide_r for r in rows)
            if overlap_guide:
                if includes_guide and has_non_guide:
                    options.append((r0, h))
            elif not includes_guide:
                options.append((r0, h))
    if not options:
        return None
    r0, h = random.choice(options)
    return r0, c0, r0 + h - 1, c0 + width - 1


def _apply_patch(tile: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.array(tile, copy=True)
    out[r0 : r1 + 1, c0 : c1 + 1] = GRAY
    return out


def _validate_case(
    inp: np.ndarray,
    repaired: np.ndarray,
    orient: str,
    index: int,
    guide_color: int,
) -> bool:
    try:
        patch, bbox = _gray_patch_mask(inp)
    except AssertionError:
        return False

    if orient == "row":
        if not _valid_row_tile(repaired, index, guide_color):
            return False
    else:
        if not _valid_row_tile(repaired.T, index, guide_color):
            return False

    if int(np.sum(patch)) <= 0:
        return False
    if not bool(np.any(patch & (np.arange(inp.shape[0])[:, None] != index if orient == "row" else np.arange(inp.shape[1])[None, :] != index))):
        return False

    candidates = _candidate_repairs(inp)
    if len(candidates) != 1:
        return False
    cand = candidates[0]
    if cand["orient"] != orient or int(cand["index"]) != int(index) or int(cand["guide_color"]) != int(guide_color):
        return False
    if not np.array_equal(cand["tile"], repaired):
        return False

    oracle = _tile_band(repaired, orient, index)
    shortcut = _tile_band(inp, orient, index)
    if np.array_equal(shortcut, oracle):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, oracle):
        return False

    r0, c0, r1, c1 = bbox
    if orient == "row":
        if c0 <= (inp.shape[1] - 1) // 2 <= c1 and inp.shape[1] % 2 == 1:
            return False
    else:
        if r0 <= (inp.shape[0] - 1) // 2 <= r1 and inp.shape[0] % 2 == 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([4, 5])
    orient = random.choice(["row", "col"])
    guide_index = random.randrange(n)
    guide_color = random.choice(PALETTE)

    base_row_tile = _make_row_tile(n, guide_index, guide_color)
    if base_row_tile is None:
        return None

    overlap_guide = random.choice([False, True])
    bbox = _row_patch_bbox(n, guide_index, overlap_guide)
    if bbox is None:
        return None

    row_input = _apply_patch(base_row_tile, bbox)
    repaired = base_row_tile
    inp = row_input
    if orient == "col":
        repaired = repaired.T.copy()
        inp = inp.T.copy()

    if not _validate_case(inp, repaired, orient, guide_index, guide_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the repaired-band-tiling puzzle")
