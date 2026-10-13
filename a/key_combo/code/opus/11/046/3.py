import math
from collections import deque

import numpy as np


FG = 8
BG = 0
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _span_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _share_row_or_col(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    return _span_overlap(box_a[0], box_a[2], box_b[0], box_b[2]) or _span_overlap(
        box_a[1], box_a[3], box_b[1], box_b[3]
    )


def _merge_fragment_groups(fragments: list[dict]) -> list[dict]:
    n = len(fragments)
    groups = []
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        members = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            for j in range(n):
                if seen[j]:
                    continue
                if _share_row_or_col(fragments[cur]["bbox"], fragments[j]["bbox"]):
                    seen[j] = True
                    stack.append(j)
        coords = np.vstack([fragments[k]["coords"] for k in members])
        box = _bbox(coords)
        groups.append({"coords": coords, "bbox": box, "members": members})
    return groups


def _gcd_of_diffs(values: list[int]) -> int:
    uniq = sorted(set(int(v) for v in values))
    diffs = [b - a for a, b in zip(uniq, uniq[1:]) if b > a]
    if not diffs:
        raise AssertionError("need at least two distinct anchors")
    g = diffs[0]
    for d in diffs[1:]:
        g = math.gcd(g, d)
    return int(g)


def _infer_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BG, FG}), "expected only black and teal"

    fragments = _components_mask(g == FG)
    assert fragments, "expected at least one teal fragment"

    raw = [{"coords": comp, "bbox": _bbox(comp)} for comp in fragments]
    groups = _merge_fragment_groups(raw)
    assert len(groups) >= 3, "expected at least three visible tiles"

    boxes = [grp["bbox"] for grp in groups]
    heights = {box[2] - box[0] + 1 for box in boxes}
    widths = {box[3] - box[1] + 1 for box in boxes}
    assert len(heights) == 1 and len(widths) == 1, "all merged tiles must share one size"
    tile_h = heights.pop()
    tile_w = widths.pop()
    assert tile_h in (4, 5) and tile_w in (4, 5) and tile_h == tile_w

    template = np.zeros((tile_h, tile_w), dtype=bool)
    rows = []
    cols = []
    for grp in groups:
        r0, c0, r1, c1 = grp["bbox"]
        rows.append(r0)
        cols.append(c0)
        local = _crop_mask(g == FG, grp["bbox"])
        template |= local

    assert template.any(), "template cannot be empty"
    assert np.any(~template), "template must contain background texture"

    row_step = _gcd_of_diffs(rows)
    col_step = _gcd_of_diffs(cols)

    row_positions = list(range(min(rows), max(rows) + 1, row_step))
    col_positions = list(range(min(cols), max(cols) + 1, col_step))
    assert len(row_positions) == len(groups), "generator expects one visible tile per lattice row"
    assert len(col_positions) == len(groups), "generator expects one visible tile per lattice col"

    return {
        "groups": groups,
        "template": template,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "row_step": row_step,
        "col_step": col_step,
        "row_positions": row_positions,
        "col_positions": col_positions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Merge teal fragments into full tile bboxes, then use the recovered tile to fill the missing periodic tiling positions.

    Input structure:
    - The grid is black background with teal foreground only.
    - Teal cells form several 4-connected fragments belonging to equal-size 4x4 or 5x5 tiles.
    - Fragments from the same tile are disconnected by black gaps, but their bounding boxes overlap in row span or column span; visible tiles occupy distinct lattice rows and distinct lattice columns.

    Rule:
    - Find all teal 4-connected fragments and merge them transitively when their bounding boxes overlap in row span or in column span.
    - Treat each merged group's bounding box as one tile location, and recover the common tile pattern by overlaying the visible teal cells inside those equal-size boxes.
    - Infer the vertical and horizontal lattice periods from the merged tile top-left coordinates, then fill every lattice position in the rectangle spanned by those coordinates with the recovered tile pattern.

    Key insight:
    - The fragment merge is only an intermediate step: it reveals the true tile bbox and anchor positions, which are then needed to recover the periodic Cartesian tiling and fill both broken visible tiles and entirely missing tiles.
    """
    info = _infer_structure(np.asarray(grid, dtype=int))
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    patch = np.where(info["template"], FG, BG)
    for r0 in info["row_positions"]:
        for c0 in info["col_positions"]:
            out[r0 : r0 + info["tile_h"], c0 : c0 + info["tile_w"]] = np.maximum(
                out[r0 : r0 + info["tile_h"], c0 : c0 + info["tile_w"]], patch
            )
    return out


_TEMPLATE_LIBRARY = {
    4: [
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
                [1, 0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ],
    5: [
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 1, 1, 1],
                [1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1],
                [1, 1, 1, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1, 0],
                [1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1],
                [0, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ],
}


def _transform_template(template: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = template.copy()
    out = np.rot90(out, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    return out.copy()


def _group_merge_connected(comps: list[np.ndarray]) -> bool:
    if not comps:
        return False
    boxes = [_bbox(comp) for comp in comps]
    n = len(boxes)
    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        cur = stack.pop()
        for j in range(n):
            if seen[j]:
                continue
            if _share_row_or_col(boxes[cur], boxes[j]):
                seen[j] = True
                stack.append(j)
    return all(seen)


def _fragment_template(template: np.ndarray, target_components: int, rng: np.random.Generator) -> np.ndarray | None:
    cells = np.argwhere(template)
    filled = len(cells)
    side = template.shape[0]
    assert side == template.shape[1]
    whole_box = (0, 0, side - 1, side - 1)

    for _ in range(300):
        keep = template.copy()
        remove_low = max(1, filled // 6)
        remove_high = max(remove_low + 1, filled // 2)
        remove_n = int(rng.integers(remove_low, remove_high + 1))
        remove_idx = rng.choice(filled, size=remove_n, replace=False)
        for idx in remove_idx:
            rr, cc = cells[idx]
            keep[rr, cc] = False

        if not keep[0].any() or not keep[-1].any() or not keep[:, 0].any() or not keep[:, -1].any():
            continue
        comps = _components_mask(keep)
        if len(comps) != target_components:
            continue
        if any(len(comp) < 2 for comp in comps):
            continue
        if _bbox(np.argwhere(keep)) != whole_box:
            continue
        if not _group_merge_connected(comps):
            continue
        return keep
    return None


def _paint_tile(grid: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    patch = np.where(mask, FG, BG)
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    sub[patch == FG] = FG


def _expected_output(shape: tuple[int, int], rows: list[int], cols: list[int], template: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r0 in rows:
        for c0 in cols:
            _paint_tile(out, r0, c0, template)
    return out


def _visible_only_output(shape: tuple[int, int], visible_positions: list[tuple[int, int]], template: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r0, c0 in visible_positions:
        _paint_tile(out, r0, c0, template)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile_size = int(rng.choice([4, 5]))
    template = _transform_template(rng.choice(_TEMPLATE_LIBRARY[tile_size]), rng)
    step = tile_size + 1

    n = int(rng.choice([3, 4] if tile_size == 4 else [3]))
    span = n * tile_size + (n - 1)
    extra_top = int(rng.integers(0, 4))
    extra_bottom = int(rng.integers(0, 4))
    extra_left = int(rng.integers(0, 4))
    extra_right = int(rng.integers(0, 4))
    h = span + extra_top + extra_bottom
    w = span + extra_left + extra_right
    if h > 30 or w > 30:
        return None

    rows = [extra_top + i * step for i in range(n)]
    cols = [extra_left + i * step for i in range(n)]
    perm = rng.permutation(n)
    visible_positions = [(rows[i], cols[int(perm[i])]) for i in range(n)]

    fragmented_count = int(rng.integers(1, n))
    frag_indices = set(int(x) for x in rng.choice(n, size=fragmented_count, replace=False))
    complete_count = n - fragmented_count
    if complete_count < 1:
        return None

    tile_masks = []
    has_three_way = False
    for i in range(n):
        if i not in frag_indices:
            tile_masks.append(template)
            continue
        target_components = int(rng.choice([2, 3], p=[0.55, 0.45]))
        frag = _fragment_template(template, target_components, rng)
        if frag is None:
            return None
        has_three_way |= target_components == 3
        tile_masks.append(frag)

    grid = np.zeros((h, w), dtype=int)
    for (r0, c0), mask in zip(visible_positions, tile_masks):
        _paint_tile(grid, r0, c0, mask)

    expected = _expected_output(grid.shape, rows, cols, template)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    info = _infer_structure(grid)
    if info["row_step"] != step or info["col_step"] != step:
        return None
    if not np.array_equal(info["template"], template):
        return None
    if info["row_positions"] != rows or info["col_positions"] != cols:
        return None

    initial_fragments = _components_mask(grid == FG)
    if len(initial_fragments) <= len(info["groups"]):
        return None
    if sum(mask.sum() < template.sum() for mask in tile_masks) < 1:
        return None
    if np.array_equal(_visible_only_output(grid.shape, visible_positions, template), expected):
        return None
    if n * n - n < 2:
        return None
    if not np.any(template == 0):
        return None

    # Keep branch coverage healthy: do not let the generator collapse to only 2-fragment cases.
    if fragmented_count >= 2 and not has_three_way and bool(rng.integers(0, 3) == 0):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragmented-periodic tiling input")
