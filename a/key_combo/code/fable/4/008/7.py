from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 800


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mode_nonblack_first(vals: list[int]) -> int:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    for idx, val in enumerate(vals):
        color = int(val)
        if color == BLACK:
            continue
        counts[color] += 1
        first_seen.setdefault(color, int(idx))
    if counts:
        return max(counts, key=lambda color: (counts[color], -first_seen[color]))
    return BLACK


def _extract_patches(
    crop: np.ndarray, rows: int, cols: int, cell_h: int, cell_w: int, gap: int
) -> list[list[np.ndarray]]:
    patches: list[list[np.ndarray]] = []
    for gr in range(rows):
        row: list[np.ndarray] = []
        r0 = gr * (cell_h + gap)
        for gc in range(cols):
            c0 = gc * (cell_w + gap)
            row.append(crop[r0 : r0 + cell_h, c0 : c0 + cell_w])
        patches.append(row)
    return patches


def _gaps_are_black(crop: np.ndarray, rows: int, cols: int, cell_h: int, cell_w: int, gap: int) -> bool:
    if gap == 0:
        return True
    for gr in range(rows - 1):
        r = (gr + 1) * cell_h + gr * gap
        if bool(np.any(crop[r : r + gap, :] != BLACK)):
            return False
    for gc in range(cols - 1):
        c = (gc + 1) * cell_w + gc * gap
        if bool(np.any(crop[:, c : c + gap] != BLACK)):
            return False
    return True


def _template_from_patches(patches: list[list[np.ndarray]]) -> np.ndarray:
    rows = len(patches)
    cols = len(patches[0])
    cell_h, cell_w = patches[0][0].shape
    template = np.zeros((cell_h, cell_w), dtype=int)
    for r in range(cell_h):
        for c in range(cell_w):
            vals = [int(patches[gr][gc][r, c]) for gr in range(rows) for gc in range(cols)]
            template[r, c] = _mode_nonblack_first(vals)
    return template


def _candidate_defects(
    patches: list[list[np.ndarray]], template: np.ndarray
) -> list[dict[str, object]] | None:
    defects: list[dict[str, object]] = []
    template_fg = template != BLACK
    for gr, row in enumerate(patches):
        for gc, patch in enumerate(row):
            diff = patch != template
            if not bool(diff.any()):
                continue
            if bool(np.any(diff & ((patch != BLACK) | ~template_fg))):
                return None
            defects.append({"grid_pos": (int(gr), int(gc)), "mask": diff.copy()})
    return defects


def _parse_tiling(grid: np.ndarray) -> dict[str, object]:
    box = _bbox(grid != BLACK)
    assert box is not None, "expected at least one tiled sprite"
    br0, bc0, br1, bc1 = box
    crop = grid[br0 : br1 + 1, bc0 : bc1 + 1]
    H, W = crop.shape
    valid: list[dict[str, object]] = []

    for gap in (0, 1):
        for rows in range(2, H + 1):
            inner_h = H - (rows - 1) * gap
            if inner_h < rows or inner_h % rows != 0:
                continue
            cell_h = inner_h // rows
            if cell_h < 2:
                continue
            for cols in range(2, W + 1):
                inner_w = W - (cols - 1) * gap
                if inner_w < cols or inner_w % cols != 0:
                    continue
                cell_w = inner_w // cols
                if cell_w < 2:
                    continue
                if rows * cols < 4:
                    continue
                if not _gaps_are_black(crop, rows, cols, cell_h, cell_w, gap):
                    continue

                patches = _extract_patches(crop, rows, cols, cell_h, cell_w, gap)
                template = _template_from_patches(patches)
                if int(np.count_nonzero(template != BLACK)) < 4:
                    continue
                defects = _candidate_defects(patches, template)
                if defects is None or len(defects) != 2:
                    continue

                (r1, c1) = defects[0]["grid_pos"]  # type: ignore[index]
                (r2, c2) = defects[1]["grid_pos"]  # type: ignore[index]
                mask1 = defects[0]["mask"]  # type: ignore[index]
                mask2 = defects[1]["mask"]  # type: ignore[index]
                if r1 == r2 or c1 == c2:
                    continue
                if np.array_equal(mask1, mask2):
                    continue
                if not bool(mask1.any()) or not bool(mask2.any()):
                    continue

                valid.append(
                    {
                        "origin": (int(br0), int(bc0)),
                        "rows": int(rows),
                        "cols": int(cols),
                        "cell_h": int(cell_h),
                        "cell_w": int(cell_w),
                        "gap": int(gap),
                        "template": template,
                        "defects": defects,
                    }
                )

    assert valid, "expected exactly two defective cells in a regular tiled sprite grid"
    valid.sort(key=lambda item: (-int(item["rows"]) * int(item["cols"]), int(item["cell_h"]) * int(item["cell_w"]), int(item["gap"])))
    best = valid[0]
    best_key = (
        int(best["rows"]) * int(best["cols"]),
        int(best["cell_h"]) * int(best["cell_w"]),
        int(best["gap"]),
    )
    tied = [
        item
        for item in valid
        if (
            int(item["rows"]) * int(item["cols"]),
            int(item["cell_h"]) * int(item["cell_w"]),
            int(item["gap"]),
        )
        == best_key
    ]
    assert len(tied) == 1, "tiling interpretation must be unique"
    return best


def _cell_origin(meta: dict[str, object], gr: int, gc: int) -> tuple[int, int]:
    r0, c0 = meta["origin"]  # type: ignore[misc]
    cell_h = int(meta["cell_h"])
    cell_w = int(meta["cell_w"])
    gap = int(meta["gap"])
    return int(r0) + int(gr) * (cell_h + gap), int(c0) + int(gc) * (cell_w + gap)


def _paint_cell(out: np.ndarray, meta: dict[str, object], gr: int, gc: int, patch: np.ndarray) -> None:
    r0, c0 = _cell_origin(meta, gr, gc)
    h, w = patch.shape
    out[r0 : r0 + h, c0 : c0 + w] = patch


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    meta = _parse_tiling(grid)
    template = meta["template"]  # type: ignore[assignment]
    defects = meta["defects"]  # type: ignore[assignment]

    first, second = defects
    r1, c1 = first["grid_pos"]  # type: ignore[index]
    r2, c2 = second["grid_pos"]  # type: ignore[index]
    mask1 = first["mask"]  # type: ignore[index]
    mask2 = second["mask"]  # type: ignore[index]

    out = grid.copy()
    _paint_cell(out, meta, int(r1), int(c1), template)
    _paint_cell(out, meta, int(r2), int(c2), template)

    cross1 = template.copy()
    cross1[mask1] = BLACK
    _paint_cell(out, meta, int(r1), int(c2), cross1)

    cross2 = template.copy()
    cross2[mask2] = BLACK
    _paint_cell(out, meta, int(r2), int(c1), cross2)
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair two defective tiled sprites and transfer their notches to the row-column crossing cells.

    Input structure:
    - The background is black.
    - One multicolor sprite is repeated as a regular row-by-column tiled grid, with either no separator or a one-cell black separator between adjacent cells.
    - Exactly two cells are defective: each is the same complete sprite with a nonempty black missing-pixel notch, and the two defective cells lie in different tile rows and different tile columns.

    Rule:
    - Infer the complete sprite template from the repeated cells.
    - Restore both original defective cells to the complete sprite.
    - Move each defect along its own tile row to the other defective cell's tile column: the cell at (r1,c2) receives the first row's exact relative notch mask, and the cell at (r2,c1) receives the second row's exact relative notch mask.
    - Leave all other cells, separators, margins, colors, and relative notch positions unchanged.

    Key insight:
    - The defect is conserved but crosses columns, and the notch shape is inherited from the defective cell in the same row rather than from the destination column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"
    out, _meta = _solve_with_meta(g)
    return out


def _notch_masks(cell_h: int, cell_w: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    # Edge notches with varied side, length, depth, and offset. None removes a whole row or column.
    for side in ("top", "bottom"):
        max_depth = min(2, cell_h - 1)
        for depth in range(1, max_depth + 1):
            for length in range(1, min(3, cell_w - 1) + 1):
                for start in range(0, cell_w - length + 1):
                    m = np.zeros((cell_h, cell_w), dtype=bool)
                    rr = slice(0, depth) if side == "top" else slice(cell_h - depth, cell_h)
                    m[rr, start : start + length] = True
                    masks.append(m)
    for side in ("left", "right"):
        max_depth = min(2, cell_w - 1)
        for depth in range(1, max_depth + 1):
            for length in range(1, min(3, cell_h - 1) + 1):
                for start in range(0, cell_h - length + 1):
                    m = np.zeros((cell_h, cell_w), dtype=bool)
                    cc = slice(0, depth) if side == "left" else slice(cell_w - depth, cell_w)
                    m[start : start + length, cc] = True
                    masks.append(m)
    # A few corner L notches make area-only or side-only shortcuts fail.
    for r_edge in (0, cell_h - 1):
        for c_edge in (0, cell_w - 1):
            inward_r = 1 if r_edge == 0 else -1
            inward_c = 1 if c_edge == 0 else -1
            m = np.zeros((cell_h, cell_w), dtype=bool)
            m[r_edge, c_edge] = True
            m[r_edge + inward_r, c_edge] = True
            m[r_edge, c_edge + inward_c] = True
            masks.append(m)
    unique: dict[bytes, np.ndarray] = {}
    for m in masks:
        unique[m.tobytes()] = m
    return list(unique.values())


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    cells = {(int(r), int(c)) for r, c in pts}
    q: deque[tuple[int, int]] = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _make_sprite(cell_h: int, cell_w: int) -> np.ndarray:
    palette = random.sample(COLORS, random.randint(3, 5))
    sprite = np.zeros((cell_h, cell_w), dtype=int)
    for r in range(cell_h):
        for c in range(cell_w):
            base = (2 * r + 3 * c + (r * c) % len(palette)) % len(palette)
            if random.random() < 0.30:
                base = random.randrange(len(palette))
            sprite[r, c] = int(palette[base])
    # Force asymmetric anchors so smaller accidental periods are rejected by the parser.
    anchors = [(0, 0), (0, cell_w - 1), (cell_h - 1, 0), (cell_h - 1, cell_w - 1)]
    random.shuffle(anchors)
    for idx, (r, c) in enumerate(anchors[: min(len(palette), 4)]):
        sprite[r, c] = int(palette[idx])
    return sprite


def _render_full(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    gap: int,
    margin_r: int,
    margin_c: int,
    sprite: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    tiled_h = rows * cell_h + (rows - 1) * gap
    tiled_w = cols * cell_w + (cols - 1) * gap
    grid = np.zeros((tiled_h + margin_r + random.randint(0, 2), tiled_w + margin_c + random.randint(0, 2)), dtype=int)
    meta = {
        "origin": (int(margin_r), int(margin_c)),
        "rows": int(rows),
        "cols": int(cols),
        "cell_h": int(cell_h),
        "cell_w": int(cell_w),
        "gap": int(gap),
        "template": sprite.copy(),
    }
    for gr in range(rows):
        for gc in range(cols):
            r0 = margin_r + gr * (cell_h + gap)
            c0 = margin_c + gc * (cell_w + gap)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = sprite
    return grid, meta


def _expected_from_specs(
    inp: np.ndarray,
    meta: dict[str, object],
    a: tuple[int, int, np.ndarray],
    b: tuple[int, int, np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    sprite = meta["template"]  # type: ignore[assignment]
    ar, ac, am = a
    br, bc, bm = b
    _paint_cell(out, meta, ar, ac, sprite)
    _paint_cell(out, meta, br, bc, sprite)
    cross_a = sprite.copy()
    cross_a[am] = BLACK
    _paint_cell(out, meta, ar, bc, cross_a)
    cross_b = sprite.copy()
    cross_b[bm] = BLACK
    _paint_cell(out, meta, br, ac, cross_b)
    return out


def _apply_defect(grid: np.ndarray, meta: dict[str, object], gr: int, gc: int, mask: np.ndarray) -> None:
    r0, c0 = _cell_origin(meta, gr, gc)
    sub = grid[r0 : r0 + int(meta["cell_h"]), c0 : c0 + int(meta["cell_w"])]
    sub[mask] = BLACK


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: tuple[tuple[int, int, np.ndarray], tuple[int, int, np.ndarray]]) -> bool:
    try:
        solved, parsed = _solve_with_meta(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    defects = parsed["defects"]  # type: ignore[assignment]
    masks = [d["mask"] for d in defects]  # type: ignore[index]
    if len(masks) != 2 or np.array_equal(masks[0], masks[1]):
        return False

    a, b = specs
    if a[0] == b[0] or a[1] == b[1]:
        return False
    if np.array_equal(a[2], b[2]):
        return False

    repair_only = inp.copy()
    sprite = parsed["template"]  # type: ignore[assignment]
    for d in defects:
        gr, gc = d["grid_pos"]  # type: ignore[index]
        _paint_cell(repair_only, parsed, int(gr), int(gc), sprite)
    if np.array_equal(repair_only, expected):
        return False

    wrong_axis = repair_only.copy()
    ar, ac, am = a
    br, bc, bm = b
    swap_a = sprite.copy()
    swap_a[bm] = BLACK
    _paint_cell(wrong_axis, parsed, ar, bc, swap_a)
    swap_b = sprite.copy()
    swap_b[am] = BLACK
    _paint_cell(wrong_axis, parsed, br, ac, swap_b)
    if np.array_equal(wrong_axis, expected):
        return False

    return bool(np.any(inp != expected))


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    gap = random.randint(0, 1)
    cell_h = random.randint(4, 6)
    cell_w = random.randint(4, 6)

    max_r_margin = 30 - (rows * cell_h + (rows - 1) * gap)
    max_c_margin = 30 - (cols * cell_w + (cols - 1) * gap)
    if max_r_margin < 0 or max_c_margin < 0:
        return None
    top_margin = random.randint(0, min(2, max_r_margin))
    left_margin = random.randint(0, min(2, max_c_margin))

    sprite = _make_sprite(cell_h, cell_w)
    inp, meta = _render_full(rows, cols, cell_h, cell_w, gap, top_margin, left_margin, sprite)
    if max(inp.shape) > 30:
        return None

    r1, r2 = random.sample(range(rows), 2)
    c1, c2 = random.sample(range(cols), 2)
    masks = [m for m in _notch_masks(cell_h, cell_w) if _connected(m)]
    random.shuffle(masks)
    if len(masks) < 2:
        return None
    mask1 = masks[0]
    different_masks = [m for m in masks[1:] if not np.array_equal(m, mask1)]
    if not different_masks:
        return None
    different_area = [m for m in different_masks if int(m.sum()) != int(mask1.sum())]
    mask2 = random.choice(different_area or different_masks)
    if np.array_equal(mask1, mask2):
        return None

    _apply_defect(inp, meta, r1, c1, mask1)
    _apply_defect(inp, meta, r2, c2, mask2)

    expected = _expected_from_specs(inp, meta, (r1, c1, mask1), (r2, c2, mask2))
    if not _semantic_checks(inp, expected, ((r1, c1, mask1), (r2, c2, mask2))):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid defective-transfer tiled sprite grid")
