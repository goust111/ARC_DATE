from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 800
LAYOUT_RETRIES = 250
RECT_RETRIES = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _rect_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _tile_grid(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    th, tw = tile.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % th, c % tw])
    return out


def _draw_canvas(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _draw_nested_boxes(grid: np.ndarray, rect: tuple[int, int, int, int], colors: list[int]) -> None:
    r0, c0, r1, c1 = rect
    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    inner_r1 = r1 - 1
    inner_c1 = c1 - 1
    grid[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] = BLACK
    center_r = (inner_r0 + inner_r1) // 2
    center_c = (inner_c0 + inner_c1) // 2
    for layer, color in enumerate(colors):
        rr0 = center_r - layer
        rr1 = center_r + layer
        cc0 = center_c - layer
        cc1 = center_c + layer
        grid[rr0, cc0 : cc1 + 1] = color
        grid[rr1, cc0 : cc1 + 1] = color
        grid[rr0 : rr1 + 1, cc0] = color
        grid[rr0 : rr1 + 1, cc1] = color


def _candidate_canvas_rects(h: int, w: int) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for side in range(5, min(h, w) + 1, 2):
        rects.extend(
            [
                (0, 0, side - 1, side - 1),
                (0, w - side, side - 1, w - 1),
                (h - side, 0, h - 1, side - 1),
                (h - side, w - side, h - 1, w - 1),
            ]
        )
    return rects


def _detect_canvas(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    h, w = grid.shape
    matches: list[tuple[int, int, int, int]] = []
    for rect in _candidate_canvas_rects(h, w):
        r0, c0, r1, c1 = rect
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not bool(np.all(patch[0, :] == GREY)):
            continue
        if not bool(np.all(patch[-1, :] == GREY)):
            continue
        if not bool(np.all(patch[:, 0] == GREY)):
            continue
        if not bool(np.all(patch[:, -1] == GREY)):
            continue
        if not bool(np.all(patch[1:-1, 1:-1] == BLACK)):
            continue
        matches.append(rect)
    assert len(matches) == 1, "expected exactly one corner canvas"
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = matches[0]
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return matches[0], mask


def _majority_color(counts: dict[int, int]) -> int:
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _evaluate_period(
    grid: np.ndarray, canvas_mask: np.ndarray, tile_h: int, tile_w: int
) -> dict[str, object]:
    h, w = grid.shape
    buckets: list[list[dict[int, int]]] = [[{} for _ in range(tile_w)] for _ in range(tile_h)]
    coverage = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(h):
        for c in range(w):
            if bool(canvas_mask[r, c]):
                continue
            bucket = buckets[r % tile_h][c % tile_w]
            color = int(grid[r, c])
            bucket[color] = bucket.get(color, 0) + 1
            coverage[r % tile_h, c % tile_w] += 1
    if bool((coverage == 0).any()):
        return {
            "valid": False,
            "score": (-1, -1, -1, -1, -1, -1),
            "tile": np.zeros((tile_h, tile_w), dtype=int),
            "pred": np.zeros_like(grid),
            "occluders": [],
        }

    tile = np.zeros((tile_h, tile_w), dtype=int)
    for rr in range(tile_h):
        for cc in range(tile_w):
            tile[rr, cc] = _majority_color(buckets[rr][cc])

    pred = _tile_grid(tile, h, w)
    mismatch = (~canvas_mask) & (grid != pred)
    tile_colors = {int(v) for v in tile.ravel()}
    occluders: list[dict[str, object]] = []
    used_colors: set[int] = set()
    valid = True
    for comp in _components(mismatch):
        r0, c0, r1, c1 = _rect_bbox(comp)
        color = int(grid[comp[0][0], comp[0][1]])
        if color in tile_colors or color in {BLACK, GREY}:
            valid = False
            break
        if color in used_colors:
            valid = False
            break
        if any(int(grid[r, c]) != color for r, c in comp):
            valid = False
            break
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(comp) != area:
            valid = False
            break
        occluders.append(
            {
                "color": color,
                "area": area,
                "bbox": (r0, c0, r1, c1),
            }
        )
        used_colors.add(color)
    if not (2 <= len(occluders) <= 4):
        valid = False

    mismatch_area = int(mismatch.sum())
    score = (
        1 if valid else 0,
        -mismatch_area,
        -(tile_h * tile_w),
        -tile_h,
        -tile_w,
        -len(occluders),
    )
    return {"valid": valid, "score": score, "tile": tile, "pred": pred, "occluders": occluders}


def _infer_pattern_and_occluders(
    grid: np.ndarray, canvas_mask: np.ndarray
) -> tuple[np.ndarray, list[dict[str, object]]]:
    best: dict[str, object] | None = None
    fallback: dict[str, object] | None = None
    for tile_h in range(1, 5):
        for tile_w in range(1, 5):
            cand = _evaluate_period(grid, canvas_mask, tile_h, tile_w)
            if fallback is None or tuple(cand["score"]) > tuple(fallback["score"]):
                fallback = cand
            if bool(cand["valid"]) and (best is None or tuple(cand["score"]) > tuple(best["score"])):
                best = cand
    chosen = best if best is not None else fallback
    assert chosen is not None
    return np.asarray(chosen["pred"], dtype=int), list(chosen["occluders"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a periodic tiling hidden by colored rectangles, then encode the rectangle sizes as nested boxes in the corner canvas.

    Input structure:
    - Most of the grid is filled by a small periodic tile pattern using 2-3 colors.
    - A corner contains a square black canvas surrounded by a grey 1-cell frame.
    - Two to four solid monochrome rectangles with colors outside the tile palette cover parts of the periodic region.

    Rule:
    - Detect the grey-framed corner canvas and exclude it from period inference.
    - Infer the smallest repeating tile from the visible non-canvas cells by taking the majority color in each row/column residue class, then use that tile to restore all covered cells.
    - Treat the cells that disagree with the restored pattern as the occluding rectangles, sort those rectangles by area from smallest to largest, and draw their colors as nested square boxes inside the canvas: 1x1 in the center, then 3x3, 5x5, and so on.

    Key insight:
    - The puzzle separates into two linked deductions: periodic repair identifies which cells belong to the hidden tiling, and the repaired-vs-input differences reveal the rectangles whose ordered areas determine the nested-box code.
    """
    g = np.asarray(grid, dtype=int)
    canvas_rect, canvas_mask = _detect_canvas(g)
    pred, occluders = _infer_pattern_and_occluders(g, canvas_mask)

    out = pred.copy()
    out[canvas_mask] = g[canvas_mask]

    sorted_occluders = sorted(
        occluders,
        key=lambda item: (int(item["area"]), int(item["bbox"][0]), int(item["bbox"][1])),
    )
    r0, c0, r1, c1 = canvas_rect
    inner_side = r1 - r0 - 1
    assert inner_side == 2 * len(sorted_occluders) - 1, "canvas size must match rectangle count"
    _draw_nested_boxes(out, canvas_rect, [int(item["color"]) for item in sorted_occluders])
    return out


def _canvas_rect_for_corner(corner: str, h: int, w: int, side: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, side - 1, side - 1
    if corner == "tr":
        return 0, w - side, side - 1, w - 1
    if corner == "bl":
        return h - side, 0, h - 1, side - 1
    return h - side, w - side, h - 1, w - 1


def _row_shift_period(tile: np.ndarray, shift: int) -> bool:
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) != int(tile[(r + shift) % h, c]):
                return False
    return True


def _col_shift_period(tile: np.ndarray, shift: int) -> bool:
    h, w = tile.shape
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) != int(tile[r, (c + shift) % w]):
                return False
    return True


def _tile_has_minimal_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for shift in range(1, h):
        if _row_shift_period(tile, shift):
            return False
    for shift in range(1, w):
        if _col_shift_period(tile, shift):
            return False
    return True


def _sample_tile() -> np.ndarray:
    for _ in range(LAYOUT_RETRIES):
        tile_h = random.randint(2, 4)
        tile_w = random.randint(2, 4)
        n_colors = random.randint(2, 3)
        colors = random.sample(COLOR_POOL, n_colors)
        tile = np.array(
            [[random.choice(colors) for _ in range(tile_w)] for _ in range(tile_h)],
            dtype=int,
        )
        if len({int(v) for v in tile.ravel()}) != n_colors:
            continue
        if _tile_has_minimal_period(tile):
            return tile
    raise RuntimeError("failed to sample a non-degenerate tile")


def _place_rectangles(
    h: int, w: int, canvas_rect: tuple[int, int, int, int], count: int
) -> list[tuple[int, int, int, int]]:
    max_h = max(2, min(6, h - 4))
    max_w = max(2, min(7, w - 4))
    for _ in range(RECT_RETRIES):
        rects: list[tuple[int, int, int, int]] = []
        used_areas: set[int] = set()
        ok = True
        for _ in range(count):
            placed = False
            for _ in range(RECT_RETRIES):
                rh = random.randint(1, max_h)
                rw = random.randint(1, max_w)
                area = rh * rw
                if area in used_areas:
                    continue
                r0 = random.randint(1, h - rh - 1)
                c0 = random.randint(1, w - rw - 1)
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if _rect_intersects(rect, canvas_rect, margin=1):
                    continue
                if any(_rect_intersects(rect, other, margin=1) for other in rects):
                    continue
                rects.append(rect)
                used_areas.add(area)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return rects
    raise RuntimeError("failed to place non-overlapping rectangles")


def _visible_residue_ok(
    h: int,
    w: int,
    tile_h: int,
    tile_w: int,
    canvas_rect: tuple[int, int, int, int],
    rects: list[tuple[int, int, int, int]],
) -> bool:
    visible = np.ones((h, w), dtype=bool)
    cr0, cc0, cr1, cc1 = canvas_rect
    visible[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
    for r0, c0, r1, c1 in rects:
        visible[r0 : r1 + 1, c0 : c1 + 1] = False
    for rr in range(tile_h):
        for cc in range(tile_w):
            if int(visible[rr::tile_h, cc::tile_w].sum()) < 2:
                return False
    return True


def _generate_pair() -> tuple[np.ndarray, np.ndarray]:
    tile = _sample_tile()
    tile_h, tile_w = tile.shape
    rect_count = random.randint(2, 4)
    inner_side = 2 * rect_count - 1
    canvas_side = inner_side + 2
    min_size = max(12, canvas_side + 5)
    h = random.randint(min_size, 30)
    w = random.randint(min_size, 30)
    canvas_corner = random.choice(("tl", "tr", "bl", "br"))
    canvas_rect = _canvas_rect_for_corner(canvas_corner, h, w, canvas_side)
    rects = _place_rectangles(h, w, canvas_rect, rect_count)
    if not _visible_residue_ok(h, w, tile_h, tile_w, canvas_rect, rects):
        raise RuntimeError("insufficient visible residues")

    base = _tile_grid(tile, h, w)
    tile_colors = {int(v) for v in tile.ravel()}
    available = [c for c in COLOR_POOL if c not in tile_colors]
    rect_colors = random.sample(available, rect_count)

    inp = base.copy()
    out = base.copy()
    _draw_canvas(inp, canvas_rect)
    _draw_canvas(out, canvas_rect)

    rect_specs: list[dict[str, object]] = []
    for color, rect in zip(rect_colors, rects, strict=False):
        r0, c0, r1, c1 = rect
        inp[r0 : r1 + 1, c0 : c1 + 1] = color
        rect_specs.append({"color": color, "area": (r1 - r0 + 1) * (c1 - c0 + 1), "bbox": rect})

    sorted_colors = [
        int(item["color"])
        for item in sorted(rect_specs, key=lambda item: (int(item["area"]), int(item["bbox"][0]), int(item["bbox"][1])))
    ]
    _draw_nested_boxes(out, canvas_rect, sorted_colors)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, out = _generate_pair()
        except RuntimeError:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, out):
            return inp
    raise RuntimeError("failed to generate a valid periodic-repair ARC sample")
