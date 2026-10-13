import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNER_OFFSETS = {
    "TL": lambda ih, iw, ph, pw: (0, 0),
    "TR": lambda ih, iw, ph, pw: (0, iw - pw),
    "BL": lambda ih, iw, ph, pw: (ih - ph, 0),
    "BR": lambda ih, iw, ph, pw: (ih - ph, iw - pw),
}
ALL_CORNERS = ("TL", "TR", "BL", "BR")
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.lexsort((vals, -counts))
    return int(vals[int(order[0])])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _frame_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    best = None
    best_score = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            if len(coords) != 2 * (hh + ww) - 4:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((hh, ww), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            if np.any(sub[1:-1, 1:-1] != bg):
                continue
            score = (hh * ww, len(coords), -r0, -c0)
            if best_score is None or score > best_score:
                best_score = score
                best = (color, (r0, c0, r1, c1))
    assert best is not None, "expected one hollow rectangular frame"
    color, bbox = best
    return int(color), bbox, _frame_border_mask(grid.shape, bbox)


def _placement_candidates(inner_shape: tuple[int, int], piece_shape: tuple[int, int]) -> list[tuple[int, int]]:
    ih, iw = inner_shape
    ph, pw = piece_shape
    offsets = []
    for name in ALL_CORNERS:
        dr, dc = CORNER_OFFSETS[name](ih, iw, ph, pw)
        if dr < 0 or dc < 0:
            continue
        off = (int(dr), int(dc))
        if off not in offsets:
            offsets.append(off)
    return offsets


def _placement_score(mask: np.ndarray, inner_shape: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
    ih, iw = inner_shape
    dr, dc = offset
    total = 0
    touched = set()
    for r, c in np.argwhere(mask):
        rr = dr + int(r)
        cc = dc + int(c)
        if rr == 0:
            total += 1
            touched.add("T")
        if rr == ih - 1:
            total += 1
            touched.add("B")
        if cc == 0:
            total += 1
            touched.add("L")
        if cc == iw - 1:
            total += 1
            touched.add("R")
    return total, len(touched)


def _infer_offset(mask: np.ndarray, inner_shape: tuple[int, int]) -> tuple[int, int]:
    candidates = _placement_candidates(inner_shape, mask.shape)
    assert candidates, "piece must fit inside template interior"
    scored = []
    for off in candidates:
        score = _placement_score(mask, inner_shape, off)
        scored.append((score, tuple(-x for x in off), off))
    scored.sort(reverse=True)
    return scored[0][2]


def _draw_piece(canvas: np.ndarray, origin: tuple[int, int], mask: np.ndarray, color: int) -> None:
    r0, c0 = origin
    for r, c in np.argwhere(mask):
        canvas[r0 + int(r), c0 + int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered two-color fragments into the hollow template frame.

    Fragments are translation-only corner-anchored shapes. Their shared marker
    color indicates draw priority via marker count; markers disappear after the
    fragment is repainted inside the template using its body color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frame_bbox, frame_mask = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_origin = (fr0 + 1, fc0 + 1)
    inner_shape = (fr1 - fr0 - 1, fc1 - fc0 - 1)

    obj_mask = (g != bg) & (~frame_mask)
    comps = _components(obj_mask)
    assert len(comps) >= 2, "expected at least two fragments"

    present = Counter()
    occurrences = Counter()
    comp_colors = []
    for coords in comps:
        colors = [int(g[r, c]) for r, c in coords]
        color_set = sorted(set(colors) - {bg, frame_color})
        comp_colors.append(color_set)
        for color in set(color_set):
            present[int(color)] += 1
        occurrences.update(colors)

    fragment_count = len(comps)
    shared = [color for color, cnt in present.items() if cnt == fragment_count]
    assert shared, "expected one marker color shared by all fragments"
    marker_color = min(shared, key=lambda color: (occurrences[int(color)], int(color)))

    fragments = []
    for coords in comps:
        mask, _ = _coords_to_mask(coords)
        values = [int(g[r, c]) for r, c in coords]
        colors = set(values)
        assert marker_color in colors and len(colors) == 2, "fragments must be two-color objects"
        body_color = int(next(color for color in colors if color != marker_color))
        marker_count = sum(1 for value in values if value == marker_color)
        offset = _infer_offset(mask, inner_shape)
        fragments.append(
            {
                "mask": mask,
                "body_color": body_color,
                "marker_count": int(marker_count),
                "offset": offset,
            }
        )

    out = np.full_like(g, int(bg))
    out[frame_mask] = int(frame_color)
    for frag in sorted(fragments, key=lambda item: (item["marker_count"], item["body_color"])):
        dr, dc = frag["offset"]
        base_r = inner_origin[0] + dr
        base_c = inner_origin[1] + dc
        _draw_piece(out, (base_r, base_c), frag["mask"], frag["body_color"])
    return out


def _transform_from_tl(mask: np.ndarray, corner: str) -> np.ndarray:
    out = mask
    if corner in ("TR", "BR"):
        out = np.fliplr(out)
    if corner in ("BL", "BR"):
        out = np.flipud(out)
    return out.copy()


def _internal_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[r + dr, c + dc] for dr, dc in DIR4):
                cells.append((int(r), int(c)))
    return cells


def _grow_piece_tl(
    rng: random.Random,
    box_h: int,
    box_w: int,
    arm_h: int,
    arm_w: int,
    fill_ratio: float,
) -> np.ndarray:
    mask = np.zeros((box_h, box_w), dtype=bool)
    mask[:arm_h, :] = True
    mask[:, :arm_w] = True
    max_cells = box_h * box_w - 1
    base_cells = int(mask.sum())
    target = min(max_cells, max(base_cells, int(round(fill_ratio * box_h * box_w))))
    while int(mask.sum()) < target:
        frontier = []
        weights = []
        for r in range(box_h):
            for c in range(box_w):
                if mask[r, c]:
                    continue
                if not any(0 <= r + dr < box_h and 0 <= c + dc < box_w and mask[r + dr, c + dc] for dr, dc in DIR4):
                    continue
                frontier.append((r, c))
                weights.append(1 + r + c)
        if not frontier:
            break
        pick_r, pick_c = rng.choices(frontier, weights=weights, k=1)[0]
        mask[pick_r, pick_c] = True
    return mask


def _piece_offset(corner: str, inner_shape: tuple[int, int], piece_shape: tuple[int, int]) -> tuple[int, int]:
    ih, iw = inner_shape
    ph, pw = piece_shape
    return tuple(int(x) for x in CORNER_OFFSETS[corner](ih, iw, ph, pw))


def _compose_expected(
    canvas_shape: tuple[int, int],
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    pieces: list[dict],
) -> np.ndarray:
    out = np.full(canvas_shape, int(bg), dtype=int)
    frame_mask = _frame_border_mask(canvas_shape, frame_bbox)
    out[frame_mask] = int(frame_color)
    inner_origin = (frame_bbox[0] + 1, frame_bbox[1] + 1)
    for piece in sorted(pieces, key=lambda item: item["marker_count"]):
        dr, dc = piece["offset"]
        _draw_piece(
            out,
            (inner_origin[0] + dr, inner_origin[1] + dc),
            piece["mask"],
            int(piece["body_color"]),
        )
    return out


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_fragment_patch(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    zone: tuple[int, int, int, int],
) -> bool:
    zh0, zw0, zh1, zw1 = zone
    ph, pw = patch.shape
    if zh1 - zh0 + 1 < ph or zw1 - zw0 + 1 < pw:
        return False
    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(zh0, zh1 - ph + 1)
        c0 = rng.randint(zw0, zw1 - pw + 1)
        r1 = r0 + ph - 1
        c1 = c0 + pw - 1
        if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        mask = patch >= 0
        for r, c in np.argwhere(mask):
            val = int(patch[int(r), int(c)])
            if val >= 0:
                grid[r0 + int(r), c0 + int(c)] = val
        _mark_rect(blocked, r0, c0, r1, c1, pad=1)
        return True
    return False


def _make_fragment_patch(mask: np.ndarray, body_color: int, marker_color: int, marker_cells: list[tuple[int, int]]) -> np.ndarray:
    patch = np.full(mask.shape, -1, dtype=int)
    patch[mask] = int(body_color)
    for r, c in marker_cells:
        patch[int(r), int(c)] = int(marker_color)
    return patch


def _choose_corners(rng: random.Random, piece_count: int) -> list[str]:
    if piece_count == 2:
        return list(rng.choice((("TL", "BR"), ("TR", "BL"))))
    missing = rng.choice(ALL_CORNERS)
    return [corner for corner in ALL_CORNERS if corner != missing]


def _generate_piece(
    rng: random.Random,
    corner: str,
    inner_shape: tuple[int, int],
    marker_count: int,
    piece_count: int,
) -> dict | None:
    ih, iw = inner_shape
    min_h = max(4, int(np.ceil((0.68 if piece_count == 2 else 0.56) * ih)))
    min_w = max(4, int(np.ceil((0.68 if piece_count == 2 else 0.56) * iw)))
    box_h = rng.randint(min_h, ih)
    box_w = rng.randint(min_w, iw)
    arm_h = rng.randint(2, min(4, box_h - 1))
    arm_w = rng.randint(2, min(4, box_w - 1))
    fill_ratio = rng.uniform(0.62, 0.84)
    tl_mask = _grow_piece_tl(rng, box_h, box_w, arm_h, arm_w, fill_ratio)
    mask = _transform_from_tl(tl_mask, corner)
    if int(mask.sum()) <= marker_count + 3:
        return None
    internal = _internal_cells(mask)
    if len(internal) < marker_count:
        return None
    offset = _piece_offset(corner, inner_shape, mask.shape)
    inferred = _infer_offset(mask, inner_shape)
    if inferred != offset:
        return None
    return {"mask": mask, "offset": offset, "marker_pool": internal}


def _coverage_ok(inner_shape: tuple[int, int], pieces: list[dict]) -> bool:
    cover = np.zeros(inner_shape, dtype=int)
    for piece in pieces:
        dr, dc = piece["offset"]
        mask = piece["mask"]
        cover[dr : dr + mask.shape[0], dc : dc + mask.shape[1]] += mask.astype(int)
    return bool(np.all(cover >= 1) and np.any(cover >= 2))


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = rng.choice((2, 3))
    inner_h = rng.randint(7, 10)
    inner_w = rng.randint(7, 10)
    inner_shape = (inner_h, inner_w)
    corners = _choose_corners(rng, piece_count)
    marker_counts = [1, 3] if piece_count == 2 else [1, 2, 3]
    rng.shuffle(marker_counts)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    frame_color = rng.choice(palette)
    palette.remove(frame_color)
    marker_color = rng.choice(palette)
    palette.remove(marker_color)
    body_colors = rng.sample(palette, piece_count)

    pieces = []
    for corner, marker_count, body_color in zip(corners, marker_counts, body_colors):
        piece = _generate_piece(rng, corner, inner_shape, marker_count, piece_count)
        if piece is None:
            return None
        marker_cells = rng.sample(piece["marker_pool"], marker_count)
        piece.update(
            {
                "corner": corner,
                "marker_count": marker_count,
                "body_color": int(body_color),
                "marker_cells": marker_cells,
            }
        )
        pieces.append(piece)

    if not _coverage_ok(inner_shape, pieces):
        return None

    max_piece_h = max(piece["mask"].shape[0] for piece in pieces)
    max_piece_w = max(piece["mask"].shape[1] for piece in pieces)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    orientation = rng.choice(("left", "top"))
    if orientation == "left":
        min_h = max(frame_h + 4, max_piece_h + 6, 22)
        min_w = frame_w + max_piece_w + 7
        if min_h > 30 or min_w > 30:
            return None
        grid_h = rng.randint(min_h, 30)
        grid_w = rng.randint(min_w, 30)
        fr0 = rng.randint(1, grid_h - frame_h - 1)
        fc0 = rng.randint(1, 2)
        zone = (1, fc0 + frame_w + 2, grid_h - 2, grid_w - 2)
    else:
        min_h = frame_h + max_piece_h + 7
        min_w = max(frame_w + 4, max_piece_w + 6, 22)
        if min_h > 30 or min_w > 30:
            return None
        grid_h = rng.randint(min_h, 30)
        grid_w = rng.randint(min_w, 30)
        fr0 = rng.randint(1, 2)
        fc0 = rng.randint(1, grid_w - frame_w - 1)
        zone = (fr0 + frame_h + 2, 1, grid_h - 2, grid_w - 2)

    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    frame_mask = _frame_border_mask(grid.shape, frame_bbox)
    grid[frame_mask] = int(frame_color)

    blocked = np.zeros_like(grid, dtype=bool)
    _mark_rect(blocked, frame_bbox[0], frame_bbox[1], frame_bbox[2], frame_bbox[3], pad=1)

    place_order = pieces[:]
    rng.shuffle(place_order)
    for piece in place_order:
        patch = _make_fragment_patch(piece["mask"], piece["body_color"], marker_color, piece["marker_cells"])
        if not _place_fragment_patch(rng, grid, blocked, patch, zone):
            return None

    expected = _compose_expected(grid.shape, bg, frame_color, frame_bbox, pieces)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input after retries")
