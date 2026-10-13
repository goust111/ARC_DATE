import random
from collections import Counter, deque

import numpy as np


BG = 8
TOOTH = 9
MARK = 3
BODY_COLORS = [0, 1, 2, 4, 5, 6, 7]
SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
GEN_ATTEMPTS = 300
PLACE_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            comps.append({"bbox": (r0, c0, r1, c1), "coords": coords, "patch": patch})
    return comps


def _body_color(patch: np.ndarray, bg: int) -> int:
    vals = [
        int(v)
        for v in patch.reshape(-1)
        if int(v) not in (bg, TOOTH, MARK)
    ]
    assert vals, "each piece must contain a visible body color"
    return int(Counter(vals).most_common(1)[0][0])


def _clean_piece(patch: np.ndarray, body: int) -> np.ndarray:
    out = patch.copy()
    out[(out == TOOTH) | (out == MARK)] = body
    return out


def _side_descriptor(patch: np.ndarray, bg: int, side: str) -> dict:
    mask = patch != bg
    h, w = patch.shape
    if side in ("left", "right"):
        profile = []
        boundary = []
        for r in range(h):
            cols = np.flatnonzero(mask[r])
            assert cols.size > 0, "piece bbox rows must stay occupied"
            c = int(cols[0] if side == "left" else cols[-1])
            profile.append(c)
            boundary.append((r, c))
        signature = tuple(
            int(c) for c in (profile if side == "left" else [w - 1 - c for c in profile])
        )
        span = h
    else:
        profile = []
        boundary = []
        for c in range(w):
            rows = np.flatnonzero(mask[:, c])
            assert rows.size > 0, "piece bbox columns must stay occupied"
            r = int(rows[0] if side == "top" else rows[-1])
            profile.append(r)
            boundary.append((r, c))
        signature = tuple(
            int(r) for r in (profile if side == "top" else [h - 1 - r for r in profile])
        )
        span = w
    toothed = all(int(patch[r, c]) == TOOTH for r, c in boundary)
    return {
        "side": side,
        "profile": tuple(int(x) for x in profile),
        "signature": signature,
        "span": int(span),
        "toothed": bool(toothed),
    }


def _occupied_set(patch: np.ndarray, bg: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(patch != bg)
    }


def _match_delta(desc_a: dict, desc_b: dict) -> tuple[int, int] | None:
    pa = desc_a["profile"]
    pb = desc_b["profile"]
    if desc_a["side"] == "right" and desc_b["side"] == "left":
        vals = [a + 1 - b for a, b in zip(pa, pb)]
        return (0, int(vals[0])) if len(set(vals)) == 1 else None
    if desc_a["side"] == "left" and desc_b["side"] == "right":
        vals = [a - 1 - b for a, b in zip(pa, pb)]
        return (0, int(vals[0])) if len(set(vals)) == 1 else None
    if desc_a["side"] == "bottom" and desc_b["side"] == "top":
        vals = [a + 1 - b for a, b in zip(pa, pb)]
        return (int(vals[0]), 0) if len(set(vals)) == 1 else None
    if desc_a["side"] == "top" and desc_b["side"] == "bottom":
        vals = [a - 1 - b for a, b in zip(pa, pb)]
        return (int(vals[0]), 0) if len(set(vals)) == 1 else None
    return None


def _touching_without_overlap(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]], delta: tuple[int, int]) -> bool:
    dr, dc = delta
    shifted_b = {(r + dr, c + dc) for r, c in cells_b}
    if not cells_a.isdisjoint(shifted_b):
        return False
    for r, c in cells_a:
        if (
            (r - 1, c) in shifted_b
            or (r + 1, c) in shifted_b
            or (r, c - 1) in shifted_b
            or (r, c + 1) in shifted_b
        ):
            return True
    return False


def _extract_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    pieces: list[dict] = []
    for comp in _components(grid, bg):
        patch = comp["patch"]
        body = _body_color(patch, bg)
        clean = _clean_piece(patch, body)
        sides = {side: _side_descriptor(patch, bg, side) for side in SIDES}
        pieces.append(
            {
                "patch": patch,
                "clean": clean,
                "body": int(body),
                "bbox": comp["bbox"],
                "sides": sides,
                "cells_raw": _occupied_set(patch, bg),
                "paint_cells": [
                    (int(r), int(c), int(clean[r, c]))
                    for r, c in np.argwhere(clean != bg)
                ],
                "has_marker": bool(np.any(patch == MARK)),
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble maroon-serrated fragments into the unique full rectangle, anchored by the green marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    pieces = _extract_pieces(g, bg)
    assert pieces, "expected at least one fragment"

    anchor_ids = [i for i, p in enumerate(pieces) if p["has_marker"]]
    assert len(anchor_ids) == 1, "expected exactly one anchor fragment"
    anchor_idx = anchor_ids[0]

    matches: dict[tuple[int, str], tuple[int, tuple[int, int]]] = {}
    for i, piece_a in enumerate(pieces):
        for side in SIDES:
            desc_a = piece_a["sides"][side]
            if not desc_a["toothed"]:
                continue
            target_side = OPPOSITE[side]
            candidates: list[tuple[int, tuple[int, int]]] = []
            for j, piece_b in enumerate(pieces):
                if i == j:
                    continue
                desc_b = piece_b["sides"][target_side]
                if not desc_b["toothed"]:
                    continue
                if desc_a["span"] != desc_b["span"]:
                    continue
                delta = _match_delta(desc_a, desc_b)
                if delta is None:
                    continue
                if not _touching_without_overlap(piece_a["cells_raw"], piece_b["cells_raw"], delta):
                    continue
                candidates.append((j, delta))
            assert len(candidates) == 1, "each toothed edge must have a unique match"
            matches[(i, side)] = candidates[0]

    for (i, side), (j, delta) in matches.items():
        back = matches.get((j, OPPOSITE[side]))
        assert back is not None and back[0] == i and back[1] == (-delta[0], -delta[1])

    marker_coords = np.argwhere(pieces[anchor_idx]["patch"] == MARK)
    assert marker_coords.size > 0, "anchor must contain green marker"
    mr0, mc0 = marker_coords.min(axis=0)
    placements: dict[int, tuple[int, int]] = {anchor_idx: (-int(mr0), -int(mc0))}
    queue = deque([anchor_idx])

    while queue:
        idx = queue.popleft()
        base_r, base_c = placements[idx]
        for side in SIDES:
            if (idx, side) not in matches:
                continue
            nbr, (dr, dc) = matches[(idx, side)]
            pos = (base_r + dr, base_c + dc)
            if nbr in placements:
                assert placements[nbr] == pos, "connector graph implies inconsistent placement"
            else:
                placements[nbr] = pos
                queue.append(nbr)

    assert len(placements) == len(pieces), "all fragments must connect to the anchor"

    min_r = min(pos[0] + r for idx, pos in placements.items() for r, _c, _v in pieces[idx]["paint_cells"])
    min_c = min(pos[1] + c for idx, pos in placements.items() for _r, c, _v in pieces[idx]["paint_cells"])
    max_r = max(pos[0] + r for idx, pos in placements.items() for r, _c, _v in pieces[idx]["paint_cells"])
    max_c = max(pos[1] + c for idx, pos in placements.items() for _r, c, _v in pieces[idx]["paint_cells"])

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    for idx, (pr, pc) in placements.items():
        for r, c, val in pieces[idx]["paint_cells"]:
            tr = pr + r - min_r
            tc = pc + c - min_c
            assert int(out[tr, tc]) == bg, "assembled fragments must not overlap"
            out[tr, tc] = val

    assert np.all(out != bg), "assembled output must be a full rectangle"
    return out


def _random_pattern(length: int, used: set[tuple[int, ...]], rng: random.Random) -> tuple[int, ...]:
    for _ in range(PLACE_ATTEMPTS):
        pattern = tuple(rng.randint(0, 1) for _ in range(length))
        ones = sum(pattern)
        if ones < 2 or ones > length - 2:
            continue
        if len(set(pattern)) < 2:
            continue
        if pattern in used:
            continue
        used.add(pattern)
        return pattern
    raise RuntimeError("failed to sample a unique tooth pattern")


def _make_vertical_piece(
    height: int,
    width: int,
    left_pattern: tuple[int, ...] | None,
    right_pattern: tuple[int, ...] | None,
    color: int,
    marker_shape: tuple[int, int] | None,
) -> np.ndarray:
    extra = 1 if right_pattern is not None else 0
    patch = np.full((height, width + extra), BG, dtype=int)
    for r in range(height):
        left_inset = 1 if left_pattern is not None and left_pattern[r] else 0
        patch[r, left_inset:width] = color
        if right_pattern is not None and right_pattern[r]:
            patch[r, width] = color
        if left_pattern is not None:
            patch[r, left_inset] = TOOTH
        if right_pattern is not None:
            patch[r, width if right_pattern[r] else width - 1] = TOOTH
    if marker_shape is not None:
        mh, mw = marker_shape
        patch[:mh, :mw] = MARK
    return patch


def _make_horizontal_piece(
    height: int,
    width: int,
    top_pattern: tuple[int, ...] | None,
    bottom_pattern: tuple[int, ...] | None,
    color: int,
    marker_shape: tuple[int, int] | None,
) -> np.ndarray:
    extra = 1 if bottom_pattern is not None else 0
    patch = np.full((height + extra, width), BG, dtype=int)
    for c in range(width):
        top_inset = 1 if top_pattern is not None and top_pattern[c] else 0
        patch[top_inset:height, c] = color
        if bottom_pattern is not None and bottom_pattern[c]:
            patch[height, c] = color
        if top_pattern is not None:
            patch[top_inset, c] = TOOTH
        if bottom_pattern is not None:
            patch[height if bottom_pattern[c] else height - 1, c] = TOOTH
    if marker_shape is not None:
        mh, mw = marker_shape
        patch[:mh, :mw] = MARK
    return patch


def _assemble_expected(
    pieces: list[np.ndarray],
    colors: list[int],
    orientation: str,
    core_sizes: list[int],
) -> np.ndarray:
    if orientation == "vertical":
        out = np.full((pieces[0].shape[0], sum(core_sizes)), BG, dtype=int)
        x = 0
        for patch, color, width in zip(pieces, colors, core_sizes):
            clean = _clean_piece(patch, color)
            for r, c in np.argwhere(clean != BG):
                rr, cc = int(r), x + int(c)
                assert int(out[rr, cc]) == BG
                out[rr, cc] = int(clean[r, c])
            x += width
    else:
        out = np.full((sum(core_sizes), pieces[0].shape[1]), BG, dtype=int)
        y = 0
        for patch, color, height in zip(pieces, colors, core_sizes):
            clean = _clean_piece(patch, color)
            for r, c in np.argwhere(clean != BG):
                rr, cc = y + int(r), int(c)
                assert int(out[rr, cc]) == BG
                out[rr, cc] = int(clean[r, c])
            y += height
    assert np.all(out != BG)
    return out


def _boxes_far_enough(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _scatter_pieces(pieces: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in pieces)
    max_h = max(int(p.shape[0]) for p in pieces)
    max_w = max(int(p.shape[1]) for p in pieces)
    base = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 2.6))))
    canvas_h = base + rng.randint(-2, 3)
    canvas_w = base + rng.randint(-2, 3)
    canvas_h = max(canvas_h, max_h + 2)
    canvas_w = max(canvas_w, max_w + 2)
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)

    order = sorted(range(len(pieces)), key=lambda i: int(pieces[i].shape[0] * pieces[i].shape[1]), reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        patch = pieces[idx]
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = rng.randint(0, canvas_h - ph)
            c0 = rng.randint(0, canvas_w - pw)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_boxes_far_enough(box, other) for other in placed_boxes):
                placements[idx] = (r0, c0)
                placed_boxes.append(box)
                placed = True
                break
        if not placed:
            return None

    for idx, (r0, c0) in placements.items():
        patch = pieces[idx]
        ph, pw = patch.shape
        sub = canvas[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != BG
        if np.any(sub[mask] != BG):
            return None
        sub[mask] = patch[mask]
    return canvas


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["vertical", "horizontal"])
    piece_count = rng.randint(3, 5)
    colors = rng.sample(BODY_COLORS, piece_count)
    marker_shape = rng.choice([(2, 3), (3, 2)])

    pieces: list[np.ndarray] = []
    core_sizes: list[int] = []
    used_patterns: set[tuple[int, ...]] = set()

    if orientation == "vertical":
        height = rng.randint(6, 10)
        widths = [rng.randint(4, 6) for _ in range(piece_count)]
        patterns = [_random_pattern(height, used_patterns, rng) for _ in range(piece_count - 1)]
        for i in range(piece_count):
            piece = _make_vertical_piece(
                height=height,
                width=widths[i],
                left_pattern=patterns[i - 1] if i > 0 else None,
                right_pattern=patterns[i] if i < piece_count - 1 else None,
                color=colors[i],
                marker_shape=marker_shape if i == 0 else None,
            )
            pieces.append(piece)
        core_sizes = widths
    else:
        width = rng.randint(6, 10)
        heights = [rng.randint(4, 6) for _ in range(piece_count)]
        patterns = [_random_pattern(width, used_patterns, rng) for _ in range(piece_count - 1)]
        for i in range(piece_count):
            piece = _make_horizontal_piece(
                height=heights[i],
                width=width,
                top_pattern=patterns[i - 1] if i > 0 else None,
                bottom_pattern=patterns[i] if i < piece_count - 1 else None,
                color=colors[i],
                marker_shape=marker_shape if i == 0 else None,
            )
            pieces.append(piece)
        core_sizes = heights

    expected = _assemble_expected(pieces, colors, orientation, core_sizes)
    grid = _scatter_pieces(pieces, rng)
    if grid is None:
        return None

    if len(_components(grid, BG)) != piece_count:
        return None
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
