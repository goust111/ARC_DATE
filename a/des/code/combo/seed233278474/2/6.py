from __future__ import annotations

from collections import deque
import random

import numpy as np


BG_COLOR = 0
OBSTACLE_COLOR = 5
SIDES = ("top", "bottom", "left", "right")
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _mode_int(values: list[int] | np.ndarray) -> int:
    vals = np.asarray(values, dtype=int)
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _classify_side(piece_bbox: tuple[int, int, int, int], obstacle_bbox: tuple[int, int, int, int]) -> str:
    pr0, pc0, pr1, pc1 = piece_bbox
    or0, oc0, or1, oc1 = obstacle_bbox
    if pr1 < or0:
        return "top"
    if pr0 > or1:
        return "bottom"
    if pc1 < oc0:
        return "left"
    if pc0 > oc1:
        return "right"
    raise AssertionError("piece must be outside the obstacle")


def _boundary_profile(comp: dict, side: str) -> np.ndarray:
    r0, c0, _r1, _c1 = comp["bbox"]
    mask = _mask_from_component(comp)
    if side == "top":
        return r0 + np.array([int(np.flatnonzero(mask[:, j])[0]) for j in range(mask.shape[1])], dtype=int)
    if side == "bottom":
        return r0 + np.array([int(np.flatnonzero(mask[:, j])[-1]) for j in range(mask.shape[1])], dtype=int)
    if side == "left":
        return c0 + np.array([int(np.flatnonzero(mask[i])[0]) for i in range(mask.shape[0])], dtype=int)
    if side == "right":
        return c0 + np.array([int(np.flatnonzero(mask[i])[-1]) for i in range(mask.shape[0])], dtype=int)
    raise ValueError(side)


def _normalized_profile(values: np.ndarray) -> np.ndarray:
    base = _mode_int(values)
    return values - base


def _match_profile(piece_raw: np.ndarray, obstacle_raw: np.ndarray) -> tuple[int, int]:
    piece_norm = _normalized_profile(piece_raw)
    obstacle_norm = _normalized_profile(obstacle_raw)
    matches: list[tuple[int, int]] = []
    plen = len(piece_raw)
    for start in range(len(obstacle_raw) - plen + 1):
        window_norm = obstacle_norm[start : start + plen]
        if not np.array_equal(window_norm, piece_norm):
            continue
        delta = int(obstacle_raw[start] - piece_raw[0])
        if np.all(obstacle_raw[start : start + plen] - piece_raw == delta):
            matches.append((start, delta))
    assert len(matches) == 1, f"expected one profile match, found {len(matches)}"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered monochrome pieces onto the toothed boundary of a central gray obstacle.

    The obstacle is the largest non-background component. Each other component is a piece that
    stays in its original orientation and moves only by translation. A piece's relative side
    (above / below / left / right of the obstacle) determines which obstacle edge it belongs to.
    Its jagged boundary profile is matched to the unique matching interval on that obstacle edge,
    then the piece is translated there and painted over the obstacle/background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_int(g.ravel())
    comps = _find_components(g, bg)
    assert len(comps) >= 3, "expected one obstacle and at least two pieces"

    obstacle = max(comps, key=lambda comp: int(comp["area"]))
    pieces = [comp for comp in comps if comp is not obstacle]
    obstacle_bbox = obstacle["bbox"]

    obstacle_profiles = {
        "top": _boundary_profile(obstacle, "top"),
        "bottom": _boundary_profile(obstacle, "bottom"),
        "left": _boundary_profile(obstacle, "left"),
        "right": _boundary_profile(obstacle, "right"),
    }

    placements: list[tuple[dict, int, int]] = []
    for piece in pieces:
        side = _classify_side(piece["bbox"], obstacle_bbox)
        pr0, pc0, _pr1, _pc1 = piece["bbox"]

        if side == "top":
            piece_raw = _boundary_profile(piece, "bottom")
            start, dr = _match_profile(piece_raw, obstacle_profiles["top"])
            dc = int(obstacle_bbox[1] + start - pc0)
        elif side == "bottom":
            piece_raw = _boundary_profile(piece, "top")
            start, dr = _match_profile(piece_raw, obstacle_profiles["bottom"])
            dc = int(obstacle_bbox[1] + start - pc0)
        elif side == "left":
            piece_raw = _boundary_profile(piece, "right")
            start, dc = _match_profile(piece_raw, obstacle_profiles["left"])
            dr = int(obstacle_bbox[0] + start - pr0)
        else:
            piece_raw = _boundary_profile(piece, "left")
            start, dc = _match_profile(piece_raw, obstacle_profiles["right"])
            dr = int(obstacle_bbox[0] + start - pr0)

        placements.append((piece, int(dr), int(dc)))

    out = g.copy()
    obstacle_color = int(obstacle["color"])
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = bg

    for piece, dr, dc in placements:
        for r, c in piece["coords"]:
            rr = r + dr
            cc = c + dc
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
            assert int(out[rr, cc]) in (bg, obstacle_color), "pieces must land only on bg or obstacle"
            out[rr, cc] = int(piece["color"])

    return out


def _make_code(length: int) -> list[int]:
    while True:
        code = [0] * length
        code[0] = random.choice((-1, 1))
        code[-1] = random.choice((-1, 1))
        for idx in range(1, length - 1):
            code[idx] = random.choices((0, -1, 1), weights=(5, 1, 1), k=1)[0]
        if 0 not in code or 1 not in code or -1 not in code:
            continue
        if code.count(0) <= code.count(1) or code.count(0) <= code.count(-1):
            continue
        return code


def _make_obstacle_mask(
    h: int,
    w: int,
    base_bbox: tuple[int, int, int, int],
    specs: list[dict],
) -> np.ndarray:
    r0, c0, r1, c1 = base_bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True

    for spec in specs:
        side = spec["side"]
        start = spec["start"]
        code = spec["code"]
        if side == "top":
            for idx, val in enumerate(code):
                col = c0 + start + idx
                if val == -1:
                    mask[r0 - 1, col] = True
                elif val == 1:
                    mask[r0, col] = False
        elif side == "bottom":
            for idx, val in enumerate(code):
                col = c0 + start + idx
                if val == -1:
                    mask[r1 + 1, col] = True
                elif val == 1:
                    mask[r1, col] = False
        elif side == "left":
            for idx, val in enumerate(code):
                row = r0 + start + idx
                if val == -1:
                    mask[row, c0 - 1] = True
                elif val == 1:
                    mask[row, c0] = False
        else:
            for idx, val in enumerate(code):
                row = r0 + start + idx
                if val == -1:
                    mask[row, c1 + 1] = True
                elif val == 1:
                    mask[row, c1] = False
    return mask


def _final_piece_coords(spec: dict, base_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = base_bbox
    side = spec["side"]
    start = spec["start"]
    code = spec["code"]
    thickness = spec["thickness"]
    coords: list[tuple[int, int]] = []

    if side == "top":
        top = r0 - thickness
        for idx, val in enumerate(code):
            col = c0 + start + idx
            bottom = r0 + val
            coords.extend((row, col) for row in range(top, bottom + 1))
    elif side == "bottom":
        bottom = r1 + thickness
        for idx, val in enumerate(code):
            col = c0 + start + idx
            top = r1 - val
            coords.extend((row, col) for row in range(top, bottom + 1))
    elif side == "left":
        left = c0 - thickness
        for idx, val in enumerate(code):
            row = r0 + start + idx
            right = c0 + val
            coords.extend((row, col) for col in range(left, right + 1))
    else:
        right = c1 + thickness
        for idx, val in enumerate(code):
            row = r0 + start + idx
            left = c1 - val
            coords.extend((row, col) for col in range(left, right + 1))

    return sorted(set(coords))


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int], int, int]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    rel = sorted((r - r0, c - c0) for r, c in coords)
    return rel, (r0, c0, r1, c1), r1 - r0 + 1, c1 - c0 + 1


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _scatter_piece(
    side: str,
    rel_coords: list[tuple[int, int]],
    shape_h: int,
    shape_w: int,
    grid_h: int,
    grid_w: int,
    obstacle_bbox: tuple[int, int, int, int],
    forbidden_bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]] | None:
    or0, oc0, or1, oc1 = obstacle_bbox

    if side == "top":
        r_lo, r_hi = 1, or0 - shape_h - 2
        c_lo, c_hi = max(1, oc0 - shape_w + 2), min(grid_w - shape_w - 2, oc1 - 1)
        if c_lo > c_hi:
            c_lo, c_hi = 1, grid_w - shape_w - 2
    elif side == "bottom":
        r_lo, r_hi = or1 + 2, grid_h - shape_h - 2
        c_lo, c_hi = max(1, oc0 - shape_w + 2), min(grid_w - shape_w - 2, oc1 - 1)
        if c_lo > c_hi:
            c_lo, c_hi = 1, grid_w - shape_w - 2
    elif side == "left":
        c_lo, c_hi = 1, oc0 - shape_w - 2
        r_lo, r_hi = max(1, or0 - shape_h + 2), min(grid_h - shape_h - 2, or1 - 1)
    else:
        c_lo, c_hi = oc1 + 2, grid_w - shape_w - 2
        r_lo, r_hi = max(1, or0 - shape_h + 2), min(grid_h - shape_h - 2, or1 - 1)

    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(120):
        top = random.randint(r_lo, r_hi)
        left = random.randint(c_lo, c_hi)
        bbox = (top, left, top + shape_h - 1, left + shape_w - 1)
        if any(_bboxes_overlap(bbox, other, gap=1) for other in forbidden_bboxes):
            continue
        return [(top + rr, left + cc) for rr, cc in rel_coords]
    return None


def _render_grid(
    grid_h: int,
    grid_w: int,
    obstacle_mask: np.ndarray,
    piece_coords: list[list[tuple[int, int]]],
    piece_colors: list[int],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    grid[obstacle_mask] = OBSTACLE_COLOR
    for coords, color in zip(piece_coords, piece_colors):
        for r, c in coords:
            grid[r, c] = color
    return grid


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(22, 30)
    grid_w = random.randint(22, 30)
    base_h = random.randint(9, min(14, grid_h - 10))
    base_w = random.randint(9, min(16, grid_w - 10))
    margin = 6
    if grid_h - base_h - 2 * margin < 0 or grid_w - base_w - 2 * margin < 0:
        return None

    base_r0 = random.randint(margin, grid_h - base_h - margin)
    base_c0 = random.randint(margin, grid_w - base_w - margin)
    base_bbox = (base_r0, base_c0, base_r0 + base_h - 1, base_c0 + base_w - 1)

    piece_count = random.randint(2, 4)
    sides = random.sample(SIDES, piece_count)
    colors = random.sample(PALETTE, piece_count)

    specs: list[dict] = []
    for side, color in zip(sides, colors):
        side_len = base_w if side in ("top", "bottom") else base_h
        max_span = min(9, side_len - 4)
        if max_span < 5:
            return None
        span = random.randint(5, max_span)
        start = random.randint(2, side_len - span - 2)
        specs.append(
            {
                "side": side,
                "start": start,
                "code": _make_code(span),
                "thickness": random.randint(2, 4),
                "color": color,
            }
        )

    obstacle_mask = _make_obstacle_mask(grid_h, grid_w, base_bbox, specs)
    obstacle_bbox = _mask_bbox(obstacle_mask)

    final_piece_coords: list[list[tuple[int, int]]] = []
    scattered_piece_coords: list[list[tuple[int, int]]] = []
    forbidden_bboxes = [obstacle_bbox]

    for spec in specs:
        coords = _final_piece_coords(spec, base_bbox)
        rel_coords, _bbox, shape_h, shape_w = _normalize_coords(coords)
        scattered = _scatter_piece(
            spec["side"],
            rel_coords,
            shape_h,
            shape_w,
            grid_h,
            grid_w,
            obstacle_bbox,
            forbidden_bboxes,
        )
        if scattered is None:
            return None
        final_piece_coords.append(coords)
        scattered_piece_coords.append(scattered)
        forbidden_bboxes.append(_bbox_from_coords(scattered))

    inp = _render_grid(
        grid_h,
        grid_w,
        obstacle_mask,
        scattered_piece_coords,
        [spec["color"] for spec in specs],
    )
    expected = _render_grid(
        grid_h,
        grid_w,
        obstacle_mask,
        final_piece_coords,
        [spec["color"] for spec in specs],
    )
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
