from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _signature(comp: dict) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return tuple(sorted((r - r0, c - c0) for r, c in comp["coords"]))


def _touches_corner(comp: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return any(cell in corners for cell in comp["coords"])


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _pair_relation(a: dict, b: dict) -> dict | None:
    if int(a["color"]) == int(b["color"]):
        return None

    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    horizontal = _intervals_overlap(ar0, ar1, br0, br1) and (ac1 < bc0 or bc1 < ac0)
    vertical = _intervals_overlap(ac0, ac1, bc0, bc1) and (ar1 < br0 or br1 < ar0)
    if horizontal == vertical:
        return None

    if horizontal:
        left, right = (a, b) if ac0 < bc0 else (b, a)
        gap = int(right["bbox"][1] - left["bbox"][3] - 1)
        return {"orientation": "H", "first": left, "second": right, "gap": gap}

    top, bottom = (a, b) if ar0 < br0 else (b, a)
    gap = int(bottom["bbox"][0] - top["bbox"][2] - 1)
    return {"orientation": "V", "first": top, "second": bottom, "gap": gap}


def _match_group(group: list[dict]) -> list[dict]:
    assert len(group) % 2 == 0 and group, "each shape signature must occur an even number of times"
    relations: dict[tuple[int, int], dict] = {}

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            rel = _pair_relation(group[i], group[j])
            if rel is not None:
                assert rel["gap"] >= 1, "paired shapes must have a visible gap"
                relations[(i, j)] = rel

    @lru_cache(maxsize=None)
    def rec(remaining: tuple[int, ...]) -> tuple[int, tuple[tuple[int, int], ...]] | None:
        if not remaining:
            return 0, ()

        i = remaining[0]
        best_cost: int | None = None
        best_pairs: tuple[tuple[int, int], ...] | None = None
        unique = True

        for pos in range(1, len(remaining)):
            j = remaining[pos]
            key = (i, j) if i < j else (j, i)
            if key not in relations:
                continue
            tail = remaining[1:pos] + remaining[pos + 1 :]
            tail_result = rec(tail)
            if tail_result is None:
                continue
            tail_cost, tail_pairs = tail_result
            cost = relations[key]["gap"] + tail_cost
            pairs = ((i, j),) + tail_pairs
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pairs = pairs
                unique = True
            elif cost == best_cost:
                unique = False

        if best_cost is None:
            return None
        if not unique:
            raise AssertionError("shape pairing must be unique")
        assert best_pairs is not None
        return best_cost, best_pairs

    result = rec(tuple(range(len(group))))
    assert result is not None, "every shape must find a compatible mate"
    _cost, pair_ids = result
    return [relations[(i, j) if i < j else (j, i)] for i, j in pair_ids]


def _edge_cells(comp: dict, side: str) -> list[tuple[int, int]]:
    coords = comp["coords"]
    if side == "left":
        edge = min(c for _r, c in coords)
        return [(r, c) for r, c in coords if c == edge]
    if side == "right":
        edge = max(c for _r, c in coords)
        return [(r, c) for r, c in coords if c == edge]
    if side == "top":
        edge = min(r for r, _c in coords)
        return [(r, c) for r, c in coords if r == edge]
    if side == "bottom":
        edge = max(r for r, _c in coords)
        return [(r, c) for r, c in coords if r == edge]
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair equal shapes, recolor the sides facing each other with the corner marker color, and remove the marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    marker_candidates = [comp for comp in comps if comp["area"] == 2 and _touches_corner(comp, g.shape)]
    assert len(marker_candidates) == 1, "expected one 2-pixel corner marker"
    marker = marker_candidates[0]
    fusion_color = int(marker["color"])

    shape_comps = [comp for comp in comps if comp is not marker]
    assert shape_comps and len(shape_comps) % 2 == 0
    assert all(4 <= comp["area"] <= 8 for comp in shape_comps)

    by_sig: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in shape_comps:
        by_sig[_signature(comp)].append(comp)

    pairings: list[dict] = []
    for group in by_sig.values():
        pairings.extend(_match_group(group))

    out = g.copy()
    for r, c in marker["coords"]:
        out[r, c] = bg

    for rel in pairings:
        if rel["orientation"] == "H":
            first_side, second_side = "right", "left"
        else:
            first_side, second_side = "bottom", "top"
        for r, c in _edge_cells(rel["first"], first_side):
            out[r, c] = fusion_color
        for r, c in _edge_cells(rel["second"], second_side):
            out[r, c] = fusion_color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
PAIR_COUNT_RANGE = (2, 4)
SHAPE_AREA_RANGE = (4, 8)
SHAPE_BBOX_MAX = 4
GAP_RANGE = (2, 6)
PLACE_RETRIES = 120
SHAPE_RETRIES = 120
GEN_RETRIES = 400
PALETTE = tuple(range(10))


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _sample_shape(existing: set[tuple[tuple[int, int], ...]]) -> tuple[tuple[tuple[int, int], ...], int, int] | None:
    for _ in range(SHAPE_RETRIES):
        area = random.randint(*SHAPE_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for cell in sorted(cells):
                for nbr in _neighbors(cell):
                    if nbr not in cells:
                        frontier.append(nbr)
            cells.add(random.choice(frontier))

        sig = _normalize(cells)
        rows = [r for r, _ in sig]
        cols = [c for _, c in sig]
        bbox_h = max(rows) + 1
        bbox_w = max(cols) + 1
        if bbox_h < 2 or bbox_w < 2:
            continue
        if bbox_h > SHAPE_BBOX_MAX or bbox_w > SHAPE_BBOX_MAX:
            continue
        if sig in existing:
            continue
        return sig, bbox_h, bbox_w
    return None


def _stamp(grid: np.ndarray, sig: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for dr, dc in sig:
        grid[top + dr, left + dc] = color


def _placeable(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    return all(0 <= r < h and 0 <= c < w and not blocked[r, c] for r, c in cells)


def _reserve(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _cells_at(sig: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in sig]


def _marker_cells(h: int, w: int, corner: str, axis: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, 0), (0, 1)] if axis == "h" else [(0, 0), (1, 0)]
    if corner == "tr":
        return [(0, w - 1), (0, w - 2)] if axis == "h" else [(0, w - 1), (1, w - 1)]
    if corner == "bl":
        return [(h - 1, 0), (h - 1, 1)] if axis == "h" else [(h - 1, 0), (h - 2, 0)]
    return [(h - 1, w - 1), (h - 1, w - 2)] if axis == "h" else [(h - 1, w - 1), (h - 2, w - 1)]


def _render_expected(
    grid: np.ndarray,
    bg: int,
    marker_cells: list[tuple[int, int]],
    fusion_color: int,
    placements: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for r, c in marker_cells:
        out[r, c] = bg
    for item in placements:
        sig = item["sig"]
        first_top, first_left = item["first"]
        second_top, second_left = item["second"]
        if item["orientation"] == "H":
            edge_a = max(dc for _dr, dc in sig)
            edge_b = min(dc for _dr, dc in sig)
            for dr, dc in sig:
                if dc == edge_a:
                    out[first_top + dr, first_left + dc] = fusion_color
                if dc == edge_b:
                    out[second_top + dr, second_left + dc] = fusion_color
        else:
            edge_a = max(dr for dr, _dc in sig)
            edge_b = min(dr for dr, _dc in sig)
            for dr, dc in sig:
                if dr == edge_a:
                    out[first_top + dr, first_left + dc] = fusion_color
                if dr == edge_b:
                    out[second_top + dr, second_left + dc] = fusion_color
    return out


def _place_pair(
    blocked: np.ndarray,
    h: int,
    w: int,
    bbox_h: int,
    bbox_w: int,
    orientation: str,
    gap: int,
    sig: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(PLACE_RETRIES):
        if orientation == "H":
            if 2 * bbox_w + gap > w or bbox_h > h:
                return None
            top = random.randint(0, h - bbox_h)
            left = random.randint(0, w - (2 * bbox_w + gap))
            first = (top, left)
            second = (top, left + bbox_w + gap)
        else:
            if 2 * bbox_h + gap > h or bbox_w > w:
                return None
            top = random.randint(0, h - (2 * bbox_h + gap))
            left = random.randint(0, w - bbox_w)
            first = (top, left)
            second = (top + bbox_h + gap, left)

        cells_a = _cells_at(sig, *first)
        cells_b = _cells_at(sig, *second)
        if not _placeable(blocked, cells_a + cells_b):
            continue
        _reserve(blocked, cells_a + cells_b)
        return first, second
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    bg, fusion_color = random.sample(PALETTE, 2)
    remaining_colors = [color for color in PALETTE if color not in {bg, fusion_color}]
    if len(remaining_colors) < 2 * pair_count:
        return None
    shape_colors = random.sample(remaining_colors, 2 * pair_count)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    marker_axis = random.choice(("h", "v"))
    marker_cells = _marker_cells(h, w, corner, marker_axis)
    for r, c in marker_cells:
        grid[r, c] = fusion_color
    _reserve(blocked, marker_cells)

    placements: list[dict] = []
    used_signatures: set[tuple[tuple[int, int], ...]] = set()
    orientations = [random.choice(("H", "V")) for _ in range(pair_count)]

    for idx in range(pair_count):
        sampled = _sample_shape(used_signatures)
        if sampled is None:
            return None
        sig, bbox_h, bbox_w = sampled
        used_signatures.add(sig)
        gap = random.randint(*GAP_RANGE)
        pair_pos = _place_pair(blocked, h, w, bbox_h, bbox_w, orientations[idx], gap, sig)
        if pair_pos is None:
            return None

        first, second = pair_pos
        color_a = shape_colors[2 * idx]
        color_b = shape_colors[2 * idx + 1]
        _stamp(grid, sig, first[0], first[1], color_a)
        _stamp(grid, sig, second[0], second[1], color_b)
        placements.append(
            {
                "sig": sig,
                "orientation": orientations[idx],
                "first": first,
                "second": second,
            }
        )

    expected = _render_expected(grid, bg, marker_cells, fusion_color, placements)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
