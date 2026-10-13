from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ALL_COLORS = tuple(range(10))
PATTERN_KINDS = ("h", "v", "hv", "none")
LEGEND_ORDER = ("h", "v", "hv")

GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (23, 30)
PATTERN_COUNT_RANGE = (4, 6)
PATTERN_SIZE_RANGE = (3, 6)
LEGEND_BLOCK_WIDTH_RANGE = (3, 5)
LEGEND_GAP_RANGE = (1, 3)
PLACEMENT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["cells"]:
        mask[r - r0, c - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, coords[0])
    seen[start_r, start_c] = True
    q = deque([(start_r, start_c)])
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _bbox_is_full(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return (
        int(coords[:, 0].min()) == 0
        and int(coords[:, 0].max()) == mask.shape[0] - 1
        and int(coords[:, 1].min()) == 0
        and int(coords[:, 1].max()) == mask.shape[1] - 1
    )


def _is_solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _classify_symbol(mask: np.ndarray) -> str | None:
    area = int(mask.sum())
    if area == 3 and mask.shape == (1, 3):
        return "h"
    if area == 3 and mask.shape == (3, 1):
        return "v"
    cross = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    if area == 5 and mask.shape == (3, 3) and np.array_equal(mask, cross):
        return "hv"
    return None


def _find_legend_mapping(grid: np.ndarray, comps: list[dict]) -> tuple[dict[str, int], set[int]]:
    h, _w = grid.shape
    symbol_ids: list[tuple[int, str]] = []
    for idx, comp in enumerate(comps):
        r0, _c0, r1, _c1 = comp["bbox"]
        if r0 < h - 3 or r1 < h - 2:
            continue
        kind = _classify_symbol(_component_mask(comp))
        if kind is not None:
            symbol_ids.append((idx, kind))

    assert len(symbol_ids) == 3, "expected exactly three legend symbols"
    mapping: dict[str, int] = {}
    used_ids: set[int] = set()
    for sym_idx, kind in symbol_ids:
        sr0, sc0, _sr1, sc1 = comps[sym_idx]["bbox"]
        candidates: list[tuple[int, int, int, int]] = []
        for block_idx, comp in enumerate(comps):
            if block_idx == sym_idx or not _is_solid_rect(comp):
                continue
            br0, bc0, br1, bc1 = comp["bbox"]
            if br1 >= sr0:
                continue
            gap = sr0 - br1
            if gap > 2:
                continue
            overlap = min(sc1, bc1) - max(sc0, bc0) + 1
            if overlap <= 0:
                continue
            center_delta = abs((bc0 + bc1) - (sc0 + sc1))
            candidates.append((gap, -overlap, center_delta, block_idx))

        assert candidates, "each legend symbol must have a block directly above it"
        candidates.sort()
        best = candidates[0]
        if len(candidates) > 1:
            assert best[:3] != candidates[1][:3], "legend block pairing must be unambiguous"
        block_idx = best[3]
        mapping[kind] = int(comps[block_idx]["color"])
        used_ids.add(sym_idx)
        used_ids.add(block_idx)

    assert set(mapping) == set(LEGEND_ORDER), "legend must define the three symmetric classes"
    return mapping, used_ids


def _draw_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 - 1, c0 - 1 : c1 + 2] = color
    out[r1 + 1, c0 - 1 : c1 + 2] = color
    out[r0 - 1 : r1 + 2, c0 - 1] = color
    out[r0 - 1 : r1 + 2, c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each pattern's reflection symmetry, look up the legend color for that
    symmetry class, and draw a one-cell border around the pattern's bounding box.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    legend_colors, legend_ids = _find_legend_mapping(g, comps)

    out = g.copy()
    for idx, comp in enumerate(comps):
        if idx in legend_ids:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        assert r0 > 0 and c0 > 0 and r1 + 1 < g.shape[0] and c1 + 1 < g.shape[1]
        mask = _component_mask(comp)
        has_h = np.array_equal(mask, np.flipud(mask))
        has_v = np.array_equal(mask, np.fliplr(mask))
        if has_h and has_v:
            border = legend_colors["hv"]
        elif has_h:
            border = legend_colors["h"]
        elif has_v:
            border = legend_colors["v"]
        else:
            continue
        _draw_frame(out, comp["bbox"], border)
    return out


def _center_rows(h: int) -> list[int]:
    return [h // 2] if h % 2 else [h // 2 - 1, h // 2]


def _center_cols(w: int) -> list[int]:
    return [w // 2] if w % 2 else [w // 2 - 1, w // 2]


def _make_h_only(h: int, w: int) -> np.ndarray:
    rows = _center_rows(h)
    top_depth = min(rows)
    profile = [random.randint(0, top_depth) if random.random() < 0.7 else 0 for _ in range(w)]
    if max(profile) == 0:
        profile[random.randrange(w)] = top_depth
    else:
        profile[random.choice([i for i, v in enumerate(profile) if v > 0])] = top_depth
    if profile == profile[::-1]:
        col = random.randrange(max(1, w // 2))
        mirror = w - 1 - col
        options = [v for v in range(top_depth + 1) if v != profile[mirror]]
        profile[col] = random.choice(options or [top_depth])
        if max(profile) == 0:
            profile[col] = top_depth

    mask = np.zeros((h, w), dtype=bool)
    mask[rows, :] = True
    for c, depth in enumerate(profile):
        if depth <= 0:
            continue
        mask[min(rows) - depth : min(rows), c] = True
        mask[max(rows) + 1 : max(rows) + 1 + depth, c] = True
    assert np.array_equal(mask, np.flipud(mask))
    assert not np.array_equal(mask, np.fliplr(mask))
    return mask


def _make_v_only(h: int, w: int) -> np.ndarray:
    cols = _center_cols(w)
    left_depth = min(cols)
    profile = [random.randint(0, left_depth) if random.random() < 0.7 else 0 for _ in range(h)]
    if max(profile) == 0:
        profile[random.randrange(h)] = left_depth
    else:
        profile[random.choice([i for i, v in enumerate(profile) if v > 0])] = left_depth
    if profile == profile[::-1]:
        row = random.randrange(max(1, h // 2))
        mirror = h - 1 - row
        options = [v for v in range(left_depth + 1) if v != profile[mirror]]
        profile[row] = random.choice(options or [left_depth])
        if max(profile) == 0:
            profile[row] = left_depth

    mask = np.zeros((h, w), dtype=bool)
    mask[:, cols] = True
    for r, depth in enumerate(profile):
        if depth <= 0:
            continue
        mask[r, min(cols) - depth : min(cols)] = True
        mask[r, max(cols) + 1 : max(cols) + 1 + depth] = True
    assert np.array_equal(mask, np.fliplr(mask))
    assert not np.array_equal(mask, np.flipud(mask))
    return mask


def _make_hv(h: int, w: int) -> np.ndarray:
    rows = _center_rows(h)
    cols = _center_cols(w)
    top_depth = min(rows)
    left_depth = min(cols)

    mask = np.zeros((h, w), dtype=bool)
    mask[rows, :] = True
    mask[:, cols] = True

    for c in range((w + 1) // 2):
        depth = random.randint(0, top_depth)
        if depth <= 0:
            continue
        for cc in {c, w - 1 - c}:
            mask[min(rows) - depth : min(rows), cc] = True
            mask[max(rows) + 1 : max(rows) + 1 + depth, cc] = True

    for r in range((h + 1) // 2):
        depth = random.randint(0, left_depth)
        if depth <= 0:
            continue
        for rr in {r, h - 1 - r}:
            mask[rr, min(cols) - depth : min(cols)] = True
            mask[rr, max(cols) + 1 : max(cols) + 1 + depth] = True

    assert np.array_equal(mask, np.flipud(mask))
    assert np.array_equal(mask, np.fliplr(mask))
    return mask


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    out = [(r, c)]
    while (r, c) != b:
        choices: list[tuple[int, int]] = []
        if r < b[0]:
            choices.append((r + 1, c))
        if r > b[0]:
            choices.append((r - 1, c))
        if c < b[1]:
            choices.append((r, c + 1))
        if c > b[1]:
            choices.append((r, c - 1))
        r, c = random.choice(choices)
        out.append((r, c))
    return out


def _grow_connected(mask: np.ndarray, target_area: int) -> None:
    h, w = mask.shape
    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True


def _make_none(h: int, w: int) -> np.ndarray:
    for _ in range(200):
        hub = (random.randint(1, h - 2), random.randint(1, w - 2))
        anchors = [
            (0, random.randint(0, w - 1)),
            (h - 1, random.randint(0, w - 1)),
            (random.randint(0, h - 1), 0),
            (random.randint(0, h - 1), w - 1),
        ]
        mask = np.zeros((h, w), dtype=bool)
        for anchor in anchors:
            for r, c in _manhattan_path(anchor, hub):
                mask[r, c] = True
        max_area = min(h * w - 1, int(mask.sum()) + max(2, h * w // 4))
        target_area = random.randint(int(mask.sum()), max_area)
        _grow_connected(mask, target_area)
        if not _bbox_is_full(mask):
            continue
        if not _is_connected(mask):
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric pattern")


def _make_pattern(kind: str) -> np.ndarray:
    for _ in range(200):
        h = random.randint(*PATTERN_SIZE_RANGE)
        w = random.randint(*PATTERN_SIZE_RANGE)
        if kind == "h":
            mask = _make_h_only(h, w)
        elif kind == "v":
            mask = _make_v_only(h, w)
        elif kind == "hv":
            mask = _make_hv(h, w)
        elif kind == "none":
            mask = _make_none(h, w)
        else:
            raise ValueError(kind)
        if _bbox_is_full(mask) and _is_connected(mask):
            return mask
    raise RuntimeError(f"failed to build pattern {kind}")


def _symbol_mask(kind: str) -> np.ndarray:
    if kind == "h":
        return np.array([[True, True, True]], dtype=bool)
    if kind == "v":
        return np.array([[True], [True], [True]], dtype=bool)
    if kind == "hv":
        return np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ],
            dtype=bool,
        )
    raise ValueError(kind)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _render_pattern(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = color


def _render_legend(grid: np.ndarray, legend_colors: dict[str, int], mark_color: int) -> None:
    h, w = grid.shape
    block_widths = [random.randint(*LEGEND_BLOCK_WIDTH_RANGE) for _ in LEGEND_ORDER]
    gaps = [random.randint(*LEGEND_GAP_RANGE) for _ in range(len(LEGEND_ORDER) - 1)]
    total_width = sum(block_widths) + sum(gaps)
    assert total_width + 2 <= w, "legend must fit with one-column side margins"
    start = random.randint(1, w - total_width - 1)
    row = h - 4
    col = start
    for idx, kind in enumerate(LEGEND_ORDER):
        block_w = block_widths[idx]
        grid[row, col : col + block_w] = legend_colors[kind]
        sym = _symbol_mask(kind)
        sym_h, sym_w = sym.shape
        sym_top = row + 1 + (3 - sym_h) // 2
        sym_left = col + (block_w - sym_w) // 2
        sym_view = grid[sym_top : sym_top + sym_h, sym_left : sym_left + sym_w]
        sym_view[sym] = mark_color
        col += block_w
        if idx + 1 < len(LEGEND_ORDER):
            col += gaps[idx]


def _expected_output(
    inp: np.ndarray,
    placed: list[dict],
    legend_colors: dict[str, int],
) -> np.ndarray:
    out = inp.copy()
    for item in placed:
        if item["kind"] == "none":
            continue
        border = legend_colors[item["kind"]]
        _draw_frame(out, item["bbox"], border)
    return out


def _sample_pattern_kinds() -> list[str]:
    total = random.randint(*PATTERN_COUNT_RANGE)
    kinds = ["h", "v", "hv", "none"]
    while len(kinds) < total:
        kinds.append(random.choice(PATTERN_KINDS))
    random.shuffle(kinds)
    return kinds


def _sample_grid_shape() -> tuple[int, int]:
    return (
        random.randint(*GRID_HEIGHT_RANGE),
        random.randint(*GRID_WIDTH_RANGE),
    )


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_grid_shape()
    kinds = _sample_pattern_kinds()

    available_colors = [c for c in ALL_COLORS if c != BG]
    mark_color = random.choice(available_colors)
    available_colors.remove(mark_color)
    legend_palette = random.sample(available_colors, len(LEGEND_ORDER))
    legend_colors = {kind: color for kind, color in zip(LEGEND_ORDER, legend_palette, strict=False)}
    pattern_colors = [c for c in available_colors if c not in legend_palette]
    if not pattern_colors:
        return None

    grid = np.full((h, w), BG, dtype=int)
    placed: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []
    bottom_limit = h - 6

    for kind in kinds:
        mask = _make_pattern(kind)
        ph, pw = mask.shape
        if ph + 1 > bottom_limit or pw + 1 > w - 2:
            return None

        placed_here = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, bottom_limit - ph)
            left = random.randint(1, w - pw - 1)
            bbox = (top, left, top + ph - 1, left + pw - 1)
            frame_box = (top - 1, left - 1, top + ph, left + pw)
            if any(_boxes_overlap(frame_box, other) for other in reserved):
                continue
            color = random.choice(pattern_colors)
            _render_pattern(grid, mask, top, left, color)
            placed.append({"kind": kind, "bbox": bbox, "color": color, "mask": mask})
            reserved.append(frame_box)
            placed_here = True
            break
        if not placed_here:
            return None

    _render_legend(grid, legend_colors, mark_color)
    expected = _expected_output(grid, placed, legend_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
