from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
MAROON = 9
STAMP_COLORS = (1, 2, 3, 4, 6, 7)
BOARD_COLOR = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0, r1, c1)


def _detect_board(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]:
    counts = Counter(int(v) for v in grid.flat if int(v) not in {BG, GREY, MAROON})
    assert counts, "expected a colored board"
    board_color = max(counts, key=counts.get)
    board_coords = [tuple(map(int, p)) for p in np.argwhere(grid == int(board_color))]
    r0, c0, r1, c1 = _bbox(board_coords)
    slots = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == MAROON)
        if r0 <= int(r) <= r1 and c0 <= int(c) <= c1
    ]
    assert len(slots) >= 3, "expected board slots"
    slot_rows = {r for r, _ in slots}
    assert len(slot_rows) == 1, "board slots must lie on one row"
    slots.sort(key=lambda p: p[1])
    return int(board_color), (int(r0), int(c0), int(r1), int(c1)), slots


def _extract_bars(grid: np.ndarray) -> list[dict]:
    bars = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        assert r0 == r1, "count bars must be horizontal"
        bars.append(
            {
                "row": int(r0),
                "c0": int(c0),
                "c1": int(c1),
                "length": int(len(coords)),
            }
        )
    return bars


def _extract_stamps(grid: np.ndarray, board_bbox: tuple[int, int, int, int], bars: list[dict]) -> list[dict]:
    stamp_mask = (grid != BG) & (grid != GREY)
    br0, bc0, br1, bc1 = board_bbox
    stamp_mask[br0 : br1 + 1, bc0 : bc1 + 1] = False

    stamps = []
    for coords in _components(stamp_mask):
        values = {int(grid[r, c]) for r, c in coords}
        body_colors = values - {MAROON}
        assert len(body_colors) == 1, "each stamp must have one body color"
        anchor_cells = [(r, c) for r, c in coords if int(grid[r, c]) == MAROON]
        assert len(anchor_cells) == 1, "each stamp must have one maroon anchor"
        patch, bbox = _crop_patch(grid, coords)
        r0, c0, r1, c1 = bbox
        anchor_r, anchor_c = anchor_cells[0]
        rel_anchor = (int(anchor_r - r0), int(anchor_c - c0))

        below = [bar for bar in bars if int(bar["row"]) == int(r1 + 1)]
        assert below, "each stamp must have a count bar below it"
        bar = min(
            below,
            key=lambda item: max(0, max(c0 - int(item["c1"]), int(item["c0"]) - c1)),
        )
        stamps.append(
            {
                "patch": patch,
                "anchor": rel_anchor,
                "count": int(bar["length"]),
            }
        )

    counts = [int(stamp["count"]) for stamp in stamps]
    assert len(stamps) in {3, 4}
    assert len(set(counts)) == len(counts), "count bars must be unique"
    return stamps


def _render_output(
    shape: tuple[int, int],
    board_color: int,
    board_bbox: tuple[int, int, int, int],
    slots: list[tuple[int, int]],
    stamps: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = board_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = int(board_color)
    for r, c in slots:
        out[int(r), int(c)] = MAROON

    ordered_slots = list(slots)
    for stamp in sorted(stamps, key=lambda item: int(item["count"])):
        idx = int(stamp["count"]) - 1
        assert 0 <= idx < len(ordered_slots), "bar length must address an existing slot"
        target_r, target_c = ordered_slots[idx]
        patch = np.asarray(stamp["patch"], dtype=int)
        anchor_r, anchor_c = stamp["anchor"]
        top = int(target_r - anchor_r)
        left = int(target_c - anchor_c)
        for pr, pc in np.argwhere(patch != BG):
            rr = int(top + pr)
            cc = int(left + pc)
            assert r0 <= rr <= r1 and c0 <= cc <= c1, "stamped patch must stay on the board"
            out[rr, cc] = int(patch[pr, pc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    board_color, board_bbox, slots = _detect_board(g)
    bars = _extract_bars(g)
    stamps = _extract_stamps(g, board_bbox, bars)
    return _render_output(g.shape, board_color, board_bbox, slots, stamps)


def _random_stamp_patch(rng: np.random.Generator, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    canvas = np.full((5, 5), BG, dtype=int)
    anchor = (2, 2)
    occupied = {anchor, (1, 2), (2, 1), (2, 3), (3, 2)}
    target_area = int(rng.integers(5, 10))

    while len(occupied) < target_area:
        frontier = set()
        for r, c in occupied:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in occupied:
                    frontier.add((nr, nc))
        frontier.discard(anchor)
        choices = list(frontier)
        occupied.add(choices[int(rng.integers(len(choices)))])

    for r, c in occupied:
        canvas[r, c] = color
    canvas[anchor] = MAROON

    k = int(rng.integers(4))
    if k:
        canvas = np.rot90(canvas, k)
    if bool(rng.integers(2)):
        canvas = np.fliplr(canvas)

    coords = [tuple(map(int, p)) for p in np.argwhere(canvas != BG)]
    patch, _bbox0 = _crop_patch(canvas, coords)
    anchor_pos = tuple(map(int, np.argwhere(patch == MAROON)[0]))
    return patch, anchor_pos


def _build_input(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    stamp_count = int(rng.integers(3, 5))
    counts = sorted(rng.choice(np.arange(1, 5), size=stamp_count, replace=False).tolist())
    slot_count = max(counts) + int(rng.integers(0, 2))
    slot_count = min(slot_count, 5)

    colors = rng.choice(np.array(STAMP_COLORS), size=stamp_count, replace=False).tolist()
    stamp_specs = []
    for color, count in zip(colors, counts):
        patch, anchor = _random_stamp_patch(rng, int(color))
        stamp_specs.append({"color": int(color), "count": int(count), "patch": patch, "anchor": anchor})

    max_up = max(int(spec["anchor"][0]) for spec in stamp_specs)
    max_down = max(int(spec["patch"].shape[0] - 1 - spec["anchor"][0]) for spec in stamp_specs)
    max_left = max(int(spec["anchor"][1]) for spec in stamp_specs)
    max_right = max(int(spec["patch"].shape[1] - 1 - spec["anchor"][1]) for spec in stamp_specs)

    slot_spacing = 2
    board_h_min = max_up + max_down + 3
    board_h = int(board_h_min + rng.integers(0, 4))
    board_w_min = max_left + max_right + (slot_count - 1) * slot_spacing + 3
    board_w = int(board_w_min + rng.integers(0, 4))

    place_order = list(range(stamp_count))
    rng.shuffle(place_order)
    top_margin = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(1, 3))
    gaps = [int(rng.integers(1, 3)) for _ in range(max(0, stamp_count - 1))]

    placements = []
    y = top_margin
    left_band_w = max(8, max(int(spec["patch"].shape[1]) for spec in stamp_specs) + 3)
    for idx_in_order, spec_idx in enumerate(place_order):
        spec = stamp_specs[spec_idx]
        patch = spec["patch"]
        ph, pw = patch.shape
        max_x = max(1, left_band_w - pw - 1)
        x = int(rng.integers(1, max_x + 1))
        bar_row = int(y + ph)
        bar_left = int(x + spec["anchor"][1] - (spec["count"] - 1) // 2)
        bar_left = max(1, min(bar_left, left_band_w - spec["count"]))
        placements.append({"spec_idx": spec_idx, "top": int(y), "left": int(x), "bar_row": bar_row, "bar_left": bar_left})
        y = int(bar_row + 1 + (gaps[idx_in_order] if idx_in_order < len(gaps) else 0))
    stamp_zone_h = int(y + bottom_margin)

    height = max(stamp_zone_h, board_h + 2)
    board_top = int(rng.integers(1, height - board_h + 1))
    gap_w = int(rng.integers(3, 6))
    board_left = int(left_band_w + gap_w)
    width = int(board_left + board_w + rng.integers(1, 3))

    slot_row = int(rng.integers(max_up + 1, board_h - max_down - 1))
    first_slot_min = max_left + 1
    first_slot_max = board_w - max_right - 2 - (slot_count - 1) * slot_spacing
    first_slot = int(rng.integers(first_slot_min, first_slot_max + 1))
    slots = [(int(board_top + slot_row), int(board_left + first_slot + i * slot_spacing)) for i in range(slot_count)]
    board_bbox = (int(board_top), int(board_left), int(board_top + board_h - 1), int(board_left + board_w - 1))

    grid = np.full((height, width), BG, dtype=int)
    br0, bc0, br1, bc1 = board_bbox
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = BOARD_COLOR
    for r, c in slots:
        grid[r, c] = MAROON

    for placement in placements:
        spec = stamp_specs[placement["spec_idx"]]
        patch = spec["patch"]
        ph, pw = patch.shape
        top = placement["top"]
        left = placement["left"]
        region = grid[top : top + ph, left : left + pw]
        assert not np.any((patch != BG) & (region != BG))
        mask = patch != BG
        region[mask] = patch[mask]
        row = placement["bar_row"]
        start = placement["bar_left"]
        assert not np.any(grid[row, start : start + spec["count"]] != BG)
        grid[row, start : start + spec["count"]] = GREY

    expected = _render_output(
        grid.shape,
        BOARD_COLOR,
        board_bbox,
        slots,
        [{"patch": spec["patch"], "anchor": spec["anchor"], "count": spec["count"]} for spec in stamp_specs],
    )
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected = _build_input(rng)
            solved = main(grid)
            if np.array_equal(solved, expected):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid sample")
