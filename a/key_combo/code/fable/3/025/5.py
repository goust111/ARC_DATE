from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
SIDES = ("top", "right", "bottom", "left")
ROT_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
GEN_ATTEMPTS = 800


@dataclass(frozen=True)
class Source:
    body_color: int
    nose_color: int
    r0: int
    c0: int
    patch: np.ndarray


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _divider_col(g: np.ndarray, bg: int) -> int:
    h, w = g.shape
    cols = []
    for c in range(w):
        col = g[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            cols.append(c)
    assert len(cols) == 1, "expected exactly one full-height non-background divider column"
    div = int(cols[0])
    assert 0 < div < w - 1, "divider must split non-empty left and right regions"
    return div


def _parse_sources(g: np.ndarray, div_col: int, bg: int) -> tuple[dict[int, Source], int, int]:
    left = g[:, :div_col]
    comps = _components(left != bg)
    assert 2 <= len(comps) <= 3, "expected two or three left-side source sprites"

    sources: dict[int, Source] = {}
    side: int | None = None
    nose_colors: set[int] = set()

    for coords in sorted(comps, key=lambda pts: (_bbox(pts)[0], _bbox(pts)[1])):
        r0, c0, r1, c1 = _bbox(coords)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        assert h == w and h >= 3 and h % 2 == 1, "source sprite bbox must be an odd square"
        if side is None:
            side = h
        assert h == side, "all source sprites must have the same square size"

        patch = left[r0 : r1 + 1, c0 : c1 + 1].copy()
        non_bg = patch[patch != bg]
        colors, counts = np.unique(non_bg, return_counts=True)
        assert len(colors) == 2, "each source uses one body color and one shared nose color"

        singleton = [int(color) for color, count in zip(colors, counts) if int(count) == 1]
        assert len(singleton) == 1, "the nose color must occur exactly once in each source"
        nose_color = singleton[0]
        body_color = int(next(color for color in colors if int(color) != nose_color))
        assert body_color not in sources, "source body colors must be unique"

        nose_cells = np.argwhere(patch == nose_color)
        assert nose_cells.shape == (1, 2), "source must have exactly one nose pixel"
        nr, nc = map(int, nose_cells[0])
        assert (nr, nc) == (0, h // 2), "source nose must start at the top edge midpoint"
        assert np.count_nonzero(patch == body_color) >= h, "source body must be visible"
        assert np.all((patch == bg) | (patch == body_color) | (patch == nose_color))

        nose_colors.add(nose_color)
        sources[body_color] = Source(body_color, nose_color, int(r0), int(c0), patch)

    assert side is not None
    assert len(nose_colors) == 1, "all source sprites must share the same nose color"
    return sources, int(side), int(next(iter(nose_colors)))


def _marker_side(local_r: int, local_c: int, side: int) -> str:
    mid = side // 2
    positions = {
        (0, mid): "top",
        (mid, side - 1): "right",
        (side - 1, mid): "bottom",
        (mid, 0): "left",
    }
    key = (int(local_r), int(local_c))
    assert key in positions, "marker must sit at the midpoint of one slot edge"
    return positions[key]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate and stamp the left source sprite selected by each right-slot side marker.

    Input structure:
    - A dominant background is split by one full-height monochrome divider column.
    - The left side contains two or three separated square source sprites; each has a unique body color and the same nose-color pixel at the top-edge midpoint.
    - The right side is tiled into same-size square slots; a marked slot has exactly one colored pixel at a slot-edge midpoint.

    Rule:
    - For every marked slot, use the marker color to choose the left source whose body color matches it.
    - Rotate that source so its nose points toward the marker's touched edge, then replace the whole slot with the rotated sprite.
    - Leave unmatched-color markers, empty slots, the divider, and the left source sprites unchanged.

    Key insight:
    - A single marker pixel carries two independent instructions: its color chooses the sprite, and its side position chooses the rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0 and max(g.shape) <= 30, "expected a non-empty ARC grid"
    bg = _background(g)
    div_col = _divider_col(g, bg)
    sources, side, _nose = _parse_sources(g, div_col, bg)

    h, w = g.shape
    right_w = w - div_col - 1
    assert h % side == 0 and right_w % side == 0, "right region must be tiled by source-size square slots"

    out = g.copy()
    for r0 in range(0, h, side):
        for c0 in range(div_col + 1, w, side):
            slot = g[r0 : r0 + side, c0 : c0 + side]
            marks = np.argwhere(slot != bg)
            if marks.size == 0:
                continue
            assert marks.shape == (1, 2), "each marked slot must contain one marker pixel only"
            local_r, local_c = map(int, marks[0])
            marker_color = int(slot[local_r, local_c])
            if marker_color not in sources:
                continue
            side_name = _marker_side(local_r, local_c, side)
            stamp = np.rot90(sources[marker_color].patch, ROT_K[side_name])
            out[r0 : r0 + side, c0 : c0 + side] = stamp
    return out


# ----------------------------- generator -----------------------------


def _path_cells(start: tuple[int, int], target: tuple[int, int], rng: random.Random) -> set[tuple[int, int]]:
    r, c = start
    tr, tc = target
    cells = {(r, c)}
    while (r, c) != (tr, tc):
        steps: list[tuple[int, int]] = []
        if r < tr:
            steps.append((1, 0))
        elif r > tr:
            steps.append((-1, 0))
        if c < tc:
            steps.append((0, 1))
        elif c > tc:
            steps.append((0, -1))
        dr, dc = rng.choice(steps)
        r += dr
        c += dc
        cells.add((r, c))
    return cells


def _body_mask(side: int, used: set[tuple[tuple[int, int], ...]], rng: random.Random) -> np.ndarray | None:
    mid = side // 2
    nose = (0, mid)
    seed = (1, mid)
    for _ in range(300):
        left_target = (rng.randint(1, side - 1), 0)
        right_target = (rng.randint(1, side - 1), side - 1)
        bottom_target = (side - 1, rng.randint(0, side - 1))
        cells: set[tuple[int, int]] = {seed}
        for target in (left_target, right_target, bottom_target):
            cells.update(_path_cells(seed, target, rng))
        cells.discard(nose)

        low = max(len(cells), max(side + 1, int(side * side * 0.40)))
        high = max(low, int(side * side * 0.72))
        target_area = rng.randint(low, min(high, side * side - 1))
        while len(cells) < target_area:
            candidates = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) != nose and (nr, nc) not in cells:
                        candidates.append((nr, nc))
            if not candidates:
                break
            cells.add(rng.choice(candidates))

        if len(cells) < low:
            continue
        rows = {r for r, _c in cells} | {0}
        cols = {c for _r, c in cells}
        if min(rows) != 0 or max(rows) != side - 1 or min(cols) != 0 or max(cols) != side - 1:
            continue
        key = tuple(sorted(cells))
        if key in used:
            continue
        used.add(key)
        mask = np.zeros((side, side), dtype=bool)
        rr, cc = zip(*cells)
        mask[rr, cc] = True
        return mask
    return None


def _make_source_patch(side: int, body_color: int, nose_color: int, mask: np.ndarray) -> np.ndarray:
    patch = np.full((side, side), BG, dtype=int)
    patch[mask] = int(body_color)
    patch[0, side // 2] = int(nose_color)
    return patch


def _slot_local(side_name: str, side: int) -> tuple[int, int]:
    mid = side // 2
    return {
        "top": (0, mid),
        "right": (mid, side - 1),
        "bottom": (side - 1, mid),
        "left": (mid, 0),
    }[side_name]


def _choose_layout(source_count: int, side: int, min_slots: int, rng: random.Random):
    min_h = source_count * side + source_count + 1
    min_rows = (min_h + side - 1) // side
    max_rows = 30 // side
    if min_rows > max_rows:
        return None

    for _ in range(80):
        slot_rows = rng.randint(min_rows, max_rows)
        left_w = side + rng.randint(2, 4)
        min_cols = max(2, (min_slots + slot_rows - 1) // slot_rows)
        max_cols = min(5, (30 - left_w - 1) // side)
        if min_cols <= max_cols:
            slot_cols = rng.randint(min_cols, max_cols)
            return slot_rows, slot_cols, left_w
    return None


def _place_sources(
    grid: np.ndarray,
    patches: list[np.ndarray],
    left_w: int,
    side: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h = grid.shape[0]
    source_count = len(patches)
    for _ in range(200):
        gaps = [rng.randint(1, 3) for _ in range(source_count - 1)]
        total_h = source_count * side + sum(gaps)
        if total_h > h - 2:
            continue
        r = rng.randint(1, h - total_h - 1)
        c = rng.randint(0, left_w - side - 1)
        locs = []
        for i, patch in enumerate(patches):
            grid[r : r + side, c : c + side] = patch
            locs.append((r, c))
            r += side + (gaps[i] if i < len(gaps) else 0)
        return locs
    raise ValueError("could not place left sources")


def _marker_plan(body_colors: list[int], nose_color: int, divider_color: int, slot_count: int, rng: random.Random):
    matched = []
    colors_for_sides = body_colors[:]
    while len(colors_for_sides) < len(SIDES):
        colors_for_sides.append(rng.choice(body_colors))
    rng.shuffle(colors_for_sides)
    side_order = list(SIDES)
    rng.shuffle(side_order)
    for side_name, color in zip(side_order, colors_for_sides):
        matched.append({"side": side_name, "color": int(color), "matched": True})

    remaining = slot_count - len(matched)
    if remaining < 2:
        return None
    extra_match_count = rng.randint(0, min(2, remaining - 2))
    for _ in range(extra_match_count):
        matched.append({"side": rng.choice(SIDES), "color": int(rng.choice(body_colors)), "matched": True})

    remaining = slot_count - len(matched)
    distractor_count = rng.randint(1, min(2, remaining - 1))
    unmatched_pool = [c for c in PALETTE if c not in body_colors and c != divider_color]
    distractors = []
    for i in range(distractor_count):
        if i == 0 and nose_color in unmatched_pool and rng.random() < 0.7:
            color = nose_color
        else:
            color = rng.choice(unmatched_pool)
        distractors.append({"side": rng.choice(SIDES), "color": int(color), "matched": False})

    records = matched + distractors
    rng.shuffle(records)
    return records


def _render_markers_and_expected(
    grid: np.ndarray,
    source_by_color: dict[int, np.ndarray],
    records: list[dict],
    side: int,
    div_col: int,
    slot_rows: int,
    slot_cols: int,
    rng: random.Random,
) -> np.ndarray:
    expected = grid.copy()
    slots = [(r, c) for r in range(slot_rows) for c in range(slot_cols)]
    rng.shuffle(slots)
    assert len(records) < len(slots), "generator must leave at least one empty slot"

    for record, (sr, sc) in zip(records, slots):
        r0 = sr * side
        c0 = div_col + 1 + sc * side
        lr, lc = _slot_local(record["side"], side)
        grid[r0 + lr, c0 + lc] = int(record["color"])
        expected[r0 + lr, c0 + lc] = int(record["color"])
        record["slot"] = (sr, sc)
        if bool(record["matched"]):
            stamp = np.rot90(source_by_color[int(record["color"])], ROT_K[str(record["side"])])
            expected[r0 : r0 + side, c0 : c0 + side] = stamp
    return expected


def _semantic_validate(
    grid: np.ndarray,
    expected: np.ndarray,
    patches: dict[int, np.ndarray],
    records: list[dict],
    side: int,
    div_col: int,
    slot_rows: int,
    slot_cols: int,
    nose_color: int,
) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(grid, expected):
        return False

    matched = [r for r in records if bool(r["matched"])]
    distractors = [r for r in records if not bool(r["matched"])]
    if {str(r["side"]) for r in matched} != set(SIDES):
        return False
    if {int(r["color"]) for r in matched} != set(patches):
        return False
    if not distractors:
        return False

    body_masks = {color: tuple(map(tuple, np.argwhere(patch == color))) for color, patch in patches.items()}
    if len(set(body_masks.values())) != len(body_masks):
        return False

    used_slots = {tuple(r["slot"]) for r in records}
    if len(used_slots) != len(records):
        return False
    if len(used_slots) >= slot_rows * slot_cols:
        return False

    for record in records:
        sr, sc = map(int, record["slot"])
        r0 = sr * side
        c0 = div_col + 1 + sc * side
        slot = grid[r0 : r0 + side, c0 : c0 + side]
        marks = np.argwhere(slot != BG)
        if marks.shape != (1, 2):
            return False
        lr, lc = map(int, marks[0])
        if _slot_local(str(record["side"]), side) != (lr, lc):
            return False
        if int(slot[lr, lc]) != int(record["color"]):
            return False

        out_slot = expected[r0 : r0 + side, c0 : c0 + side]
        if bool(record["matched"]):
            wanted = np.rot90(patches[int(record["color"])], ROT_K[str(record["side"])])
            if not np.array_equal(out_slot, wanted):
                return False
            if int(out_slot[lr, lc]) != int(nose_color):
                return False
        else:
            if int(record["color"]) in patches:
                return False
            if not np.array_equal(out_slot, slot):
                return False

    for sr in range(slot_rows):
        for sc in range(slot_cols):
            if (sr, sc) in used_slots:
                continue
            r0 = sr * side
            c0 = div_col + 1 + sc * side
            if np.any(grid[r0 : r0 + side, c0 : c0 + side] != BG):
                return False
            if np.any(expected[r0 : r0 + side, c0 : c0 + side] != BG):
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice((3, 5, 7))
    source_count = rng.choice((2, 3))
    min_slots = len(SIDES) + 2
    layout = _choose_layout(source_count, side, min_slots, rng)
    if layout is None:
        return None
    slot_rows, slot_cols, left_w = layout
    h = slot_rows * side
    w = left_w + 1 + slot_cols * side

    colors = list(PALETTE)
    rng.shuffle(colors)
    nose_color = colors.pop()
    divider_color = colors.pop()
    body_colors = [colors.pop() for _ in range(source_count)]

    used_masks: set[tuple[tuple[int, int], ...]] = set()
    patches: dict[int, np.ndarray] = {}
    patch_list = []
    for color in body_colors:
        mask = _body_mask(side, used_masks, rng)
        if mask is None:
            return None
        patch = _make_source_patch(side, int(color), int(nose_color), mask)
        patches[int(color)] = patch
        patch_list.append(patch)

    grid = np.full((h, w), BG, dtype=int)
    div_col = left_w
    grid[:, div_col] = int(divider_color)
    try:
        _place_sources(grid, patch_list, left_w, side, rng)
    except ValueError:
        return None

    records = _marker_plan(body_colors, int(nose_color), int(divider_color), slot_rows * slot_cols, rng)
    if records is None:
        return None
    expected = _render_markers_and_expected(grid, patches, records, side, div_col, slot_rows, slot_cols, rng)

    if not _semantic_validate(grid, expected, patches, records, side, div_col, slot_rows, slot_cols, int(nose_color)):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orientation-stamp input")
