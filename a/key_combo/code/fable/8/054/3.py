from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SEPARATOR = 4
SIZE_LEDGER = 5
OBJECT_COLORS = (1, 2, 3, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(r - min_r, c - min_c) for r, c in cells}
    max_r = max(r for r, _ in shifted)
    max_c = max(c for _, c in shifted)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _random_polyomino(area: int, rng: random.Random) -> np.ndarray:
    if area == 1:
        return np.array([[True]], dtype=bool)

    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        cells.add(rng.choice(frontier))

    mask = _normalize_cells(cells)
    for _ in range(rng.randrange(4)):
        mask = np.rot90(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    return np.array(mask, dtype=bool)


def _find_separators(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = []
    for r in range(h):
        vals = {int(v) for v in grid[r, :]}
        if len(vals) == 1:
            color = next(iter(vals))
            if color != BG:
                row_candidates.append((r, color))

    col_candidates = []
    for c in range(w):
        vals = {int(v) for v in grid[:, c]}
        if len(vals) == 1:
            color = next(iter(vals))
            if color != BG:
                col_candidates.append((c, color))

    matches = [(r, c, color) for r, color in row_candidates for c, color2 in col_candidates if color == color2]
    assert len(matches) == 1, "expected one full separator row and one full separator column"
    sep_r, sep_c, sep_color = matches[0]
    assert 0 < sep_r < h - 1 and 0 < sep_c < w - 1, "separator must split the grid into three usable regions"
    return int(sep_r), int(sep_c), int(sep_color)


def _find_objects(field: np.ndarray) -> list[dict]:
    h, w = field.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            color = int(field[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(field[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            cell_set = set(cells)
            for cr, cc in cell_set:
                for nr in range(cr - 1, cr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and int(field[nr, nc]) == color:
                            assert False, "objects of the same color must not touch diagonally"

            objects.append({"color": color, "cells": cells, "area": len(cells)})

    assert objects, "expected at least one object"
    return objects


def _parse_color_ledger(bottom: np.ndarray) -> dict[int, int]:
    h, w = bottom.shape
    counts: dict[int, int] = {}
    for c in range(w):
        rows = np.flatnonzero(bottom[:, c] != BG)
        if rows.size == 0:
            continue
        color_values = {int(v) for v in bottom[rows, c]}
        assert len(color_values) == 1, "each color bar must be monochrome"
        assert int(rows[-1]) == h - 1, "color bars must rise from the bottom edge"
        expected_rows = np.arange(int(rows[0]), h)
        assert np.array_equal(rows, expected_rows), "color bars must be contiguous vertical suffixes"
        color = next(iter(color_values))
        assert color != SIZE_LEDGER, "size-ledger color cannot appear in the color ledger"
        assert color not in counts, "expected exactly one color bar per object color"
        counts[color] = int(rows.size)
    assert counts, "expected at least one color bar"
    return counts


def _parse_size_ledger(right: np.ndarray) -> dict[int, int]:
    h, w = right.shape
    non_bg = {int(v) for v in right.flatten() if int(v) != BG}
    assert non_bg == {SIZE_LEDGER}, "right ledger must contain only grey counting bars"

    lengths = []
    for r in range(h):
        cols = np.flatnonzero(right[r, :] != BG)
        if cols.size == 0:
            lengths.append(0)
            continue
        assert int(cols[-1]) == w - 1, "size bars must grow from the right edge"
        expected_cols = np.arange(int(cols[0]), w)
        assert np.array_equal(cols, expected_cols), "size bars must be contiguous horizontal suffixes"
        lengths.append(int(cols.size))

    last_nonzero = max((i for i, length in enumerate(lengths) if length > 0), default=-1)
    assert last_nonzero >= 0, "expected at least one size bar"
    assert all(length > 0 for length in lengths[: last_nonzero + 1]), "sizes 1..K must all have visible bars"
    assert all(length == 0 for length in lengths[last_nonzero + 1 :]), "unused size rows must be empty"
    return {size: lengths[size - 1] for size in range(1, last_nonzero + 2)}


def _missing_key(actual: Counter, registered: dict[int, int], label: str) -> int:
    all_keys = set(actual) | set(registered)
    diffs = {key: int(actual.get(key, 0)) - int(registered.get(key, 0)) for key in all_keys}
    missing = [key for key, diff in diffs.items() if diff == 1]
    assert len(missing) == 1, f"expected exactly one missing {label}"
    assert all(diff == 0 for key, diff in diffs.items() if key not in missing), f"unexpected {label} ledger mismatch"
    return int(missing[0])


def _infer_intruder(grid: np.ndarray) -> dict:
    sep_r, sep_c, _sep_color = _find_separators(grid)
    field = grid[:sep_r, :sep_c]
    bottom = grid[sep_r + 1 :, :sep_c]
    right = grid[:sep_r, sep_c + 1 :]
    assert field.size > 0 and bottom.size > 0 and right.size > 0

    objects = _find_objects(field)
    actual_colors = Counter(obj["color"] for obj in objects)
    actual_sizes = Counter(obj["area"] for obj in objects)
    registered_colors = _parse_color_ledger(bottom)
    registered_sizes = _parse_size_ledger(right)

    missing_color = _missing_key(actual_colors, registered_colors, "color")
    missing_size = _missing_key(actual_sizes, registered_sizes, "size")
    candidates = [obj for obj in objects if obj["color"] == missing_color and obj["area"] == missing_size]
    assert len(candidates) == 1, "color and size ledgers must cross-check to a unique object"

    return {
        "sep_row": sep_r,
        "sep_col": sep_c,
        "missing_color": missing_color,
        "missing_size": missing_size,
        "candidate": candidates[0],
        "objects": objects,
        "actual_colors": actual_colors,
        "actual_sizes": actual_sizes,
        "registered_colors": registered_colors,
        "registered_sizes": registered_sizes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the one field object that is missing from both the bottom color ledger and the right size ledger.

    Input structure:
    - A full horizontal separator row and a full vertical separator column split the grid into a top-left object field, a bottom color ledger, and a right size ledger.
    - The object field has black background and several pairwise separated monochrome 4-connected objects; colors and sizes may repeat.
    - The bottom ledger contains one colored vertical bar per object color, while the right ledger contains grey horizontal bars whose top-to-bottom order encodes object sizes.

    Rule:
    - Count the field objects by color and compare those counts with the bottom bars to find the single color whose ledger is short by one object.
    - Count the field objects by pixel area and compare those counts with the right bars to find the single size whose ledger is short by one object.
    - Exactly one field object has both that color and that size; erase only that object to black and keep the ledgers unchanged.

    Key insight:
    - Neither ledger identifies the intruder alone; the answer is the unique intersection of the color-count discrepancy and the size-count discrepancy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _infer_intruder(g)
    out = g.copy()
    for r, c in info["candidate"]["cells"]:
        out[r, c] = BG
    return out


def _choose_object_specs(rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n_colors = rng.randint(3, 4)
    max_size = rng.randint(max(4, n_colors), 5)
    colors = rng.sample(list(OBJECT_COLORS), n_colors)
    shift = rng.randint(1, n_colors - 1)

    specs = []
    base_pairs = []
    for size in range(1, max_size + 1):
        primary = colors[(size - 1) % n_colors]
        secondary = colors[(size - 1 + shift) % n_colors]
        base_pairs.append((primary, size))
        base_pairs.append((secondary, size))
        specs.extend([(primary, size), (secondary, size)])

    intruder = rng.choice(base_pairs)

    for _ in range(rng.randint(0, 2)):
        candidates = [(color, size) for color in colors for size in range(1, max_size + 1) if (color, size) != intruder]
        specs.append(rng.choice(candidates))

    rng.shuffle(specs)
    return specs, intruder


def _can_place(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    for dr, dc in np.argwhere(mask):
        if blocked[r0 + int(dr), c0 + int(dc)]:
            return False
    return True


def _place_objects(field_h: int, field_w: int, specs: list[tuple[int, int]], intruder: tuple[int, int], rng: random.Random):
    field = np.zeros((field_h, field_w), dtype=int)
    blocked = np.zeros((field_h, field_w), dtype=bool)
    objects = []

    ordered_specs = sorted(list(enumerate(specs)), key=lambda item: item[1][1], reverse=True)
    for original_index, (color, size) in ordered_specs:
        placed = False
        for _ in range(60):
            mask = _random_polyomino(size, rng)
            h, w = mask.shape
            if h > field_h or w > field_w:
                continue

            positions = []
            for r0 in range(field_h - h + 1):
                for c0 in range(field_w - w + 1):
                    if _can_place(blocked, mask, r0, c0):
                        positions.append((r0, c0))
            if not positions:
                continue

            r0, c0 = rng.choice(positions)
            cells = []
            for dr, dc in np.argwhere(mask):
                rr = r0 + int(dr)
                cc = c0 + int(dc)
                field[rr, cc] = color
                cells.append((rr, cc))
                for nr in range(rr - 1, rr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if 0 <= nr < field_h and 0 <= nc < field_w:
                            blocked[nr, nc] = True
            objects.append({"index": original_index, "color": color, "area": size, "cells": cells})
            placed = True
            break
        if not placed:
            return None

    objects.sort(key=lambda obj: obj["index"])
    intruder_index = next(i for i, spec in enumerate(specs) if spec == intruder)
    intruder_object = objects[intruder_index]
    return field, objects, intruder_object


def _build_ledgers(field_h: int, field_w: int, objects: list[dict], intruder_object: dict, rng: random.Random):
    actual_color_counts = Counter(obj["color"] for obj in objects)
    actual_size_counts = Counter(obj["area"] for obj in objects)
    color_counts = dict(actual_color_counts)
    size_counts = dict(actual_size_counts)
    color_counts[intruder_object["color"]] -= 1
    size_counts[intruder_object["area"]] -= 1

    assert min(color_counts.values()) >= 1
    assert min(size_counts.values()) >= 1

    bottom_h = max(color_counts.values()) + rng.randint(1, 2)
    right_w = max(size_counts.values()) + rng.randint(1, 2)
    bottom = np.zeros((bottom_h, field_w), dtype=int)
    right = np.zeros((field_h, right_w), dtype=int)

    bar_cols = sorted(rng.sample(range(field_w), len(color_counts)))
    color_items = list(color_counts.items())
    rng.shuffle(color_items)
    for col, (color, count) in zip(bar_cols, color_items):
        bottom[bottom_h - count :, col] = color

    max_size = max(size_counts)
    assert field_h >= max_size + 1
    for size in range(1, max_size + 1):
        count = size_counts[size]
        right[size - 1, right_w - count :] = SIZE_LEDGER

    return bottom, right


def _assemble_grid(field: np.ndarray, bottom: np.ndarray, right: np.ndarray) -> np.ndarray:
    field_h, field_w = field.shape
    bottom_h, _ = bottom.shape
    _, right_w = right.shape
    grid = np.zeros((field_h + 1 + bottom_h, field_w + 1 + right_w), dtype=int)
    grid[:field_h, :field_w] = field
    grid[field_h, :] = SEPARATOR
    grid[:, field_w] = SEPARATOR
    grid[field_h + 1 :, :field_w] = bottom
    grid[:field_h, field_w + 1 :] = right
    return grid


def _validate_instance(grid: np.ndarray, intruder_object: dict) -> bool:
    info = _infer_intruder(grid)
    candidate_cells = {tuple(cell) for cell in info["candidate"]["cells"]}
    expected_cells = {tuple(cell) for cell in intruder_object["cells"]}
    if candidate_cells != expected_cells:
        return False

    same_color = [obj for obj in info["objects"] if obj["color"] == info["missing_color"]]
    same_size = [obj for obj in info["objects"] if obj["area"] == info["missing_size"]]
    if len(same_color) < 2 or len(same_size) < 2:
        return False

    expected = grid.copy()
    for r, c in intruder_object["cells"]:
        expected[r, c] = BG
    return np.array_equal(main(grid), expected)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    specs, intruder = _choose_object_specs(rng)
    field_h = rng.randint(12, 17)
    field_w = rng.randint(12, 18)
    placed = _place_objects(field_h, field_w, specs, intruder, rng)
    if placed is None:
        return None
    field, objects, intruder_object = placed

    bottom, right = _build_ledgers(field_h, field_w, objects, intruder_object, rng)
    grid = _assemble_grid(field, bottom, right)
    if max(grid.shape) > 30:
        return None
    if not _validate_instance(grid, intruder_object):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-ledger intruder grid")
