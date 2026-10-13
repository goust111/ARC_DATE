from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SPRITE = 1
RED = 2
GREY = 5
DIVIDER = 7
TEAL = 8
CYCLE = (RED, TEAL, GREY)

MAX_ATTEMPTS = 500
_GEN_CALL_COUNT = 0


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _unique_widest_row(coords: np.ndarray) -> int:
    best_width = -1
    best_row = -1
    seen_best = False
    for row in sorted(set(map(int, coords[:, 0]))):
        cols = np.sort(coords[coords[:, 0] == row, 1])
        if cols.size > 1:
            expected = np.arange(int(cols[0]), int(cols[-1]) + 1)
            assert np.array_equal(cols, expected), "each occupied row must form one contiguous segment"
        width = int(cols.size)
        if width > best_width:
            best_width = width
            best_row = int(row)
            seen_best = False
        elif width == best_width:
            seen_best = True
    assert best_width > 1, "multi-column sprites must have a widest row longer than 1"
    assert not seen_best, "multi-column sprites must have a unique widest row"
    return best_row


def _scan_direction(top_lefts: list[tuple[int, int]]) -> str:
    rows = [r for r, _c in top_lefts]
    cols = [c for _r, c in top_lefts]
    row_span = max(rows) - min(rows) + 1
    col_span = max(cols) - min(cols) + 1
    return "col" if col_span >= row_span else "row"


def _ordered_ids(descs: list[dict], key_name: str) -> list[int]:
    direction = _scan_direction([tuple(map(int, d[key_name])) for d in descs])
    if direction == "col":
        ordered = sorted(descs, key=lambda d: (int(d[key_name][1]), int(d[key_name][0]), int(d["id"])))
    else:
        ordered = sorted(descs, key=lambda d: (int(d[key_name][0]), int(d[key_name][1]), int(d["id"])))
    return [int(d["id"]) for d in ordered]


def _color_map_from_order(order: list[int]) -> dict[int, int]:
    return {sprite_id: int(CYCLE[i % len(CYCLE)]) for i, sprite_id in enumerate(order)}


def _single_local(height: int) -> np.ndarray:
    return np.array([(r, 0) for r in range(height)], dtype=int)


def _multi_local(height: int, width: int, bar_row: int, spine_col: int) -> np.ndarray:
    pts = {(r, spine_col) for r in range(height)}
    pts.update((bar_row, c) for c in range(width))
    return np.array(sorted(pts), dtype=int)


def _composition(total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[random.randrange(parts)] += 1
    return shares


def _place_starts(widths: list[int], grid_w: int, target_direction: str) -> list[int] | None:
    base = sum(widths) + max(0, len(widths) - 1)
    extra = grid_w - base
    if extra < 0:
        return None
    if target_direction == "row":
        left_pad = random.randint(0, extra)
        starts = []
        c = left_pad
        for i, width in enumerate(widths):
            starts.append(c)
            c += width
            if i < len(widths) - 1:
                c += 1
        return starts

    gaps = _composition(extra, len(widths) + 1)
    starts = []
    c = gaps[0]
    for i, width in enumerate(widths):
        starts.append(c)
        c += width
        if i < len(widths) - 1:
            c += 1 + gaps[i + 1]
    return starts


def _render_input(h: int, w: int, divider_r: int, descs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[divider_r, :] = DIVIDER
    for desc in descs:
        coords = desc["orig_coords"]
        grid[coords[:, 0], coords[:, 1]] = SPRITE
    return grid


def _render_expected_output(h: int, w: int, divider_r: int, descs: list[dict]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    out[divider_r, :] = DIVIDER
    for desc in descs:
        if desc["kind"] == "single":
            out[divider_r, int(desc["landed_tl"][1])] = BG
    final_order = _ordered_ids(descs, "landed_tl")
    color_map = _color_map_from_order(final_order)
    for desc in descs:
        color = color_map[int(desc["id"])]
        coords = desc["landed_coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _sprite_descriptor(sprite_id: int, kind: str, target_direction: str) -> dict:
    if kind == "single":
        if target_direction == "row":
            height = random.randint(3, 8)
        else:
            height = random.randint(4, 5)
        local = _single_local(height)
        return {
            "id": sprite_id,
            "kind": "single",
            "height": height,
            "width": 1,
            "bar_row": None,
            "spine_col": 0,
            "local": local,
        }

    width = random.choice((2, 3))
    if target_direction == "row":
        height = random.randint(3, 6)
        if random.random() < 0.5:
            bar_row = 0
        else:
            bar_row = height - 1
    else:
        height = random.randint(3, 4)
        bar_row = 1 if height > 2 else 0
    spine_col = random.randrange(width)
    local = _multi_local(height, width, bar_row, spine_col)
    area = int(local.shape[0])
    assert 3 <= area <= 8
    return {
        "id": sprite_id,
        "kind": "multi",
        "height": height,
        "width": width,
        "bar_row": bar_row,
        "spine_col": spine_col,
        "local": local,
    }


def _decorate_with_positions(h: int, divider_r: int, starts: list[int], descs: list[dict]) -> bool:
    occupied = np.zeros((h, max(starts) + max(int(d["width"]) for d in descs) + 1), dtype=bool)
    for desc, left in zip(descs, starts):
        if desc["kind"] == "single":
            landed_top = h - int(desc["height"])
        else:
            landed_top = divider_r - 1 - int(desc["bar_row"])
        landed = desc["local"] + np.array([landed_top, left], dtype=int)
        if landed[:, 0].min() < 0 or landed[:, 0].max() >= h:
            return False
        if occupied[landed[:, 0], landed[:, 1]].any():
            return False
        occupied[landed[:, 0], landed[:, 1]] = True
        desc["left"] = int(left)
        desc["landed_coords"] = landed
        desc["landed_tl"] = (int(landed_top), int(left))
    return True


def _assign_original_positions(divider_r: int, descs: list[dict], final_direction: str) -> bool:
    max_tops = [divider_r - int(desc["height"]) - 1 for desc in descs]
    if min(max_tops) < 0:
        return False

    final_order = _ordered_ids(descs, "landed_tl")
    final_color_map = _color_map_from_order(final_order)

    for _ in range(200):
        if final_direction == "row":
            band_width = random.choice((1, 2))
            band_cap = min(max_tops)
            if band_cap < band_width:
                return False
            start = random.randint(0, band_cap - band_width)
            orig_tops = [random.randint(start, start + band_width) for _ in descs]
        else:
            orig_tops = [random.randint(0, limit) for limit in max_tops]

        for desc, top in zip(descs, orig_tops):
            desc["orig_coords"] = desc["local"] + np.array([top, int(desc["left"])], dtype=int)
            desc["orig_tl"] = (int(top), int(desc["left"]))

        orig_direction = _scan_direction([d["orig_tl"] for d in descs])
        if orig_direction == final_direction:
            continue

        orig_order = _ordered_ids(descs, "orig_tl")
        orig_color_map = _color_map_from_order(orig_order)
        if orig_color_map == final_color_map:
            continue

        return True
    return False


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, descs: list[dict], divider_r: int, final_direction: str) -> bool:
    if inp.shape != expected.shape:
        return False
    if not (inp.ndim == 2 and expected.ndim == 2):
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.all(inp[divider_r, :] == DIVIDER):
        return False
    if np.any(inp[divider_r + 1 :, :] == SPRITE):
        return False

    widths = []
    single_count = 0
    multi_count = 0
    crossed_divider = 0
    for desc in descs:
        lr0, lc0, lr1, lc1 = _bbox(desc["orig_coords"])
        if lr1 >= divider_r - 1:
            return False
        width = lc1 - lc0 + 1
        widths.append(width)
        if width == 1:
            single_count += 1
        else:
            multi_count += 1
        if np.any(desc["landed_coords"][:, 0] >= divider_r):
            crossed_divider += 1

    if single_count < 1 or multi_count < 2:
        return False
    if len(descs) < 4 or len(descs) > 6:
        return False
    if final_direction != _scan_direction([d["landed_tl"] for d in descs]):
        return False
    if _scan_direction([d["orig_tl"] for d in descs]) == final_direction:
        return False
    if crossed_divider < 1:
        return False
    if len(set(map(tuple, [d["landed_tl"] for d in descs]))) != len(descs):
        return False

    actual = main(inp)
    return bool(np.array_equal(actual, expected) and not np.array_equal(inp, expected))


def _gen_attempt(final_direction: str) -> np.ndarray | None:
    if final_direction == "row":
        h = random.randint(18, 22)
        w = random.randint(13, 16)
        total = random.choice((4, 4, 5))
        single_count = 2 if total == 4 else random.choice((1, 2))
    else:
        h = random.randint(18, 22)
        w = random.randint(14, 17)
        total = random.choice((4, 5))
        single_count = random.choice((1, 2))
    multi_count = total - single_count
    if multi_count < 2:
        return None

    divider_r = random.randint(h - 7, h - 4)
    descs = [
        _sprite_descriptor(i, "single" if i < single_count else "multi", final_direction)
        for i in range(total)
    ]
    random.shuffle(descs)
    widths = [int(desc["width"]) for desc in descs]
    starts = _place_starts(widths, w, final_direction)
    if starts is None:
        return None

    if not _decorate_with_positions(h, divider_r, starts, descs):
        return None

    if _scan_direction([d["landed_tl"] for d in descs]) != final_direction:
        return None

    if not _assign_original_positions(divider_r, descs, final_direction):
        return None

    inp = _render_input(h, w, divider_r, descs)
    expected = _render_expected_output(h, w, divider_r, descs)
    if not _semantic_checks(inp, expected, descs, divider_r, final_direction):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the blue sprites, derive a scan order from their landed bbox corners, and recolor them cyclically.

    Input structure:
    - The background is black.
    - Exactly one full-width orange horizontal divider row sits in the lower part of the grid.
    - Several blue 4-connected sprites lie strictly above the divider, with some one-column sprites and some wider sprites.

    Rule:
    - Clear the original blue pixels.
    - For each sprite, inspect its bbox width: width-1 sprites fall straight to the bottom in their column and punch a black hole through the divider at that column; wider sprites fall until their unique widest horizontal row lands immediately above the divider, possibly covering divider cells below it.
    - After all sprites land, take each landed sprite's bbox top-left coordinate. If those top-lefts span at least as much in columns as in rows, order sprites column-major; otherwise order them row-major.
    - Recolor the ordered landed sprites by cycling red, teal, grey over whole sprites, leaving the background and any untouched divider cells unchanged.

    Key insight:
    The final colors are not determined by the original sprite order: width controls the fall, the fall changes the landed top-left positions, and those landed positions determine the recoloring order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    divider_rows = np.where(np.all(g == DIVIDER, axis=1))[0]
    assert divider_rows.size == 1, "expected exactly one full-width divider row"
    divider_r = int(divider_rows[0])

    comps = _components_of_color(g, SPRITE)
    assert comps, "expected at least one sprite"

    landed: list[dict] = []
    out = g.copy()
    out[g == SPRITE] = BG

    occupied = np.zeros((h, w), dtype=bool)
    for sprite_id, coords in enumerate(comps):
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 < divider_r - 0, "all sprites must start above the divider"
        width = c1 - c0 + 1
        if width == 1:
            shift = h - 1 - r1
            out[divider_r, c0] = BG
        else:
            widest_r = _unique_widest_row(coords)
            shift = divider_r - 1 - widest_r
        moved = coords.copy()
        moved[:, 0] += shift
        assert moved[:, 0].min() >= 0 and moved[:, 0].max() < h
        assert moved[:, 1].min() >= 0 and moved[:, 1].max() < w
        if occupied[moved[:, 0], moved[:, 1]].any():
            raise AssertionError("landed sprites must not overlap")
        occupied[moved[:, 0], moved[:, 1]] = True
        landed.append(
            {
                "id": sprite_id,
                "coords": moved,
                "tl": (int(moved[:, 0].min()), int(moved[:, 1].min())),
            }
        )

    direction = _scan_direction([item["tl"] for item in landed])
    if direction == "col":
        ordered = sorted(landed, key=lambda item: (item["tl"][1], item["tl"][0], item["id"]))
    else:
        ordered = sorted(landed, key=lambda item: (item["tl"][0], item["tl"][1], item["id"]))

    for i, item in enumerate(ordered):
        color = int(CYCLE[i % len(CYCLE)])
        coords = item["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    target_direction = "row" if _GEN_CALL_COUNT % 2 == 0 else "col"
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(target_direction)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input for puzzle 2 ({target_direction})")
