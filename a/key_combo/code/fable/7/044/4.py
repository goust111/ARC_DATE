from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
TEAL = 8
BAR_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

GRID_SIZE_MIN = 16
GRID_SIZE_MAX = 29
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
BAR_LENGTH_MIN = 5
BAR_LENGTH_MAX = 13
DECOY_COUNT_RANGE = (0, 5)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_value(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[dict]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            dense = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            dense[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": dense,
                }
            )
    return comps


def _is_contiguous(indices: np.ndarray) -> bool:
    return indices.size > 0 and int(indices[-1] - indices[0] + 1) == int(indices.size)


def _grows_by_row(mask: np.ndarray) -> bool:
    starts: list[int] = []
    ends: list[int] = []
    for row in mask:
        cols = np.flatnonzero(row)
        if not _is_contiguous(cols):
            return False
        starts.append(int(cols[0]))
        ends.append(int(cols[-1]))
    return all(
        starts[i + 1] == starts[i] - 1 and ends[i + 1] == ends[i] + 1
        for i in range(len(starts) - 1)
    )


def _canonical_horizontal_half(mask: np.ndarray) -> np.ndarray | None:
    if _grows_by_row(mask):
        return mask
    flipped = np.flipud(mask)
    if _grows_by_row(flipped):
        return flipped
    return None


def _canonical_vertical_half(mask: np.ndarray) -> np.ndarray | None:
    if _grows_by_row(mask.T):
        return mask
    flipped = np.fliplr(mask)
    if _grows_by_row(flipped.T):
        return flipped
    return None


def _collect_hidden_bars(grid: np.ndarray) -> list[dict]:
    teal_components = _components(grid == TEAL)
    horizontal_parts: list[dict] = []
    vertical_parts: list[dict] = []

    for index, comp in enumerate(teal_components):
        r0, c0, r1, c1 = comp["bbox"]
        ch = int(r1 - r0 + 1)
        cw = int(c1 - c0 + 1)
        if cw > ch:
            canonical = _canonical_horizontal_half(comp["mask"])
            assert canonical is not None, "invalid horizontal envelope half"
            horizontal_parts.append(
                {
                    "index": index,
                    "bbox": comp["bbox"],
                    "canonical": canonical,
                }
            )
        elif ch > cw:
            canonical = _canonical_vertical_half(comp["mask"])
            assert canonical is not None, "invalid vertical envelope half"
            vertical_parts.append(
                {
                    "index": index,
                    "bbox": comp["bbox"],
                    "canonical": canonical,
                }
            )
        else:
            raise AssertionError("ambiguous teal component shape")

    bars: list[dict] = []
    used: set[int] = set()

    for part in sorted(horizontal_parts, key=lambda item: item["bbox"]):
        if part["index"] in used:
            continue
        r0, c0, r1, c1 = part["bbox"]
        candidates = []
        for other in horizontal_parts:
            if other["index"] in used or other["index"] == part["index"]:
                continue
            or0, oc0, or1, oc1 = other["bbox"]
            if (c0, c1) != (oc0, oc1):
                continue
            top, bottom = (part, other) if r0 < or0 else (other, part)
            tr0, tc0, tr1, tc1 = top["bbox"]
            br0, bc0, br1, bc1 = bottom["bbox"]
            if tc0 != bc0 or tc1 != bc1:
                continue
            if br0 != tr1 + 2:
                continue
            if not np.array_equal(top["canonical"], bottom["canonical"]):
                continue
            candidates.append((top, bottom))

        assert len(candidates) == 1, "expected one mate for each horizontal teal half"
        top, bottom = candidates[0]
        used.add(top["index"])
        used.add(bottom["index"])
        tr0, tc0, tr1, tc1 = top["bbox"]
        bars.append(
            {
                "orientation": "H",
                "axis": tr1 + 1,
                "start": tc0 - 1,
                "end": tc1 + 1,
            }
        )

    for part in sorted(vertical_parts, key=lambda item: item["bbox"]):
        if part["index"] in used:
            continue
        r0, c0, r1, c1 = part["bbox"]
        candidates = []
        for other in vertical_parts:
            if other["index"] in used or other["index"] == part["index"]:
                continue
            or0, oc0, or1, oc1 = other["bbox"]
            if (r0, r1) != (or0, or1):
                continue
            left, right = (part, other) if c0 < oc0 else (other, part)
            lr0, lc0, lr1, lc1 = left["bbox"]
            rr0, rc0, rr1, rc1 = right["bbox"]
            if lr0 != rr0 or lr1 != rr1:
                continue
            if rc0 != lc1 + 2:
                continue
            if not np.array_equal(left["canonical"], right["canonical"]):
                continue
            candidates.append((left, right))

        assert len(candidates) == 1, "expected one mate for each vertical teal half"
        left, right = candidates[0]
        used.add(left["index"])
        used.add(right["index"])
        lr0, lc0, lr1, lc1 = left["bbox"]
        bars.append(
            {
                "orientation": "V",
                "axis": lc1 + 1,
                "start": lr0 - 1,
                "end": lr1 + 1,
            }
        )

    assert len(used) == len(teal_components), "all teal components must belong to exactly one envelope"
    assert bars, "expected at least one hidden bar"
    return sorted(bars, key=lambda item: (item["orientation"], item["axis"], item["start"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each erased generator bar from its teal diamond envelope and recolor it from the unique collinear singleton pixel.

    Input structure:
    - The background is black.
    - Teal pixels appear as disconnected top/bottom or left/right halves of diamond envelopes, each wrapped around one missing horizontal or vertical middle bar.
    - Non-teal foreground pixels are isolated singletons; exactly one lies on each envelope axis and the rest are decoys.

    Rule:
    - Pair the teal halves into full envelopes by matching mirrored shape and shared span.
    - Recover the erased center bar midway between the halves; its length is two cells longer than the teal segment nearest the center.
    - Color that bar with the unique singleton on the same row or column, and delete every singleton pixel.

    Key insight:
    - The teal shapes are not the missing object but the shrunken projection of it, so inverting the shrink step reveals the hidden bar while a lone collinear marker supplies its color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_value(g)

    markers = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != bg) & (g != TEAL))
    ]
    out = g.copy()
    out[(g != bg) & (g != TEAL)] = bg

    hidden_bars = _collect_hidden_bars(g)
    used_markers: set[tuple[int, int]] = set()

    for bar in hidden_bars:
        if bar["orientation"] == "H":
            axis = int(bar["axis"])
            candidates = [
                (r, c, color)
                for r, c, color in markers
                if r == axis and (r, c) not in used_markers
            ]
            assert len(candidates) == 1, "each horizontal envelope must have one collinear singleton"
            _, _, color = candidates[0]
            used_markers.add((candidates[0][0], candidates[0][1]))
            out[axis, bar["start"] : bar["end"] + 1] = color
        else:
            axis = int(bar["axis"])
            candidates = [
                (r, c, color)
                for r, c, color in markers
                if c == axis and (r, c) not in used_markers
            ]
            assert len(candidates) == 1, "each vertical envelope must have one collinear singleton"
            _, _, color = candidates[0]
            used_markers.add((candidates[0][0], candidates[0][1]))
            out[bar["start"] : bar["end"] + 1, axis] = color

    return out


def _reserve_with_padding(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0] - 1, r1 + pad)
    cc1 = min(reserved.shape[1] - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_orientations(count: int) -> list[str]:
    if count == 1:
        return [random.choice(("H", "V"))]
    if random.random() < 0.7:
        h_count = random.randint(1, count - 1)
    else:
        h_count = random.choice((0, count))
    orientations = ["H"] * h_count + ["V"] * (count - h_count)
    random.shuffle(orientations)
    return orientations


def _sample_object_specs(rows: int, cols: int) -> list[dict]:
    count = random.choice(OBJECT_COUNT_CHOICES)
    specs = []
    for orientation in _sample_orientations(count):
        max_length = min(BAR_LENGTH_MAX, cols if orientation == "H" else rows)
        length = random.randint(BAR_LENGTH_MIN, max_length)
        radius = (length - 1) // 2
        full_h = 2 * radius + 1 if orientation == "H" else length
        full_w = length if orientation == "H" else 2 * radius + 1
        specs.append(
            {
                "orientation": orientation,
                "length": length,
                "radius": radius,
                "full_h": full_h,
                "full_w": full_w,
            }
        )
    return sorted(specs, key=lambda spec: -(spec["full_h"] * spec["full_w"]))


def _place_objects(rows: int, cols: int) -> list[dict] | None:
    reserved = np.zeros((rows, cols), dtype=bool)
    objects: list[dict] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for spec in _sample_object_specs(rows, cols):
        candidates = []
        full_h = int(spec["full_h"])
        full_w = int(spec["full_w"])
        radius = int(spec["radius"])
        for r0 in range(0, rows - full_h + 1):
            for c0 in range(0, cols - full_w + 1):
                r1 = r0 + full_h - 1
                c1 = c0 + full_w - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(rows - 1, r1 + 1)
                cc1 = min(cols - 1, c1 + 1)
                if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                if spec["orientation"] == "H":
                    axis_row = r0 + radius
                    if axis_row in used_rows:
                        continue
                    candidates.append((r0, c0, axis_row))
                else:
                    axis_col = c0 + radius
                    if axis_col in used_cols:
                        continue
                    candidates.append((r0, c0, axis_col))

        if not candidates:
            return None

        r0, c0, axis = random.choice(candidates)
        r1 = r0 + full_h - 1
        c1 = c0 + full_w - 1
        _reserve_with_padding(reserved, r0, c0, r1, c1, pad=1)

        obj = dict(spec)
        obj["bbox"] = (r0, c0, r1, c1)
        if spec["orientation"] == "H":
            obj["axis_row"] = axis
            obj["left"] = c0
            obj["right"] = c1
            used_rows.add(axis)
        else:
            obj["axis_col"] = axis
            obj["top"] = r0
            obj["bottom"] = r1
            used_cols.add(axis)
        objects.append(obj)

    return objects


def _free_cell_candidates_on_row(
    reserved: np.ndarray, row: int, forbidden_cols: set[int]
) -> list[tuple[int, int]]:
    return [
        (row, c)
        for c in range(reserved.shape[1])
        if c not in forbidden_cols and not reserved[row, c]
    ]


def _free_cell_candidates_on_col(
    reserved: np.ndarray, col: int, forbidden_rows: set[int]
) -> list[tuple[int, int]]:
    return [
        (r, col)
        for r in range(reserved.shape[0])
        if r not in forbidden_rows and not reserved[r, col]
    ]


def _paint_envelope(grid: np.ndarray, obj: dict) -> None:
    radius = int(obj["radius"])
    if obj["orientation"] == "H":
        row = int(obj["axis_row"])
        left = int(obj["left"])
        right = int(obj["right"])
        for offset in range(1, radius + 1):
            grid[row - offset, left + offset : right - offset + 1] = TEAL
            grid[row + offset, left + offset : right - offset + 1] = TEAL
    else:
        col = int(obj["axis_col"])
        top = int(obj["top"])
        bottom = int(obj["bottom"])
        for offset in range(1, radius + 1):
            grid[top + offset : bottom - offset + 1, col - offset] = TEAL
            grid[top + offset : bottom - offset + 1, col + offset] = TEAL


def _build_input_and_expected(rows: int, cols: int, objects: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    reserved = np.zeros((rows, cols), dtype=bool)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        _reserve_with_padding(reserved, r0, c0, r1, c1, pad=1)

    axis_rows = {int(obj["axis_row"]) for obj in objects if obj["orientation"] == "H"}
    axis_cols = {int(obj["axis_col"]) for obj in objects if obj["orientation"] == "V"}
    marker_colors = random.sample(BAR_COLORS, len(objects))

    for obj, color in zip(objects, marker_colors):
        obj["bar_color"] = int(color)
        if obj["orientation"] == "H":
            choices = _free_cell_candidates_on_row(reserved, int(obj["axis_row"]), axis_cols)
        else:
            choices = _free_cell_candidates_on_col(reserved, int(obj["axis_col"]), axis_rows)
        if not choices:
            return None
        marker = random.choice(choices)
        obj["marker"] = marker
        _reserve_with_padding(reserved, marker[0], marker[1], marker[0], marker[1], pad=1)

    decoy_target = random.randint(*DECOY_COUNT_RANGE)
    decoys: list[tuple[int, int, int]] = []
    for _ in range(decoy_target):
        free_decoys = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if not reserved[r, c] and r not in axis_rows and c not in axis_cols
        ]
        if not free_decoys:
            break
        r, c = random.choice(free_decoys)
        color = random.choice(BAR_COLORS)
        decoys.append((r, c, color))
        _reserve_with_padding(reserved, r, c, r, c, pad=1)

    inp = np.full((rows, cols), BACKGROUND, dtype=int)
    expected = np.full((rows, cols), BACKGROUND, dtype=int)

    for obj in objects:
        _paint_envelope(inp, obj)
        _paint_envelope(expected, obj)
        mr, mc = obj["marker"]
        inp[mr, mc] = int(obj["bar_color"])
        if obj["orientation"] == "H":
            expected[int(obj["axis_row"]), int(obj["left"]) : int(obj["right"]) + 1] = int(obj["bar_color"])
        else:
            expected[int(obj["top"]) : int(obj["bottom"]) + 1, int(obj["axis_col"])] = int(obj["bar_color"])

    for r, c, color in decoys:
        inp[r, c] = int(color)

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    cols = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    objects = _place_objects(rows, cols)
    if objects is None:
        return None

    built = _build_input_and_expected(rows, cols, objects)
    if built is None:
        return None
    inp, expected = built

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
