from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEMPLATE_COLORS = (BLUE, RED, GREEN)

GLYPH_LIBRARY = (
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _diag_neighbors(r: int, c: int):
    yield r - 1, c - 1
    yield r - 1, c + 1
    yield r + 1, c - 1
    yield r + 1, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= dst.shape[0] and c0 + w <= dst.shape[1]
    if transparent == BG:
        mask = patch != BG
        dst[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    else:
        dst[r0 : r0 + h, c0 : c0 + w] = patch


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    out: list[np.ndarray] = []
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
                if connectivity == 4:
                    nbrs = _neighbors4(rr, cc)
                else:
                    nbrs = (
                        (rr + dr, cc + dc)
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if not (dr == 0 and dc == 0)
                    )
                for nr, nc in nbrs:
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _is_diag_noise(mask: np.ndarray, r: int, c: int) -> bool:
    orth = any(_in_bounds(mask.shape, nr, nc) and mask[nr, nc] for nr, nc in _neighbors4(r, c))
    diag = any(_in_bounds(mask.shape, nr, nc) and mask[nr, nc] for nr, nc in _diag_neighbors(r, c))
    return (not orth) and diag


def _detect_noise(grid: np.ndarray, color: int, work_r0: int) -> np.ndarray:
    mask = grid == int(color)
    noise = np.zeros_like(mask, dtype=bool)
    for r in range(work_r0, grid.shape[0]):
        for c in range(1, grid.shape[1] - 1):
            if mask[r, c] and _is_diag_noise(mask, r, c):
                noise[r, c] = True
    return noise


def _fill_line_gaps(clean_mask: np.ndarray, noise: np.ndarray, work_r0: int) -> np.ndarray:
    out = clean_mask.copy()
    span = 1
    for comp in _components(noise, connectivity=8):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        span = max(span, int(r1 - r0 + 1), int(c1 - c0 + 1))

    h, w = clean_mask.shape
    for r in range(work_r0, h):
        cols = np.where(clean_mask[r])[0]
        if cols.size < 2:
            continue
        starts = [int(cols[0])]
        ends: list[int] = []
        prev = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
            else:
                ends.append(prev)
                starts.append(c)
                prev = c
        ends.append(prev)
        for end0, start1 in zip(ends, starts[1:]):
            gap0 = end0 + 1
            gap1 = start1 - 1
            if gap1 < gap0 or gap1 - gap0 + 1 > span:
                continue
            rr0 = max(work_r0, r - 1)
            rr1 = min(h - 1, r + 1)
            cc0 = max(1, gap0 - 1)
            cc1 = min(w - 2, gap1 + 1)
            if noise[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                out[r, gap0 : gap1 + 1] = True

    for c in range(1, w - 1):
        rows = np.where(clean_mask[:, c])[0]
        rows = rows[rows >= work_r0]
        if rows.size < 2:
            continue
        starts = [int(rows[0])]
        ends: list[int] = []
        prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r == prev + 1:
                prev = r
            else:
                ends.append(prev)
                starts.append(r)
                prev = r
        ends.append(prev)
        for end0, start1 in zip(ends, starts[1:]):
            gap0 = end0 + 1
            gap1 = start1 - 1
            if gap1 < gap0 or gap1 - gap0 + 1 > span:
                continue
            rr0 = max(work_r0, gap0 - 1)
            rr1 = min(h - 1, gap1 + 1)
            cc0 = max(1, c - 1)
            cc1 = min(w - 2, c + 1)
            if noise[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                out[gap0 : gap1 + 1, c] = True
    return out


def _extract_line(mask: np.ndarray, work_r0: int) -> dict[str, int]:
    coords = np.argwhere(mask)
    coords = coords[coords[:, 0] >= work_r0]
    assert coords.size > 0
    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    if len(rows) == 1:
        row = rows[0]
        c0 = cols[0]
        c1 = cols[-1]
        assert np.all(mask[row, c0 : c1 + 1])
        return {"orientation": "H", "r": row, "c0": c0, "c1": c1}
    if len(cols) == 1:
        col = cols[0]
        r0 = rows[0]
        r1 = rows[-1]
        assert np.all(mask[r0 : r1 + 1, col])
        return {"orientation": "V", "c": col, "r0": r0, "r1": r1}
    raise AssertionError("work object must restore to one straight line")


def _expected_markers(line: dict[str, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if line["orientation"] == "H":
        r = int(line["r"])
        c0 = int(line["c0"])
        c1 = int(line["c1"])
        return (r - 1, c0 - 1), (r + 1, c1 + 1)
    c = int(line["c"])
    r0 = int(line["r0"])
    r1 = int(line["r1"])
    return (r0 - 1, c - 1), (r1 + 1, c + 1)


def _extract_templates(grid: np.ndarray, mid: int) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    top = grid[:mid]
    for color in TEMPLATE_COLORS:
        mask = top == int(color)
        if not mask.any():
            continue
        comps = _components(mask, connectivity=4)
        assert len(comps) == 1
        bbox = _bbox_from_coords(comps[0])
        patch = _crop_bbox(top, bbox)
        assert patch.shape[0] in (3, 5) and patch.shape[1] in (3, 5)
        assert patch.shape[0] % 2 == 1 and patch.shape[1] % 2 == 1
        templates[int(color)] = patch
    assert templates
    return templates


def _clear_work_color(out: np.ndarray, grid: np.ndarray, color: int, work_r0: int) -> None:
    mask = grid == int(color)
    mask[:work_r0, :] = False
    out[mask] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase diagonal staircase noise, restore the hidden endpoint anchors, and rebuild each work glyph from the color-matched template.

    Input structure:
    - The background is black, and grey border bars fill the full left and right columns.
    - The upper half contains 1 or 2 complete monochrome glyph templates in blue, red, or green.
    - The lower half contains one disturbed horizontal or vertical line per template color, plus yellow endpoint markers near the line ends; some same-color diagonal-only noise pixels break the line and may cover one marker.

    Rule:
    - Read the glyph template for each foreground color from the upper half.
    - In the lower half, delete same-color pixels that have no orthogonal same-color neighbor but do have a diagonal same-color neighbor.
    - Use the deleted diagonal span to refill the short horizontal or vertical gap of that color, recovering one complete straight line.
    - Infer the two yellow endpoint markers from the restored line ends, treating a deleted noise pixel at an expected marker position as a hidden marker.
    - Remove that work line and its markers, then draw the matching template centered at the midpoint of the two markers.
    - Keep the border bars and the upper template area unchanged.

    Key insight:
    The diagonal noise is not just clutter: deleting it both reveals a hidden anchor marker and provides the local evidence needed to complete the broken straight line, whose color selects the glyph template to reconstruct.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2
    assert np.all(g[:, 0] == GREY) and np.all(g[:, w - 1] == GREY)

    mid = h // 2
    templates = _extract_templates(g, mid)
    out = g.copy()

    for color, patch in templates.items():
        noise = _detect_noise(g, color, mid)
        clean_mask = (g == int(color)) & (~noise)
        clean_mask[:mid, :] = False
        restored = _fill_line_gaps(clean_mask, noise, mid)
        line = _extract_line(restored, mid)
        marker_a, marker_b = _expected_markers(line)
        for mr, mc in (marker_a, marker_b):
            assert _in_bounds(g.shape, mr, mc)
            assert int(g[mr, mc]) == YELLOW or bool(noise[mr, mc])

        _clear_work_color(out, g, color, mid)
        for mr, mc in (marker_a, marker_b):
            out[mr, mc] = BG

        center_r = (marker_a[0] + marker_b[0]) // 2
        center_c = (marker_a[1] + marker_b[1]) // 2
        ph, pw = patch.shape
        top_r = center_r - ph // 2
        left_c = center_c - pw // 2
        _paste(out, patch, top_r, left_c, transparent=BG)

    return out


def _tint(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _bbox_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _coords_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(coords, dtype=int)
    return _bbox_from_coords(arr)


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _marker_positions(item: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    if item["orientation"] == "H":
        r = int(item["row"])
        c0 = int(item["start"])
        c1 = int(item["end"])
        return (r - 1, c0 - 1), (r + 1, c1 + 1)
    c = int(item["col"])
    r0 = int(item["start"])
    r1 = int(item["end"])
    return (r0 - 1, c - 1), (r1 + 1, c + 1)


def _build_item(
    *,
    shape: tuple[int, int],
    mid: int,
    color: int,
    glyph: np.ndarray,
    orientation: str,
    hide_side: str | None,
    blocked: set[tuple[int, int]],
    reserved_boxes: list[tuple[int, int, int, int]],
) -> dict | None:
    h, w = shape
    gh, gw = glyph.shape
    lengths = [5, 7, 9]
    random.shuffle(lengths)

    for line_len in lengths:
        half = line_len // 2
        for _ in range(120):
            if orientation == "H":
                row_lo = max(mid + gh // 2, mid + 2)
                row_hi = min(h - 1 - gh // 2, h - 3)
                col_lo = max(gw // 2 + 1, half + 2)
                col_hi = min(w - 2 - gw // 2, w - 3 - half)
                if row_lo > row_hi or col_lo > col_hi:
                    continue
                row = random.randint(row_lo, row_hi)
                center_c = random.randint(col_lo, col_hi)
                c0 = center_c - half
                c1 = center_c + half
                line = [(row, c) for c in range(c0, c1 + 1)]
                gap_len = random.randint(1, min(2, line_len - 4))
                gap_start = random.randint(c0 + 2, c1 - gap_len - 1)
                gap = [(row, c) for c in range(gap_start, gap_start + gap_len)]
                gap_noise = [(row + 1 + i, gap_start + i) for i in range(gap_len)]
                item = {
                    "orientation": "H",
                    "row": row,
                    "start": c0,
                    "end": c1,
                    "center": (row, center_c),
                    "gap": gap,
                    "gap_noise": gap_noise,
                }
            else:
                col_lo = max(gw // 2, 2)
                col_hi = min(w - 1 - gw // 2, w - 3)
                row_lo = max(mid + gh // 2 + 1, mid + 2 + half)
                row_hi = min(h - 1 - gh // 2, h - 3 - half)
                if row_lo > row_hi or col_lo > col_hi:
                    continue
                col = random.randint(col_lo, col_hi)
                center_r = random.randint(row_lo, row_hi)
                r0 = center_r - half
                r1 = center_r + half
                line = [(r, col) for r in range(r0, r1 + 1)]
                gap_len = random.randint(1, min(2, line_len - 4))
                gap_start = random.randint(r0 + 2, r1 - gap_len - 1)
                gap = [(r, col) for r in range(gap_start, gap_start + gap_len)]
                gap_noise = [(gap_start + i, col + 1 + i) for i in range(gap_len)]
                item = {
                    "orientation": "V",
                    "col": col,
                    "start": r0,
                    "end": r1,
                    "center": (center_r, col),
                    "gap": gap,
                    "gap_noise": gap_noise,
                }

            marker_a, marker_b = _marker_positions(item)
            hidden_noise: list[tuple[int, int]] = []
            visible_markers = {marker_a, marker_b}
            if hide_side == "first":
                hidden_noise = [marker_a]
                if orientation == "H":
                    hidden_noise.append((marker_a[0] - 1, marker_a[1] + 1))
                else:
                    hidden_noise.append((marker_a[0] + 1, marker_a[1] - 1))
                visible_markers.remove(marker_a)
            elif hide_side == "second":
                hidden_noise = [marker_b]
                if orientation == "H":
                    hidden_noise.append((marker_b[0] + 1, marker_b[1] - 1))
                else:
                    hidden_noise.append((marker_b[0] - 1, marker_b[1] + 1))
                visible_markers.remove(marker_b)

            input_cells = set(line) | set(gap_noise) | set(hidden_noise) | set(visible_markers)
            if any(not _in_bounds(shape, r, c) or c in (0, w - 1) for r, c in input_cells):
                continue
            if any(r < mid for r, _c in input_cells):
                continue
            if blocked & input_cells:
                continue

            output_box = (
                item["center"][0] - gh // 2,
                item["center"][1] - gw // 2,
                item["center"][0] + gh // 2,
                item["center"][1] + gw // 2,
            )
            if output_box[0] < mid or output_box[1] <= 0 or output_box[2] >= h or output_box[3] >= w - 1:
                continue
            if any(not _bbox_far(output_box, box, pad=1) for box in reserved_boxes):
                continue

            line_after_gap = [cell for cell in line if cell not in set(gap)]
            item.update(
                {
                    "color": color,
                    "glyph": glyph,
                    "line": line,
                    "line_after_gap": line_after_gap,
                    "marker_a": marker_a,
                    "marker_b": marker_b,
                    "visible_markers": sorted(visible_markers),
                    "hidden_noise": hidden_noise,
                    "input_cells": sorted(input_cells),
                    "noise_cells": sorted(set(gap_noise) | set(hidden_noise)),
                    "output_box": output_box,
                }
            )
            return item
    return None


def _place_templates(
    grid: np.ndarray,
    chosen: list[tuple[int, int, np.ndarray]],
    mid: int,
) -> list[tuple[int, int, int, int]]:
    placed: list[tuple[int, int, int, int]] = []
    for color, _idx, glyph in chosen:
        gh, gw = glyph.shape
        candidates = []
        for r0 in range(1, mid - gh):
            for c0 in range(2, grid.shape[1] - 2 - gw + 1):
                box = (r0, c0, r0 + gh - 1, c0 + gw - 1)
                if any(not _bbox_far(box, old, pad=2) for old in placed):
                    continue
                candidates.append((r0, c0, box))
        if not candidates:
            return []
        r0, c0, box = random.choice(candidates)
        _paste(grid, _tint(glyph, color), r0, c0, transparent=BG)
        placed.append(box)
    return placed


def _expected_output(base: np.ndarray, items: list[dict]) -> np.ndarray:
    out = base.copy()
    for item in items:
        cr, cc = item["center"]
        glyph = item["glyph"]
        _paste(out, _tint(glyph, item["color"]), cr - glyph.shape[0] // 2, cc - glyph.shape[1] // 2)
    return out


def _render_input(base: np.ndarray, items: list[dict]) -> np.ndarray:
    inp = base.copy()
    for item in items:
        color = int(item["color"])
        for r, c in item["line_after_gap"]:
            inp[r, c] = color
        for r, c in item["gap_noise"]:
            inp[r, c] = color
        for r, c in item["hidden_noise"]:
            inp[r, c] = color
        for r, c in item["visible_markers"]:
            inp[r, c] = YELLOW
    return inp


def _validate_case(inp: np.ndarray, expected: np.ndarray, items: list[dict], mid: int) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    for item in items:
        color = int(item["color"])
        noise = _detect_noise(inp, color, mid)
        if set(map(tuple, np.argwhere(noise))) != set(item["noise_cells"]):
            return False
        clean_mask = (inp == color) & (~noise)
        clean_mask[:mid, :] = False
        restored = _fill_line_gaps(clean_mask, noise, mid)
        line = _extract_line(restored, mid)
        expected_markers = {item["marker_a"], item["marker_b"]}
        if set(_expected_markers(line)) != expected_markers:
            return False
        if not set(item["gap"]).difference(set(item["line_after_gap"])):
            return False
        if set(item["gap"]).issubset(set(item["line_after_gap"])):
            return False
        if item["hidden_noise"]:
            if inp[item["hidden_noise"][0]] == YELLOW:
                return False
        else:
            if len(item["visible_markers"]) != 2:
                return False

    if len(items) == 2:
        swapped = expected.copy()
        first, second = items
        for item in items:
            glyph = second["glyph"] if item is first else first["glyph"]
            cr, cc = item["center"]
            box = (
                cr - item["glyph"].shape[0] // 2,
                cc - item["glyph"].shape[1] // 2,
                cr + item["glyph"].shape[0] // 2,
                cc + item["glyph"].shape[1] // 2,
            )
            swapped[box[0] : box[2] + 1, box[1] : box[3] + 1] = BG
            _paste(swapped, _tint(glyph, item["color"]), cr - glyph.shape[0] // 2, cc - glyph.shape[1] // 2)
        if np.array_equal(swapped, expected):
            return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    mid = h // 2
    base = np.full((h, w), BG, dtype=int)
    base[:, 0] = GREY
    base[:, w - 1] = GREY

    n_templates = random.choices([1, 2], weights=[2, 5], k=1)[0]
    colors = random.sample(list(TEMPLATE_COLORS), n_templates)
    glyph_indices = random.sample(range(len(GLYPH_LIBRARY)), n_templates)
    chosen = [(color, idx, GLYPH_LIBRARY[idx]) for color, idx in zip(colors, glyph_indices)]

    template_boxes = _place_templates(base, chosen, mid)
    if len(template_boxes) != n_templates:
        raise RuntimeError("template placement failed")

    items: list[dict] = []
    blocked = _expand_cells(set(), base.shape, radius=0)
    for box in template_boxes:
        for r in range(box[0], box[2] + 1):
            for c in range(box[1], box[3] + 1):
                blocked.add((r, c))
    blocked = _expand_cells(blocked, base.shape, radius=1)

    reserved_boxes = list(template_boxes)
    hide_index = random.randrange(n_templates)
    orientations = ["H", "V"] if n_templates == 2 else [random.choice(["H", "V"])]
    random.shuffle(orientations)

    for i, (color, _idx, glyph) in enumerate(chosen):
        hide_side = None
        if i == hide_index:
            hide_side = random.choice(["first", "second"])
        item = _build_item(
            shape=base.shape,
            mid=mid,
            color=color,
            glyph=glyph,
            orientation=orientations[i if n_templates == 2 else 0],
            hide_side=hide_side,
            blocked=blocked,
            reserved_boxes=reserved_boxes,
        )
        if item is None:
            raise RuntimeError("item placement failed")
        items.append(item)
        blocked = _expand_cells(blocked | set(item["input_cells"]), base.shape, radius=1)
        reserved_boxes.append(item["output_box"])

    inp = _render_input(base, items)
    out = _expected_output(base, items)
    if not _validate_case(inp, out, items, mid):
        raise RuntimeError("semantic validation failed")
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, _out = _generate_case()
            return inp
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 8")
