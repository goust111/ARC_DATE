from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
TABLE_BG = 8
TABLE_SEP = 5
TABLE_ROWS = 2
TABLE_COLS = 3
CELL = 3
TABLE_H = TABLE_ROWS * CELL + (TABLE_ROWS - 1)
TABLE_W = TABLE_COLS * CELL + (TABLE_COLS - 1)
LOWER_START_ROW = 9

REF_COLORS = [1, 2, 3, 4, 6, 7, 9]
RECT_SHAPES_BY_AREA = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4)],
    9: [(3, 3)],
    12: [(3, 4)],
}
GLYPH_MASKS = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BG) -> bool:
    h, w = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + h)
    cc1 = min(dst.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return False
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    mask = view != int(transparent)
    dst[rr0:rr1, cc0:cc1][mask] = view[mask]
    return True


def _spans_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(int(x) for x in separators), int(length)]
    spans = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        if b - a > 1:
            spans.append((a + 1, b))
    return spans


def _parse_reference_table(grid: np.ndarray) -> dict:
    mask = (grid == TABLE_BG) | (grid == TABLE_SEP)
    coords = np.argwhere(mask)
    assert coords.size > 0, "missing reference table"
    box = _bbox(coords)
    patch = _crop(grid, box)

    sep_rows = [r for r in range(patch.shape[0]) if np.all(patch[r, :] == TABLE_SEP)]
    sep_cols = [c for c in range(patch.shape[1]) if np.all(patch[:, c] == TABLE_SEP)]
    row_spans = _spans_from_separators(patch.shape[0], sep_rows)
    col_spans = _spans_from_separators(patch.shape[1], sep_cols)
    assert len(row_spans) == TABLE_ROWS
    assert len(col_spans) == TABLE_COLS
    assert all(r1 - r0 == CELL for r0, r1 in row_spans)
    assert all(c1 - c0 == CELL for c0, c1 in col_spans)

    by_color: dict[int, dict] = {}
    cells: list[dict] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            cell = patch[r0:r1, c0:c1]
            colors = [int(v) for v in np.unique(cell) if v not in (TABLE_BG, TABLE_SEP, BG)]
            assert len(colors) == 1, "each glyph cell must use one drawing color"
            color = int(colors[0])
            glyph = cell == color
            assert int(glyph.sum()) >= 3, "glyphs should be visually identifiable"
            info = {"row": ri, "col": ci, "color": color, "glyph": glyph.copy()}
            assert color not in by_color, "marker colors must resolve uniquely"
            by_color[color] = info
            cells.append(info)

    return {
        "bbox": box,
        "cells": cells,
        "by_color": by_color,
        "mask": mask,
    }


def _parse_rectangles(grid: np.ndarray, table_box: tuple[int, int, int, int]) -> list[dict]:
    lower = grid.copy()
    r0, c0, r1, c1 = table_box
    lower[r0 : r1 + 1, c0 : c1 + 1] = BG
    comps = _components(lower != BG)
    rects: list[dict] = []
    for idx, coords in enumerate(comps):
        box = _bbox(coords)
        patch = _crop(lower, box)
        h, w = patch.shape
        assert int(coords.shape[0]) == h * w, "lower objects must be filled rectangles"
        counts = Counter(int(v) for v in patch.flat)
        assert BG not in counts
        marker_colors = [color for color, count in counts.items() if count == 1]
        assert len(marker_colors) == 1, "each rectangle needs one single-pixel marker"
        marker_color = int(marker_colors[0])
        fill_color, fill_count = max(counts.items(), key=lambda item: item[1])
        assert int(fill_count) == h * w - 1, "rectangle body must be solid except the marker"
        marker_pos = tuple(map(int, np.argwhere(patch == marker_color)[0]))
        rects.append(
            {
                "id": idx,
                "bbox": box,
                "patch": patch,
                "h": h,
                "w": w,
                "area": h * w,
                "marker_color": marker_color,
                "marker_pos": marker_pos,
                "fill_color": int(fill_color),
            }
        )
    assert rects, "expected marked rectangles"
    return sorted(rects, key=lambda item: (item["bbox"][0], item["bbox"][1]))


def _visible_placements(rect: dict, copies: int, direction: int, shape: tuple[int, int], *, drop_partial: bool = False) -> list[tuple[int, int, bool]]:
    h, w = int(rect["h"]), int(rect["w"])
    r0, c0, _r1, _c1 = rect["bbox"]
    step_c = w if int(direction) > 0 else -w
    h_lim, w_lim = shape
    places = [(r0, c0, False)]
    for k in range(1, int(copies) + 1):
        rr = int(r0 + k * h)
        cc = int(c0 + k * step_c)
        visible = rr < h_lim and rr + h > 0 and cc < w_lim and cc + w > 0
        if not visible:
            continue
        partial = rr < 0 or cc < 0 or rr + h > h_lim or cc + w > w_lim
        if drop_partial and partial:
            continue
        places.append((rr, cc, partial))
    return places


def _render(grid: np.ndarray, table: dict, rects: list[dict], *, force_direction: int | None = None, force_copies: int | None = None, reverse_priority: bool = False, drop_partial: bool = False) -> np.ndarray:
    out = np.zeros_like(grid)
    out[table["mask"]] = grid[table["mask"]]
    ordered = sorted(rects, key=lambda item: (item["area"], item["id"]), reverse=not reverse_priority)
    for rect in ordered:
        lookup = table["by_color"][rect["marker_color"]]
        direction = force_direction if force_direction is not None else (1 if lookup["row"] == 0 else -1)
        copies = int(force_copies) if force_copies is not None else int(lookup["col"] + 1)
        for rr, cc, _partial in _visible_placements(rect, copies, direction, out.shape, drop_partial=drop_partial):
            _paste_clip(out, rect["patch"], rr, cc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a glyph lookup table to parameterize one-direction diagonal stamping of marked rectangles.

    Input structure:
    - Black background with a teal reference table in the top band, split by one-pixel grey separators into a 2x3 grid of 3x3 glyph cells.
    - Each table cell contains one monochrome glyph whose drawing color uniquely identifies that cell.
    - Below the table are 2-4 solid rectangles; each rectangle contains exactly one one-pixel marker of a table-glyph color.

    Rule:
    - Parse the table and map each glyph color to its table row and column.
    - For each lower rectangle, use its marker color to find the matching table cell.
    - Table row 0 means stamp the whole marked rectangle down-right by steps of (+h, +w); table row 1 means stamp it down-left by (+h, -w), where h and w are that rectangle's own height and width.
    - Stamp the original rectangle plus `col + 1` diagonal copies, clipping any out-of-bounds part, and resolve overlaps by letting smaller-area rectangles overwrite larger-area rectangles.

    Key insight:
    - The table is a two-parameter encoder: the matched glyph's row chooses the diagonal direction, while its column chooses how many times the rectangle is repeated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    table = _parse_reference_table(g)
    rects = _parse_rectangles(g, table["bbox"])
    for rect in rects:
        assert rect["marker_color"] in table["by_color"], "rectangle marker must match exactly one table glyph color"
    return _render(g, table, rects)


def _build_table() -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    patch = np.full((TABLE_H, TABLE_W), TABLE_BG, dtype=int)
    patch[CELL, :] = TABLE_SEP
    patch[:, CELL] = TABLE_SEP
    patch[:, CELL * 2 + 1] = TABLE_SEP

    colors = random.sample(REF_COLORS, TABLE_ROWS * TABLE_COLS)
    masks = random.sample(GLYPH_MASKS, TABLE_ROWS * TABLE_COLS)
    mapping: dict[int, tuple[int, int]] = {}
    idx = 0
    for ri in range(TABLE_ROWS):
        for ci in range(TABLE_COLS):
            r0 = ri * (CELL + 1)
            c0 = ci * (CELL + 1)
            color = int(colors[idx])
            glyph = masks[idx]
            cell = np.full((CELL, CELL), TABLE_BG, dtype=int)
            cell[glyph] = color
            patch[r0 : r0 + CELL, c0 : c0 + CELL] = cell
            mapping[color] = (ri, ci)
            idx += 1
    return patch, mapping


def _choose_lookup_positions(count: int) -> list[tuple[int, int]]:
    if random.random() < 0.5:
        chosen = [(0, 2), (1, 0)]
    else:
        chosen = [(0, 0), (1, 2)]
    if count >= 3:
        chosen.append(random.choice([(0, 1), (1, 1)]))
    if count >= 4:
        pool = [pos for pos in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)] if pos not in chosen]
        chosen.append(random.choice(pool))
    random.shuffle(chosen)
    return chosen


def _shape_choices(count: int) -> list[tuple[int, int]]:
    areas = random.sample(sorted(RECT_SHAPES_BY_AREA), count)
    shapes = []
    for area in areas:
        shapes.append(random.choice(RECT_SHAPES_BY_AREA[area]))
    random.shuffle(shapes)
    return shapes


def _placement_domain(h: int, w: int, copies: int, direction: int, H: int, W: int, want_clip: bool) -> list[tuple[int, int]]:
    max_r0 = H - (copies + 1) * h
    if max_r0 < LOWER_START_ROW:
        return []
    rows = list(range(LOWER_START_ROW, max_r0 + 1))
    if direction > 0:
        if want_clip:
            c_min = max(0, W - copies * w - w + 1)
            c_max = min(W - w, W - copies * w - 1)
        else:
            c_min = 0
            c_max = W - (copies + 1) * w
    else:
        if want_clip:
            c_min = max(0, copies * w - w + 1)
            c_max = min(W - w, copies * w - 1)
        else:
            c_min = copies * w
            c_max = W - w
    if c_max < c_min:
        return []
    cols = list(range(c_min, c_max + 1))
    return [(r0, c0) for r0 in rows for c0 in cols]


def _input_boxes_separate(boxes: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in boxes:
        if not (r1 < pr0 - 1 or pr1 < r0 - 1 or c1 < pc0 - 1 or pc1 < c0 - 1):
            return False
    return True


def _visible_cells(r0: int, c0: int, h: int, w: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = shape
    cells = set()
    for rr in range(max(0, r0), min(H, r0 + h)):
        for cc in range(max(0, c0), min(W, c0 + w)):
            cells.add((rr, cc))
    return cells


def _layout_stats(rects: list[dict], shape: tuple[int, int]) -> dict:
    by_rect: dict[int, list[set[tuple[int, int]]]] = {}
    clipped = 0
    for rect in rects:
        lookup = rect["lookup"]
        direction = 1 if lookup[0] == 0 else -1
        copies = lookup[1] + 1
        covers = []
        for rr, cc, partial in _visible_placements(rect, copies, direction, shape):
            covers.append(_visible_cells(rr, cc, rect["h"], rect["w"], shape))
            clipped += int(partial)
        by_rect[rect["id"]] = covers

    overlaps: list[tuple[int, int, int]] = []
    rect_list = list(rects)
    for i in range(len(rect_list)):
        for j in range(i + 1, len(rect_list)):
            a = rect_list[i]
            b = rect_list[j]
            if a["area"] == b["area"]:
                continue
            best = 0
            for ca in by_rect[a["id"]]:
                for cb in by_rect[b["id"]]:
                    best = max(best, len(ca & cb))
            if best:
                overlaps.append((a["id"], b["id"], best))
    return {"clipped": clipped, "overlaps": overlaps}


def _build_rectangle_patch(h: int, w: int, fill_color: int, marker_color: int) -> np.ndarray:
    patch = np.full((h, w), int(fill_color), dtype=int)
    marker_r = random.randrange(h)
    marker_c = random.randrange(w)
    patch[marker_r, marker_c] = int(marker_color)
    return patch


def _compose_candidate() -> np.ndarray | None:
    rect_count = random.choices([2, 3, 4], weights=[1, 4, 2], k=1)[0]
    lookup_positions = _choose_lookup_positions(rect_count)
    shapes = _shape_choices(rect_count)
    H = random.randint(20, 26)
    W = random.randint(18, 26)
    if TABLE_W > W:
        return None

    table_patch, color_to_pos = _build_table()
    pos_to_color = {pos: color for color, pos in color_to_pos.items()}

    rect_specs = []
    for idx, (lookup_pos, (h, w)) in enumerate(zip(lookup_positions, shapes, strict=False)):
        marker_color = pos_to_color[lookup_pos]
        direction = 1 if lookup_pos[0] == 0 else -1
        copies = lookup_pos[1] + 1
        rect_specs.append(
            {
                "id": idx,
                "lookup": lookup_pos,
                "marker_color": marker_color,
                "direction": direction,
                "copies": copies,
                "h": h,
                "w": w,
                "area": h * w,
            }
        )

    clip_index = max(range(len(rect_specs)), key=lambda idx: (rect_specs[idx]["copies"], rect_specs[idx]["area"]))
    domains = []
    for idx, spec in enumerate(rect_specs):
        domain = _placement_domain(spec["h"], spec["w"], spec["copies"], spec["direction"], H, W, idx == clip_index)
        if not domain:
            return None
        random.shuffle(domain)
        domains.append(domain)

    fill_colors = random.sample(REF_COLORS, rect_count)
    order = list(range(rect_count))
    order.sort(key=lambda idx: rect_specs[idx]["area"], reverse=True)
    placed: dict[int, dict] = {}
    boxes: list[tuple[int, int, int, int]] = []
    for _attempt in range(200):
        placed.clear()
        boxes.clear()
        ok = True
        for idx in order:
            spec = rect_specs[idx]
            choices = domains[idx][:]
            random.shuffle(choices)
            chosen = None
            for r0, c0 in choices[:120]:
                box = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
                if _input_boxes_separate(boxes, box):
                    chosen = (r0, c0)
                    boxes.append(box)
                    break
            if chosen is None:
                ok = False
                break
            placed[idx] = {**spec, "bbox": (chosen[0], chosen[1], chosen[0] + spec["h"] - 1, chosen[1] + spec["w"] - 1)}
        if not ok:
            continue

        rects = []
        for idx in range(rect_count):
            spec = placed[idx]
            fill_color = fill_colors[idx]
            if fill_color == spec["marker_color"]:
                alternatives = [c for c in REF_COLORS if c != spec["marker_color"] and c not in fill_colors[:idx]]
                if not alternatives:
                    ok = False
                    break
                fill_color = random.choice(alternatives)
            patch = _build_rectangle_patch(spec["h"], spec["w"], fill_color, spec["marker_color"])
            rects.append(
                {
                    **spec,
                    "patch": patch,
                    "fill_color": fill_color,
                }
            )
        if not ok:
            continue

        stats = _layout_stats(rects, (H, W))
        if stats["clipped"] < 1:
            continue
        if not stats["overlaps"]:
            continue
        if max(size for _a, _b, size in stats["overlaps"]) > 4:
            continue

        grid = np.zeros((H, W), dtype=int)
        table_left = random.randint(0, W - TABLE_W)
        grid[:TABLE_H, table_left : table_left + TABLE_W] = table_patch
        for rect in rects:
            r0, c0, r1, c1 = rect["bbox"]
            grid[r0 : r1 + 1, c0 : c1 + 1] = rect["patch"]
        return grid
    return None


def _validate_semantics(grid: np.ndarray) -> None:
    table = _parse_reference_table(grid)
    rects = _parse_rectangles(grid, table["bbox"])
    for rect in rects:
        lookup = table["by_color"][rect["marker_color"]]
        rect["lookup"] = (lookup["row"], lookup["col"])

    used_rows = {rect["lookup"][0] for rect in rects}
    used_counts = {rect["lookup"][1] + 1 for rect in rects}
    assert used_rows == {0, 1}, "both diagonal directions must appear"
    assert len(used_counts) >= 2, "copy count must matter"

    stats = _layout_stats(rects, grid.shape)
    assert stats["clipped"] >= 1, "need at least one clipped stamp"
    assert stats["overlaps"], "need at least one overlap between different-sized rectangles"

    correct = _render(grid, table, rects)
    assert not np.array_equal(correct, grid), "output should add visible stamps"

    wrong_count = _render(grid, table, rects, force_copies=1)
    wrong_dir = _render(grid, table, rects, force_direction=1)
    wrong_priority = _render(grid, table, rects, reverse_priority=True)
    wrong_clip = _render(grid, table, rects, drop_partial=True)

    assert not np.array_equal(correct, wrong_count), "count lookup must affect the result"
    assert not np.array_equal(correct, wrong_dir), "row-controlled direction must affect the result"
    assert not np.array_equal(correct, wrong_priority), "smaller-over-larger priority must affect the result"
    assert not np.array_equal(correct, wrong_clip), "clipping must affect the result"


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _compose_candidate()
        if grid is None:
            continue
        try:
            _validate_semantics(grid)
            out = main(grid)
        except AssertionError:
            continue
        if out.shape == grid.shape and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid instance for puzzle 9")
