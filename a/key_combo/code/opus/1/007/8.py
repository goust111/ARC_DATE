from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
CROSS_COLOR = 4
COLOR_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []

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
            result.append(np.array(coords, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_structure(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must stay in 0..9"
    assert np.any(g != BACKGROUND), "expected non-background content"

    objects = []
    for coords in _components(g != BACKGROUND):
        box = _bbox(coords)
        objects.append({"coords": coords, "bbox": box, "area": int(coords.shape[0])})

    max_area = max(obj["area"] for obj in objects)
    pattern_candidates = [obj for obj in objects if obj["area"] == max_area]
    assert len(pattern_candidates) == 1, "expected one unique largest central pattern"
    pattern = pattern_candidates[0]
    r0, c0, r1, c1 = pattern["bbox"]
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert np.all(patch != BACKGROUND), "central pattern must be a filled rectangle"
    assert patch.size == pattern["area"], "central pattern must occupy its full bounding box"

    legends: list[dict[str, object]] = []
    for obj in objects:
        if obj is pattern:
            continue
        coords = obj["coords"]
        assert obj["area"] == 2, "every non-central object must be a 2-cell legend"
        rows = sorted({int(r) for r in coords[:, 0]})
        cols = sorted(int(c) for c in coords[:, 1])
        assert len(rows) == 1 and cols[1] == cols[0] + 1, "legends must be horizontal dominoes"
        rr = rows[0]
        left_c, right_c = cols
        replacement = int(g[rr, left_c])
        source = int(g[rr, right_c])
        assert replacement != source, "legend colors must differ"
        legends.append(
            {
                "replacement": replacement,
                "source": source,
                "row": rr,
                "left_col": left_c,
            }
        )

    assert 1 <= len(legends) <= 2, "expected one or two legends"
    sources = [int(legend["source"]) for legend in legends]
    assert len(sources) == len(set(sources)), "legend source colors must be distinct"
    legends.sort(key=lambda legend: (int(legend["row"]), int(legend["left_col"])))

    return {
        "grid": g,
        "pattern_bbox": (r0, c0, r1, c1),
        "patch": patch,
        "legends": legends,
    }


def _apply_legends(patch: np.ndarray, legends: list[dict[str, object]] | tuple[dict[str, object], ...]) -> np.ndarray:
    out = np.array(patch, copy=True)
    for legend in legends:
        replacement = int(legend["replacement"])
        source = int(legend["source"])
        out[out == source] = replacement
    return out


def _horizontal_segments(patch: np.ndarray) -> list[dict[str, object]]:
    h, w = patch.shape
    segments: list[dict[str, object]] = []

    for r in range(h):
        c = 0
        while c < w:
            color = int(patch[r, c])
            end = c
            while end + 1 < w and int(patch[r, end + 1]) == color:
                end += 1
            if color != BACKGROUND and end - c + 1 >= 3:
                thin = True
                for cc in range(c, end + 1):
                    if r > 0 and int(patch[r - 1, cc]) == color:
                        thin = False
                        break
                    if r + 1 < h and int(patch[r + 1, cc]) == color:
                        thin = False
                        break
                if thin:
                    segments.append(
                        {
                            "row": r,
                            "start": c,
                            "end": end,
                            "color": color,
                            "cells": {(r, cc) for cc in range(c, end + 1)},
                        }
                    )
            c = end + 1
    return segments


def _vertical_segments(patch: np.ndarray) -> list[dict[str, object]]:
    h, w = patch.shape
    segments: list[dict[str, object]] = []

    for c in range(w):
        r = 0
        while r < h:
            color = int(patch[r, c])
            end = r
            while end + 1 < h and int(patch[end + 1, c]) == color:
                end += 1
            if color != BACKGROUND and end - r + 1 >= 3:
                thin = True
                for rr in range(r, end + 1):
                    if c > 0 and int(patch[rr, c - 1]) == color:
                        thin = False
                        break
                    if c + 1 < w and int(patch[rr, c + 1]) == color:
                        thin = False
                        break
                if thin:
                    segments.append(
                        {
                            "col": c,
                            "start": r,
                            "end": end,
                            "color": color,
                            "cells": {(rr, c) for rr in range(r, end + 1)},
                        }
                    )
            r = end + 1
    return segments


def _find_unique_solution(patch: np.ndarray) -> tuple[dict[str, object], dict[str, object]] | None:
    h_segments = _horizontal_segments(patch)
    v_segments = _vertical_segments(patch)
    if len(h_segments) != 1 or len(v_segments) != 1:
        return None
    h_seg = h_segments[0]
    v_seg = v_segments[0]
    cross = (int(h_seg["row"]), int(v_seg["col"]))
    if int(patch[cross]) in {int(h_seg["color"]), int(v_seg["color"])}:
        return None
    return h_seg, v_seg


def _semantic_output(grid: np.ndarray) -> np.ndarray:
    info = _extract_structure(grid)
    g = info["grid"]
    r0, c0, r1, c1 = info["pattern_bbox"]
    legends = info["legends"]
    recolored_patch = _apply_legends(info["patch"], legends)
    solution = _find_unique_solution(recolored_patch)
    assert solution is not None, "recolored patch must reveal exactly one horizontal and one vertical segment"
    h_seg, v_seg = solution
    assert int(h_seg["color"]) != CROSS_COLOR and int(v_seg["color"]) != CROSS_COLOR

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = recolored_patch

    local_h_cells = set(h_seg["cells"])
    local_v_cells = set(v_seg["cells"])
    global_row = r0 + int(h_seg["row"])
    global_col = c0 + int(v_seg["col"])

    for c in range(g.shape[1]):
        if c0 <= c <= c1:
            local = (int(h_seg["row"]), c - c0)
            if local in local_h_cells or int(recolored_patch[local]) == BACKGROUND:
                out[global_row, c] = int(h_seg["color"])
        else:
            out[global_row, c] = int(h_seg["color"])

    for r in range(g.shape[0]):
        if r0 <= r <= r1:
            local = (r - r0, int(v_seg["col"]))
            if local in local_v_cells or int(recolored_patch[local]) == BACKGROUND:
                out[r, global_col] = int(v_seg["color"])
        else:
            out[r, global_col] = int(v_seg["color"])

    out[global_row, global_col] = CROSS_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use legend dominoes to recolor a central patch, reveal hidden thin segments, and extend them into a cross.

    Input structure:
    - A black background contains one large filled rectangular multicolor patch and 1 or 2 separate horizontal 1x2 legend dominoes.
    - In each legend, the left cell is the replacement color and the right cell is the source color to rewrite inside the central patch.
    - The patch hides one horizontal and one vertical one-cell-thick segment, but some of their cells use source colors so the segments are not fully visible before the rewrite.

    Rule:
    - Treat the largest non-black component as the central patch and apply every legend replacement only inside that patch.
    - After recoloring, find the unique horizontal run and unique vertical run of length at least 3 whose orthogonal neighbors are not the same color.
    - Extend the horizontal run across its whole row and the vertical run across its whole column; outside the patch they overwrite anything, but inside the patch they only overwrite the segment cells or background.
    - Recolor the row/column intersection cell yellow 4 and keep the rest of the recolored patch unchanged.

    Key insight:
    - The legends are not the final answer by themselves: the color rewrite changes the patch structure, and only the rewritten patch reveals which row and column must be extended.
    """
    return _semantic_output(np.asarray(grid, dtype=int))


def _all_non_source_colors(legend_count: int, colors: tuple[int, ...]) -> list[int]:
    if legend_count == 1:
        seg_target, _source, fill_a, fill_b = colors
        return [fill_a, fill_b, seg_target]
    h_target, v_target, _h_source, _v_source, filler = colors
    return [filler, h_target, v_target]


def _make_pattern(ph: int, pw: int, colors: list[int]) -> np.ndarray:
    offset = random.randrange(len(colors))
    patch = np.zeros((ph, pw), dtype=int)
    for r in range(ph):
        for c in range(pw):
            patch[r, c] = int(colors[(r + 2 * c + offset) % len(colors)])
    return patch


def _pick_geometry(ph: int, pw: int) -> dict[str, int | str]:
    h_side = random.choice(["left", "right"])
    v_side = random.choice(["up", "down"])

    col_choices = [
        c
        for c in range(1, pw - 1)
        if (h_side == "left" and c >= 3) or (h_side == "right" and pw - c - 1 >= 3)
    ]
    row_choices = [
        r
        for r in range(1, ph - 1)
        if (v_side == "up" and r >= 3) or (v_side == "down" and ph - r - 1 >= 3)
    ]
    cross_col = random.choice(col_choices)
    cross_row = random.choice(row_choices)

    if h_side == "left":
        h_start = random.randint(0, cross_col - 3)
        h_end = random.randint(h_start + 2, cross_col - 1)
    else:
        h_start = random.randint(cross_col + 1, pw - 3)
        h_end = random.randint(h_start + 2, pw - 1)

    if v_side == "up":
        v_start = random.randint(0, cross_row - 3)
        v_end = random.randint(v_start + 2, cross_row - 1)
    else:
        v_start = random.randint(cross_row + 1, ph - 3)
        v_end = random.randint(v_start + 2, ph - 1)

    return {
        "h_side": h_side,
        "v_side": v_side,
        "cross_row": cross_row,
        "cross_col": cross_col,
        "h_start": h_start,
        "h_end": h_end,
        "v_start": v_start,
        "v_end": v_end,
    }


def _alternating_values(length: int, target: int, source: int) -> list[int]:
    start_with_source = bool(random.getrandbits(1))
    values = []
    for i in range(length):
        use_source = (i % 2 == 0) == start_with_source
        values.append(source if use_source else target)
    return values


def _choose_safe_color(
    patch: np.ndarray,
    r: int,
    c: int,
    allowed: list[int],
    forbidden: set[int],
) -> int:
    neighbors = set()
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1]:
            neighbors.add(int(patch[nr, nc]))

    choices = [color for color in allowed if color not in forbidden]
    for color in choices:
        if color not in neighbors:
            return int(color)
    return int(choices[0])


def _make_patch(ph: int, pw: int, legend_count: int) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, object]]:
    if legend_count == 1:
        seg_target, source, fill_a, fill_b = random.sample(COLOR_POOL, 4)
        colors = (seg_target, source, fill_a, fill_b)
        h_target = v_target = seg_target
        h_source = v_source = source
        legends = [(seg_target, source)]
        base_colors = [seg_target, fill_a, fill_b]
        cross_fill = random.choice([fill_a, fill_b])
    else:
        h_target, v_target, h_source, v_source, filler = random.sample(COLOR_POOL, 5)
        colors = (h_target, v_target, h_source, v_source, filler)
        legends = [(h_target, h_source), (v_target, v_source)]
        base_colors = [filler, h_target, v_target]
        cross_fill = filler

    patch = _make_pattern(ph, pw, base_colors)
    geom = _pick_geometry(ph, pw)
    h_cells = {(int(geom["cross_row"]), c) for c in range(int(geom["h_start"]), int(geom["h_end"]) + 1)}
    v_cells = {(r, int(geom["cross_col"])) for r in range(int(geom["v_start"]), int(geom["v_end"]) + 1)}
    reserved = set(h_cells) | set(v_cells) | {(int(geom["cross_row"]), int(geom["cross_col"]))}
    non_source_colors = _all_non_source_colors(legend_count, colors)

    h_values = _alternating_values(int(geom["h_end"]) - int(geom["h_start"]) + 1, h_target, h_source)
    for offset, c in enumerate(range(int(geom["h_start"]), int(geom["h_end"]) + 1)):
        patch[int(geom["cross_row"]), c] = h_values[offset]

    v_values = _alternating_values(int(geom["v_end"]) - int(geom["v_start"]) + 1, v_target, v_source)
    for offset, r in enumerate(range(int(geom["v_start"]), int(geom["v_end"]) + 1)):
        patch[r, int(geom["cross_col"])] = v_values[offset]

    patch[int(geom["cross_row"]), int(geom["cross_col"])] = cross_fill

    for c in range(int(geom["h_start"]), int(geom["h_end"]) + 1):
        for rr in (int(geom["cross_row"]) - 1, int(geom["cross_row"]) + 1):
            if 0 <= rr < ph and (rr, c) not in reserved and int(patch[rr, c]) == h_target:
                patch[rr, c] = _choose_safe_color(patch, rr, c, non_source_colors, {h_target})
    for c in (int(geom["h_start"]) - 1, int(geom["h_end"]) + 1):
        if 0 <= c < pw and (int(geom["cross_row"]), c) not in reserved and int(patch[int(geom["cross_row"]), c]) == h_target:
            patch[int(geom["cross_row"]), c] = _choose_safe_color(
                patch,
                int(geom["cross_row"]),
                c,
                non_source_colors,
                {h_target},
            )

    for r in range(int(geom["v_start"]), int(geom["v_end"]) + 1):
        for cc in (int(geom["cross_col"]) - 1, int(geom["cross_col"]) + 1):
            if 0 <= cc < pw and (r, cc) not in reserved and int(patch[r, cc]) == v_target:
                patch[r, cc] = _choose_safe_color(patch, r, cc, non_source_colors, {v_target})
    for r in (int(geom["v_start"]) - 1, int(geom["v_end"]) + 1):
        if 0 <= r < ph and (r, int(geom["cross_col"])) not in reserved and int(patch[r, int(geom["cross_col"])]) == v_target:
            patch[r, int(geom["cross_col"])] = _choose_safe_color(
                patch,
                r,
                int(geom["cross_col"]),
                non_source_colors,
                {v_target},
            )

    meta = {
        "geom": geom,
        "legend_count": legend_count,
        "legends": legends,
        "h_target": h_target,
        "v_target": v_target,
    }
    return patch, legends, meta


def _place_legends(
    grid: np.ndarray,
    legends: list[tuple[int, int]],
    cross_col_global: int,
) -> None:
    h, w = grid.shape
    items = legends[:]
    random.shuffle(items)

    if len(items) == 1:
        rows = [random.choice([0, h - 1])]
    else:
        rows = [0, h - 1]
        if random.random() < 0.5:
            rows.reverse()

    for i, (replacement, source) in enumerate(items):
        row = rows[i if len(items) == 2 else 0]
        if i == 0 and random.random() < 0.7:
            aligned = [start for start in {cross_col_global - 1, cross_col_global} if 0 <= start <= w - 2]
            start = random.choice(aligned) if aligned else random.randint(0, w - 2)
        else:
            start = random.randint(0, w - 2)
        grid[row, start] = replacement
        grid[row, start + 1] = source


def _all_proper_legend_subsets(legends: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    subsets: list[list[dict[str, object]]] = []
    n = len(legends)
    for mask in range(1 << n):
        if mask == (1 << n) - 1:
            continue
        subset = [legends[i] for i in range(n) if (mask >> i) & 1]
        subsets.append(subset)
    return subsets


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        info = _extract_structure(grid)
        patch = info["patch"]
        legends = info["legends"]
        assert 3 <= len(set(map(int, np.unique(patch)))) <= 5

        recolored = _apply_legends(patch, legends)
        solution = _find_unique_solution(recolored)
        assert solution is not None
        h_seg, v_seg = solution
        assert int(h_seg["color"]) != CROSS_COLOR and int(v_seg["color"]) != CROSS_COLOR

        cross = (int(h_seg["row"]), int(v_seg["col"]))
        assert int(recolored[cross]) not in {int(h_seg["color"]), int(v_seg["color"])}

        h_cells = set(h_seg["cells"])
        v_cells = set(v_seg["cells"])
        for legend in legends:
            replacement = int(legend["replacement"])
            source = int(legend["source"])
            source_cells = {tuple(map(int, pos)) for pos in np.argwhere(patch == source)}
            assert source_cells, "every legend source color must appear in the patch"

            allowed = set()
            if int(h_seg["color"]) == replacement:
                allowed |= h_cells
            if int(v_seg["color"]) == replacement:
                allowed |= v_cells
            assert allowed, "legend replacement color must match at least one revealed segment"
            assert source_cells <= allowed, "source color may only occur on the hidden segment cells"
            assert source_cells & allowed, "source color must actively disguise the segment"

        for subset in _all_proper_legend_subsets(legends):
            partial = _apply_legends(patch, subset)
            assert _find_unique_solution(partial) is None, "all legends must be necessary for the full solve"

        expected = _semantic_output(grid)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        return True
    except (AssertionError, IndexError, ValueError):
        return False


def _build_candidate() -> np.ndarray | None:
    ph = random.randint(6, 8)
    pw = random.randint(6, 8)
    legend_count = random.choice([1, 2])
    patch, legends, meta = _make_patch(ph, pw, legend_count)

    h = random.randint(max(13, ph + 5), min(19, 30))
    w = random.randint(max(13, pw + 5), min(19, 30))

    top_low = 2
    top_high = h - ph - 3
    left_low = 2
    left_high = w - pw - 2
    if top_low > top_high or left_low > left_high:
        return None

    top = random.randint(top_low, top_high)
    left = random.randint(left_low, left_high)
    grid = np.zeros((h, w), dtype=int)
    grid[top : top + ph, left : left + pw] = patch

    geom = meta["geom"]
    cross_col_global = left + int(geom["cross_col"])
    _place_legends(grid, legends, cross_col_global)

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-segment legend puzzle")
