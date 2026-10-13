from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    target = int(color)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != target:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == target:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    markers: list[tuple[int, int, int]] = []
    legend_cells: set[tuple[int, int]] = set()
    r = 0
    while r <= h - 2:
        block = g[r : r + 2, 0:2]
        if (
            block.shape == (2, 2)
            and np.all(block != int(bg))
            and np.all(block[:, 0] == int(block[0, 0]))
            and np.all(block[:, 1] == int(block[0, 1]))
        ):
            if r > 0:
                assert np.all(g[r - 1, 0:2] == int(bg)), "legend markers must be separated vertically"
            src = int(block[0, 0])
            halo = int(block[0, 1])
            assert src != halo, "legend source and halo colors must differ"
            markers.append((r, src, halo))
            for rr in range(r, r + 2):
                for cc in range(2):
                    legend_cells.add((rr, cc))
            r += 2
            continue
        r += 1

    assert markers, "expected at least one left-edge legend marker"
    markers.sort()
    sources = [src for _r, src, _halo in markers]
    assert len(set(sources)) == len(sources), "legend source colors must be unique"
    for (r0, _s0, _h0), (r1, _s1, _h1) in zip(markers, markers[1:]):
        assert r1 >= r0 + 3, "legend markers must have at least one blank row between them"

    legend_mask = np.zeros_like(g, dtype=bool)
    top = markers[0][0]
    bottom = markers[-1][0] + 1
    legend_mask[top : bottom + 1, 0:2] = True
    return {src: halo for _r, src, halo in markers}, legend_mask, legend_cells


def _extract_objects(
    grid: np.ndarray, legend_map: dict[int, int], legend_cells: set[tuple[int, int]]
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict[str, object]] = []
    for color in legend_map:
        comps = _components_of_color(g, color)
        non_legend = [comp for comp in comps if not set(comp).issubset(legend_cells)]
        assert len(non_legend) == 1, "each legend source color must label exactly one object"
        comp = non_legend[0]
        r0, c0, r1, c1 = _bbox(comp)
        objects.append(
            {
                "color": int(color),
                "coords": tuple(sorted(comp)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    keys = [(obj["bbox"][1], obj["bbox"][0]) for obj in objects]
    assert len(set(keys)) == len(keys), "objects must have a unique column-primary order"
    return objects


def _ordered_objects(objects: list[dict[str, object]], mode: str = "colrow") -> list[dict[str, object]]:
    if mode == "colrow":
        return sorted(objects, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    if mode == "rowcol":
        return sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    if mode == "reverse_colrow":
        return sorted(objects, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]), reverse=True)
    raise ValueError(f"unknown order mode: {mode}")


def _render_halos(
    grid: np.ndarray,
    legend_map: dict[int, int],
    legend_mask: np.ndarray,
    objects: list[dict[str, object]],
    *,
    order_mode: str = "colrow",
    neighbor_mode: str = "cyclic_right",
    paint_mode: str = "forward",
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0
    out = g.copy()
    ordered = _ordered_objects(objects, mode=order_mode)
    halo_colors: list[int] = []
    for idx, obj in enumerate(ordered):
        if neighbor_mode == "cyclic_right":
            ref_obj = ordered[(idx + 1) % len(ordered)]
        elif neighbor_mode == "self":
            ref_obj = obj
        else:
            raise ValueError(f"unknown neighbor mode: {neighbor_mode}")
        halo_colors.append(int(legend_map[int(ref_obj["color"])]))

    indices = list(range(len(ordered)))
    if paint_mode == "reverse":
        indices.reverse()
    elif paint_mode != "forward":
        raise ValueError(f"unknown paint mode: {paint_mode}")

    input_bg = g == int(bg)
    for idx in indices:
        obj = ordered[idx]
        halo_color = halo_colors[idx]
        r0, c0, r1, c1 = obj["bbox"]
        pr0 = max(0, r0 - 1)
        pc0 = max(0, c0 - 1)
        pr1 = min(h - 1, r1 + 1)
        pc1 = min(w - 1, c1 + 1)
        candidate = np.zeros_like(g, dtype=bool)
        candidate[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        candidate &= input_bg
        candidate &= ~legend_mask
        out[candidate & (out == int(bg))] = int(halo_color)
    return out


def _scene_from_grid(grid: np.ndarray) -> tuple[int, dict[int, int], np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert int(bg) == 0, "background must be black"
    legend_map, legend_mask, legend_cells = _detect_legend(g, bg)
    objects = _extract_objects(g, legend_map, legend_cells)
    assert len(objects) == len(legend_map), "number of objects must match the legend count"
    return bg, legend_map, legend_mask, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add one-cell halo rectangles around the objects, but color each halo from that object's cyclic right neighbor in the legend.

    Input structure:
    - The background is black.
    - Along the left edge sits a vertical stack of separated 2x2 legend blocks; each block's left column is an object color and its right column is the halo color paired with that object color.
    - Elsewhere there is exactly one 4-connected monochrome object for each legend source color, and each object is separated from the others.

    Rule:
    - Keep every original pixel unchanged.
    - Sort the non-legend objects by the top-left corner of their bounding box using column first, then row.
    - Treat that order as a cycle, so each object's right neighbor is the next object in the sorted list, wrapping around at the end.
    - For each object, expand its bounding box by one cell in every direction, clip to the grid, and mark the background cells in that expanded rectangle as its halo candidates.
    - The halo color for an object is not taken from its own legend row: it is the halo color paired with the right neighbor's object color.
    - Paint halos in sorted order, writing only onto cells that are still background; earlier halos and all original foreground, including the legend area, win ties.

    Key insight:
    - The legend gives a color lookup table, but you must first determine the spatial cyclic order of the objects because each object borrows the halo color of its cyclic right neighbor rather than its own.
    """
    _bg, legend_map, legend_mask, objects = _scene_from_grid(np.asarray(grid, dtype=int))
    return _render_halos(np.asarray(grid, dtype=int), legend_map, legend_mask, objects)


_PALETTE = tuple(range(1, 10))
_GRID_H_RANGE = (14, 19)
_GRID_W_RANGE = (15, 20)
_OBJECT_COUNT_RANGE = (3, 5)
_MAX_ATTEMPTS = 600


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _make_shape(area: int, height: int, width: int) -> np.ndarray | None:
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(120):
        start = random.choice(all_cells)
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if 0 not in rows or height - 1 not in rows or 0 not in cols or width - 1 not in cols:
            continue
        patch = np.zeros((height, width), dtype=int)
        for r, c in cells:
            patch[r, c] = 1
        return patch
    return None


def _random_patch() -> np.ndarray | None:
    for _ in range(120):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        area = random.randint(max(4, height + width - 1), min(8, height * width))
        patch = _make_shape(area, height, width)
        if patch is not None:
            return patch
    return None


def _stamp(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(patch == 1):
        r = int(r0 + rr)
        c = int(c0 + cc)
        grid[r, c] = int(color)
        coords.append((r, c))
    return coords


def _can_place(
    occupied: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    gap: int = 1,
) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    rs, cs = np.argwhere(patch == 1).T
    cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in zip(rs, cs)]
    for r, c in cells:
        if occupied[r, c]:
            return False
        rmin = max(0, r - gap)
        rmax = min(h - 1, r + gap)
        cmin = max(0, c - gap)
        cmax = min(w - 1, c + gap)
        if np.any(occupied[rmin : rmax + 1, cmin : cmax + 1]):
            return False
    return True


def _bbox_from_patch(patch: np.ndarray, r0: int, c0: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1)


def _candidate_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return max(ar0 - 1, br0 - 1) <= min(ar1 + 1, br1 + 1) and max(ac0 - 1, bc0 - 1) <= min(ac1 + 1, bc1 + 1)


def _row_major_differs(objects: list[dict[str, object]]) -> bool:
    colrow = [obj["color"] for obj in _ordered_objects(objects, "colrow")]
    rowcol = [obj["color"] for obj in _ordered_objects(objects, "rowcol")]
    return colrow != rowcol


def _legend_layout(height: int, count: int) -> tuple[int, list[int]] | None:
    min_span = 2 * count + (count - 1)
    if min_span > height:
        return None
    extra = height - min_span
    gap_rows = [1] * (count - 1)
    for _ in range(extra):
        if random.random() < 0.65:
            gap_rows[random.randrange(count - 1)] += 1
    span = 2 * count + sum(gap_rows)
    top = random.randint(0, height - span)
    rows = []
    cur = top
    for idx in range(count):
        rows.append(cur)
        cur += 2
        if idx != count - 1:
            cur += gap_rows[idx]
    return top, rows


def _choose_colors(count: int) -> tuple[list[int], dict[int, int]] | None:
    if count > 5:
        return None
    sources = random.sample(_PALETTE, count)
    remaining = [c for c in _PALETTE if c not in sources]
    halos: list[int] = []
    if len(remaining) >= count and random.random() < 0.8:
        halos = random.sample(remaining, count)
    else:
        choices = [c for c in _PALETTE]
        for src in sources:
            local = [c for c in choices if c != src]
            halos.append(random.choice(local))
    if len(set(halos)) < 2:
        return None
    return sources, {src: halo for src, halo in zip(sources, halos)}


def _place_objects(
    height: int,
    width: int,
    count: int,
    patches: list[np.ndarray],
) -> tuple[list[tuple[int, int]], bool] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * count
    overlap_pair = None
    want_overlap = random.random() < 0.7 and count >= 2
    want_same_col = random.random() < 0.55 and count >= 2
    want_edge_touch = random.random() < 0.4

    if want_overlap:
        i = random.randrange(count - 1)
        p0 = patches[i]
        p1 = patches[i + 1]
        max_c = width - max(p0.shape[1], p1.shape[1])
        if max_c < 4:
            return None
        c0 = random.randint(4, max_c)
        max_r = height - (p0.shape[0] + p1.shape[0] + 1)
        if max_r < 0:
            return None
        r0 = random.randint(0, max_r)
        r1 = r0 + p0.shape[0] + 1
        if not _can_place(occupied, p0, r0, c0):
            return None
        placements[i] = (r0, c0)
        for rr, cc in np.argwhere(p0 == 1):
            occupied[r0 + rr, c0 + cc] = True

        c1 = c0 if want_same_col else min(width - p1.shape[1], c0 + random.choice([0, 1]))
        if not _can_place(occupied, p1, r1, c1):
            return None
        placements[i + 1] = (r1, c1)
        for rr, cc in np.argwhere(p1 == 1):
            occupied[r1 + rr, c1 + cc] = True
        overlap_pair = i

    for idx, patch in enumerate(patches):
        if placements[idx] != (-1, -1):
            continue
        for _ in range(160):
            if want_edge_touch and idx == count - 1 and random.random() < 0.7:
                r0 = random.choice([0, height - patch.shape[0]])
                c0 = random.randint(4, width - patch.shape[1])
            else:
                r0 = random.randint(0, height - patch.shape[0])
                c0 = random.randint(4, width - patch.shape[1])
            if overlap_pair is not None and c0 == placements[overlap_pair][1]:
                continue
            if _can_place(occupied, patch, r0, c0):
                placements[idx] = (r0, c0)
                for rr, cc in np.argwhere(patch == 1):
                    occupied[r0 + rr, c0 + cc] = True
                break
        else:
            return None

    for idx in range(count):
        assert placements[idx] != (-1, -1)

    has_overlap = False
    bboxes = [_bbox_from_patch(patch, r0, c0) for patch, (r0, c0) in zip(patches, placements)]
    for i in range(count):
        for j in range(i + 1, count):
            if _candidate_overlap(bboxes[i], bboxes[j]):
                has_overlap = True
    return placements, has_overlap


def _alt_outputs(grid: np.ndarray) -> dict[str, np.ndarray]:
    _bg, legend_map, legend_mask, objects = _scene_from_grid(grid)
    return {
        "correct": _render_halos(grid, legend_map, legend_mask, objects),
        "self": _render_halos(grid, legend_map, legend_mask, objects, neighbor_mode="self"),
        "rowcol": _render_halos(grid, legend_map, legend_mask, objects, order_mode="rowcol"),
        "reverse": _render_halos(grid, legend_map, legend_mask, objects, paint_mode="reverse"),
    }


def _valid_semantics(grid: np.ndarray, expect_overlap: bool) -> bool:
    try:
        bg, legend_map, legend_mask, objects = _scene_from_grid(grid)
    except AssertionError:
        return False
    if int(bg) != 0 or len(objects) != len(legend_map):
        return False
    if not _row_major_differs(objects):
        return False

    outs = _alt_outputs(grid)
    correct = outs["correct"]
    if not np.array_equal(correct, main(grid)):
        return False
    if np.array_equal(correct, outs["self"]):
        return False
    if np.array_equal(correct, outs["rowcol"]):
        return False

    overlap_exists = False
    ordered = _ordered_objects(objects, "colrow")
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if _candidate_overlap(ordered[i]["bbox"], ordered[j]["bbox"]):
                overlap_exists = True
    if expect_overlap:
        if not overlap_exists:
            return False
        if np.array_equal(correct, outs["reverse"]):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*_GRID_H_RANGE)
    count = random.randint(*_OBJECT_COUNT_RANGE)
    layout = _legend_layout(height, count)
    if layout is None:
        return None
    _legend_top, legend_rows = layout
    width = random.randint(*_GRID_W_RANGE)
    color_choice = _choose_colors(count)
    if color_choice is None:
        return None
    sources, legend_map = color_choice

    patches: list[np.ndarray] = []
    for _ in range(count):
        patch = _random_patch()
        if patch is None:
            return None
        patches.append(patch)

    placed = _place_objects(height, width, count, patches)
    if placed is None:
        return None
    placements, has_overlap = placed

    grid = np.zeros((height, width), dtype=int)
    for row, src in zip(legend_rows, sources):
        halo = legend_map[src]
        grid[row : row + 2, 0] = int(src)
        grid[row : row + 2, 1] = int(halo)

    perm = sources[:]
    random.shuffle(perm)
    for color, patch, (r0, c0) in zip(perm, patches, placements):
        _stamp(grid, patch, r0, c0, color)

    if not _valid_semantics(grid, expect_overlap=has_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {_MAX_ATTEMPTS} attempts")
