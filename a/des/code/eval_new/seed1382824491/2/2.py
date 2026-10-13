from __future__ import annotations

import random
from collections import deque
from functools import lru_cache

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
EDGES = ("top", "bottom", "left", "right")
ARC_MAX = 30
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bool_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    dirs = DIRS4 if connectivity == 4 else DIRS8

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for dr, dc in dirs:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(coords)
    return components


def _is_connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    cells = int(mask.sum())
    if cells == 0:
        return False
    return len(_bool_components(mask, connectivity=connectivity)) == 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _component_record(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    color = int(grid[coords[0]])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        assert int(grid[r, c]) == color
        mask[r - r0, c - c0] = True
    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
    }


def _extract_fragments(grid: np.ndarray) -> list[dict[str, object]]:
    interior = grid != BG
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    components = _bool_components(interior, connectivity=8)
    fragments = [_component_record(grid, coords) for coords in components]
    colors = [int(fragment["color"]) for fragment in fragments]
    assert len(colors) == len(set(colors)), "fragment colors must be unique"
    return fragments


def _edge_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(values)
    while i < n:
        if int(values[i]) == BG:
            i += 1
            continue
        j = i
        while j < n and int(values[j]) != BG:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _edge_marker_specs(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    specs: list[dict[str, object]] = []

    def add_runs(edge: str, values: np.ndarray, offset: int = 0) -> None:
        for start, end in _edge_runs(values):
            assert end - start == 2, "each edge marker group must be a single adjacent pair"
            if edge == "top":
                cells = [(0, start), (0, start + 1)]
            elif edge == "bottom":
                cells = [(h - 1, start), (h - 1, start + 1)]
            elif edge == "left":
                cells = [(start + offset, 0), (start + offset + 1, 0)]
            else:
                cells = [(start + offset, w - 1), (start + offset + 1, w - 1)]
            colors = tuple(int(grid[r, c]) for r, c in cells)
            specs.append({"edge": edge, "cells": cells, "colors": colors})

    add_runs("top", grid[0, :])
    add_runs("bottom", grid[-1, :])
    if h > 2:
        add_runs("left", grid[1:-1, 0], offset=1)
        add_runs("right", grid[1:-1, -1], offset=1)

    color_sets = [frozenset(spec["colors"]) for spec in specs]
    assert len(color_sets) == len(set(color_sets)), "marker color pairs must be unique"
    return specs


def _complements(a: dict[str, object], b: dict[str, object]) -> bool:
    ma = np.asarray(a["mask"], dtype=bool)
    mb = np.asarray(b["mask"], dtype=bool)
    if ma.shape != mb.shape:
        return False
    return not bool(np.any(ma & mb)) and bool(np.all(ma | mb))


def _unique_fragment_pairing(fragments: list[dict[str, object]]) -> list[tuple[int, int]]:
    n = len(fragments)
    assert n % 2 == 0 and n > 0
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            compat[i][j] = compat[j][i] = _complements(fragments[i], fragments[j])

    @lru_cache(maxsize=None)
    def search(remaining: tuple[int, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
        if not remaining:
            return ((),)
        i = remaining[0]
        solutions: list[tuple[tuple[int, int], ...]] = []
        rem_set = set(remaining)
        for j in remaining[1:]:
            if not compat[i][j]:
                continue
            nxt = tuple(k for k in remaining if k not in (i, j))
            for tail in search(nxt):
                solutions.append(((i, j),) + tail)
                if len(solutions) >= 2:
                    return tuple(solutions)
        return tuple(solutions)

    solutions = search(tuple(range(n)))
    assert len(solutions) == 1, "fragment complement pairing must be unique"
    return list(solutions[0])


def _assembly_for_pair(a: dict[str, object], b: dict[str, object]) -> np.ndarray:
    ma = np.asarray(a["mask"], dtype=bool)
    mb = np.asarray(b["mask"], dtype=bool)
    rect = np.full(ma.shape, BG, dtype=int)
    rect[ma] = int(a["color"])
    rect[mb] = int(b["color"])
    assert np.all(rect != BG)
    return rect


def _placement_top_left(edge: str, marker_cells: list[tuple[int, int]], rect_shape: tuple[int, int],
                        grid_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = rect_shape
    gh, gw = grid_shape
    (r1, c1), (r2, c2) = marker_cells
    if edge == "top":
        return 0, (c1 + c2 - w + 1) // 2
    if edge == "bottom":
        return gh - h, (c1 + c2 - w + 1) // 2
    if edge == "left":
        return (r1 + r2 - h + 1) // 2, 0
    if edge == "right":
        return (r1 + r2 - h + 1) // 2, gw - w
    raise ValueError(edge)


def _solved_pair_specs(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fragments = _extract_fragments(grid)
    markers = _edge_marker_specs(grid)
    marker_by_colors = {frozenset(spec["colors"]): spec for spec in markers}
    pairing = _unique_fragment_pairing(fragments)
    pair_specs: list[dict[str, object]] = []

    for i, j in pairing:
        a = fragments[i]
        b = fragments[j]
        key = frozenset((int(a["color"]), int(b["color"])))
        assert key in marker_by_colors, "each complementary pair must have a matching edge marker pair"
        marker = marker_by_colors[key]
        pair_specs.append({"a": a, "b": b, "marker": marker})

    assert len(pair_specs) == len(markers), "every marker pair must map to one fragment pair"
    return fragments, pair_specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble complementary fragment pairs and move each completed two-color rectangle to its edge anchor.

    Input structure:
    - The background is black.
    - Interior non-black cells form scattered monochrome 4-connected fragments; each fragment has a unique color.
    - Fragments come in complementary pairs: the two fragments in a pair have equal bbox size and their normalized occupancies exactly fill that bbox.
    - The grid edge contains isolated adjacent marker pairs; each marker color matches one fragment color, and each fragment pair has exactly one same-edge adjacent marker pair.

    Rule:
    - Find all interior fragments and pair them by complementary normalized shape, not by spatial proximity.
    - For each paired colors {A, B}, locate the edge marker pair with the same two colors.
    - Merge the two normalized fragment masks inside their shared bbox into a solid rectangle that keeps A cells colored A and B cells colored B.
    - Erase the original fragment locations, then place the merged rectangle flush against the indicated edge and centered on the midpoint of the two adjacent markers.

    Key insight:
    - The edge colors only tell you where a completed rectangle belongs; the actual pairing must first be inferred from bbox-normalized shape complementarity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= ARC_MAX
    assert int(g[0, 0]) == BG or BG in set(map(int, g.flat))

    fragments, pair_specs = _solved_pair_specs(g)
    out = g.copy()

    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = BG

    for pair in pair_specs:
        a = pair["a"]
        b = pair["b"]
        marker = pair["marker"]
        rect = _assembly_for_pair(a, b)
        top, left = _placement_top_left(
            str(marker["edge"]),
            list(marker["cells"]),
            rect.shape,
            out.shape,
        )
        h, w = rect.shape
        assert 0 <= top <= out.shape[0] - h and 0 <= left <= out.shape[1] - w

        occupied = out[top:top + h, left:left + w] != BG
        allowed = np.zeros((h, w), dtype=bool)
        for mr, mc in marker["cells"]:
            if top <= mr < top + h and left <= mc < left + w:
                allowed[mr - top, mc - left] = True
        assert not bool(np.any(occupied & (~allowed))), "destination rectangles must not collide"

        out[top:top + h, left:left + w] = rect

    return out


def _monotone_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    row_steps = [1 if er > r else -1] * abs(er - r)
    col_steps = [1 if ec > c else -1] * abs(ec - c)
    steps = [("r", step) for step in row_steps] + [("c", step) for step in col_steps]
    random.shuffle(steps)

    path = [(r, c)]
    for axis, step in steps:
        if axis == "r":
            r += step
        else:
            c += step
        path.append((r, c))
    assert path[-1] == end
    return path


def _valid_partition(mask: np.ndarray) -> bool:
    comp = ~mask
    if int(mask.sum()) == 0 or int(comp.sum()) == 0:
        return False
    return (
        _is_connected(mask, connectivity=8)
        and _is_connected(comp, connectivity=8)
        and _touches_all_sides(mask)
        and _touches_all_sides(comp)
    )


def _canonical_pair_signature(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int, bytes]:
    a = mask_a.astype(np.uint8).tobytes()
    b = mask_b.astype(np.uint8).tobytes()
    return mask_a.shape[0], mask_a.shape[1], min(a, b)


def _generate_complementary_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(250):
        row_start = random.randint(1, h - 2)
        col_start = random.randint(1, w - 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, col_start:] = True
        mask[row_start:, 0] = True

        r, c = 0, col_start
        mask[r, c] = True
        while (r, c) != (row_start, 0):
            options: list[tuple[int, int]] = []
            if r < row_start:
                options.append((r + 1, c))
            if c > 0:
                options.append((r, c - 1))
            if r < row_start and c > 0:
                options.append((r + 1, c - 1))
            r, c = random.choice(options)
            mask[r, c] = True

        orientation = random.choice(("nw", "ne", "sw", "se"))
        if "s" in orientation:
            mask = np.flipud(mask)
        if "e" in orientation:
            mask = np.fliplr(mask)
        mask = np.array(mask, dtype=bool, copy=True)

        if random.random() < 0.5:
            mask = ~mask
        comp = ~mask
        if min(int(mask.sum()), int(comp.sum())) < 4:
            continue
        if _valid_partition(mask):
            return mask, comp
    return None


def _rect_is_free(occupied: np.ndarray, top: int, left: int, height: int, width: int, margin: int = 0) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + height + margin)
    c1 = min(w, left + width + margin)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int, margin: int = 0) -> None:
    h, w = occupied.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + height + margin)
    c1 = min(w, left + width + margin)
    occupied[r0:r1, c0:c1] = True


def _destination_bbox(edge: str, start: int, h: int, w: int, grid_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    gh, gw = grid_shape
    if edge == "top":
        return 0, start, h - 1, start + w - 1
    if edge == "bottom":
        return gh - h, start, gh - 1, start + w - 1
    if edge == "left":
        return start, 0, start + h - 1, w - 1
    if edge == "right":
        return start, gw - w, start + h - 1, gw - 1
    raise ValueError(edge)


def _center_markers(edge: str, top: int, left: int, h: int, w: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, left + w // 2 - 1), (0, left + w // 2)]
    if edge == "bottom":
        return [(top + h - 1, left + w // 2 - 1), (top + h - 1, left + w // 2)]
    if edge == "left":
        return [(top + h // 2 - 1, 0), (top + h // 2, 0)]
    if edge == "right":
        return [(top + h // 2 - 1, left + w - 1), (top + h // 2, left + w - 1)]
    raise ValueError(edge)


def _sample_destination_placements(grid_shape: tuple[int, int], pair_defs: list[dict[str, object]]) -> bool:
    occupied = np.zeros(grid_shape, dtype=bool)
    indices = list(range(len(pair_defs)))
    random.shuffle(indices)

    for idx in indices:
        pair = pair_defs[idx]
        edge = str(pair["edge"])
        h = int(pair["height"])
        w = int(pair["width"])
        gh, gw = grid_shape

        if edge in ("top", "bottom"):
            starts = list(range(0, gw - w + 1))
        else:
            starts = list(range(0, gh - h + 1))
        random.shuffle(starts)

        placed = False
        for start in starts:
            r0, c0, r1, c1 = _destination_bbox(edge, start, h, w, grid_shape)
            if not _rect_is_free(occupied, r0, c0, h, w, margin=0):
                continue
            pair["dest_top"] = r0
            pair["dest_left"] = c0
            pair["marker_cells"] = _center_markers(edge, r0, c0, h, w)
            _mark_rect(occupied, r0, c0, h, w, margin=0)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_fragment_bboxes(grid_shape: tuple[int, int], pair_defs: list[dict[str, object]]) -> bool:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)

    for pair in pair_defs:
        dh = int(pair["height"])
        dw = int(pair["width"])
        _mark_rect(occupied, int(pair["dest_top"]), int(pair["dest_left"]), dh, dw, margin=1)
        for r, c in pair["marker_cells"]:
            occupied[r, c] = True

    fragments: list[tuple[int, int, int, int, str]] = []
    for idx, pair in enumerate(pair_defs):
        fragments.append((idx, int(pair["height"]), int(pair["width"]), 0, "a"))
        fragments.append((idx, int(pair["height"]), int(pair["width"]), 0, "b"))
    random.shuffle(fragments)

    for idx, fh, fw, _dummy, which in fragments:
        rows = list(range(2, h - fh - 1))
        cols = list(range(2, w - fw - 1))
        random.shuffle(rows)
        placed = False
        for top in rows:
            shuffled_cols = cols[:]
            random.shuffle(shuffled_cols)
            for left in shuffled_cols:
                if not _rect_is_free(occupied, top, left, fh, fw, margin=1):
                    continue
                pair_defs[idx][f"{which}_top"] = top
                pair_defs[idx][f"{which}_left"] = left
                _mark_rect(occupied, top, left, fh, fw, margin=1)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _render_input(pair_defs: list[dict[str, object]], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)

    for pair in pair_defs:
        cells = list(pair["marker_cells"])
        colors = [int(pair["color_a"]), int(pair["color_b"])]
        if random.random() < 0.5:
            colors.reverse()
        for (r, c), color in zip(cells, colors):
            grid[r, c] = color

        for which in ("a", "b"):
            top = int(pair[f"{which}_top"])
            left = int(pair[f"{which}_left"])
            mask = np.asarray(pair[f"mask_{which}"], dtype=bool)
            color = int(pair[f"color_{which}"])
            grid[top:top + mask.shape[0], left:left + mask.shape[1]][mask] = color

    return grid


def _render_expected_output(inp: np.ndarray, pair_defs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for pair in pair_defs:
        for which in ("a", "b"):
            top = int(pair[f"{which}_top"])
            left = int(pair[f"{which}_left"])
            mask = np.asarray(pair[f"mask_{which}"], dtype=bool)
            out[top:top + mask.shape[0], left:left + mask.shape[1]][mask] = BG

    for pair in pair_defs:
        rect = np.full((int(pair["height"]), int(pair["width"])), BG, dtype=int)
        mask_a = np.asarray(pair["mask_a"], dtype=bool)
        mask_b = np.asarray(pair["mask_b"], dtype=bool)
        rect[mask_a] = int(pair["color_a"])
        rect[mask_b] = int(pair["color_b"])
        top = int(pair["dest_top"])
        left = int(pair["dest_left"])
        out[top:top + rect.shape[0], left:left + rect.shape[1]] = rect
    return out


def _pair_dimensions(edge: str, grid_shape: tuple[int, int]) -> tuple[int, int] | None:
    gh, gw = grid_shape
    if edge in ("top", "bottom"):
        h_max = min(7, gh - 6)
        w_max = min(8, gw - 2)
        if h_max < 4 or w_max < 4:
            return None
        height = random.randint(4, h_max)
        width_choices = [val for val in range(4, w_max + 1) if val % 2 == 0]
        if not width_choices:
            return None
        width = random.choice(width_choices)
        return height, width

    h_max = min(8, gh - 2)
    w_max = min(7, gw - 6)
    if h_max < 4 or w_max < 4:
        return None
    height_choices = [val for val in range(4, h_max + 1) if val % 2 == 0]
    if not height_choices:
        return None
    height = random.choice(height_choices)
    width = random.randint(4, w_max)
    return height, width


def _gen_attempt() -> np.ndarray | None:
    grid_shape = (random.randint(22, ARC_MAX), random.randint(22, ARC_MAX))
    pair_count = random.randint(2, 4)
    edges = random.sample(EDGES, pair_count)
    colors = random.sample(range(1, 10), pair_count * 2)

    pair_defs: list[dict[str, object]] = []
    signatures: set[tuple[int, int, bytes]] = set()
    color_idx = 0
    for edge in edges:
        dims = _pair_dimensions(edge, grid_shape)
        if dims is None:
            return None
        height, width = dims
        masks = _generate_complementary_masks(height, width)
        if masks is None:
            return None
        mask_a, mask_b = masks
        sig = _canonical_pair_signature(mask_a, mask_b)
        if sig in signatures:
            return None
        signatures.add(sig)

        pair_defs.append({
            "edge": edge,
            "height": height,
            "width": width,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "color_a": colors[color_idx],
            "color_b": colors[color_idx + 1],
        })
        color_idx += 2

    if not _sample_destination_placements(grid_shape, pair_defs):
        return None
    if not _place_fragment_bboxes(grid_shape, pair_defs):
        return None

    inp = _render_input(pair_defs, grid_shape)
    expected = _render_expected_output(inp, pair_defs)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    solved_fragments, solved_pairs = _solved_pair_specs(inp)
    if len(solved_fragments) != pair_count * 2 or len(solved_pairs) != pair_count:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment input")
