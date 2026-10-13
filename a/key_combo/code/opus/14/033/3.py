from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
CYAN = 8
PATTERN_COLORS = (1, 2, 4, 6, 7, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SMALL_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 1), (1, 0), (1, 1)),
)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _share_row_or_col_span(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _spans_overlap(a[0], a[2], b[0], b[2]) or _spans_overlap(a[1], a[3], b[1], b[3])


def _merge_fragment_groups(components: list[np.ndarray]) -> list[dict]:
    groups = [{"coords": comp.copy(), "bbox": _bbox_coords(comp)} for comp in components]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if _share_row_or_col_span(groups[i]["bbox"], groups[j]["bbox"]):
                    merged = np.vstack([groups[i]["coords"], groups[j]["coords"]])
                    groups[i] = {"coords": merged, "bbox": _bbox_coords(merged)}
                    del groups[j]
                    changed = True
                    break
    return groups


def _green_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components_of_color(grid, GREEN)
    assert len(comps) == 1, "expected one green occlusion rectangle"
    box = _bbox_coords(comps[0])
    r0, c0, r1, c1 = box
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN), "green object must be solid"
    return box


def _recover_patch_with_axis(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    axis: str,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    cr = H // 2
    cc = W // 2
    r0, c0, r1, c1 = box
    patch = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rr, r in enumerate(range(r0, r1 + 1)):
        for cc_idx, c in enumerate(range(c0, c1 + 1)):
            if axis == "horizontal":
                sr, sc = 2 * cr - r, c
            else:
                sr, sc = r, 2 * cc - c
            assert 0 <= sr < H and 0 <= sc < W
            assert g[sr, sc] != GREEN, "mirror source must be visible"
            patch[rr, cc_idx] = int(g[sr, sc])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the green-occluded central pattern using the symmetry axis encoded by merged cyan fragments.

    Input structure:
    - A black background contains a centered multicolor pattern and one solid green rectangle hiding part of that pattern.
    - Several small cyan fragments lie in the outer black border and are not part of the pattern.
    - The cyan fragments are split into multiple 4-connected components whose bounding boxes can be transitively merged by overlapping row spans or column spans.

    Rule:
    - Find all cyan 4-connected components and merge them transitively whenever two component bounding boxes share any row span or any column span.
    - Read the merged group's overall bounding box: if it is wider than tall, use the grid's horizontal center row as the mirror axis; if it is taller than wide, use the grid's vertical center column as the mirror axis.
    - Replace every green cell by the color at its reflected position across that chosen axis, then erase all cyan fragment cells to black.

    Key insight:
    - The cyan fragments are an instruction code, not damaged pattern pixels: only after span-based fragment merging does their combined aspect ratio reveal which symmetry axis repairs the hidden region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cyan_components = _components_of_color(g, CYAN)
    assert cyan_components, "expected cyan signal fragments"
    merged = _merge_fragment_groups(cyan_components)
    assert len(merged) == 1, "expected a single merged cyan instruction group"
    fr0, fc0, fr1, fc1 = merged[0]["bbox"]
    frag_h = fr1 - fr0 + 1
    frag_w = fc1 - fc0 + 1
    assert frag_w != frag_h, "merged cyan bbox must encode a non-square aspect ratio"
    axis = "horizontal" if frag_w > frag_h else "vertical"

    r0, c0, r1, c1 = _green_bbox(g)
    patch = _recover_patch_with_axis(g, (r0, c0, r1, c1), axis)

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = patch
    out[out == CYAN] = BLACK
    return out


def _manhattan_path(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = goal
    path = [(r, c)]
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        path.append((r, c))
    return path


def _grow_connected_region(
    allowed: np.ndarray,
    fixed_cells: set[tuple[int, int]],
    target_count: int,
) -> set[tuple[int, int]] | None:
    cells = set(fixed_cells)
    if len(cells) > target_count:
        return None

    while len(cells) < target_count:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nr, nc]:
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
        if not frontier:
            return None
        weights = []
        for r, c in frontier:
            center_bias = abs(r - allowed.shape[0] / 2.0) + abs(c - allowed.shape[1] / 2.0)
            weights.append(1.0 / (1.0 + center_bias))
        pick = random.choices(list(frontier), weights=weights, k=1)[0]
        cells.add(pick)
    return cells


def _mirror_cells(cells: set[tuple[int, int]], axis: str, H: int, W: int) -> set[tuple[int, int]]:
    cr = H // 2
    cc = W // 2
    out = set()
    for r, c in cells:
        out.add((r, c))
        if axis == "horizontal":
            out.add((2 * cr - r, c))
        else:
            out.add((r, 2 * cc - c))
    return out


def _assign_pattern_colors(mask: np.ndarray, axis: str) -> np.ndarray:
    H, W = mask.shape
    cr = H // 2
    cc = W // 2
    palette = random.sample(PATTERN_COLORS, k=random.choice((2, 3)))
    reps = []
    if axis == "horizontal":
        for r in range(H):
            for c in range(W):
                if mask[r, c] and r <= cr:
                    reps.append((r, c))
    else:
        for r in range(H):
            for c in range(W):
                if mask[r, c] and c <= cc:
                    reps.append((r, c))
    assert len(reps) >= len(palette)

    seed_points = random.sample(reps, k=len(palette))
    color_of_rep: dict[tuple[int, int], int] = {}
    for seed, color in zip(seed_points, palette):
        color_of_rep[seed] = int(color)

    grid = np.zeros((H, W), dtype=int)
    for rep in reps:
        if rep not in color_of_rep:
            distances = [abs(rep[0] - s[0]) + abs(rep[1] - s[1]) for s in seed_points]
            best = int(np.argmin(distances))
            color_of_rep[rep] = int(palette[best])
        r, c = rep
        color = color_of_rep[rep]
        grid[r, c] = color
        if axis == "horizontal":
            grid[2 * cr - r, c] = color
        else:
            grid[r, 2 * cc - c] = color
    return grid


def _make_support_and_occ(
    H: int,
    W: int,
    axis: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    cr = H // 2
    cc = W // 2

    occ_h = random.randint(3, 5)
    occ_w = random.randint(3, 5)

    if axis == "horizontal":
        if cr - occ_h - 1 < 1 or W - occ_w - 3 < 2:
            return None
        r0 = random.randint(1, cr - occ_h - 1)
        c0 = random.randint(2, W - occ_w - 3)
        r1 = r0 + occ_h - 1
        c1 = c0 + occ_w - 1
        mirror_r0 = 2 * cr - r1

        bridge_w = random.randint(3, 5)
        bridge_c0 = max(2, min(W - bridge_w - 2, c0 + occ_w // 2 - bridge_w // 2))
        center_row_cells = {(cr, c) for c in range(bridge_c0, bridge_c0 + bridge_w)}

        allowed = np.zeros((H, W), dtype=bool)
        allowed[1:cr, 2 : W - 2] = True
        top_fixed = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        path_start = (r1, c0 + occ_w // 2)
        path_goal = (cr - 1, random.randint(bridge_c0, bridge_c0 + bridge_w - 1))
        top_fixed.update(_manhattan_path(path_start, path_goal))

        feasible = [n for n in range(max(len(top_fixed), 18), 29) if 40 <= 2 * n + bridge_w <= 60]
        if not feasible:
            return None
        top_target = random.choice(feasible)
        top_cells = _grow_connected_region(allowed, top_fixed, top_target)
        if top_cells is None:
            return None
        full_cells = _mirror_cells(top_cells, axis, H, W) | center_row_cells
    else:
        if cc - occ_w - 1 < 1 or H - occ_h - 3 < 2:
            return None
        r0 = random.randint(2, H - occ_h - 3)
        c0 = random.randint(1, cc - occ_w - 1)
        r1 = r0 + occ_h - 1
        c1 = c0 + occ_w - 1
        mirror_c0 = 2 * cc - c1

        bridge_h = random.randint(3, 5)
        bridge_r0 = max(2, min(H - bridge_h - 2, r0 + occ_h // 2 - bridge_h // 2))
        center_col_cells = {(r, cc) for r in range(bridge_r0, bridge_r0 + bridge_h)}

        allowed = np.zeros((H, W), dtype=bool)
        allowed[2 : H - 2, 1:cc] = True
        left_fixed = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        path_start = (r0 + occ_h // 2, c1)
        path_goal = (random.randint(bridge_r0, bridge_r0 + bridge_h - 1), cc - 1)
        left_fixed.update(_manhattan_path(path_start, path_goal))

        feasible = [n for n in range(max(len(left_fixed), 18), 29) if 40 <= 2 * n + bridge_h <= 60]
        if not feasible:
            return None
        left_target = random.choice(feasible)
        left_cells = _grow_connected_region(allowed, left_fixed, left_target)
        if left_cells is None:
            return None
        full_cells = _mirror_cells(left_cells, axis, H, W) | center_col_cells

    mask = np.zeros((H, W), dtype=bool)
    for r, c in full_cells:
        mask[r, c] = True

    if axis == "horizontal":
        assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
        assert np.all(mask[mirror_r0 : mirror_r0 + occ_h, c0 : c1 + 1])
        if np.array_equal(mask, np.fliplr(mask)):
            return None
    else:
        assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
        assert np.all(mask[r0 : r1 + 1, mirror_c0 : mirror_c0 + occ_w])
        if np.array_equal(mask, np.flipud(mask)):
            return None

    return mask, (r0, c0, r1, c1)


def _build_fragments(H: int, W: int, axis: str) -> np.ndarray | None:
    grid = np.zeros((H, W), dtype=int)
    count = random.randint(3, 5)
    mode = random.choice(("direct", "chain"))

    if axis == "horizontal":
        use_bottom = random.random() < 0.5
        base_r = H - 3 if use_bottom else 0
        current_c = 0
        for i in range(count):
            shape = random.choice([s for s in SMALL_SHAPES if max(r for r, _ in s) <= 2 and max(c for _, c in s) <= 2])
            shape_h = 1 + max(r for r, _ in shape)
            shape_w = 1 + max(c for _, c in shape)
            if mode == "direct":
                row_offset = random.randint(0, 3 - shape_h)
            else:
                band_start = 0 if i % 2 == 0 else 1
                if shape_h > 2:
                    row_offset = 0
                else:
                    row_offset = band_start
            if current_c + shape_w > W:
                return None
            for dr, dc in shape:
                grid[base_r + row_offset + dr, current_c + dc] = CYAN
            current_c += shape_w + random.randint(1, 2)
        if current_c <= 5:
            return None
    else:
        use_right = random.random() < 0.5
        base_c = W - 3 if use_right else 0
        current_r = 0
        for i in range(count):
            shape = random.choice([s for s in SMALL_SHAPES if max(r for r, _ in s) <= 2 and max(c for _, c in s) <= 2])
            shape_h = 1 + max(r for r, _ in shape)
            shape_w = 1 + max(c for _, c in shape)
            if mode == "direct":
                col_offset = random.randint(0, 3 - shape_w)
            else:
                band_start = 0 if i % 2 == 0 else 1
                if shape_w > 2:
                    col_offset = 0
                else:
                    col_offset = band_start
            if current_r + shape_h > H:
                return None
            for dr, dc in shape:
                grid[current_r + dr, base_c + col_offset + dc] = CYAN
            current_r += shape_h + random.randint(1, 2)
        if current_r <= 5:
            return None

    comps = _components_of_color(grid, CYAN)
    if len(comps) != count:
        return None
    merged = _merge_fragment_groups(comps)
    if len(merged) != 1:
        return None
    r0, c0, r1, c1 = merged[0]["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if axis == "horizontal" and not (w > h):
        return None
    if axis == "vertical" and not (h > w):
        return None
    return grid


def _wrong_axis_differs(pattern: np.ndarray, box: tuple[int, int, int, int], axis: str) -> bool:
    correct = _recover_patch_with_axis(pattern, box, axis)
    wrong_axis = "vertical" if axis == "horizontal" else "horizontal"
    wrong = _recover_patch_with_axis(pattern, box, wrong_axis)
    return not np.array_equal(correct, wrong)


def _validate_instance(inp: np.ndarray, expected: np.ndarray, axis: str) -> None:
    out = main(inp)
    assert np.array_equal(out, expected)

    green_box = _green_bbox(inp)
    cyan_components = _components_of_color(inp, CYAN)
    assert 3 <= len(cyan_components) <= 5
    merged = _merge_fragment_groups(cyan_components)
    assert len(merged) == 1
    r0, c0, r1, c1 = merged[0]["bbox"]
    frag_h = r1 - r0 + 1
    frag_w = c1 - c0 + 1
    decoded = "horizontal" if frag_w > frag_h else "vertical"
    assert decoded == axis

    non_special = set(int(v) for v in inp.flatten()) - {BLACK, GREEN, CYAN}
    assert 2 <= len(non_special) <= 3
    assert _wrong_axis_differs(expected, green_box, axis)
    assert np.any(inp == GREEN)
    assert np.any(inp == CYAN)
    assert not np.array_equal(inp, expected)


def _attempt_generate() -> np.ndarray | None:
    axis = random.choice(("horizontal", "vertical"))
    H = random.choice((13, 15, 17))
    W = random.choice((13, 15, 17))

    support_result = _make_support_and_occ(H, W, axis)
    if support_result is None:
        return None
    support_mask, green_box = support_result

    pattern = _assign_pattern_colors(support_mask, axis)
    if not _wrong_axis_differs(pattern, green_box, axis):
        return None

    fragments = _build_fragments(H, W, axis)
    if fragments is None:
        return None
    if np.any((pattern != BLACK) & (fragments == CYAN)):
        return None

    inp = pattern.copy()
    r0, c0, r1, c1 = green_box
    inp[r0 : r1 + 1, c0 : c1 + 1] = GREEN
    inp[fragments == CYAN] = CYAN

    expected = pattern.copy()
    _validate_instance(inp, expected, axis)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
