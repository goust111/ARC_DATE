from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MARKER_COLORS = (BLUE, RED, GREEN)
STRUCT4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BLACK) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in STRUCT4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _components_mask(grid == color):
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _point_to_object_sqdist(point: tuple[int, int], coords: np.ndarray) -> int:
    rr, cc = point
    diffs = coords - np.asarray([[rr, cc]], dtype=int)
    return int((diffs * diffs).sum(axis=1).min())


def _analyze(grid: np.ndarray) -> tuple[list[dict], dict[int, tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    grey_objects = _find_monochrome_components(g, GREY)
    assert 4 <= len(grey_objects) <= 6, "expected 4-6 grey objects"

    markers = {}
    marker_cells = np.argwhere((g != BLACK) & (g != GREY))
    assert 2 <= len(marker_cells) <= 3, "expected 2-3 marker pixels"
    for r, c in marker_cells:
        color = int(g[r, c])
        assert color in MARKER_COLORS, "marker colors must be blue/red/green"
        assert color not in markers, "marker colors must be distinct"
        for dr, dc in STRUCT4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                assert g[nr, nc] == BLACK, "markers must be isolated from foreground by 4-neighborhood"
        markers[color] = (int(r), int(c))

    analyzed = []
    usage = defaultdict(int)
    for obj in grey_objects:
        scores = []
        for color, marker in markers.items():
            scores.append((_point_to_object_sqdist(marker, obj["coords"]), int(color)))
        scores.sort()
        assert len(scores) >= 2
        assert scores[0][0] != scores[1][0], "nearest marker must be unique"
        target = scores[0][1]
        usage[target] += 1
        analyzed.append(
            {
                "coords": obj["coords"],
                "bbox": obj["bbox"],
                "target_color": int(target),
                "left": int(obj["bbox"][1]),
                "top": int(obj["bbox"][0]),
                "height": int(obj["bbox"][2] - obj["bbox"][0] + 1),
                "width": int(obj["bbox"][3] - obj["bbox"][1] + 1),
            }
        )

    assert set(usage) == set(markers), "every marker color must be nearest to at least one grey object"
    return grey_objects, markers, analyzed


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor grey objects by their uniquely nearest marker, then group and bottom-stack them by original x-order.

    Input structure:
    A black background contains 4-6 pairwise separated 4-connected grey objects
    and 2-3 isolated single-pixel markers whose colors are chosen from blue,
    red, and green.

    Rule:
    For each grey object, compute the minimum squared Euclidean distance from any
    object pixel to each marker pixel; the unique closest marker color becomes
    that object's new color. Erase all marker pixels, group the recolored
    objects by that target color, sort groups by the minimum original bbox-left
    column in each group, sort objects inside a group by descending original
    bbox-left column with upper rows breaking ties, crop every object to its
    bbox, and stack each group top-to-bottom with one blank row between objects,
    bottom-aligned on an output grid of the same size. Place the color groups
    left-to-right with one blank column between groups, starting at column 0.

    Key insight:
    The visible grouping color is not given by the grey objects themselves; it is
    encoded by which isolated marker pixel is spatially closest to each object,
    while the packed layout order still depends on the objects' original
    positions rather than on color ids.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _grey_objects, markers, analyzed = _analyze(g)

    groups: list[dict] = []
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in analyzed:
        by_color[int(obj["target_color"])].append(obj)

    for color, objs in by_color.items():
        min_left = min(obj["left"] for obj in objs)
        group_width = max(obj["width"] for obj in objs)
        stack_height = sum(obj["height"] for obj in objs) + len(objs) - 1
        assert stack_height <= h, "stack height exceeds canvas"
        groups.append(
            {
                "color": int(color),
                "objects": objs,
                "min_left": int(min_left),
                "width": int(group_width),
                "stack_height": int(stack_height),
            }
        )

    group_lefts = [group["min_left"] for group in groups]
    assert len(set(group_lefts)) == len(group_lefts), "group order must be unique"
    groups.sort(key=lambda group: group["min_left"])
    total_width = sum(group["width"] for group in groups) + len(groups) - 1
    assert total_width <= w, "packed groups exceed canvas width"
    assert set(group["color"] for group in groups) == set(markers), "all marker colors must survive in grouped output"

    out = np.zeros_like(g)
    c0 = 0
    for group in groups:
        objs = sorted(group["objects"], key=lambda obj: (-obj["left"], obj["top"]))
        lefts = [obj["left"] for obj in objs]
        assert len(set(lefts)) == len(lefts), "within-group x-order must be unique"
        r = h - group["stack_height"]
        for idx, obj in enumerate(objs):
            patch = _crop_mask(obj["coords"], group["color"])
            _paste(out, patch, r, c0, transparent=BLACK)
            r += patch.shape[0]
            if idx != len(objs) - 1:
                r += 1
        c0 += group["width"] + 1

    return out


def _random_object_patch(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 5)
        target = rng.randint(4, min(12, h * w))
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            rr, cc = rng.choice(tuple(cells))
            frontier = []
            for dr, dc in STRUCT4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        coords = np.asarray(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        coords[:, 0] -= r0
        coords[:, 1] -= c0
        ch = r1 - r0 + 1
        cw = c1 - c0 + 1
        if ch < 2 or cw < 2:
            continue
        if coords.shape[0] == ch * cw and rng.random() < 0.8:
            continue
        patch = np.zeros((ch, cw), dtype=int)
        patch[coords[:, 0], coords[:, 1]] = GREY
        return patch
    raise RuntimeError("failed to build object patch")


def _sample_counts(rng: random.Random, n_groups: int, total_objects: int) -> list[int]:
    cuts = sorted(rng.sample(range(1, total_objects), n_groups - 1))
    parts = []
    prev = 0
    for cut in cuts + [total_objects]:
        parts.append(cut - prev)
        prev = cut
    rng.shuffle(parts)
    if max(parts) == 1:
        parts[0] += 1
        parts[-1] -= 1
    return parts


def _sample_markers(rng: random.Random, shape: tuple[int, int], colors: list[int]) -> dict[int, tuple[int, int]] | None:
    h, w = shape
    candidate_cols = list(range(2, w - 2))
    if len(candidate_cols) < len(colors):
        return None
    for _ in range(300):
        cols = sorted(rng.sample(candidate_cols, len(colors)))
        if any(cols[i + 1] - cols[i] < 5 for i in range(len(cols) - 1)):
            continue
        markers = {}
        for idx, color in enumerate(colors):
            row_choices = [1, 2, h - 3, h - 2]
            row = row_choices[(idx + rng.randrange(len(row_choices))) % len(row_choices)]
            markers[int(color)] = (int(row), int(cols[idx]))
        return markers
    return None


def _base_columns_from_markers(
    rng: random.Random, shape: tuple[int, int], groups: list[list[np.ndarray]], markers: dict[int, tuple[int, int]], colors: list[int]
) -> list[int] | None:
    w = shape[1]
    bases = []
    for group_index, patches in enumerate(groups):
        marker_col = markers[int(colors[group_index])][1]
        max_width = max(patch.shape[1] for patch in patches)
        low = max(0, marker_col - 6)
        high = min(marker_col - 1, w - max_width - 1)
        if high < low:
            return None
        bases.append(rng.randint(low, high))
    bases.sort()
    if len(set(bases)) != len(bases):
        return None
    if any(bases[i + 1] - bases[i] < 2 for i in range(len(bases) - 1)):
        return None
    return bases


def _fits(mask: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    patch_coords = np.argwhere(patch == GREY)
    patch_set = {(int(r0 + r), int(c0 + c)) for r, c in patch_coords}
    for rr, cc in patch_set:
        if mask[rr, cc]:
            return False
        for dr, dc in STRUCT4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in patch_set and mask[nr, nc]:
                return False
    return True


def _object_respects_markers(coords: np.ndarray, expected_color: int, markers: dict[int, tuple[int, int]]) -> bool:
    scores = sorted((_point_to_object_sqdist(marker, coords), color) for color, marker in markers.items())
    return len(scores) >= 2 and scores[0][0] != scores[1][0] and int(scores[0][1]) == int(expected_color)


def _markers_match_assignments(placed: list[dict], markers: dict[int, tuple[int, int]]) -> bool:
    usage = defaultdict(int)
    for obj in placed:
        scores = sorted((_point_to_object_sqdist(marker, obj["coords"]), color) for color, marker in markers.items())
        if len(scores) < 2 or scores[0][0] == scores[1][0]:
            return False
        if int(scores[0][1]) != int(obj["color"]):
            return False
        usage[int(scores[0][1])] += 1
    return set(usage) == set(markers)


def _place_objects_with_markers(
    rng: random.Random,
    shape: tuple[int, int],
    groups: list[list[np.ndarray]],
    colors: list[int],
    base_cols: list[int],
    markers: dict[int, tuple[int, int]],
) -> list[dict] | None:
    h, w = shape
    occ = np.zeros(shape, dtype=bool)
    placed: list[dict] = []
    protected = set(markers.values())

    for group_index, patches in enumerate(groups):
        base = base_cols[group_index]
        count = len(patches)
        offsets = list(range(min(7, w - base - 1)))
        valid_offsets = []
        for offset in offsets:
            if all(base + offset + patch.shape[1] <= w for patch in patches):
                valid_offsets.append(offset)
        if len(valid_offsets) < count:
            return None
        chosen_offsets = rng.sample(valid_offsets, count)
        if 0 not in chosen_offsets:
            chosen_offsets[0] = 0
        if len(set(chosen_offsets)) != len(chosen_offsets):
            return None

        order = list(range(count))
        rng.shuffle(order)
        for local_idx, offset in zip(order, chosen_offsets):
            patch = patches[local_idx]
            left = base + offset
            rows = list(range(0, h - patch.shape[0] + 1))
            rng.shuffle(rows)
            success = None
            for r0 in rows:
                coords = np.argwhere(patch == GREY) + np.asarray([[r0, left]], dtype=int)
                if not _fits(occ, patch, r0, left):
                    continue
                patch_cells = {tuple(map(int, rc)) for rc in coords}
                if any(cell in protected for cell in patch_cells):
                    continue
                if any(abs(cell[0] - mr) + abs(cell[1] - mc) == 1 for cell in patch_cells for mr, mc in protected):
                    continue
                if not _object_respects_markers(coords, colors[group_index], markers):
                    continue
                success = coords
                break
            if success is None:
                return None
            occ[success[:, 0], success[:, 1]] = True
            placed.append(
                {
                    "group_index": group_index,
                    "color": int(colors[group_index]),
                    "patch": patch,
                    "local_idx": int(local_idx),
                    "left": int(left),
                    "top": int(success[:, 0].min()),
                    "coords": success,
                    "bbox": _bbox(success),
                    "height": int(patch.shape[0]),
                    "width": int(patch.shape[1]),
                }
            )
    return placed if _markers_match_assignments(placed, markers) else None


def _render_input(shape: tuple[int, int], placed: list[dict], markers: dict[int, tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in placed:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = GREY
    for color, (r, c) in markers.items():
        grid[r, c] = int(color)
    return grid


def _stack_variant_wrong_inner_order(grid: np.ndarray) -> np.ndarray:
    _grey, markers, analyzed = _analyze(grid)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in analyzed:
        by_color[int(obj["target_color"])].append(obj)
    groups = []
    for color, objs in by_color.items():
        groups.append(
            {
                "color": int(color),
                "objects": objs,
                "min_left": min(obj["left"] for obj in objs),
                "width": max(obj["width"] for obj in objs),
                "stack_height": sum(obj["height"] for obj in objs) + len(objs) - 1,
            }
        )
    groups.sort(key=lambda group: group["min_left"])
    out = np.zeros_like(grid)
    c0 = 0
    h = grid.shape[0]
    for group in groups:
        objs = sorted(group["objects"], key=lambda obj: (obj["top"], obj["left"]))
        r = h - group["stack_height"]
        for idx, obj in enumerate(objs):
            patch = _crop_mask(obj["coords"], group["color"])
            _paste(out, patch, r, c0)
            r += patch.shape[0]
            if idx != len(objs) - 1:
                r += 1
        c0 += group["width"] + 1
    assert set(markers) == set(by_color)
    return out


def _stack_variant_wrong_group_order(grid: np.ndarray) -> np.ndarray:
    _grey, markers, analyzed = _analyze(grid)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in analyzed:
        by_color[int(obj["target_color"])].append(obj)
    groups = []
    for color, objs in by_color.items():
        groups.append(
            {
                "color": int(color),
                "objects": objs,
                "width": max(obj["width"] for obj in objs),
                "stack_height": sum(obj["height"] for obj in objs) + len(objs) - 1,
            }
        )
    groups.sort(key=lambda group: group["color"])
    out = np.zeros_like(grid)
    c0 = 0
    h = grid.shape[0]
    for group in groups:
        objs = sorted(group["objects"], key=lambda obj: (-obj["left"], obj["top"]))
        r = h - group["stack_height"]
        for idx, obj in enumerate(objs):
            patch = _crop_mask(obj["coords"], group["color"])
            _paste(out, patch, r, c0)
            r += patch.shape[0]
            if idx != len(objs) - 1:
                r += 1
        c0 += group["width"] + 1
    assert set(markers) == set(by_color)
    return out


def _validate_generated(inp: np.ndarray) -> None:
    out = main(inp)
    _grey, markers, analyzed = _analyze(inp)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in analyzed:
        by_color[int(obj["target_color"])].append(obj)

    group_order = [color for color, _objs in sorted(by_color.items(), key=lambda item: min(obj["left"] for obj in item[1]))]
    assert group_order != sorted(group_order), "group order must depend on spatial position, not color id"

    group_mins = [min(obj["left"] for obj in objs) for objs in by_color.values()]
    assert len(group_mins) == len(set(group_mins))
    assert set(group_order) == set(markers)

    overlap_like = False
    order_conflict = False
    for objs in by_color.values():
        lefts = [obj["left"] for obj in objs]
        assert len(lefts) == len(set(lefts)), "within-group x-order must be unique"
        if len(objs) >= 2:
            by_left = [id(obj) for obj in sorted(objs, key=lambda obj: (-obj["left"], obj["top"]))]
            by_row = [id(obj) for obj in sorted(objs, key=lambda obj: (obj["top"], obj["left"]))]
            if by_left != by_row:
                order_conflict = True
    ordered_groups = sorted(by_color.values(), key=lambda objs: min(obj["left"] for obj in objs))
    for i in range(len(ordered_groups) - 1):
        left_group = ordered_groups[i]
        right_group = ordered_groups[i + 1]
        if max(obj["left"] for obj in left_group) > min(obj["left"] for obj in right_group):
            overlap_like = True
    assert overlap_like, "group x-ranges should not separate cleanly into trivial clusters"
    assert order_conflict, "at least one group must need the specified within-group ordering rule"

    wrong_inner = _stack_variant_wrong_inner_order(inp)
    wrong_group = _stack_variant_wrong_group_order(inp)
    assert not np.array_equal(out, wrong_inner), "inner ordering step must matter"
    assert not np.array_equal(out, wrong_group), "group ordering step must matter"
    assert np.count_nonzero(out == GREY) == 0


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 25)
    w = rng.randint(18, 25)
    n_groups = rng.randint(2, 3)
    total_objects = rng.randint(max(4, n_groups), 7)
    counts = _sample_counts(rng, n_groups, total_objects)
    colors = list(rng.sample(MARKER_COLORS, n_groups))

    groups: list[list[np.ndarray]] = []
    for count in counts:
        patches = [_random_object_patch(rng) for _ in range(count)]
        groups.append(patches)

    markers = _sample_markers(rng, (h, w), colors)
    if markers is None:
        return None

    bases = _base_columns_from_markers(rng, (h, w), groups, markers, colors)
    if bases is None:
        return None

    placed = _place_objects_with_markers(rng, (h, w), groups, colors, bases, markers)
    if placed is None:
        return None

    inp = _render_input((h, w), placed, markers)
    try:
        _validate_generated(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
