from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, paste, random_free_location


BG = 0
FILL = 5
MARKER_COLORS = (1, 2, 3)
STAMP_COLORS = (4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500

Coord = tuple[int, int]


def _component_patch_data(grid: np.ndarray, comp: dict) -> tuple[np.ndarray, np.ndarray, set[int]]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, comp["bbox"])
    mask = np.zeros(patch.shape, dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    colors = {int(grid[int(r), int(c)]) for r, c in coords}
    return patch, mask, colors


def _normalize(coords: set[Coord]) -> frozenset[Coord]:
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _unique_marker_offset(src_marker: np.ndarray, dst_marker: np.ndarray) -> tuple[int, int]:
    src_coords = {(int(r), int(c)) for r, c in np.argwhere(src_marker)}
    dst_coords = {(int(r), int(c)) for r, c in np.argwhere(dst_marker)}
    assert src_coords and dst_coords, "marker pixels must exist in source and destination"
    assert len(src_coords) == len(dst_coords), "marker groups must have the same size"
    assert _normalize(src_coords) == _normalize(dst_coords), "marker groups must match by translation"

    src_r0 = min(r for r, _c in src_coords)
    src_c0 = min(c for _r, c in src_coords)
    dst_r0 = min(r for r, _c in dst_coords)
    dst_c0 = min(c for _r, c in dst_coords)
    dr, dc = dst_r0 - src_r0, dst_c0 - src_c0
    shifted = {(r + dr, c + dc) for r, c in src_coords}
    assert shifted == dst_coords, "marker translation must be unique"
    return int(dr), int(dc)


def _visible_counts(grid: np.ndarray, *, bg: int = BG) -> dict[int, int]:
    counts = Counter(int(v) for v in grid.flat if int(v) != int(bg))
    return dict(counts)


def _render_histogram(composite: np.ndarray) -> np.ndarray:
    counts = _visible_counts(composite, bg=BG)
    assert counts, "composite must contain foreground colors"
    values = list(counts.values())
    assert len(values) == len(set(values)), "visible color counts must be unique for a deterministic ranking"

    h, w = composite.shape
    hist_h = max(values)
    out = np.full((h + 1 + hist_h, max(w, len(counts))), BG, dtype=int)
    out[:h, :w] = composite

    ranking = sorted(counts.items(), key=lambda item: -item[1])
    for col, (color, count) in enumerate(ranking):
        out[h + 1 + hist_h - count : h + 1 + hist_h, col] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the marked base, overlay the external sprites by their shared marker groups, then render visible-color frequencies as a histogram.

    Input structure:
    - A black background contains one unique largest 8-connected multicolor base object whose dominant color is grey.
    - The base object is a filled rectangle-like region with one or more isolated marker pixels/groups in blue, red, or green.
    - Each smaller 8-connected sprite contains exactly one marker color shared with the base plus one stamp color that should be painted onto the base after alignment.

    Rule:
    - Identify the unique largest 8-connected foreground component as the base and crop the output to its bounding box.
    - For each smaller sprite, find its shared marker color and translate the sprite so its whole marker group coincides with the same-colored marker group in the cropped base.
    - Overlay sprites in decreasing area order, painting only their stamp-color pixels so original base markers remain visible.
    - Count the visible pixels of every non-background color in the final composite.
    - Append a one-row black separator and, below it, a compact bottom-aligned histogram whose columns are ordered from highest visible count to lowest.

    Key insight:
    - The histogram is driven by visible pixels after aligned overlay and occlusion, so marker matching, crop choice, and large-first small-last layering all change the final bar heights.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) >= 3, "expected one base object and at least two sprites"

    areas = [int(comp["area"]) for comp in comps]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "largest base object must be unique"
    dst_index = areas.index(max_area)
    dst_comp = comps[dst_index]
    dst_patch, dst_mask, _dst_colors = _component_patch_data(g, dst_comp)

    dst_coords = dst_comp["coords"]
    dst_color_counts = Counter(int(g[int(r), int(c)]) for r, c in dst_coords)
    fill_area = max(dst_color_counts.values())
    fill_colors = [color for color, area in dst_color_counts.items() if area == fill_area]
    assert len(fill_colors) == 1, "base fill color must be unique"
    fill_color = fill_colors[0]

    marker_masks: dict[int, np.ndarray] = {}
    for color in dst_color_counts:
        if color == fill_color:
            continue
        marker_mask = (dst_patch == int(color)) & dst_mask
        assert marker_mask.any(), "base marker color must be present in the crop"
        marker_masks[int(color)] = marker_mask
    assert marker_masks, "base must contain marker colors"

    sources = [comp for idx, comp in enumerate(comps) if idx != dst_index]
    source_areas = [int(comp["area"]) for comp in sources]
    assert len(set(source_areas)) == len(source_areas), "sprite areas must be distinct for layering"
    sources.sort(key=lambda comp: -int(comp["area"]))

    out = dst_patch.copy()
    base_marker_mask = np.zeros(out.shape, dtype=bool)
    for marker_mask in marker_masks.values():
        base_marker_mask |= marker_mask

    out_h, out_w = out.shape
    marker_colors = set(marker_masks)
    for src_comp in sources:
        src_patch, src_mask, src_colors = _component_patch_data(g, src_comp)
        common_marker_colors = src_colors & marker_colors
        assert len(common_marker_colors) == 1, "each sprite must share exactly one marker color with the base"
        marker_color = next(iter(common_marker_colors))

        stamp_colors = src_colors - {marker_color}
        assert len(stamp_colors) == 1, "each sprite must contain one stamp color plus its marker color"
        stamp_color = next(iter(stamp_colors))

        src_marker = (src_patch == int(marker_color)) & src_mask
        dst_marker = marker_masks[int(marker_color)]
        top, left = _unique_marker_offset(src_marker, dst_marker)

        src_h, src_w = src_patch.shape
        assert 0 <= top and 0 <= left, "aligned sprite must fit within the base crop"
        assert top + src_h <= out_h and left + src_w <= out_w, "aligned sprite must stay inside the base crop"

        stamp_coords = np.argwhere((src_patch == int(stamp_color)) & src_mask)
        for r, c in stamp_coords:
            rr = top + int(r)
            cc = left + int(c)
            assert 0 <= rr < out_h and 0 <= cc < out_w
            if not base_marker_mask[rr, cc]:
                out[rr, cc] = int(stamp_color)

    return _render_histogram(out)


def _neighbors8(shape: tuple[int, int], cell: Coord) -> list[Coord]:
    h, w = shape
    r, c = cell
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((int(rr), int(cc)))
    return out


def _connected(cells: set[Coord], shape: tuple[int, int]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    queue = deque(seen)
    while queue:
        cell = queue.popleft()
        for nxt in _neighbors8(shape, cell):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def _bfs_path(
    shape: tuple[int, int],
    start: Coord,
    goals: set[Coord],
    blocked: set[Coord],
) -> list[Coord] | None:
    queue = deque([start])
    prev: dict[Coord, Coord | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell in goals:
            path = [cell]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            path.reverse()
            return path
        for nxt in _neighbors8(shape, cell):
            if nxt in prev:
                continue
            if nxt in blocked and nxt not in goals:
                continue
            prev[nxt] = cell
            queue.append(nxt)
    return None


def _sample_marker_groups(h: int, w: int, counts: list[int]) -> list[set[Coord]] | None:
    candidates = [(r, c) for r in range(1, h - 1, 2) for c in range(1, w - 1, 2)]
    if sum(counts) > len(candidates):
        return None

    for _ in range(80):
        pool = candidates[:]
        random.shuffle(pool)
        groups: list[set[Coord]] = []
        ok = True
        idx = 0
        for count in counts:
            group = set(pool[idx : idx + count])
            idx += count
            if len(group) != count:
                ok = False
                break
            rows = {r for r, _c in group}
            cols = {c for _r, c in group}
            if count >= 3 and (len(rows) == 1 or len(cols) == 1):
                ok = False
                break
            groups.append(group)
        if ok:
            return groups
    return None


def _choose_overlap_core(h: int, w: int, marker_cells: set[Coord]) -> set[Coord] | None:
    shapes = [
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0)},
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (0, 2)},
    ]
    anchors = [
        (r, c)
        for r in range(max(1, h // 2 - 2), min(h - 1, h // 2 + 1))
        for c in range(max(1, w // 2 - 2), min(w - 1, w // 2 + 1))
    ]
    random.shuffle(anchors)
    random.shuffle(shapes)
    for ar, ac in anchors:
        for shape in shapes:
            cells = {(ar + dr, ac + dc) for dr, dc in shape}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                continue
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
                continue
            if cells & marker_cells:
                continue
            return cells
    return None


def _build_initial_stamp(
    shape: tuple[int, int],
    marker_cells: set[Coord],
    all_markers: set[Coord],
    core: set[Coord],
) -> set[Coord] | None:
    stamp = set(core)
    for marker in marker_cells:
        blocked = set(all_markers) - {marker}
        path = _bfs_path(shape, marker, core, blocked)
        if path is None:
            return None
        for cell in path:
            if cell not in all_markers:
                stamp.add(cell)
    component = set(marker_cells) | stamp
    if not _connected(component, shape):
        return None
    return stamp


def _expand_stamp(
    shape: tuple[int, int],
    marker_cells: set[Coord],
    all_markers: set[Coord],
    stamp: set[Coord],
    target_size: int,
) -> set[Coord] | None:
    stamp = set(stamp)
    occupied_markers = set(all_markers)
    while len(stamp) < target_size:
        frontier: set[Coord] = set()
        support = stamp | marker_cells
        for cell in support:
            for nxt in _neighbors8(shape, cell):
                if nxt in occupied_markers or nxt in stamp:
                    continue
                frontier.add(nxt)
        if not frontier:
            return None
        cell = random.choice(sorted(frontier))
        stamp.add(cell)
    component = set(marker_cells) | stamp
    if not _connected(component, shape):
        return None
    return stamp


def _build_base_crop(h: int, w: int, marker_info: list[tuple[int, set[Coord]]]) -> np.ndarray:
    base = np.full((h, w), FILL, dtype=int)
    for color, cells in marker_info:
        for r, c in cells:
            base[r, c] = int(color)
    return base


def _compose_from_spec(base: np.ndarray, sprites: list[dict], *, descending: bool) -> np.ndarray:
    out = base.copy()
    base_marker_mask = (base != FILL)
    order = sorted(sprites, key=lambda spec: int(spec["area"]), reverse=descending)
    for spec in order:
        for r, c in spec["stamp_cells"]:
            if not base_marker_mask[r, c]:
                out[r, c] = int(spec["stamp_color"])
    return out


def _expected_output(base: np.ndarray, sprites: list[dict]) -> np.ndarray:
    composite = _compose_from_spec(base, sprites, descending=True)
    return _render_histogram(composite)


def _pairwise_overlap_ok(sprites: list[dict]) -> bool:
    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            if sprites[i]["stamp_cells"] & sprites[j]["stamp_cells"]:
                return True
    return False


def _semantic_checks(base: np.ndarray, sprites: list[dict]) -> bool:
    if not _pairwise_overlap_ok(sprites):
        return False

    composite = _compose_from_spec(base, sprites, descending=True)
    reverse = _compose_from_spec(base, sprites, descending=False)
    counts = _visible_counts(composite, bg=BG)
    reverse_counts = _visible_counts(reverse, bg=BG)

    expected_color_count = 1 + 2 * len(sprites)
    if len(counts) != expected_color_count:
        return False
    if len(set(counts.values())) != len(counts):
        return False
    if counts == reverse_counts:
        return False

    raw_stamp_counts = {int(spec["stamp_color"]): len(spec["stamp_cells"]) for spec in sprites}
    if not any(counts[color] < raw_stamp_counts[color] for color in raw_stamp_counts):
        return False

    out = _render_histogram(composite)
    return max(out.shape) <= 30


def _sprite_patch(spec: dict) -> np.ndarray:
    cells = set(spec["marker_cells"]) | set(spec["stamp_cells"])
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in spec["stamp_cells"]:
        patch[r - r0, c - c0] = int(spec["stamp_color"])
    for r, c in spec["marker_cells"]:
        patch[r - r0, c - c0] = int(spec["marker_color"])
    return patch


def _make_grid(base: np.ndarray, sprites: list[dict]) -> np.ndarray | None:
    patches = [_sprite_patch(spec) for spec in sprites]
    max_h = max([base.shape[0]] + [patch.shape[0] for patch in patches])
    max_w = max([base.shape[1]] + [patch.shape[1] for patch in patches])
    min_h = max(18, max_h + 4)
    min_w = max(18, max_w + 4)
    if min_h > 24 or min_w > 24:
        return None

    grid_h = random.randint(min_h, 24)
    grid_w = random.randint(min_w, 24)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    try:
        br, bc = random_free_location(grid, base, transparent=BG, border=1, padding=2, padding_connectivity=8)
        paste(grid, base, br, bc, transparent=BG)

        indices = list(range(len(sprites)))
        random.shuffle(indices)
        for idx in indices:
            patch = patches[idx]
            sr, sc = random_free_location(grid, patch, transparent=BG, border=1, padding=2, padding_connectivity=8)
            paste(grid, patch, sr, sc, transparent=BG)
    except ValueError:
        return None

    comps = find_components(grid, bg=BG, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 1 + len(sprites):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.choice([2, 3])
    h = random.choice([5, 6])
    w = random.choice([7, 8])

    marker_colors = random.sample(MARKER_COLORS, sprite_count)
    marker_counts = random.sample([1, 2, 3], sprite_count)
    marker_groups = _sample_marker_groups(h, w, marker_counts)
    if marker_groups is None:
        return None

    marker_info = list(zip(marker_colors, marker_groups))
    all_markers = set().union(*(cells for _color, cells in marker_info))
    core = _choose_overlap_core(h, w, all_markers)
    if core is None:
        return None

    stamp_colors = random.sample(STAMP_COLORS, sprite_count)
    sprites: list[dict] = []
    current_areas: list[int] = []
    for marker_color, marker_cells, stamp_color in zip(marker_colors, marker_groups, stamp_colors):
        stamp = _build_initial_stamp((h, w), marker_cells, all_markers, core)
        if stamp is None:
            return None
        spec = {
            "marker_color": int(marker_color),
            "marker_cells": set(marker_cells),
            "stamp_color": int(stamp_color),
            "stamp_cells": set(stamp),
        }
        current_areas.append(len(spec["marker_cells"]) + len(spec["stamp_cells"]))
        sprites.append(spec)

    for _ in range(80):
        bonuses = random.sample([1, 2, 3, 4, 5], sprite_count)
        target_areas = [current_areas[i] + bonuses[i] for i in range(sprite_count)]
        if len(set(target_areas)) == sprite_count:
            break
    else:
        return None

    for spec, target_area in zip(sprites, target_areas):
        target_stamp_size = target_area - len(spec["marker_cells"])
        expanded = _expand_stamp((h, w), spec["marker_cells"], all_markers, spec["stamp_cells"], target_stamp_size)
        if expanded is None:
            return None
        spec["stamp_cells"] = expanded
        spec["area"] = len(spec["marker_cells"]) + len(spec["stamp_cells"])

    if len({int(spec["area"]) for spec in sprites}) != len(sprites):
        return None

    base = _build_base_crop(h, w, marker_info)
    if not _semantic_checks(base, sprites):
        return None

    expected = _expected_output(base, sprites)
    if max(expected.shape) > 30:
        return None

    grid = _make_grid(base, sprites)
    if grid is None:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
