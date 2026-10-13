from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import (  # type: ignore
    bbox,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    is_contiguous,
    neighbors4,
    paste,
    random_free_location,
)


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
OWN_COLORS = (BLUE, GREEN, MAGENTA)
MARKER_COLORS = (RED, YELLOW)

Coord = tuple[int, int]
CoordSet = frozenset[Coord]


def _component_data(grid: np.ndarray, comp: dict) -> tuple[np.ndarray, np.ndarray, set[int]]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    mask = np.zeros(patch.shape, dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    colors = {int(grid[int(r), int(c)]) for r, c in coords}
    return patch, mask, colors


def _coord_set(coords: np.ndarray) -> CoordSet:
    return frozenset((int(r), int(c)) for r, c in coords)


def _normalized(coords: CoordSet) -> CoordSet:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _component_sets(mask: np.ndarray) -> list[CoordSet]:
    sets = [_coord_set(coords) for coords in components_mask(mask, connectivity=4)]
    assert sets, "marker mask must contain at least one component"
    return sets


def _unique_marker_offset(src_marker: np.ndarray, dst_marker: np.ndarray) -> tuple[int, int]:
    src_components = _component_sets(src_marker)
    dst_components = _component_sets(dst_marker)
    dst_component_set = set(dst_components)

    candidate_offsets: set[tuple[int, int]] = set()
    for src_comp in src_components:
        src_norm = _normalized(src_comp)
        sr0 = min(r for r, _ in src_comp)
        sc0 = min(c for _, c in src_comp)
        for dst_comp in dst_components:
            if len(dst_comp) != len(src_comp) or _normalized(dst_comp) != src_norm:
                continue
            dr0 = min(r for r, _ in dst_comp)
            dc0 = min(c for _, c in dst_comp)
            candidate_offsets.add((dr0 - sr0, dc0 - sc0))

    valid_offsets = []
    for dr, dc in candidate_offsets:
        if all(frozenset((r + dr, c + dc) for r, c in src_comp) in dst_component_set for src_comp in src_components):
            valid_offsets.append((dr, dc))

    assert len(valid_offsets) == 1, "marker components must determine a unique translation"
    return valid_offsets[0]


def _flip_by_aspect(arr: np.ndarray) -> np.ndarray:
    h, w = map(int, arr.shape)
    assert h != w, "source sprite must be non-square"
    return np.fliplr(arr) if w > h else np.flipud(arr)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each smaller sprite by its own aspect ratio, then stamp it onto the largest target object by shared marker alignment.

    Input structure:
    - A black background contains one largest 4-connected multicolor target object made only of red and yellow pixels.
    - Two or three smaller 4-connected source sprites are separated from the target; each source mixes one own color with exactly one shared marker color from the target.
    - The shared marker color inside a source forms one or more connected components whose shapes match connected components of the same color inside the target.

    Rule:
    - Find the unique largest object and crop its tight bounding box; this crop is the output canvas.
    - For each smaller source sprite, crop its own bbox and mirror it first: wide sprites flip left-right, tall sprites flip top-bottom.
    - In the mirrored sprite, align the full shared-marker component set to the identically colored component set inside the target by translation.
    - Stamp the mirrored sprites onto the cropped target in decreasing source-area order, with later non-black pixels overwriting earlier ones.
    - Clip any stamped pixels that fall outside the cropped target canvas.

    Key insight:
    - The alignment anchor is the shared marker component geometry, but the visible effect comes from mirroring the whole source before alignment, and the mirror axis depends only on the source sprite's bbox aspect ratio.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert len(comps) >= 3, "expected one target and multiple source sprites"

    areas = [int(comp["area"]) for comp in comps]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "target object must be uniquely largest"
    target_idx = areas.index(max_area)
    target_comp = comps[target_idx]
    target_patch, target_mask, target_colors = _component_data(g, target_comp)
    assert target_colors == {RED, YELLOW}, "target must use exactly red and yellow"

    sources = [comp for idx, comp in enumerate(comps) if idx != target_idx]
    sources.sort(key=lambda comp: (-int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))

    out = target_patch.copy()
    for src_comp in sources:
        src_patch, src_mask, src_colors = _component_data(g, src_comp)
        shared_colors = src_colors & target_colors
        assert len(shared_colors) == 1, "each source must share exactly one marker color with the target"
        marker_color = int(next(iter(shared_colors)))

        flipped_patch = _flip_by_aspect(src_patch)
        flipped_mask = _flip_by_aspect(src_mask.astype(np.uint8)).astype(bool)

        src_marker = (flipped_patch == marker_color) & flipped_mask
        dst_marker = (target_patch == marker_color) & target_mask
        top, left = _unique_marker_offset(src_marker, dst_marker)

        paste(out, flipped_patch, top, left, mask=flipped_mask, overwrite=True, clip=True)

    return out


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


WIDE_TEMPLATES: dict[int, np.ndarray] = {
    6: _mask(["1000", "1111", "0010"]),
    7: _mask(["1100", "1111", "0010"]),
    8: _mask(["10000", "11111", "00110"]),
    9: _mask(["10000", "11110", "00110", "00110"]),
    10: _mask(["11000", "11110", "00110", "00110"]),
}
TALL_TEMPLATES: dict[int, np.ndarray] = {area: mask.T.copy() for area, mask in WIDE_TEMPLATES.items()}

MARKER_SHAPES: tuple[tuple[Coord, ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
)

GRID_SIDE_RANGE = (18, 24)
SOURCE_COUNT_OPTIONS = (2, 3)
TARGET_H_RANGE = (7, 9)
TARGET_W_RANGE = (7, 10)
TARGET_AREA_RANGE = (20, 30)
GEN_RETRIES = 400


def _shape_dims(shape: tuple[Coord, ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _normalize_shape(coords: list[Coord] | tuple[Coord, ...] | CoordSet) -> tuple[Coord, ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_cells(shape: tuple[Coord, ...], top: int, left: int) -> set[Coord]:
    return {(top + r, left + c) for r, c in shape}


def _touches(existing: set[Coord], cells: set[Coord]) -> bool:
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in existing and (nr, nc) not in cells:
                return True
    return False


def _place_target_markers(h: int, w: int, shapes: list[tuple[Coord, ...]]) -> list[dict] | None:
    placed: list[dict] = []
    occupied: set[Coord] = set()
    for shape in sorted(shapes, key=len, reverse=True):
        sh, sw = _shape_dims(shape)
        candidates = []
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                cells = _shape_cells(shape, r0, c0)
                if cells & occupied:
                    continue
                if _touches(occupied, cells):
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        cells = random.choice(candidates)
        occupied |= cells
        placed.append({"shape": _normalize_shape(cells), "coords": cells})
    return placed


def _free_adjacent(cells: set[Coord], h: int, w: int, blocked: np.ndarray) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc]:
                out.add((int(nr), int(nc)))
    return out


def _bfs_path(shape: tuple[int, int], blocked: np.ndarray, starts: set[Coord], goals: set[Coord]) -> list[Coord] | None:
    if not starts or not goals:
        return None
    goal_set = set(goals)
    queue = deque(starts)
    prev: dict[Coord, Coord | None] = {cell: None for cell in starts}
    h, w = map(int, shape)

    while queue:
        cell = queue.popleft()
        if cell in goal_set:
            path = []
            cur: Coord | None = cell
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        r, c = cell
        for nr, nc in neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc] or nxt in prev:
                continue
            prev[nxt] = cell
            queue.append(nxt)
    return None


def _bbox_area(coords: set[Coord]) -> int:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _grow_target(object_cells: set[Coord], blocked: np.ndarray, goal_area: int) -> set[Coord]:
    h, w = blocked.shape
    cells = set(object_cells)
    while len(cells) < goal_area:
        frontier = []
        current_bbox_area = _bbox_area(cells)
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                nxt = (int(nr), int(nc))
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if blocked[nr, nc] or nxt in cells:
                    continue
                trial = set(cells)
                trial.add(nxt)
                score = (_bbox_area(trial) - current_bbox_area, random.random())
                frontier.append((score, nxt))
        if not frontier:
            break
        frontier.sort(key=lambda item: item[0], reverse=True)
        top = frontier[: min(6, len(frontier))]
        cells.add(random.choice(top)[1])
    return cells


def _build_target_patch(n_sources: int) -> tuple[np.ndarray, int, list[dict]] | None:
    marker_color = random.choice(MARKER_COLORS)
    scaffold_color = YELLOW if marker_color == RED else RED
    chosen_shapes = random.sample(MARKER_SHAPES, n_sources)

    for _ in range(120):
        h = random.randint(*TARGET_H_RANGE)
        w = random.randint(*TARGET_W_RANGE)
        markers = _place_target_markers(h, w, chosen_shapes)
        if markers is None:
            continue

        blocked = np.zeros((h, w), dtype=bool)
        for marker in markers:
            for r, c in marker["coords"]:
                blocked[r, c] = True

        scaffold: set[Coord] = set()
        object_cells = set(markers[0]["coords"])
        for marker in markers[1:]:
            starts = _free_adjacent(object_cells, h, w, blocked)
            goals = _free_adjacent(marker["coords"], h, w, blocked)
            path = _bfs_path((h, w), blocked, starts, goals)
            if path is None:
                break
            scaffold.update(path)
            object_cells.update(path)
            object_cells.update(marker["coords"])
        else:
            goal_area = random.randint(max(TARGET_AREA_RANGE[0], len(object_cells)), min(TARGET_AREA_RANGE[1], h * w))
            full_cells = _grow_target(object_cells, blocked, goal_area)
            grid = np.zeros((h, w), dtype=int)
            for r, c in full_cells:
                grid[r, c] = scaffold_color
            for marker in markers:
                for r, c in marker["coords"]:
                    grid[r, c] = marker_color

            bb = bbox(grid != BG)
            assert bb is not None
            patch = crop_bbox(grid, bb)
            if min(patch.shape) < 6 or max(patch.shape) > 10:
                continue
            if not is_contiguous(patch, background=BG, connectivity=4):
                continue

            patch_markers = find_components(
                patch,
                bg=BG,
                connectivity=4,
                monochromatic=True,
                colors=[marker_color],
                return_format="dict",
            )
            if len(patch_markers) != n_sources:
                continue
            norm_shapes = {_normalize_shape([tuple(map(int, coord)) for coord in comp["coords"]]) for comp in patch_markers}
            if len(norm_shapes) != n_sources:
                continue
            return patch, marker_color, patch_markers

    return None


def _template_bank(orientation: str) -> dict[int, np.ndarray]:
    return WIDE_TEMPLATES if orientation == "wide" else TALL_TEMPLATES


def _marker_placements(mask: np.ndarray, shape: tuple[Coord, ...]) -> list[set[Coord]]:
    h, w = mask.shape
    sh, sw = _shape_dims(shape)
    placements: list[set[Coord]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            cells = _shape_cells(shape, r0, c0)
            if all(mask[r, c] for r, c in cells):
                placements.append(cells)
    return placements


def _build_stamp(mask: np.ndarray, placement: set[Coord], marker_color: int, own_color: int) -> np.ndarray:
    stamp = np.zeros(mask.shape, dtype=int)
    stamp[mask] = own_color
    for r, c in placement:
        stamp[r, c] = marker_color
    return stamp


def _edge_bias(cells: set[Coord], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return min(min(r, h - 1 - r, c, w - 1 - c) for r, c in cells), random.randint(0, 10_000)


def _build_source_spec(
    target_patch: np.ndarray,
    marker_color: int,
    target_marker_comp: dict,
    orientation: str,
    area: int,
    own_color: int,
) -> dict | None:
    template = _template_bank(orientation)[area]
    target_shape = _normalize_shape([tuple(map(int, coord)) for coord in target_marker_comp["coords"]])
    placements = _marker_placements(template, target_shape)
    if not placements:
        return None

    placements.sort(key=lambda cells: _edge_bias(cells, template.shape))
    placement = random.choice(placements[: min(4, len(placements))])
    stamp = _build_stamp(template, placement, marker_color, own_color)
    if np.array_equal(stamp, _flip_by_aspect(stamp)):
        return None

    marker_mask = stamp == marker_color
    marker_comps = components_mask(marker_mask, connectivity=4)
    if len(marker_comps) != 1:
        return None
    if _normalize_shape([tuple(map(int, coord)) for coord in marker_comps[0]]) != target_shape:
        return None

    top, left = _unique_marker_offset(marker_mask, target_patch == marker_color)
    input_patch = _flip_by_aspect(stamp)
    return {
        "area": int(np.count_nonzero(stamp)),
        "stamp": stamp,
        "input": input_patch,
        "offset": (top, left),
        "marker_shape": target_shape,
    }


def _visible_coords(spec: dict, out_shape: tuple[int, int]) -> set[Coord]:
    h, w = out_shape
    top, left = spec["offset"]
    coords: set[Coord] = set()
    for r, c in np.argwhere(spec["stamp"] != BG):
        rr = top + int(r)
        cc = left + int(c)
        if 0 <= rr < h and 0 <= cc < w:
            coords.add((rr, cc))
    return coords


def _layout_ok(target_patch: np.ndarray, specs: list[dict], marker_areas: list[int]) -> bool:
    covered = []
    for spec, marker_area in zip(specs, marker_areas):
        visible = _visible_coords(spec, target_patch.shape)
        if len(visible) < marker_area + 2:
            return False
        covered.append(visible)

    for i in range(len(covered)):
        for j in range(i + 1, len(covered)):
            if len(covered[i] & covered[j]) > 4:
                return False
    return True


def _expected_output(target_patch: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = target_patch.copy()
    for spec in sorted(specs, key=lambda item: -item["area"]):
        paste(out, spec["stamp"], spec["offset"][0], spec["offset"][1], mask=spec["stamp"] != BG, clip=True)
    return out


def _compose_input(target_patch: np.ndarray, source_inputs: list[np.ndarray]) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(*GRID_SIDE_RANGE)
        w = random.randint(*GRID_SIDE_RANGE)
        canvas = np.zeros((h, w), dtype=int)
        pieces = [target_patch, *source_inputs]
        try:
            for patch in sorted(pieces, key=lambda arr: -int(np.count_nonzero(arr))):
                r0, c0 = random_free_location(canvas, patch, transparent=BG, border=1, padding=2, padding_connectivity=8)
                paste(canvas, patch, r0, c0, transparent=BG)
        except ValueError:
            continue
        return canvas
    return None


def _gen_attempt() -> np.ndarray | None:
    n_sources = random.choice(SOURCE_COUNT_OPTIONS)
    target_bundle = _build_target_patch(n_sources)
    if target_bundle is None:
        return None
    target_patch, marker_color, marker_components = target_bundle
    own_colors = random.sample(OWN_COLORS, n_sources)

    orientations = ["wide", "tall"]
    if n_sources == 3:
        orientations.append(random.choice(["wide", "tall"]))
    random.shuffle(orientations)

    areas = random.sample(sorted(WIDE_TEMPLATES), n_sources)
    marker_components = random.sample(marker_components, n_sources)
    specs: list[dict] = []
    marker_areas: list[int] = []
    for comp, orientation, area, own_color in zip(marker_components, orientations, areas, own_colors):
        marker_areas.append(int(comp["area"]))
        spec = _build_source_spec(target_patch, marker_color, comp, orientation, area, own_color)
        if spec is None:
            return None
        specs.append(spec)

    if len({spec["area"] for spec in specs}) != len(specs):
        return None
    if not _layout_ok(target_patch, specs, marker_areas):
        return None

    expected = _expected_output(target_patch, specs)
    inp = _compose_input(target_patch, [spec["input"] for spec in specs])
    if inp is None:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 039/5")
