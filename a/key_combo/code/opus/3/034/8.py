from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SIDES = ("up", "down", "left", "right")
OPPOSITE_SIDE = {"up": "down", "down": "up", "left": "right", "right": "left"}
MARKER_PALETTE = (3, 4, 6, 8)

BG_COLOR = 5
FRAME_COLOR = 0
SKELETON_COLOR = 1
ANCHOR_COLOR = 2

GEN_RETRIES = 400
GRID_H_RANGE = (22, 29)
GRID_W_RANGE = (22, 29)
OBJECT_COUNT_CHOICES = (3, 4)
TARGET_OUTER_RANGE = (6, 9)
SOURCE_OUTER_MAX = 9
PLACEMENT_GAP = 2

_SHAPE_ANCHOR_POOL: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] | None = None


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: tuple[int, ...] | list[int] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    if connectivity != 4:
        raise ValueError("only 4-connectivity is used in this solver")
    if bg is None:
        bg = get_bg(grid)
    if colors is None:
        colors_iter = sorted(set(map(int, np.unique(grid))) - {int(bg)})
    else:
        colors_iter = sorted({int(color) for color in colors if int(color) != int(bg)})

    height, width = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    out = []
    for color in colors_iter:
        for r in range(height):
            for c in range(width):
                if seen[r, c] or int(grid[r, c]) != color:
                    continue
                queue = deque([(r, c)])
                seen[r, c] = True
                coords = []
                while queue:
                    rr, cc = queue.popleft()
                    coords.append((rr, cc))
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                coord_arr = np.array(coords, dtype=int)
                out.append(
                    {
                        "color": int(color),
                        "coords": coord_arr,
                        "bbox": bbox_coords(coord_arr),
                        "area": int(coord_arr.shape[0]),
                    }
                )
    out.sort(key=lambda comp: comp["bbox"])
    return out


def _patch_from_cells(cells: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    coords = np.array([(r, c) for r, c, _color in cells], dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c, color in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _colored_cells(patch: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(patch[r, c])) for r, c in np.argwhere(patch != int(bg))]


def _transform_patch(patch: np.ndarray, k: int, flip: bool) -> np.ndarray:
    transformed = np.rot90(patch, int(k))
    if flip:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _unique_transform_params(patch: np.ndarray) -> list[tuple[int, bool]]:
    out: list[tuple[int, bool]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(patch, k)
        for flip in (False, True):
            variant = np.fliplr(rotated) if flip else rotated
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                out.append((int(k), bool(flip)))
    return out


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _matches_tail(
    core_bbox: tuple[int, int, int, int],
    tail_side: str,
    tail: set[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = core_bbox
    if len(tail) != 2:
        return False
    rows = sorted(r for r, _c in tail)
    cols = sorted(c for _r, c in tail)
    if tail_side == "up":
        return len(set(cols)) == 1 and rows == [r0 - 2, r0 - 1] and c0 < cols[0] < c1
    if tail_side == "down":
        return len(set(cols)) == 1 and rows == [r1 + 1, r1 + 2] and c0 < cols[0] < c1
    if tail_side == "left":
        return len(set(rows)) == 1 and cols == [c0 - 2, c0 - 1] and r0 < rows[0] < r1
    if tail_side == "right":
        return len(set(rows)) == 1 and cols == [c1 + 1, c1 + 2] and r0 < rows[0] < r1
    raise ValueError(f"unknown tail side: {tail_side}")


def _parse_container(comp: dict) -> dict | None:
    coords = {tuple(map(int, cell)) for cell in comp["coords"].tolist()}
    r0, c0, r1, c1 = comp["bbox"]
    candidates = []
    for tail_side in SIDES:
        if tail_side == "up":
            if r1 - r0 + 1 < 5:
                continue
            core_bbox = (r0 + 2, c0, r1, c1)
        elif tail_side == "down":
            if r1 - r0 + 1 < 5:
                continue
            core_bbox = (r0, c0, r1 - 2, c1)
        elif tail_side == "left":
            if c1 - c0 + 1 < 5:
                continue
            core_bbox = (r0, c0 + 2, r1, c1)
        else:
            if c1 - c0 + 1 < 5:
                continue
            core_bbox = (r0, c0, r1, c1 - 2)

        cr0, cc0, cr1, cc1 = core_bbox
        if cr1 - cr0 + 1 < 5 or cc1 - cc0 + 1 < 5:
            continue
        frame = _perimeter_coords(core_bbox)
        if not frame.issubset(coords):
            continue
        tail = coords - frame
        if _matches_tail(core_bbox, tail_side, tail):
            candidates.append(
                {
                    "color": int(comp["color"]),
                    "core_bbox": core_bbox,
                    "tail": tuple(sorted(tail)),
                    "tail_side": tail_side,
                    "open_side": OPPOSITE_SIDE[tail_side],
                    "interior_bbox": (cr0 + 1, cc0 + 1, cr1 - 1, cc1 - 1),
                }
            )

    if not candidates:
        return None
    assert len(candidates) == 1, "container geometry must determine a unique tail side"
    return candidates[0]


def _interior_patch(grid: np.ndarray, container: dict) -> np.ndarray:
    return crop_bbox(grid, container["interior_bbox"])


def _infer_role_colors(grid: np.ndarray, containers: list[dict], bg: int) -> tuple[int, int, set[int]]:
    counts = []
    for container in containers:
        patch = _interior_patch(grid, container)
        vals = patch[patch != int(bg)]
        counts.append(Counter(map(int, vals.tolist())))

    shared = {
        int(color)
        for color in set().union(*(set(counter) for counter in counts))
        if all(counter.get(int(color), 0) > 0 for counter in counts)
    }
    assert len(shared) >= 2, "all containers must share skeleton and anchor colors"

    anchor_candidates = [color for color in shared if all(counter.get(int(color), 0) == 1 for counter in counts)]
    assert len(anchor_candidates) == 1, "anchor color must appear exactly once in every container"
    anchor = int(anchor_candidates[0])

    skeleton_candidates = [color for color in shared if color != anchor]
    assert skeleton_candidates, "missing skeleton color"
    if len(skeleton_candidates) == 1:
        skeleton = int(skeleton_candidates[0])
    else:
        totals = {color: sum(counter.get(int(color), 0) for counter in counts) for color in skeleton_candidates}
        max_total = max(totals.values())
        winners = [color for color, total in totals.items() if total == max_total]
        assert len(winners) == 1, "shared skeleton color must be unique"
        skeleton = int(winners[0])

    extras = set()
    for container in containers:
        patch = _interior_patch(grid, container)
        extras |= {int(color) for color in np.unique(patch) if int(color) not in {int(bg), skeleton, anchor}}
    return skeleton, anchor, extras


def _extract_source_patch(grid: np.ndarray, container: dict, bg: int) -> np.ndarray:
    interior = _interior_patch(grid, container)
    coords = np.argwhere(interior != int(bg))
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return interior[r0 : r1 + 1, c0 : c1 + 1].copy()


def _translated_markers(
    variant: np.ndarray,
    target_skeleton: set[tuple[int, int]],
    target_anchor: tuple[int, int],
    bg: int,
    skeleton: int,
    anchor: int,
) -> tuple[tuple[int, int, int], ...] | None:
    variant_skeleton = {tuple(map(int, coord)) for coord in np.argwhere(variant == int(skeleton)).tolist()}
    variant_anchor_cells = [tuple(map(int, coord)) for coord in np.argwhere(variant == int(anchor)).tolist()]
    if len(variant_anchor_cells) != 1:
        return None
    if len(variant_skeleton) != len(target_skeleton):
        return None

    ar, ac = variant_anchor_cells[0]
    tr, tc = target_anchor
    dr, dc = tr - ar, tc - ac
    shifted_skeleton = {(r + dr, c + dc) for r, c in variant_skeleton}
    if shifted_skeleton != target_skeleton:
        return None

    markers = []
    marker_mask = (variant != int(bg)) & (variant != int(skeleton)) & (variant != int(anchor))
    for r, c in np.argwhere(marker_mask):
        markers.append((int(r + dr), int(c + dc), int(variant[int(r), int(c)])))
    return tuple(sorted(markers))


def _inside_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= int(r) <= r1 and c0 <= int(c) <= c1


def _visible_from_opening(
    r: int,
    c: int,
    skeleton: set[tuple[int, int]],
    interior_bbox: tuple[int, int, int, int],
    open_side: str,
) -> bool:
    r0, c0, r1, c1 = interior_bbox
    if open_side == "up":
        for rr in range(r - 1, r0 - 1, -1):
            if (rr, c) in skeleton:
                return False
        return True
    if open_side == "down":
        for rr in range(r + 1, r1 + 1):
            if (rr, c) in skeleton:
                return False
        return True
    if open_side == "left":
        for cc in range(c - 1, c0 - 1, -1):
            if (r, cc) in skeleton:
                return False
        return True
    if open_side == "right":
        for cc in range(c + 1, c1 + 1):
            if (r, cc) in skeleton:
                return False
        return True
    raise ValueError(f"unknown open side: {open_side}")


def _solve(grid: np.ndarray, *, return_debug: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    raw_components = find_components(g, bg=bg, connectivity=4)
    containers = []
    for comp in raw_components:
        parsed = _parse_container(comp)
        if parsed is not None:
            containers.append(parsed)

    assert 3 <= len(containers) <= 4, "expected 3 or 4 rectangular hollow containers"
    frame_colors = {container["color"] for container in containers}
    assert len(frame_colors) == 1, "all containers must share the same frame color"

    skeleton, anchor, extra_colors = _infer_role_colors(g, containers, bg)
    source_indices = []
    for idx, container in enumerate(containers):
        patch = _interior_patch(g, container)
        patch_colors = {int(color) for color in np.unique(patch) if int(color) != int(bg)}
        if patch_colors - {int(skeleton), int(anchor)}:
            source_indices.append(idx)
    assert len(source_indices) == 1, "expected exactly one fully annotated source container"
    source_idx = int(source_indices[0])

    source_patch = _extract_source_patch(g, containers[source_idx], bg)
    variants = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(source_patch, k)
        for flip in (False, True):
            variant = np.fliplr(rotated) if flip else rotated
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())

    source_extra_abs = []
    source_interior = _interior_patch(g, containers[source_idx])
    ir0, ic0, _ir1, _ic1 = containers[source_idx]["interior_bbox"]
    for r, c in np.argwhere(source_interior != int(bg)):
        color = int(source_interior[int(r), int(c)])
        if color not in {int(skeleton), int(anchor)}:
            source_extra_abs.append((int(ir0 + r), int(ic0 + c), color))

    out = g.copy()
    for container in containers:
        for tr, tc in container["tail"]:
            out[int(tr), int(tc)] = int(bg)

    target_debug = []
    for idx, container in enumerate(containers):
        if idx == source_idx:
            continue

        patch = _interior_patch(g, container)
        target_skeleton = {
            (container["interior_bbox"][0] + int(r), container["interior_bbox"][1] + int(c))
            for r, c in np.argwhere(patch == int(skeleton))
        }
        target_anchor_cells = [
            (container["interior_bbox"][0] + int(r), container["interior_bbox"][1] + int(c))
            for r, c in np.argwhere(patch == int(anchor))
        ]
        assert len(target_anchor_cells) == 1, "each target must contain exactly one anchor"
        target_anchor = target_anchor_cells[0]

        candidates = [
            cells
            for variant in variants
            if (cells := _translated_markers(variant, target_skeleton, target_anchor, bg, skeleton, anchor)) is not None
        ]
        unique_candidates = sorted(set(candidates))
        assert len(unique_candidates) == 1, "target alignment must be unique"
        mapped = list(unique_candidates[0])

        inside = [cell for cell in mapped if _inside_bbox(cell[0], cell[1], container["interior_bbox"])]
        visible = [
            cell
            for cell in inside
            if _visible_from_opening(cell[0], cell[1], target_skeleton, container["interior_bbox"], container["open_side"])
        ]
        for r, c, color in visible:
            assert int(out[int(r), int(c)]) == int(bg), "transferred label must land on empty interior"
            out[int(r), int(c)] = int(color)
        target_debug.append(
            {
                "mapped_markers": mapped,
                "inside_markers": inside,
                "visible_markers": visible,
                "open_side": container["open_side"],
            }
        )

    if return_debug:
        return out, {"source_markers": source_extra_abs, "targets": target_debug}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Erase every tail, align the internal skeletons, and copy only the source labels that remain visible from each target container's opening side.

    Input structure:
    - The dominant background surrounds 3-4 black rectangular hollow containers, each formed by a rectangular frame plus a straight 2-pixel tail on one side.
    - Inside every frame sits the same blue skeleton pattern under some rotation/reflection, together with exactly one red anchor pixel.
    - Exactly one container also contains extra colored label pixels inside its hollow interior; the other containers contain only the skeleton and anchor.

    Rule:
    - Remove the 2-pixel tail from every container; the side opposite the removed tail is that container's opening side.
    - Use the fully labeled container as the source template, and find for each other container the unique dihedral transform that aligns the source blue skeleton and red anchor to the target skeleton and anchor.
    - Transform every extra source label through that alignment, keep only positions that land inside the target interior, and then discard any landed label whose straight line toward the target opening side is blocked by a blue skeleton pixel.
    - Preserve the rectangular frame, the remaining skeleton pixels, the anchor pixels, and the full source labeling.

    Key insight:
    - The skeleton+anchor pair determines how labels should rotate or flip between containers, but each target container keeps only the subset of those transferred labels that has unobstructed line of sight toward its own opening side.
    """
    return _solve(grid, return_debug=False)


def _normalize_shape(shape: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _c in shape)
    c0 = min(c for _r, c in shape)
    return frozenset((r - r0, c - c0) for r, c in shape)


def _shape_anchor_pool() -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    global _SHAPE_ANCHOR_POOL
    if _SHAPE_ANCHOR_POOL is not None:
        return _SHAPE_ANCHOR_POOL

    pool: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = []
    shapes = {frozenset({(0, 0)})}
    for size in range(2, 7):
        next_shapes = set()
        for shape in shapes:
            frontier = set()
            for r, c in shape:
                frontier.update(neighbors4(r, c))
            for cell in frontier - set(shape):
                next_shapes.add(_normalize_shape(set(shape) | {cell}))
        shapes = next_shapes
        if size < 4:
            continue
        for shape in shapes:
            rows = [r for r, _c in shape]
            cols = [c for _r, c in shape]
            if max(rows) - min(rows) + 1 > 3 or max(cols) - min(cols) + 1 > 3:
                continue
            for anchor in shape:
                cells = [
                    (int(r), int(c), ANCHOR_COLOR if (r, c) == anchor else SKELETON_COLOR)
                    for r, c in sorted(shape)
                ]
                patch = _patch_from_cells(cells, BG_COLOR)
                if len(_unique_transform_params(patch)) >= 4:
                    pool.append((tuple(sorted(shape)), tuple(map(int, anchor))))
    assert pool, "shape-anchor pool should not be empty"
    _SHAPE_ANCHOR_POOL = pool
    return pool


def _marker_bucket(cell: tuple[int, int], anchor: tuple[int, int]) -> str:
    dr, dc = cell[0] - anchor[0], cell[1] - anchor[1]
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    if dr != 0:
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _choose_marker_cells(
    shape: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
) -> list[tuple[int, int, int]]:
    occupied = set(shape)
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    candidates = []
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if (r, c) in occupied:
                continue
            if any(max(abs(r - sr), abs(c - sc)) == 1 for sr, sc in occupied):
                candidates.append((r, c))
    assert len(candidates) >= 4

    for _ in range(80):
        count = random.randint(3, 4)
        cells = random.sample(candidates, count)
        if len({_marker_bucket(cell, anchor) for cell in cells}) >= 2:
            colors = random.sample(MARKER_PALETTE, count)
            return [(int(r), int(c), int(color)) for (r, c), color in zip(cells, colors)]

    colors = random.sample(MARKER_PALETTE, 3)
    cells = candidates[:3]
    return [(int(r), int(c), int(color)) for (r, c), color in zip(cells, colors)]


def _tail_axis_positions(length: int) -> list[int]:
    if length % 2 == 1:
        return [length // 2]
    return [length // 2 - 1, length // 2]


def _object_patch(
    outer_h: int,
    outer_w: int,
    tail_side: str,
    interior_cells: list[tuple[int, int, int]],
    *,
    bg: int,
    frame: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    if tail_side in {"up", "down"}:
        patch = np.full((outer_h + 2, outer_w), int(bg), dtype=int)
        core_r0 = 2 if tail_side == "up" else 0
        core_c0 = 0
    else:
        patch = np.full((outer_h, outer_w + 2), int(bg), dtype=int)
        core_r0 = 0
        core_c0 = 2 if tail_side == "left" else 0

    patch[core_r0, core_c0 : core_c0 + outer_w] = int(frame)
    patch[core_r0 + outer_h - 1, core_c0 : core_c0 + outer_w] = int(frame)
    patch[core_r0 : core_r0 + outer_h, core_c0] = int(frame)
    patch[core_r0 : core_r0 + outer_h, core_c0 + outer_w - 1] = int(frame)

    if tail_side == "up":
        col = core_c0 + random.choice(_tail_axis_positions(outer_w))
        patch[core_r0 - 1, col] = int(frame)
        patch[core_r0 - 2, col] = int(frame)
    elif tail_side == "down":
        col = core_c0 + random.choice(_tail_axis_positions(outer_w))
        patch[core_r0 + outer_h, col] = int(frame)
        patch[core_r0 + outer_h + 1, col] = int(frame)
    elif tail_side == "left":
        row = core_r0 + random.choice(_tail_axis_positions(outer_h))
        patch[row, core_c0 - 1] = int(frame)
        patch[row, core_c0 - 2] = int(frame)
    else:
        row = core_r0 + random.choice(_tail_axis_positions(outer_h))
        patch[row, core_c0 + outer_w] = int(frame)
        patch[row, core_c0 + outer_w + 1] = int(frame)

    for ir, ic, color in interior_cells:
        patch[core_r0 + 1 + int(ir), core_c0 + 1 + int(ic)] = int(color)
    return patch, (int(core_r0), int(core_c0))


def _non_bg_cells_with_offset(patch: np.ndarray, top: int, left: int, bg: int) -> list[tuple[int, int, int]]:
    return [(top + r, left + c, int(patch[r, c])) for r, c in np.argwhere(patch != int(bg))]


def _place_objects(
    height: int,
    width: int,
    object_patches: list[dict],
) -> np.ndarray | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    placed_rects = []
    order = sorted(range(len(object_patches)), key=lambda idx: object_patches[idx]["patch"].shape[0] * object_patches[idx]["patch"].shape[1], reverse=True)

    for idx in order:
        patch = object_patches[idx]["patch"]
        ph, pw = patch.shape
        if ph > height or pw > width:
            return None
        placed = False
        for _ in range(200):
            top = random.randint(0, height - ph)
            left = random.randint(0, width - pw)
            rect = (top, left, top + ph - 1, left + pw - 1)
            if any(
                not (
                    rect[2] + PLACEMENT_GAP < other[0]
                    or other[2] + PLACEMENT_GAP < rect[0]
                    or rect[3] + PLACEMENT_GAP < other[1]
                    or other[3] + PLACEMENT_GAP < rect[1]
                )
                for other in placed_rects
            ):
                continue
            placed_rects.append(rect)
            object_patches[idx]["top"] = int(top)
            object_patches[idx]["left"] = int(left)
            for r, c, color in _non_bg_cells_with_offset(patch, top, left, BG_COLOR):
                grid[int(r), int(c)] = int(color)
            placed = True
            break
        if not placed:
            return None
    return grid


def _build_generated_object(
    *,
    source: bool,
    sa_patch: np.ndarray,
    full_patch: np.ndarray,
) -> dict | None:
    tail_side = random.choice(SIDES)
    if source:
        full_h, full_w = full_patch.shape
        min_outer_h = max(TARGET_OUTER_RANGE[0], full_h + 4)
        min_outer_w = max(TARGET_OUTER_RANGE[0], full_w + 4)
        if min_outer_h > SOURCE_OUTER_MAX or min_outer_w > SOURCE_OUTER_MAX:
            return None
        outer_h = random.randint(min_outer_h, SOURCE_OUTER_MAX)
        outer_w = random.randint(min_outer_w, SOURCE_OUTER_MAX)
        interior_h, interior_w = outer_h - 2, outer_w - 2
        top = random.randint(1, interior_h - full_h - 1)
        left = random.randint(1, interior_w - full_w - 1)
        interior_cells = [(top + r, left + c, color) for r, c, color in _colored_cells(full_patch, BG_COLOR)]
    else:
        sa_h, sa_w = sa_patch.shape
        min_outer_h = max(TARGET_OUTER_RANGE[0], sa_h + 2)
        min_outer_w = max(TARGET_OUTER_RANGE[0], sa_w + 2)
        if min_outer_h > TARGET_OUTER_RANGE[1] or min_outer_w > TARGET_OUTER_RANGE[1]:
            return None
        outer_h = random.randint(min_outer_h, TARGET_OUTER_RANGE[1])
        outer_w = random.randint(min_outer_w, TARGET_OUTER_RANGE[1])
        interior_h, interior_w = outer_h - 2, outer_w - 2
        row_limit = interior_h - sa_h
        col_limit = interior_w - sa_w
        top = random.choice([0, row_limit]) if row_limit > 0 and random.random() < 0.6 else random.randint(0, row_limit)
        left = random.choice([0, col_limit]) if col_limit > 0 and random.random() < 0.6 else random.randint(0, col_limit)
        interior_cells = [(top + r, left + c, color) for r, c, color in _colored_cells(sa_patch, BG_COLOR)]

    patch, core_offset = _object_patch(
        outer_h,
        outer_w,
        tail_side,
        interior_cells,
        bg=BG_COLOR,
        frame=FRAME_COLOR,
    )
    return {
        "patch": patch,
        "tail_side": tail_side,
        "core_offset": core_offset,
        "source": source,
    }


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    shape, anchor = random.choice(_shape_anchor_pool())
    markers = _choose_marker_cells(shape, anchor)

    sa_cells = [
        (int(r), int(c), ANCHOR_COLOR if (r, c) == anchor else SKELETON_COLOR)
        for r, c in shape
    ]
    full_cells = sa_cells + markers

    sa_patch = _patch_from_cells(sa_cells, BG_COLOR)
    full_patch = _patch_from_cells(full_cells, BG_COLOR)
    transforms = _unique_transform_params(sa_patch)
    if len(transforms) < 2:
        return None

    source_transform = random.choice(transforms)
    source_sa_patch = _transform_patch(sa_patch, *source_transform)
    source_full_patch = _transform_patch(full_patch, *source_transform)

    source_obj = _build_generated_object(source=True, sa_patch=source_sa_patch, full_patch=source_full_patch)
    if source_obj is None:
        return None

    objects = [source_obj]
    for _ in range(n_objects - 1):
        target_transform = random.choice(transforms)
        target_sa_patch = _transform_patch(sa_patch, *target_transform)
        target_obj = _build_generated_object(source=False, sa_patch=target_sa_patch, full_patch=target_sa_patch)
        if target_obj is None:
            return None
        objects.append(target_obj)

    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    random.shuffle(objects)
    grid = _place_objects(height, width, objects)
    if grid is None:
        return None

    try:
        out, debug = _solve(grid, return_debug=True)
    except AssertionError:
        return None

    source_marker_count = len(debug["source_markers"])
    visible_counts = [len(target["visible_markers"]) for target in debug["targets"]]
    inside_counts = [len(target["inside_markers"]) for target in debug["targets"]]
    mapped_counts = [len(target["mapped_markers"]) for target in debug["targets"]]

    if source_marker_count < 3:
        return None
    if not visible_counts or max(visible_counts) == 0:
        return None
    if not any(count < source_marker_count for count in visible_counts):
        return None
    if len(set(visible_counts)) < 2 and not any(i < m for i, m in zip(inside_counts, mapped_counts)) and not any(v < i for v, i in zip(visible_counts, inside_counts)):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
