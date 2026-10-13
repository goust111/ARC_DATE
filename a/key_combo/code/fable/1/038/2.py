from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BBox = tuple[int, int, int, int]

_CORNER_PATTERNS = {
    ((0, 0), (0, 1), (1, 0)): "TL",
    ((0, 0), (0, 1), (1, 1)): "TR",
    ((0, 0), (1, 0), (1, 1)): "BL",
    ((0, 1), (1, 0), (1, 1)): "BR",
}
_ROLES = ("TL", "TR", "BL", "BR")

_GEN_BG = 0
_GEN_COLORS = list(range(1, 10))
_GEN_DEPTHS = (2, 3)
_GEN_RETRY_LIMIT = 400
_GEN_SYSTEM_RETRIES = 200
_GEN_MAX_PATCH_SIDE = 13
_GEN_MIN_GAP = 2
_GEN_MAX_GAP = 4
_GEN_BORDER = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _local_mask(coords: np.ndarray) -> tuple[np.ndarray, BBox]:
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, box


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_full_frame(coords: np.ndarray) -> bool:
    mask, _box = _local_mask(coords)
    height, width = map(int, mask.shape)
    if height < 3 or width < 3:
        return False
    return np.array_equal(mask, _frame_mask(height, width))


def _corner_role(coords: np.ndarray) -> str | None:
    mask, box = _local_mask(coords)
    if mask.shape != (2, 2) or int(mask.sum()) != 3:
        return None
    r0, c0, _r1, _c1 = box
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return _CORNER_PATTERNS.get(rel)


def _corner_anchor(role: str, box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if role == "TL":
        return int(r0), int(c0)
    if role == "TR":
        return int(r0), int(c1)
    if role == "BL":
        return int(r1), int(c0)
    assert role == "BR"
    return int(r1), int(c1)


def _solve_corner_frames(role_points: dict[str, list[tuple[int, int]]]) -> list[BBox]:
    counts = [len(role_points[role]) for role in _ROLES]
    assert len(set(counts)) == 1, "corner-only colors must expose four roles equally"
    count = counts[0]
    if count == 0:
        return []

    tls = role_points["TL"]
    trs = role_points["TR"]
    bls = role_points["BL"]
    brs = role_points["BR"]
    br_lookup = {(r, c): idx for idx, (r, c) in enumerate(brs)}

    candidates_by_tl: list[list[tuple[int, int, int, BBox]]] = [[] for _ in range(count)]
    for i, (top, left) in enumerate(tls):
        for j, (top2, right) in enumerate(trs):
            if top2 != top or right - left < 2:
                continue
            for k, (bottom, left2) in enumerate(bls):
                if left2 != left or bottom - top < 2:
                    continue
                l = br_lookup.get((bottom, right))
                if l is None:
                    continue
                candidates_by_tl[i].append((j, k, l, (top, left, bottom, right)))

    assert all(candidates_by_tl), "each top-left corner must belong to a rectangle"

    used_tl = [False] * count
    used_tr = [False] * count
    used_bl = [False] * count
    used_br = [False] * count
    chosen: list[BBox] = []
    solution: tuple[BBox, ...] | None = None
    multiple = False

    def rec() -> None:
        nonlocal solution, multiple
        if multiple:
            return
        next_tl = None
        for idx in range(count):
            if not used_tl[idx]:
                next_tl = idx
                break
        if next_tl is None:
            boxes = tuple(sorted(chosen))
            if solution is None:
                solution = boxes
            elif solution != boxes:
                multiple = True
            return

        for j, k, l, box in candidates_by_tl[next_tl]:
            if used_tr[j] or used_bl[k] or used_br[l]:
                continue
            used_tl[next_tl] = True
            used_tr[j] = True
            used_bl[k] = True
            used_br[l] = True
            chosen.append(box)
            rec()
            chosen.pop()
            used_tl[next_tl] = False
            used_tr[j] = False
            used_bl[k] = False
            used_br[l] = False

    rec()
    assert solution is not None and not multiple, "corner roles must determine unique rectangles"
    return list(solution)


def _strict_contains(outer: BBox, inner: BBox) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _contains_or_equal(outer: BBox, inner: BBox) -> bool:
    return (
        int(outer[0]) <= int(inner[0])
        and int(outer[1]) <= int(inner[1])
        and int(outer[2]) >= int(inner[2])
        and int(outer[3]) >= int(inner[3])
    )


def _bbox_area(box: BBox) -> int:
    return int((box[2] - box[0] + 1) * (box[3] - box[1] + 1))


def _paint_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _paint_corner_markers(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    v = int(color)
    grid[r0, c0] = v
    grid[r0, c0 + 1] = v
    grid[r0 + 1, c0] = v

    grid[r0, c1 - 1] = v
    grid[r0, c1] = v
    grid[r0 + 1, c1] = v

    grid[r1 - 1, c0] = v
    grid[r1, c0] = v
    grid[r1, c0 + 1] = v

    grid[r1 - 1, c1] = v
    grid[r1, c1 - 1] = v
    grid[r1, c1] = v


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    corner_points: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {role: [] for role in _ROLES}
    )

    for color in [int(c) for c in np.unique(grid) if int(c) != int(bg)]:
        for coords in _components_mask(grid == int(color)):
            role = _corner_role(coords)
            if role is not None:
                corner_points[int(color)][role].append(_corner_anchor(role, _bbox_coords(coords)))
            elif _is_full_frame(coords):
                frames.append({"color": int(color), "bbox": _bbox_coords(coords)})
            else:
                raise AssertionError("expected only full rectangular frames or 2x2 L-corners")

    for color, role_points in corner_points.items():
        if sum(len(points) for points in role_points.values()) == 0:
            continue
        for box in _solve_corner_frames(role_points):
            frames.append({"color": int(color), "bbox": box})

    boxes = [frame["bbox"] for frame in frames]
    assert len(boxes) == len(set(boxes)), "each depth layer must be represented by one rectangle"
    assert frames, "expected at least one frame"
    return frames


def _split_systems(frames: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    boxes = [frame["bbox"] for frame in frames]
    outer_indices = [
        idx
        for idx, box in enumerate(boxes)
        if not any(other != idx and _strict_contains(boxes[other], box) for other in range(len(boxes)))
    ]
    assert len(outer_indices) == 2, "expected exactly two outermost systems"

    systems: list[list[dict[str, object]]] = []
    covered: set[int] = set()
    for outer_idx in outer_indices:
        outer_box = boxes[outer_idx]
        members = [idx for idx, box in enumerate(boxes) if _contains_or_equal(outer_box, box)]
        members.sort(key=lambda idx: _bbox_area(boxes[idx]), reverse=True)
        assert all(
            _strict_contains(boxes[members[i]], boxes[members[i + 1]])
            for i in range(len(members) - 1)
        ), "each system must be a strict nesting chain"
        systems.append([frames[idx] for idx in members])
        covered.update(members)

    assert len(covered) == len(frames), "every frame must belong to one system"
    assert len(systems[0]) == len(systems[1]), "both systems must have the same depth count"
    return systems


def _shift_box(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = map(int, box)
    return int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild corner-only frames and swap the colors of two nested frame systems layer by layer.

    Input structure:
    - The grid has a black background and exactly two disconnected systems of strictly nested, 1-cell-thick rectangular frames.
    - One system is fully drawn; the other mixes complete frames with frames shown only as four same-color 2x2 L-corner triominoes.
    - Both systems have the same number of depth layers, but their sizes, shapes, and positions differ.

    Rule:
    - Identify every complete monochrome frame and every quartet of same-color corner triominoes, then complete each corner quartet into its missing rectangle frame.
    - Partition the reconstructed rectangles into the two outer-to-inner nesting chains.
    - For each depth, swap the colors of the two corresponding rectangles while keeping every rectangle's geometry and location unchanged.

    Key insight:
    - The only stable correspondence between the two systems is nesting depth, and the corner-only rectangles must be restored first so the depth pairing does not shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    frames = _extract_frames(g, bg)
    systems = _split_systems(frames)
    out = np.full(g.shape, int(bg), dtype=int)

    for frame_a, frame_b in zip(systems[0], systems[1]):
        _paint_frame(out, frame_a["bbox"], int(frame_b["color"]))
        _paint_frame(out, frame_b["bbox"], int(frame_a["color"]))

    return out


def _make_system_boxes(depth: int) -> list[BBox] | None:
    for _ in range(_GEN_SYSTEM_RETRIES):
        inner_h = random.randint(3, 5)
        inner_w = random.randint(3, 5)
        boxes_rev = [(0, 0, inner_h - 1, inner_w - 1)]
        for _ in range(depth - 1):
            r0, c0, r1, c1 = boxes_rev[-1]
            boxes_rev.append(
                (
                    r0 - random.randint(2, 3),
                    c0 - random.randint(2, 3),
                    r1 + random.randint(2, 3),
                    c1 + random.randint(2, 3),
                )
            )

        min_r = min(box[0] for box in boxes_rev)
        min_c = min(box[1] for box in boxes_rev)
        boxes = [_shift_box(box, -min_r, -min_c) for box in reversed(boxes_rev)]
        outer = boxes[0]
        height = outer[2] - outer[0] + 1
        width = outer[3] - outer[1] + 1
        if height <= _GEN_MAX_PATCH_SIDE and width <= _GEN_MAX_PATCH_SIDE:
            return boxes
    return None


def _render_system(
    boxes: list[BBox],
    colors: list[int],
    corner_depths: set[int],
) -> np.ndarray:
    outer = boxes[0]
    height = outer[2] - outer[0] + 1
    width = outer[3] - outer[1] + 1
    patch = np.full((height, width), _GEN_BG, dtype=int)
    for depth, box in enumerate(boxes):
        if depth in corner_depths:
            _paint_corner_markers(patch, box, colors[depth])
        else:
            _paint_frame(patch, box, colors[depth])
    return patch


def _corner_eligible_depths(boxes: list[BBox]) -> list[int]:
    eligible: list[int] = []
    for depth, box in enumerate(boxes):
        height = int(box[2] - box[0] + 1)
        width = int(box[3] - box[1] + 1)
        if height >= 5 and width >= 5:
            eligible.append(depth)
    return eligible


def _place_systems(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    h1, w1 = map(int, patch_a.shape)
    h2, w2 = map(int, patch_b.shape)
    gap = random.randint(_GEN_MIN_GAP, _GEN_MAX_GAP)
    options: list[str] = []
    if w1 + w2 + gap + 2 * _GEN_BORDER <= 30 and max(h1, h2) + 2 * _GEN_BORDER <= 30:
        options.append("horizontal")
    if h1 + h2 + gap + 2 * _GEN_BORDER <= 30 and max(w1, w2) + 2 * _GEN_BORDER <= 30:
        options.append("vertical")
    if not options:
        return None

    layout = random.choice(options)
    if layout == "horizontal":
        height = random.randint(max(h1, h2) + 2 * _GEN_BORDER, 30)
        width = random.randint(w1 + w2 + gap + 2 * _GEN_BORDER, 30)
        left_idx, right_idx = random.sample([0, 1], 2)
        widths = [w1, w2]
        heights = [h1, h2]
        left_col = random.randint(
            _GEN_BORDER,
            width - _GEN_BORDER - (widths[left_idx] + gap + widths[right_idx]),
        )
        right_col = random.randint(
            left_col + widths[left_idx] + gap,
            width - _GEN_BORDER - widths[right_idx],
        )
        positions = [None, None]
        positions[left_idx] = (
            random.randint(_GEN_BORDER, height - _GEN_BORDER - heights[left_idx]),
            left_col,
        )
        positions[right_idx] = (
            random.randint(_GEN_BORDER, height - _GEN_BORDER - heights[right_idx]),
            right_col,
        )
    else:
        height = random.randint(h1 + h2 + gap + 2 * _GEN_BORDER, 30)
        width = random.randint(max(w1, w2) + 2 * _GEN_BORDER, 30)
        top_idx, bottom_idx = random.sample([0, 1], 2)
        widths = [w1, w2]
        heights = [h1, h2]
        top_row = random.randint(
            _GEN_BORDER,
            height - _GEN_BORDER - (heights[top_idx] + gap + heights[bottom_idx]),
        )
        bottom_row = random.randint(
            top_row + heights[top_idx] + gap,
            height - _GEN_BORDER - heights[bottom_idx],
        )
        positions = [None, None]
        positions[top_idx] = (
            top_row,
            random.randint(_GEN_BORDER, width - _GEN_BORDER - widths[top_idx]),
        )
        positions[bottom_idx] = (
            bottom_row,
            random.randint(_GEN_BORDER, width - _GEN_BORDER - widths[bottom_idx]),
        )

    canvas = np.full((height, width), _GEN_BG, dtype=int)
    (r1, c1), (r2, c2) = positions
    canvas[r1 : r1 + h1, c1 : c1 + w1] = patch_a
    canvas[r2 : r2 + h2, c2 : c2 + w2] = np.where(
        patch_b != _GEN_BG, patch_b, canvas[r2 : r2 + h2, c2 : c2 + w2]
    )
    return canvas, positions[0], positions[1]


def _build_expected(
    shape: tuple[int, int],
    boxes_a: list[BBox],
    boxes_b: list[BBox],
    colors_a: list[int],
    colors_b: list[int],
) -> np.ndarray:
    out = np.full(shape, _GEN_BG, dtype=int)
    for depth in range(len(boxes_a)):
        _paint_frame(out, boxes_a[depth], colors_b[depth])
        _paint_frame(out, boxes_b[depth], colors_a[depth])
    return out


def _generate_case(
    *,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, dict[str, object]]:
    for _ in range(_GEN_RETRY_LIMIT):
        depth = random.choice(_GEN_DEPTHS)
        colors = random.sample(_GEN_COLORS, 2 * depth)
        colors_a = colors[:depth]
        colors_b = colors[depth:]

        boxes_a = _make_system_boxes(depth)
        boxes_b = _make_system_boxes(depth)
        if boxes_a is None or boxes_b is None:
            continue

        outer_a = boxes_a[0]
        outer_b = boxes_b[0]
        h1 = outer_a[2] - outer_a[0] + 1
        w1 = outer_a[3] - outer_a[1] + 1
        h2 = outer_b[2] - outer_b[0] + 1
        w2 = outer_b[3] - outer_b[1] + 1
        if (h1, w1) == (h2, w2):
            continue
        if h1 * w2 == h2 * w1:
            continue

        partial_idx = random.randrange(2)
        corner_depths_a = set()
        corner_depths_b = set()
        eligible_a = _corner_eligible_depths(boxes_a)
        eligible_b = _corner_eligible_depths(boxes_b)
        eligible = eligible_a if partial_idx == 0 else eligible_b
        if not eligible:
            continue
        max_partial = min(len(eligible), depth - 1)
        if max_partial <= 0:
            continue
        partial_count = random.randint(1, max_partial)
        partial_depths = set(random.sample(eligible, partial_count))
        if partial_idx == 0:
            corner_depths_a = partial_depths
        else:
            corner_depths_b = partial_depths

        patch_a = _render_system(boxes_a, colors_a, corner_depths_a)
        patch_b = _render_system(boxes_b, colors_b, corner_depths_b)
        placed = _place_systems(patch_a, patch_b)
        if placed is None:
            continue

        inp, offset_a, offset_b = placed
        global_boxes_a = [_shift_box(box, offset_a[0], offset_a[1]) for box in boxes_a]
        global_boxes_b = [_shift_box(box, offset_b[0], offset_b[1]) for box in boxes_b]
        expected = _build_expected(inp.shape, global_boxes_a, global_boxes_b, colors_a, colors_b)

        if not np.array_equal(main(inp), expected):
            continue

        meta = {
            "depth": depth,
            "partial_idx": partial_idx,
            "systems": [
                {"boxes": global_boxes_a, "colors": colors_a, "corner_depths": sorted(corner_depths_a)},
                {"boxes": global_boxes_b, "colors": colors_b, "corner_depths": sorted(corner_depths_b)},
            ],
        }
        if return_meta:
            return inp, expected, meta
        return inp

    raise RuntimeError("failed to generate a valid input for nested depth color swap")


def generate_input() -> np.ndarray:
    return _generate_case()
