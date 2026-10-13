from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_COLORS = tuple(range(1, 10))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _find_components(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or labels[r, c] >= 0:
                continue

            comp_id = len(components)
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] < 0 and int(grid[nr, nc]) == color:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))

            coord_arr = np.asarray(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": _bbox_from_coords(coord_arr),
                    "area": int(coord_arr.shape[0]),
                }
            )
    return labels, components


def _is_contiguous_axis(values: np.ndarray) -> bool:
    lo = int(values.min())
    hi = int(values.max())
    return values.shape[0] == hi - lo + 1 and len(set(map(int, values))) == values.shape[0]


def _orientation_candidates(component: dict) -> list[str]:
    coords = component["coords"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    candidates: list[str] = []
    if len(set(map(int, rows))) == 1 and _is_contiguous_axis(cols):
        candidates.append("H")
    if len(set(map(int, cols))) == 1 and _is_contiguous_axis(rows):
        candidates.append("V")
    return candidates


def _validate_bridge_orientation(
    grid: np.ndarray,
    labels: np.ndarray,
    components: list[dict],
    comp_id: int,
    orientation: str,
) -> dict | None:
    component = components[comp_id]
    coords = component["coords"]
    bridge_color = int(component["color"])
    h, w = grid.shape

    if orientation == "H":
        r = int(coords[0, 0])
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        if c0 == 0 or c1 == w - 1:
            return None
        left_id = int(labels[r, c0 - 1])
        right_id = int(labels[r, c1 + 1])
        if left_id < 0 or right_id < 0 or left_id == comp_id or right_id == comp_id or left_id == right_id:
            return None
        neighbor_ids: set[int] = set()
        for c in range(c0, c1 + 1):
            for dr, dc in FOUR_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nbr_id = int(labels[nr, nc])
                    if nbr_id >= 0 and nbr_id != comp_id:
                        neighbor_ids.add(nbr_id)
        if neighbor_ids != {left_id, right_id}:
            return None
        object_ids = (left_id, right_id)
        bbox = (r, c0, r, c1)
    else:
        c = int(coords[0, 1])
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        if r0 == 0 or r1 == h - 1:
            return None
        top_id = int(labels[r0 - 1, c])
        bottom_id = int(labels[r1 + 1, c])
        if top_id < 0 or bottom_id < 0 or top_id == comp_id or bottom_id == comp_id or top_id == bottom_id:
            return None
        neighbor_ids = set()
        for r in range(r0, r1 + 1):
            for dr, dc in FOUR_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nbr_id = int(labels[nr, nc])
                    if nbr_id >= 0 and nbr_id != comp_id:
                        neighbor_ids.add(nbr_id)
        if neighbor_ids != {top_id, bottom_id}:
            return None
        object_ids = (top_id, bottom_id)
        bbox = (r0, c, r1, c)

    object_colors = [int(components[obj_id]["color"]) for obj_id in object_ids]
    if object_colors[0] != object_colors[1] or object_colors[0] == bridge_color:
        return None

    return {
        "bridge_id": comp_id,
        "bridge_color": bridge_color,
        "orientation": orientation,
        "object_ids": object_ids,
        "bbox": bbox,
    }


def _detect_bridges(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    labels, components = _find_components(grid)
    color_counts: dict[int, int] = {}
    for component in components:
        color = int(component["color"])
        color_counts[color] = color_counts.get(color, 0) + 1

    bridges: list[dict] = []
    used_objects: set[int] = set()
    for comp_id, component in enumerate(components):
        if color_counts[int(component["color"])] != 1:
            continue
        candidates = [
            info
            for orientation in _orientation_candidates(component)
            if (info := _validate_bridge_orientation(grid, labels, components, comp_id, orientation)) is not None
        ]
        if len(candidates) != 1:
            continue
        info = candidates[0]
        a, b = info["object_ids"]
        assert a not in used_objects and b not in used_objects, "an object cannot belong to multiple bridges"
        used_objects.add(a)
        used_objects.add(b)
        bridges.append(info)

    bridges.sort(key=lambda info: int(info["bridge_id"]))
    return bridges, components


def _scale_index(dst_idx: int, dst_len: int, src_len: int) -> int:
    if src_len == 1 or dst_len == 1:
        return 0
    return int(round(dst_idx * (src_len - 1) / (dst_len - 1)))


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = mask.shape
    dst_h, dst_w = target_shape
    out = np.zeros((dst_h, dst_w), dtype=bool)
    for r in range(dst_h):
        sr = _scale_index(r, dst_h, src_h)
        for c in range(dst_w):
            sc = _scale_index(c, dst_w, src_w)
            out[r, c] = bool(mask[sr, sc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find same-color object pairs joined by a one-cell bridge. Recolor each paired
    object with the bridge color and redraw the bridge cells with the union of the
    two normalized object shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g != BG), "expected foreground content"

    bridges, components = _detect_bridges(g)
    assert bridges, "expected at least one valid bridge"

    out = np.full_like(g, BG)
    for info in bridges:
        color = int(info["bridge_color"])
        for obj_id in info["object_ids"]:
            coords = components[obj_id]["coords"]
            out[coords[:, 0], coords[:, 1]] = color

        r0, c0, r1, c1 = info["bbox"]
        target_shape = (r1 - r0 + 1, c1 - c0 + 1)
        masks = [_component_mask(components[obj_id]) for obj_id in info["object_ids"]]
        overlay = _resize_mask(masks[0], target_shape) | _resize_mask(masks[1], target_shape)
        out[r0 : r1 + 1, c0 : c1 + 1][overlay] = color

    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _random_connected_mask(height: int, width: int, anchor: tuple[int, int]) -> np.ndarray:
    area_lo = max(1, (height * width + 3) // 4)
    area_hi = max(area_lo, height * width - random.randint(0, max(0, (height * width) // 3)))
    target_area = random.randint(area_lo, area_hi)

    cells = {anchor}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in FOUR_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*anchor)
    while len(cells) < target_area and frontier:
        ranked = sorted(
            frontier,
            key=lambda rc: sum((rc[0] + dr, rc[1] + dc) in cells for dr, dc in FOUR_NEIGHBORS),
            reverse=True,
        )
        top_k = max(1, min(len(ranked), 4))
        cell = random.choice(ranked[:top_k] if random.random() < 0.7 else ranked)
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _anchor_for_side(height: int, width: int, side: str) -> tuple[int, int]:
    if side == "left":
        return random.randrange(height), 0
    if side == "right":
        return random.randrange(height), width - 1
    if side == "top":
        return 0, random.randrange(width)
    if side == "bottom":
        return height - 1, random.randrange(width)
    raise ValueError(f"unknown side {side}")


def _object_patch(color: int, side: str) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(40):
        height = random.randint(1, 5)
        width = random.randint(1, 5)
        if height == width == 1 and random.random() < 0.8:
            continue
        anchor = _anchor_for_side(height, width, side)
        mask = _random_connected_mask(height, width, anchor)
        rows = np.flatnonzero(np.any(mask, axis=1))
        cols = np.flatnonzero(np.any(mask, axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        anchor = (anchor[0] - r0, anchor[1] - c0)
        patch = np.where(mask, color, BG)
        return patch, anchor
    raise RuntimeError("failed to build anchored object patch")


def _pair_patch(object_color: int, bridge_color: int, orientation: str) -> np.ndarray:
    if orientation == "H":
        left_patch, left_anchor = _object_patch(object_color, "right")
        right_patch, right_anchor = _object_patch(object_color, "left")
        gap = random.randint(2, 8)

        top_left = max(0, right_anchor[0] - left_anchor[0])
        top_right = max(0, left_anchor[0] - right_anchor[0])
        bridge_row = top_left + left_anchor[0]
        height = max(top_left + left_patch.shape[0], top_right + right_patch.shape[0])
        width = left_patch.shape[1] + gap + right_patch.shape[1]

        patch = np.full((height, width), BG, dtype=int)
        patch[top_left : top_left + left_patch.shape[0], : left_patch.shape[1]][left_patch != BG] = object_color
        right_c0 = left_patch.shape[1] + gap
        patch[top_right : top_right + right_patch.shape[0], right_c0 : right_c0 + right_patch.shape[1]][
            right_patch != BG
        ] = object_color
        patch[bridge_row, left_patch.shape[1] : left_patch.shape[1] + gap] = bridge_color
        return patch

    top_patch, top_anchor = _object_patch(object_color, "bottom")
    bottom_patch, bottom_anchor = _object_patch(object_color, "top")
    gap = random.randint(2, 8)

    left_top = max(0, bottom_anchor[1] - top_anchor[1])
    left_bottom = max(0, top_anchor[1] - bottom_anchor[1])
    bridge_col = left_top + top_anchor[1]
    height = top_patch.shape[0] + gap + bottom_patch.shape[0]
    width = max(left_top + top_patch.shape[1], left_bottom + bottom_patch.shape[1])

    patch = np.full((height, width), BG, dtype=int)
    patch[: top_patch.shape[0], left_top : left_top + top_patch.shape[1]][top_patch != BG] = object_color
    bottom_r0 = top_patch.shape[0] + gap
    patch[bottom_r0 : bottom_r0 + bottom_patch.shape[0], left_bottom : left_bottom + bottom_patch.shape[1]][
        bottom_patch != BG
    ] = object_color
    patch[top_patch.shape[0] : top_patch.shape[0] + gap, bridge_col] = bridge_color
    return patch


def _distractor_patch(color: int) -> np.ndarray:
    side = random.choice(("left", "right", "top", "bottom"))
    patch, _anchor = _object_patch(color, side)
    return patch


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False

    fg = patch != BG
    occ = grid != BG
    blocked = _dilate4(occ)
    region = blocked[r0 : r0 + ph, c0 : c0 + pw]
    return not np.any(region & fg)


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    region = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BG
    region[mask] = patch[mask]


def _place_randomly(grid: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for r0, c0 in positions[:120]:
        if _can_place_patch(grid, patch, r0, c0):
            _paste_patch(grid, patch, r0, c0)
            return True
    return False


def _sample_orientations(pair_count: int) -> list[str]:
    if pair_count == 1:
        return [random.choice(("H", "V"))]
    if pair_count == 2 and random.random() < 0.6:
        return random.sample(["H", "V"], k=2)
    if pair_count == 3 and random.random() < 0.7:
        orientations = ["H", "V", random.choice(("H", "V"))]
        random.shuffle(orientations)
        return orientations
    return [random.choice(("H", "V")) for _ in range(pair_count)]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(15, 28)
    width = random.randint(15, 28)
    grid = np.full((height, width), BG, dtype=int)

    pair_count = random.randint(1, 3)
    max_distractors = min(2, len(ALL_COLORS) - 2 * pair_count)
    distractor_count = random.randint(1, max_distractors)
    colors = random.sample(ALL_COLORS, 2 * pair_count + distractor_count)
    orientations = _sample_orientations(pair_count)

    for idx in range(pair_count):
        object_color = int(colors[2 * idx])
        bridge_color = int(colors[2 * idx + 1])
        patch = _pair_patch(object_color, bridge_color, orientations[idx])
        if not _place_randomly(grid, patch):
            return None

    for idx in range(distractor_count):
        color = int(colors[2 * pair_count + idx])
        patch = _distractor_patch(color)
        if not _place_randomly(grid, patch):
            return None

    bridges, components = _detect_bridges(grid)
    if len(bridges) != pair_count:
        return None

    bridged_objects = {obj_id for info in bridges for obj_id in info["object_ids"]}
    if len(bridged_objects) != 2 * pair_count:
        return None

    total_components = len(components)
    if total_components <= 2 * pair_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or not np.any(out != BG):
        return None
    if len(set(orientations)) == 1 and random.random() < 0.15:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-overlay ARC input")
