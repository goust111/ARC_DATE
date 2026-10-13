from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 5
ORANGE = 7
GRID_SIZE = 15
CENTER = (GRID_SIZE // 2, GRID_SIZE // 2)

CORNER_SPECS = (
    ("TL", (0, 0), 2),
    ("TR", (0, GRID_SIZE - 1), 1),
    ("BL", (GRID_SIZE - 1, 0), 3),
    ("BR", (GRID_SIZE - 1, GRID_SIZE - 1), 4),
)
CORNER_BY_NAME = {name: (coord, color) for name, coord, color in CORNER_SPECS}
CORNER_BY_COLOR = {color: (name, coord) for name, coord, color in CORNER_SPECS}

HOLE_FRAME_DIMS = (
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (3, 5),
    (5, 3),
)

OPEN_SHAPE_LIBRARY = (
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
)

GEN_ATTEMPTS = 500


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any((~mask) & (~exterior)))


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _can_place(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        return False
    mask = patch != BG
    region = dst[r0 : r0 + h, c0 : c0 + w]
    return bool(np.all(region[mask] == BG))


def _stop_reason(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> str:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        return "boundary"
    return "occupied"


def _bbox_center2_from_top_left(r0: int, c0: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return 2 * r0 + h - 1, 2 * c0 + w - 1


def _bbox_center2_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def _corner_distance(center2: tuple[int, int], corner: tuple[int, int]) -> int:
    return abs(center2[0] - 2 * corner[0]) + abs(center2[1] - 2 * corner[1])


def _center_distance(center2: tuple[int, int], target: tuple[int, int]) -> int:
    return abs(center2[0] - 2 * target[0]) + abs(center2[1] - 2 * target[1])


def _nearest_corner_for_bbox(
    r0: int, c0: int, shape: tuple[int, int], grid_shape: tuple[int, int]
) -> tuple[str, int, bool]:
    center2 = _bbox_center2_from_top_left(r0, c0, shape)
    h, w = grid_shape
    corners = (
        ("TL", (0, 0), 2),
        ("TR", (0, w - 1), 1),
        ("BL", (h - 1, 0), 3),
        ("BR", (h - 1, w - 1), 4),
    )
    scored = [
        (_corner_distance(center2, coord), coord[0], coord[1], name, color)
        for name, coord, color in corners
    ]
    scored.sort()
    best_dist = scored[0][0]
    tied = [item for item in scored if item[0] == best_dist]
    chosen = tied[0]
    tie_break_used = len(tied) > 1
    return chosen[3], chosen[4], tie_break_used


def _extract_objects(grid: np.ndarray) -> list[dict]:
    mask = grid != BG
    for _name, (r, c), _color in CORNER_SPECS:
        mask[r, c] = False

    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        assert int(grid[nr, nc]) == color
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local_mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "patch": _patch_from_mask(local_mask, color),
                    "has_hole": _mask_has_hole(local_mask),
                }
            )

    return objects


def _slide_to_corner(
    out: np.ndarray, patch: np.ndarray, start: tuple[int, int], corner_name: str
) -> tuple[tuple[int, int], int, str]:
    r, c = start
    target, _color = CORNER_BY_NAME[corner_name]
    dr = _sign(target[0] - r)
    dc = _sign(target[1] - c)
    steps = 0
    reason = "boundary"

    while True:
        nr = r + dr
        nc = c + dc
        if not _can_place(out, patch, nr, nc):
            reason = _stop_reason(out, patch, nr, nc)
            break
        r, c = nr, nc
        steps += 1

    _paste(out, patch, r, c)
    return (r, c), steps, reason


def _slide_to_center(
    out: np.ndarray, patch: np.ndarray, start: tuple[int, int], center: tuple[int, int]
) -> tuple[tuple[int, int], int, str]:
    r, c = start
    target_r2, target_c2 = 2 * center[0], 2 * center[1]
    steps = 0

    while True:
        cur_r2, cur_c2 = _bbox_center2_from_top_left(r, c, patch.shape)
        diff_r = target_r2 - cur_r2
        diff_c = target_c2 - cur_c2
        dr = _sign(diff_r) if abs(diff_r) >= 2 else 0
        dc = _sign(diff_c) if abs(diff_c) >= 2 else 0
        if dr == 0 and dc == 0:
            _paste(out, patch, r, c)
            return (r, c), steps, "center"

        nr = r + dr
        nc = c + dc
        if not _can_place(out, patch, nr, nc):
            reason = _stop_reason(out, patch, nr, nc)
            _paste(out, patch, r, c)
            return (r, c), steps, reason

        r, c = nr, nc
        steps += 1


def _transform(grid: np.ndarray, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    center = (h // 2, w // 2)

    out = np.full_like(g, BG)
    for _name, (r, c), color in CORNER_SPECS:
        out[r, c] = color

    objects = _extract_objects(g)
    holes = [obj for obj in objects if obj["has_hole"]]
    solids = [obj for obj in objects if not obj["has_hole"]]

    hole_traces: list[dict] = []
    solid_traces: list[dict] = []

    hole_order = sorted(
        holes,
        key=lambda obj: (
            _corner_distance(
                _bbox_center2_from_bbox(obj["bbox"]),
                CORNER_BY_COLOR[obj["color"]][1],
            ),
            obj["bbox"][0],
            obj["bbox"][1],
        ),
    )
    for obj in hole_order:
        corner_name, _coord = CORNER_BY_COLOR[obj["color"]]
        final_rc, steps, reason = _slide_to_corner(
            out, obj["patch"], (obj["bbox"][0], obj["bbox"][1]), corner_name
        )
        hole_traces.append(
            {
                "color": obj["color"],
                "target_corner": corner_name,
                "start": (obj["bbox"][0], obj["bbox"][1]),
                "end": final_rc,
                "steps": steps,
                "stop_reason": reason,
            }
        )

    solid_meta = []
    for obj in solids:
        corner_name, new_color, tie_break = _nearest_corner_for_bbox(
            obj["bbox"][0], obj["bbox"][1], obj["patch"].shape, g.shape
        )
        recolored = np.full(obj["patch"].shape, BG, dtype=int)
        recolored[obj["patch"] != BG] = new_color
        center2 = _bbox_center2_from_bbox(obj["bbox"])
        solid_meta.append(
            (
                _center_distance(center2, center),
                obj["bbox"][0],
                obj["bbox"][1],
                obj,
                recolored,
                corner_name,
                new_color,
                tie_break,
            )
        )

    for _dist, _r0, _c0, obj, recolored, corner_name, new_color, tie_break in sorted(
        solid_meta, key=lambda item: (-item[0], item[1], item[2])
    ):
        final_rc, steps, reason = _slide_to_center(
            out, recolored, (obj["bbox"][0], obj["bbox"][1]), center
        )
        end_center2 = _bbox_center2_from_top_left(final_rc[0], final_rc[1], recolored.shape)
        solid_traces.append(
            {
                "old_color": obj["color"],
                "new_color": new_color,
                "nearest_corner": corner_name,
                "tie_break_used": tie_break,
                "start": (obj["bbox"][0], obj["bbox"][1]),
                "end": final_rc,
                "steps": steps,
                "stop_reason": reason,
                "reached_center": end_center2 == (2 * center[0], 2 * center[1]),
            }
        )

    if return_trace:
        return out, {"holes": hole_traces, "solids": solid_traces}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify objects by enclosed holes, send hole-objects to their matching corner, and recolor holeless objects before pulling them toward the center.

    Input structure:
    - The background is grey, and the four corner cells are fixed single-pixel markers: top-left red, top-right blue, bottom-left green, bottom-right yellow.
    - Inside the grid are several monochromatic 4-connected objects separated by at least one background cell.
    - Some objects are closed frames whose pixels enclose one or more grey holes; the rest are orange open shapes with no enclosed hole.

    Rule:
    - Detect whether each object contains an enclosed background hole inside its tight bounding box.
    - A hole-object keeps its color and slides diagonally toward the corner marker of the same color, stopping when the next step would leave the grid or hit an already placed object.
    - A holeless object finds the nearest corner marker by Manhattan distance from its bounding-box center, breaking ties in favor of the smaller row marker, is recolored to that marker's color, and then slides step by step toward the grid center until it reaches that center or the next step would collide.
    - The four corner marker pixels remain unchanged in the output.

    Key insight:
    - The same corner markers carry two different roles at once: for hole-objects they name the destination corner, while for holeless objects they only supply the recoloring chosen by geometric proximity before the center-packing step.
    """
    return _transform(grid, return_trace=False)


def _make_hole_patch(color: int) -> np.ndarray:
    h, w = random.choice(HOLE_FRAME_DIMS)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return _patch_from_mask(mask, color)


def _make_open_patch() -> np.ndarray:
    mask = random.choice(OPEN_SHAPE_LIBRARY).copy()
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    k = random.randrange(4)
    mask = np.rot90(mask, k)
    mask = _crop_mask(mask)
    return _patch_from_mask(mask, ORANGE)


def _touches_foreground(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    mask = patch != BG
    for rr, cc in np.argwhere(mask):
        gr = r0 + int(rr)
        gc = c0 + int(cc)
        if grid[gr, gc] != BG:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = gr + dr
                nc = gc + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] != BG and not (dr == 0 and dc == 0):
                        return True
    return False


def _place_input_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    if _touches_foreground(grid, patch, r0, c0):
        return False
    _paste(grid, patch, r0, c0)
    return True


def _choose_hole_corners(count: int) -> list[str]:
    if count == 2:
        options = [
            ("TL", "TR"),
            ("BL", "BR"),
            ("TL", "BL"),
            ("TR", "BR"),
            ("TL", "BR"),
            ("TR", "BL"),
        ]
        weights = [4, 4, 2, 2, 1, 1]
        return list(random.choices(options, weights=weights, k=1)[0])
    return random.sample(["TL", "TR", "BL", "BR"], count)


def _hole_candidates(corner_name: str, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch.shape
    rows = range(1, GRID_SIZE - h)
    cols = range(1, GRID_SIZE - w)
    out = []
    for r in rows:
        for c in cols:
            if corner_name == "TL" and (r < 4 or c < 4):
                continue
            if corner_name == "TR" and (r < 4 or c + w > GRID_SIZE - 4):
                continue
            if corner_name == "BL" and (r + h > GRID_SIZE - 4 or c < 4):
                continue
            if corner_name == "BR" and (r + h > GRID_SIZE - 4 or c + w > GRID_SIZE - 4):
                continue
            out.append((r, c))
    random.shuffle(out)
    return out


def _pair_positions(side: str, far_patch: np.ndarray, near_patch: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    fh, fw = far_patch.shape
    nh, nw = near_patch.shape
    fr = CENTER[0] - fh // 2
    nr = CENTER[0] - nh // 2
    pairs = []
    if side == "left":
        far_cols = range(1, GRID_SIZE - fw)
        near_cols = range(1, GRID_SIZE - nw)
        for fc in far_cols:
            if _bbox_center2_from_top_left(fr, fc, far_patch.shape)[1] >= 2 * CENTER[1]:
                continue
            for nc in near_cols:
                if _bbox_center2_from_top_left(nr, nc, near_patch.shape)[1] >= 2 * CENTER[1]:
                    continue
                if fc + fw + 1 > nc:
                    continue
                if _center_distance(_bbox_center2_from_top_left(fr, fc, far_patch.shape), CENTER) <= _center_distance(
                    _bbox_center2_from_top_left(nr, nc, near_patch.shape), CENTER
                ):
                    continue
                pairs.append(((fr, fc), (nr, nc)))
    else:
        far_cols = range(1, GRID_SIZE - fw)
        near_cols = range(1, GRID_SIZE - nw)
        for fc in far_cols:
            if _bbox_center2_from_top_left(fr, fc, far_patch.shape)[1] <= 2 * CENTER[1]:
                continue
            for nc in near_cols:
                if _bbox_center2_from_top_left(nr, nc, near_patch.shape)[1] <= 2 * CENTER[1]:
                    continue
                if nc + nw + 1 > fc:
                    continue
                if _center_distance(_bbox_center2_from_top_left(fr, fc, far_patch.shape), CENTER) <= _center_distance(
                    _bbox_center2_from_top_left(nr, nc, near_patch.shape), CENTER
                ):
                    continue
                pairs.append(((fr, fc), (nr, nc)))
    random.shuffle(pairs)
    return pairs


def _random_solid_candidates(target_corner: str, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch.shape
    candidates = []
    for r in range(1, GRID_SIZE - h):
        for c in range(1, GRID_SIZE - w):
            if abs((r + h // 2) - CENTER[0]) <= 1 and abs((c + w // 2) - CENTER[1]) <= 1:
                continue
            corner_name, _color, tie = _nearest_corner_for_bbox(r, c, patch.shape, (GRID_SIZE, GRID_SIZE))
            if corner_name != target_corner:
                continue
            if tie:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _build_candidate_grid() -> np.ndarray | None:
    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for _name, (r, c), color in CORNER_SPECS:
        grid[r, c] = color

    n_holes, n_solids = random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))

    pair_side = random.choice(("left", "right"))
    pair_patches = [_make_open_patch(), _make_open_patch()]
    pair_positions = _pair_positions(pair_side, pair_patches[0], pair_patches[1])
    placed_pair = None
    for (far_rc, near_rc) in pair_positions:
        temp = grid.copy()
        if not _place_input_patch(temp, pair_patches[0], far_rc[0], far_rc[1]):
            continue
        if not _place_input_patch(temp, pair_patches[1], near_rc[0], near_rc[1]):
            continue
        placed_pair = (temp, far_rc, near_rc)
        break
    if placed_pair is None:
        return None
    grid, _far_rc, _near_rc = placed_pair

    if n_solids == 3:
        third_patch = _make_open_patch()
        pair_target = "TL" if pair_side == "left" else "TR"
        third_target = random.choice([name for name in ("TL", "TR", "BL", "BR") if name != pair_target])
        placed = False
        for r0, c0 in _random_solid_candidates(third_target, third_patch):
            temp = grid.copy()
            if _place_input_patch(temp, third_patch, r0, c0):
                grid = temp
                placed = True
                break
        if not placed:
            return None

    hole_corners = _choose_hole_corners(n_holes)
    for corner_name in hole_corners:
        color = CORNER_BY_NAME[corner_name][1]
        patch = _make_hole_patch(color)
        placed = False
        for r0, c0 in _hole_candidates(corner_name, patch):
            temp = grid.copy()
            if _place_input_patch(temp, patch, r0, c0):
                grid = temp
                placed = True
                break
        if not placed:
            return None

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out, trace = _transform(grid, return_trace=True)
    except Exception:
        return False

    if np.array_equal(grid, out):
        return False

    holes = trace["holes"]
    solids = trace["solids"]
    if not (2 <= len(holes) <= 3 and 2 <= len(solids) <= 3 and 4 <= len(holes) + len(solids) <= 6):
        return False

    if not any(item["steps"] >= 2 for item in holes):
        return False
    if not any(item["steps"] >= 2 for item in solids):
        return False
    if not any(item["tie_break_used"] for item in solids):
        return False
    if not any(item["stop_reason"] == "occupied" for item in solids):
        return False

    hole_targets = {item["target_corner"] for item in holes}
    if len(hole_targets) < 2:
        return False

    if len(solids) >= 3 and len({item["new_color"] for item in solids}) < 2:
        return False

    for _name, (r, c), color in CORNER_SPECS:
        if int(out[r, c]) != color:
            return False

    if np.count_nonzero(grid == ORANGE) == 0:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate_grid()
        if grid is None:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
