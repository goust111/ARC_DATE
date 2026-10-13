from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 5
OBJ = 1
GUIDE = 2

BRANCHES = ("left", "right", "top", "bottom")
HORIZONTAL_BRANCHES = {"left", "right"}
VERTICAL_BRANCHES = {"top", "bottom"}

GEN_RETRIES = 500
SHAPE_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _full_bbox(mask: np.ndarray) -> bool:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return (
        rows.size > 0
        and cols.size > 0
        and rows[0] == 0
        and rows[-1] == mask.shape[0] - 1
        and cols[0] == 0
        and cols[-1] == mask.shape[1] - 1
    )


def _connected_random_mask(
    h: int,
    w: int,
    min_area: int,
    max_area: int,
    *,
    require_all_rotations_distinct: bool,
    require_non_180: bool,
) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(SHAPE_RETRIES):
        area = random.randint(min_area, max_area)
        seed = random.choice(cells)
        chosen = {seed}
        frontier = {
            nb
            for nb in _neighbors4(*seed)
            if 0 <= nb[0] < h and 0 <= nb[1] < w
        }
        while len(chosen) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            chosen.add(cell)
            for nb in _neighbors4(*cell):
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in chosen:
                    frontier.add(nb)
        if len(chosen) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if not _full_bbox(mask):
            continue
        if require_all_rotations_distinct and any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if require_non_180 and np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    return None


def _rotation_k(branch: str) -> int:
    if branch == "left":
        return 3
    if branch == "right":
        return 1
    if branch == "top":
        return 2
    if branch == "bottom":
        return 0
    raise ValueError(branch)


def _stamp_rotated(
    out_mask: np.ndarray,
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    branch: str,
) -> np.ndarray:
    rotated = np.rot90(patch, _rotation_k(branch))
    r0, c0, r1, c1 = box
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    h2, w2 = rotated.shape
    dst_r0 = int(np.floor(center_r - (h2 - 1) / 2.0 + 1e-9))
    dst_c0 = int(np.floor(center_c - (w2 - 1) / 2.0 + 1e-9))
    dst_r1 = dst_r0 + h2
    dst_c1 = dst_c0 + w2
    assert 0 <= dst_r0 and dst_r1 <= out_mask.shape[0]
    assert 0 <= dst_c0 and dst_c1 <= out_mask.shape[1]
    for rr, cc in np.argwhere(rotated):
        out_mask[dst_r0 + int(rr), dst_c0 + int(cc)] = True
    return rotated


def _guide_rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    rows = [r for r in range(1, h - 1) if grid[r, 0] == GUIDE and grid[r, w - 1] == GUIDE]
    cols = [c for c in range(1, w - 1) if grid[0, c] == GUIDE and grid[h - 1, c] == GUIDE]
    return rows, cols


def _contact_for_object(
    coords: np.ndarray,
    guide_rows: list[int],
    guide_cols: list[int],
    shape: tuple[int, int],
) -> tuple[str, str]:
    h, w = shape
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        row_to_cols.setdefault(r, []).append(c)
        col_to_rows.setdefault(c, []).append(r)

    candidates: list[tuple[str, str, int]] = []
    for r in guide_rows:
        cols = row_to_cols.get(int(r))
        if not cols:
            continue
        left_gap = min(cols)
        right_gap = (w - 1) - max(cols)
        assert left_gap != right_gap, "horizontal guide contact would be ambiguous"
        side = "left" if left_gap < right_gap else "right"
        candidates.append(("horizontal", side, int(r)))
    for c in guide_cols:
        rows = col_to_rows.get(int(c))
        if not rows:
            continue
        top_gap = min(rows)
        bottom_gap = (h - 1) - max(rows)
        assert top_gap != bottom_gap, "vertical guide contact would be ambiguous"
        side = "top" if top_gap < bottom_gap else "bottom"
        candidates.append(("vertical", side, int(c)))

    assert len(candidates) == 1, "each object must have exactly one guide contact"
    axis, side, _ = candidates[0]
    return axis, side


def _transformed_blue_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    guide_rows, guide_cols = _guide_rows_cols(grid)
    blue = grid == OBJ
    objects = _components(blue)
    assert 3 <= len(objects) <= 4
    out_mask = np.zeros((h, w), dtype=bool)
    for coords in objects:
        box = _bbox(coords)
        patch = _crop_mask(blue, box)
        axis, side = _contact_for_object(coords, guide_rows, guide_cols, grid.shape)
        branch = {
            ("horizontal", "left"): "left",
            ("horizontal", "right"): "right",
            ("vertical", "top"): "top",
            ("vertical", "bottom"): "bottom",
        }[(axis, side)]
        _stamp_rotated(out_mask, patch, box, branch)
    return out_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete border-defined guide lines, use the guide-side contact direction to rotate each blue object, and stamp it back in place.

    Input structure:
    - The background is grey.
    - Blue pixels form 3 to 4 small 4-connected objects, each separated from the others.
    - Red border-marker pairs on opposite sides define full guide rows or guide columns.

    Rule:
    - Complete every left/right marker pair into a full horizontal guide row and every top/bottom pair into a full vertical guide column.
    - Each blue object is crossed by exactly one completed guide along one of its occupied rows or columns; the nearer emitting border on that shared row/column determines the contact side.
    - A horizontal guide contacting from the left rotates the object 90 degrees clockwise, from the right rotates it 90 degrees counterclockwise, a vertical guide from above rotates it 180 degrees, and a vertical guide from below leaves it unchanged.
    - Rotate the cropped object around its bbox center, stamp the rotated shape back around the same center, restore uncovered cells to grey, and keep red guide lines on top of any overlaps.

    Key insight:
    - The blue shapes do not encode their own rotation; the decisive information is the orientation of the external guide and which border side reaches the object first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)
    out[_transformed_blue_mask(g)] = OBJ
    guide_rows, guide_cols = _guide_rows_cols(g)
    for r in guide_rows:
        out[int(r), :] = GUIDE
    for c in guide_cols:
        out[:, int(c)] = GUIDE
    return out


def _zone_candidates(branch: str, obj_h: int, obj_w: int, h: int, w: int) -> list[tuple[int, int]]:
    r_max = h - obj_h - 1
    c_max = w - obj_w - 1
    if r_max < 1 or c_max < 1:
        return []
    mid_r0 = max(1, h // 3 - 1)
    mid_r1 = max(mid_r0, min(r_max, (2 * h) // 3 - obj_h))
    mid_c0 = max(1, w // 3 - 1)
    mid_c1 = max(mid_c0, min(c_max, (2 * w) // 3 - obj_w))
    left_c1 = max(1, min(c_max, w // 4))
    right_c0 = min(c_max, max(1, w - obj_w - 1 - w // 4))
    top_r1 = max(1, min(r_max, h // 4))
    bottom_r0 = min(r_max, max(1, h - obj_h - 1 - h // 4))

    positions = []
    if branch == "left":
        for r0 in range(mid_r0, mid_r1 + 1):
            for c0 in range(1, left_c1 + 1):
                positions.append((r0, c0))
    elif branch == "right":
        for r0 in range(mid_r0, mid_r1 + 1):
            for c0 in range(right_c0, c_max + 1):
                positions.append((r0, c0))
    elif branch == "top":
        for r0 in range(1, top_r1 + 1):
            for c0 in range(mid_c0, mid_c1 + 1):
                positions.append((r0, c0))
    elif branch == "bottom":
        for r0 in range(bottom_r0, r_max + 1):
            for c0 in range(mid_c0, mid_c1 + 1):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _rows_plus_margin(rows: set[int], limit: int) -> set[int]:
    out = set()
    for x in rows:
        for dx in (-1, 0, 1):
            y = x + dx
            if 0 <= y < limit:
                out.add(y)
    return out


def _boxes_far_enough(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> bool:
    r0a, c0a, r1a, c1a = box1
    r0b, c0b, r1b, c1b = box2
    return not (
        max(0, max(r0a, r0b) - min(r1a, r1b) - 1) < 2
        and max(0, max(c0a, c0b) - min(c1a, c1b) - 1) < 2
    )


def _branch_shape(branch: str) -> tuple[np.ndarray, int]:
    if branch in HORIZONTAL_BRANCHES:
        mask = _connected_random_mask(
            3,
            3,
            4,
            7,
            require_all_rotations_distinct=True,
            require_non_180=True,
        )
        assert mask is not None
        rows = [r for r in range(3) if mask[r].any()]
        guide_row = random.choice(rows)
        return mask, guide_row
    obj_h = 2 if random.random() < 0.45 else 3
    mask = _connected_random_mask(
        obj_h,
        3,
        4,
        min(6, obj_h * 3),
        require_all_rotations_distinct=False,
        require_non_180=True,
    )
    assert mask is not None
    cols = [c for c in range(3) if mask[:, c].any()]
    guide_col = random.choice(cols)
    return mask, guide_col


def _side_margin_ok(branch: str, abs_rows: list[int], abs_cols: list[int], h: int, w: int) -> bool:
    if branch == "left":
        return min(abs_cols) + 1 < (w - 1) - max(abs_cols)
    if branch == "right":
        return min(abs_cols) > (w - 1) - max(abs_cols) + 1
    if branch == "top":
        return min(abs_rows) + 1 < (h - 1) - max(abs_rows)
    if branch == "bottom":
        return min(abs_rows) > (h - 1) - max(abs_rows) + 1
    raise ValueError(branch)


def _render_input(placements: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for rec in placements:
        mask = rec["mask"]
        r0, c0 = rec["top_left"]
        for rr, cc in np.argwhere(mask):
            grid[r0 + int(rr), c0 + int(cc)] = OBJ
    for rec in placements:
        if rec["branch"] in HORIZONTAL_BRANCHES:
            r = rec["guide_abs"]
            grid[int(r), 0] = GUIDE
            grid[int(r), w - 1] = GUIDE
        else:
            c = rec["guide_abs"]
            grid[0, int(c)] = GUIDE
            grid[h - 1, int(c)] = GUIDE
    return grid


def _placement_records(h: int, w: int, branches: list[str]) -> list[dict] | None:
    placements: list[dict] = []
    random.shuffle(branches)
    for branch in branches:
        mask, guide_local = _branch_shape(branch)
        obj_h, obj_w = mask.shape
        candidates = _zone_candidates(branch, obj_h, obj_w, h, w)
        if not candidates:
            return None
        placed = False
        for r0, c0 in candidates:
            abs_rows = [r0 + int(rr) for rr, _ in np.argwhere(mask)]
            abs_cols = [c0 + int(cc) for _, cc in np.argwhere(mask)]
            box = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
            if branch in HORIZONTAL_BRANCHES:
                guide_abs = r0 + int(guide_local)
                row_cols = [c0 + int(cc) for cc in np.where(mask[int(guide_local)])[0]]
                if not row_cols:
                    continue
                side_rows = [guide_abs]
                side_cols = row_cols
            else:
                guide_abs = c0 + int(guide_local)
                col_rows = [r0 + int(rr) for rr in np.where(mask[:, int(guide_local)])[0]]
                if not col_rows:
                    continue
                side_rows = col_rows
                side_cols = [guide_abs]
            if not _side_margin_ok(branch, side_rows, side_cols, h, w):
                continue
            ok = True
            candidate_rows = set(abs_rows)
            candidate_cols = set(abs_cols)
            candidate_rows_margin = _rows_plus_margin(candidate_rows, h)
            candidate_cols_margin = _rows_plus_margin(candidate_cols, w)
            for other in placements:
                if not _boxes_far_enough(box, other["box"]):
                    ok = False
                    break
                if branch in HORIZONTAL_BRANCHES and guide_abs in other["rows_margin"]:
                    ok = False
                    break
                if branch in VERTICAL_BRANCHES and guide_abs in other["cols_margin"]:
                    ok = False
                    break
                if other["branch"] in HORIZONTAL_BRANCHES and other["guide_abs"] in candidate_rows_margin:
                    ok = False
                    break
                if other["branch"] in VERTICAL_BRANCHES and other["guide_abs"] in candidate_cols_margin:
                    ok = False
                    break
            if not ok:
                continue
            placements.append(
                {
                    "branch": branch,
                    "mask": mask,
                    "top_left": (r0, c0),
                    "guide_abs": int(guide_abs),
                    "box": box,
                    "rows_margin": candidate_rows_margin,
                    "cols_margin": candidate_cols_margin,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return placements


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    unique = set(map(int, np.unique(grid)))
    assert unique <= {BG, OBJ, GUIDE}
    guide_rows, guide_cols = _guide_rows_cols(grid)
    assert 1 <= len(guide_rows) <= 2
    assert 1 <= len(guide_cols) <= 2
    objects = _components(grid == OBJ)
    assert 3 <= len(objects) <= 4

    contacts = []
    transformed = np.zeros(grid.shape, dtype=bool)
    changed_count = 0
    for coords in objects:
        box = _bbox(coords)
        patch = _crop_mask(grid == OBJ, box)
        axis, side = _contact_for_object(coords, guide_rows, guide_cols, grid.shape)
        contacts.append((axis, side))
        branch = {
            ("horizontal", "left"): "left",
            ("horizontal", "right"): "right",
            ("vertical", "top"): "top",
            ("vertical", "bottom"): "bottom",
        }[(axis, side)]
        before = set(map(tuple, coords.tolist()))
        scratch = np.zeros(grid.shape, dtype=bool)
        rotated = _stamp_rotated(scratch, patch, box, branch)
        after = set(map(tuple, np.argwhere(scratch).tolist()))
        if branch != "bottom":
            assert before != after
            changed_count += 1
        assert not np.any(transformed & scratch)
        transformed |= scratch
        if branch in HORIZONTAL_BRANCHES:
            assert patch.shape == (3, 3)
            assert rotated.shape == (3, 3)
        else:
            assert patch.shape[1] == 3
    assert changed_count >= 2
    assert len(contacts) == len(objects)

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)
    for r in guide_rows:
        assert np.all(out[int(r), :] == GUIDE)
    for c in guide_cols:
        assert np.all(out[:, int(c)] == GUIDE)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(13, 17)
        w = random.randint(13, 17)
        count = 4 if random.random() < 0.7 else 3
        if count == 4:
            branches = list(BRANCHES)
        else:
            branches = random.sample(BRANCHES, 3)
        placements = _placement_records(h, w, branches)
        if placements is None:
            continue
        grid = _render_input(placements, h, w)
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
