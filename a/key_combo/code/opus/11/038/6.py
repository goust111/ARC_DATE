from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PAIR_COLORS = (1, 2, 3)
BODY_COLORS = (4, 5, 6, 7, 8, 9)
SIDE_DELTA = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _connector_pos(patch: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(patch == color)
    assert pts.shape == (1, 2)
    return int(pts[0, 0]), int(pts[0, 1])


def _exposed_sides(shape: tuple[int, int], pos: tuple[int, int]) -> set[str]:
    h, w = shape
    r, c = pos
    sides: set[str] = set()
    if r == 0:
        sides.add("top")
    if r == h - 1:
        sides.add("bottom")
    if c == 0:
        sides.add("left")
    if c == w - 1:
        sides.add("right")
    assert sides
    return sides


def _role_from_sides(sides: set[str]) -> str:
    if sides <= {"top", "left"}:
        return "moving"
    if sides <= {"bottom", "right"}:
        return "stationary"
    raise AssertionError(f"ambiguous connector sides: {sorted(sides)}")


def _depth_from_left(mask: np.ndarray, row: int) -> int:
    if row < 0 or row >= mask.shape[0]:
        return 0
    d = 0
    while d < mask.shape[1] and mask[row, d]:
        d += 1
    return int(d)


def _depth_from_right(mask: np.ndarray, row: int) -> int:
    if row < 0 or row >= mask.shape[0]:
        return 0
    d = 0
    w = mask.shape[1]
    while d < w and mask[row, w - 1 - d]:
        d += 1
    return int(d)


def _depth_from_top(mask: np.ndarray, col: int) -> int:
    if col < 0 or col >= mask.shape[1]:
        return 0
    d = 0
    while d < mask.shape[0] and mask[d, col]:
        d += 1
    return int(d)


def _depth_from_bottom(mask: np.ndarray, col: int) -> int:
    if col < 0 or col >= mask.shape[1]:
        return 0
    d = 0
    h = mask.shape[0]
    while d < h and mask[h - 1 - d, col]:
        d += 1
    return int(d)


def _profile_match(
    stationary_patch: np.ndarray,
    stationary_side: str,
    stationary_conn: tuple[int, int],
    moving_patch: np.ndarray,
    moving_side: str,
    moving_conn: tuple[int, int],
) -> bool:
    stat_mask = stationary_patch != BG
    mov_mask = moving_patch != BG

    if stationary_side == "right" and moving_side == "left":
        row_shift = stationary_conn[0] - moving_conn[0]
        lo = min(0, row_shift)
        hi = max(stat_mask.shape[0] - 1, row_shift + mov_mask.shape[0] - 1)
        total = None
        for rr in range(lo, hi + 1):
            ds = _depth_from_right(stat_mask, rr)
            dm = _depth_from_left(mov_mask, rr - row_shift)
            if ds == 0 and dm == 0:
                return False
            s = ds + dm
            total = s if total is None else total
            if s != total:
                return False
        return total is not None and total >= 2

    if stationary_side == "bottom" and moving_side == "top":
        col_shift = stationary_conn[1] - moving_conn[1]
        lo = min(0, col_shift)
        hi = max(stat_mask.shape[1] - 1, col_shift + mov_mask.shape[1] - 1)
        total = None
        for cc in range(lo, hi + 1):
            ds = _depth_from_bottom(stat_mask, cc)
            dm = _depth_from_top(mov_mask, cc - col_shift)
            if ds == 0 and dm == 0:
                return False
            s = ds + dm
            total = s if total is None else total
            if s != total:
                return False
        return total is not None and total >= 2

    return False


def _fits(shape: tuple[int, int], patch: np.ndarray, origin: tuple[int, int]) -> bool:
    r0, c0 = origin
    h, w = patch.shape
    return 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]


def _paint(out: np.ndarray, patch: np.ndarray, origin: tuple[int, int]) -> None:
    r0, c0 = origin
    h, w = patch.shape
    mask = patch != BG
    view = out[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(mask & (view != BG))
    view[mask] = patch[mask]


def _extract_object(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    marker_coords = [(int(r), int(c)) for r, c in np.argwhere(np.isin(patch, PAIR_COLORS))]
    assert len(marker_coords) == 1
    conn_local = marker_coords[0]
    conn_color = int(patch[conn_local])
    sides = _exposed_sides(patch.shape, conn_local)
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "conn_color": conn_color,
        "conn_local": conn_local,
        "conn_global": (r0 + conn_local[0], c0 + conn_local[1]),
        "sides": sides,
        "role": _role_from_sides(sides),
    }


def _pair_valid_rotations(moving: dict, stationary: dict, grid_shape: tuple[int, int] | None) -> list[dict]:
    stationary_side = "right" if "right" in stationary["sides"] else "bottom"
    needed_side = "left" if stationary_side == "right" else "top"
    sr, sc = stationary["conn_global"]
    dr, dc = SIDE_DELTA[stationary_side]

    valid: list[dict] = []
    for k in range(4):
        rotated = np.rot90(moving["patch"], k)
        conn_local = _connector_pos(rotated, moving["conn_color"])
        sides = _exposed_sides(rotated.shape, conn_local)
        if needed_side not in sides:
            continue
        if not _profile_match(
            stationary["patch"],
            stationary_side,
            stationary["conn_local"],
            rotated,
            needed_side,
            conn_local,
        ):
            continue
        origin = (sr + dr - conn_local[0], sc + dc - conn_local[1])
        if grid_shape is not None and not _fits(grid_shape, rotated, origin):
            continue
        valid.append(
            {
                "rotation": k,
                "patch": rotated,
                "origin": origin,
                "needed_side": needed_side,
                "conn_local": conn_local,
            }
        )
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """Pair equal-colored connectors, find the unique rotation whose docking-edge profiles are complementary, and stitch each moving piece onto its stationary mate.

    Input structure:
    - The grid has black background and 2-3 connector-color pairs of foreground objects.
    - Each object is one 4-connected component with a single body color and one connector pixel of color 1, 2, or 3.
    - For each pair, one object's connector lies on its top/left boundary set (the moving piece) and the other's lies on its bottom/right boundary (the stationary piece).

    Rule:
    - Group objects by connector color and identify the moving vs stationary object from the connector side.
    - Try the moving object under all four 90-degree rotations; after each rotation, place its connector adjacent to the stationary connector on the stationary connector's outward side.
    - Keep the unique rotation whose docking-edge row/column depth profiles are perfectly complementary, then erase the moving object's old location and paste the rotated piece into the docking position.

    Key insight:
    - Connector color only says which two objects interact; the actual rotation is determined by a one-dimensional complement test on the two docking-edge profiles, not by a fixed preset angle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = [_extract_object(g, comp) for comp in _find_components(g)]
    assert 4 <= len(objects) <= 6

    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        by_color[obj["conn_color"]].append(obj)
    assert 2 <= len(by_color) <= 3
    assert all(len(v) == 2 for v in by_color.values())

    out = g.copy()
    for obj in objects:
        if obj["role"] == "moving":
            rr, cc = obj["coords"][:, 0], obj["coords"][:, 1]
            out[rr, cc] = BG

    placements: list[dict] = []
    for pair in by_color.values():
        moving = [obj for obj in pair if obj["role"] == "moving"]
        stationary = [obj for obj in pair if obj["role"] == "stationary"]
        assert len(moving) == 1 and len(stationary) == 1
        valid = _pair_valid_rotations(moving[0], stationary[0], g.shape)
        assert len(valid) == 1
        placements.append(valid[0])

    for placed in placements:
        _paint(out, placed["patch"], placed["origin"])
    return out


def _build_right_patch(profile: list[int], body_color: int, conn_color: int, conn_row: int) -> np.ndarray:
    h = len(profile)
    w = max(profile)
    patch = np.zeros((h, w), dtype=int)
    for r, depth in enumerate(profile):
        patch[r, w - depth : w] = body_color
    patch[conn_row, w - 1] = conn_color
    return patch


def _build_left_patch(profile: list[int], body_color: int, conn_color: int, conn_row: int) -> np.ndarray:
    h = len(profile)
    w = max(profile)
    patch = np.zeros((h, w), dtype=int)
    for r, depth in enumerate(profile):
        patch[r, :depth] = body_color
    patch[conn_row, 0] = conn_color
    return patch


def _build_bottom_patch(profile: list[int], body_color: int, conn_color: int, conn_col: int) -> np.ndarray:
    w = len(profile)
    h = max(profile)
    patch = np.zeros((h, w), dtype=int)
    for c, depth in enumerate(profile):
        patch[h - depth : h, c] = body_color
    patch[h - 1, conn_col] = conn_color
    return patch


def _build_top_patch(profile: list[int], body_color: int, conn_color: int, conn_col: int) -> np.ndarray:
    w = len(profile)
    h = max(profile)
    patch = np.zeros((h, w), dtype=int)
    for c, depth in enumerate(profile):
        patch[:depth, c] = body_color
    patch[0, conn_col] = conn_color
    return patch


def _shape_info_from_patch(patch: np.ndarray, conn_color: int) -> dict:
    conn = _connector_pos(patch, conn_color)
    sides = _exposed_sides(patch.shape, conn)
    return {"patch": patch, "conn_color": conn_color, "conn_local": conn, "sides": sides}


def _intrinsic_valid_rotations(moving_patch: np.ndarray, conn_color: int, stationary_patch: np.ndarray, stationary_sides: set[str], stationary_conn: tuple[int, int]) -> list[int]:
    moving = _shape_info_from_patch(moving_patch, conn_color)
    stationary = {
        "patch": stationary_patch,
        "conn_color": conn_color,
        "conn_local": stationary_conn,
        "conn_global": stationary_conn,
        "sides": stationary_sides,
    }
    return [item["rotation"] for item in _pair_valid_rotations(moving, stationary, None)]


def _relative_output_bbox(stationary_patch: np.ndarray, stationary_side: str, stationary_conn: tuple[int, int], moved_patch: np.ndarray, moved_conn: tuple[int, int]) -> tuple[int, int, int, int]:
    if stationary_side == "right":
        moved_origin = (stationary_conn[0] - moved_conn[0], stationary_patch.shape[1])
    else:
        moved_origin = (stationary_patch.shape[0], stationary_conn[1] - moved_conn[1])
    r0 = min(0, moved_origin[0])
    c0 = min(0, moved_origin[1])
    r1 = max(stationary_patch.shape[0] - 1, moved_origin[0] + moved_patch.shape[0] - 1)
    c1 = max(stationary_patch.shape[1] - 1, moved_origin[1] + moved_patch.shape[1] - 1)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_rect(existing: list[tuple[int, int, int, int]], canvas: tuple[int, int], size: tuple[int, int], gap: int) -> tuple[int, int] | None:
    h, w = size
    H, W = canvas
    choices = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(choices)
    for r, c in choices:
        rect = (r, c, r + h - 1, c + w - 1)
        if all(_rect_gap_ok(rect, old, gap) for old in existing):
            return r, c
    return None


def _make_horizontal_pair(conn_color: int, stationary_body: int, moving_body: int, target_rotation: int) -> dict | None:
    assert target_rotation in (0, 1)
    for _ in range(200):
        hs = random.randint(3, 4)
        hm = random.randint(3, 4)
        stationary_conn_row = random.randint(1, hs - 2)
        moving_conn_row = 0 if target_rotation == 0 else hm - 1
        row_shift = stationary_conn_row - moving_conn_row
        lo = min(0, row_shift)
        hi = max(hs - 1, row_shift + hm - 1)
        total_depth = random.randint(2, 4)

        stationary_profile = [0] * hs
        moving_profile = [0] * hm
        overlap_splits = 0
        for rr in range(lo, hi + 1):
            s_idx = rr
            m_idx = rr - row_shift
            s_here = 0 <= s_idx < hs
            m_here = 0 <= m_idx < hm
            if s_here and m_here:
                split = random.randint(1, total_depth - 1)
                stationary_profile[s_idx] = split
                moving_profile[m_idx] = total_depth - split
                overlap_splits += split not in (0, total_depth)
            elif s_here:
                stationary_profile[s_idx] = total_depth
            else:
                moving_profile[m_idx] = total_depth

        if overlap_splits < 2:
            continue
        if len(set(stationary_profile)) == 1 and len(set(moving_profile)) == 1:
            continue
        if not (4 <= sum(stationary_profile) <= 8 and 4 <= sum(moving_profile) <= 8):
            continue

        stationary_patch = _build_right_patch(stationary_profile, stationary_body, conn_color, stationary_conn_row)
        moved_patch = _build_left_patch(moving_profile, moving_body, conn_color, moving_conn_row)
        input_patch = np.rot90(moved_patch, -target_rotation)
        input_info = _shape_info_from_patch(input_patch, conn_color)
        if input_info["sides"] != {"top", "left"}:
            continue

        intrinsic = _intrinsic_valid_rotations(
            input_patch,
            conn_color,
            stationary_patch,
            {"right"},
            (stationary_conn_row, stationary_patch.shape[1] - 1),
        )
        if intrinsic != [target_rotation]:
            continue

        candidate_by_side = []
        for k in range(4):
            rotated = np.rot90(input_patch, k)
            sides = _exposed_sides(rotated.shape, _connector_pos(rotated, conn_color))
            if "left" in sides:
                candidate_by_side.append(k)
        if sorted(candidate_by_side) != [0, 1]:
            continue

        return {
            "stationary_patch": stationary_patch,
            "moving_input_patch": input_patch,
            "moving_output_patch": moved_patch,
            "conn_color": conn_color,
            "stationary_side": "right",
            "target_rotation": target_rotation,
            "stationary_conn_local": (stationary_conn_row, stationary_patch.shape[1] - 1),
            "moving_output_conn_local": _connector_pos(moved_patch, conn_color),
        }
    return None


def _make_vertical_pair(conn_color: int, stationary_body: int, moving_body: int, target_rotation: int) -> dict | None:
    assert target_rotation in (0, 3)
    for _ in range(200):
        ws = random.randint(3, 4)
        wm = random.randint(3, 4)
        stationary_conn_col = random.randint(1, ws - 2)
        moving_conn_col = 0 if target_rotation == 0 else wm - 1
        col_shift = stationary_conn_col - moving_conn_col
        lo = min(0, col_shift)
        hi = max(ws - 1, col_shift + wm - 1)
        total_depth = random.randint(2, 4)

        stationary_profile = [0] * ws
        moving_profile = [0] * wm
        overlap_splits = 0
        for cc in range(lo, hi + 1):
            s_idx = cc
            m_idx = cc - col_shift
            s_here = 0 <= s_idx < ws
            m_here = 0 <= m_idx < wm
            if s_here and m_here:
                split = random.randint(1, total_depth - 1)
                stationary_profile[s_idx] = split
                moving_profile[m_idx] = total_depth - split
                overlap_splits += split not in (0, total_depth)
            elif s_here:
                stationary_profile[s_idx] = total_depth
            else:
                moving_profile[m_idx] = total_depth

        if overlap_splits < 2:
            continue
        if len(set(stationary_profile)) == 1 and len(set(moving_profile)) == 1:
            continue
        if not (4 <= sum(stationary_profile) <= 8 and 4 <= sum(moving_profile) <= 8):
            continue

        stationary_patch = _build_bottom_patch(stationary_profile, stationary_body, conn_color, stationary_conn_col)
        moved_patch = _build_top_patch(moving_profile, moving_body, conn_color, moving_conn_col)
        input_patch = np.rot90(moved_patch, -target_rotation)
        input_info = _shape_info_from_patch(input_patch, conn_color)
        if input_info["sides"] != {"top", "left"}:
            continue

        intrinsic = _intrinsic_valid_rotations(
            input_patch,
            conn_color,
            stationary_patch,
            {"bottom"},
            (stationary_patch.shape[0] - 1, stationary_conn_col),
        )
        if intrinsic != [target_rotation]:
            continue

        candidate_by_side = []
        for k in range(4):
            rotated = np.rot90(input_patch, k)
            sides = _exposed_sides(rotated.shape, _connector_pos(rotated, conn_color))
            if "top" in sides:
                candidate_by_side.append(k)
        if sorted(candidate_by_side) != [0, 3]:
            continue

        return {
            "stationary_patch": stationary_patch,
            "moving_input_patch": input_patch,
            "moving_output_patch": moved_patch,
            "conn_color": conn_color,
            "stationary_side": "bottom",
            "target_rotation": target_rotation,
            "stationary_conn_local": (stationary_patch.shape[0] - 1, stationary_conn_col),
            "moving_output_conn_local": _connector_pos(moved_patch, conn_color),
        }
    return None


def _make_pair(conn_color: int, stationary_body: int, moving_body: int, orientation: str, target_rotation: int) -> dict | None:
    if orientation == "H":
        return _make_horizontal_pair(conn_color, stationary_body, moving_body, target_rotation)
    return _make_vertical_pair(conn_color, stationary_body, moving_body, target_rotation)


def _render_output(canvas: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.zeros(canvas, dtype=int)
    for pair in pairs:
        _paint(out, pair["stationary_patch"], pair["stationary_origin"])
        _paint(out, pair["moving_output_patch"], pair["moving_output_origin"])
    return out


def _render_input(canvas: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    inp = np.zeros(canvas, dtype=int)
    for pair in pairs:
        _paint(inp, pair["stationary_patch"], pair["stationary_origin"])
        _paint(inp, pair["moving_input_patch"], pair["moving_input_origin"])
    return inp


def _place_pairs(pairs: list[dict], canvas: tuple[int, int]) -> bool:
    pair_order = list(range(len(pairs)))
    random.shuffle(pair_order)
    occupied_unions: list[tuple[int, int, int, int]] = []
    for idx in pair_order:
        pair = pairs[idx]
        stat_patch = pair["stationary_patch"]
        moved_patch = pair["moving_output_patch"]
        stat_side = pair["stationary_side"]
        rel = _relative_output_bbox(
            stat_patch,
            stat_side,
            pair["stationary_conn_local"],
            moved_patch,
            pair["moving_output_conn_local"],
        )
        rel_h = rel[2] - rel[0] + 1
        rel_w = rel[3] - rel[1] + 1
        top_left = _place_rect(occupied_unions, canvas, (rel_h, rel_w), gap=2)
        if top_left is None:
            return False
        union_r0, union_c0 = top_left
        union_rect = (union_r0, union_c0, union_r0 + rel_h - 1, union_c0 + rel_w - 1)
        occupied_unions.append(union_rect)

        stationary_origin = (union_r0 - rel[0], union_c0 - rel[1])
        if stat_side == "right":
            moving_output_origin = (
                stationary_origin[0] + pair["stationary_conn_local"][0] - pair["moving_output_conn_local"][0],
                stationary_origin[1] + stat_patch.shape[1],
            )
        else:
            moving_output_origin = (
                stationary_origin[0] + stat_patch.shape[0],
                stationary_origin[1] + pair["stationary_conn_local"][1] - pair["moving_output_conn_local"][1],
            )
        pair["stationary_origin"] = stationary_origin
        pair["moving_output_origin"] = moving_output_origin
        pair["output_union_rect"] = union_rect
    return True


def _place_input_movers(pairs: list[dict], canvas: tuple[int, int]) -> bool:
    occupied: list[tuple[int, int, int, int]] = []
    for pair in pairs:
        sr, sc = pair["stationary_origin"]
        sh, sw = pair["stationary_patch"].shape
        occupied.append((sr, sc, sr + sh - 1, sc + sw - 1))

    mover_order = list(range(len(pairs)))
    random.shuffle(mover_order)
    for idx in mover_order:
        pair = pairs[idx]
        mh, mw = pair["moving_input_patch"].shape
        choices = [(r, c) for r in range(canvas[0] - mh + 1) for c in range(canvas[1] - mw + 1)]
        random.shuffle(choices)
        placed = None
        final_r, final_c = pair["moving_output_origin"]
        final_rect = (final_r, final_c, final_r + pair["moving_output_patch"].shape[0] - 1, final_c + pair["moving_output_patch"].shape[1] - 1)
        for r, c in choices:
            rect = (r, c, r + mh - 1, c + mw - 1)
            if not all(_rect_gap_ok(rect, old, 2) for old in occupied):
                continue
            if not _rect_gap_ok(rect, final_rect, 1):
                continue
            placed = (r, c)
            break
        if placed is None:
            return False
        pair["moving_input_origin"] = placed
        occupied.append((placed[0], placed[1], placed[0] + mh - 1, placed[1] + mw - 1))
    return True


def _validate_generated(inp: np.ndarray, out: np.ndarray) -> bool:
    if np.array_equal(inp, out):
        return False
    if not np.array_equal(main(inp), out):
        return False

    objects = [_extract_object(inp, comp) for comp in _find_components(inp)]
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        by_color[obj["conn_color"]].append(obj)

    if not (2 <= len(by_color) <= 3):
        return False
    any_rotated = False
    orientations = set()
    for pair in by_color.values():
        moving = [obj for obj in pair if obj["role"] == "moving"]
        stationary = [obj for obj in pair if obj["role"] == "stationary"]
        if len(moving) != 1 or len(stationary) != 1:
            return False
        stationary_side = "right" if "right" in stationary[0]["sides"] else "bottom"
        orientations.add(stationary_side)
        valids = _pair_valid_rotations(moving[0], stationary[0], None)
        if len(valids) != 1:
            return False
        any_rotated |= valids[0]["rotation"] != 0

        candidate_by_side = []
        for k in range(4):
            rotated = np.rot90(moving[0]["patch"], k)
            sides = _exposed_sides(rotated.shape, _connector_pos(rotated, moving[0]["conn_color"]))
            if stationary_side == "right" and "left" in sides:
                candidate_by_side.append(k)
            if stationary_side == "bottom" and "top" in sides:
                candidate_by_side.append(k)
        if len(candidate_by_side) != 2:
            return False

    return any_rotated and len(orientations) >= 1


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([2, 3])
    canvas = (random.randint(16, 20), random.randint(16, 20))
    body_colors = random.sample(BODY_COLORS, pair_count * 2)

    if pair_count == 2:
        orientations = random.choice([["H", "V"], ["H", "H"], ["V", "V"]])
    else:
        orientations = ["H", "V", random.choice(["H", "V"])]
        random.shuffle(orientations)

    rotation_choices = []
    for orientation in orientations:
        rotation_choices.append(random.choice([0, 1]) if orientation == "H" else random.choice([0, 3]))
    if all(k == 0 for k in rotation_choices):
        idx = random.randrange(pair_count)
        rotation_choices[idx] = 1 if orientations[idx] == "H" else 3

    pairs: list[dict] = []
    for idx in range(pair_count):
        pair = _make_pair(
            PAIR_COLORS[idx],
            body_colors[2 * idx],
            body_colors[2 * idx + 1],
            orientations[idx],
            rotation_choices[idx],
        )
        if pair is None:
            return None
        pairs.append(pair)

    if not _place_pairs(pairs, canvas):
        return None
    if not _place_input_movers(pairs, canvas):
        return None

    out = _render_output(canvas, pairs)
    inp = _render_input(canvas, pairs)
    if not _validate_generated(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid rotating complementary docking example")
