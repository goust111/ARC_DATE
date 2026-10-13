from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
HOLE_AREAS = [1, 2, 3, 4, 6]
PALETTE = list(range(10))

GRID_MIN = 22
GRID_MAX = 30
MARGIN = 1
GAP = 1
MAX_GEN_ATTEMPTS = 500
MAX_PLACE_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _sign(x: float) -> int:
    if x > 1e-9:
        return 1
    if x < -1e-9:
        return -1
    return 0


def _interior_holes(local_mask: np.ndarray) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    for hole in _components_from_mask(~local_mask):
        if (
            np.any(hole[:, 0] == 0)
            or np.any(hole[:, 1] == 0)
            or np.any(hole[:, 0] == local_mask.shape[0] - 1)
            or np.any(hole[:, 1] == local_mask.shape[1] - 1)
        ):
            continue
        holes.append(hole)
    return holes


def _hole_signature(local_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int] | None:
    holes = _interior_holes(local_mask)
    if len(holes) != 1:
        return None
    hole = holes[0]
    obj_center = ((local_mask.shape[0] - 1) / 2.0, (local_mask.shape[1] - 1) / 2.0)
    hole_center = (float(hole[:, 0].mean()), float(hole[:, 1].mean()))
    direction = (_sign(hole_center[0] - obj_center[0]), _sign(hole_center[1] - obj_center[1]))
    if direction == (0, 0):
        return None
    return hole, direction, int(hole.shape[0])


def _connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask)
    return len(comps) == 1


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components_from_mask(grid == color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "local_mask": local,
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each hollow object in the direction encoded by its hole offset, using the hole area as step count.

    Input structure:
    - A uniform background contains several disconnected monochrome foreground objects.
    - Some objects enclose exactly one 4-connected background hole inside their own bounding box.
    - The hole sits off-center relative to that object's bounding-box center; other objects have no hole.

    Rule:
    - For each hole-bearing object, compare the hole center with the object's bounding-box center.
    - Take the sign of the row offset and the sign of the column offset to get one of the eight movement directions.
    - Translate the whole object by n cells along that direction, where n is the hole area; leave solid objects unchanged.

    Key insight:
    - The hole is not damage to repair: its offset encodes direction and its area encodes distance, so the object's own interior geometry is the movement instruction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = _find_objects(g, bg)

    moving: list[tuple[int, np.ndarray, np.ndarray]] = []
    to_clear = np.zeros(g.shape, dtype=bool)
    for obj in objects:
        sig = _hole_signature(obj["local_mask"])
        if sig is None:
            continue
        _hole, direction, distance = sig
        shift = np.asarray([direction[0] * distance, direction[1] * distance], dtype=int)
        shifted = obj["coords"] + shift
        assert np.all(shifted[:, 0] >= 0) and np.all(shifted[:, 0] < g.shape[0])
        assert np.all(shifted[:, 1] >= 0) and np.all(shifted[:, 1] < g.shape[1])
        moving.append((int(obj["color"]), obj["coords"], shifted))
        to_clear[obj["coords"][:, 0], obj["coords"][:, 1]] = True

    out = g.copy()
    out[to_clear] = int(bg)
    for color, _coords, shifted in moving:
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == int(bg))
        out[shifted[:, 0], shifted[:, 1]] = int(color)
    return out


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, area + 1):
        if area % h == 0:
            out.append((h, area // h))
    return out


def _padding_pair(sign_value: int) -> tuple[int, int]:
    if sign_value == 0:
        side = random.randint(1, 2)
        return side, side
    small = random.randint(1, 2)
    big = small + random.randint(1, 2)
    return (big, small) if sign_value > 0 else (small, big)


def _add_arm(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        arm_h = random.randint(1, 2)
        arm_w = random.randint(max(1, w // 2), w)
        start = random.randint(0, w - arm_w)
        new = np.zeros((h + arm_h, w), dtype=bool)
        if side == "top":
            new[arm_h:, :] = mask
            new[:arm_h, start : start + arm_w] = True
        else:
            new[:h, :] = mask
            new[h:, start : start + arm_w] = True
    else:
        arm_w = random.randint(1, 2)
        arm_h = random.randint(max(1, h // 2), h)
        start = random.randint(0, h - arm_h)
        new = np.zeros((h, w + arm_w), dtype=bool)
        if side == "left":
            new[:, arm_w:] = mask
            new[start : start + arm_h, :arm_w] = True
        else:
            new[:, :w] = mask
            new[start : start + arm_h, w:] = True
    return _crop_mask(new)


def _build_hollow_mask(direction: tuple[int, int], area: int) -> np.ndarray:
    for _ in range(100):
        hole_h, hole_w = random.choice(_factor_pairs(area))
        if random.random() < 0.5:
            hole_h, hole_w = hole_w, hole_h
        top, bottom = _padding_pair(direction[0])
        left, right = _padding_pair(direction[1])

        mask = np.ones((top + hole_h + bottom, left + hole_w + right), dtype=bool)
        mask[top : top + hole_h, left : left + hole_w] = False
        if random.random() < 0.8:
            mask = _add_arm(mask)
        if random.random() < 0.35:
            mask = _add_arm(mask)

        sig = _hole_signature(mask)
        if sig is None:
            continue
        _hole, actual_dir, actual_area = sig
        if actual_dir != direction or actual_area != area:
            continue
        if not _connected(mask):
            continue
        return mask
    raise RuntimeError("failed to build hollow mask")


def _build_solid_mask() -> np.ndarray:
    for _ in range(120):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        target = random.randint(5, min(12, h * w - 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True
        mask = _crop_mask(mask)
        if not _connected(mask):
            continue
        if _interior_holes(mask):
            continue
        return mask
    raise RuntimeError("failed to build solid mask")


def _coords_at(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask).copy()
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _can_place(coords: np.ndarray, occupied: np.ndarray, gap: int) -> bool:
    h, w = occupied.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - gap)
        r1 = min(h, r + gap + 1)
        c0 = max(0, c - gap)
        c1 = min(w, c + gap + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _place_object(
    obj: dict,
    input_occ: np.ndarray,
    output_occ: np.ndarray,
    h: int,
    w: int,
) -> dict | None:
    mask = obj["mask"]
    mh, mw = mask.shape
    dr, dc = obj["move"]

    if obj["moving"]:
        row_lo = MARGIN + max(0, dr)
        row_hi = h - MARGIN - mh + min(0, dr)
        col_lo = MARGIN + max(0, dc)
        col_hi = w - MARGIN - mw + min(0, dc)
        if row_lo > row_hi or col_lo > col_hi:
            return None
        for _ in range(MAX_PLACE_ATTEMPTS):
            out_top = random.randint(row_lo, row_hi)
            out_left = random.randint(col_lo, col_hi)
            in_top = out_top - dr
            in_left = out_left - dc
            out_coords = _coords_at(mask, out_top, out_left)
            in_coords = _coords_at(mask, in_top, in_left)
            if not _can_place(out_coords, output_occ, GAP):
                continue
            if not _can_place(in_coords, input_occ, GAP):
                continue
            output_occ[out_coords[:, 0], out_coords[:, 1]] = True
            input_occ[in_coords[:, 0], in_coords[:, 1]] = True
            return {
                "color": obj["color"],
                "mask": mask,
                "input_top": in_top,
                "input_left": in_left,
                "output_top": out_top,
                "output_left": out_left,
                "moving": True,
                "move": (dr, dc),
            }
        return None

    row_lo = MARGIN
    row_hi = h - MARGIN - mh
    col_lo = MARGIN
    col_hi = w - MARGIN - mw
    if row_lo > row_hi or col_lo > col_hi:
        return None
    for _ in range(MAX_PLACE_ATTEMPTS):
        top = random.randint(row_lo, row_hi)
        left = random.randint(col_lo, col_hi)
        coords = _coords_at(mask, top, left)
        if not _can_place(coords, input_occ, GAP):
            continue
        if not _can_place(coords, output_occ, GAP):
            continue
        input_occ[coords[:, 0], coords[:, 1]] = True
        output_occ[coords[:, 0], coords[:, 1]] = True
        return {
            "color": obj["color"],
            "mask": mask,
            "input_top": top,
            "input_left": left,
            "output_top": top,
            "output_left": left,
            "moving": False,
            "move": (0, 0),
        }
    return None


def _render_scene(shape: tuple[int, int], bg: int, placements: list[dict], *, use_output: bool) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    top_key = "output_top" if use_output else "input_top"
    left_key = "output_left" if use_output else "input_left"
    for item in placements:
        coords = _coords_at(item["mask"], item[top_key], item[left_key])
        grid[coords[:, 0], coords[:, 1]] = int(item["color"])
    return grid


def _describe_scene(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    desc = []
    for obj in _find_objects(grid, bg):
        sig = _hole_signature(obj["local_mask"])
        entry = {
            "color": obj["color"],
            "coords_set": frozenset((int(r), int(c)) for r, c in obj["coords"]),
            "local_mask": obj["local_mask"],
            "moving": sig is not None,
            "direction": None,
            "distance": None,
        }
        if sig is not None:
            _hole, direction, distance = sig
            entry["direction"] = direction
            entry["distance"] = distance
        desc.append(entry)
    return desc


def _semantic_validate(inp: np.ndarray, expected_out: np.ndarray) -> None:
    produced = main(inp)
    assert np.array_equal(produced, expected_out)
    assert not np.array_equal(inp, expected_out)

    inp_desc = {item["color"]: item for item in _describe_scene(inp)}
    out_desc = {item["color"]: item for item in _describe_scene(expected_out)}
    assert set(inp_desc) == set(out_desc)

    hollow = [item for item in inp_desc.values() if item["moving"]]
    solid = [item for item in inp_desc.values() if not item["moving"]]
    assert len(hollow) >= 3
    assert len(solid) >= 1
    assert any((d["direction"][0] == 0) ^ (d["direction"][1] == 0) for d in hollow)
    assert any(d["direction"][0] != 0 and d["direction"][1] != 0 for d in hollow)
    assert any(
        a["distance"] == b["distance"] and a["direction"] != b["direction"]
        for i, a in enumerate(hollow)
        for b in hollow[i + 1 :]
    )
    assert any(
        a["direction"] == b["direction"] and a["distance"] != b["distance"]
        for i, a in enumerate(hollow)
        for b in hollow[i + 1 :]
    )

    for color, before in inp_desc.items():
        after = out_desc[color]
        if before["moving"]:
            dr = before["direction"][0] * before["distance"]
            dc = before["direction"][1] * before["distance"]
            shifted = frozenset((r + dr, c + dc) for r, c in before["coords_set"])
            assert shifted == after["coords_set"]
            assert after["moving"]
            assert after["direction"] == before["direction"]
            assert after["distance"] == before["distance"]
        else:
            assert before["coords_set"] == after["coords_set"]
            assert not after["moving"]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)
        bg = random.choice(PALETTE)

        cardinal_dir = random.choice(CARDINAL_DIRS)
        diagonal_dir = random.choice(DIAGONAL_DIRS)
        base_area = random.choice(HOLE_AREAS)
        alt_area = random.choice([a for a in HOLE_AREAS if a != base_area])

        moving_specs = [
            {"mask": _build_hollow_mask(cardinal_dir, base_area), "move": (cardinal_dir[0] * base_area, cardinal_dir[1] * base_area), "moving": True},
            {"mask": _build_hollow_mask(diagonal_dir, base_area), "move": (diagonal_dir[0] * base_area, diagonal_dir[1] * base_area), "moving": True},
            {"mask": _build_hollow_mask(cardinal_dir, alt_area), "move": (cardinal_dir[0] * alt_area, cardinal_dir[1] * alt_area), "moving": True},
        ]
        if random.random() < 0.45:
            extra_dir = random.choice(ALL_DIRS)
            extra_area = random.choice(HOLE_AREAS)
            moving_specs.append(
                {
                    "mask": _build_hollow_mask(extra_dir, extra_area),
                    "move": (extra_dir[0] * extra_area, extra_dir[1] * extra_area),
                    "moving": True,
                }
            )

        solid_specs = [{"mask": _build_solid_mask(), "move": (0, 0), "moving": False} for _ in range(random.randint(1, 2))]
        objects = moving_specs + solid_specs

        if len(objects) > len(PALETTE) - 1:
            continue
        colors = random.sample([c for c in PALETTE if c != bg], len(objects))
        for obj, color in zip(objects, colors):
            obj["color"] = int(color)

        input_occ = np.zeros((height, width), dtype=bool)
        output_occ = np.zeros((height, width), dtype=bool)
        placements: list[dict] = []
        order = sorted(objects, key=lambda item: (-int(item["mask"].shape[0] * item["mask"].shape[1]), -int(item["mask"].sum())))
        ok = True
        for obj in order:
            placed = _place_object(obj, input_occ, output_occ, height, width)
            if placed is None:
                ok = False
                break
            placements.append(placed)
        if not ok:
            continue

        inp = _render_scene((height, width), bg, placements, use_output=False)
        expected_out = _render_scene((height, width), bg, placements, use_output=True)
        try:
            _semantic_validate(inp, expected_out)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid input for puzzle 7")
