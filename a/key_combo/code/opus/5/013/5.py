from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

try:
    from arc_utils import find_components, get_bg, neighbors4
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_fallback", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
    neighbors4 = _ARC_UTILS.neighbors4


BG = 0
WALL = 5
SEED = 9
ARROW_COLORS = (1, 2, 3)
SIDES = ("top", "right", "bottom", "left")
DIR_TO_SIDE = {(-1, 0): "top", (0, 1): "right", (1, 0): "bottom", (0, -1): "left"}
SIDE_TO_DIR = {v: k for k, v in DIR_TO_SIDE.items()}
OUTWARD_BY_ZONE = {
    "UL": ("top", "left"),
    "UR": ("top", "right"),
    "LL": ("bottom", "left"),
    "LR": ("bottom", "right"),
}
OTHER_ZONE_POS = {
    "UL": ((0, 0), (0, 1), (1, 0)),
    "UR": ((0, 1), (0, 0), (1, 1)),
    "LL": ((1, 0), (0, 0), (1, 1)),
    "LR": ((1, 1), (0, 1), (1, 0)),
}
GEN_RETRIES = 400


def _arrow_patch(direction: tuple[int, int], color: int) -> np.ndarray:
    if direction == (-1, 0):
        arr = np.zeros((4, 3), dtype=int)
        arr[:, 1] = color
        arr[1, :] = color
        return arr
    if direction == (1, 0):
        arr = np.zeros((4, 3), dtype=int)
        arr[:, 1] = color
        arr[2, :] = color
        return arr
    if direction == (0, -1):
        arr = np.zeros((3, 4), dtype=int)
        arr[1, :] = color
        arr[:, 1] = color
        return arr
    if direction == (0, 1):
        arr = np.zeros((3, 4), dtype=int)
        arr[1, :] = color
        arr[:, 2] = color
        return arr
    raise ValueError(direction)


def _local_seed_region(mask: np.ndarray, seed_local: tuple[int, int]) -> set[tuple[int, int]]:
    sr, sc = seed_local
    seen = {(int(sr), int(sc))}
    stack = [(int(sr), int(sc))]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                stack.append((int(nr), int(nc)))
    return seen


def _openings_from_seed_region(
    grid: np.ndarray, bbox: tuple[int, int, int, int], seed_pos: tuple[int, int]
) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    sr, sc = seed_pos
    region = _local_seed_region(sub != WALL, (sr - r0, sc - c0))
    openings: dict[str, list[tuple[int, int]]] = {}
    top = [(r0 + r, c0 + c) for r, c in region if r == 0]
    right = [(r0 + r, c0 + c) for r, c in region if c == sub.shape[1] - 1]
    bottom = [(r0 + r, c0 + c) for r, c in region if r == sub.shape[0] - 1]
    left = [(r0 + r, c0 + c) for r, c in region if c == 0]
    if top:
        openings["top"] = sorted(top)
    if right:
        openings["right"] = sorted(right)
    if bottom:
        openings["bottom"] = sorted(bottom)
    if left:
        openings["left"] = sorted(left)
    return openings


def _decode_arrow_direction(comp: dict) -> tuple[int, int]:
    coords = comp["coords"]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    local = coords - np.array([[r0, c0]], dtype=int)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if (h, w) == (4, 3):
        row_counts = [int(np.sum(local[:, 0] == rr)) for rr in range(4)]
        col_counts = [int(np.sum(local[:, 1] == cc)) for cc in range(3)]
        cross_rows = [rr for rr, cnt in enumerate(row_counts) if cnt == 3]
        assert col_counts == [1, 4, 1]
        assert len(cross_rows) == 1 and cross_rows[0] in (1, 2)
        return (-1, 0) if cross_rows[0] == 1 else (1, 0)
    if (h, w) == (3, 4):
        row_counts = [int(np.sum(local[:, 0] == rr)) for rr in range(3)]
        col_counts = [int(np.sum(local[:, 1] == cc)) for cc in range(4)]
        cross_cols = [cc for cc, cnt in enumerate(col_counts) if cnt == 3]
        assert row_counts == [1, 4, 1]
        assert len(cross_cols) == 1 and cross_cols[0] in (1, 2)
        return (0, -1) if cross_cols[0] == 1 else (0, 1)
    raise AssertionError(f"unexpected arrow bbox: {(h, w)}")


def _sqdist(a: np.ndarray, b: np.ndarray) -> int:
    dr = a[:, None, 0] - b[None, :, 0]
    dc = a[:, None, 1] - b[None, :, 1]
    return int(np.min(dr * dr + dc * dc))


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    grey_comps: list[dict] = []
    arrows: list[dict] = []
    seeds: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        if color == WALL:
            grey_comps.append({"coords": comp["coords"], "bbox": tuple(map(int, comp["bbox"]))})
        elif area == 6:
            arrows.append(
                {
                    "coords": comp["coords"],
                    "bbox": tuple(map(int, comp["bbox"])),
                    "color": color,
                    "direction": _decode_arrow_direction(comp),
                }
            )
        elif area == 1:
            r, c = map(int, comp["coords"][0])
            seeds.append({"pos": (r, c), "color": color})

    assert 2 <= len(seeds) <= 3
    assert len(arrows) == len(seeds)
    assert len(grey_comps) >= len(seeds)
    walls: list[dict] = []
    used_grey: set[int] = set()
    for seed in seeds:
        sr, sc = seed["pos"]
        member_idxs = []
        for idx, comp in enumerate(grey_comps):
            d = _sqdist(np.array([[sr, sc]], dtype=int), comp["coords"])
            if d <= 16:
                member_idxs.append(idx)
        assert member_idxs
        for idx in member_idxs:
            assert idx not in used_grey
        used_grey.update(member_idxs)

        coords = np.vstack([grey_comps[idx]["coords"] for idx in member_idxs])
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        temp = np.zeros_like(grid)
        temp[coords[:, 0], coords[:, 1]] = WALL
        openings = _openings_from_seed_region(temp, (r0, c0, r1, c1), seed["pos"])
        assert len(openings) == 2
        walls.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "seed": seed["pos"],
                "openings": openings,
            }
        )

    assert len(walls) == len(seeds) == len(arrows)
    assert used_grey == set(range(len(grey_comps)))

    for arrow in arrows:
        distances = [_sqdist(arrow["coords"], wall["coords"]) for wall in walls]
        best = min(distances)
        assert distances.count(best) == 1
        arrow["wall_idx"] = distances.index(best)

    paired = [arrow["wall_idx"] for arrow in arrows]
    assert sorted(paired) == list(range(len(walls)))
    return walls, arrows


def _fill_from_wall(
    grid: np.ndarray,
    walls: list[dict],
    wall_idx: int,
    exit_side: str,
) -> np.ndarray:
    wall = walls[wall_idx]
    assert exit_side in wall["openings"]
    r0, c0, r1, c1 = wall["bbox"]
    inside_bbox = np.zeros_like(grid, dtype=bool)
    inside_bbox[r0 : r1 + 1, c0 : c1 + 1] = True
    side_mask = np.zeros_like(grid, dtype=bool)
    if exit_side == "top":
        side_mask[:r0, :] = True
    elif exit_side == "bottom":
        side_mask[r1 + 1 :, :] = True
    elif exit_side == "left":
        side_mask[:, :c0] = True
    else:
        side_mask[:, c1 + 1 :] = True

    passable = (grid == BG) & (inside_bbox | side_mask)
    sr, sc = wall["seed"]
    passable[sr, sc] = True

    for idx, other in enumerate(walls):
        for side, cells in other["openings"].items():
            if idx == wall_idx and side == exit_side:
                continue
            for r, c in cells:
                passable[r, c] = False

    seen = np.zeros_like(grid, dtype=bool)
    stack = [wall["seed"]]
    seen[sr, sc] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each seeded enclosure and let it flood outward only through the arrow-selected opening.

    Input structure:
    - Black background contains 2 or 3 grey wall structures around single-pixel seeds; each local structure occupies a small square neighborhood and leaves two visible exit sides from the seed region.
    - A wall structure may be split into nearby grey fragments, but each seed has exactly one associated local wall structure and one nearby colored 6-pixel arrow sprite.
    - Arrow colors differ from the grey walls and act as the eventual fill colors.

    Rule:
    - Decode each arrow direction from which end of its shaft the crossbar is closer to.
    - Pair each arrow with the unique nearest grey wall structure around a seed.
    - For that wall, keep only the opening on the side matching the arrow direction; treat all other wall openings as temporarily blocked.
    - Starting from the enclosed seed, 4-neighbor flood through background cells inside the local wall box and then continue only into the selected outside half-plane, recoloring the reached region, including the seed, with the arrow color.
    - Keep walls and arrows unchanged.

    Key insight:
    The wall geometry alone is ambiguous because each enclosure has several exits; the arrow's internal geometry selects the one usable opening, while its color supplies the paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    walls, arrows = _parse_scene(g)
    out = g.copy()
    painted = np.zeros_like(g, dtype=bool)

    for arrow in arrows:
        wall_idx = int(arrow["wall_idx"])
        exit_side = DIR_TO_SIDE[arrow["direction"]]
        fill_mask = _fill_from_wall(g, walls, wall_idx, exit_side)
        assert not np.any(painted & fill_mask)
        out[fill_mask] = int(arrow["color"])
        painted |= fill_mask

    return out


def _zone_origin(zone: str, shape: tuple[int, int], size: int) -> tuple[int, int] | None:
    h, w = shape
    ranges = {
        "UL": ((3, 5), (3, 5)),
        "UR": ((3, 5), (w - size - 5, w - size - 3)),
        "LL": ((h - size - 5, h - size - 3), (3, 5)),
        "LR": ((h - size - 5, h - size - 3), (w - size - 5, w - size - 3)),
    }
    (rlo, rhi), (clo, chi) = ranges[zone]
    if rlo > rhi or clo > chi:
        return None
    return random.randint(rlo, rhi), random.randint(clo, chi)


def _seed_pos(size: int) -> tuple[int, int]:
    return random.randint(2, size - 3), random.randint(2, size - 3)


def _opening_offset(size: int) -> int:
    return random.randint(2, size - 3)


def _wall_patch(size: int, openings: dict[str, int]) -> np.ndarray:
    patch = np.full((size, size), WALL, dtype=int)
    patch[2 : size - 2, 2 : size - 2] = BG
    for side, offset in openings.items():
        if side == "top":
            patch[0:3, offset] = BG
        elif side == "bottom":
            patch[size - 3 : size, offset] = BG
        elif side == "left":
            patch[offset, 0:3] = BG
        elif side == "right":
            patch[offset, size - 3 : size] = BG
    return patch


def _make_wall_patch(size: int, side_a: str, side_b: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(100):
        openings = {side_a: _opening_offset(size), side_b: _opening_offset(size)}
        patch = _wall_patch(size, openings)
        seed_local = _seed_pos(size)
        if int(patch[seed_local]) != BG:
            continue
        openings_found = _openings_from_seed_region(
            np.where(patch == WALL, WALL, BG), (0, 0, size - 1, size - 1), seed_local
        )
        if set(openings_found) == {side_a, side_b}:
            return patch, seed_local
    return None


def _candidate_arrow_positions(
    grid_shape: tuple[int, int],
    wall_bbox: tuple[int, int, int, int],
    arrow_shape: tuple[int, int],
    placement_side: str,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = wall_bbox
    ah, aw = arrow_shape
    positions: list[tuple[int, int]] = []
    if placement_side == "top":
        top = r0 - ah - 1
        if top < 0:
            return positions
        lo = max(0, c0 - aw + 2)
        hi = min(w - aw, c1 - 1)
        for left in range(lo, hi + 1):
            positions.append((top, left))
    elif placement_side == "bottom":
        top = r1 + 2
        if top + ah > h:
            return positions
        lo = max(0, c0 - aw + 2)
        hi = min(w - aw, c1 - 1)
        for left in range(lo, hi + 1):
            positions.append((top, left))
    elif placement_side == "left":
        left = c0 - aw - 1
        if left < 0:
            return positions
        lo = max(0, r0 - ah + 2)
        hi = min(h - ah, r1 - 1)
        for top in range(lo, hi + 1):
            positions.append((top, left))
    else:
        left = c1 + 2
        if left + aw > w:
            return positions
        lo = max(0, r0 - ah + 2)
        hi = min(h - ah, r1 - 1)
        for top in range(lo, hi + 1):
            positions.append((top, left))
    random.shuffle(positions)
    return positions


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    vr = max(0, max(a[0], b[0]) - min(a[2], b[2]) - 1)
    vc = max(0, max(a[1], b[1]) - min(a[3], b[3]) - 1)
    if vr == 0:
        return vc
    if vc == 0:
        return vr
    return max(vr, vc)


def _occupy(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    assert not np.any(mask & (sub != BG))
    sub[mask] = patch[mask]


def _render_case(case: dict) -> np.ndarray:
    grid = np.zeros(case["shape"], dtype=int)
    for wall in case["walls"]:
        patch = wall["patch"]
        _occupy(grid, patch, int(wall["top"]), int(wall["left"]))
        sr, sc = wall["seed_local"]
        grid[wall["top"] + sr, wall["left"] + sc] = SEED
    for arrow in case["arrows"]:
        patch = _arrow_patch(arrow["direction"], arrow["color"])
        _occupy(grid, patch, int(arrow["top"]), int(arrow["left"]))
    return grid


def _build_case() -> dict | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    n_walls = random.choice((2, 3))
    zones = random.sample(("UL", "UR", "LL", "LR"), n_walls)
    colors = random.sample(list(ARROW_COLORS), n_walls)
    walls = []

    for zone in zones:
        size = 7
        origin = _zone_origin(zone, (h, w), size)
        if origin is None:
            return None
        top, left = origin
        bbox = (top, left, top + size - 1, left + size - 1)
        if any(_rect_gap(bbox, other["bbox"]) < 4 for other in walls):
            return None
        selected_side = random.choice(OUTWARD_BY_ZONE[zone])
        other_side = random.choice([side for side in SIDES if side != selected_side])
        made = _make_wall_patch(size, selected_side, other_side)
        if made is None:
            return None
        patch, seed_local = made
        walls.append(
            {
                "zone": zone,
                "size": size,
                "top": top,
                "left": left,
                "bbox": bbox,
                "selected_side": selected_side,
                "other_side": other_side,
                "patch": patch,
                "seed_local": seed_local,
            }
        )

    arrows = []
    mismatch_count = 0
    arrow_boxes = []
    for wall, color in zip(walls, colors):
        direction = SIDE_TO_DIR[wall["selected_side"]]
        patch = _arrow_patch(direction, color)
        placement_sides = list(SIDES)
        random.shuffle(placement_sides)
        placement_sides.sort(key=lambda side: side == wall["selected_side"])
        placed = None
        for placement_side in placement_sides:
            candidates = _candidate_arrow_positions((h, w), wall["bbox"], patch.shape, placement_side)
            for top, left in candidates:
                bbox = (top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1)
                if any(_rect_gap(bbox, other) < 1 for other in arrow_boxes):
                    continue
                if any(_rect_gap(bbox, other["bbox"]) < 1 for other in walls if other is not wall):
                    continue
                arrow_coords = np.argwhere(patch != BG) + np.array([[top, left]], dtype=int)
                distances = []
                for other in walls:
                    wall_coords = np.argwhere(other["patch"] == WALL)
                    wall_coords = wall_coords + np.array([[other["top"], other["left"]]], dtype=int)
                    distances.append(_sqdist(arrow_coords, wall_coords))
                best = min(distances)
                if distances.count(best) != 1 or distances.index(best) != walls.index(wall):
                    continue
                placed = {
                    "top": top,
                    "left": left,
                    "bbox": bbox,
                    "direction": direction,
                    "color": color,
                    "placement_side": placement_side,
                }
                break
            if placed is not None:
                break
        if placed is None:
            return None
        mismatch_count += int(placed["placement_side"] != wall["selected_side"])
        arrow_boxes.append(placed["bbox"])
        arrows.append(placed)

    if mismatch_count == 0:
        return None
    return {"shape": (h, w), "walls": walls, "arrows": arrows}


def _semantic_checks(grid: np.ndarray) -> bool:
    walls, arrows = _parse_scene(grid)
    side_by_wall = {arrow["wall_idx"]: DIR_TO_SIDE[arrow["direction"]] for arrow in arrows}
    fill_masks = []
    for idx, wall in enumerate(walls):
        chosen = side_by_wall[idx]
        if chosen not in wall["openings"]:
            return False
        alts = [side for side in wall["openings"] if side != chosen]
        if not alts:
            return False
        good = _fill_from_wall(grid, walls, idx, chosen)
        if not any(not np.array_equal(good, _fill_from_wall(grid, walls, idx, alt)) for alt in alts):
            return False
        fill_masks.append(good)

    for i in range(len(fill_masks)):
        for j in range(i + 1, len(fill_masks)):
            if np.any(fill_masks[i] & fill_masks[j]):
                return False

    out = main(grid)
    if out.shape != grid.shape:
        return False

    for arrow in arrows:
        color = int(arrow["color"])
        idx = int(arrow["wall_idx"])
        fill_mask = fill_masks[idx]
        if not np.all(out[fill_mask] == color):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is None:
            continue
        grid = _render_case(case)
        try:
            if _semantic_checks(grid):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid arrow-guided flood-fill puzzle")


if __name__ == "__main__":
    p = Path(__file__)
    mod = SourceFileLoader("arc_task_self", str(p)).load_module()
    for _ in range(5):
        x = mod.generate_input()
        y = mod.main(x)
        print(x.shape, y.shape, np.unique(x), np.unique(y))
