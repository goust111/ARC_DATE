from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = [2, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _shape_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _reflect_coords(coords: np.ndarray, axis_kind: str, axis_value: int) -> np.ndarray:
    out = coords.copy()
    if axis_kind == "horizontal":
        out[:, 0] = 2 * int(axis_value) - out[:, 0]
    else:
        out[:, 1] = 2 * int(axis_value) - out[:, 1]
    return out


def _select_anchor(comps: list[dict]) -> dict:
    assert comps, "expected at least one component"
    return min(comps, key=lambda comp: (-int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))


def _axis_from_anchor(anchor: dict) -> tuple[str, int]:
    r0, c0, r1, c1 = anchor["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert width != height, "anchor bbox must have unequal width and height"
    if width > height:
        return "horizontal", (r0 + r1) // 2
    return "vertical", (c0 + c1) // 2


def _scene_analysis(grid: np.ndarray) -> dict:
    comps = _components(grid)
    assert 4 <= len(comps) <= 6, "expected 4 to 6 components"
    colors = [int(comp["color"]) for comp in comps]
    assert len(colors) == len(set(colors)), "components must use distinct colors"
    for comp in comps:
        assert 3 <= int(comp["area"]) <= 10, "component size out of range"

    anchor = _select_anchor(comps)
    anchor_set = {tuple(map(int, rc)) for rc in anchor["coords"]}
    axis_kind, axis_value = _axis_from_anchor(anchor)
    others = [comp for comp in comps if comp is not anchor]
    others.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))

    transformed = []
    painted_small = set()
    small_overlap_exists = False
    for comp in others:
        mapped = _reflect_coords(comp["coords"], axis_kind, axis_value)
        in_bounds = set()
        anchor_hits = set()
        out_count = 0
        for rr, cc in mapped:
            cell = (int(rr), int(cc))
            if 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]:
                in_bounds.add(cell)
                if cell in anchor_set:
                    anchor_hits.add(cell)
            else:
                out_count += 1
        visible = in_bounds - anchor_set
        if visible & painted_small:
            small_overlap_exists = True
        painted_small |= visible
        transformed.append(
            {
                "comp": comp,
                "mapped": mapped,
                "in_bounds": in_bounds,
                "anchor_hits": anchor_hits,
                "visible": visible,
                "out_count": out_count,
            }
        )

    return {
        "components": comps,
        "anchor": anchor,
        "axis_kind": axis_kind,
        "axis_value": axis_value,
        "transformed": transformed,
        "anchor_set": anchor_set,
        "small_overlap_exists": small_overlap_exists,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the largest component fixed, use its bbox to define a mirror axis, and mirror every other component around that axis.

    Input structure:
    - The grid is a black-background field containing 4 to 6 single-color 4-connected components.
    - Exactly one component has strictly more pixels than every other component.
    - The largest component sits near the middle and its bbox is not square, so its longer side uniquely determines the mirror-axis direction.

    Rule:
    - Find the largest 4-connected component; ties are broken by smaller bbox top row, then smaller bbox left column.
    - If that anchor bbox is wider than tall, use the horizontal line through bbox row `(top + bottom) // 2` as the mirror axis; otherwise use the vertical line through bbox column `(left + right) // 2`.
    - Leave the anchor unchanged. Mirror every non-anchor component across that axis, erase its original position, drop mirrored pixels that leave the grid or land on the anchor, and process mirrored components in bbox top-left order so later ones overwrite earlier mirrored pixels.

    Key insight:
    - The largest component is both the selector and the parameter source: its area chooses it as the anchor, its bbox aspect ratio chooses the axis direction, and its occupied cells block mirrored results.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    analysis = _scene_analysis(g)
    anchor = analysis["anchor"]
    axis_kind = analysis["axis_kind"]
    axis_value = analysis["axis_value"]
    anchor_set = analysis["anchor_set"]

    out = np.zeros_like(g)
    coords = anchor["coords"]
    out[coords[:, 0], coords[:, 1]] = int(anchor["color"])

    for item in analysis["transformed"]:
        color = int(item["comp"]["color"])
        for rr, cc in item["mapped"]:
            r = int(rr)
            c = int(cc)
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and (r, c) not in anchor_set:
                out[r, c] = color

    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _mask_from_local(coords: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _random_connected_shape(h: int, w: int, area: int, *, rng: random.Random, attempts: int = 300) -> np.ndarray | None:
    if area < max(h, w) or area >= h * w:
        return None
    for _ in range(attempts):
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue
        coords = np.asarray(sorted(cells), dtype=int)
        if _bbox(coords) != (0, 0, h - 1, w - 1):
            continue
        mask = _mask_from_local(coords, h, w)
        if bool(mask.all()):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return coords
    return None


def _choose_anchor_shape(axis_kind: str, rng: random.Random) -> np.ndarray | None:
    dims = []
    if axis_kind == "horizontal":
        for h in (3, 4):
            for w in range(max(4, h + 1), 7):
                dims.append((h, w))
    else:
        for w in (3, 4):
            for h in range(max(4, w + 1), 7):
                dims.append((h, w))
    rng.shuffle(dims)
    for h, w in dims:
        min_area = max(7, h + w - 1)
        max_area = min(10, h * w - 1)
        if min_area > max_area:
            continue
        areas = list(range(min_area, max_area + 1))
        rng.shuffle(areas)
        for area in areas:
            shape = _random_connected_shape(h, w, area, rng=rng, attempts=300)
            if shape is not None:
                return shape
    return None


def _choose_small_shape(max_area: int, rng: random.Random) -> np.ndarray | None:
    dims = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3)]
    rng.shuffle(dims)
    for h, w in dims:
        min_area = max(3, h + w - 1)
        area_hi = min(max_area, h * w - 1)
        if min_area > area_hi:
            continue
        areas = list(range(min_area, area_hi + 1))
        rng.shuffle(areas)
        for area in areas:
            shape = _random_connected_shape(h, w, area, rng=rng, attempts=250)
            if shape is not None:
                return shape
    return None


def _translate(coords: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return coords + np.array([r0, c0], dtype=int)


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, max(a[0] - b[2] - 1, b[0] - a[2] - 1))
    col_gap = max(0, max(a[1] - b[3] - 1, b[1] - a[3] - 1))
    return row_gap + col_gap


def _candidate_positions(
    local_coords: np.ndarray,
    grid_shape: tuple[int, int],
    occupied: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
    *,
    rng: random.Random,
) -> list[tuple[int, int, np.ndarray]]:
    h = int(local_coords[:, 0].max()) + 1
    w = int(local_coords[:, 1].max()) + 1
    blocked = _dilate4(occupied)
    candidates = []
    for r0 in range(grid_shape[0] - h + 1):
        for c0 in range(grid_shape[1] - w + 1):
            coords = _translate(local_coords, r0, c0)
            if blocked[coords[:, 0], coords[:, 1]].any():
                continue
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _bbox_distance(bbox, anchor_bbox) > 8:
                continue
            candidates.append((r0, c0, coords))
    rng.shuffle(candidates)
    return candidates


def _role_ok(
    coords: np.ndarray,
    axis_kind: str,
    axis_value: int,
    grid_shape: tuple[int, int],
    anchor_set: set[tuple[int, int]],
    *,
    role: str,
    seed_visible: set[tuple[int, int]] | None = None,
    later_than: tuple[int, int] | None = None,
) -> bool:
    mapped = _reflect_coords(coords, axis_kind, axis_value)
    in_bounds = set()
    out_count = 0
    for rr, cc in mapped:
        cell = (int(rr), int(cc))
        if 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]:
            in_bounds.add(cell)
        else:
            out_count += 1
    anchor_hits = in_bounds & anchor_set
    visible = in_bounds - anchor_set

    if role == "anchor_overlap":
        return 1 <= len(anchor_hits) < len(in_bounds) and len(visible) >= 2
    if role == "out_of_bounds":
        return out_count >= 1 and len(visible) >= 2
    if role == "overlap_seed":
        return len(visible) >= 3 and not anchor_hits
    if role == "overlap_later":
        if seed_visible is None or later_than is None:
            return False
        bbox = _bbox(coords)
        if (bbox[0], bbox[1]) <= tuple(map(int, later_than)):
            return False
        overlap = visible & seed_visible
        return 1 <= len(overlap) <= max(2, len(visible) - 1) and len(visible) >= 3
    return len(visible) >= 2


def _place_component(
    local_coords: np.ndarray,
    role: str,
    grid_shape: tuple[int, int],
    occupied: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
    axis_kind: str,
    axis_value: int,
    anchor_set: set[tuple[int, int]],
    *,
    rng: random.Random,
    seed_visible: set[tuple[int, int]] | None = None,
    later_than: tuple[int, int] | None = None,
) -> np.ndarray | None:
    for _r0, _c0, coords in _candidate_positions(local_coords, grid_shape, occupied, anchor_bbox, rng=rng):
        if _role_ok(
            coords,
            axis_kind,
            axis_value,
            grid_shape,
            anchor_set,
            role=role,
            seed_visible=seed_visible,
            later_than=later_than,
        ):
            return coords
    return None


def _render_grid(shape: tuple[int, int], pieces: list[tuple[int, np.ndarray]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, coords in pieces:
        grid[coords[:, 0], coords[:, 1]] = int(color)
    return grid


def _semantic_validate(grid: np.ndarray, *, want_out_of_bounds: bool, want_small_overlap: bool) -> bool:
    try:
        analysis = _scene_analysis(grid)
    except AssertionError:
        return False

    areas = sorted(int(comp["area"]) for comp in analysis["components"])
    if areas[-1] == areas[-2]:
        return False

    h, w = grid.shape
    ar0, ac0, ar1, ac1 = analysis["anchor"]["bbox"]
    anchor_center_r = (ar0 + ar1) / 2.0
    anchor_center_c = (ac0 + ac1) / 2.0
    if not (h * 0.25 <= anchor_center_r <= h * 0.75 and w * 0.25 <= anchor_center_c <= w * 0.75):
        return False

    if not any(item["anchor_hits"] and item["visible"] for item in analysis["transformed"]):
        return False

    if want_out_of_bounds and not any(item["out_count"] > 0 for item in analysis["transformed"]):
        return False

    if want_small_overlap and not analysis["small_overlap_exists"]:
        return False

    if np.array_equal(main(grid), grid):
        return False

    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    axis_kind = rng.choice(["horizontal", "vertical"])
    want_out_of_bounds = rng.random() < 0.5
    want_small_overlap = rng.random() < 0.5

    min_small = 3 + (1 if want_out_of_bounds else 0) + (1 if want_small_overlap else 0)
    small_count = rng.randint(min_small, 5)
    total_components = small_count + 1
    assert 4 <= total_components <= 6

    grid_h = rng.randint(14, 18)
    grid_w = rng.randint(14, 18)
    anchor_local = _choose_anchor_shape(axis_kind, rng)
    if anchor_local is None:
        return None

    anchor_h = int(anchor_local[:, 0].max()) + 1
    anchor_w = int(anchor_local[:, 1].max()) + 1
    anchor_r0 = max(1, min(grid_h - anchor_h - 1, grid_h // 2 - anchor_h // 2 + rng.randint(-1, 1)))
    anchor_c0 = max(1, min(grid_w - anchor_w - 1, grid_w // 2 - anchor_w // 2 + rng.randint(-1, 1)))
    anchor_coords = _translate(anchor_local, anchor_r0, anchor_c0)
    anchor_bbox = _bbox(anchor_coords)
    anchor_area = int(anchor_coords.shape[0])
    anchor_set = {tuple(map(int, rc)) for rc in anchor_coords}
    axis_value = (anchor_bbox[0] + anchor_bbox[2]) // 2 if axis_kind == "horizontal" else (anchor_bbox[1] + anchor_bbox[3]) // 2

    roles = ["anchor_overlap"]
    if want_small_overlap:
        roles += ["overlap_seed", "overlap_later"]
    if want_out_of_bounds:
        roles.append("out_of_bounds")
    while len(roles) < small_count:
        roles.append("generic")
    rng.shuffle(roles[1:])

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[anchor_coords[:, 0], anchor_coords[:, 1]] = True

    pieces: list[tuple[str, np.ndarray]] = []
    seed_visible = None
    seed_topleft = None
    max_small_area = anchor_area - 1

    for role in roles:
        local = _choose_small_shape(max_small_area, rng)
        if local is None:
            return None
        coords = _place_component(
            local,
            role,
            (grid_h, grid_w),
            occupied,
            anchor_bbox,
            axis_kind,
            axis_value,
            anchor_set,
            rng=rng,
            seed_visible=seed_visible,
            later_than=seed_topleft,
        )
        if coords is None:
            return None
        occupied[coords[:, 0], coords[:, 1]] = True
        pieces.append((role, coords))
        if role == "overlap_seed":
            mapped = _reflect_coords(coords, axis_kind, axis_value)
            visible = {
                (int(rr), int(cc))
                for rr, cc in mapped
                if 0 <= int(rr) < grid_h and 0 <= int(cc) < grid_w and (int(rr), int(cc)) not in anchor_set
            }
            seed_visible = visible
            bb = _bbox(coords)
            seed_topleft = (bb[0], bb[1])

    colors = rng.sample(PALETTE, total_components)
    colored_pieces = [(colors[0], anchor_coords)]
    for color, (_role, coords) in zip(colors[1:], pieces):
        colored_pieces.append((color, coords))
    grid = _render_grid((grid_h, grid_w), colored_pieces)

    if not _semantic_validate(grid, want_out_of_bounds=want_out_of_bounds, want_small_overlap=want_small_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 4")
