from __future__ import annotations

import numpy as np


BG = 0
PATTERN_COLOR = 3
WALL_COLORS = (1, 2, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPENINGS = ("up", "down", "left", "right")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _opposite(opening: str) -> str:
    return {"up": "down", "down": "up", "left": "right", "right": "left"}[opening]


def _opening_from_wall(local_wall: np.ndarray) -> str:
    h, w = local_wall.shape
    counts = {
        "up": int(np.count_nonzero(local_wall[0, :])),
        "down": int(np.count_nonzero(local_wall[-1, :])),
        "left": int(np.count_nonzero(local_wall[:, 0])),
        "right": int(np.count_nonzero(local_wall[:, -1])),
    }
    lengths = {"up": w, "down": w, "left": h, "right": h}
    mins = min(counts.values())
    candidates = [side for side, count in counts.items() if count == mins]
    assert len(candidates) == 1, "cup opening must be unique"
    opening = candidates[0]
    assert mins == 2, "opening side must expose only the two corner wall pixels"
    assert counts[_opposite(opening)] == lengths[_opposite(opening)], "cup bottom wall missing"
    if opening in {"up", "down"}:
        assert counts["left"] == h and counts["right"] == h
    else:
        assert counts["up"] == w and counts["down"] == w
    assert not local_wall[1:-1, 1:-1].any(), "cup wall must stay on its bbox border"
    return opening


def _interior_bbox_from_wall_bbox(
    wall_bbox: tuple[int, int, int, int], opening: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = wall_bbox
    if opening == "up":
        return r0, c0 + 1, r1 - 1, c1 - 1
    if opening == "down":
        return r0 + 1, c0 + 1, r1, c1 - 1
    if opening == "left":
        return r0 + 1, c0, r1 - 1, c1 - 1
    return r0 + 1, c0 + 1, r1 - 1, c1


def _mirror_coord(
    r: int,
    c: int,
    interior_bbox: tuple[int, int, int, int],
    opening: str,
) -> tuple[int, int]:
    ir0, ic0, ir1, ic1 = interior_bbox
    if opening in {"left", "right"}:
        return int(r), int(ic0 + ic1 - c)
    return int(ir0 + ir1 - r), int(c)


def _detect_cups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    cups: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BG, PATTERN_COLOR}):
        for coords in _component_coords(g == color):
            wall_bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = wall_bbox
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            shifted = coords - np.array([r0, c0])
            local[shifted[:, 0], shifted[:, 1]] = True
            opening = _opening_from_wall(local)
            interior_bbox = _interior_bbox_from_wall_bbox(wall_bbox, opening)
            ir0, ic0, ir1, ic1 = interior_bbox
            pattern_cells = [
                (int(rr), int(cc))
                for rr, cc in np.argwhere(g[ir0 : ir1 + 1, ic0 : ic1 + 1] == PATTERN_COLOR)
                if True
            ]
            pattern_cells = [(ir0 + rr, ic0 + cc) for rr, cc in pattern_cells]
            cups.append(
                {
                    "color": color,
                    "wall_bbox": wall_bbox,
                    "opening": opening,
                    "interior_bbox": interior_bbox,
                    "pattern_cells": sorted(pattern_cells),
                }
            )
    cups.sort(key=lambda cup: (cup["wall_bbox"][0], cup["wall_bbox"][1], cup["color"]))
    return cups


def _apply_cup_transform(out: np.ndarray, cup: dict) -> tuple[int, int]:
    added_green = 0
    added_fill = 0
    for r, c in cup["pattern_cells"]:
        mr, mc = _mirror_coord(r, c, cup["interior_bbox"], cup["opening"])
        if int(out[mr, mc]) == BG:
            out[mr, mc] = PATTERN_COLOR
            added_green += 1

    ir0, ic0, ir1, ic1 = cup["interior_bbox"]
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            if int(out[r, c]) == BG:
                out[r, c] = cup["color"]
                added_fill += 1
    return added_green, added_fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each cup's interior by mirroring its green half-pattern across the cup-axis, then fill remaining voids with the wall color.

    Input structure:
    - The grid has a black background and 2-3 separated cup-shaped wall objects.
    - Each cup is a one-pixel-thick monochrome U-shape with exactly one open side.
    - Inside each cup, only the half nearer the closed bottom contains a small green 4-connected pattern; the other half is black.

    Rule:
    - Detect every cup and infer its opening side from the missing wall edge.
    - Use the midpoint line of the cup interior perpendicular to the opening direction as the reflection axis.
    - Reflect the existing green interior pixels across that axis, writing only into black target cells.
    - After mirroring, fill every still-black interior cell with that cup's wall color; leave walls and exterior pixels unchanged.

    Key insight:
    - The opening side is not decorative: it selects whether the mirror axis is vertical or horizontal, so the same green shape must be completed differently in differently oriented cups.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for cup in _detect_cups(g):
        _apply_cup_transform(out, cup)
    return out


def _cup_geometry(opening: str, interior_h: int, interior_w: int) -> tuple[int, int, tuple[int, int, int, int]]:
    if opening == "up":
        return interior_h + 1, interior_w + 2, (0, 1, interior_h - 1, interior_w)
    if opening == "down":
        return interior_h + 1, interior_w + 2, (1, 1, interior_h, interior_w)
    if opening == "left":
        return interior_h + 2, interior_w + 1, (1, 0, interior_h, interior_w - 1)
    return interior_h + 2, interior_w + 1, (1, 1, interior_h, interior_w)


def _half_side_cells(interior_bbox: tuple[int, int, int, int], opening: str) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = interior_bbox
    cells: list[tuple[int, int]] = []
    axis_r = (ir0 + ir1) / 2.0
    axis_c = (ic0 + ic1) / 2.0
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            if opening == "up" and r > axis_r:
                cells.append((r, c))
            elif opening == "down" and r < axis_r:
                cells.append((r, c))
            elif opening == "left" and c > axis_c:
                cells.append((r, c))
            elif opening == "right" and c < axis_c:
                cells.append((r, c))
    return cells


def _random_connected_pattern(
    interior_bbox: tuple[int, int, int, int], opening: str, rng: np.random.Generator
) -> list[tuple[int, int]]:
    allowed = _half_side_cells(interior_bbox, opening)
    allowed_set = set(allowed)
    max_size = min(4, len(allowed))
    for _ in range(80):
        size = int(rng.integers(2, max_size + 1))
        seed = allowed[int(rng.integers(0, len(allowed)))]
        cells = {seed}
        while len(cells) < size:
            frontier = []
            for r, c in sorted(cells):
                for dr, dc in DIR4:
                    nb = (r + dr, c + dc)
                    if nb in allowed_set and nb not in cells:
                        frontier.append(nb)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != size:
            continue
        mirrored = {_mirror_coord(r, c, interior_bbox, opening) for r, c in cells}
        if mirrored & cells:
            continue
        area = (interior_bbox[2] - interior_bbox[0] + 1) * (interior_bbox[3] - interior_bbox[1] + 1)
        if len(cells) + len(mirrored) >= area:
            continue
        return sorted(cells)
    raise RuntimeError("failed to build non-degenerate cup pattern")


def _build_cup_patch(
    opening: str,
    wall_color: int,
    interior_h: int,
    interior_w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    patch_h, patch_w, interior_bbox = _cup_geometry(opening, interior_h, interior_w)
    patch = np.zeros((patch_h, patch_w), dtype=int)
    if opening == "up":
        patch[-1, :] = wall_color
        patch[:, 0] = wall_color
        patch[:, -1] = wall_color
    elif opening == "down":
        patch[0, :] = wall_color
        patch[:, 0] = wall_color
        patch[:, -1] = wall_color
    elif opening == "left":
        patch[0, :] = wall_color
        patch[-1, :] = wall_color
        patch[:, -1] = wall_color
    else:
        patch[0, :] = wall_color
        patch[-1, :] = wall_color
        patch[:, 0] = wall_color

    pattern_cells = _random_connected_pattern(interior_bbox, opening, rng)
    for r, c in pattern_cells:
        patch[r, c] = PATTERN_COLOR

    meta = {
        "color": int(wall_color),
        "opening": opening,
        "wall_bbox_local": (0, 0, patch_h - 1, patch_w - 1),
        "interior_bbox_local": interior_bbox,
        "pattern_local": pattern_cells,
    }
    return patch, meta


def _candidate_positions(
    reserved: np.ndarray,
    patch_shape: tuple[int, int],
    border: int,
) -> list[tuple[int, int]]:
    h, w = reserved.shape
    ph, pw = patch_shape
    positions = []
    for r in range(border, h - ph - border + 1):
        for c in range(border, w - pw - border + 1):
            if reserved[r : r + ph, c : c + pw].any():
                continue
            positions.append((r, c))
    return positions


def _place_patches(
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
    rng: np.random.Generator,
    border: int = 1,
    padding: int = 1,
) -> list[tuple[int, int]] | None:
    reserved = np.zeros(grid_shape, dtype=bool)
    placements: list[tuple[int, int]] = []
    for patch in patches:
        positions = _candidate_positions(reserved, patch.shape, border)
        if not positions:
            return None
        r, c = positions[int(rng.integers(0, len(positions)))]
        placements.append((r, c))
        ph, pw = patch.shape
        r0 = max(0, r - padding)
        c0 = max(0, c - padding)
        r1 = min(grid_shape[0], r + ph + padding)
        c1 = min(grid_shape[1], c + pw + padding)
        reserved[r0:r1, c0:c1] = True
    return placements


def _paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _globalize_meta(local_meta: dict, top: int, left: int) -> dict:
    def shift(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        r0, c0, r1, c1 = box
        return r0 + top, c0 + left, r1 + top, c1 + left

    return {
        "color": local_meta["color"],
        "opening": local_meta["opening"],
        "wall_bbox": shift(local_meta["wall_bbox_local"]),
        "interior_bbox": shift(local_meta["interior_bbox_local"]),
        "pattern_cells": sorted((top + r, left + c) for r, c in local_meta["pattern_local"]),
    }


def _expected_from_metadata(inp: np.ndarray, metas: list[dict]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = inp.copy()
    deltas = []
    for meta in metas:
        deltas.append(_apply_cup_transform(out, meta))
    return out, deltas


def _same_cup_signature(cup: dict) -> tuple:
    return cup["wall_bbox"], cup["opening"], cup["color"], tuple(cup["pattern_cells"])


def _validate_generated_case(inp: np.ndarray, metas: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    detected = _detect_cups(inp)
    if len(detected) != len(metas):
        return False
    expected, deltas = _expected_from_metadata(inp, metas)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    expected_signatures = sorted(_same_cup_signature(cup) for cup in metas)
    detected_signatures = sorted(_same_cup_signature(cup) for cup in detected)
    if expected_signatures != detected_signatures:
        return False

    if any(added_green <= 0 or added_fill <= 0 for added_green, added_fill in deltas):
        return False

    openings = {meta["opening"] for meta in metas}
    if not openings & {"left", "right"} or not openings & {"up", "down"}:
        return False

    fill_only = inp.copy()
    for meta in metas:
        ir0, ic0, ir1, ic1 = meta["interior_bbox"]
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 + 1):
                if int(fill_only[r, c]) == BG:
                    fill_only[r, c] = meta["color"]
    if np.array_equal(fill_only, expected):
        return False

    mirror_only = inp.copy()
    any_green_added = False
    for meta in metas:
        for r, c in meta["pattern_cells"]:
            mr, mc = _mirror_coord(r, c, meta["interior_bbox"], meta["opening"])
            if int(mirror_only[mr, mc]) == BG:
                mirror_only[mr, mc] = PATTERN_COLOR
                any_green_added = True
    if not any_green_added or np.array_equal(mirror_only, expected):
        return False

    return True


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    cup_count = 3 if rng.random() < 0.65 else 2
    grid_h = int(rng.integers(14, 21))
    grid_w = int(rng.integers(14, 21))

    horiz = ["left", "right"][int(rng.integers(0, 2))]
    vert = ["up", "down"][int(rng.integers(0, 2))]
    directions = [horiz, vert]
    remaining = [d for d in OPENINGS if d not in directions]
    if cup_count == 3:
        directions.append(remaining[int(rng.integers(0, len(remaining)))])
    rng.shuffle(directions)

    colors = list(rng.choice(np.array(WALL_COLORS), size=cup_count, replace=False))
    patches: list[np.ndarray] = []
    metas_local: list[dict] = []
    for opening, color in zip(directions, colors):
        if opening in {"left", "right"}:
            interior_h = int(rng.integers(3, 6))
            interior_w = int(rng.integers(4, 6))
        else:
            interior_h = int(rng.integers(4, 6))
            interior_w = int(rng.integers(3, 6))
        patch, meta = _build_cup_patch(opening, int(color), interior_h, interior_w, rng)
        patches.append(patch)
        metas_local.append(meta)

    order = list(rng.permutation(cup_count))
    ordered_patches = [patches[i] for i in order]
    ordered_local = [metas_local[i] for i in order]
    placements = _place_patches((grid_h, grid_w), ordered_patches, rng)
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    metas: list[dict] = []
    for patch, meta_local, (r0, c0) in zip(ordered_patches, ordered_local, placements):
        _paste_patch(grid, patch, r0, c0)
        metas.append(_globalize_meta(meta_local, r0, c0))

    if not _validate_generated_case(grid, metas):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-mirror ARC input")
