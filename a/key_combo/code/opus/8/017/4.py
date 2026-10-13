from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    overwrite: bool = True,
    clip: bool = False,
) -> None:
    h, w = patch.shape
    rows, cols = dst.shape

    if not clip:
        if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
            raise ValueError("patch out of bounds")
        rr0, cc0, rr1, cc1 = r0, c0, r0 + h, c0 + w
        pr0, pc0 = 0, 0
    else:
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(rows, r0 + h)
        cc1 = min(cols, c0 + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return
        pr0 = rr0 - r0
        pc0 = cc0 - c0

    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    dst_view = dst[rr0:rr1, cc0:cc1]
    patch_view = patch[pr0:pr1, pc0:pc1]
    mask = np.ones_like(patch_view, dtype=bool)
    if transparent is not None:
        mask &= patch_view != int(transparent)
        if not overwrite:
            mask &= dst_view == int(transparent)
    elif not overwrite:
        raise ValueError("overwrite=False requires transparent")
    dst_view[mask] = patch_view[mask]


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    monochromatic: bool,
    colors: Iterable[int] | None = None,
) -> list[dict]:
    comps: list[dict] = []
    if monochromatic:
        color_values = sorted(set(map(int, colors if colors is not None else np.unique(grid))) - {int(bg)})
        for color in color_values:
            for coords in _components_from_mask(grid == int(color)):
                comps.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return comps

    if colors is None:
        mask = grid != int(bg)
    else:
        mask = np.isin(grid, list(colors))
    for coords in _components_from_mask(mask):
        colors_here = set(map(int, np.unique(grid[coords[:, 0], coords[:, 1]]))) - {int(bg)}
        comps.append(
            {
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
                "colors": colors_here,
            }
        )
    return comps


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _quadrant_of(r: int, c: int, shape: tuple[int, int]) -> str:
    rows, cols = map(int, shape)
    upper = 2 * int(r) < rows
    left = 2 * int(c) < cols
    if upper and left:
        return "ul"
    if upper and not left:
        return "ur"
    if not upper and left:
        return "ll"
    return "lr"


def _rotation_steps_for_quadrant(quadrant: str) -> int:
    if quadrant == "ul":
        return 0
    if quadrant == "ur":
        return 3
    if quadrant == "ll":
        return 1
    return 2


def _find_template(grid: np.ndarray, bg: int) -> tuple[dict, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    mixed = [
        comp
        for comp in _find_components(grid, bg=bg, monochromatic=False)
        if comp["colors"] == {BLUE, RED}
    ]
    assert len(mixed) == 1, "expected one mixed blue-red template"
    template = mixed[0]
    patch = _crop_bbox(grid, template["bbox"])
    source_rel = [tuple(map(int, rc)) for rc in np.argwhere(patch == BLUE)]
    marker_rel = [tuple(map(int, rc)) for rc in np.argwhere(patch == RED)]
    assert source_rel and marker_rel, "template must contain both blue and red cells"
    return template, patch, sorted(source_rel), sorted(marker_rel)


def _find_target_squares(
    grid: np.ndarray, bg: int, template_coords: set[tuple[int, int]]
) -> list[dict]:
    targets: list[dict] = []
    for comp in _find_components(grid, bg=bg, monochromatic=True, colors=[RED]):
        coords = _coord_set(comp["coords"])
        if coords <= template_coords:
            continue
        assert coords.isdisjoint(template_coords), "target squares must be outside the template"
        r0, c0, r1, c1 = comp["bbox"]
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        assert side_r == side_c, "target markers must be square"
        assert int(comp["area"]) == side_r * side_r, "target markers must be solid squares"
        targets.append({"id": len(targets), "r0": r0, "c0": c0, "side": side_r})
    assert targets, "expected detached red target squares"
    scales = {int(t["side"]) for t in targets}
    assert len(scales) == 1, "all detached squares must have the same side length"
    side = scales.pop()
    assert side >= 2, "scale must be at least 2"
    return targets


def _match_instances(targets: list[dict], marker_rel: list[tuple[int, int]]) -> list[dict]:
    by_pos = {(int(t["r0"]), int(t["c0"])): t for t in targets}
    assert len(by_pos) == len(targets), "target square top-lefts must be unique"
    scale = int(targets[0]["side"])
    instances: dict[frozenset[int], tuple[int, int]] = {}

    for anchor in targets:
        ar, ac = int(anchor["r0"]), int(anchor["c0"])
        for mr, mc in marker_rel:
            origin = (ar - mr * scale, ac - mc * scale)
            member_ids: list[int] = []
            for rr, cc in marker_rel:
                target = by_pos.get((origin[0] + rr * scale, origin[1] + cc * scale))
                if target is None:
                    break
                member_ids.append(int(target["id"]))
            else:
                key = frozenset(member_ids)
                prev = instances.get(key)
                assert prev in (None, origin), "ambiguous target grouping"
                instances[key] = origin

    assert instances, "detached squares must realize the template marker layout"
    covered = sorted(target_id for key in instances for target_id in key)
    assert covered == list(range(len(targets))), "each target square must belong to exactly one instance"
    out = []
    for key, origin in sorted(instances.items(), key=lambda item: item[1]):
        out.append({"origin": origin, "target_ids": sorted(key), "scale": scale})
    return out


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    template, patch, source_rel, marker_rel = _find_template(g, bg)
    target_squares = _find_target_squares(g, bg, _coord_set(template["coords"]))
    instances = _match_instances(target_squares, marker_rel)

    h, w = patch.shape
    source_mask = np.zeros((h, w), dtype=bool)
    for r, c in source_rel:
        source_mask[r, c] = True

    by_id = {int(t["id"]): t for t in target_squares}
    for inst in instances:
        quadrants = {
            _quadrant_of(by_id[target_id]["r0"], by_id[target_id]["c0"], g.shape)
            for target_id in inst["target_ids"]
        }
        assert len(quadrants) == 1, "every matched marker group must lie inside one quadrant"
        inst["quadrant"] = next(iter(quadrants))

    return {
        "grid": g,
        "bg": bg,
        "template_bbox": template["bbox"],
        "source_mask": source_mask,
        "marker_rel": marker_rel,
        "targets": target_squares,
        "instances": instances,
    }


def _render_from_scene(scene: dict, quadrant_override: str | None = None) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    bg = int(scene["bg"])
    source_mask = scene["source_mask"]
    for inst in scene["instances"]:
        quadrant = quadrant_override or inst["quadrant"]
        rotated = np.rot90(source_mask, _rotation_steps_for_quadrant(quadrant))
        scale = int(inst["scale"])
        block = np.full((scale, scale), BLUE, dtype=int)
        origin_r, origin_c = map(int, inst["origin"])
        for rr, cc in np.argwhere(rotated):
            _paste(
                out,
                block,
                origin_r + int(rr) * scale,
                origin_c + int(cc) * scale,
                transparent=bg,
                overwrite=False,
                clip=True,
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp scaled blue copies of a mixed template, rotating each copy by the quadrant of its red marker group.

    Input structure:
    - A black grid contains one 4-connected mixed template made of blue source pixels and red marker pixels.
    - Elsewhere, detached red objects are solid equal-size squares whose top-left corners match the template marker offsets at one common scale.
    - The detached squares are spread across multiple quadrants of the whole grid.

    Rule:
    - Find the mixed blue-red template and use its bbox as the template coordinate system.
    - The common side length k of the detached red squares is the scale factor, and matching their top-left corners to the template red offsets identifies each instance origin.
    - For each matched marker group, rotate the template blue pattern by the quadrant of that group's square top-lefts, then expand each rotated blue cell to a k x k block and stamp it at the instance origin while keeping the red squares and template unchanged.

    Key insight:
    - The same detached red squares encode two independent controls: their size gives the global scale, while their quadrant gives the local rotation for the blue stamp.
    """
    scene = _analyze_scene(grid)
    return _render_from_scene(scene)


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask)
    return len(comps) == 1


def _sample_template(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(2000):
        side = int(rng.integers(3, 5))
        total_cells = side * side
        marker_count = int(rng.choice([2, 3]))
        blue_count = int(rng.integers(3, min(6, total_cells - marker_count + 1)))
        chosen = rng.choice(total_cells, size=marker_count + blue_count, replace=False)
        marker_idx = set(map(int, chosen[:marker_count]))
        blue_idx = set(map(int, chosen[marker_count:]))

        marker_mask = np.zeros((side, side), dtype=bool)
        blue_mask = np.zeros((side, side), dtype=bool)
        for idx in marker_idx:
            marker_mask[idx // side, idx % side] = True
        for idx in blue_idx:
            blue_mask[idx // side, idx % side] = True

        if marker_mask[0, 0]:
            continue
        if not _mask_connected(marker_mask | blue_mask):
            continue

        rotations = [np.rot90(blue_mask, k) for k in range(4)]
        if len({rot.astype(np.uint8).tobytes() for rot in rotations}) != 4:
            continue
        if any(np.any(rot & marker_mask) for rot in rotations):
            continue
        return blue_mask, marker_mask
    raise RuntimeError("failed to sample a non-degenerate template")


def _origin_range(
    shape: tuple[int, int],
    side: int,
    scale: int,
    marker_rel: list[tuple[int, int]],
    quadrant: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rows, cols = map(int, shape)
    max_origin_r = rows - side * scale
    max_origin_c = cols - side * scale
    if max_origin_r < 0 or max_origin_c < 0:
        return None

    marker_rows = [r for r, _ in marker_rel]
    marker_cols = [c for _, c in marker_rel]
    min_mr, max_mr = min(marker_rows), max(marker_rows)
    min_mc, max_mc = min(marker_cols), max(marker_cols)

    upper_max_r = (rows - 1) // 2
    left_max_c = (cols - 1) // 2
    lower_min_r = (rows + 1) // 2
    right_min_c = (cols + 1) // 2

    if quadrant in {"ul", "ur"}:
        r_lo, r_hi = 0, min(max_origin_r, upper_max_r - max_mr * scale)
    else:
        r_lo = max(0, lower_min_r - min_mr * scale)
        r_hi = max_origin_r

    if quadrant in {"ul", "ll"}:
        c_lo, c_hi = 0, min(max_origin_c, left_max_c - max_mc * scale)
    else:
        c_lo = max(0, right_min_c - min_mc * scale)
        c_hi = max_origin_c

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (int(r_lo), int(r_hi)), (int(c_lo), int(c_hi))


def _block_cells(r0: int, c0: int, side: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(side) for dc in range(side)}


def _instance_cells(
    origin: tuple[int, int],
    quadrant: str,
    source_mask: np.ndarray,
    marker_rel: list[tuple[int, int]],
    scale: int,
) -> set[tuple[int, int]]:
    origin_r, origin_c = map(int, origin)
    cells: set[tuple[int, int]] = set()
    for rr, cc in marker_rel:
        cells |= _block_cells(origin_r + rr * scale, origin_c + cc * scale, scale)
    rotated = np.rot90(source_mask, _rotation_steps_for_quadrant(quadrant))
    for rr, cc in np.argwhere(rotated):
        cells |= _block_cells(origin_r + int(rr) * scale, origin_c + int(cc) * scale, scale)
    return cells


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    rows, cols = map(int, shape)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    out.add((nr, nc))
    return out


def _build_grid(
    shape: tuple[int, int],
    template_origin: tuple[int, int],
    blue_mask: np.ndarray,
    marker_mask: np.ndarray,
    scale: int,
    placements: list[tuple[str, tuple[int, int]]],
) -> np.ndarray:
    rows, cols = map(int, shape)
    side = blue_mask.shape[0]
    grid = np.full((rows, cols), BG, dtype=int)

    tr, tc = map(int, template_origin)
    for rr, cc in np.argwhere(blue_mask):
        grid[tr + int(rr), tc + int(cc)] = BLUE
    for rr, cc in np.argwhere(marker_mask):
        grid[tr + int(rr), tc + int(cc)] = RED

    marker_rel = [tuple(map(int, rc)) for rc in np.argwhere(marker_mask)]
    for _quadrant, (origin_r, origin_c) in placements:
        for rr, cc in marker_rel:
            grid[
                origin_r + rr * scale : origin_r + (rr + 1) * scale,
                origin_c + cc * scale : origin_c + (cc + 1) * scale,
            ] = RED
    return grid


def _semantic_checks(
    grid: np.ndarray,
    expected_instances: int,
    expected_scale: int,
) -> bool:
    try:
        scene = _analyze_scene(grid)
    except AssertionError:
        return False

    if len(scene["instances"]) != expected_instances:
        return False
    if {inst["scale"] for inst in scene["instances"]} != {expected_scale}:
        return False
    if {inst["quadrant"] for inst in scene["instances"]} != {"ul", "ur", "ll", "lr"}:
        return False

    source_mask = scene["source_mask"]
    if len({np.rot90(source_mask, k).astype(np.uint8).tobytes() for k in range(4)}) != 4:
        return False

    out = _render_from_scene(scene)
    for wrong_quadrant in ("ul", "ur", "ll", "lr"):
        if np.array_equal(out, _render_from_scene(scene, quadrant_override=wrong_quadrant)):
            return False
    return True


def _sample_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(24, 31))
    cols = int(rng.integers(24, 31))
    blue_mask, marker_mask = _sample_template(rng)
    side = int(blue_mask.shape[0])
    marker_rel = [tuple(map(int, rc)) for rc in np.argwhere(marker_mask)]

    template_origin = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
    if template_origin[0] + side > rows or template_origin[1] + side > cols:
        return None

    scale_options = []
    for scale in (2, 3, 4):
        if any(_origin_range((rows, cols), side, scale, marker_rel, q) is None for q in ("ul", "ur", "ll", "lr")):
            continue
        scale_options.append(scale)
    if not scale_options:
        return None
    scale = int(rng.choice(scale_options))

    quadrant_order = ["ul", "ur", "ll", "lr"]
    extras = int(rng.integers(0, 3))
    quadrant_order += [str(rng.choice(["ul", "ur", "ll", "lr"])) for _ in range(extras)]
    rng.shuffle(quadrant_order)

    template_cells = {
        (template_origin[0] + r, template_origin[1] + c)
        for r in range(side)
        for c in range(side)
        if blue_mask[r, c] or marker_mask[r, c]
    }
    blocked = _expand(template_cells, (rows, cols), pad=1)
    placements: list[tuple[str, tuple[int, int]]] = []

    for quadrant in quadrant_order:
        ranges = _origin_range((rows, cols), side, scale, marker_rel, quadrant)
        if ranges is None:
            return None
        (r_lo, r_hi), (c_lo, c_hi) = ranges
        candidates: list[tuple[int, int]] = []
        for _ in range(250):
            origin = (int(rng.integers(r_lo, r_hi + 1)), int(rng.integers(c_lo, c_hi + 1)))
            cells = _instance_cells(origin, quadrant, blue_mask, marker_rel, scale)
            if cells & blocked:
                continue
            candidates.append(origin)
        if not candidates:
            return None
        origin = candidates[int(rng.integers(0, len(candidates)))]
        placements.append((quadrant, origin))
        blocked |= _expand(_instance_cells(origin, quadrant, blue_mask, marker_rel, scale), (rows, cols), pad=1)

    grid = _build_grid((rows, cols), template_origin, blue_mask, marker_mask, scale, placements)
    if not _semantic_checks(grid, len(placements), scale):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _sample_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid puzzle")
