from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
TRACK = 9
STRIPE_COLORS = (1, 2, 3, 8)
SPRITE_COLORS = (4, 5, 6, 7)

DIR_LEFT = "left"
DIR_RIGHT = "right"
DIR_UP = "up"
DIR_DOWN = "down"

SHAPES = (
    np.array(
        [
            [0, 1, 2],
            [2, 2, 0],
            [0, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [2, 1, 0],
            [0, 2, 2],
            [2, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [2, 1, 0],
            [2, 2, 2],
            [0, 0, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 2],
            [2, 2, 2],
            [2, 0, 0],
        ],
        dtype=int,
    ),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _transform_patch(patch: np.ndarray, family: str, rot_k: int) -> np.ndarray:
    out = patch
    if family == "flip_h":
        out = np.fliplr(out)
    elif family == "flip_v":
        out = np.flipud(out)
    elif family != "rot":
        raise ValueError(f"unknown transform family: {family}")
    return np.rot90(out, int(rot_k))


def _transform_tiers() -> list[list[tuple[str, int]]]:
    return [
        [("rot", k) for k in range(4)],
        [("flip_h", k) for k in range(4)] + [("flip_v", k) for k in range(4)],
    ]


def _all_unique_transforms(patch: np.ndarray) -> list[np.ndarray]:
    seen: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for family, rot_k in _transform_tiers()[0] + _transform_tiers()[1]:
        out = _transform_patch(patch, family, rot_k)
        key = (tuple(map(int, out.shape)), out.tobytes())
        seen.setdefault(key, out)
    return list(seen.values())


def _marker_coord(patch: np.ndarray, marker_color: int) -> tuple[int, int]:
    coords = np.argwhere(patch == int(marker_color))
    assert coords.shape == (1, 2), "expected exactly one marker pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _marker_faces_direction(patch: np.ndarray, marker_color: int, direction: str) -> bool:
    r, c = _marker_coord(patch, marker_color)
    h, w = patch.shape
    if direction == DIR_LEFT:
        return c == 0
    if direction == DIR_RIGHT:
        return c == w - 1
    if direction == DIR_UP:
        return r == 0
    if direction == DIR_DOWN:
        return r == h - 1
    raise ValueError(f"unknown direction: {direction}")


def _orientation_candidates(
    patch: np.ndarray, marker_color: int, direction: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tiers: list[list[np.ndarray]] = []
    for ops in _transform_tiers():
        seen: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
        for family, rot_k in ops:
            out = _transform_patch(patch, family, rot_k)
            if not _marker_faces_direction(out, marker_color, direction):
                continue
            key = (tuple(map(int, out.shape)), out.tobytes())
            seen.setdefault(key, out)
        tiers.append(list(seen.values()))
    return tiers[0], tiers[1]


def _orient_patch_to_direction(patch: np.ndarray, marker_color: int, direction: str) -> np.ndarray:
    rot_only, with_flip = _orientation_candidates(patch, marker_color, direction)
    if rot_only:
        assert len(rot_only) == 1, "rotation-only orientation must be unique"
        return rot_only[0]
    assert with_flip, "expected at least one flip-assisted orientation"
    assert len(with_flip) == 1, "flip-assisted orientation must be unique"
    return with_flip[0]


def _detect_stripes(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    stripes: list[dict] = []
    stripe_mask = np.zeros_like(grid, dtype=bool)

    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) not in (BG, TRACK):
            stripes.append({"orientation": "v", "index": c, "color": int(col[0])})
            stripe_mask[:, c] = True
    for r in range(h):
        row = grid[r, :]
        if np.all(row == row[0]) and int(row[0]) not in (BG, TRACK):
            stripes.append({"orientation": "h", "index": r, "color": int(row[0])})
            stripe_mask[r, :] = True

    assert 2 <= len(stripes) <= 3, "expected 2 or 3 full-width/full-height stripes"
    orientations = {item["orientation"] for item in stripes}
    assert len(orientations) == 1, "generated cases use stripes of one orientation"
    colors = [item["color"] for item in stripes]
    assert len(set(colors)) == len(colors), "stripe colors must be distinct"
    return stripes, stripe_mask


def _detect_track(grid: np.ndarray) -> dict:
    coords = np.argwhere(grid == TRACK)
    assert coords.size > 0, "expected visible track points"

    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    if len(rows) == 1:
        positions = sorted(int(c) for _r, c in coords)
        orientation = "h"
        fixed = rows[0]
    else:
        assert len(cols) == 1, "track points must lie on one row or one column"
        positions = sorted(int(r) for r, _c in coords)
        orientation = "v"
        fixed = cols[0]

    assert len(positions) >= 2, "need at least two visible track points"
    diffs = [b - a for a, b in zip(positions, positions[1:])]
    spacing = min(diffs)
    assert spacing >= 2, "track spacing must be at least 2"
    assert all(d % spacing == 0 for d in diffs), "visible points must fit one arithmetic track"
    full_positions = list(range(positions[0], positions[-1] + 1, spacing))
    assert full_positions[-1] == positions[-1]

    return {
        "orientation": orientation,
        "fixed": fixed,
        "visible_positions": positions,
        "full_positions": full_positions,
        "spacing": spacing,
    }


def _extract_sprites(grid: np.ndarray, stripe_mask: np.ndarray, stripe_colors: set[int]) -> list[dict]:
    mask = (grid != BG) & (grid != TRACK) & (~stripe_mask)
    comps = _connected_components(mask)
    sprites: list[dict] = []
    for coords in comps:
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        marker_coords: list[tuple[int, int]] = []
        other_colors: set[int] = set()
        abs_cells: set[tuple[int, int]] = set()
        for r, c in coords:
            color = int(grid[r, c])
            patch[r - r0, c - c0] = color
            abs_cells.add((int(r), int(c)))
            if color in stripe_colors:
                marker_coords.append((int(r - r0), int(c - c0)))
            else:
                other_colors.add(color)
        assert len(marker_coords) == 1, "each sprite must have exactly one stripe-colored marker"
        assert len(other_colors) == 1, "each sprite must have one main color"
        marker_r, marker_c = marker_coords[0]
        marker_color = int(patch[marker_r, marker_c])
        sprites.append(
            {
                "bbox": bbox,
                "patch": patch,
                "coords": coords,
                "abs_cells": abs_cells,
                "marker_color": marker_color,
                "main_color": next(iter(other_colors)),
            }
        )
    return sprites


def _track_slots_covered(track: dict, cells: Iterable[tuple[int, int]]) -> list[int]:
    cell_set = set(cells)
    covered: list[int] = []
    if track["orientation"] == "h":
        row = track["fixed"]
        for pos in track["full_positions"]:
            if (row, pos) in cell_set:
                covered.append(pos)
    else:
        col = track["fixed"]
        for pos in track["full_positions"]:
            if (pos, col) in cell_set:
                covered.append(pos)
    return covered


def _target_direction(sprite: dict, stripes: list[dict], track: dict) -> str:
    marker_color = sprite["marker_color"]
    bbox = sprite["bbox"]
    stripe = next(item for item in stripes if item["color"] == marker_color)
    if track["orientation"] == "h":
        assert stripe["orientation"] == "v", "horizontal track should pair with vertical stripes"
        stripe_c = stripe["index"]
        if stripe_c < bbox[1]:
            return DIR_LEFT
        assert stripe_c > bbox[3], "marker stripe must be strictly to one side"
        return DIR_RIGHT
    assert stripe["orientation"] == "h", "vertical track should pair with horizontal stripes"
    stripe_r = stripe["index"]
    if stripe_r < bbox[0]:
        return DIR_UP
    assert stripe_r > bbox[2], "marker stripe must be strictly to one side"
    return DIR_DOWN


def _shift_to_next_slot(
    top_left: tuple[int, int], patch_shape: tuple[int, int], direction: str, track: dict
) -> tuple[int, int] | None:
    r0, c0 = top_left
    h, w = patch_shape
    slots = track["full_positions"]
    if direction == DIR_LEFT:
        targets = [pos for pos in slots if pos < c0]
        if not targets:
            return None
        return r0, targets[-1]
    if direction == DIR_RIGHT:
        front = c0 + w - 1
        targets = [pos for pos in slots if pos > front]
        if not targets:
            return None
        return r0, targets[0] - (w - 1)
    if direction == DIR_UP:
        targets = [pos for pos in slots if pos < r0]
        if not targets:
            return None
        return targets[-1], c0
    if direction == DIR_DOWN:
        front = r0 + h - 1
        targets = [pos for pos in slots if pos > front]
        if not targets:
            return None
        return targets[0] - (h - 1), c0
    raise ValueError(f"unknown direction: {direction}")


def _paint_patch(dst: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _parse_scene(grid: np.ndarray) -> dict:
    stripes, stripe_mask = _detect_stripes(grid)
    track = _detect_track(grid)
    stripe_colors = {item["color"] for item in stripes}
    sprites = _extract_sprites(grid, stripe_mask, stripe_colors)
    assert 2 <= len(sprites) <= 3, "expected 2 or 3 sprites"
    main_colors = [item["main_color"] for item in sprites]
    assert len(set(main_colors)) == len(main_colors), "sprite main colors must be distinct"
    stripe_orientation = stripes[0]["orientation"]
    assert stripe_orientation != track["orientation"], "track axis should be perpendicular to stripes"

    for sprite in sprites:
        sprite["direction"] = _target_direction(sprite, stripes, track)
    return {
        "stripes": stripes,
        "stripe_mask": stripe_mask,
        "track": track,
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each sprite to face its same-colored partition stripe, then slide it one track slot toward that stripe.

    Input structure:
    - Black background with 2 or 3 single-pixel partition stripes, all full rows or full columns and all different colors.
    - One maroon dotted track lies on a single row or single column with equal spacing between visible points.
    - Two or three small 4-connected sprites sit on the track; each has one main color and exactly one marker pixel whose color matches one stripe.

    Rule:
    - Detect the full-row/full-column stripes and the equal-step track.
    - For each sprite, use the marker color to identify the matching stripe, then determine whether that stripe lies above, below, left, or right of the sprite.
    - Reorient the sprite so its marker pixel lies on the bbox side facing that stripe, preferring pure rotation and using one flip plus rotation only if no rotation works.
    - Keep the reoriented sprite's top-left fixed, then slide it along the track so its front bbox edge reaches the next track point in that direction.
    - Erase the original sprite, restore the full track, and draw the moved sprite on top of the track.

    Key insight:
    - The marker pixel does not encode a direction by itself: its stripe color identifies which partition stripe to face, and that same facing direction also determines the one-slot track movement.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    out = g.copy()

    for sprite in scene["sprites"]:
        for r, c in sprite["abs_cells"]:
            out[r, c] = BG

    track = scene["track"]
    if track["orientation"] == "h":
        row = track["fixed"]
        for c in track["full_positions"]:
            out[row, c] = TRACK
    else:
        col = track["fixed"]
        for r in track["full_positions"]:
            out[r, col] = TRACK

    used = np.zeros_like(g, dtype=bool)
    used[scene["stripe_mask"]] = True
    for sprite in scene["sprites"]:
        r0, c0, _r1, _c1 = sprite["bbox"]
        moved_patch = _orient_patch_to_direction(
            sprite["patch"], sprite["marker_color"], sprite["direction"]
        )
        target_top_left = _shift_to_next_slot((r0, c0), moved_patch.shape, sprite["direction"], track)
        assert target_top_left is not None, "each sprite must have an empty next slot"
        rr, cc = target_top_left
        h, w = moved_patch.shape
        assert 0 <= rr and 0 <= cc and rr + h <= g.shape[0] and cc + w <= g.shape[1]
        cells = np.argwhere(moved_patch != BG)
        for pr, pc in cells:
            ar = rr + int(pr)
            ac = cc + int(pc)
            assert not scene["stripe_mask"][ar, ac], "moved sprite must stay off stripes"
            assert not used[ar, ac], "moved sprites may not overlap"
        for pr, pc in cells:
            used[rr + int(pr), cc + int(pc)] = True
        _paint_patch(out, moved_patch, target_top_left)

    return out


def _colorize_shape(shape: np.ndarray, main_color: int, marker_color: int) -> np.ndarray:
    patch = np.zeros_like(shape, dtype=int)
    patch[shape == 2] = int(main_color)
    patch[shape == 1] = int(marker_color)
    return patch


def _patch_cells(patch: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    cells = np.argwhere(patch != BG)
    return {(r0 + int(r), c0 + int(c)) for r, c in cells}


def _expand_cells(
    cells: Iterable[tuple[int, int]], shape: tuple[int, int], radius: int = 1
) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) > radius:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _slot_coord(track: dict, pos: int) -> tuple[int, int]:
    if track["orientation"] == "h":
        return track["fixed"], pos
    return pos, track["fixed"]


def _covers_exactly_one_slot(track: dict, cells: set[tuple[int, int]]) -> tuple[bool, int | None]:
    covered = _track_slots_covered(track, cells)
    if len(covered) == 1:
        return True, covered[0]
    return False, None


def _candidate_positions(
    canvas_shape: tuple[int, int],
    region_bbox: tuple[int, int, int, int],
    track: dict,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    direction: str,
) -> list[dict]:
    h, w = canvas_shape
    rmin, cmin, rmax, cmax = region_bbox
    in_h, in_w = patch_in.shape
    out_h, out_w = patch_out.shape
    slot_first = track["full_positions"][0]
    slot_last = track["full_positions"][-1]
    candidates: list[dict] = []

    for r0 in range(rmin, rmax - in_h + 2):
        for c0 in range(cmin, cmax - in_w + 2):
            in_cells = _patch_cells(patch_in, (r0, c0))
            ok, hidden_pos = _covers_exactly_one_slot(track, in_cells)
            if not ok or hidden_pos is None:
                continue
            if hidden_pos in (slot_first, slot_last):
                continue
            out_top_left = _shift_to_next_slot((r0, c0), patch_out.shape, direction, track)
            if out_top_left is None:
                continue
            rr, cc = out_top_left
            if not (rmin <= rr <= rmax - out_h + 1 and cmin <= cc <= cmax - out_w + 1):
                continue
            out_cells = _patch_cells(patch_out, out_top_left)
            if not _track_slots_covered(track, out_cells):
                continue
            hidden_index = track["full_positions"].index(hidden_pos)
            candidates.append(
                {
                    "input_top_left": (r0, c0),
                    "output_top_left": out_top_left,
                    "input_cells": in_cells,
                    "output_cells": out_cells,
                    "input_buffer": _expand_cells(in_cells, canvas_shape, radius=1),
                    "output_buffer": _expand_cells(out_cells, canvas_shape, radius=1),
                    "hidden_index": hidden_index,
                }
            )
    random.shuffle(candidates)
    return candidates


def _direction_plan(n_sprites: int) -> list[str]:
    if n_sprites == 2:
        if random.random() < 0.7:
            return [DIR_LEFT, DIR_RIGHT]
        return [random.choice([DIR_LEFT, DIR_RIGHT]) for _ in range(2)]
    dirs = [DIR_LEFT, DIR_RIGHT, random.choice([DIR_LEFT, DIR_RIGHT])]
    random.shuffle(dirs)
    return dirs


def _build_horizontal_layout(
    n_sprites: int, n_stripes: int
) -> tuple[np.ndarray, dict, tuple[int, int, int, int], dict] | None:
    for _ in range(200):
        h = random.randint(12, 18)
        w_min = 20 if n_sprites == 3 else 16
        w_min = max(w_min, 24 if n_stripes == 3 else w_min)
        w = random.randint(w_min, 28)
        region_side = "full"
        if n_stripes == 3:
            region_side = random.choice(["left", "right"])
            min_region_width = 15 if n_sprites == 2 else 18
            if region_side == "left":
                mid_lo = min_region_width + 1
                mid_hi = w - 6
            else:
                mid_lo = 5
                mid_hi = w - min_region_width - 2
            if mid_lo > mid_hi:
                continue
            mid = random.randint(mid_lo, mid_hi)
        else:
            mid = None

        stripe_colors = random.sample(list(STRIPE_COLORS), n_stripes)
        grid = np.zeros((h, w), dtype=int)
        grid[:, 0] = stripe_colors[0]
        if n_stripes == 2:
            grid[:, w - 1] = stripe_colors[1]
            region_left, region_right = 1, w - 2
            boundary_colors = {DIR_LEFT: stripe_colors[0], DIR_RIGHT: stripe_colors[1]}
        else:
            assert mid is not None
            grid[:, mid] = stripe_colors[1]
            grid[:, w - 1] = stripe_colors[2]
            if region_side == "left":
                region_left, region_right = 1, mid - 1
                boundary_colors = {DIR_LEFT: stripe_colors[0], DIR_RIGHT: stripe_colors[1]}
            else:
                region_left, region_right = mid + 1, w - 2
                boundary_colors = {DIR_LEFT: stripe_colors[1], DIR_RIGHT: stripe_colors[2]}

        min_region_width = 18 if n_sprites == 3 else 15
        if region_right - region_left + 1 < min_region_width:
            continue

        slot_count = n_sprites + 3
        slot_choices: list[tuple[int, int]] = []
        for spacing in (3, 4):
            usable = region_right - region_left + 1
            needed = 1 + (slot_count - 1) * spacing
            if needed > usable - 2:
                continue
            start_min = region_left + 1
            start_max = region_right - 1 - (slot_count - 1) * spacing
            for start in range(start_min, start_max + 1):
                slot_choices.append((spacing, start))
        if not slot_choices:
            continue

        spacing, start = random.choice(slot_choices)
        slots = [start + i * spacing for i in range(slot_count)]
        track_row = random.randint(2, h - 3)
        track = {
            "orientation": "h",
            "fixed": track_row,
            "visible_positions": slots[:],
            "full_positions": slots[:],
            "spacing": spacing,
        }
        region_bbox = (0, region_left, h - 1, region_right)

        return grid, track, region_bbox, {"n_sprites": n_sprites, "boundary_colors": boundary_colors}
    return None


def _assign_candidates(
    specs: list[dict], track: dict
) -> list[dict] | None:
    order = sorted(range(len(specs)), key=lambda idx: len(specs[idx]["candidates"]))
    used_in: set[tuple[int, int]] = set()
    used_out: set[tuple[int, int]] = set()
    hidden_indices: set[int] = set()
    picked: dict[int, dict] = {}

    def rec(pos: int) -> bool:
        if pos == len(order):
            visible = [i for i in range(len(track["full_positions"])) if i not in hidden_indices]
            if len(visible) < 2:
                return False
            if not any(b - a == 1 for a, b in zip(visible, visible[1:])):
                return False
            return True

        spec_idx = order[pos]
        for cand in specs[spec_idx]["candidates"]:
            if cand["hidden_index"] in hidden_indices:
                continue
            if cand["input_buffer"] & used_in:
                continue
            if cand["output_buffer"] & used_out:
                continue
            picked[spec_idx] = cand
            used_in.update(cand["input_buffer"])
            used_out.update(cand["output_buffer"])
            hidden_indices.add(cand["hidden_index"])
            if rec(pos + 1):
                return True
            del picked[spec_idx]
            used_in.difference_update(cand["input_buffer"])
            used_out.difference_update(cand["output_buffer"])
            hidden_indices.remove(cand["hidden_index"])
        return False

    if not rec(0):
        return None
    return [picked[i] for i in range(len(specs))]


def _extract_output_sprites_by_color(out: np.ndarray) -> dict[int, dict]:
    stripes, stripe_mask = _detect_stripes(out)
    sprites = _extract_sprites(out, stripe_mask, {item["color"] for item in stripes})
    return {item["main_color"]: item for item in sprites}


def _validate_generated_input(grid: np.ndarray) -> None:
    scene = _parse_scene(grid)
    assert len(scene["track"]["full_positions"]) >= len(scene["sprites"]) + 2
    assert any(
        b - a == scene["track"]["spacing"]
        for a, b in zip(scene["track"]["visible_positions"], scene["track"]["visible_positions"][1:])
    )

    hidden_slots: list[int] = []
    for sprite in scene["sprites"]:
        covered = _track_slots_covered(scene["track"], sprite["abs_cells"])
        assert len(covered) == 1, "each input sprite should hide exactly one slot"
        hidden_slots.extend(covered)
        rot_only, with_flip = _orientation_candidates(
            sprite["patch"], sprite["marker_color"], sprite["direction"]
        )
        assert (len(rot_only) == 1) or (len(rot_only) == 0 and len(with_flip) == 1)

    assert len(hidden_slots) == len(set(hidden_slots)), "sprites must hide different slots"
    out = main(grid)
    out_sprites = _extract_output_sprites_by_color(out)
    assert set(out_sprites) == {item["main_color"] for item in scene["sprites"]}

    for sprite in scene["sprites"]:
        r0, c0, _r1, _c1 = sprite["bbox"]
        oriented = _orient_patch_to_direction(
            sprite["patch"], sprite["marker_color"], sprite["direction"]
        )
        target_top_left = _shift_to_next_slot(
            (r0, c0), oriented.shape, sprite["direction"], scene["track"]
        )
        assert target_top_left is not None
        out_sprite = out_sprites[sprite["main_color"]]
        assert out_sprite["bbox"][:2] == target_top_left
        assert np.array_equal(out_sprite["patch"], oriented)

    assert not np.array_equal(grid, out), "output must differ from input"


def _generate_horizontal_case(n_sprites: int, n_stripes: int) -> np.ndarray | None:
    layout = _build_horizontal_layout(n_sprites, n_stripes)
    if layout is None:
        return None
    base, track, region_bbox, meta = layout
    boundary_colors = meta["boundary_colors"]
    directions = _direction_plan(n_sprites)
    main_colors = random.sample(list(SPRITE_COLORS), n_sprites)

    specs: list[dict] = []
    nontrivial = False
    for main_color, direction in zip(main_colors, directions):
        marker_color = boundary_colors[direction]
        made = None
        for _ in range(60):
            shape = random.choice(SHAPES)
            canonical = _colorize_shape(shape, main_color, marker_color)
            variants = _all_unique_transforms(canonical)
            patch_in = random.choice(variants)
            patch_out = _orient_patch_to_direction(patch_in, marker_color, direction)
            if np.array_equal(patch_in, patch_out) and random.random() < 0.8:
                continue
            candidates = _candidate_positions(base.shape, region_bbox, track, patch_in, patch_out, direction)
            if candidates:
                made = {
                    "main_color": main_color,
                    "marker_color": marker_color,
                    "direction": direction,
                    "patch_in": patch_in,
                    "patch_out": patch_out,
                    "candidates": candidates,
                }
                if not np.array_equal(patch_in, patch_out):
                    nontrivial = True
                break
        if made is None:
            return None
        specs.append(made)

    if not nontrivial:
        return None

    placements = _assign_candidates(specs, track)
    if placements is None:
        return None

    grid = base.copy()
    for c in track["full_positions"]:
        grid[track["fixed"], c] = TRACK
    for spec, placement in zip(specs, placements):
        _paint_patch(grid, spec["patch_in"], placement["input_top_left"])

    return grid


def generate_input() -> np.ndarray:
    configs = [(2, 2), (2, 3)] + [(3, 2)] * 8
    for _ in range(400):
        n_sprites, n_stripes = random.choice(configs)
        grid = _generate_horizontal_case(n_sprites, n_stripes)
        if grid is None:
            continue
        if random.random() < 0.5:
            grid = grid.T.copy()
        try:
            _validate_generated_input(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
