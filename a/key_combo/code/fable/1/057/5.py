from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BG = Color.BLACK
WALL = Color.GREY
SIDES = ("top", "bottom", "left", "right")
PATTERNS = ((0,), (0, 1), (0, 1, 2), (0, 2))
WIDTH3_PATTERNS = ((0, 1, 2), (0, 2))
BLOCK_COLORS = [c for c in Color.NOT_BLACK if c != WALL]


def _axis_key(side: str):
    return (lambda rc: (rc[1], rc[0])) if side in ("top", "bottom") else (lambda rc: (rc[0], rc[1]))


def _group_positions(values: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for value in values:
        if not groups or value - groups[-1][-1] > 2:
            groups.append([int(value)])
        else:
            groups[-1].append(int(value))
    return groups


def _run_length(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int], color: int) -> int:
    r, c = start
    dr, dc = step
    length = 0
    while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) == color:
        length += 1
        r += dr
        c += dc
    return length


def _scan_side_notches(grid: np.ndarray, side: str, wall: int = WALL) -> list[dict]:
    h, w = grid.shape
    if side == "top":
        positions = sorted(int(c) for c in np.flatnonzero(grid[0] == wall))
        if not positions:
            return []
        thickness = min(_run_length(grid, (0, c), (1, 0), wall) - 1 for c in positions)
        inner_coord = thickness
        return [
            {
                "side": side,
                "signature": (side, tuple(c - group[0] for c in group)),
                "outer_cells": [(0, int(c)) for c in group],
                "target_cells": [(int(inner_coord), int(c)) for c in group],
            }
            for group in _group_positions(positions)
        ]
    if side == "bottom":
        positions = sorted(int(c) for c in np.flatnonzero(grid[h - 1] == wall))
        if not positions:
            return []
        thickness = min(_run_length(grid, (h - 1, c), (-1, 0), wall) - 1 for c in positions)
        inner_coord = h - 1 - thickness
        return [
            {
                "side": side,
                "signature": (side, tuple(c - group[0] for c in group)),
                "outer_cells": [(h - 1, int(c)) for c in group],
                "target_cells": [(int(inner_coord), int(c)) for c in group],
            }
            for group in _group_positions(positions)
        ]
    if side == "left":
        positions = sorted(int(r) for r in np.flatnonzero(grid[:, 0] == wall))
        if not positions:
            return []
        thickness = min(_run_length(grid, (r, 0), (0, 1), wall) - 1 for r in positions)
        inner_coord = thickness
        return [
            {
                "side": side,
                "signature": (side, tuple(r - group[0] for r in group)),
                "outer_cells": [(int(r), 0) for r in group],
                "target_cells": [(int(r), int(inner_coord)) for r in group],
            }
            for group in _group_positions(positions)
        ]
    positions = sorted(int(r) for r in np.flatnonzero(grid[:, w - 1] == wall))
    if not positions:
        return []
    thickness = min(_run_length(grid, (r, w - 1), (0, -1), wall) - 1 for r in positions)
    inner_coord = w - 1 - thickness
    return [
        {
            "side": side,
            "signature": (side, tuple(r - group[0] for r in group)),
            "outer_cells": [(int(r), w - 1) for r in group],
            "target_cells": [(int(r), int(inner_coord)) for r in group],
        }
        for group in _group_positions(positions)
    ]


def _parse_block(grid: np.ndarray, component: dict, bg: int) -> dict:
    bbox = tuple(map(int, component["bbox"]))
    color = int(component["color"])
    patch = crop_bbox(grid, bbox)
    mask = patch == color
    h, w = mask.shape
    candidates: list[dict] = []

    if h >= 2 and 0 < int(mask[0].sum()) < w and bool(np.all(mask[1:, :])):
        cols = np.flatnonzero(mask[0])
        candidates.append(
            {
                "side": "top",
                "line_positions": [int(c) for c in cols],
                "tooth_rel": [(0, int(c)) for c in cols],
            }
        )
    if h >= 2 and 0 < int(mask[-1].sum()) < w and bool(np.all(mask[:-1, :])):
        cols = np.flatnonzero(mask[-1])
        candidates.append(
            {
                "side": "bottom",
                "line_positions": [int(c) for c in cols],
                "tooth_rel": [(h - 1, int(c)) for c in cols],
            }
        )
    if w >= 2 and 0 < int(mask[:, 0].sum()) < h and bool(np.all(mask[:, 1:])):
        rows = np.flatnonzero(mask[:, 0])
        candidates.append(
            {
                "side": "left",
                "line_positions": [int(r) for r in rows],
                "tooth_rel": [(int(r), 0) for r in rows],
            }
        )
    if w >= 2 and 0 < int(mask[:, -1].sum()) < h and bool(np.all(mask[:, :-1])):
        rows = np.flatnonzero(mask[:, -1])
        candidates.append(
            {
                "side": "right",
                "line_positions": [int(r) for r in rows],
                "tooth_rel": [(int(r), w - 1) for r in rows],
            }
        )

    assert len(candidates) == 1, "each block must have exactly one toothed side"
    chosen = candidates[0]
    line_positions = chosen["line_positions"]
    signature = (chosen["side"], tuple(p - line_positions[0] for p in line_positions))
    r0, c0, _r1, _c1 = bbox
    tooth_abs = [(r0 + rr, c0 + cc) for rr, cc in chosen["tooth_rel"]]

    return {
        "bbox": bbox,
        "patch": patch,
        "color": color,
        "side": chosen["side"],
        "signature": signature,
        "tooth_abs": sorted(tooth_abs, key=_axis_key(chosen["side"])),
        "coords": component["coords"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve unseen inner lockholes from the wall's outer tabs, then slide each toothed block into its complementary notch.

    Input structure:
    - A grey thick wall occupies one or more canvas sides, leaving a one-cell exterior gap at the border.
    - The wall is flat on its inner face, but its outer face has several one-cell-deep tab patterns on the gap side.
    - Separate monochrome field blocks each have exactly one toothed side; the tooth shape matches one outer wall tab.

    Rule:
    - For every outer wall tab, project straight through the wall thickness and remove the same pattern from the opposite inner wall face, creating a notch.
    - Match each block to the unique notch with the same side orientation and tooth pattern; translate the block without rotating or flipping so its teeth fill that notch and its body sits flush against the wall.
    - Clear the blocks from their original positions, keep the wall, and include the newly carved notches now occupied by the moved blocks.

    Key insight:
    - The lockholes never appear in the input; they must be inferred from the opposite wall edge first, and only then can tooth-shape matching determine where each block belongs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    notches = [feature for side in SIDES for feature in _scan_side_notches(g, side, wall=WALL)]
    assert notches, "expected at least one wall tab"

    notch_by_signature: dict[tuple[str, tuple[int, ...]], dict] = {}
    for feature in notches:
        signature = feature["signature"]
        assert signature not in notch_by_signature, "wall tabs must be uniquely identifiable"
        notch_by_signature[signature] = feature

    blocks_grid = g.copy()
    blocks_grid[g == WALL] = bg
    components = find_components(blocks_grid, bg=bg, connectivity=4, monochromatic=True)
    blocks = [_parse_block(blocks_grid, component, bg) for component in components]
    assert len(blocks) == len(notches), "every wall tab must have one matching block"

    block_signatures = [block["signature"] for block in blocks]
    assert len(set(block_signatures)) == len(block_signatures), "block tooth signatures must be unique"
    assert set(block_signatures) == set(notch_by_signature), "blocks and inferred notches must match exactly"

    out = g.copy()
    for block in blocks:
        out[block["coords"][:, 0], block["coords"][:, 1]] = bg
    for feature in notches:
        for r, c in feature["target_cells"]:
            out[r, c] = bg

    for block in blocks:
        target_cells = sorted(notch_by_signature[block["signature"]]["target_cells"], key=_axis_key(block["side"]))
        source_cells = block["tooth_abs"]
        dr = target_cells[0][0] - source_cells[0][0]
        dc = target_cells[0][1] - source_cells[0][1]
        shifted = [(r + dr, c + dc) for r, c in source_cells]
        assert shifted == target_cells, "tooth pattern translation must align exactly"

        r0, c0, _r1, _c1 = block["bbox"]
        patch = block["patch"]
        rr = r0 + dr
        cc = c0 + dc
        active = np.argwhere(patch != bg)
        for pr, pc in active:
            tr = rr + int(pr)
            tc = cc + int(pc)
            assert 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1], "translated block must stay in bounds"
            assert int(out[tr, tc]) == bg, "translated block may only occupy free cells and carved notches"
        paste(out, patch, rr, cc, transparent=bg)

    return out


def _build_wall(shape: tuple[int, int], thickness: int, sides: tuple[str, ...]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    if "top" in sides:
        grid[1 : thickness + 1, 1 : w - 1] = WALL
    if "bottom" in sides:
        grid[h - 1 - thickness : h - 1, 1 : w - 1] = WALL
    if "left" in sides:
        grid[1 : h - 1, 1 : thickness + 1] = WALL
    if "right" in sides:
        grid[1 : h - 1, w - 1 - thickness : w - 1] = WALL
    return grid


def _choose_sides() -> tuple[str, ...]:
    side_count = random.choice((1, 2, 2, 3))
    return tuple(random.sample(SIDES, side_count))


def _choose_block_count(side_count: int) -> int:
    if side_count == 1:
        return random.choice((2, 3))
    if side_count == 2:
        return random.choice((3, 4))
    return 4


def _assign_signatures(sides: tuple[str, ...], block_count: int) -> list[dict]:
    counts = {side: 1 for side in sides}
    focus_side = random.choice(sides)
    remaining = block_count - len(sides)
    counts[focus_side] += 1
    remaining -= 1
    while remaining > 0:
        side = random.choice(sides)
        if counts[side] < 3:
            counts[side] += 1
            remaining -= 1

    specs: list[dict] = []
    for side in sides:
        count = counts[side]
        if count == 1:
            patterns = [random.choice(PATTERNS)]
        elif count == 2:
            patterns = list(WIDTH3_PATTERNS)
            random.shuffle(patterns)
        else:
            patterns = list(WIDTH3_PATTERNS) + [random.choice(((0,), (0, 1)))]
            random.shuffle(patterns)
        for pattern in patterns:
            specs.append({"side": side, "pattern": tuple(pattern)})
    random.shuffle(specs)
    return specs


def _place_patterns_on_side(axis_len: int, patterns: list[tuple[int, ...]], margin: int) -> list[list[int]] | None:
    ordered = sorted(patterns, key=lambda pattern: (-len(pattern), -(pattern[-1] + 1)))
    for _ in range(200):
        starts: list[tuple[tuple[int, ...], int]] = []
        used_starts: list[int] = []
        ok = True
        for pattern in ordered:
            span = pattern[-1] + 1
            candidates = [
                start
                for start in range(margin, axis_len - margin - span + 1)
                if all(abs(start - other) >= 6 for other in used_starts)
            ]
            if not candidates:
                ok = False
                break
            start = random.choice(candidates)
            used_starts.append(start)
            starts.append((pattern, start))
        if ok:
            pattern_to_group = {pattern: [int(start + offset) for offset in pattern] for pattern, start in starts}
            return [pattern_to_group[pattern] for pattern in patterns]
    return None


def _add_outer_tabs(grid: np.ndarray, specs: list[dict], thickness: int, sides: tuple[str, ...]) -> list[dict] | None:
    del thickness
    h, w = grid.shape
    margin = 5
    by_side = {side: [spec for spec in specs if spec["side"] == side] for side in sides}
    placed: list[dict] = []
    for side in sides:
        side_specs = by_side[side]
        groups = _place_patterns_on_side(
            w if side in ("top", "bottom") else h,
            [spec["pattern"] for spec in side_specs],
            margin,
        )
        if groups is None:
            return None
        for spec, group in zip(side_specs, groups):
            spec = dict(spec)
            spec["outer_positions"] = group
            if side == "top":
                for c in group:
                    grid[0, c] = WALL
            elif side == "bottom":
                for c in group:
                    grid[h - 1, c] = WALL
            elif side == "left":
                for r in group:
                    grid[r, 0] = WALL
            else:
                for r in group:
                    grid[r, w - 1] = WALL
            placed.append(spec)
    return placed


def _wall_inner_limits(shape: tuple[int, int], thickness: int, sides: tuple[str, ...]) -> dict[str, int]:
    h, w = shape
    return {
        "top": thickness + 1 if "top" in sides else 1,
        "bottom": h - 2 - thickness if "bottom" in sides else h - 2,
        "left": thickness + 1 if "left" in sides else 1,
        "right": w - 2 - thickness if "right" in sides else w - 2,
    }


def _make_block_patch(side: str, pattern: tuple[int, ...], color: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if side in ("top", "bottom"):
        span = pattern[-1] + 1
        extra_left = 1
        extra_right = 1
        width = span + extra_left + extra_right
        body_h = random.randint(2, 4)
        patch = np.full((body_h + 1, width), BG, dtype=int)
        if side == "top":
            patch[1:, :] = color
            teeth = [(0, extra_left + offset) for offset in pattern]
        else:
            patch[:-1, :] = color
            teeth = [(body_h, extra_left + offset) for offset in pattern]
    else:
        span = pattern[-1] + 1
        extra_top = 1
        extra_bottom = 1
        height = span + extra_top + extra_bottom
        body_w = random.randint(2, 4)
        patch = np.full((height, body_w + 1), BG, dtype=int)
        if side == "left":
            patch[:, 1:] = color
            teeth = [(extra_top + offset, 0) for offset in pattern]
        else:
            patch[:, :-1] = color
            teeth = [(extra_top + offset, body_w) for offset in pattern]
    for r, c in teeth:
        patch[r, c] = color
    return patch, teeth


def _target_cells_for_spec(spec: dict, shape: tuple[int, int], thickness: int) -> list[tuple[int, int]]:
    h, w = shape
    if spec["side"] == "top":
        return [(thickness, c) for c in spec["outer_positions"]]
    if spec["side"] == "bottom":
        return [(h - 1 - thickness, c) for c in spec["outer_positions"]]
    if spec["side"] == "left":
        return [(r, thickness) for r in spec["outer_positions"]]
    return [(r, w - 1 - thickness) for r in spec["outer_positions"]]


def _patch_target_origin(
    side: str,
    patch: np.ndarray,
    teeth: list[tuple[int, int]],
    target_cells: list[tuple[int, int]],
) -> tuple[int, int]:
    sorted_teeth = sorted(teeth, key=_axis_key(side))
    sorted_targets = sorted(target_cells, key=_axis_key(side))
    dr = sorted_targets[0][0] - sorted_teeth[0][0]
    dc = sorted_targets[0][1] - sorted_teeth[0][1]
    shifted = [(r + dr, c + dc) for r, c in sorted_teeth]
    assert shifted == sorted_targets
    return dr, dc


def _can_overlay(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    active = patch != BG
    window = grid[r0 : r0 + h, c0 : c0 + w]
    return not bool(np.any(active & (window != BG)))


def _can_place_freely(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    if r0 < 1 or c0 < 1 or r0 + h >= grid.shape[0] - 1 or c0 + w >= grid.shape[1] - 1:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(grid.shape[0], r0 + h + padding)
    cc1 = min(grid.shape[1], c0 + w + padding)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BG))


def _preferred_ranges(side: str, shape: tuple[int, int], patch_shape: tuple[int, int], limits: dict[str, int]) -> tuple[range, range]:
    h, w = shape
    ph, pw = patch_shape
    row_min = limits["top"] + 1
    row_max = limits["bottom"] - ph
    col_min = limits["left"] + 1
    col_max = limits["right"] - pw
    if row_min > row_max or col_min > col_max:
        return range(0), range(0)
    if side == "top":
        return range(row_min, min(row_max, row_min + 4) + 1), range(col_min, col_max + 1)
    if side == "bottom":
        return range(max(row_min, row_max - 4), row_max + 1), range(col_min, col_max + 1)
    if side == "left":
        return range(row_min, row_max + 1), range(col_min, min(col_max, col_min + 4) + 1)
    return range(row_min, row_max + 1), range(max(col_min, col_max - 4), col_max + 1)


def _place_input_blocks(
    wall_grid: np.ndarray,
    block_infos: list[dict],
    thickness: int,
    sides: tuple[str, ...],
) -> np.ndarray | None:
    grid = wall_grid.copy()
    limits = _wall_inner_limits(grid.shape, thickness, sides)
    decoy_index = random.randrange(len(block_infos)) if len(sides) > 1 else None

    for index, info in enumerate(block_infos):
        target_side = info["side"]
        if decoy_index == index:
            bait_choices = [side for side in sides if side != target_side]
            bait_side = random.choice(bait_choices) if bait_choices else target_side
        else:
            bait_side = random.choice(sides)

        candidate_positions: list[tuple[int, int]] = []
        row_range, col_range = _preferred_ranges(bait_side, grid.shape, info["patch"].shape, limits)
        candidate_positions.extend((r, c) for r in row_range for c in col_range)

        row_min = limits["top"] + 1
        row_max = limits["bottom"] - info["patch"].shape[0]
        col_min = limits["left"] + 1
        col_max = limits["right"] - info["patch"].shape[1]
        candidate_positions.extend((r, c) for r in range(row_min, row_max + 1) for c in range(col_min, col_max + 1))
        random.shuffle(candidate_positions)

        placed = False
        for r0, c0 in candidate_positions:
            if abs(r0 - info["final_origin"][0]) + abs(c0 - info["final_origin"][1]) < 5:
                continue
            if not _can_place_freely(grid, info["patch"], r0, c0, padding=1):
                continue
            paste(grid, info["patch"], r0, c0, transparent=BG)
            info["input_origin"] = (int(r0), int(c0))
            placed = True
            break
        if not placed:
            return None
    return grid


def _nearest_side(origin: tuple[int, int], patch: np.ndarray, limits: dict[str, int], sides: tuple[str, ...]) -> str:
    h, w = patch.shape
    r0, c0 = origin
    center_r = r0 + (h - 1) / 2.0
    center_c = c0 + (w - 1) / 2.0
    distances = {
        "top": center_r - limits["top"],
        "bottom": limits["bottom"] - center_r,
        "left": center_c - limits["left"],
        "right": limits["right"] - center_c,
    }
    return min((side for side in sides), key=lambda side: distances[side])


def _same_width_counterexample(specs: list[dict]) -> bool:
    by_side: dict[str, list[int]] = {}
    for spec in specs:
        by_side.setdefault(spec["side"], []).append(spec["pattern"][-1] + 1)
    for widths in by_side.values():
        if len(widths) >= 2 and len(widths) != len(set(widths)):
            return True
    return False


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    specs: list[dict],
    block_infos: list[dict],
    thickness: int,
    sides: tuple[str, ...],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    recovered_notches = [feature for side in SIDES for feature in _scan_side_notches(inp, side, wall=WALL)]
    if len(recovered_notches) != len(specs):
        return False
    if len({feature["signature"] for feature in recovered_notches}) != len(recovered_notches):
        return False
    if set(feature["side"] for feature in recovered_notches) != set(sides):
        return False
    if not _same_width_counterexample(specs):
        return False

    limits = _wall_inner_limits(inp.shape, thickness, sides)
    if len(sides) > 1:
        if not any(_nearest_side(info["input_origin"], info["patch"], limits, sides) != info["side"] for info in block_infos):
            return False

    for info in block_infos:
        if info["signature"] not in {feature["signature"] for feature in recovered_notches}:
            return False
        if info["input_origin"] == info["final_origin"]:
            return False

    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(28, 30)
    w = random.randint(28, 30)
    thickness = random.randint(2, 3)
    sides = _choose_sides()
    block_count = _choose_block_count(len(sides))
    specs = _assign_signatures(sides, block_count)

    wall_in = _build_wall((h, w), thickness, sides)
    specs = _add_outer_tabs(wall_in, specs, thickness, sides)
    if specs is None:
        raise ValueError("could not place wall tabs")

    expected = wall_in.copy()
    for spec in specs:
        spec["target_cells"] = _target_cells_for_spec(spec, (h, w), thickness)
        spec["signature"] = (spec["side"], spec["pattern"])
        for r, c in spec["target_cells"]:
            expected[r, c] = BG

    final_grid = expected.copy()
    colors = random.sample(BLOCK_COLORS, len(specs))
    block_infos: list[dict] = []
    for spec, color in zip(specs, colors):
        placed = False
        for _ in range(120):
            patch, teeth = _make_block_patch(spec["side"], spec["pattern"], color)
            origin = _patch_target_origin(spec["side"], patch, teeth, spec["target_cells"])
            if not _can_overlay(final_grid, patch, origin[0], origin[1]):
                continue
            paste(final_grid, patch, origin[0], origin[1], transparent=BG)
            block_infos.append(
                {
                    "side": spec["side"],
                    "pattern": spec["pattern"],
                    "signature": spec["signature"],
                    "patch": patch,
                    "final_origin": (int(origin[0]), int(origin[1])),
                    "target_cells": spec["target_cells"],
                }
            )
            placed = True
            break
        if not placed:
            raise ValueError("could not build a flush final block")

    inp = _place_input_blocks(wall_in, block_infos, thickness, sides)
    if inp is None:
        raise ValueError("could not place input blocks")

    if not _validate_case(inp, final_grid, specs, block_infos, thickness, sides):
        raise ValueError("semantic validation failed")

    return inp, final_grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            inp, _out = _generate_case()
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid lockhole sample")
