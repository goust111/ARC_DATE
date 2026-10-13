from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, is_contiguous, paste, random_sprite


Coord = tuple[int, int]


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _component_colors(grid: np.ndarray, component: dict) -> set[int]:
    return {int(grid[int(r), int(c)]) for r, c in component["coords"]}


def _mono_component_color(grid: np.ndarray, component: dict) -> int:
    colors = _component_colors(grid, component)
    assert len(colors) == 1
    return next(iter(colors))


def _is_solid_2x2(grid: np.ndarray, component: dict) -> bool:
    if int(component["area"]) != 4:
        return False
    r0, c0, r1, c1 = map(int, component["bbox"])
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    color = _mono_component_color(grid, component)
    return bool((grid[r0 : r1 + 1, c0 : c1 + 1] == color).all())


def _movement_from_bboxes(sprite_bbox: tuple[int, int, int, int], anchor_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    sr0, sc0, sr1, sc1 = map(int, sprite_bbox)
    ar0, ac0, ar1, ac1 = map(int, anchor_bbox)
    candidates: list[tuple[int, int]] = []
    if sc1 < ac0 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append((1, 1))
    if ac1 < sc0 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append((1, -1))
    if sr1 < ar0 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append((0, 1))
    if ar1 < sr0 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append((0, -1))
    assert len(candidates) == 1
    return candidates[0]


def _sort_key(axis_value: int, sign: int) -> int:
    return -int(axis_value) if sign > 0 else int(axis_value)


def _transform(grid: np.ndarray, direction: str) -> np.ndarray:
    if direction == "right":
        return grid.copy()
    if direction == "left":
        return np.fliplr(grid)
    if direction == "down":
        return grid.T.copy()
    if direction == "up":
        return np.flipud(grid.T)
    raise ValueError(f"unknown direction {direction}")


def _paint_cells(out: np.ndarray, cells: list[tuple[int, int, int]], bg: int) -> None:
    for r, c, color in cells:
        assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
        assert int(out[r, c]) == int(bg), "paint collision"
        out[r, c] = int(color)


def _detect_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert len(components) >= 4

    sprite_idx = None
    anchor_idx = None
    for idx, comp in enumerate(components):
        colors = _component_colors(g, comp)
        if len(colors) >= 2:
            assert sprite_idx is None, "expected exactly one multicolor sprite"
            sprite_idx = idx
        if _is_solid_2x2(g, comp):
            assert anchor_idx is None, "expected exactly one 2x2 anchor"
            anchor_idx = idx

    assert sprite_idx is not None
    assert anchor_idx is not None and anchor_idx != sprite_idx

    sprite = components[sprite_idx]
    anchor = components[anchor_idx]
    axis, sign = _movement_from_bboxes(sprite["bbox"], anchor["bbox"])

    gate_components = [comp for idx, comp in enumerate(components) if idx not in {sprite_idx, anchor_idx}]
    assert gate_components

    sr0, sc0, sr1, sc1 = map(int, sprite["bbox"])
    if axis == 1:
        perp0, perp1 = sr0, sr1
        source_edge = sc1 if sign > 0 else sc0
        anchor_edge = anchor["bbox"][1] if sign > 0 else anchor["bbox"][3]
    else:
        perp0, perp1 = sc0, sc1
        source_edge = sr1 if sign > 0 else sr0
        anchor_edge = anchor["bbox"][0] if sign > 0 else anchor["bbox"][2]

    gates_by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in gate_components:
        color = _mono_component_color(g, comp)
        coords = comp["coords"]
        line_values = {int(rc[axis]) for rc in coords}
        assert len(line_values) == 1
        perp_values = [int(rc[1 - axis]) for rc in coords]
        gates_by_color[color].append(
            {
                "component": comp,
                "line": next(iter(line_values)),
                "perp_min": min(perp_values),
                "perp_max": max(perp_values),
            }
        )

    gates: list[dict] = []
    for color, pieces in gates_by_color.items():
        assert len(pieces) == 2
        assert pieces[0]["line"] == pieces[1]["line"]
        line = int(pieces[0]["line"])
        lower = min(pieces, key=lambda item: item["perp_max"])
        upper = max(pieces, key=lambda item: item["perp_min"])
        assert lower["perp_max"] < perp0
        assert upper["perp_min"] > perp1
        if sign > 0:
            assert source_edge < line < anchor_edge
        else:
            assert anchor_edge < line < source_edge
        gates.append({"color": int(color), "line": line, "pieces": [piece["component"] for piece in pieces]})

    gates.sort(key=lambda item: item["line"], reverse=sign < 0)
    assert len(gates) == len({gate["line"] for gate in gates}), "gate lines must be distinct"

    return {
        "grid": g,
        "bg": bg,
        "sprite": sprite,
        "anchor": anchor,
        "gates": gates,
        "axis": axis,
        "sign": sign,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the multicolor sprite toward the 2x2 anchor while each colored gate detains matching pixels on its own line.

    Input structure:
    - The background is the most common color.
    - One 8-connected multicolor sprite and one solid single-color 2x2 square anchor are separated along exactly one cardinal direction.
    - Several gates lie between them: each gate is a pair of same-color short segments on the two sides of the corridor, and gate colors are distinct.

    Rule:
    - Move the sprite straight toward the anchor.
    - Whenever the sprite crosses a gate line, all sprite pixels whose color matches that gate are detained on that line, keeping their perpendicular coordinate; multiple detained pixels on the same motion axis stack from the gate line back toward the source in arrival order.
    - Pixels whose colors do not match any gate keep moving until the remaining sprite's front edge touches the anchor's near side; erase the original sprite and keep all gates and the anchor fixed.

    Key insight:
    - Each gate color removes one subset of the traveling sprite, and the back-stacked order on every gate line preserves the original front-to-back order of those pixels inside the moving body.
    """
    scene = _detect_scene(np.asarray(grid, dtype=int))
    g = scene["grid"]
    bg = int(scene["bg"])
    axis = int(scene["axis"])
    sign = int(scene["sign"])
    sprite = scene["sprite"]
    anchor = scene["anchor"]
    gates = scene["gates"]

    out = np.full_like(g, bg)
    for comp in [anchor] + [piece for gate in gates for piece in gate["pieces"]]:
        color = _mono_component_color(g, comp)
        for r, c in comp["coords"]:
            out[int(r), int(c)] = color

    gate_by_color = {gate["color"]: gate["line"] for gate in gates}
    detained: dict[int, list[tuple[int, int]]] = defaultdict(list)
    remaining: list[tuple[int, int, int]] = []
    for raw_r, raw_c in sprite["coords"]:
        r, c = int(raw_r), int(raw_c)
        color = int(g[r, c])
        if color in gate_by_color:
            detained[color].append((r, c))
        else:
            remaining.append((r, c, color))

    detained_cells: list[tuple[int, int, int]] = []
    for gate in gates:
        color = int(gate["color"])
        line = int(gate["line"])
        rows_or_cols: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r, c in detained.get(color, []):
            rows_or_cols[int(r if axis == 1 else c)].append((r, c))
        for perp, cells in rows_or_cols.items():
            cells.sort(key=lambda rc: _sort_key(rc[axis], sign))
            for idx, (r, c) in enumerate(cells):
                dest_axis = line - sign * idx
                if axis == 1:
                    detained_cells.append((perp, dest_axis, color))
                else:
                    detained_cells.append((dest_axis, perp, color))

    moved_remaining: list[tuple[int, int, int]] = []
    if remaining:
        anchor_bbox = tuple(map(int, anchor["bbox"]))
        near_side = anchor_bbox[1] if axis == 1 and sign > 0 else anchor_bbox[3] if axis == 1 else anchor_bbox[0] if sign > 0 else anchor_bbox[2]
        target_front = near_side - sign
        current_front = max(cell[axis] for cell in remaining) if sign > 0 else min(cell[axis] for cell in remaining)
        delta = int(target_front - current_front)
        for r, c, color in remaining:
            if axis == 1:
                moved_remaining.append((r, c + delta, color))
            else:
                moved_remaining.append((r + delta, c, color))

    _paint_cells(out, detained_cells, bg)
    _paint_cells(out, moved_remaining, bg)
    return out


_GEN_RETRIES = 400
_LOCAL_H_RANGE = (3, 5)
_LOCAL_W_RANGE = (4, 7)
_GATE_GAP_RANGE = (2, 4)
_GATE_SEG_LEN_CHOICES = (1, 2)
_GLOBAL_MARGIN_RANGE = (0, 4)
_DIRECTIONS = ("right", "left", "down", "up")


def _canonical_mask() -> np.ndarray | None:
    h = random.randint(*_LOCAL_H_RANGE)
    w = random.randint(*_LOCAL_W_RANGE)
    for _ in range(80):
        density = random.uniform(0.55, 0.8)
        sprite = random_sprite(
            h,
            w,
            density=density,
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=8,
            background=0,
        )
        mask = sprite != 0
        area = int(mask.sum())
        if area < max(7, h + w - 1) or area > int(0.85 * h * w):
            continue
        line_lengths = [int(mask[r, :].sum()) for r in range(h)]
        if max(line_lengths) < 2 or sum(length > 0 for length in line_lengths) < 2:
            continue
        if not is_contiguous(mask, background=False, connectivity=8):
            continue
        return mask
    return None


def _canonical_row_groups(mask: np.ndarray) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for r in range(mask.shape[0]):
        cols = [int(c) for c in np.where(mask[r])[0]]
        if cols:
            groups[int(r)] = cols
    return groups


def _color_canonical_sprite(mask: np.ndarray, colors: list[int], front_detained: bool, present_gate_count: int) -> tuple[np.ndarray, dict]:
    h, w = mask.shape
    occupied = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    front_col = max(c for _r, c in occupied)
    front_cells = [(r, c) for r, c in occupied if c == front_col]
    non_front = [(r, c) for r, c in occupied if c != front_col]
    if front_detained and not non_front:
        raise ValueError("need at least one non-front cell")

    row_groups = _canonical_row_groups(mask)
    stack_rows = [r for r, cols in row_groups.items() if len(cols) >= 2]
    if not stack_rows:
        raise ValueError("need a stackable row")
    preferred_rows = [r for r in stack_rows if row_groups[r][-1] == front_col]
    stack_row = random.choice(preferred_rows or stack_rows)

    gate_colors = colors[:present_gate_count]
    other_colors = colors[present_gate_count:]
    assert gate_colors and other_colors

    assignment: dict[tuple[int, int], int] = {}
    stack_color = gate_colors[0]

    if front_detained:
        for cell in front_cells:
            assignment[cell] = stack_color

    stack_candidates = [(stack_row, c) for c in row_groups[stack_row]]
    random.shuffle(stack_candidates)
    for cell in stack_candidates:
        if cell not in assignment:
            assignment[cell] = stack_color
        if Counter(assignment.values())[stack_color] >= max(2, len(front_cells)):
            break
    if Counter(assignment.values())[stack_color] < 2:
        raise ValueError("failed to form a detained stack")

    available = [cell for cell in occupied if cell not in assignment]
    random.shuffle(available)

    if not front_detained:
        ungated_front = random.choice(front_cells)
        assignment[ungated_front] = other_colors[0]
        available = [cell for cell in available if cell != ungated_front]
        remaining_front_cells = [cell for cell in front_cells if cell != ungated_front]
        if remaining_front_cells:
            gate_front_color = gate_colors[min(1, present_gate_count - 1)]
            for cell in remaining_front_cells[: random.randint(0, len(remaining_front_cells))]:
                if cell not in assignment:
                    assignment[cell] = gate_front_color

    for color in gate_colors[1:]:
        if not available:
            break
        count = 1 if len(available) == 1 else random.randint(1, min(2, len(available)))
        for cell in available[:count]:
            assignment[cell] = color
        available = [cell for cell in available if cell not in assignment]

    for cell in occupied:
        if cell not in assignment:
            assignment[cell] = random.choice(other_colors)

    used_colors = {assignment[cell] for cell in occupied}
    if len(used_colors) < 3:
        spare_cells = [cell for cell in occupied if assignment[cell] != stack_color]
        if spare_cells:
            assignment[random.choice(spare_cells)] = other_colors[-1]
            used_colors = {assignment[cell] for cell in occupied}
    if len(used_colors) < 3:
        raise ValueError("sprite should expose at least three colors")
    if not all(any(assignment[cell] == color for cell in occupied) for color in gate_colors):
        raise ValueError("every present gate color must appear in the sprite")

    remaining_cells = [cell for cell in occupied if assignment[cell] not in gate_colors]
    if not remaining_cells:
        raise ValueError("need ungated remainder")

    if front_detained:
        remaining_front = max(c for _r, c in remaining_cells)
        if remaining_front >= front_col:
            raise ValueError("front_detained branch not achieved")
    else:
        remaining_front = max(c for _r, c in remaining_cells)
        if remaining_front != front_col:
            raise ValueError("front should still be visible")

    sprite = np.zeros((h, w), dtype=int)
    for (r, c), color in assignment.items():
        sprite[r, c] = int(color)

    return sprite, {
        "assignment": assignment,
        "gate_colors": gate_colors,
        "remaining_cells": remaining_cells,
        "front_col": front_col,
        "stack_color": stack_color,
    }


def _canonical_expected(sprite: np.ndarray, bg: int, anchor_color: int, gate_specs: list[dict], anchor_top: int, anchor_left: int) -> np.ndarray:
    h, w = sprite.shape
    top_margin = 2
    local_h = top_margin + h + 2
    local_w = anchor_left + 2
    out = np.full((local_h, local_w), bg, dtype=int)
    source_r = top_margin

    out[anchor_top : anchor_top + 2, anchor_left : anchor_left + 2] = anchor_color
    for gate in gate_specs:
        col = gate["col"]
        seg_len = gate["seg_len"]
        color = gate["color"]
        out[source_r - seg_len : source_r, col] = color
        out[source_r + h : source_r + h + seg_len, col] = color

    gate_by_color = {gate["color"]: gate["col"] for gate in gate_specs}
    gated_rows: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    remaining: list[tuple[int, int, int]] = []

    for r, c in np.argwhere(sprite != 0):
        rr, cc = int(r), int(c)
        color = int(sprite[rr, cc])
        if color in gate_by_color:
            gated_rows[color][rr].append(cc)
        else:
            remaining.append((rr, cc, color))

    for gate in gate_specs:
        color = gate["color"]
        col = gate["col"]
        for row, cols in gated_rows[color].items():
            for idx, src_col in enumerate(sorted(cols, reverse=True)):
                dest_r = source_r + row
                dest_c = col - idx
                assert int(out[dest_r, dest_c]) == int(bg)
                out[dest_r, dest_c] = color

    assert remaining
    delta = anchor_left - 1 - max(c for _r, c, _color in remaining)
    for r, c, color in remaining:
        dest_r = source_r + r
        dest_c = c + delta
        assert int(out[dest_r, dest_c]) == int(bg)
        out[dest_r, dest_c] = color

    return out


def _canonical_input(sprite: np.ndarray, bg: int, anchor_color: int, gate_specs: list[dict], anchor_top: int, anchor_left: int) -> np.ndarray:
    h, w = sprite.shape
    top_margin = 2
    local_h = top_margin + h + 2
    local_w = anchor_left + 2
    inp = np.full((local_h, local_w), bg, dtype=int)
    source_r = top_margin
    paste(inp, sprite, source_r, 0, transparent=0)
    inp[anchor_top : anchor_top + 2, anchor_left : anchor_left + 2] = anchor_color
    for gate in gate_specs:
        col = gate["col"]
        seg_len = gate["seg_len"]
        color = gate["color"]
        inp[source_r - seg_len : source_r, col] = color
        inp[source_r + h : source_r + h + seg_len, col] = color
    return inp


def _generate_canonical_pair() -> tuple[np.ndarray, np.ndarray] | None:
    mask = _canonical_mask()
    if mask is None:
        return None

    gate_pattern = random.choice([(2, 0), (2, 1), (1, 1), (3, 0)])
    present_gate_count, absent_gate_count = gate_pattern
    front_detained = bool(random.getrandbits(1))

    bg = random.randrange(10)
    palette = [color for color in range(10) if color not in {bg, 0}]
    random.shuffle(palette)

    used_sprite_color_count = present_gate_count + (2 if present_gate_count == 1 else 1)
    sprite_colors = palette[:used_sprite_color_count]
    anchor_color = palette[used_sprite_color_count]
    absent_gate_colors = palette[used_sprite_color_count + 1 : used_sprite_color_count + 1 + absent_gate_count]
    if len(absent_gate_colors) != absent_gate_count:
        return None

    try:
        sprite, sprite_meta = _color_canonical_sprite(mask, sprite_colors, front_detained, present_gate_count)
    except ValueError:
        return None

    gate_colors = list(sprite_meta["gate_colors"]) + absent_gate_colors
    if len(gate_colors) < 2:
        return None

    remaining_cells = sprite_meta["remaining_cells"]
    rem_cols = [c for _r, c in remaining_cells]
    rem_span = max(rem_cols) - min(rem_cols)
    h, w = sprite.shape

    gate_cols: list[int] = []
    cur = w - 1
    for _ in gate_colors:
        cur += random.randint(*_GATE_GAP_RANGE)
        gate_cols.append(cur)

    clearance = random.randint(2, 4)
    anchor_left = gate_cols[-1] + rem_span + clearance + 1
    anchor_top = random.randint(2, h)
    gate_specs = [
        {"color": int(color), "col": int(col), "seg_len": random.choice(_GATE_SEG_LEN_CHOICES)}
        for color, col in zip(gate_colors, gate_cols)
    ]

    inp = _canonical_input(sprite, bg, anchor_color, gate_specs, anchor_top, anchor_left)
    out = _canonical_expected(sprite, bg, anchor_color, gate_specs, anchor_top, anchor_left)

    if np.array_equal(inp, out):
        return None

    last_gate = gate_cols[-1]
    final_min_col = min(c for c in np.where(out != bg)[1] if c > last_gate) if np.any(out[:, last_gate + 1 :] != bg) else None
    if final_min_col is None or final_min_col <= last_gate:
        return None

    return inp, out


def _embed_scene(local_inp: np.ndarray, local_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = random.choice(_DIRECTIONS)
    bg = int(get_bg(local_inp))
    tinp = _transform(local_inp, direction)
    tout = _transform(local_out, direction)

    extra_top = random.randint(*_GLOBAL_MARGIN_RANGE)
    extra_left = random.randint(*_GLOBAL_MARGIN_RANGE)
    extra_bottom = random.randint(*_GLOBAL_MARGIN_RANGE)
    extra_right = random.randint(*_GLOBAL_MARGIN_RANGE)

    H = tinp.shape[0] + extra_top + extra_bottom
    W = tinp.shape[1] + extra_left + extra_right
    if H > 30 or W > 30:
        raise ValueError("scene exceeds ARC bounds")

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)
    paste(inp, tinp, extra_top, extra_left, transparent=bg)
    paste(out, tout, extra_top, extra_left, transparent=bg)
    return inp, out


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        scene = _detect_scene(inp)
    except Exception:
        return False

    gates = scene["gates"]
    gate_colors = {gate["color"] for gate in gates}
    sprite_color_counter = Counter(int(inp[int(r), int(c)]) for r, c in scene["sprite"]["coords"])
    if len(gates) < 2 or not any(sprite_color_counter[color] > 0 for color in gate_colors):
        return False

    stackable = False
    axis = int(scene["axis"])
    by_color_perp: dict[int, Counter] = defaultdict(Counter)
    for r, c in scene["sprite"]["coords"]:
        color = int(inp[int(r), int(c)])
        if color in gate_colors:
            perp = int(r if axis == 1 else c)
            by_color_perp[color][perp] += 1
            if by_color_perp[color][perp] >= 2:
                stackable = True
    if not stackable:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        pair = _generate_canonical_pair()
        if pair is None:
            continue
        local_inp, local_out = pair
        try:
            inp, expected = _embed_scene(local_inp, local_out)
        except ValueError:
            continue
        if _validate_generated(inp, expected):
            return inp
    raise RuntimeError(f"failed to generate a valid customs-corridor input after {_GEN_RETRIES} attempts")
