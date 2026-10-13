from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
PALETTE = [color for color in Color.NOT_BLACK]
GEN_RETRIES = 400


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _component_colors(grid: np.ndarray, component: dict) -> set[int]:
    coords = component["coords"]
    return {int(grid[int(r), int(c)]) for r, c in coords}


def _find_anchor_and_marker(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], tuple[int, int], int]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False)

    marker_candidates = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(marker_candidates) == 1, "expected exactly one isolated marker pixel"
    marker = marker_candidates[0]["coords"][0]
    marker_rc = (int(marker[0]), int(marker[1]))
    marker_color = int(grid[marker_rc])

    anchor_candidates = []
    for comp in comps:
        if int(comp["area"]) < 4:
            continue
        h, w = _bbox_hw(comp["bbox"])
        if h <= 1 or w <= 1:
            continue
        if int(comp["area"]) != h * w:
            continue
        if len(_component_colors(grid, comp)) != 1:
            continue
        anchor_candidates.append(comp)

    assert len(anchor_candidates) == 1, "expected exactly one solid anchor block"
    return anchor_candidates[0]["bbox"], marker_rc, marker_color


def _extract_legend_map(grid: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    for comp in comps:
        if int(comp["area"]) != 2:
            continue
        colors = _component_colors(grid, comp)
        if len(colors) != 2:
            continue

        r0, c0, r1, c1 = comp["bbox"]
        h, w = _bbox_hw(comp["bbox"])
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if h == 1 and w == 2:
            coords.sort(key=lambda rc: rc[1])
        elif h == 2 and w == 1:
            coords.sort(key=lambda rc: rc[0])
        else:
            continue

        src = int(grid[coords[0]])
        dst = int(grid[coords[1]])
        assert src not in mapping, "legend sources must be unique"
        mapping[src] = dst

    assert len(mapping) >= 2, "expected a nontrivial color legend"
    assert set(mapping.keys()) == set(mapping.values()), "legend must define a visible closed evolution cycle"
    return mapping


def _path_spec(
    marker: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
) -> tuple[bool, int, list[int]]:
    mr, mc = marker
    r0, c0, r1, c1 = anchor_bbox

    if r0 <= mr <= r1 and mc < c0:
        return True, int(mr), list(range(int(mc) + 1, int(c0)))
    if r0 <= mr <= r1 and mc > c1:
        return True, int(mr), list(range(int(mc) - 1, int(c1), -1))
    if c0 <= mc <= c1 and mr < r0:
        return False, int(mc), list(range(int(mr) + 1, int(r0)))
    if c0 <= mc <= c1 and mr > r1:
        return False, int(mc), list(range(int(mr) - 1, int(r1), -1))
    raise AssertionError("marker must align with the anchor along one axis")


def _perpendicular_segments(grid: np.ndarray, bg: int, horizontal_path: bool) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h, w = _bbox_hw(comp["bbox"])
        if horizontal_path:
            if w != 1 or h < 2:
                continue
            groups.setdefault(int(c0), []).append(comp)
        else:
            if h != 1 or w < 2:
                continue
            groups.setdefault(int(r0), []).append(comp)
    return groups


def _gate_state(segments: list[dict], path_coord: int, horizontal_path: bool) -> str | None:
    if not segments:
        return None

    if horizontal_path:
        start_idx, end_idx = 0, 2
    else:
        start_idx, end_idx = 1, 3

    crossing = [seg for seg in segments if int(seg["bbox"][start_idx]) <= path_coord <= int(seg["bbox"][end_idx])]
    if crossing:
        return "closed"

    before = [seg for seg in segments if int(seg["bbox"][end_idx]) == path_coord - 1]
    after = [seg for seg in segments if int(seg["bbox"][start_idx]) == path_coord + 1]
    matches = [
        (seg_a, seg_b)
        for seg_a in before
        for seg_b in after
        if int(seg_a["color"]) == int(seg_b["color"])
    ]
    if matches:
        return "open"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the marker toward the anchor and evolve its color once per traversed open gate.

    Input structure:
    - The background contains one solid anchor block and one isolated marker pixel aligned with it by row or column.
    - Between marker and anchor, perpendicular gate segments may either span the path cell (closed) or leave a one-cell gap exactly on the path (open).
    - Elsewhere there are unrelated line-segment distractors plus adjacent two-cell legend dominoes that show a visible color-to-color evolution.

    Rule:
    - Remove the marker from its start cell and move it straight toward the anchor.
    - Each time the marker passes a true open gate, update its color by one legend step.
    - Stop immediately before the first closed gate; if no closed gate blocks the path, stop on the last empty cell next to the anchor.
    - Keep the anchor, all segments, distractors, and legends unchanged; only the marker moves and may recolor.

    Key insight:
    - The same ray scan simultaneously determines both the stopping cell and the final color: only segments that genuinely straddle the marker-to-anchor ray count as gates, and the legend is iterated exactly once per passed open gate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    anchor_bbox, marker, color = _find_anchor_and_marker(g, bg)
    legend_map = _extract_legend_map(g, bg)
    assert color in legend_map, "marker color must participate in the legend cycle"

    horizontal_path, path_coord, positions = _path_spec(marker, anchor_bbox)
    groups = _perpendicular_segments(g, bg, horizontal_path)

    final = marker
    for pos in positions:
        state = _gate_state(groups.get(int(pos), []), path_coord, horizontal_path)
        if state == "closed":
            break
        final = (path_coord, int(pos)) if horizontal_path else (int(pos), path_coord)
        if state == "open":
            color = int(legend_map[color])

    out = g.copy()
    out[marker] = bg
    out[final] = color
    return out


def _build_track(axis_len: int, block_size: int, gate_count: int, positive: bool) -> dict:
    min_dist = 2 * gate_count + 2
    max_dist = axis_len - block_size - 3
    if max_dist < min_dist:
        raise ValueError("axis too short for requested gate count")

    dist = random.randint(min_dist, max_dist)
    gate_steps = [int(idx) + 2 for idx in _sample_non_adjacent_slots(dist - 3, gate_count)]

    if positive:
        anchor_start = random.randint(dist + 1, axis_len - block_size - 2)
        anchor_end = anchor_start + block_size - 1
        marker = anchor_start - dist

        def to_abs(step: int) -> int:
            return int(marker + step)

    else:
        anchor_end = random.randint(block_size, axis_len - dist - 2)
        anchor_start = anchor_end - block_size + 1
        marker = anchor_end + dist

        def to_abs(step: int) -> int:
            return int(marker - step)

    return {
        "marker": int(marker),
        "anchor_start": int(anchor_start),
        "anchor_end": int(anchor_end),
        "path_positions": [to_abs(step) for step in range(1, dist)],
        "gate_positions": [to_abs(step) for step in gate_steps],
    }


def _sample_gate_states(gate_positions: list[int]) -> dict[int, str]:
    first_closed = None if random.random() < 0.5 else random.randrange(len(gate_positions))
    states: dict[int, str] = {}
    for idx, pos in enumerate(gate_positions):
        if first_closed is None or idx < first_closed:
            states[int(pos)] = "open"
        elif idx == first_closed:
            states[int(pos)] = "closed"
        else:
            states[int(pos)] = random.choice(["open", "closed"])
    return states


def _occupy(placement: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    occ = (patch != BG).astype(int)
    paste(placement, occ, int(r0), int(c0), transparent=0)


def _place_patch(grid: np.ndarray, placement: np.ndarray, patch: np.ndarray) -> None:
    r0, c0 = random_free_location(
        placement,
        patch,
        transparent=BG,
        border=1,
        padding=1,
        padding_connectivity=4,
    )
    paste(grid, patch, int(r0), int(c0), transparent=BG)
    _occupy(placement, patch, int(r0), int(c0))


def _line_patch(length: int, vertical: bool, color: int) -> np.ndarray:
    if vertical:
        return np.full((int(length), 1), int(color), dtype=int)
    return np.full((1, int(length)), int(color), dtype=int)


def _expected_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    marker = meta["marker"]
    color = int(meta["initial_color"])
    final = marker
    out[marker] = int(meta["bg"])

    for cell in meta["path_cells"]:
        state = meta["cell_states"].get(cell)
        if state == "closed":
            break
        final = cell
        if state == "open":
            color = int(meta["legend_map"][color])

    out[final] = color
    return out


def _sample_non_adjacent_slots(max_len: int, n_slots: int) -> list[int]:
    free_len = int(max_len) - int(n_slots) + 1
    if free_len < n_slots:
        raise ValueError("not enough slots for spaced gate placement")
    base = sorted(random.sample(range(free_len), int(n_slots)))
    return [int(idx) + offset for offset, idx in enumerate(base)]


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    height = random.randint(18, 29)
    width = random.randint(18, 29)
    horizontal_path = random.choice([True, False])

    cycle_len = random.randint(3, 4)
    cycle_colors = random.sample(PALETTE, cycle_len)
    legend_pairs = [
        (int(cycle_colors[idx]), int(cycle_colors[(idx + 1) % cycle_len]))
        for idx in range(cycle_len)
    ]
    legend_map = {src: dst for src, dst in legend_pairs}
    marker_color = random.choice(cycle_colors)

    anchor_choices = [color for color in PALETTE if color not in cycle_colors]
    anchor_color = random.choice(anchor_choices)
    segment_palette = [color for color in PALETTE if color not in cycle_colors and color != anchor_color]
    if not segment_palette:
        segment_palette = [color for color in PALETTE if color != anchor_color]

    gate_count = random.randint(2, 5)
    anchor_h = random.randint(2, 4)
    anchor_w = random.randint(2, 4)
    grid = np.full((height, width), BG, dtype=int)
    placement = np.zeros_like(grid, dtype=int)

    if horizontal_path:
        path_row = random.randint(4, height - 5)
        r0_low = max(1, path_row - anchor_h + 1)
        r0_high = min(path_row, height - anchor_h - 2)
        if r0_low > r0_high:
            raise ValueError("anchor rows do not fit")
        anchor_r0 = random.randint(r0_low, r0_high)
        anchor_r1 = anchor_r0 + anchor_h - 1

        track = _build_track(width, anchor_w, gate_count, positive=random.choice([True, False]))
        anchor_c0 = int(track["anchor_start"])
        anchor_c1 = int(track["anchor_end"])
        marker = (int(path_row), int(track["marker"]))

        anchor_patch = np.full((anchor_h, anchor_w), anchor_color, dtype=int)
        paste(grid, anchor_patch, anchor_r0, anchor_c0, transparent=BG)
        _occupy(placement, anchor_patch, anchor_r0, anchor_c0)
        grid[marker] = marker_color
        placement[marker] = 1

        gate_states = _sample_gate_states(track["gate_positions"])
        for gate_col in track["gate_positions"]:
            gate_color = random.choice(segment_palette)
            top = random.randint(1, path_row - 2)
            bottom = random.randint(path_row + 2, height - 2)
            if gate_states[gate_col] == "closed":
                grid[top : bottom + 1, int(gate_col)] = gate_color
                placement[top : bottom + 1, int(gate_col)] = 1
            else:
                grid[top:path_row, int(gate_col)] = gate_color
                grid[path_row + 1 : bottom + 1, int(gate_col)] = gate_color
                placement[top:path_row, int(gate_col)] = 1
                placement[path_row + 1 : bottom + 1, int(gate_col)] = 1

        placement[path_row, :] = 1
        path_cells = [(int(path_row), int(col)) for col in track["path_positions"]]
        cell_states = {(int(path_row), int(col)): gate_states[int(col)] for col in track["gate_positions"]}
    else:
        path_col = random.randint(4, width - 5)
        c0_low = max(1, path_col - anchor_w + 1)
        c0_high = min(path_col, width - anchor_w - 2)
        if c0_low > c0_high:
            raise ValueError("anchor cols do not fit")
        anchor_c0 = random.randint(c0_low, c0_high)
        anchor_c1 = anchor_c0 + anchor_w - 1

        track = _build_track(height, anchor_h, gate_count, positive=random.choice([True, False]))
        anchor_r0 = int(track["anchor_start"])
        anchor_r1 = int(track["anchor_end"])
        marker = (int(track["marker"]), int(path_col))

        anchor_patch = np.full((anchor_h, anchor_w), anchor_color, dtype=int)
        paste(grid, anchor_patch, anchor_r0, anchor_c0, transparent=BG)
        _occupy(placement, anchor_patch, anchor_r0, anchor_c0)
        grid[marker] = marker_color
        placement[marker] = 1

        gate_states = _sample_gate_states(track["gate_positions"])
        for gate_row in track["gate_positions"]:
            gate_color = random.choice(segment_palette)
            left = random.randint(1, path_col - 2)
            right = random.randint(path_col + 2, width - 2)
            if gate_states[gate_row] == "closed":
                grid[int(gate_row), left : right + 1] = gate_color
                placement[int(gate_row), left : right + 1] = 1
            else:
                grid[int(gate_row), left:path_col] = gate_color
                grid[int(gate_row), path_col + 1 : right + 1] = gate_color
                placement[int(gate_row), left:path_col] = 1
                placement[int(gate_row), path_col + 1 : right + 1] = 1

        placement[:, path_col] = 1
        path_cells = [(int(row), int(path_col)) for row in track["path_positions"]]
        cell_states = {(int(row), int(path_col)): gate_states[int(row)] for row in track["gate_positions"]}

    legend_order = legend_pairs[:]
    random.shuffle(legend_order)
    for src, dst in legend_order:
        legend_patch = np.array([[int(src), int(dst)]], dtype=int)
        _place_patch(grid, placement, legend_patch)

    distractor_count = random.randint(1, 3)
    gate_vertical = bool(horizontal_path)
    orientations = [gate_vertical]
    while len(orientations) < distractor_count:
        orientations.append(random.choice([gate_vertical, not gate_vertical]))
    random.shuffle(orientations)

    for vertical in orientations:
        length = random.randint(3, 6)
        distractor_color = random.choice(segment_palette)
        distractor_patch = _line_patch(length, vertical, distractor_color)
        _place_patch(grid, placement, distractor_patch)

    meta = {
        "bg": BG,
        "marker": marker,
        "initial_color": marker_color,
        "legend_map": legend_map,
        "path_cells": path_cells,
        "cell_states": cell_states,
        "anchor_bbox": (anchor_r0, anchor_c0, anchor_r1, anchor_c1),
    }
    expected = _expected_output(grid, meta)
    return grid, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected, _meta = _generate_case()
            out = main(inp)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate a valid gate-chain recolor input")
