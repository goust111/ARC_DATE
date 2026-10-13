from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
BLOCK_COLORS = (BLUE, RED, GREEN)
TRACK_STEP = 3
BLOCK_SIZE = 2
GEN_MAX_ATTEMPTS = 400
COLOR_ATTEMPTS = 200


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_blocks(grid: np.ndarray) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    for color in BLOCK_COLORS:
        for comp in _components_of_color(grid, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if len(comp) != 4 or r1 - r0 != 1 or c1 - c0 != 1:
                raise AssertionError("colored objects must be solid 2x2 blocks")
            if any(int(grid[r, c]) != color for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
                raise AssertionError("colored objects must be solid rectangles")
            blocks.append({"color": color, "r0": r0, "c0": c0})
    if not blocks:
        raise AssertionError("expected at least one colored block")
    return sorted(blocks, key=lambda b: (b["r0"], b["c0"], b["color"]))


def _detect_orientation(grey_coords: list[tuple[int, int]], blocks: list[dict[str, int]]) -> str:
    rows = {r for r, _ in grey_coords} | {b["r0"] for b in blocks}
    cols = {c for _, c in grey_coords} | {b["c0"] for b in blocks}
    if len(rows) < len(cols):
        return "horizontal"
    if len(cols) < len(rows):
        return "vertical"
    row_hits = max((sum(1 for r, _ in grey_coords if r == rr) for rr in rows), default=0)
    col_hits = max((sum(1 for _, c in grey_coords if c == cc) for cc in cols), default=0)
    return "horizontal" if row_hits >= col_hits else "vertical"


def _extract_scene(grid: np.ndarray) -> tuple[str, list[dict[str, object]], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    blocks = _find_blocks(g)
    grey_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    if not grey_coords:
        raise AssertionError("expected visible grey track dots")

    orientation = _detect_orientation(grey_coords, blocks)
    tracks: list[dict[str, object]] = []

    if orientation == "horizontal":
        track_rows = sorted({r for r, _ in grey_coords} | {b["r0"] for b in blocks})
        for row in track_rows:
            known = sorted({c for r, c in grey_coords if r == row} | {b["c0"] for b in blocks if b["r0"] == row})
            if len(known) < 2:
                raise AssertionError("horizontal track must expose at least two slots")
            full = list(range(known[0], known[-1] + 1, TRACK_STEP))
            if full != known:
                raise AssertionError("horizontal track slots must form a contiguous step-3 progression")
            tracks.append({"axis": row, "slots": full})
    else:
        track_cols = sorted({c for _, c in grey_coords} | {b["c0"] for b in blocks})
        for col in track_cols:
            known = sorted({r for r, c in grey_coords if c == col} | {b["r0"] for b in blocks if b["c0"] == col})
            if len(known) < 2:
                raise AssertionError("vertical track must expose at least two slots")
            full = list(range(known[0], known[-1] + 1, TRACK_STEP))
            if full != known:
                raise AssertionError("vertical track slots must form a contiguous step-3 progression")
            tracks.append({"axis": col, "slots": full})

    return orientation, tracks, blocks


def _move_blocks(
    orientation: str, tracks: list[dict[str, object]], blocks: list[dict[str, int]]
) -> list[dict[str, int]]:
    slot_map = {int(track["axis"]): list(track["slots"]) for track in tracks}
    moved: list[dict[str, int]] = []
    for block in blocks:
        axis = block["r0"] if orientation == "horizontal" else block["c0"]
        pos = block["c0"] if orientation == "horizontal" else block["r0"]
        slots = slot_map[axis]
        idx = slots.index(pos)
        next_pos = slots[min(idx + 1, len(slots) - 1)]
        if orientation == "horizontal":
            moved.append({"color": block["color"], "r0": block["r0"], "c0": next_pos})
        else:
            moved.append({"color": block["color"], "r0": next_pos, "c0": block["c0"]})
    return moved


def _paint_block(out: np.ndarray, block: dict[str, int]) -> None:
    r0 = int(block["r0"])
    c0 = int(block["c0"])
    out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = int(block["color"])


def _fill_between_aligned_pairs(out: np.ndarray, blocks: list[dict[str, int]]) -> None:
    for a, b in itertools.combinations(blocks, 2):
        if int(a["color"]) != int(b["color"]):
            continue

        if int(a["r0"]) == int(b["r0"]):
            left, right = sorted((a, b), key=lambda x: int(x["c0"]))
            fill_c0 = int(left["c0"]) + BLOCK_SIZE
            fill_c1 = int(right["c0"]) - 1
            if fill_c0 <= fill_c1:
                for r in range(int(left["r0"]), int(left["r0"]) + BLOCK_SIZE):
                    for c in range(fill_c0, fill_c1 + 1):
                        if int(out[r, c]) == BLACK:
                            out[r, c] = int(a["color"])

        if int(a["c0"]) == int(b["c0"]):
            top, bottom = sorted((a, b), key=lambda x: int(x["r0"]))
            fill_r0 = int(top["r0"]) + BLOCK_SIZE
            fill_r1 = int(bottom["r0"]) - 1
            if fill_r0 <= fill_r1:
                for r in range(fill_r0, fill_r1 + 1):
                    for c in range(int(top["c0"]), int(top["c0"]) + BLOCK_SIZE):
                        if int(out[r, c]) == BLACK:
                            out[r, c] = int(a["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 2x2 block one track slot forward, then connect same-color blocks that become row- or column-aligned.

    Input structure:
    - The grid has a black background with 2-3 parallel grey-dot tracks, all horizontal or all vertical.
    - Each track is a step-3 sequence of grey slot markers, with 1-2 colored 2x2 blocks covering some slots.
    - Colored blocks use blue, red, or green, while uncovered track slots remain visible as grey singleton dots.

    Rule:
    - Recover each track as the full step-3 progression formed by its visible grey dots together with the occupied slots under the 2x2 blocks.
    - Move every colored 2x2 block one slot forward along its own track, using increasing column order for horizontal tracks or increasing row order for vertical tracks; a block at the final slot stays put.
    - Redraw all grey slot markers, place the moved blocks, then for every same-color pair whose moved 2x2 centers share a row or column, fill the black cells strictly between their facing edges with that color.
    - Filling never overwrites non-black pixels, so grey dots and blocks remain visible if the connector path crosses them.

    Key insight:
    - The grey dots define discrete motion slots, and the only connectors to add are the same-color alignments that appear after the one-step slide.
    """
    g = np.asarray(grid, dtype=int)
    orientation, tracks, blocks = _extract_scene(g)
    moved = _move_blocks(orientation, tracks, blocks)

    out = np.full_like(g, BLACK)
    if orientation == "horizontal":
        for track in tracks:
            row = int(track["axis"])
            for col in track["slots"]:
                out[row, int(col)] = GREY
    else:
        for track in tracks:
            col = int(track["axis"])
            for row in track["slots"]:
                out[int(row), col] = GREY

    for block in moved:
        _paint_block(out, block)

    _fill_between_aligned_pairs(out, moved)
    return out


def _actual_coord(margin: int, unit: int) -> int:
    return int(margin + TRACK_STEP * unit)


def _post_unit(unit: int, end_unit: int) -> int:
    return min(unit + 1, end_unit)


def _actual_block_from_unit(
    orientation: str, perp: int, axis_margin: int, unit: int, color: int
) -> dict[str, int]:
    axis_coord = _actual_coord(axis_margin, unit)
    if orientation == "horizontal":
        return {"color": color, "r0": perp, "c0": axis_coord}
    return {"color": color, "r0": axis_coord, "c0": perp}


def _pair_aligned(a: dict[str, int], b: dict[str, int]) -> bool:
    return int(a["r0"]) == int(b["r0"]) or int(a["c0"]) == int(b["c0"])


def _pick_track_rows_or_cols(n_tracks: int) -> list[int]:
    first = random.randint(1, 4)
    positions = [first]
    for _ in range(1, n_tracks):
        positions.append(positions[-1] + random.choice((4, 5)))
    return positions


def _sample_track_specs(n_tracks: int) -> list[dict[str, int]]:
    track_specs: list[dict[str, int]] = []
    for _ in range(n_tracks):
        track_specs.append({"start": random.randint(0, 2), "length": random.randint(4, 6)})
    return track_specs


def _feasible_trigger_pairs(track_specs: list[dict[str, int]]) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for mover_idx, mover in enumerate(track_specs):
        mover_start = mover["start"]
        mover_end = mover_start + mover["length"] - 1
        for stayer_idx, stayer in enumerate(track_specs):
            if mover_idx == stayer_idx:
                continue
            target = stayer["start"] + stayer["length"] - 1
            if mover_start < target <= mover_end:
                pairs.append((mover_idx, stayer_idx, target))
    return pairs


def _mapped_units(units: set[int], end_unit: int) -> list[int]:
    return [_post_unit(unit, end_unit) for unit in sorted(units)]


def _choose_track_occupancies(
    track_specs: list[dict[str, int]], mover_idx: int, stayer_idx: int, target: int
) -> list[list[int]] | None:
    occupancies: list[set[int]] = [set() for _ in track_specs]
    occupancies[mover_idx].add(target - 1)
    occupancies[stayer_idx].add(target)

    desired_counts = [random.choice((1, 2)) for _ in track_specs]
    desired_counts[mover_idx] = max(desired_counts[mover_idx], 1)
    desired_counts[stayer_idx] = max(desired_counts[stayer_idx], 1)

    order = list(range(len(track_specs)))
    random.shuffle(order)
    for idx in order:
        spec = track_specs[idx]
        start = spec["start"]
        end = start + spec["length"] - 1
        candidates = list(range(start, end + 1))
        while len(occupancies[idx]) < desired_counts[idx]:
            choices = []
            for unit in candidates:
                if unit in occupancies[idx]:
                    continue
                trial = set(occupancies[idx])
                trial.add(unit)
                mapped = _mapped_units(trial, end)
                if len(set(mapped)) != len(mapped):
                    continue
                choices.append(unit)
            if not choices:
                return None
            occupancies[idx].add(random.choice(choices))

    return [sorted(units) for units in occupancies]


def _build_objects(
    orientation: str,
    perps: list[int],
    axis_margin: int,
    track_specs: list[dict[str, int]],
    occupancies: list[list[int]],
) -> tuple[list[dict[str, int]], list[dict[str, int]], dict[tuple[int, int], int]]:
    objects_pre: list[dict[str, int]] = []
    objects_post: list[dict[str, int]] = []
    id_map: dict[tuple[int, int], int] = {}
    for track_idx, (perp, spec, units) in enumerate(zip(perps, track_specs, occupancies)):
        end_unit = spec["start"] + spec["length"] - 1
        for unit in units:
            obj_id = len(objects_pre)
            id_map[(track_idx, unit)] = obj_id
            objects_pre.append(_actual_block_from_unit(orientation, perp, axis_margin, unit, BLACK))
            objects_post.append(_actual_block_from_unit(orientation, perp, axis_margin, _post_unit(unit, end_unit), BLACK))
    return objects_pre, objects_post, id_map


def _assign_colors(
    objects_pre: list[dict[str, int]],
    objects_post: list[dict[str, int]],
    trigger_pair: tuple[int, int],
) -> list[int] | None:
    n = len(objects_pre)
    diff_edges = [set() for _ in range(n)]
    trigger_a, trigger_b = trigger_pair
    for i, j in itertools.combinations(range(n), 2):
        must_differ = _pair_aligned(objects_pre[i], objects_pre[j]) or _pair_aligned(objects_post[i], objects_post[j])
        if {i, j} == {trigger_a, trigger_b}:
            must_differ = False
        if must_differ:
            diff_edges[i].add(j)
            diff_edges[j].add(i)

    trigger_color = random.choice(BLOCK_COLORS)
    colors: list[int | None] = [None] * n
    colors[trigger_a] = trigger_color
    colors[trigger_b] = trigger_color

    order = [idx for idx in range(n) if idx not in trigger_pair]
    order.sort(key=lambda idx: len(diff_edges[idx]), reverse=True)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        palette = list(BLOCK_COLORS)
        random.shuffle(palette)
        for color in palette:
            if any(colors[other] == color for other in diff_edges[idx]):
                continue
            colors[idx] = color
            if backtrack(pos + 1):
                return True
            colors[idx] = None
        return False

    if not backtrack(0):
        return None
    return [int(color) for color in colors]


def _apply_colors(blocks: list[dict[str, int]], colors: list[int]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for block, color in zip(blocks, colors):
        colored = dict(block)
        colored["color"] = color
        out.append(colored)
    return out


def _render_input(
    orientation: str,
    perps: list[int],
    axis_margin: int,
    track_specs: list[dict[str, int]],
    blocks: list[dict[str, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    if orientation == "horizontal":
        for row, spec in zip(perps, track_specs):
            for unit in range(spec["start"], spec["start"] + spec["length"]):
                grid[row, _actual_coord(axis_margin, unit)] = GREY
    else:
        for col, spec in zip(perps, track_specs):
            for unit in range(spec["start"], spec["start"] + spec["length"]):
                grid[_actual_coord(axis_margin, unit), col] = GREY

    for block in blocks:
        _paint_block(grid, block)
    return grid


def _count_new_aligned_pairs(pre: list[dict[str, int]], post: list[dict[str, int]]) -> int:
    count = 0
    for a, b in itertools.combinations(range(len(pre)), 2):
        if int(pre[a]["color"]) != int(pre[b]["color"]):
            continue
        if not _pair_aligned(pre[a], pre[b]) and _pair_aligned(post[a], post[b]):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("horizontal", "vertical"))
    n_tracks = random.choice((2, 3))
    perps = _pick_track_rows_or_cols(n_tracks)
    track_specs = _sample_track_specs(n_tracks)
    trigger_options = _feasible_trigger_pairs(track_specs)
    if not trigger_options:
        return None

    mover_track, stayer_track, target = random.choice(trigger_options)
    occupancies = _choose_track_occupancies(track_specs, mover_track, stayer_track, target)
    if occupancies is None:
        return None

    axis_margin = random.randint(1, 4)
    objects_pre_blank, objects_post_blank, id_map = _build_objects(
        orientation, perps, axis_margin, track_specs, occupancies
    )
    trigger_pair = (id_map[(mover_track, target - 1)], id_map[(stayer_track, target)])

    colors = None
    for _ in range(COLOR_ATTEMPTS):
        colors = _assign_colors(objects_pre_blank, objects_post_blank, trigger_pair)
        if colors is not None:
            break
    if colors is None:
        return None

    objects_pre = _apply_colors(objects_pre_blank, colors)
    objects_post = _apply_colors(objects_post_blank, colors)

    if _pair_aligned(objects_pre[trigger_pair[0]], objects_pre[trigger_pair[1]]):
        return None
    if not _pair_aligned(objects_post[trigger_pair[0]], objects_post[trigger_pair[1]]):
        return None
    if _count_new_aligned_pairs(objects_pre, objects_post) < 1:
        return None

    max_track_unit = max(spec["start"] + spec["length"] - 1 for spec in track_specs)
    if orientation == "horizontal":
        height = perps[-1] + BLOCK_SIZE + random.randint(1, 3)
        width = _actual_coord(axis_margin, max_track_unit) + BLOCK_SIZE + random.randint(1, 3)
    else:
        height = _actual_coord(axis_margin, max_track_unit) + BLOCK_SIZE + random.randint(1, 3)
        width = perps[-1] + BLOCK_SIZE + random.randint(1, 3)
    if max(height, width) > 30:
        return None

    inp = _render_input(orientation, perps, axis_margin, track_specs, objects_pre, (height, width))
    out = main(inp)

    if not isinstance(out, np.ndarray) or out.shape != inp.shape:
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
