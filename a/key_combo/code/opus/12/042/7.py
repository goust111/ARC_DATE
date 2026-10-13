from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


Coord = tuple[int, int]
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
MOVE_TO_ORIENT = {(0, 1): "h", (0, -1): "h", (1, 0): "v", (-1, 0): "v"}
NON_FILL_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
FILL_COLOR = 8
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> tuple[Coord, ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_by_color(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _direction(a: Coord, b: Coord) -> Coord:
    return (b[0] - a[0], b[1] - a[1])


def _turn_count(points: list[Coord]) -> int:
    if len(points) < 3:
        return 0
    turns = 0
    prev = _direction(points[0], points[1])
    for i in range(1, len(points) - 1):
        cur = _direction(points[i], points[i + 1])
        if cur != prev:
            turns += 1
        prev = cur
    return turns


def _ordered_path(attach: Coord, remaining: set[Coord]) -> list[Coord] | None:
    starts = [cell for cell in remaining if sum((nb in remaining) for nb in _neighbors4(*cell)) <= 1]
    attach_neighbors = [cell for cell in remaining if attach in _neighbors4(*cell)]
    if len(attach_neighbors) != 1:
        return None
    start = attach_neighbors[0]
    ordered = [start]
    used = {start}
    prev = attach
    cur = start
    while True:
        nxt = [nb for nb in _neighbors4(*cur) if nb in remaining and nb != prev]
        if len(nxt) > 1:
            return None
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur in used:
            return None
        used.add(cur)
        ordered.append(cur)
    if used != remaining:
        return None
    if start not in starts and len(remaining) > 1:
        return None
    return ordered


def _parse_component(comp: dict) -> dict:
    cells = set(comp["coords"])
    color = int(comp["color"])
    candidates: list[dict] = []
    for a in cells:
        for b in _neighbors4(*a):
            if b not in cells or a > b:
                continue
            remainder = cells - {a, b}
            if not remainder:
                continue
            for attach, other in ((a, b), (b, a)):
                attach_links = sum((nb in remainder) for nb in _neighbors4(*attach))
                other_links = sum((nb in remainder) for nb in _neighbors4(*other))
                if attach_links != 1 or other_links != 0:
                    continue
                path = _ordered_path(attach, remainder)
                if path is None:
                    continue
                chain = [other, attach] + path
                if _turn_count(path) > 1:
                    continue
                candidates.append(
                    {
                        "color": color,
                        "other": other,
                        "attach": attach,
                        "path": path,
                        "body_vec": (other[0] - attach[0], other[1] - attach[1]),
                        "path_len": len(path),
                        "chain_turns": _turn_count(chain),
                    }
                )
    # Straight same-color chains are ambiguous; generated inputs avoid them.
    uniq: list[dict] = []
    for cand in candidates:
        key = (cand["attach"], cand["other"], tuple(cand["path"]))
        if all((key != (u["attach"], u["other"], tuple(u["path"]))) for u in uniq):
            uniq.append(cand)
    assert len(uniq) == 1, "component must identify exactly one domino/path interpretation"
    return uniq[0]


def _parse_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for comp in _components_by_color(grid):
        obj = _parse_component(comp)
        obj["initial_cells"] = [obj["attach"], obj["other"]]
        anchor = min(obj["initial_cells"])
        obj["sort_key"] = (obj["path_len"], anchor[0], anchor[1])
        objects.append(obj)
    return objects


def _settle_objects(objects: list[dict], shape: tuple[int, int]) -> list[dict]:
    occupied = np.zeros(shape, dtype=bool)
    settled: list[dict] = []
    for obj in sorted(objects, key=lambda item: item["sort_key"]):
        body_vec = obj["body_vec"]
        chosen_attach: Coord | None = None
        chosen_cells: list[Coord] | None = None
        for attach in reversed(obj["path"]):
            cells = [attach, (attach[0] + body_vec[0], attach[1] + body_vec[1])]
            if not all(_in_bounds(shape, r, c) for r, c in cells):
                continue
            if any(occupied[r, c] for r, c in cells):
                continue
            chosen_attach = attach
            chosen_cells = cells
            break
        assert chosen_attach is not None and chosen_cells is not None, "every domino needs a valid settled spot"
        for r, c in chosen_cells:
            occupied[r, c] = True
        settled.append(
            {
                "color": obj["color"],
                "cells": chosen_cells,
                "attach": chosen_attach,
                "body_vec": body_vec,
                "path": list(obj["path"]),
                "blocked": chosen_attach != obj["path"][-1],
            }
        )
    return settled


def _paint_bridges(out: np.ndarray, settled: list[dict]) -> None:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in settled:
        by_color[int(obj["color"])].append(obj)

    base = out.copy()
    for color, items in by_color.items():
        _ = color
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = sorted(items[i]["cells"])
                b = sorted(items[j]["cells"])
                rows_a = {r for r, _ in a}
                rows_b = {r for r, _ in b}
                cols_a = {c for _, c in a}
                cols_b = {c for _, c in b}

                if len(rows_a) == len(rows_b) == 1 and next(iter(rows_a)) == next(iter(rows_b)):
                    row = next(iter(rows_a))
                    a_cols = sorted(c for _, c in a)
                    b_cols = sorted(c for _, c in b)
                    left_end = min(a_cols[-1], b_cols[-1]) if min(a_cols) > min(b_cols) else a_cols[-1]
                    right_start = max(a_cols[0], b_cols[0]) if min(a_cols) > min(b_cols) else b_cols[0]
                    if min(a_cols) > min(b_cols):
                        left_end = b_cols[-1]
                        right_start = a_cols[0]
                    if right_start - left_end > 1:
                        gap = [(row, c) for c in range(left_end + 1, right_start)]
                        if all(base[r, c] == 0 for r, c in gap):
                            for r, c in gap:
                                out[r, c] = FILL_COLOR

                if len(cols_a) == len(cols_b) == 1 and next(iter(cols_a)) == next(iter(cols_b)):
                    col = next(iter(cols_a))
                    a_rows = sorted(r for r, _ in a)
                    b_rows = sorted(r for r, _ in b)
                    if min(a_rows) > min(b_rows):
                        top_end = b_rows[-1]
                        bottom_start = a_rows[0]
                    else:
                        top_end = a_rows[-1]
                        bottom_start = b_rows[0]
                    if bottom_start - top_end > 1:
                        gap = [(r, col) for r in range(top_end + 1, bottom_start)]
                        if all(base[r, c] == 0 for r, c in gap):
                            for r, c in gap:
                                out[r, c] = FILL_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each two-cell domino along its attached same-color trail and bridge aligned same-color pairs afterward.

    Input structure:
    - The grid has a black background, 4-6 monochrome dominoes, and one same-color 4-connected trail attached to one end of each domino.
    - Each trail is a one-cell-wide orthogonal polyline attached to exactly one domino and disconnected from every other object.
    - Dominoes are exactly two 4-connected cells; trails encode where their attachment endpoint should travel.

    Rule:
    - For every monochrome component, identify which two cells are the domino and trace the attached trail from the attachment endpoint to its terminal pixel.
    - Settle dominoes in increasing trail-length order, breaking ties by the domino's top-left cell; each domino tries to move its attachment endpoint to the trail end and, if blocked, backs up along the discrete trail to the nearest non-conflicting placement.
    - Clear all original domino and trail pixels, keep only the settled dominoes, then fill with color 8 every all-black gap between two settled dominoes of the same color that end up entirely on the same row or entirely on the same column.

    Key insight:
    - The trail is a private motion script, but the cyan bridge is an emergent post-motion effect that only appears after all dominoes have settled and same-color alignment can be checked.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _parse_objects(g)
    settled = _settle_objects(objects, g.shape)

    out = np.zeros_like(g)
    for obj in settled:
        for r, c in obj["cells"]:
            out[r, c] = int(obj["color"])
    _paint_bridges(out, settled)
    return out


def _cells_for_domino(anchor: Coord, orient: str) -> list[Coord]:
    r, c = anchor
    if orient == "h":
        return [(r, c), (r, c + 1)]
    return [(r, c), (r + 1, c)]


def _touches_any(cells: set[Coord], occupied: set[Coord], shape: tuple[int, int]) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return True
        if (r, c) in occupied:
            return True
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc) and (nr, nc) in occupied:
                return True
    return False


def _place_bridge_pair(shape: tuple[int, int], occupied: set[Coord], color: int) -> list[dict] | None:
    h, w = shape
    for _ in range(200):
        orient = random.choice(("h", "v"))
        gap = random.randint(2, 4)
        if orient == "h":
            r = random.randint(2, h - 3)
            c1 = random.randint(1, w - (gap + 6))
            c2 = c1 + gap + 3
            cells_a = [(r, c1), (r, c1 + 1)]
            cells_b = [(r, c2), (r, c2 + 1)]
        else:
            c = random.randint(2, w - 3)
            r1 = random.randint(1, h - (gap + 6))
            r2 = r1 + gap + 3
            cells_a = [(r1, c), (r1 + 1, c)]
            cells_b = [(r2, c), (r2 + 1, c)]
        if _touches_any(set(cells_a) | set(cells_b), occupied, shape):
            continue
        return [{"color": color, "cells": cells_a}, {"color": color, "cells": cells_b}]
    return None


def _random_final_domino(shape: tuple[int, int], occupied: set[Coord], color: int) -> dict | None:
    h, w = shape
    for _ in range(200):
        orient = random.choice(("h", "v"))
        if orient == "h":
            anchor = (random.randint(1, h - 2), random.randint(1, w - 3))
        else:
            anchor = (random.randint(1, h - 3), random.randint(1, w - 2))
        cells = _cells_for_domino(anchor, orient)
        if _touches_any(set(cells), occupied, shape):
            continue
        return {"color": color, "cells": cells}
    return None


def _place_distractor_pair(shape: tuple[int, int], occupied: set[Coord], color: int) -> list[dict] | None:
    first = _random_final_domino(shape, occupied, color)
    if first is None:
        return None
    occ = occupied | set(first["cells"])
    for _ in range(200):
        second = _random_final_domino(shape, occ, color)
        if second is None:
            continue
        rows_a = {r for r, _ in first["cells"]}
        rows_b = {r for r, _ in second["cells"]}
        cols_a = {c for _, c in first["cells"]}
        cols_b = {c for _, c in second["cells"]}
        same_row = len(rows_a) == len(rows_b) == 1 and next(iter(rows_a)) == next(iter(rows_b))
        same_col = len(cols_a) == len(cols_b) == 1 and next(iter(cols_a)) == next(iter(cols_b))
        if same_row or same_col:
            continue
        return [first, second]
    return None


def _place_collision_pair(shape: tuple[int, int], occupied: set[Coord], blocker_color: int, collider_color: int) -> list[dict] | None:
    h, w = shape
    for _ in range(200):
        orient = random.choice(("h", "v"))
        if orient == "h":
            r = random.randint(2, h - 3)
            start = random.randint(1, w - 5)
            cells_collider = [(r, start), (r, start + 1)]
            cells_blocker = [(r, start + 2), (r, start + 3)]
            slide_dir = (0, 1)
        else:
            c = random.randint(2, w - 3)
            start = random.randint(1, h - 5)
            cells_collider = [(start, c), (start + 1, c)]
            cells_blocker = [(start + 2, c), (start + 3, c)]
            slide_dir = (1, 0)
        all_cells = set(cells_collider) | set(cells_blocker)
        if _touches_any(all_cells, occupied, shape):
            continue
        return [
            {"color": blocker_color, "cells": cells_blocker, "preferred_slide": None},
            {"color": collider_color, "cells": cells_collider, "preferred_slide": slide_dir},
        ]
    return None


def _final_attach_options(cells: list[Coord]) -> list[Coord]:
    a, b = cells
    return [a, b]


def _perpendiculars(d: Coord) -> tuple[Coord, Coord]:
    dr, dc = d
    return ((dc, dr), (-dc, -dr))


def _candidate_input_shapes(
    final_cells: list[Coord],
    shape: tuple[int, int],
    total_len: int,
    force_slide: Coord | None,
    allow_blocked: bool,
) -> list[dict]:
    candidates: list[dict] = []
    cells_sorted = sorted(final_cells)
    for attach_final in _final_attach_options(cells_sorted):
        other_final = next(cell for cell in cells_sorted if cell != attach_final)
        body_vec = (other_final[0] - attach_final[0], other_final[1] - attach_final[1])
        slide_dirs = [force_slide] if force_slide is not None else list(DIRS4)
        for d2 in slide_dirs:
            if d2 is None:
                continue
            for d1 in _perpendiculars(d2):
                if d1 == (-body_vec[0], -body_vec[1]):
                    continue
                ext_choices = [1] if allow_blocked else [0]
                for ext in ext_choices:
                    for len1 in range(1, total_len):
                        len2_actual = total_len - len1 - ext
                        if len2_actual < 1:
                            continue
                        if len2_actual + ext < 2:
                            continue
                        turn = (attach_final[0] - len2_actual * d2[0], attach_final[1] - len2_actual * d2[1])
                        attach0 = (turn[0] - len1 * d1[0], turn[1] - len1 * d1[1])
                        other0 = (attach0[0] + body_vec[0], attach0[1] + body_vec[1])
                        seg1 = [(attach0[0] + i * d1[0], attach0[1] + i * d1[1]) for i in range(1, len1 + 1)]
                        seg2 = [(turn[0] + i * d2[0], turn[1] + i * d2[1]) for i in range(1, len2_actual + ext + 1)]
                        path = seg1 + seg2
                        all_cells = {attach0, other0, *path}
                        if len(all_cells) != len(path) + 2:
                            continue
                        if not all(_in_bounds(shape, r, c) for r, c in all_cells):
                            continue
                        if path[-1] == attach_final and ext != 0:
                            continue
                        candidate = {
                            "attach0": attach0,
                            "other0": other0,
                            "path": path,
                            "attach_final": attach_final,
                            "cells_final": set(final_cells),
                            "body_vec": body_vec,
                            "slide_dir": d2,
                            "all_cells": all_cells,
                        }
                        try:
                            parsed = _parse_component({"color": 1, "coords": list(all_cells)})
                        except AssertionError:
                            continue
                        if parsed["attach"] != attach0 or parsed["other"] != other0 or tuple(parsed["path"]) != tuple(path):
                            continue
                        candidates.append(candidate)
    random.shuffle(candidates)
    return candidates


def _render_input(records: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rec in records:
        color = int(rec["color"])
        for r, c in rec["path"]:
            grid[r, c] = color
        grid[rec["attach0"]] = color
        grid[rec["other0"]] = color
    return grid


def _assign_records(
    final_specs: list[dict],
    shape: tuple[int, int],
    lengths: list[int],
    order: list[int],
    collision_ids: set[int],
) -> list[dict] | None:
    candidate_cache: dict[int, list[dict]] = {}
    for idx in order:
        spec = final_specs[idx]
        force_slide = spec.get("preferred_slide")
        allow_blocked = idx in collision_ids
        cands = _candidate_input_shapes(spec["cells"], shape, lengths[idx], force_slide, allow_blocked)
        if not cands:
            return None
        candidate_cache[idx] = cands[:60]

    records: list[dict | None] = [None] * len(final_specs)
    occupied: set[Coord] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        spec = final_specs[idx]
        for cand in candidate_cache[idx]:
            if _touches_any(cand["all_cells"], occupied, shape):
                continue
            records[idx] = {"color": spec["color"], **cand}
            added = set(cand["all_cells"])
            occupied.update(added)
            if dfs(pos + 1):
                return True
            occupied.difference_update(added)
            records[idx] = None
        return False

    return [rec for rec in records if rec is not None] if dfs(0) else None


def _same_color_features(out: np.ndarray) -> tuple[bool, bool]:
    comps = []
    for comp in _components_by_color(out):
        color = int(comp["color"])
        if color == FILL_COLOR:
            continue
        if len(comp["coords"]) == 2:
            comps.append({"color": color, "cells": sorted(comp["coords"])})
    by_color: dict[int, list[list[Coord]]] = defaultdict(list)
    for comp in comps:
        by_color[comp["color"]].append(comp["cells"])

    has_bridge = False
    has_non_collinear = False
    for items in by_color.values():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                rows_a = {r for r, _ in a}
                rows_b = {r for r, _ in b}
                cols_a = {c for _, c in a}
                cols_b = {c for _, c in b}
                same_row = len(rows_a) == len(rows_b) == 1 and next(iter(rows_a)) == next(iter(rows_b))
                same_col = len(cols_a) == len(cols_b) == 1 and next(iter(cols_a)) == next(iter(cols_b))
                if same_row:
                    row = next(iter(rows_a))
                    a_cols = sorted(c for _, c in a)
                    b_cols = sorted(c for _, c in b)
                    if min(a_cols) > min(b_cols):
                        left_end = b_cols[-1]
                        right_start = a_cols[0]
                    else:
                        left_end = a_cols[-1]
                        right_start = b_cols[0]
                    if right_start - left_end > 1 and np.all(out[row, left_end + 1 : right_start] == FILL_COLOR):
                        has_bridge = True
                elif same_col:
                    col = next(iter(cols_a))
                    a_rows = sorted(r for r, _ in a)
                    b_rows = sorted(r for r, _ in b)
                    if min(a_rows) > min(b_rows):
                        top_end = b_rows[-1]
                        bottom_start = a_rows[0]
                    else:
                        top_end = a_rows[-1]
                        bottom_start = b_rows[0]
                    if bottom_start - top_end > 1 and np.all(out[top_end + 1 : bottom_start, col] == FILL_COLOR):
                        has_bridge = True
                else:
                    has_non_collinear = True
    return has_bridge, has_non_collinear


def _attempt_generate() -> np.ndarray | None:
    shape = (random.randint(16, 19), random.randint(16, 19))
    count = random.randint(4, 6)
    colors = random.sample(list(NON_FILL_COLORS), count)
    bridge_color, distract_color = colors[0], colors[1]
    final_specs: list[dict] = []
    occupied: set[Coord] = set()

    bridge_pair = _place_bridge_pair(shape, occupied, bridge_color)
    if bridge_pair is None:
        return None
    for item in bridge_pair:
        occupied.update(item["cells"])
        final_specs.append(item)

    distract_pair = _place_distractor_pair(shape, occupied, distract_color)
    if distract_pair is None:
        return None
    for item in distract_pair:
        occupied.update(item["cells"])
        final_specs.append(item)

    want_collision = count >= 5 and random.random() < 0.7
    collision_ids: set[int] = set()
    if want_collision:
        collision_pair = _place_collision_pair(shape, occupied, colors[2], colors[3] if count >= 6 else colors[2])
        if collision_pair is None:
            return None
        blocker = collision_pair[0]
        collider = collision_pair[1]
        if blocker["color"] == collider["color"]:
            collider["color"] = colors[-1]
        occupied.update(blocker["cells"])
        occupied.update(collider["cells"])
        final_specs.extend([blocker, collider])
        collision_ids.add(len(final_specs) - 1)

    while len(final_specs) < count:
        color = colors[len(final_specs)]
        item = _random_final_domino(shape, occupied, color)
        if item is None:
            return None
        occupied.update(item["cells"])
        final_specs.append(item)

    lengths = list(range(4, 4 + len(final_specs)))
    random.shuffle(lengths)
    if collision_ids:
        collider_idx = next(iter(collision_ids))
        blocker_idx = collider_idx - 1
        lengths[blocker_idx], lengths[collider_idx] = min(lengths), max(lengths)

    order = list(range(len(final_specs)))
    if collision_ids:
        collider_idx = next(iter(collision_ids))
        order = [collider_idx - 1, collider_idx] + [i for i in order if i not in {collider_idx, collider_idx - 1}]

    records = _assign_records(final_specs, shape, lengths, order, collision_ids)
    if records is None:
        return None

    inp = _render_input(records, shape)

    try:
        parsed = _parse_objects(inp)
        settled = _settle_objects(parsed, shape)
        out = main(inp)
    except AssertionError:
        return None

    has_bridge, has_non_collinear = _same_color_features(out)
    if not has_bridge or not has_non_collinear:
        return None

    if len(parsed) != count:
        return None
    if collision_ids and not any(obj["blocked"] for obj in settled):
        return None
    if not any(_turn_count([obj["other"], obj["attach"]] + obj["path"]) >= 2 for obj in parsed):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
