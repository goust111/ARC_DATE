import random
from collections import deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [1, 2, 3, 4]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _frame_info(grid: np.ndarray, comp: dict) -> dict | None:
    color = int(comp["color"])
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None
    coord_set = set(coords)
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if not coord_set.issubset(border):
        return None
    missing = sorted(border - coord_set)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != 0):
        return None
    if gap[0] == r0:
        gap_side = "up"
    elif gap[0] == r1:
        gap_side = "down"
    elif gap[1] == c0:
        gap_side = "left"
    else:
        gap_side = "right"
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "cells": coord_set,
        "gap": gap,
        "gap_side": gap_side,
        "gap_dir": DIRS[gap_side],
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    frame_cells: set[tuple[int, int]] = set()
    for comp in _components(g):
        info = _frame_info(g, comp)
        if info is None:
            continue
        frames.append(info)
        frame_cells |= info["cells"]
    border_lookup: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in frame["cells"]:
            if cell in border_lookup:
                raise AssertionError("frame borders must not overlap")
            border_lookup[cell] = idx

    sources: list[dict] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == 0 or (r, c) in frame_cells:
                continue
            if color not in PALETTE:
                raise AssertionError("unexpected source color")
            sources.append({"start": (r, c), "color": color})

    for src in sources:
        r, c = src["start"]
        if any(_in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != 0 for nr, nc in _neighbors4(r, c)):
            raise AssertionError("sources must be isolated single pixels")

    sources.sort(key=lambda item: (item["start"][1], item["start"][0]))
    return frames, sources


def _simulate_particle(
    start: tuple[int, int],
    color: int,
    frames: list[dict],
    border_lookup: dict[tuple[int, int], int],
    shape: tuple[int, int],
    *,
    same_color_pass: bool = True,
    redirect: bool = True,
) -> dict:
    r, c = start
    dr, dc = DIRS["down"]
    seen: set[tuple[int, int, int, int]] = set()
    trace = [start]
    encounters: list[int] = []

    while True:
        state = (r, c, dr, dc)
        if state in seen:
            raise AssertionError("particle loop detected")
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            return {"final": (r, c), "trace": trace, "encounters": encounters, "reason": "edge"}

        hit = border_lookup.get((nr, nc))
        if hit is not None:
            frame = frames[hit]
            encounters.append(hit)
            if same_color_pass and color == frame["color"]:
                r, c = frame["gap"]
                if redirect:
                    dr, dc = frame["gap_dir"]
                trace.append((r, c))
                continue
            return {"final": (r, c), "trace": trace, "encounters": encounters, "reason": "blocked"}

        r, c = nr, nc
        trace.append((r, c))


def _solve(
    grid: np.ndarray,
    *,
    same_color_pass: bool = True,
    redirect: bool = True,
    return_stats: bool = False,
):
    g = np.asarray(grid, dtype=int)
    frames, sources = _parse_scene(g)
    border_lookup: dict[tuple[int, int], int] = {}
    out = np.zeros_like(g)
    for idx, frame in enumerate(frames):
        for r, c in frame["cells"]:
            border_lookup[(r, c)] = idx
            out[r, c] = frame["color"]

    stats: list[dict] = []
    for src in sources:
        sim = _simulate_particle(
            src["start"],
            src["color"],
            frames,
            border_lookup,
            g.shape,
            same_color_pass=same_color_pass,
            redirect=redirect,
        )
        fr, fc = sim["final"]
        out[fr, fc] = src["color"]
        sim["start"] = src["start"]
        sim["color"] = src["color"]
        stats.append(sim)

    if return_stats:
        return out, {"frames": frames, "sources": sources, "stats": stats}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each top pixel through hollow color-coded frames until it is blocked or reaches the edge.

    Input structure:
    - The grid has a black background, 2-4 isolated colored source pixels, and 3-5 hollow rectangular frame objects.
    - Every frame is a 1-pixel-thick monochromatic rectangle with a black interior and exactly one missing non-corner border cell.
    - Source pixels use colors from blue/red/green/yellow, and the same palette is reused by the frames.

    Rule:
    - Each source pixel starts moving downward one cell at a time.
    - When the next cell is a frame border: a same-colored frame is traversed by jumping to that frame's gap, then the motion direction becomes the outward direction of the gap; a different-colored frame blocks the pixel in the previous cell.
    - If no blocking frame stops it, the pixel keeps moving until the next step would leave the grid, then it remains on the last in-bounds cell.
    - Frames stay fixed, the original source pixels disappear, and settled pixels are written left-to-right by initial source column order.

    Key insight:
    - A frame's color decides whether it acts as a portal or a wall, and its single gap is the portal's exit vector for the next segment of the path.
    """
    return _solve(grid)


def _frame_patch(h: int, w: int, color: int, gap_side: str, gap_index: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if gap_side == "up":
        patch[0, gap_index] = 0
    elif gap_side == "down":
        patch[-1, gap_index] = 0
    elif gap_side == "left":
        patch[gap_index, 0] = 0
    else:
        patch[gap_index, -1] = 0
    return patch


def _gap_coord(bbox: tuple[int, int, int, int], gap_side: str, gap_index: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if gap_side == "up":
        return r0, c0 + gap_index
    if gap_side == "down":
        return r1, c0 + gap_index
    if gap_side == "left":
        return r0 + gap_index, c0
    return r0 + gap_index, c1


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 < br0:
        gap_r = br0 - ar1 - 1
    elif br1 < ar0:
        gap_r = ar0 - br1 - 1
    else:
        gap_r = 0
    if ac1 < bc0:
        gap_c = bc0 - ac1 - 1
    elif bc1 < ac0:
        gap_c = ac0 - bc1 - 1
    else:
        gap_c = 0
    return gap_r, gap_c


def _frames_well_spaced(frames: list[dict]) -> bool:
    bboxes = [frame["bbox"] for frame in frames]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            gap_r, gap_c = _bbox_gap(bboxes[i], bboxes[j])
            if gap_r == 0 and gap_c == 0:
                return False
            if max(gap_r, gap_c) < 2:
                return False
    return True


def _pick_template(n_sources: int) -> list[str]:
    templates = {
        2: [
            ["left", "right"],
            ["left", "top1"],
            ["right", "top1"],
        ],
        3: [
            ["left", "right", "top1"],
            ["left", "top1", "top2"],
            ["right", "top1", "top2"],
        ],
        4: [["left", "right", "top1", "top2"]],
    }
    return random.choice(templates[n_sources]).copy()


def _choose_special(slots: list[str]) -> str:
    has_left = "left" in slots
    has_right = "right" in slots
    top_slots = [slot for slot in slots if slot.startswith("top")]
    if has_left and has_right:
        return random.choice(["left", "right"])
    if top_slots and random.random() < 0.45:
        return random.choice(top_slots)
    side_slots = [slot for slot in slots if slot in ("left", "right")]
    return random.choice(side_slots or top_slots)


def _build_attempt() -> np.ndarray | None:
    n_sources = random.randint(2, 4)
    slots = _pick_template(n_sources)
    top_slots = [slot for slot in slots if slot.startswith("top")]
    has_left = "left" in slots
    has_right = "right" in slots

    left_w = random.randint(5, 7) if has_left else 0
    right_w = random.randint(5, 7) if has_right else 0
    bw = bh = 7

    min_w = 18
    if has_left:
        min_w = max(min_w, left_w + bw + 6)
    if has_right:
        min_w = max(min_w, right_w + bw + 6)
    if has_left and has_right:
        min_w = max(min_w, left_w + right_w + bw + 6)
    if min_w > 30:
        return None
    max_w = min(30, min_w + 4)
    w = random.randint(min_w, max_w)

    if len(top_slots) >= 2:
        min_h = 24
    elif len(top_slots) == 1:
        min_h = 18
    else:
        min_h = 16
    if min_h > 30:
        return None
    max_h = min(30, min_h + 5)
    h = random.randint(min_h, max_h)

    bc0_min = left_w + 3 if has_left else 4
    bc0_max = w - bw - right_w - 3 if has_right else w - bw - 4
    if bc0_min > bc0_max:
        return None
    bc0 = random.randint(bc0_min, bc0_max)

    bottom_margin = random.randint(2, 4)
    br0 = h - bh - bottom_margin
    if br0 < 9:
        return None

    colors = random.sample(PALETTE, len(slots))
    slot_colors = {slot: color for slot, color in zip(slots, colors)}
    special_slot = _choose_special(slots)

    frame_specs: list[dict] = []
    source_specs: list[dict] = []

    interior_rows = list(range(br0 + 1, br0 + bh - 1))
    interior_cols = list(range(bc0 + 1, bc0 + bw - 1))
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    left_row = None
    if has_left:
        left_row = random.choice([interior_rows[1], interior_rows[-2]]) if len(interior_rows) >= 4 else random.choice(interior_rows)
        used_rows.add(left_row)
        rh = random.randint(5, 7)
        rw = left_w
        gap_index = random.randint(1, rh - 2)
        rr0 = left_row - gap_index
        if rr0 < 1 or rr0 + rh > h:
            return None
        rc0 = 1
        bbox = (rr0, rc0, rr0 + rh - 1, rc0 + rw - 1)
        start_c = random.randint(rc0 + 1, rc0 + rw - 2)
        frame_specs.append(
            {
                "name": "left",
                "bbox": bbox,
                "color": slot_colors["left"],
                "gap_side": "right",
                "gap_index": gap_index,
            }
        )
        source_specs.append({"slot": "left", "start": (0, start_c), "color": slot_colors["left"]})

    right_row = None
    if has_right:
        choices = [row for row in interior_rows if row not in used_rows] or interior_rows
        right_row = random.choice(choices)
        used_rows.add(right_row)
        rh = random.randint(5, 7)
        rw = right_w
        gap_index = random.randint(1, rh - 2)
        rr0 = right_row - gap_index
        if rr0 < 1 or rr0 + rh > h:
            return None
        rc0 = w - rw - 1
        bbox = (rr0, rc0, rr0 + rh - 1, rc0 + rw - 1)
        start_c = random.randint(rc0 + 1, rc0 + rw - 2)
        frame_specs.append(
            {
                "name": "right",
                "bbox": bbox,
                "color": slot_colors["right"],
                "gap_side": "left",
                "gap_index": gap_index,
            }
        )
        source_specs.append({"slot": "right", "start": (0, start_c), "color": slot_colors["right"]})

    top_order = top_slots[:]
    random.shuffle(top_order)
    for idx, slot in enumerate(top_order):
        if len(top_slots) >= 2:
            rh = rw = 5
            top_c = interior_cols[0] if slot.endswith("1") else interior_cols[-1]
            rc0 = top_c - 1 if top_c == interior_cols[0] else top_c - 3
            rr1 = br0 - 3 if idx == 0 else br0 - 10
            rr0 = rr1 - rh + 1
        else:
            rh = random.randint(5, 7)
            rw = random.randint(5, 7)
            choices = [col for col in interior_cols if col not in used_cols] or interior_cols
            top_c = random.choice(choices)
            rc0_lo = max(1, top_c - (rw - 2))
            rc0_hi = min(w - rw - 1, top_c - 1)
            if rc0_lo > rc0_hi:
                return None
            rc0 = random.randint(rc0_lo, rc0_hi)
            rr1 = br0 - 3
            rr0 = rr1 - rh + 1
        if rr0 < 1:
            return None
        used_cols.add(top_c)
        bbox = (rr0, rc0, rr0 + rh - 1, rc0 + rw - 1)
        frame_specs.append(
            {
                "name": slot,
                "bbox": bbox,
                "color": slot_colors[slot],
                "gap_side": "down",
                "gap_index": top_c - rc0,
            }
        )
        source_specs.append({"slot": slot, "start": (0, top_c), "color": slot_colors[slot]})

    special_info: tuple[str, int] | tuple[str, str]
    if special_slot in ("left", "right"):
        special_gap_side = "down" if top_slots else random.choice(["up", "down"])
        free_cols = [col for col in interior_cols if col not in used_cols]
        if not free_cols:
            free_cols = interior_cols[:]
        special_gap_col = random.choice(free_cols)
        special_info = (special_gap_side, special_gap_col - bc0)
    else:
        if "left" in slots and "right" not in slots:
            special_gap_side = "right"
        elif "right" in slots and "left" not in slots:
            special_gap_side = "left"
        else:
            special_gap_side = random.choice(["left", "right"])
        free_rows = [row for row in interior_rows if row not in used_rows]
        if not free_rows:
            free_rows = interior_rows[:]
        special_gap_row = random.choice(free_rows)
        special_info = (special_gap_side, special_gap_row - br0)

    blocker_gap_side, blocker_gap_index = special_info
    frame_specs.append(
        {
            "name": "blocker",
            "bbox": (br0, bc0, br0 + bh - 1, bc0 + bw - 1),
            "color": slot_colors[special_slot],
            "gap_side": blocker_gap_side,
            "gap_index": blocker_gap_index,
        }
    )

    frames = []
    for spec in frame_specs:
        r0, c0, r1, c1 = spec["bbox"]
        h0 = r1 - r0 + 1
        w0 = c1 - c0 + 1
        patch = _frame_patch(h0, w0, spec["color"], spec["gap_side"], spec["gap_index"])
        gap = _gap_coord(spec["bbox"], spec["gap_side"], spec["gap_index"])
        frames.append(
            {
                "name": spec["name"],
                "bbox": spec["bbox"],
                "color": spec["color"],
                "gap": gap,
                "gap_side": spec["gap_side"],
                "gap_dir": DIRS[spec["gap_side"]],
                "patch": patch,
            }
        )

    if not _frames_well_spaced(frames):
        return None

    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != 0):
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1] = frame["patch"]

    for src in source_specs:
        r, c = src["start"]
        if grid[r, c] != 0:
            return None
        grid[r, c] = src["color"]

    try:
        out, meta = _solve(grid, return_stats=True)
    except AssertionError:
        return None

    if len(meta["frames"]) != len(frames) or len(meta["sources"]) != len(source_specs):
        return None
    if len(meta["frames"]) < 3 or len(meta["frames"]) > 5:
        return None
    if len(meta["sources"]) < 2 or len(meta["sources"]) > 4:
        return None

    start_cols = [src["start"][1] for src in meta["sources"]]
    if len(set(start_cols)) != len(start_cols):
        return None
    if sorted(src["color"] for src in meta["sources"]) != sorted(slot_colors[slot] for slot in slots):
        return None

    blocker_idx = None
    slot_to_source = {src["slot"]: src for src in source_specs}
    for idx, frame in enumerate(meta["frames"]):
        if frame["bbox"] == (br0, bc0, br0 + bh - 1, bc0 + bw - 1):
            blocker_idx = idx
            break
    if blocker_idx is None:
        return None

    traces = meta["stats"]
    if any(len(item["encounters"]) < 2 for item in traces):
        return None
    if any(item["encounters"][-1] != blocker_idx for item in traces):
        return None

    frame_colors_seen = {tuple(item["start"]): [] for item in traces}
    for item in traces:
        start = tuple(item["start"])
        frame_colors_seen[start] = [meta["frames"][idx]["color"] for idx in item["encounters"]]
    if not all(colors_seen[0] == next(src["color"] for src in source_specs if tuple(src["start"]) == start) for start, colors_seen in frame_colors_seen.items()):
        return None

    special_start = tuple(slot_to_source[special_slot]["start"])
    non_special = [item for item in traces if tuple(item["start"]) != special_start]
    special_trace = next(item for item in traces if tuple(item["start"]) == special_start)
    if meta["frames"][blocker_idx]["color"] != next(src["color"] for src in source_specs if tuple(src["start"]) == special_start):
        return None
    if any(meta["frames"][item["encounters"][-1]]["color"] == item["color"] for item in non_special):
        return None
    if any(item["reason"] != "blocked" for item in non_special):
        return None
    if special_trace["reason"] != "edge":
        return None

    path_cells: set[tuple[int, int]] = set()
    for item in traces:
        cells = set(item["trace"])
        if tuple(item["start"]) != special_start:
            cells.discard(item["final"])
        overlap = path_cells & cells
        if overlap:
            return None
        path_cells |= cells

    if np.array_equal(grid, out):
        return None
    if special_trace["final"] == special_trace["start"]:
        return None

    alt_block = _solve(grid, same_color_pass=False, redirect=True)
    try:
        alt_keep = _solve(grid, same_color_pass=True, redirect=False)
        alt_keep_same = np.array_equal(out, alt_keep)
    except AssertionError:
        alt_keep_same = False
    if np.array_equal(out, alt_block):
        return None
    if alt_keep_same:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-gated frame input")
