from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


L_CORNERS = {(0, 0), (0, 1), (1, 0), (1, 1)}
SOURCE_COLORS = {Color.GREY, Color.RED}
ANCHOR_COLORS = [
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
MOVE_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MISSING_FROM_DIR = {
    (-1, -1): (0, 0),
    (-1, 1): (0, 1),
    (1, -1): (1, 0),
    (1, 1): (1, 1),
}
# If an L is moving in move_dir, this is the top-left of its final 2x2 bbox
# relative to the anchor block's 3x3 top-left.
BOX_OFFSET_FROM_ANCHOR = {
    (-1, -1): (3, 3),
    (-1, 1): (3, -2),
    (1, -1): (-2, 3),
    (1, 1): (-2, -2),
}
MAX_GENERATION_ATTEMPTS = 400
BG = Color.BLACK


def _triomino_cells(r0: int, c0: int, missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(r0 + dr), int(c0 + dc))
        for dr in (0, 1)
        for dc in (0, 1)
        if (dr, dc) != missing
    )


def _anchor_cells(r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((r0 + dr, c0 + dc) for dr in range(3) for dc in range(3))


def _expand_cells(
    cells: set[tuple[int, int]] | tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    radius: int = 1,
) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _missing_corner_direction(comp: dict) -> tuple[int, int]:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    assert int(comp["area"]) == 3, "each source object must be a triomino"
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "triomino must occupy a 2x2 bbox"

    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    missing = L_CORNERS - rel
    assert len(missing) == 1, "expected exactly one missing corner"
    mr, mc = next(iter(missing))
    return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)


def _find_anchor_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    anchors: list[dict] = []
    for color in sorted(set(map(int, np.unique(g))) - {int(bg), Color.GREY, Color.RED}):
        comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[color])
        for comp in comps:
            if int(comp["area"]) != 9:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            if int(r1 - r0) != 2 or int(c1 - c0) != 2:
                continue
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(patch == int(color)):
                anchors.append(
                    {
                        "color": int(color),
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "top_left": (int(r0), int(c0)),
                    }
                )
    return anchors


def _steps_to_anchor(
    source_top_left: tuple[int, int],
    move_dir: tuple[int, int],
    anchor_top_left: tuple[int, int],
) -> int | None:
    sr, sc = map(int, source_top_left)
    dr, dc = map(int, move_dir)
    ar, ac = map(int, anchor_top_left)
    off_r, off_c = BOX_OFFSET_FROM_ANCHOR[(dr, dc)]
    dest_r = int(ar + off_r)
    dest_c = int(ac + off_c)
    delta_r = int(dest_r - sr)
    delta_c = int(dest_c - sc)
    steps = int(delta_r * dr)
    if steps <= 0:
        return None
    if delta_c * dc != steps:
        return None
    if delta_r != steps * dr or delta_c != steps * dc:
        return None
    return steps


def _first_anchor_hit(
    source_top_left: tuple[int, int],
    move_dir: tuple[int, int],
    anchors: list[dict],
) -> tuple[int | None, int | None]:
    best_idx = None
    best_steps = None
    for idx, anchor in enumerate(anchors):
        steps = _steps_to_anchor(source_top_left, move_dir, anchor["top_left"])
        if steps is None:
            continue
        if best_steps is None or steps < best_steps:
            best_idx = idx
            best_steps = steps
    return best_idx, best_steps


def _axis_aligned_anchor_ids(
    source_top_left: tuple[int, int],
    axis_dir: tuple[int, int],
    anchors: list[dict],
) -> set[int]:
    dirs = [axis_dir, (-axis_dir[0], -axis_dir[1])]
    hits: set[int] = set()
    for move_dir in dirs:
        for idx, anchor in enumerate(anchors):
            if _steps_to_anchor(source_top_left, move_dir, anchor["top_left"]) is not None:
                hits.add(idx)
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each L triomino along its missing-corner diagonal until it reaches the first anchor, then recolor it to that anchor.

    Input structure:
    - The background is dominant, with red and grey 4-connected L triominoes that each fill three cells of a 2x2 box.
    - Additional foreground objects are solid 3x3 anchor blocks in colors distinct from red and grey.
    - For every L, the diagonal axis implied by its missing 2x2 corner aligns with at most one anchor block.

    Rule:
    - Grey Ls move toward their missing corner; red Ls move in the opposite direction along that same diagonal axis.
    - An L slides until its 2x2 box is diagonally corner-touching the first anchor block on that ray, then the whole L is recolored to the anchor's color.
    - If no anchor exists on that ray, erase the L entirely; anchors and background stay unchanged.

    Key insight:
    - The missing corner identifies the diagonal axis, the source color chooses which way to travel on that axis, and the struck anchor supplies the final position and replacement color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    anchors = _find_anchor_blocks(g, bg)
    src_comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=sorted(SOURCE_COLORS),
    )

    out = g.copy()
    for comp in src_comps:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = bg

    for comp in src_comps:
        color = int(comp["color"])
        move_dir = _missing_corner_direction(comp)
        if color == int(Color.RED):
            move_dir = (-move_dir[0], -move_dir[1])
        source_top_left = (int(comp["bbox"][0]), int(comp["bbox"][1]))
        anchor_idx, steps = _first_anchor_hit(source_top_left, move_dir, anchors)
        if anchor_idx is None or steps is None:
            continue
        anchor_color = int(anchors[anchor_idx]["color"])
        for r, c in comp["coords"]:
            rr = int(r + steps * move_dir[0])
            cc = int(c + steps * move_dir[1])
            out[rr, cc] = anchor_color

    return out


def _sample_anchor_layout(shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    row_ranges = {
        "top": (1, max(1, h // 2 - 4)),
        "bottom": (min(h // 2 + 1, h - 4), h - 4),
    }
    col_ranges = {
        "left": (1, max(1, w // 2 - 4)),
        "right": (min(w // 2 + 1, w - 4), w - 4),
    }
    quadrants = [
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
    ]
    chosen = random.sample(quadrants, k=random.randint(2, 4))
    colors = random.sample(ANCHOR_COLORS, k=len(chosen))
    anchors: list[dict] = []
    for color, (row_key, col_key) in zip(colors, chosen):
        r0 = random.randint(*row_ranges[row_key])
        c0 = random.randint(*col_ranges[col_key])
        anchors.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r0 + 2, c0 + 2),
                "top_left": (r0, c0),
                "cells": set(_anchor_cells(r0, c0)),
            }
        )
    return anchors


def _collect_anchored_candidates(
    shape: tuple[int, int],
    anchors: list[dict],
    input_forbidden: set[tuple[int, int]],
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
    used_slots: set[tuple[int, tuple[int, int]]],
    src_color: int,
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    for anchor_idx, anchor in enumerate(anchors):
        for move_dir in MOVE_DIRS:
            if (anchor_idx, move_dir) in used_slots:
                continue
            final_r = int(anchor["top_left"][0] + BOX_OFFSET_FROM_ANCHOR[move_dir][0])
            final_c = int(anchor["top_left"][1] + BOX_OFFSET_FROM_ANCHOR[move_dir][1])
            if not (0 <= final_r <= h - 2 and 0 <= final_c <= w - 2):
                continue
            missing_dir = move_dir if src_color == int(Color.GREY) else (-move_dir[0], -move_dir[1])
            missing = MISSING_FROM_DIR[missing_dir]
            final_cells = set(_triomino_cells(final_r, final_c, missing))
            if final_cells & input_occupied:
                continue
            if final_cells & output_occupied:
                continue
            steps = 1
            while True:
                src_r = int(final_r - steps * move_dir[0])
                src_c = int(final_c - steps * move_dir[1])
                if not (0 <= src_r <= h - 2 and 0 <= src_c <= w - 2):
                    break
                source_cells = set(_triomino_cells(src_r, src_c, missing))
                if source_cells & input_forbidden:
                    steps += 1
                    continue
                if source_cells & output_occupied:
                    steps += 1
                    continue
                hit_idx, hit_steps = _first_anchor_hit((src_r, src_c), move_dir, anchors)
                axis_hits = _axis_aligned_anchor_ids((src_r, src_c), missing_dir, anchors)
                if hit_idx == anchor_idx and hit_steps == steps and axis_hits == {anchor_idx}:
                    candidates.append(
                        {
                            "kind": "anchored",
                            "src_color": int(src_color),
                            "move_dir": move_dir,
                            "missing": missing,
                            "missing_dir": missing_dir,
                            "steps": steps,
                            "source_top_left": (src_r, src_c),
                            "source_cells": source_cells,
                            "anchor_idx": anchor_idx,
                            "final_cells": final_cells,
                        }
                    )
                steps += 1
    long_moves = [cand for cand in candidates if int(cand["steps"]) >= 2]
    return long_moves or candidates


def _collect_deleted_candidates(
    shape: tuple[int, int],
    anchors: list[dict],
    input_forbidden: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
    src_color: int,
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    for move_dir in MOVE_DIRS:
        missing_dir = move_dir if src_color == int(Color.GREY) else (-move_dir[0], -move_dir[1])
        missing = MISSING_FROM_DIR[missing_dir]
        for src_r in range(h - 1):
            for src_c in range(w - 1):
                source_cells = set(_triomino_cells(src_r, src_c, missing))
                if source_cells & input_forbidden:
                    continue
                if source_cells & output_occupied:
                    continue
                hit_idx, _ = _first_anchor_hit((src_r, src_c), move_dir, anchors)
                if hit_idx is not None:
                    continue
                if _axis_aligned_anchor_ids((src_r, src_c), missing_dir, anchors):
                    continue
                edge_bias = min(src_r, src_c, h - 2 - src_r, w - 2 - src_c)
                candidates.append(
                    {
                        "kind": "deleted",
                        "src_color": int(src_color),
                        "move_dir": move_dir,
                        "missing": missing,
                        "missing_dir": missing_dir,
                        "source_top_left": (src_r, src_c),
                        "source_cells": source_cells,
                        "anchor_idx": None,
                        "final_cells": set(),
                        "edge_bias": edge_bias,
                    }
                )
    edge_candidates = [cand for cand in candidates if int(cand["edge_bias"]) <= 2]
    return edge_candidates or candidates


def _choose_candidate(candidates: list[dict], used_dirs: set[tuple[int, int]]) -> dict | None:
    if not candidates:
        return None
    fresh_dirs = [cand for cand in candidates if cand["move_dir"] not in used_dirs]
    pool = fresh_dirs or candidates
    return random.choice(pool)


def _render_scene(shape: tuple[int, int], anchors: list[dict], records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, int(BG), dtype=int)
    out = np.full(shape, int(BG), dtype=int)

    for anchor in anchors:
        for r, c in anchor["cells"]:
            inp[r, c] = int(anchor["color"])
            out[r, c] = int(anchor["color"])

    for rec in records:
        for r, c in rec["source_cells"]:
            inp[r, c] = int(rec["src_color"])
        if rec["anchor_idx"] is None:
            continue
        anchor_color = int(anchors[rec["anchor_idx"]]["color"])
        for r, c in rec["final_cells"]:
            out[r, c] = anchor_color

    return inp, out


def _validate_scene(inp: np.ndarray, out: np.ndarray, anchors: list[dict], records: list[dict]) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    bg = int(get_bg(inp))
    found_anchors = _find_anchor_blocks(inp, bg)
    if len(found_anchors) != len(anchors):
        return False
    if len({int(anchor["color"]) for anchor in found_anchors}) != len(found_anchors):
        return False

    comps = find_components(
        inp,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=sorted(SOURCE_COLORS),
    )
    if len(comps) != len(records):
        return False

    anchored_by_color = {int(Color.GREY): 0, int(Color.RED): 0}
    deleted_by_color = {int(Color.GREY): 0, int(Color.RED): 0}

    for comp in comps:
        comp_color = int(comp["color"])
        if int(comp["area"]) != 3:
            return False
        r0, c0, r1, c1 = comp["bbox"]
        if int(r1 - r0) != 1 or int(c1 - c0) != 1:
            return False
        missing_dir = _missing_corner_direction(comp)
        move_dir = missing_dir if comp_color == int(Color.GREY) else (-missing_dir[0], -missing_dir[1])
        axis_hits = _axis_aligned_anchor_ids((int(r0), int(c0)), missing_dir, found_anchors)
        if len(axis_hits) > 1:
            return False
        anchor_idx, steps = _first_anchor_hit((int(r0), int(c0)), move_dir, found_anchors)
        for r, c in comp["coords"]:
            if out[int(r), int(c)] != bg:
                return False
        if anchor_idx is None:
            deleted_by_color[comp_color] += 1
            continue
        anchored_by_color[comp_color] += 1
        anchor_color = int(found_anchors[anchor_idx]["color"])
        for r, c in comp["coords"]:
            rr = int(r + steps * move_dir[0])
            cc = int(c + steps * move_dir[1])
            if out[rr, cc] != anchor_color:
                return False

    return (
        anchored_by_color[int(Color.GREY)] >= 1
        and anchored_by_color[int(Color.RED)] >= 1
        and deleted_by_color[int(Color.GREY)] >= 1
        and deleted_by_color[int(Color.RED)] >= 1
    )


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 26), random.randint(18, 26))
    anchors = _sample_anchor_layout(shape)

    input_occupied = set().union(*(anchor["cells"] for anchor in anchors))
    input_forbidden = _expand_cells(input_occupied, shape, radius=1)
    output_occupied = set(input_occupied)
    used_slots: set[tuple[int, tuple[int, int]]] = set()
    used_dirs: set[tuple[int, int]] = set()
    records: list[dict] = []

    anchored_specs = [int(Color.GREY), int(Color.RED)]
    deleted_specs = [int(Color.GREY), int(Color.RED)]
    for _ in range(random.randint(1, 3)):
        if random.random() < 0.75:
            anchored_specs.append(random.choice([int(Color.GREY), int(Color.RED)]))
        else:
            deleted_specs.append(random.choice([int(Color.GREY), int(Color.RED)]))

    for src_color in anchored_specs:
        candidates = _collect_anchored_candidates(
            shape,
            anchors,
            input_forbidden,
            input_occupied,
            output_occupied,
            used_slots,
            src_color,
        )
        choice = _choose_candidate(candidates, used_dirs)
        if choice is None:
            return None
        records.append(choice)
        input_occupied.update(choice["source_cells"])
        input_forbidden.update(_expand_cells(choice["source_cells"], shape, radius=1))
        output_occupied.update(choice["final_cells"])
        used_slots.add((int(choice["anchor_idx"]), choice["move_dir"]))
        used_dirs.add(choice["move_dir"])

    for src_color in deleted_specs:
        candidates = _collect_deleted_candidates(
            shape,
            anchors,
            input_forbidden,
            output_occupied,
            src_color,
        )
        choice = _choose_candidate(candidates, used_dirs)
        if choice is None:
            return None
        records.append(choice)
        input_occupied.update(choice["source_cells"])
        input_forbidden.update(_expand_cells(choice["source_cells"], shape, radius=1))
        used_dirs.add(choice["move_dir"])

    inp, out = _render_scene(shape, anchors, records)
    if not _validate_scene(inp, out, anchors, records):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
