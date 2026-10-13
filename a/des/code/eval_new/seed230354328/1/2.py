from __future__ import annotations

import random
from collections import deque

import numpy as np

BG = 0
MIRROR = 4
FRAME = 5

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ARROW_BACK = {
    "up": ((1, -1), (1, 1)),
    "down": ((-1, -1), (-1, 1)),
    "left": ((-1, 1), (1, 1)),
    "right": ((-1, -1), (1, -1)),
}
REFLECT = {
    "\\": {
        DIRS["left"]: DIRS["down"],
        DIRS["right"]: DIRS["up"],
        DIRS["up"]: DIRS["left"],
        DIRS["down"]: DIRS["right"],
    },
    "/": {
        DIRS["left"]: DIRS["up"],
        DIRS["right"]: DIRS["down"],
        DIRS["up"]: DIRS["right"],
        DIRS["down"]: DIRS["left"],
    },
}

SOURCE_COLORS = [1, 2, 3, 6, 7, 8, 9]
MIRROR_TYPES = ("\\", "/")
GEN_RETRIES = 400
ENTITY_RETRIES = 600


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = ORTHO_DIRS if connectivity == 4 else ORTHO_DIRS + DIAG_DIRS
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in steps:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _frame_interior(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _all_in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _all_in_rect(cells: set[tuple[int, int]], rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)


def _halo(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _line_cells(
    start: tuple[int, int], end: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    cells = [start]
    r, c = start
    dr, dc = direction
    while (r, c) != end:
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _find_frames(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    frames: list[dict[str, object]] = []
    interior_lookup: dict[tuple[int, int], int] = {}
    for comp in _components(grid == FRAME, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "grey frame must enclose an interior"
        border = set(_frame_border((r0, c0, r1, c1)))
        assert set(comp) == border, "grey components must be hollow rectangles"
        interior = _frame_interior((r0, c0, r1, c1))
        assert interior and all(int(grid[r, c]) == BG for r, c in interior), "frame interior must be black"
        idx = len(frames)
        frames.append({"bbox": (r0, c0, r1, c1), "interior": interior})
        for cell in interior:
            assert cell not in interior_lookup, "frame interiors must not overlap"
            interior_lookup[cell] = idx
    assert frames, "expected at least one grey target frame"
    return frames, interior_lookup


def _find_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    for comp in _components(grid == MIRROR, connectivity=8):
        assert len(comp) >= 2, "mirror segments must have length at least 2"
        diffs = {r - c for r, c in comp}
        sums = {r + c for r, c in comp}
        if len(diffs) == 1:
            orient = "\\"
        else:
            assert len(sums) == 1, "yellow components must be straight diagonal mirrors"
            orient = "/"
        for cell in comp:
            orientations[cell] = orient
    assert orientations, "expected at least one yellow mirror"
    return orientations


def _find_sources(grid: np.ndarray) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, MIRROR, FRAME))
    assert colors, "expected at least one source color"
    for color in colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
        tips: list[tuple[int, int, str]] = []
        for r, c in pts:
            if any((r + dr, c + dc) in pts for dr, dc in ORTHO_DIRS):
                continue
            for name, backs in ARROW_BACK.items():
                if all((r + dr, c + dc) in pts for dr, dc in backs):
                    tips.append((r, c, name))
        assert len(tips) == 1, f"color {color} must contain exactly one arrow tip"
        tr, tc, name = tips[0]
        sources.append({"color": color, "tip": (tr, tc), "direction": DIRS[name]})
    return sources


def _trace_target(
    tip: tuple[int, int],
    direction: tuple[int, int],
    mirrors: dict[tuple[int, int], str],
    interior_lookup: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> int:
    h, w = shape
    r, c = tip
    dr, dc = direction
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray reflection loop detected"
        seen.add(state)
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "ray left the grid before reaching a frame"
        target = interior_lookup.get((r, c))
        if target is not None:
            return target
        orient = mirrors.get((r, c))
        if orient is not None:
            dr, dc = REFLECT[orient][(dr, dc)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored arrow-guided projection into a grey frame and fill that frame's interior.

    Input structure:
    - Black background with several small solid-colored rectangles, each paired with a same-colored 3-pixel arrow head.
    - Yellow pixels form straight 45-degree mirror segments, each entirely "/" or "\".
    - Grey pixels form hollow rectangular target frames whose interiors are black.

    Rule:
    - Detect each arrow tip to recover the outgoing direction for its colored source block.
    - Cast a straight ray from that arrow; whenever it reaches a yellow mirror segment, reflect it according to the mirror's diagonal slope.
    - Continue the ray until it enters the interior of a grey frame, then fill that whole frame interior with the source color.
    - Leave the source blocks, arrows, mirrors, frame borders, and the invisible ray path otherwise unchanged.

    Key insight:
    - The transformation combines two separate cues: the 3-pixel arrow head gives the initial cardinal direction, and the mirror slope determines the 90-degree reflection that selects the target frame.
    """
    g = np.asarray(grid, dtype=int)
    frames, interior_lookup = _find_frames(g)
    mirrors = _find_mirrors(g)
    sources = _find_sources(g)

    assignments: dict[int, int] = {}
    for source in sources:
        target = _trace_target(
            source["tip"],  # type: ignore[arg-type]
            source["direction"],  # type: ignore[arg-type]
            mirrors,
            interior_lookup,
            g.shape,
        )
        assert target not in assignments, "multiple rays cannot target the same frame"
        assignments[target] = int(source["color"])

    out = g.copy()
    for idx, color in assignments.items():
        for r, c in frames[idx]["interior"]:  # type: ignore[index]
            out[r, c] = color
    return out


def _sample_combos(n_entities: int, rng: random.Random) -> list[tuple[str, str]]:
    directions = rng.sample(list(DIRS.keys()), n_entities)
    mirror_types = list(MIRROR_TYPES[: min(2, n_entities)])
    while len(mirror_types) < n_entities:
        mirror_types.append(rng.choice(MIRROR_TYPES))
    rng.shuffle(mirror_types)
    combos = list(zip(directions, mirror_types))
    rng.shuffle(combos)
    return combos


def _mirror_segment_candidates(
    mirror_cell: tuple[int, int], orient: str, h: int, w: int, rng: random.Random
) -> list[list[tuple[int, int]]]:
    mr, mc = mirror_cell
    step = (1, 1) if orient == "\\" else (1, -1)
    candidates: list[list[tuple[int, int]]] = []
    lengths = [2, 3, 4, 5]
    rng.shuffle(lengths)
    for length in lengths:
        starts = list(range(-length + 1, 1))
        rng.shuffle(starts)
        for start in starts:
            cells = []
            ok = True
            for k in range(start, start + length):
                r = mr + k * step[0]
                c = mc + k * step[1]
                if not (0 <= r < h and 0 <= c < w):
                    ok = False
                    break
                cells.append((r, c))
            if ok:
                candidates.append(cells)
    return candidates


def _block_and_arrow_from_tip(
    tip: tuple[int, int], direction_name: str, rng: random.Random
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    tr, tc = tip
    if direction_name in ("left", "right"):
        bh = rng.randint(3, 5)
        bw = rng.randint(1, 3)
        anchor = rng.randint(1, bh - 2)
        r0 = tr - anchor
        c0 = tc - bw - 1 if direction_name == "right" else tc + 2
        block = {(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)}
        if direction_name == "right":
            arrow = {(tr, tc), (tr - 1, c0 + bw), (tr + 1, c0 + bw)}
        else:
            arrow = {(tr, tc), (tr - 1, c0 - 1), (tr + 1, c0 - 1)}
        return block, arrow

    bh = rng.randint(1, 3)
    bw = rng.randint(3, 5)
    anchor = rng.randint(1, bw - 2)
    r0 = tr + 2 if direction_name == "up" else tr - bh - 1
    c0 = tc - anchor
    block = {(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)}
    if direction_name == "up":
        arrow = {(tr, tc), (r0 - 1, tc - 1), (r0 - 1, tc + 1)}
    else:
        arrow = {(tr, tc), (r0 + bh, tc - 1), (r0 + bh, tc + 1)}
    return block, arrow


def _place_entity_in_zone(
    zone: tuple[int, int, int, int],
    direction_name: str,
    mirror_type: str,
    color: int,
    rng: random.Random,
) -> dict[str, object] | None:
    zr0, zc0, zr1, zc1 = zone
    zone_h = zr1 - zr0 + 1
    zone_w = zc1 - zc0 + 1
    if zone_h < 10 or zone_w < 10:
        return None

    in_dir = DIRS[direction_name]
    out_dir = REFLECT[mirror_type][in_dir]
    for _ in range(ENTITY_RETRIES):
        fh_max = min(6, zone_h - 4)
        fw_max = min(7, zone_w - 4)
        if fh_max < 4 or fw_max < 4:
            return None
        fh = rng.randint(4, fh_max)
        fw = rng.randint(4, fw_max)
        fr0 = rng.randint(zr0 + 1, zr1 - fh)
        fc0 = rng.randint(zc0 + 1, zc1 - fw)
        frame_bbox = (fr0, fc0, fr0 + fh - 1, fc0 + fw - 1)
        frame_border = set(_frame_border(frame_bbox))
        frame_interior = set(_frame_interior(frame_bbox))

        if out_dir[0] == 0:
            target_row = rng.randint(fr0 + 1, fr0 + fh - 2)
            gap_out = rng.randint(2, 5)
            mirror_cell = (
                target_row,
                fc0 - gap_out if out_dir == DIRS["right"] else fc0 + fw - 1 + gap_out,
            )
            enter = (
                target_row,
                fc0 + 1 if out_dir == DIRS["right"] else fc0 + fw - 2,
            )
        else:
            target_col = rng.randint(fc0 + 1, fc0 + fw - 2)
            gap_out = rng.randint(2, 5)
            mirror_cell = (
                fr0 - gap_out if out_dir == DIRS["down"] else fr0 + fh - 1 + gap_out,
                target_col,
            )
            enter = (
                fr0 + 1 if out_dir == DIRS["down"] else fr0 + fh - 2,
                target_col,
            )

        mr, mc = mirror_cell
        if not _all_in_rect({mirror_cell}, zone):
            continue

        gap_in = rng.randint(2, 5)
        if in_dir == DIRS["right"]:
            tip = (mr, mc - gap_in)
        elif in_dir == DIRS["left"]:
            tip = (mr, mc + gap_in)
        elif in_dir == DIRS["down"]:
            tip = (mr - gap_in, mc)
        else:
            tip = (mr + gap_in, mc)

        block_arrow = _block_and_arrow_from_tip(tip, direction_name, rng)
        assert block_arrow is not None
        block_cells, arrow_cells = block_arrow

        pre_path = set(_line_cells(tip, mirror_cell, in_dir))
        post_path = set(_line_cells(mirror_cell, enter, out_dir))
        structural = frame_border | frame_interior | block_cells | arrow_cells | pre_path | post_path
        if not _all_in_rect(structural, zone):
            continue

        for mirror_cells in _mirror_segment_candidates(mirror_cell, mirror_type, zr1 + 1, zc1 + 1, rng):
            mirror_set = set(mirror_cells)
            if not _all_in_rect(mirror_set, zone):
                continue
            visible = frame_border | block_cells | arrow_cells | mirror_set
            if frame_interior & visible:
                continue
            if len(visible) != len(frame_border) + len(block_cells) + len(arrow_cells) + len(mirror_set):
                continue
            return {
                "color": color,
                "tip": tip,
                "direction": in_dir,
                "frame_border": frame_border,
                "frame_interior": frame_interior,
                "mirror_cells": mirror_set,
                "source_cells": block_cells | arrow_cells,
                "expected_enter": enter,
            }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_entities = rng.randint(2, 4)
    min_side = 20 + 2 * n_entities
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    entities: list[dict[str, object]] = []

    combos = _sample_combos(n_entities, rng)
    colors = rng.sample(SOURCE_COLORS, n_entities)
    cut_r = rng.randint(h // 2 - 2, h // 2 + 2)
    cut_c = rng.randint(w // 2 - 2, w // 2 + 2)
    zones = [
        (1, 1, cut_r - 1, cut_c - 1),
        (1, cut_c + 1, cut_r - 1, w - 2),
        (cut_r + 1, 1, h - 2, cut_c - 1),
        (cut_r + 1, cut_c + 1, h - 2, w - 2),
    ]
    usable_zones = [zone for zone in zones if zone[2] - zone[0] + 1 >= 10 and zone[3] - zone[1] + 1 >= 10]
    if len(usable_zones) < n_entities:
        return None
    chosen_zones = rng.sample(usable_zones, n_entities)

    for zone, (direction_name, mirror_type), color in zip(chosen_zones, combos, colors):
        entity = _place_entity_in_zone(zone, direction_name, mirror_type, color, rng)
        if entity is None:
            return None
        entities.append(entity)

    grid = np.full((h, w), BG, dtype=int)
    for entity in entities:
        for r, c in entity["frame_border"]:  # type: ignore[index]
            grid[r, c] = FRAME
        for r, c in entity["mirror_cells"]:  # type: ignore[index]
            grid[r, c] = MIRROR
        for r, c in entity["source_cells"]:  # type: ignore[index]
            grid[r, c] = int(entity["color"])

    expected = grid.copy()
    for entity in entities:
        for r, c in entity["frame_interior"]:  # type: ignore[index]
            expected[r, c] = int(entity["color"])

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-reflection ARC input")
