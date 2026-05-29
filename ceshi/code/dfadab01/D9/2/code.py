# Variant of dfadab01
# Dimension: D9 — 字形放置的多步处理顺序 (multi_step)
# Change: 单步处理：所有标记同时被替换为字形，无先后顺序依赖... → 迭代放置：字形触发新标记
#
# Design analysis:
# spec:
#   retained_structure:
#     - Input is still a black canvas with single-pixel control markers from {2,3,5,8}.
#     - The glyph alphabet is still fixed 4x4 patterns derived from the original task.
#     - A legend entry is still optional in spirit and uses the same placement scheme:
#       one 4x4 sample glyph plus a matching label marker at (+4,+4). Here it is always
#       emitted because the extra corner colors are part of the rule students must learn.
#   D9 concretization:
#     - Iteration is driven only by the lower-right corner of a placed glyph.
#     - If that lower-right corner is painted in a different marker color, that cell becomes
#       a new marker for the next wave.
#     - To keep the shared seed visible after the child glyph lands on the same pixel, glyphs
#       2/5/8 also tint their upper-left filled cell with their own marker color. This is the
#       minimal support change needed for the propagation joint to stay visible in the final
#       image instead of being overwritten by an unrelated solid color.
#     - Trigger cycle: 8 -> 5 -> 2 -> 8. Marker 3 keeps its original ring and does not spawn.
#   behavior_trace:
#     - Example: an 8-marker at (1,1) paints an orange X whose lower-right corner is color 5.
#     - Wave 2 treats that visible 5 corner at (4,4) as a new marker and paints the 5-glyph there.
#     - That 5-glyph ends with a 2 corner at (7,7), which spawns the 2-glyph in wave 3, and so on
#       until the next seed would fall outside the grid.
#     - In the original task, only the first glyph at (1,1) would be painted.
#   complexity_calibration:
#     - Each glyph has at most one outgoing trigger, so the process is iterative but not branchy.
#     - Three propagating types are enough to produce clearly visible multi-round chains without
#       introducing a large unseen state space.
#   implicit_assumption_review:
#     - The original one-pass `argwhere` loop assumed no newly-created markers; this variant must
#       keep a wave frontier and a discovered-marker table.
#     - The original legend skip only ignored the below-right label pixel; here the whole legend
#       4x4 patch must be masked out because legend glyphs themselves contain marker-colored corners.
#     - Overwrite order is no longer harmless: later waves paint on top of earlier ones, so the
#       solver must process waves in order, not all markers simultaneously.
#   placement_legality:
#     - The generator simulates every root marker's full chain, records both painted cells and
#       generated marker anchors, and rejects any new root whose chain intersects an existing one.
#     - The legend is placed in an empty 5x5 zone so it never covers a real input marker.
#   observability_and_cleanliness:
#     - Every generated grid contains at least one chain with >= 3 placed glyphs and is checked
#       against a single-step baseline to ensure the iterative rule materially changes the output.
#     - Chains remain visually tidy because different roots are kept fully disjoint.

from __future__ import annotations

import random

import numpy as np


GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    2: (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    5: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    8: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}

NEXT_MARKER: dict[int, int] = {
    2: 8,
    5: 2,
    8: 5,
}

ROOT_DEPTH_MIN = 3
GEN_RETRIES = 400


def _build_template(marker_color: int) -> np.ndarray:
    glyph_color, mask = GLYPHS[marker_color]
    template = np.zeros((4, 4), dtype=int)
    template[mask] = glyph_color

    # Preserve the parent marker at the top-left corner whenever that corner exists.
    if bool(mask[0, 0]):
        template[0, 0] = int(marker_color)

    nxt = NEXT_MARKER.get(marker_color)
    if nxt is not None:
        assert bool(mask[3, 3])
        template[3, 3] = int(nxt)
    return template


TEMPLATES: dict[int, np.ndarray] = {marker: _build_template(marker) for marker in GLYPHS}
MARKER_COLORS = set(GLYPHS)
ALLOWED_COLORS = {0, 1, 2, 3, 4, 5, 6, 7, 8}


def _paint_template_clipped(out: np.ndarray, r0: int, c0: int, template: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            val = int(template[dr, dc])
            if 0 <= cc < w and val != 0:
                out[rr, cc] = val


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, template: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    return bool(np.array_equal(g[r0 : r0 + 4, c0 : c0 + 4], template))


def _legend_mask(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    mask = np.zeros_like(g, dtype=bool)
    for marker_color, template in TEMPLATES.items():
        for r0 in range(h - 4):
            for c0 in range(w - 4):
                if not _has_source_glyph(g, r0, c0, template):
                    continue
                lr, lc = r0 + 4, c0 + 4
                if 0 <= lr < h and 0 <= lc < w and int(g[lr, lc]) == int(marker_color):
                    mask[r0 : r0 + 4, c0 : c0 + 4] = True
                    mask[lr, lc] = True
    return mask


def _initial_markers(g: np.ndarray, masked: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val in MARKER_COLORS and not bool(masked[r, c]):
                markers.append((r, c, val))
    return markers


def _propagation_positions(r: int, c: int, marker_color: int, h: int, w: int) -> list[tuple[int, int, int]]:
    positions: list[tuple[int, int, int]] = []
    cur_r, cur_c, cur_color = int(r), int(c), int(marker_color)
    while True:
        positions.append((cur_r, cur_c, cur_color))
        nxt = NEXT_MARKER.get(cur_color)
        if nxt is None:
            break
        nr, nc = cur_r + 3, cur_c + 3
        if not (0 <= nr < h and 0 <= nc < w):
            break
        cur_r, cur_c, cur_color = nr, nc, int(nxt)
    return positions


def _positions_to_cells(
    positions: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, marker_color in positions:
        template = TEMPLATES[marker_color]
        for dr in range(4):
            rr = r0 + dr
            if rr < 0 or rr >= h:
                continue
            for dc in range(4):
                cc = c0 + dc
                if 0 <= cc < w and int(template[dr, dc]) != 0:
                    cells.add((rr, cc))
    return cells


def _single_step_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    masked = _legend_mask(g)
    out = np.zeros_like(g)
    for r, c, marker_color in _initial_markers(g, masked):
        _paint_template_clipped(out, r, c, TEMPLATES[marker_color])
    return out


def propagation_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    masked = _legend_mask(g)
    initial = _initial_markers(g, masked)
    h, w = g.shape
    chain_lengths = [len(_propagation_positions(r, c, color, h, w)) for r, c, color in initial]
    total_placed = int(sum(chain_lengths))
    max_chain = int(max(chain_lengths)) if chain_lengths else 0
    return {
        "initial_markers": int(len(initial)),
        "total_placed_glyphs": total_placed,
        "generated_glyphs": int(total_placed - len(initial)),
        "max_chain_length": max_chain,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each control pixel into its 4x4 glyph, then keep chasing any new corner-seeds.

    Input structure:
    - Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
    - A legend entry may appear as a full 4x4 sample glyph plus a matching marker at its lower-right label cell.
    - Some sample glyph corners are themselves marker-colored; inside the legend they are explanatory only.

    Rule:
    - Replace every real marker by its fixed 4x4 glyph at that marker's top-left position, clipping at the border.
    - If a placed glyph's lower-right corner is painted in a different marker color, treat that corner cell as a new marker in the next wave.
    - Continue wave by wave until no fresh marker remains; erase the legend glyph and its label from the output.

    Key insight:
    - This is no longer a one-pass substitution task: the output must be built by simulating chained seed propagation from glyph corners while ignoring legend pixels entirely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(int(x) for x in np.unique(g)).issubset(ALLOWED_COLORS)

    h, w = g.shape
    masked = _legend_mask(g)
    wave = sorted(_initial_markers(g, masked))
    assert wave

    out = np.zeros_like(g)
    discovered: dict[tuple[int, int], int] = {(r, c): color for r, c, color in wave}

    while wave:
        next_wave_map: dict[tuple[int, int], int] = {}
        for r, c, marker_color in wave:
            _paint_template_clipped(out, r, c, TEMPLATES[marker_color])
            nxt = NEXT_MARKER.get(marker_color)
            if nxt is None:
                continue
            nr, nc = r + 3, c + 3
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            coord = (nr, nc)
            prev = discovered.get(coord)
            if prev is not None:
                assert prev == int(nxt)
                continue
            prev = next_wave_map.get(coord)
            if prev is None:
                next_wave_map[coord] = int(nxt)
            else:
                assert prev == int(nxt)

        wave = []
        for (r, c), marker_color in sorted(next_wave_map.items()):
            discovered[(r, c)] = int(marker_color)
            wave.append((r, c, int(marker_color)))

    return out


def _choose_canvas() -> tuple[int, int]:
    size = random.choice([16, 18, 20, 22])
    return size, size


def _candidate_grid(h: int, w: int) -> list[tuple[int, int]]:
    for _ in range(100):
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        row_spacing = random.randint(4, 6)
        col_spacing = random.randint(4, 6)
        max_r0 = h - 1 - (rows - 1) * row_spacing
        max_c0 = w - 1 - (cols - 1) * col_spacing
        if max_r0 < 0 or max_c0 < 0:
            continue
        r0 = random.randint(0, max_r0)
        c0 = random.randint(0, max_c0)
        coords = [(r0 + i * row_spacing, c0 + j * col_spacing) for i in range(rows) for j in range(cols)]
        return coords
    raise RuntimeError("failed to make candidate grid")


def _can_place_chain(
    positions: list[tuple[int, int, int]],
    cells: set[tuple[int, int]],
    occupied_cells: set[tuple[int, int]],
    occupied_markers: set[tuple[int, int]],
) -> bool:
    marker_coords = {(r, c) for r, c, _ in positions}
    return not (
        cells & occupied_cells
        or marker_coords & occupied_markers
        or cells & occupied_markers
        or marker_coords & occupied_cells
    )


def _place_legend(inp: np.ndarray, used_markers: list[int]) -> None:
    h, w = inp.shape
    choices: list[tuple[int, int, int]] = []
    for marker_color in sorted(set(used_markers)):
        template = TEMPLATES[marker_color]
        for r0 in range(h - 4):
            for c0 in range(w - 4):
                lr, lc = r0 + 4, c0 + 4
                bbox = inp[r0 : r0 + 5, c0 : c0 + 5]
                if bbox.shape != (5, 5):
                    continue
                if np.any(bbox != 0):
                    continue
                if lr >= h or lc >= w:
                    continue
                choices.append((marker_color, r0, c0))

    if not choices:
        raise RuntimeError("no room for legend")

    marker_color, r0, c0 = random.choice(choices)
    template = TEMPLATES[marker_color]
    for dr in range(4):
        for dc in range(4):
            val = int(template[dr, dc])
            if val != 0:
                inp[r0 + dr, c0 + dc] = val
    inp[r0 + 4, c0 + 4] = int(marker_color)


def _gen_attempt() -> np.ndarray:
    h, w = _choose_canvas()
    candidates = _candidate_grid(h, w)
    random.shuffle(candidates)

    occupied_cells: set[tuple[int, int]] = set()
    occupied_markers: set[tuple[int, int]] = set()
    roots: list[tuple[int, int, int]] = []

    guaranteed: tuple[int, int, int] | None = None
    guarantee_order = [8, 5, 2]
    random.shuffle(guarantee_order)
    for r, c in sorted(candidates):
        for marker_color in guarantee_order:
            positions = _propagation_positions(r, c, marker_color, h, w)
            if len(positions) < ROOT_DEPTH_MIN:
                continue
            cells = _positions_to_cells(positions, h, w)
            if _can_place_chain(positions, cells, occupied_cells, occupied_markers):
                guaranteed = (r, c, marker_color)
                occupied_cells |= cells
                occupied_markers |= {(rr, cc) for rr, cc, _ in positions}
                roots.append(guaranteed)
                break
        if guaranteed is not None:
            break

    if guaranteed is None:
        raise RuntimeError("could not place a propagating root")

    target_roots = random.randint(3, min(6, len(candidates)))
    for r, c in candidates:
        if len(roots) >= target_roots:
            break
        if any((r, c) == (rr, cc) for rr, cc, _ in roots):
            continue
        colors = [2, 3, 5, 8]
        random.shuffle(colors)
        for marker_color in colors:
            positions = _propagation_positions(r, c, marker_color, h, w)
            cells = _positions_to_cells(positions, h, w)
            if not _can_place_chain(positions, cells, occupied_cells, occupied_markers):
                continue
            roots.append((r, c, marker_color))
            occupied_cells |= cells
            occupied_markers |= {(rr, cc) for rr, cc, _ in positions}
            break

    if len(roots) < 3:
        raise RuntimeError("too few roots")

    inp = np.zeros((h, w), dtype=int)
    for r, c, marker_color in roots:
        inp[r, c] = int(marker_color)
    _place_legend(inp, [color for _, _, color in roots])

    metrics = propagation_metrics(inp)
    if metrics["max_chain_length"] < ROOT_DEPTH_MIN:
        raise RuntimeError("degenerate propagation depth")
    if metrics["generated_glyphs"] <= 0:
        raise RuntimeError("no generated glyphs")
    if np.array_equal(main(inp), _single_step_output(inp)):
        raise RuntimeError("iterative rule collapsed to single-step output")
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp = _gen_attempt()
            _ = main(inp)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid multi-step dfadab01 variant")
