# Combo variant of dfadab01 (index 9)
# combo_spec: D9/2+D10/1
# Name: 迭代放置+重叠特殊色标记
#
# Design analysis:
# a) Concrete behavior and trace:
#    - The original task is a one-pass marker->4x4 glyph substitution with optional legend removal.
#    - This combo keeps the same four base glyph masks, but uses propagation templates whose lower-right corner
#      can itself be another marker color. We concretize "corner-triggered propagation" to that single corner,
#      matching the D9 reference while keeping the visual story compact enough to infer from few examples.
#    - Example: an 8-marker at (1,1) paints an X-template whose lower-right cell is marker 5. Wave 2 therefore
#      places the 5-template at (4,4); that template exposes marker 8 at (7,7); wave 3 places another 8-template.
#      If there is also a root marker at (4,1), its glyph overlaps the wave-2 glyph at (4,4), and every shared
#      painted cell becomes color 9. In the original solver, neither the extra wave nor the 9-collision band exists.
# b) Complexity calibration:
#    - Only two marker types propagate in generated samples (8->5->8->...); this keeps the visible palette at
#      five non-background colors {5,6,7,8,9} while still giving a genuine multi-wave chain.
#    - Overlap is rendered with a single special color 9 rather than color-pair-specific rules, so collisions are
#      structurally visible without introducing another hidden mapping table.
# c) Hidden assumption review:
#    - The original `argwhere` loop assumed no newly-created markers; here we need an explicit wave frontier and
#      a discovered-marker table.
#    - The original legend skip only ignored the label pixel. That is no longer sufficient because legend glyphs
#      now contain a propagation-colored lower-right corner; we must mask the whole 4x4 legend patch plus its
#      (+4,+4) label.
#    - The original renderer assumed last-write-wins was harmless because spacing avoided collisions. That no
#      longer holds, so rendering must aggregate coverage counts across all waves before choosing output colors.
#    - Generated markers may land on existing discovered markers. The fused solver assumes such re-hits, if any,
#      agree on marker color; the generator enforces this by rejecting conflicting samples.
# d) Placement legality:
#    - Input markers remain distinct single pixels; legends are inserted only inside empty 5x5 zones.
#    - The generator simulates the full propagation process before accepting a sample, so clipped waves, repeated
#      seeds, and collision locations are all validated against the actual solver logic.
# e) Visual observability:
#    - Every accepted sample must have at least one generated glyph, at least one overlap-colored cell, and at
#      least one overlap cell touched by a generated glyph. This prevents degeneration into either pure D9 or pure
#      D10 behavior.
#    - We also compare against a single-step collision-aware baseline and reject samples where propagation makes
#      no visible difference.
# f) Visual cleanliness:
#    - Marker roots are still drawn on a sparse rectilinear lattice, but one axis often uses spacing 3 so overlap
#      appears as readable stripes instead of chaotic full-canvas saturation.
#    - Samples with excessive 9-coverage are rejected to preserve recognizable base glyph structure.
# g) Color inference:
#    - The only special-purpose visible color is 9, and it appears exactly on shared painted cells. Propagation is
#      indicated by repeated diagonal +3,+3 anchor jumps and by alternating 5/8 corner seeds inside placed glyphs,
#      keeping the total non-background palette within five colors.
# h) Main/generator contract:
#    - `main()` assumes any complete legend patch is an exact 4x4 propagation template with its matching label at
#      (+4,+4); `generate_input()` only emits legends in that form.
#    - `main()` also assumes at least one non-legend marker exists and that no coordinate is assigned two different
#      marker colors through propagation; the generator checks both conditions before returning a sample.
#    - Generated samples deliberately restrict active roots to {5,8} so the color vocabulary stays ARC-readable.

from __future__ import annotations

import random

import numpy as np


OVERLAP_COLOR = 9
TRIGGER_OFFSET = (3, 3)
GEN_RETRIES = 400
MIN_PROPAGATION_WAVES = 3
ACTIVE_ROOT_COLORS = (5, 8)

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
    5: 8,
    8: 5,
}

MARKER_COLORS = set(GLYPHS)
ALLOWED_INPUT_COLORS = set(range(9))


def _build_template(marker_color: int) -> np.ndarray:
    glyph_color, mask = GLYPHS[marker_color]
    template = np.zeros((4, 4), dtype=int)
    template[mask] = glyph_color

    nxt = NEXT_MARKER.get(marker_color)
    if nxt is not None:
        assert bool(mask[3, 3])
        template[3, 3] = int(nxt)

    return template


TEMPLATES: dict[int, np.ndarray] = {marker_color: _build_template(marker_color) for marker_color in GLYPHS}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, template: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    return bool(np.array_equal(g[r0 : r0 + 4, c0 : c0 + 4], template))


def _legend_mask(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    masked = np.zeros_like(g, dtype=bool)
    for marker_color, template in TEMPLATES.items():
        for r0 in range(max(0, h - 4)):
            for c0 in range(max(0, w - 4)):
                if not _has_source_glyph(g, r0, c0, template):
                    continue
                lr, lc = r0 + 4, c0 + 4
                if 0 <= lr < h and 0 <= lc < w and int(g[lr, lc]) == int(marker_color):
                    masked[r0 : r0 + 4, c0 : c0 + 4] = True
                    masked[lr, lc] = True
    return masked


def _initial_markers(g: np.ndarray, masked: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val in MARKER_COLORS and not bool(masked[r, c]):
                markers.append((r, c, val))
    return markers


def _accumulate_template_clipped(
    cover_count: np.ndarray,
    cover_color: np.ndarray,
    generated_cover_count: np.ndarray,
    r0: int,
    c0: int,
    template: np.ndarray,
    is_generated: bool,
) -> None:
    h, w = cover_count.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            val = int(template[dr, dc])
            if 0 <= cc < w and val != 0:
                if cover_count[rr, cc] == 0:
                    cover_color[rr, cc] = val
                cover_count[rr, cc] += 1
                if is_generated:
                    generated_cover_count[rr, cc] += 1


def _simulate_placements(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
) -> list[tuple[int, int, int, int, bool]]:
    h, w = shape
    wave = sorted((int(r), int(c), int(color)) for r, c, color in markers)
    assert wave

    placements: list[tuple[int, int, int, int, bool]] = []
    discovered: dict[tuple[int, int], int] = {(r, c): color for r, c, color in wave}
    wave_index = 0

    while wave:
        next_wave_map: dict[tuple[int, int], int] = {}
        for r, c, marker_color in wave:
            placements.append((wave_index, r, c, marker_color, wave_index > 0))

            template = TEMPLATES[marker_color]
            dr, dc = TRIGGER_OFFSET
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = int(template[dr, dc])
            if nxt not in MARKER_COLORS:
                continue

            coord = (nr, nc)
            prev = discovered.get(coord)
            if prev is not None:
                if prev != nxt:
                    raise ValueError("conflicting discovered marker colors")
                continue

            prev = next_wave_map.get(coord)
            if prev is None:
                next_wave_map[coord] = nxt
            elif prev != nxt:
                raise ValueError("conflicting next-wave marker colors")

        wave = [(r, c, color) for (r, c), color in sorted(next_wave_map.items())]
        discovered.update(next_wave_map)
        wave_index += 1

    return placements


def _render_placements(
    shape: tuple[int, int],
    placements: list[tuple[int, int, int, int, bool]],
) -> tuple[np.ndarray, dict[str, int]]:
    cover_count = np.zeros(shape, dtype=int)
    cover_color = np.zeros(shape, dtype=int)
    generated_cover_count = np.zeros(shape, dtype=int)

    for _, r, c, marker_color, is_generated in placements:
        _accumulate_template_clipped(
            cover_count,
            cover_color,
            generated_cover_count,
            r,
            c,
            TEMPLATES[marker_color],
            is_generated,
        )

    out = np.zeros(shape, dtype=int)
    single = cover_count == 1
    overlap = cover_count >= 2
    out[single] = cover_color[single]
    out[overlap] = OVERLAP_COLOR

    metrics = {
        "overlap_pixels": int(np.count_nonzero(overlap)),
        "propagated_overlap_pixels": int(np.count_nonzero(overlap & (generated_cover_count >= 1))),
        "painted_pixels": int(np.count_nonzero(cover_count)),
    }
    return out, metrics


def _single_step_output(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    placements = [(0, int(r), int(c), int(color), False) for r, c, color in markers]
    out, _ = _render_placements(shape, placements)
    return out


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    masked = _legend_mask(g)
    markers = _initial_markers(g, masked)
    placements = _simulate_placements(g.shape, markers)
    out, render_metrics = _render_placements(g.shape, placements)
    baseline = _single_step_output(g.shape, markers)

    max_wave_index = max(wave for wave, *_ in placements)
    return {
        "initial_markers": int(len(markers)),
        "total_placed_glyphs": int(len(placements)),
        "generated_glyphs": int(sum(1 for wave, *_ in placements if wave > 0)),
        "wave_count": int(max_wave_index + 1),
        "overlap_pixels": int(render_metrics["overlap_pixels"]),
        "propagated_overlap_pixels": int(render_metrics["propagated_overlap_pixels"]),
        "painted_pixels": int(render_metrics["painted_pixels"]),
        "changed_vs_single_step": int(np.count_nonzero(out != baseline)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand real markers into propagation-enabled 4x4 glyphs and show every collision in color 9.

    Input structure:
    - The canvas is black except for single-pixel markers in colors 2, 3, 5, and/or 8.
    - A legend may also appear as one full 4x4 sample template plus its matching marker at the sample's lower-right label cell.
    - Unlike the original task, some visible sample/template corners can themselves use marker colors because those corners explain the propagation rule.

    Rule:
    - Ignore the entire legend patch and its label; every remaining marker is a true glyph anchor.
    - Place each anchor's fixed 4x4 template with its top-left at the marker position, clipping at the grid edge.
    - After each wave, if a placed template's lower-right corner cell is another marker color, treat that corner as a fresh marker for the next wave.
    - Render all waves together: cells painted by exactly one glyph keep that glyph's color, while any cell painted by two or more glyphs becomes the special overlap color 9.

    Key insight:
    - Solving requires two coupled inferences at once: the marker dictionary is still local, but the full output only appears after simulating diagonal corner-seed propagation and replacing every shared painted cell with 9.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(int(x) for x in np.unique(g)).issubset(ALLOWED_INPUT_COLORS)

    masked = _legend_mask(g)
    markers = _initial_markers(g, masked)
    assert markers

    placements = _simulate_placements(g.shape, markers)
    out, _ = _render_placements(g.shape, placements)
    return out


def _choose_canvas(rng: random.Random) -> tuple[int, int]:
    size = rng.choice([18, 20, 22])
    return size, size


def _build_root_lattice(rng: random.Random, h: int, w: int) -> tuple[list[int], list[int]]:
    for _ in range(120):
        row_step = rng.choice([3, 4, 5])
        col_step = rng.choice([3, 4, 5])
        if row_step != 3 and col_step != 3:
            continue

        rows = rng.randint(3, 4 if h <= 20 else 5)
        cols = rng.randint(3, 4 if w <= 20 else 5)
        max_r0 = h - 1 - (rows - 1) * row_step
        max_c0 = w - 1 - (cols - 1) * col_step
        if max_r0 < 0 or max_c0 < 0:
            continue

        cap_r0 = min(max_r0, h - 10)
        cap_c0 = min(max_c0, w - 10)
        if cap_r0 < 0 or cap_c0 < 0:
            continue

        r0 = rng.randint(0, cap_r0)
        c0 = rng.randint(0, cap_c0)
        rows_pos = [r0 + i * row_step for i in range(rows)]
        cols_pos = [c0 + j * col_step for j in range(cols)]
        return rows_pos, cols_pos

    raise RuntimeError("failed to build a feasible root lattice")


def _partner_coord(rows_pos: list[int], cols_pos: list[int]) -> tuple[int, int]:
    r0 = rows_pos[0]
    c0 = cols_pos[0]
    if len(rows_pos) >= 2 and rows_pos[1] - rows_pos[0] == 3:
        return rows_pos[1], c0
    if len(cols_pos) >= 2 and cols_pos[1] - cols_pos[0] == 3:
        return r0, cols_pos[1]
    raise RuntimeError("lattice does not support the forced propagated-overlap partner")


def _place_legend(rng: random.Random, grid: np.ndarray, used_markers: list[int]) -> None:
    if rng.random() >= 0.7:
        return

    h, w = grid.shape
    choices: list[tuple[int, int, int]] = []
    for marker_color in sorted(set(used_markers)):
        template = TEMPLATES[marker_color]
        for r0 in range(max(0, h - 4)):
            for c0 in range(max(0, w - 4)):
                bbox = grid[r0 : r0 + 5, c0 : c0 + 5]
                if bbox.shape != (5, 5):
                    continue
                if np.any(bbox != 0):
                    continue
                choices.append((marker_color, r0, c0))

    if not choices:
        return

    marker_color, r0, c0 = rng.choice(choices)
    template = TEMPLATES[marker_color]
    for dr in range(4):
        for dc in range(4):
            val = int(template[dr, dc])
            if val != 0:
                grid[r0 + dr, c0 + dc] = val
    grid[r0 + 4, c0 + 4] = int(marker_color)


def _gen_attempt(rng: random.Random) -> np.ndarray:
    h, w = _choose_canvas(rng)
    rows_pos, cols_pos = _build_root_lattice(rng, h, w)
    cells = [(r, c) for r in rows_pos for c in cols_pos]

    root_anchor = (rows_pos[0], cols_pos[0])
    partner_anchor = _partner_coord(rows_pos, cols_pos)
    chain_anchors = {(r, c) for _, r, c, _, _ in _simulate_placements((h, w), [(root_anchor[0], root_anchor[1], 8)])}

    available = [cell for cell in cells if cell not in chain_anchors and cell != partner_anchor]
    min_roots = 4
    max_roots = min(7, len(cells))
    if max_roots < min_roots:
        raise RuntimeError("lattice too small")
    root_count = rng.randint(min_roots, max_roots)
    extra_needed = root_count - 2
    if len(available) < extra_needed:
        raise RuntimeError("not enough free lattice cells")

    extras = rng.sample(available, extra_needed)
    roots: list[tuple[int, int, int]] = [
        (root_anchor[0], root_anchor[1], 8),
        (partner_anchor[0], partner_anchor[1], rng.choice(ACTIVE_ROOT_COLORS)),
    ]
    extra_colors = list(ACTIVE_ROOT_COLORS)
    for r, c in extras:
        roots.append((r, c, rng.choice(extra_colors)))

    rng.shuffle(roots)

    placements = _simulate_placements((h, w), roots)
    out, render_metrics = _render_placements((h, w), placements)
    baseline = _single_step_output((h, w), roots)

    metrics = {
        "total_placed_glyphs": int(len(placements)),
        "generated_glyphs": int(sum(1 for wave, *_ in placements if wave > 0)),
        "wave_count": int(max(wave for wave, *_ in placements) + 1),
        "overlap_pixels": int(render_metrics["overlap_pixels"]),
        "propagated_overlap_pixels": int(render_metrics["propagated_overlap_pixels"]),
        "painted_pixels": int(render_metrics["painted_pixels"]),
        "changed_vs_single_step": int(np.count_nonzero(out != baseline)),
        "visible_color_count": int(len({int(x) for x in np.unique(out) if int(x) != 0})),
    }

    if metrics["wave_count"] < MIN_PROPAGATION_WAVES:
        raise RuntimeError("too few propagation waves")
    if metrics["generated_glyphs"] <= 0:
        raise RuntimeError("no generated glyphs")
    if metrics["overlap_pixels"] < 3:
        raise RuntimeError("too little overlap")
    if metrics["propagated_overlap_pixels"] < 2:
        raise RuntimeError("overlap did not materially involve generated glyphs")
    if metrics["changed_vs_single_step"] < 6:
        raise RuntimeError("propagation not visibly distinct from single step")
    if metrics["overlap_pixels"] * 3 > metrics["painted_pixels"] * 2:
        raise RuntimeError("overlap saturation too high")
    if metrics["visible_color_count"] > 5:
        raise RuntimeError("too many visible output colors")

    grid = np.zeros((h, w), dtype=int)
    for r, c, color in roots:
        grid[r, c] = int(color)
    _place_legend(rng, grid, [color for _, _, color in roots])

    actual = main(grid)
    if not np.array_equal(actual, out):
        raise RuntimeError("solver/render mismatch")

    post_metrics = fusion_metrics(grid)
    if post_metrics["propagated_overlap_pixels"] < 2:
        raise RuntimeError("legend insertion changed core behavior")
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid fused dfadab01 variant")
