# Combo variant of 13e47133 (index 23)
# combo_spec: D3/3+D15/0
# Name: 边缘控制像素+图例行定义调色板

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The original task encoded each region palette by placing 0-3 interior marker
#    pixels on an inward diagonal and sorting them by Chebyshev depth. This combo
#    removes that source completely. The body now consists of full-height vertical
#    strips separated by wall columns, while row 0 is a legend row. Each strip has
#    one bottom-edge control pixel whose color is the strip's anchor. The legend is
#    also partitioned into vertical segments, but the legend contents are permuted:
#    the segment directly above a strip usually belongs to a different strip.
#    Example with three strips:
#      - strip A bottom control = 7, strip B = 4, strip C = 6
#      - top legend segments (left->right) show palettes [4,2], [6,8], [7,3]
#      - A must therefore ignore the segment above itself, find the legend segment
#        starting with 7, recover palette [7,3], and then fill by boundary depth.
#    Compared with the original:
#      - old solver would only see the bottom control 7 and collapse A to [7]
#      - plain legend-alignment would wrongly assign [4,2] to A
#      - the fused rule needs both cues: edge control selects the right legend
#        segment, legend order provides the actual palette cycle.
#
# b) Complexity calibration:
#    The geometry is intentionally restricted to 3-6 vertical strips. This keeps
#    the association puzzle visible without turning it into a generic graph match:
#    every region touches the bottom edge, every legend segment is a clean run, and
#    the only meaningful degrees of freedom are strip order, legend permutation, and
#    palette length 2-3. Those choices recur across samples so humans get repeated
#    support for the same inference pattern.
#
# c) Implicit assumption audit:
#    - Original main inferred markers from non-base cells inside each component.
#      That no longer holds: the only body clue is a bottom-edge anchor, and its
#      own depth is always 0, so it cannot define the whole palette.
#    - D15-alone style "read the legend segment directly above the region" is now
#      invalid because the legend contents are intentionally permuted across aligned
#      segments.
#    - Inferring bg/wall from the whole grid is unsafe because row 0 may contain
#      many extra palette colors. The solver must infer bg/wall from the body only.
#    - Matching a region to a legend segment by first color assumes those first
#      colors are unique. The generator enforces that by giving every strip a unique
#      non-bg, non-wall anchor color and preventing anchors from appearing in later
#      palette positions.
#
# d) Placement legality:
#    The generator first chooses strip widths, then reuses those same widths for the
#    legend segmentation, so no overlaps or partial segments occur. Bottom controls
#    are stamped strictly inside their strip on the global bottom row, away from
#    wall columns. Legend segments are separated by wall-colored cells and never
#    cross region boundaries.
#
# e) Visual observability:
#    A human can see two simultaneous clues:
#      - every strip has a lone colored pixel on its bottom edge
#      - the top row contains repeated palette snippets whose first colors match
#        those bottom anchors, but often not in the same column block
#    Across 2-3 examples, it becomes visible that vertical alignment is a decoy and
#    the repeated top-row sequence matched by anchor color is the real palette.
#
# f) Visual cleanliness:
#    The body is almost entirely base color plus straight wall columns, so the only
#    overlays are one bottom-edge control per strip and one top legend row. There is
#    no mixing of multiple symbolic systems inside the same region interior.
#
# g) Learnability of color changes:
#    No hidden numeric ordering is used. Palette recovery depends only on visible
#    repetition in the legend row, while region-to-legend assignment depends only on
#    visible color identity of the bottom-edge anchor. The generator also forces at
#    least one sample branch where bg appears inside a palette but not first, making
#    it clear that bg is not auto-prepended anymore.


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neigh (Chebyshev) distance to boundary for True-cells in component_mask.
    Boundary cells are those that touch False or the grid edge in their 8-neighborhood.
    Returns dist array with -1 outside component.
    """
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _recover_palette(seq: list[int]) -> list[int]:
    """Return the shortest repeating prefix that generates seq."""
    if not seq:
        raise ValueError("empty legend segment")
    for k in range(1, len(seq) + 1):
        if all(int(v) == int(seq[i % k]) for i, v in enumerate(seq)):
            return [int(v) for v in seq[:k]]
    return [int(v) for v in seq]


def _legend_segments(legend: np.ndarray, wall: int):
    runs = []
    c = 0
    w = len(legend)
    while c < w:
        if int(legend[c]) == int(wall):
            c += 1
            continue
        c0 = c
        seq = []
        while c < w and int(legend[c]) != int(wall):
            seq.append(int(legend[c]))
            c += 1
        palette = _recover_palette(seq)
        runs.append({"start": c0, "end": c - 1, "palette": palette, "key": int(palette[0])})
    return runs


def _component_anchor(body: np.ndarray, pts, base: int) -> int | None:
    """Return the bottom-edge control color for one strip component."""
    bh, _bw = body.shape
    controls = []
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        v = int(body[rr, cc])
        if rr == bh - 1 and v != int(base):
            controls.append((cc, v))
    if not controls:
        return None
    controls.sort()
    return int(controls[0][1])


def _component_span(pts) -> tuple[int, int]:
    cols = [int(c) for _r, c in pts]
    return min(cols), max(cols)


def _solve_with_palettes(grid: np.ndarray, palette_picker) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.shape[0] < 2:
        return g.copy()

    legend = g[0]
    body = g[1:]
    bh, bw = body.shape
    bg = _mode(body)
    other = body[body != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    comps = _cc4(body != wall)

    for pts in comps:
        comp_mask = np.zeros((bh, bw), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True
        base = _mode(body[comp_mask])
        palette = palette_picker(legend, body, pts, comp_mask, base, wall)
        if not palette:
            palette = [int(base)]
        dist = _boundary_distance(comp_mask)
        m = len(palette)
        for r, c in pts:
            out[int(r) + 1, int(c)] = int(palette[int(dist[int(r), int(c)]) % m])
    return out


def _aligned_palette_picker(legend, body, pts, comp_mask, base, wall):
    del body, comp_mask, base, wall
    c0, c1 = _component_span(pts)
    return _recover_palette([int(legend[c]) for c in range(c0, c1 + 1)])


def _control_only_palette_picker(legend, body, pts, comp_mask, base, wall):
    del legend, comp_mask, wall
    anchor = _component_anchor(body, pts, base)
    return [int(base)] if anchor is None else [int(anchor)]


def _fused_palette_picker(legend, body, pts, comp_mask, base, wall):
    del comp_mask
    anchor = _component_anchor(body, pts, base)
    if anchor is None:
        return [int(base)]
    legend_by_key = {seg["key"]: seg["palette"] for seg in _legend_segments(legend, wall)}
    return legend_by_key.get(int(anchor), [int(base)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned strips with concentric bands using edge anchors plus a legend row.

    Input structure:
    - Row 0 is a legend row split into colored segments by wall-colored separators.
    - Rows below row 0 form vertical free strips separated by wall columns.
    - Each strip interior is mostly the base color and carries a single non-base
      control pixel on the bottom edge; that control color matches the first color
      of exactly one legend segment, but not necessarily the segment above the strip.

    Rule:
    - For each strip, use its bottom-edge control color to find the matching legend
      segment whose repeated pattern starts with that same color.
    - Recover the shortest repeating cycle of that legend segment and use it as the
      strip palette exactly as written; the base color is only used if it appears
      explicitly in the legend.
    - Repaint the strip with palette[chebyshev_distance_to_strip_boundary mod len(palette)],
      while leaving the legend row and wall columns unchanged.

    Key insight: The edge control no longer encodes palette order by itself; it only
    tells you which top-row legend segment belongs to the strip. The legend provides
    the ordered cycle, and both clues are required because vertical alignment is
    intentionally misleading.
    """
    return _solve_with_palettes(grid, _fused_palette_picker)


# --------------- generator ---------------

H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MIN_REGION_W = 3
GEN_MAX_ATTEMPTS = 300


def _choose_region_count(W: int, rng: random.Random) -> int | None:
    feasible = []
    for n in range(N_REGIONS_MIN, N_REGIONS_MAX + 1):
        if W >= n * MIN_REGION_W + (n - 1):
            feasible.append(n)
    if not feasible:
        return None
    return rng.choice(feasible)


def _partition_widths(total: int, n: int, minimum: int, rng: random.Random) -> list[int]:
    widths = [minimum] * n
    extra = total - minimum * n
    for _ in range(extra):
        widths[rng.randrange(n)] += 1
    return widths


def _derangement(n: int, rng: random.Random) -> list[int]:
    order = list(range(n))
    if n <= 1:
        return order
    while True:
        perm = order[:]
        rng.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm


def _choose_colors(n_regions: int, rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    anchors = rng.sample(colors, n_regions)
    extras = [c for c in colors if c not in anchors]
    return int(bg), int(wall), [int(c) for c in anchors], [int(bg)] + [int(c) for c in extras]


def _make_palettes(widths: list[int], anchors: list[int], secondary_pool: list[int], rng: random.Random):
    max_len = min(3, min(widths), len(secondary_pool) + 1)
    if max_len < 2:
        return None

    palettes = []
    len3_target = max_len >= 3
    len3_idx = rng.randrange(len(anchors)) if len3_target else -1
    for idx, first in enumerate(anchors):
        if max_len >= 3:
            k = 3 if idx == len3_idx else rng.choice([2, 3])
        else:
            k = 2
        rest = rng.sample(secondary_pool, k - 1)
        palettes.append([int(first)] + [int(v) for v in rest])

    if not any(palette.index(secondary_pool[0]) > 0 for palette in palettes if secondary_pool[0] in palette):
        idx = rng.randrange(len(palettes))
        palettes[idx][1] = int(secondary_pool[0])

    return palettes


def _spans_from_widths(widths: list[int]) -> list[tuple[int, int]]:
    spans = []
    c = 0
    for i, width in enumerate(widths):
        spans.append((c, c + width - 1))
        c += width
        if i != len(widths) - 1:
            c += 1
    return spans


def _repeat_palette(palette: list[int], width: int) -> list[int]:
    return [int(palette[i % len(palette)]) for i in range(width)]


def _build_grid(
    H: int,
    W: int,
    bg: int,
    wall: int,
    widths: list[int],
    palettes: list[list[int]],
    legend_perm: list[int],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((H, W), int(bg), dtype=int)
    spans = _spans_from_widths(widths)

    for _i, (_c0, c1) in enumerate(spans[:-1]):
        wall_col = c1 + 1
        grid[1:, wall_col] = int(wall)
        grid[0, wall_col] = int(wall)

    for idx, (c0, c1) in enumerate(spans):
        width = c1 - c0 + 1
        legend_palette = palettes[legend_perm[idx]]
        grid[0, c0:c1 + 1] = np.array(_repeat_palette(legend_palette, width), dtype=int)

        control_offset = rng.randint(1, width - 2) if width >= 3 else 0
        grid[H - 1, c0 + control_offset] = int(palettes[idx][0])

    return grid


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    fused = main(grid)
    aligned = _solve_with_palettes(grid, _aligned_palette_picker)
    control_only = _solve_with_palettes(grid, _control_only_palette_picker)

    return {
        "diff_from_aligned": int(not np.array_equal(fused, aligned)),
        "diff_from_control_only": int(not np.array_equal(fused, control_only)),
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)
    n_regions = _choose_region_count(W, rng)
    if n_regions is None:
        return None

    free_cols = W - (n_regions - 1)
    widths = _partition_widths(free_cols, n_regions, MIN_REGION_W, rng)
    bg, wall, anchors, secondary_pool = _choose_colors(n_regions, rng)
    palettes = _make_palettes(widths, anchors, secondary_pool, rng)
    if palettes is None:
        return None

    legend_perm = _derangement(n_regions, rng)
    grid = _build_grid(H, W, bg, wall, widths, palettes, legend_perm, rng)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None

    metrics = _fusion_metrics(grid)
    if metrics["diff_from_aligned"] == 0:
        return None
    if metrics["diff_from_control_only"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
