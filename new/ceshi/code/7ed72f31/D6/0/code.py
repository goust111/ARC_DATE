# Variant of 7ed72f31
# Dimension: D6 — 网格形状与输出裁剪 (grid_shape)
# Change: 输出与输入相同尺寸(18-28 x 18-28)，在原网格上直接叠加镜像结果... → 紧凑裁剪输出

from __future__ import annotations

import numpy as np
from arc_utils import get_bg, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# variant_spec:
#   core_change:
#     original: solver reflects sprites onto a same-size canvas and returns it.
#     variant: keep the same reflection logic, then crop the final canvas to the
#       minimal bounding rectangle containing all non-background pixels.
#   reference_baseline:
#     source_task bf45cf4b also changes only the output shape; its solver keeps
#     the object logic intact and derives a variable-size output from a full-size
#     intermediate. We use the same complexity level here: one extra crop step.
#
# behavior_trace:
#   small_example:
#     input_full_grid:
#       a horizontal red axis with a payload below it is placed well inside a
#       24x24 background canvas.
#     original_solver:
#       mirrors the payload above the red row and returns the whole 24x24 grid,
#       leaving a thick background border around all active pixels.
#     variant_solver:
#       produces the same mirrored 24x24 intermediate, then slices away the
#       outer background border and returns only the bbox spanning the reflected
#       structure(s). The reflection rule is unchanged; only the reported canvas
#       size changes.
#   difference_point:
#     the change happens after all mirroring/point-reflection is finished.
#     input construction still needs large background margins so the crop is
#     visible; otherwise the variant would visually degenerate to the original.
#
# complexity_calibration:
#   preserved_dimensions:
#     marker semantics, reflection geometry, colors, and component logic stay
#     unchanged. We do not add new marker types or color remappings.
#   controlled_freedom:
#     generator uses only the three generator-backed original cases:
#       - horizontal red line + payload
#       - vertical red line + payload
#       - isolated red pixel + adjacent 8-connected sprite
#     This keeps the task readable while still showing multiple repeated examples
#     of the unchanged transform inside the new tight-crop output format.
#
# implicit_assumption_review:
#   - The original solver writes onto a copy of the input and assumes returning
#     the whole canvas is acceptable. In the variant this is no longer true,
#     so a final crop is mandatory.
#   - Background must be taken from the input/full canvas, not recomputed after
#     cropping, because in the cropped output bg may stop being the most common
#     color.
#   - The original eligibility scan can accidentally reflect unrelated same-color
#     pixels if they sit in the same bbox columns/rows. The generator therefore
#     assigns distinct payload colors per active group and avoids noisy extras.
#   - Tight crop assumes there is at least one non-background pixel. The
#     generator always creates 3+ groups, so this holds.
#
# placement_legality:
#   every generated group is built as an output patch first, then its mirrored
#   half is removed to obtain the input patch. Patches are placed with at least
#   2 cells of outer border and 1 cell of padding between groups, preventing
#   overlap, edge clipping, and ambiguous touching.
#
# visual_observability:
#   the decisive D6 cue is that the output no longer includes the large empty
#   border seen in the input. Generator placements force all active pixels to
#   lie strictly inside the input border so output.shape is always smaller than
#   input.shape. With 2-4 examples, the repeated disappearance of unused border
#   area is directly visible.
#
# cleanliness:
#   no free distractor sprites are added. This keeps each red marker/payload
#   relationship legible and lets the crop effect stay obvious.
# ---------------------------------------------------------------------------

MARK = 2


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _solve_full(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    full = _solve_full(g, bg)
    return _tight_crop_non_bg(full, bg)


def _neighbors(coords: set[tuple[int, int]], connectivity: int) -> list[tuple[int, int]]:
    deltas4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    deltas8 = deltas4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    deltas = deltas4 if int(connectivity) == 4 else deltas8
    frontier: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in deltas:
            frontier.add((int(r + dr), int(c + dc)))
    frontier.difference_update(coords)
    return sorted(frontier)


def _random_connected_shape(
    area: int,
    *,
    connectivity: int,
    max_h: int,
    max_w: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    for _ in range(80):
        start = (int(rng.integers(max_h)), int(rng.integers(max_w)))
        cells = {start}
        while len(cells) < int(area):
            frontier = []
            for rr, cc in _neighbors(cells, connectivity):
                if 0 <= rr < int(max_h) and 0 <= cc < int(max_w):
                    frontier.append((int(rr), int(cc)))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)
        if len(cells) != int(area):
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        norm = sorted((int(r - min(rs)), int(c - min(cs))) for r, c in cells)
        if len(norm) >= 2:
            return norm
        if int(area) == 1:
            return norm
    raise ValueError("shape_generation_failed")


def _build_patch(
    original: list[tuple[int, int]],
    mirrored: list[tuple[int, int]],
    markers: list[tuple[int, int]],
    *,
    color: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_coords = original + mirrored + markers
    rs = [r for r, _ in all_coords]
    cs = [c for _, c in all_coords]
    rmin = min(rs)
    cmin = min(cs)
    h = max(rs) - rmin + 1
    w = max(cs) - cmin + 1
    out_patch = np.full((int(h), int(w)), int(bg), dtype=int)
    inp_patch = np.full((int(h), int(w)), int(bg), dtype=int)

    for r, c in original:
        out_patch[int(r - rmin), int(c - cmin)] = int(color)
        inp_patch[int(r - rmin), int(c - cmin)] = int(color)
    for r, c in mirrored:
        out_patch[int(r - rmin), int(c - cmin)] = int(color)
    for r, c in markers:
        out_patch[int(r - rmin), int(c - cmin)] = int(MARK)
        inp_patch[int(r - rmin), int(c - cmin)] = int(MARK)
    return inp_patch, out_patch


def _make_row_group(color: int, bg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    line_len = int(rng.integers(2, 7))
    area = int(rng.integers(3, 12))
    shape = _random_connected_shape(area, connectivity=4, max_h=min(4, area), max_w=line_len, rng=rng)
    width = max(c for _r, c in shape) + 1
    col_shift = int(rng.integers(0, line_len - width + 1))
    below = [(int(1 + r), int(col_shift + c)) for r, c in shape]
    above = [(int(-1 - r), int(col_shift + c)) for r, c in shape]
    if bool(rng.integers(0, 2)):
        original, mirrored = below, above
    else:
        original, mirrored = above, below
    markers = [(0, c) for c in range(line_len)]
    return _build_patch(original, mirrored, markers, color=color, bg=bg)


def _make_col_group(color: int, bg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    line_len = int(rng.integers(2, 7))
    area = int(rng.integers(3, 12))
    shape = _random_connected_shape(area, connectivity=4, max_h=line_len, max_w=min(4, area), rng=rng)
    height = max(r for r, _c in shape) + 1
    row_shift = int(rng.integers(0, line_len - height + 1))
    right = [(int(row_shift + r), int(1 + c)) for r, c in shape]
    left = [(int(row_shift + r), int(-1 - c)) for r, c in shape]
    if bool(rng.integers(0, 2)):
        original, mirrored = right, left
    else:
        original, mirrored = left, right
    markers = [(r, 0) for r in range(line_len)]
    return _build_patch(original, mirrored, markers, color=color, bg=bg)


def _make_isolated_group(color: int, bg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    area = int(rng.integers(1, 9))
    shape = _random_connected_shape(area, connectivity=8, max_h=min(4, area), max_w=min(4, area), rng=rng)
    sr, sc = [(1, 1), (1, -1), (-1, 1), (-1, -1)][int(rng.integers(4))]
    original = [(int(sr * (1 + r)), int(sc * (1 + c))) for r, c in shape]
    mirrored = [(int(-r), int(-c)) for r, c in original]
    markers = [(0, 0)]
    return _build_patch(original, mirrored, markers, color=color, bg=bg)


def _generate_group(group_type: str, color: int, bg: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if group_type == "row":
        return _make_row_group(color, bg, rng)
    if group_type == "col":
        return _make_col_group(color, bg, rng)
    if group_type == "iso":
        return _make_isolated_group(color, bg, rng)
    raise ValueError(f"unknown group_type: {group_type}")


def _expected_from_full_output(full_output: np.ndarray, bg: int) -> np.ndarray:
    return _tight_crop_non_bg(full_output, bg)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray:
    H = int(rng.integers(24, 29))
    W = int(rng.integers(24, 29))
    palette = [int(x) for x in rng.permutation(np.arange(10, dtype=int))]
    bg = next(c for c in palette if c != MARK)
    color_pool = [c for c in palette if c not in {bg, MARK}]

    n_groups = int(rng.integers(3, 7))
    base_types = ["row", "col", "iso"]
    group_types = base_types[:]
    while len(group_types) < n_groups:
        group_types.append(base_types[int(rng.integers(len(base_types)))])
    rng.shuffle(group_types)

    if len(color_pool) < len(group_types):
        raise ValueError("not enough colors")

    inp = np.full((H, W), int(bg), dtype=int)
    full_out = np.full((H, W), int(bg), dtype=int)

    for idx, group_type in enumerate(group_types):
        color = int(color_pool[idx])
        inp_patch, out_patch = _generate_group(group_type, color, bg, rng)
        loc = random_free_location(inp, out_patch, transparent=bg, border=2, padding=1, padding_connectivity=8)
        paste(inp, inp_patch, int(loc[0]), int(loc[1]), transparent=bg)
        paste(full_out, out_patch, int(loc[0]), int(loc[1]), transparent=bg)

    fg = np.argwhere(full_out != int(bg))
    if fg.size == 0:
        raise ValueError("empty_output")
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    if not (int(r0) >= 2 and int(c0) >= 2 and int(r1) <= H - 3 and int(c1) <= W - 3):
        raise ValueError("crop_not_visible")

    expected = _expected_from_full_output(full_out, bg)
    got = main(inp)
    if not np.array_equal(got, expected):
        raise AssertionError("self_check_failed")
    if got.shape == inp.shape:
        raise ValueError("shape_not_cropped")
    return inp


def generate_input() -> np.ndarray:
    for _ in range(200):
        rng = np.random.default_rng()
        try:
            return _gen_attempt(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid D6 sample")
