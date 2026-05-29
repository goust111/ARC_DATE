# Variant of e8686506
# Dimension: D5 — 同色不相邻约束 (meta)
# Change: 拼合后同色碎片不可4-连通相邻（相同颜色的像素之间不能共享边）... → 同色必须相邻（聚类约束）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D5 inversion
# -----------------------------------------------------------------------------
# a) Concrete behavior / trace
#    Original task used LR symmetry + "same colors from different pieces must never
#    4-touch" to force a checker-like placement. That means a color could appear in
#    several separated output regions. This variant flips the key relation: input
#    still contains several disconnected fragments of the same color, but after
#    assembly those fragments must fuse into one 4-connected color cluster.
#    To make that visible, the output is a 10x5 symmetric rectangle whose single
#    frame color forms a connected border+bars scaffold, partitioning the interior
#    into three stacked chambers of heights 1/2/3 in random order. Each chamber is
#    monochrome in the output, but its input representation is cut into several
#    same-color fragments and scattered. Example trace:
#      - old logic: three red fragments are forbidden from touching, so they land in
#        separated slots.
#      - new logic: the same three red fragments contribute total area 6, matching
#        the unique 2x3 chamber, so the solver merges them into one red block.
# b) Complexity calibration
#    The frame is still needed; otherwise pure area matching would be too weak.
#    Three chambers give four total colors, enough repetition for humans to notice
#    that same-color fragments attract each other. Chamber areas are exactly 3, 6,
#    and 9, so color->chamber matching is unique without relying on color ids.
# c) Implicit-assumption audit against the original main()
#    - Original `can_place(... disallow_touch ...)` depended on the old D5 and is
#      invalid here; it would reject the desired clustered output.
#    - Original `_build_output/_fill_holes` deliberately avoided reusing a color for
#      adjacent chunks. That also conflicts with the new D5 and must be replaced by
#      one-color-per-chamber generation.
#    - The old frame finder remains valid in spirit, but the frame shape is adapted
#      to a single 4-connected scaffold so the frame color itself also respects the
#      clustering rule.
# d) Placement legality
#    Generator scatters only non-frame fragments, always outside the reserved frame
#    bbox, with Chebyshev padding 1, so pieces never overlap or touch. The frame is
#    pasted back as one fixed grouped component.
# e) Visual observability
#    Across examples, humans see the same motif: a symmetric scaffold already in
#    place, plus several disconnected same-color shards elsewhere. In the output,
#    each color consolidates into exactly one chamber. The changed behavior is
#    visible even without counting precise coordinates.
# f) Visual cleanliness
#    Only two structural layers exist: one connected scaffold and three chamber
#    colors. No overlapping markers or ambiguous decorative anchors are used.
# g) Color inference
#    Colors are only permuted globally; no rule depends on numeric ids. The only
#    matching cue is visible structure: the total size of a color's scattered shards
#    matches one chamber's size, and the output always clusters that color there.


_OUT_H = 10
_OUT_W = 5
_CANVAS_MIN = 14
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 200


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _find_frame_component(g: np.ndarray, H: int, W: int, bg: int) -> tuple[int, np.ndarray]:
    candidates: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            r0, c0, r1, c1 = bbox_coords(pts)
            if (r1 - r0 + 1, c1 - c0 + 1) != (H, W):
                continue
            bmp = np.zeros((H, W), dtype=bool)
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
            if np.array_equal(bmp, bmp[:, ::-1]):
                candidates.append((int(col), bmp))
    assert len(candidates) == 1, "expected exactly one LR-symmetric frame component"
    return candidates[0]


def _area_to_color(g: np.ndarray, bg: int, frame_color: int) -> dict[int, int]:
    area_map: dict[int, int] = {}
    for col in sorted(set(map(int, np.unique(g))) - {bg, frame_color}):
        comps = components_mask(g == col, connectivity=4)
        assert len(comps) >= 2, "each non-frame color must arrive as multiple fragments"
        area = sum(len(pts) for pts in comps)
        assert area not in area_map, "non-frame colors must have unique total areas"
        area_map[area] = int(col)
    return area_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered same-color shards into one left-right symmetric chambered rectangle.

    Input structure:
    - The most common color is background.
    - One non-background color already forms a symmetric scaffold whose bbox is the whole output.
    - Every other color appears as several disconnected fragments scattered away from that scaffold.

    Rule:
    - Infer the output rectangle dimensions from total non-background area.
    - Keep the scaffold in its normalized output position.
    - The scaffold partitions the rectangle into three connected chambers; merge all fragments of a
      given color into the unique chamber whose area matches that color's total shard area.

    Key insight:
    - The old exclusion rule is gone: repeated colors are supposed to coalesce.
    - A connected symmetric scaffold fixes the chamber geometry, and the three chamber areas are
      distinct, so each repeated color is identified by visible area and then clustered contiguously.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    total_area = int(np.sum(g != bg))
    assert total_area > 0, "grid must contain non-background pieces"
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_color, frame_mask = _find_frame_component(g, H, W, bg)
    holes = components_mask(~frame_mask, connectivity=4)
    hole_areas = sorted(len(h) for h in holes)
    assert hole_areas == [3, 6, 9], "frame must define the three canonical chambers"

    area_map = _area_to_color(g, bg, frame_color)
    assert sorted(area_map) == [3, 6, 9], "non-frame colors must match chamber areas 3/6/9"

    out = np.full((H, W), frame_color, dtype=int)
    for hole in holes:
        area = len(hole)
        col = area_map[area]
        for r, c in hole:
            out[int(r), int(c)] = col

    for col in sorted(area_map.values()):
        assert len(components_mask(out == col, connectivity=4)) == 1, "each output color must cluster"
    assert len(components_mask(out == frame_color, connectivity=4)) == 1, "frame color must stay connected"
    assert np.array_equal(out, out[:, ::-1]), "assembled board must be LR-symmetric"
    return out.astype(int)


def _chamber_layout() -> tuple[np.ndarray, list[dict[str, int]]]:
    frame_mask = np.zeros((_OUT_H, _OUT_W), dtype=bool)
    frame_mask[0, :] = True
    frame_mask[-1, :] = True
    frame_mask[:, 0] = True
    frame_mask[:, -1] = True

    heights = random.sample([1, 2, 3], k=3)
    chambers: list[dict[str, int]] = []
    r = 1
    for i, h in enumerate(heights):
        chambers.append({"top": r, "height": h, "area": 3 * h})
        r += h
        if i < 2:
            frame_mask[r, :] = True
            r += 1
    assert r == _OUT_H - 1
    return frame_mask, chambers


def _local_piece_templates(height: int) -> list[list[tuple[int, int]]]:
    if height == 1:
        return [[(0, 0)], [(0, 1)], [(0, 2)]]
    if height == 2:
        templates = [
            [[(0, 0), (0, 1)], [(0, 2), (1, 2)], [(1, 0), (1, 1)]],
            [[(0, 0), (1, 0)], [(0, 1), (0, 2)], [(1, 1), (1, 2)]],
        ]
        return random.choice(templates)
    if height == 3:
        templates = [
            [[(0, 0), (0, 1)], [(0, 2), (1, 2)], [(1, 0), (2, 0)], [(1, 1), (2, 1), (2, 2)]],
            [[(0, 0), (1, 0)], [(0, 1), (0, 2)], [(1, 1), (1, 2), (2, 2)], [(2, 0), (2, 1)]],
        ]
        return random.choice(templates)
    raise ValueError("unsupported chamber height")


def _patch_from_local(coords: list[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = color
    return patch


def _build_output() -> tuple[np.ndarray, int, list[dict[str, int]]]:
    frame_mask, chambers = _chamber_layout()
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    frame_color = palette[0]
    chamber_colors = palette[1 : 1 + len(chambers)]

    out = np.full((_OUT_H, _OUT_W), frame_color, dtype=int)
    for chamber, col in zip(chambers, chamber_colors):
        chamber["color"] = int(col)
        top = chamber["top"]
        height = chamber["height"]
        out[top : top + height, 1:4] = int(col)
    return out, bg, chambers


def _build_frame_patch(out: np.ndarray, frame_color: int, bg: int) -> np.ndarray:
    patch = np.full_like(out, bg)
    patch[out == frame_color] = frame_color
    return patch


def _fragment_patches(chambers: list[dict[str, int]], bg: int) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for chamber in chambers:
        col = int(chamber["color"])
        for coords in _local_piece_templates(int(chamber["height"])):
            patches.append(_patch_from_local(coords, col, bg))
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, chambers: list[dict[str, int]]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape
    frame_color = int(out[0, 0])
    frame_patch = _build_frame_patch(out, frame_color, bg)

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("frame bbox does not fit the scatter canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)
    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _fragment_patches(chambers, bg):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    paste(canvas, frame_patch, off_r, off_c, transparent=bg)
    return canvas


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg_in = get_bg(inp)
    bg_out = get_bg(out)
    frame_color = int(out[0, 0])
    in_multi = 0
    out_cluster = 0
    for col in sorted(set(map(int, np.unique(out))) - {bg_out, frame_color}):
        in_comps = len(components_mask(inp == col, connectivity=4))
        out_comps = len(components_mask(out == col, connectivity=4))
        if in_comps >= 2:
            in_multi += 1
        if out_comps == 1:
            out_cluster += 1
    return {
        "input_multi_colors": in_multi,
        "output_cluster_colors": out_cluster,
        "input_non_bg_area": int(np.sum(inp != bg_in)),
    }


def _gen_attempt() -> np.ndarray | None:
    out, bg, chambers = _build_output()
    try:
        inp = _scatter(out, bg, chambers)
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None

    metrics = _variant_metrics(inp, out)
    if metrics["input_multi_colors"] != 3 or metrics["output_cluster_colors"] != 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D5 variant input for e8686506")
