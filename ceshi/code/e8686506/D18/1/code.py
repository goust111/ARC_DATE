# Variant of e8686506
# Dimension: D18 — 拼合矩形的内部结构模式 (meta)
# Change: 拼合后大碎片（框架）在外围、小碎片填入凹陷和边角处，形成分层嵌套的矩形... → 同心环状排列

# Design analysis
# variant_spec:
#   concretization:
#     output is still a dense portrait rectangle assembled from scattered monochrome pieces,
#     but its interior is no longer an irregular frame-with-holes pattern. The assembled
#     rectangle is width 5 and has three visible concentric layers:
#       - outer ring: full border of the rectangle
#       - middle ring: border of the inner (H-2) x 3 sub-rectangle
#       - center: the remaining central vertical bar
#     every layer has a single color, so the output reads as nested rectangular frames.
#   old_vs_new_trace:
#     original flow filled arbitrary hole regions inside a zigzag outer frame, and the
#     solver relied on "same-color pieces never 4-touch" to keep component identities.
#     in the variant, outer/middle/center fragments must merge into three contiguous
#     same-color rings after assembly. For example with H=8, the old generator could place
#     color-A fragments in disconnected cavities; the new generator instead emits:
#       outer: 2 horizontal bars of length 5 + 2 vertical bars of length 6
#       middle: 2 horizontal bars of length 3 + 2 vertical bars of length 4
#       center: 1 vertical bar of length 4
#     the old disallow-touch rule would reject this because pieces of the same color must
#     meet edge-to-edge in the final board. Therefore the solver must switch from generic
#     propagation packing to role recognition followed by direct ring rendering.
#   complexity_calibration:
#     the variant keeps exactly three semantic layers and nine pieces total, which is enough
#     repetition for humans to spot the rule while avoiding a degenerate "three huge pieces"
#     simplification. Heights are limited to 6..8 so every bar remains small and visually
#     fragment-like.
#   implicit_assumption_audit:
#     original main assumptions that no longer hold:
#       - per-color pieces may not 4-touch after assembly: false in the variant
#       - frame is detected as 1-2 8-connected components whose union already outlines the
#         rectangle: false, because each ring is deliberately split into several bars
#       - optional singleton/bar anchors can be interpreted as corners/edges independent of
#         the interior pattern: no longer needed; the piece-length signature now determines
#         each ring role directly
#     new assumptions introduced:
#       - exactly three non-background colors exist
#       - each component is a straight filled bar with no rotation applied
#       - color roles are uniquely identified by component counts and bar lengths
#     generate_input enforces all three.
#   placement_legality:
#     the generator scatters the nine patches on a 13..15 square canvas with Chebyshev
#     padding 1, so pieces never overlap or touch in input. Since patches are copied
#     directly from the intended output decomposition, the solver's bar-shape assumptions
#     remain valid.
#   observability:
#     across samples, humans can see repeated 5-cell outer bars, shorter 3-cell middle bars,
#     and a single center column piece. The output consistently turns those into nested
#     rectangular rings, so the D18 change is visible without using hidden color ordering.
#   cleanliness:
#     the input contains only nine separated bars on a clean background, and the output has
#     three large uniform layers. No markers overlap, and each layer is easy to parse.
#   color_inference:
#     color ids are never sorted by numeric meaning. Roles are inferred from visible shape
#     signatures: longest bars define the outer ring, shorter bars define the middle ring,
#     and the lone vertical bar defines the center.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _bar_info(pts: np.ndarray) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(len(pts))
    if h == 1 and area == w:
        return "h", w
    if w == 1 and area == h:
        return "v", h
    raise AssertionError("every piece must be a straight filled bar")


def _match_role(comps: list[np.ndarray], H: int, W: int) -> str | None:
    bars = [_bar_info(pts) for pts in comps]
    hs = sorted(length for orient, length in bars if orient == "h")
    vs = sorted(length for orient, length in bars if orient == "v")

    if len(comps) == 4 and hs == [W, W] and vs == [H - 2, H - 2]:
        return "outer"
    if len(comps) == 4 and hs == [W - 2, W - 2] and vs == [H - 4, H - 4]:
        return "middle"
    if len(comps) == 1 and bars[0] == ("v", H - 4):
        return "center"
    return None


def _render_rings(H: int, W: int, colors: dict[str, int]) -> np.ndarray:
    out = np.full((H, W), int(colors["center"]), dtype=int)
    for r in range(H):
        for c in range(W):
            ring = min(r, c, H - 1 - r, W - 1 - c)
            if ring == 0:
                out[r, c] = int(colors["outer"])
            elif ring == 1:
                out[r, c] = int(colors["middle"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered bar fragments into a portrait rectangle of concentric color rings.

    Input structure:
    - The background is the most common color; every non-background component is a solid
      horizontal or vertical bar.
    - Exactly three non-background colors appear, representing the outer ring, the middle
      ring, and the central vertical core of the final rectangle.

    Rule:
    - Collect all non-background 4-connected components and compute the output rectangle
      size from their total area.
    - Identify which color belongs to which ring by the visible bar-length pattern:
      longest pair of width-5 bars plus two side bars for the outer ring, shorter width-3
      bars plus two shorter side bars for the middle ring, and one lone vertical bar for
      the center.
    - Paint the dense Hx5 rectangle as nested rectangular rings using those three colors.

    Key insight: the fragments no longer encode an arbitrary jigsaw packing. Their counts
    and bar lengths already specify the three ring depths, so once H is known the final
    concentric layout is determined directly.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[np.ndarray]] = {}
    total_area = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        assert comps, "each non-background color must contribute at least one piece"
        comps_by_color[int(col)] = comps
        total_area += sum(len(pts) for pts in comps)

    assert len(comps_by_color) == 3, "variant expects exactly three ring colors"
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]
    assert W == 5, "variant uses a fixed width-5 concentric-ring rectangle"
    assert H >= 6, "variant generator only emits heights large enough for three visible layers"

    role_to_color: dict[str, int] = {}
    for col, comps in comps_by_color.items():
        role = _match_role(comps, H, W)
        assert role is not None, "every color must match one concentric-ring role"
        assert role not in role_to_color, "ring roles must be uniquely identifiable"
        role_to_color[role] = int(col)

    assert set(role_to_color) == {"outer", "middle", "center"}, "all three ring roles must be present"
    return _render_rings(H, W, role_to_color).astype(int)


# --------------- generator (D18 variant) ---------------

_WIDTH = 5
_HEIGHT_CHOICES = [6, 7, 8]
_HEIGHT_WEIGHTS = [30, 35, 40]

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300


def _build_output(H: int) -> tuple[np.ndarray, list[int]]:
    W = _WIDTH
    colors = random.sample(list(range(10)), 3)
    role_to_color = {"outer": colors[0], "middle": colors[1], "center": colors[2]}
    return _render_rings(H, W, role_to_color), colors


def _ring_patches(H: int, colors: list[int], bg: int) -> list[np.ndarray]:
    outer, middle, center = map(int, colors)
    side_outer = H - 2
    side_middle = H - 4

    specs = [
        (outer, 1, _WIDTH),
        (outer, 1, _WIDTH),
        (outer, side_outer, 1),
        (outer, side_outer, 1),
        (middle, 1, _WIDTH - 2),
        (middle, 1, _WIDTH - 2),
        (middle, side_middle, 1),
        (middle, side_middle, 1),
        (center, H - 4, 1),
    ]

    patches = []
    for col, h, w in specs:
        patch = np.full((h, w), bg, dtype=int)
        patch[:, :] = int(col)
        patches.append(patch)
    random.shuffle(patches)
    return patches


def _scatter(patches: list[np.ndarray], bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)
    return canvas


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.choices(_HEIGHT_CHOICES, weights=_HEIGHT_WEIGHTS, k=1)[0]
    out, colors = _build_output(H)
    bg_candidates = [c for c in range(10) if c not in set(colors)]
    bg = random.choice(bg_candidates)
    patches = _ring_patches(H, colors, bg)
    try:
        inp = _scatter(patches, bg)
    except ValueError:
        return None
    got = main(inp)
    if np.array_equal(got, out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("failed to generate a valid e8686506 D18 variant input")
