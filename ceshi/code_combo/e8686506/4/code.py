# Combo variant of e8686506 (index 4)
# combo_spec: D4/1+D18/1
# Name: 双轴对称+同心环排列

# Design analysis
# fusion_spec:
#   concretization:
#     output:
#       - The assembled rectangle is always width 5 and has three concentric layers:
#         outer border, middle border, and center column.
#       - Unlike the original task, the finished rectangle is symmetric across both the
#         vertical and horizontal axes, so the same nested-ring layout appears in all
#         four quadrants around the center column.
#       - Every ring has a uniform color. The outer ring uses four bars, the middle
#         ring uses four shorter bars, and the center uses one vertical bar.
#     input:
#       - The input still consists of scattered monochrome puzzle fragments on a clean
#         background, but the recognizable grouped zigzag frame from the source task is
#         removed because the ring roles are now conveyed directly by straight bars.
#       - Each non-background component is a filled horizontal or vertical bar copied
#         from one ring role; no rotations are applied after extraction.
#     trace_example:
#       - Original e8686506 could place an LR-symmetric fragment in either the top half
#         or bottom half as long as left-right reflection and no-touch constraints held.
#       - In this combo, a sample with H=9 instead contains:
#         outer color -> bars of lengths 5,5,7,7
#         middle color -> bars of lengths 3,3,5,5
#         center color -> one bar of length 5
#       - The solver no longer searches over placements. It infers H from total area,
#         identifies the three ring roles from those bar signatures, and renders the only
#         dense rectangle that is both concentric and dual-axis symmetric.
#   complexity_calibration:
#     - Heights are restricted to 7 or 9. That keeps the output clearly portrait, gives
#       a visible center row for horizontal symmetry, and preserves small fragment sizes.
#     - Exactly nine pieces appear in every input. This repetition is enough for humans
#       to see the role families without turning the task into three giant obvious chunks.
#     - The center and middle side bars intentionally share length H-4; their roles stay
#       distinguishable because one color has one component while the other has four.
#   implicit_assumption_review:
#     - Original assumption: same-color pieces must not become 4-connected after assembly.
#       This is false now because each ring merges into one contiguous monochrome region.
#     - Original assumption: one or two 8-connected frame pieces already outline the final
#       rectangle in the input. This is also false; the combo uses fully scattered bars.
#     - Original assumption: corner singletons / edge markers are useful anchors. They are
#       dropped because they add noise without helping identify ring depth.
#     - New solver assumptions:
#       1. There are exactly three non-background colors.
#       2. Every component is a solid bar.
#       3. The three colors match unique role signatures: outer(4 bars), middle(4 bars),
#          center(1 vertical bar).
#       4. Total area factors to a unique closest-to-square portrait pair with W=5.
#     - The generator enforces each of these assumptions directly.
#   placement_legality:
#     - Output-first construction fixes every bar length before scattering, so all pieces
#       are in-bounds by construction.
#     - Input scattering uses `random_free_location(..., padding=1, padding_connectivity=8)`,
#       which guarantees no overlaps and no touching between distinct pieces.
#     - Canvas size stays within the source-task style range 13..15, and failed placements
#       are retried rather than forcing illegal layouts.
#   observability:
#     - From 2-3 examples a human can see that the output is always a nested-rectangle
#       pattern, not an arbitrary packed jigsaw.
#     - The added D4 constraint is visible because top and bottom rows match exactly and
#       the inner vertical bar is centered, so the output reads as four-quadrant symmetric
#       rather than merely left-right mirrored.
#     - Color ids never carry meaning. Role inference is supported entirely by visible
#       multiplicity and bar length patterns.
#   cleanliness:
#     - Only three structural layers are present and they never overlap ambiguously.
#       Inputs are sparse, outputs are compact, and each color owns one semantic depth.
#   post_edit_main_audit:
#     - The loop that groups components by color assumes every used color contributes at
#       least one bar; generator guarantees three non-background colors exactly.
#     - `_bar_info()` assumes each component is a filled 1xL or Lx1 rectangle; generator
#       creates patches only from those shapes.
#     - `_match_role()` assumes the outer and middle colors cannot swap because their bar
#       length multisets differ; with H in {7, 9} the multisets are always distinct.
#     - `main()` asserts odd H and W==5 so the dual symmetry has a visible central axis;
#       generator only emits total areas 35 and 45, which decode to 7x5 and 9x5.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_WIDTH = 5
_HEIGHT_CHOICES = [7, 9]
_HEIGHT_WEIGHTS = [45, 55]

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 400


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


def _is_dual_symmetric(grid: np.ndarray) -> bool:
    return np.array_equal(grid, grid[:, ::-1]) and np.array_equal(grid, grid[::-1, :])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered bar fragments into a dense rectangle of concentric rings with left-right and up-down symmetry.

    Input structure:
    - The dominant color is background; every non-background component is a solid horizontal or vertical bar.
    - Exactly three non-background colors appear, corresponding to the outer ring, the inner ring, and the centered core bar.

    Rule:
    - Collect all non-background 4-connected components and infer the output rectangle from their total area.
    - Determine which color plays each ring role from the visible bar multiset: the outer ring has two width-5 bars plus two side bars,
      the middle ring has two width-3 bars plus two shorter side bars, and the center is a single vertical bar.
    - Render the dense Hx5 rectangle as concentric rectangular layers so the result is symmetric across both axes.

    Key insight:
    - The fragments no longer describe an arbitrary jigsaw search space. Their counts and lengths already encode ring depth, and
      concentric rings automatically satisfy the stronger dual-axis symmetry once the three roles have been identified.
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

    assert len(comps_by_color) == 3, "combo variant expects exactly three ring colors"
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]
    assert W == _WIDTH, "combo variant uses a fixed width-5 rectangle"
    assert H in _HEIGHT_CHOICES and H % 2 == 1, "combo variant expects an odd height with a visible center row"

    role_to_color: dict[str, int] = {}
    for col, comps in comps_by_color.items():
        role = _match_role(comps, H, W)
        assert role is not None, "every color must match one concentric-ring role"
        assert role not in role_to_color, "ring roles must be uniquely identifiable"
        role_to_color[role] = int(col)

    assert set(role_to_color) == {"outer", "middle", "center"}, "all ring roles must be present"
    out = _render_rings(H, W, role_to_color).astype(int)
    assert _is_dual_symmetric(out), "rendered output must be dual-axis symmetric"
    return out


def _build_output(H: int) -> tuple[np.ndarray, list[int]]:
    colors = random.sample(list(range(10)), 3)
    role_to_color = {"outer": int(colors[0]), "middle": int(colors[1]), "center": int(colors[2])}
    out = _render_rings(H, _WIDTH, role_to_color)
    assert _is_dual_symmetric(out)
    return out, colors


def _ring_patches(H: int, colors: list[int], bg: int) -> list[np.ndarray]:
    outer, middle, center = map(int, colors)
    specs = [
        (outer, 1, _WIDTH),
        (outer, 1, _WIDTH),
        (outer, H - 2, 1),
        (outer, H - 2, 1),
        (middle, 1, _WIDTH - 2),
        (middle, 1, _WIDTH - 2),
        (middle, H - 4, 1),
        (middle, H - 4, 1),
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
    raise RuntimeError("failed to generate a valid e8686506 combo variant")
