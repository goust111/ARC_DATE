# Combo variant of e8686506 (index 44)
# combo_spec: D10/1+D18/1
# Name: 矩形环框架+同心环内部结构
#
# Design analysis
# ---------------
# concretization_and_trace:
#   - The fusion keeps the original "assemble scattered pieces into a portrait rectangle"
#     shell, but the visible structure changes in two coupled ways.
#   - D10/1 contributes a single preassembled outer frame: one non-background component is
#     already a complete rectangular ring with bbox Hx5. It stays fixed in the final answer.
#   - D18/1 contributes the cavity semantics: the ring interior is not filled arbitrarily.
#     It must become concentric layers, so with width 5 the cavity always resolves into:
#       1. a middle rectangular ring of color M
#       2. a central vertical bar of color C
#   - Concrete example for H=8:
#       original task: detect an irregular outline, then propagate many small pieces into a
#       dense 8x5 rectangle while forbidding same-color merges.
#       fused task: detect a full 8x5 border ring first; then recognize that the remaining
#       pieces are exactly two 1x3 bars, two 4x1 bars, and one 4x1 bar. The four pieces of
#       one color must merge into the inner ring, while the lone bar becomes the center.
#   - This changes the reasoning direction from "generic jigsaw packing inside a symmetric
#     box" to "fixed ring outside, role-typed concentric completion inside".
#
# complexity_calibration:
#   - A single outer ring plus only one interior block would be too weak visually and too
#     easy. Width 5 is retained specifically because it gives three visible nested layers:
#     outer ring, middle ring, center.
#   - Heights are limited to 7..10. That keeps the hole obvious, makes the middle side bars
#     length 3..6, and preserves the original factorization-based dimension recovery.
#   - The generator always emits six pieces total: one outer ring, four middle-ring bars,
#     and one center bar. This is small enough to be legible, but the required same-color
#     merging inside the cavity still breaks the original solver assumptions.
#
# implicit_assumptions_review:
#   - Original assumption "same-color pieces may not 4-touch in the output" is false here:
#     the four middle-ring pieces must connect into one ring.
#   - Original frame detection by "1-2 symmetric outline pieces" is too weak: the fused
#     variant requires one exact full perimeter ring, not just any symmetric contour.
#   - Original generic packing over the whole HxW board is also invalid: once the outer ring
#     is found, every remaining piece has a semantic role inside the ring cavity.
#   - New solver assumptions introduced here:
#       * exactly one component is a full Hx5 rectangular ring
#       * the remaining colors are identified by bar-shape signatures, not color ids
#       * the middle ring color has four bars that merge in the output
#     The generator constructs inputs that satisfy all three.
#
# placement_legality:
#   - The generator reserves the full ring bbox before scattering the inner pieces, so no
#     piece overlaps the ring or leaks into its cavity.
#   - All scattered patches are placed with Chebyshev padding 1, preserving clean object
#     separation in the input.
#   - The ring is pasted back last with its interior left as background, so the input always
#     shows a literal empty rectangular frame plus detached interior fragments.
#
# visual_observability_and_cleanliness:
#   - Across examples, humans can immediately spot the D10 change because the largest object
#     is a rectangle-with-a-hole rather than a jagged outline.
#   - The D18 change is visible because the non-ring fragments repeat a stable vocabulary:
#     two short horizontal bars, two matching vertical side bars, and one central bar.
#   - No hidden ordering by numeric color id is used. Roles are inferred only from visible
#     shape/count structure, and the output reads as clean nested color bands.
#
# inferability_contract:
#   - The crucial non-degenerate signal is that one inner color has four disconnected input
#     components but becomes one connected ring in the output. This cannot be explained by
#     the seed task's "same-color pieces stay separate" rule, so the combo behavior is
#     structurally observable from examples rather than only oracle-checkable.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_WIDTH = 5
_HEIGHT_CHOICES = [7, 8, 9, 10]
_HEIGHT_WEIGHTS = [35, 40, 30, 25]

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


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _ring_coords(H: int, W: int) -> tuple[tuple[int, int], ...]:
    cells = []
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                cells.append((r, c))
    return tuple(cells)


def _is_full_ring(coords: tuple[tuple[int, int], ...], H: int, W: int) -> bool:
    return len(coords) == (2 * H + 2 * W - 4) and coords == _ring_coords(H, W)


def _bar_info(pts: np.ndarray) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(len(pts))
    if h == 1 and area == w:
        return "h", w
    if w == 1 and area == h:
        return "v", h
    raise AssertionError("every non-ring piece must be a straight filled bar")


def _match_inner_role(comps: list[np.ndarray], H: int, W: int) -> str | None:
    bars = [_bar_info(pts) for pts in comps]
    hs = sorted(length for orient, length in bars if orient == "h")
    vs = sorted(length for orient, length in bars if orient == "v")

    if len(comps) == 4 and hs == [W - 2, W - 2] and vs == [H - 4, H - 4]:
        return "middle"
    if len(comps) == 1 and bars[0] == ("v", H - 4):
        return "center"
    return None


def _render_concentric(H: int, W: int, colors: dict[str, int]) -> np.ndarray:
    out = np.full((H, W), int(colors["center"]), dtype=int)
    for r in range(H):
        for c in range(W):
            ring = min(r, c, H - 1 - r, W - 1 - c)
            if ring == 0:
                out[r, c] = int(colors["outer"])
            elif ring == 1:
                out[r, c] = int(colors["middle"])
    return out


def _identify_roles(comps_by_color: dict[int, list[np.ndarray]], H: int, W: int) -> dict[str, int]:
    role_to_color: dict[str, int] = {}
    ring_owner: int | None = None

    for col, comps in comps_by_color.items():
        for pts in comps:
            h, w, coords = _piece_sig(pts)
            if h == H and w == W and _is_full_ring(coords, H, W):
                assert ring_owner is None, "outer ring must be unique"
                ring_owner = int(col)
    assert ring_owner is not None, "input must contain one full rectangular ring"
    assert len(comps_by_color[ring_owner]) == 1, "outer ring color must appear only in the ring"
    role_to_color["outer"] = int(ring_owner)

    for col, comps in comps_by_color.items():
        if int(col) == ring_owner:
            continue
        role = _match_inner_role(comps, H, W)
        assert role is not None, "every non-ring color must match an inner concentric role"
        assert role not in role_to_color, "inner roles must be uniquely identifiable"
        role_to_color[role] = int(col)

    assert set(role_to_color) == {"outer", "middle", "center"}, "all concentric roles must be present"
    return role_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a left-right symmetric rectangle whose outer border is already a complete ring and whose cavity resolves into concentric layers.

    Input structure:
    - The background is the most common color; one non-background component is a closed Hx5 rectangular ring.
    - Every other non-background component is a detached straight bar that belongs inside that ring's cavity.

    Rule:
    - Recover the output dimensions from total non-background area, then locate the unique full ring with that bbox.
    - Treat the ring color as the outer layer, identify the remaining colors from their bar-count/length signatures, and assign them to the middle ring and center column.
    - Render the final dense rectangle as nested rectangular layers with exact left-right symmetry.

    Key insight:
    - The outer frame is no longer something to extend, and the inner pieces are no longer arbitrary fillers: together they encode a fixed three-layer concentric pattern, so role recognition replaces generic jigsaw packing.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps_by_color: dict[int, list[np.ndarray]] = {}
    total_area = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        assert comps, "each non-background color must contribute at least one component"
        comps_by_color[int(col)] = comps
        total_area += sum(len(pts) for pts in comps)

    assert total_area > 0, "grid must contain non-background pieces"
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]
    assert W == _WIDTH, "combo variant uses a fixed width-5 concentric layout"
    assert H >= 7, "height must leave a visible ring cavity and center bar"
    assert len(comps_by_color) == 3, "combo variant expects exactly three non-background colors"

    role_to_color = _identify_roles(comps_by_color, H, W)
    return _render_concentric(H, W, role_to_color).astype(int)


def _build_output(H: int) -> tuple[np.ndarray, dict[str, int], int]:
    palette = random.sample(list(range(10)), 4)
    bg = int(palette[0])
    role_to_color = {
        "outer": int(palette[1]),
        "middle": int(palette[2]),
        "center": int(palette[3]),
    }
    return _render_concentric(H, _WIDTH, role_to_color), role_to_color, bg


def _ring_patch(H: int, W: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((H, W), bg, dtype=int)
    for r, c in _ring_coords(H, W):
        patch[r, c] = int(color)
    return patch


def _inner_patches(H: int, colors: dict[str, int], bg: int) -> list[np.ndarray]:
    middle = int(colors["middle"])
    center = int(colors["center"])
    patches = []

    specs = [
        (middle, 1, _WIDTH - 2),
        (middle, 1, _WIDTH - 2),
        (middle, H - 4, 1),
        (middle, H - 4, 1),
        (center, H - 4, 1),
    ]
    for col, h, w in specs:
        patch = np.full((h, w), bg, dtype=int)
        patch[:, :] = int(col)
        patches.append(patch)
    random.shuffle(patches)
    return patches


def _scatter(out: np.ndarray, colors: dict[str, int], bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("output bbox does not fit in the canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _inner_patches(oh, colors, bg):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = bg
    paste(canvas, _ring_patch(oh, ow, colors["outer"], bg), off_r, off_c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    H = random.choices(_HEIGHT_CHOICES, weights=_HEIGHT_WEIGHTS, k=1)[0]
    out, colors, bg = _build_output(H)
    try:
        inp = _scatter(out, colors, bg)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
