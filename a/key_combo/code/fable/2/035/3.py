from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, scale_sprite


TEMPLATE_SIZE = 3
NEGATIVE_FRAME = np.ones((3, 3), dtype=bool)
NEGATIVE_FRAME[1, 1] = False

GEN_RETRIES = 500
GRID_SIDE_MIN = 22
GRID_SIDE_MAX = 31
RESERVED_CORNER = 4
VISIBLE_PADDING = 1
AVAILABLE_COLORS = [c for c in range(10) if c != Color.GREY]


def _extract_template_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    assert grid.shape[0] >= TEMPLATE_SIZE and grid.shape[1] >= TEMPLATE_SIZE
    template = grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE]
    assert np.all((template == bg) | (template == Color.GREY))

    grey_elsewhere = grid == Color.GREY
    grey_elsewhere[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = False
    assert not grey_elsewhere.any(), "template grey must be confined to the top-left 3x3"

    mask = template == Color.GREY
    assert mask.any() and (~mask).any(), "template must mix grey and background"
    return mask


def _classify_marker(grid: np.ndarray, comp: dict, bg: int) -> dict[str, int | bool]:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and 1 <= h <= 3, "markers must be 1x1, 2x2, or 3x3 squares"

    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((crop == bg) | (crop == color))
    fg_mask = crop != bg

    if h == 3 and np.array_equal(fg_mask, NEGATIVE_FRAME):
        negative = True
    else:
        assert fg_mask.all(), "ambiguous or invalid marker shape"
        negative = False

    return {"r": r0, "c": c0, "n": h, "color": color, "negative": negative}


def _stamp_patch(template_mask: np.ndarray, n: int, color: int, bg: int, negative: bool) -> np.ndarray:
    paint_mask = ~template_mask if negative else template_mask
    scaled_mask = scale_sprite(paint_mask.astype(int), int(n)).astype(bool)
    patch = np.full((TEMPLATE_SIZE * n, TEMPLATE_SIZE * n), int(bg), dtype=int)
    patch[scaled_mask] = int(color)
    return patch


def _render_output(
    shape: tuple[int, int],
    bg: int,
    template_mask: np.ndarray,
    markers: list[dict[str, int | bool]],
    *,
    scale_override: int | None = None,
    negative_override: bool | None = None,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for marker in markers:
        n = int(marker["n"] if scale_override is None else scale_override)
        negative = bool(marker["negative"] if negative_override is None else negative_override)
        patch = _stamp_patch(template_mask, n, int(marker["color"]), int(bg), negative)
        paste(out, patch, int(marker["r"]), int(marker["c"]), transparent=int(bg), clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Replace square markers with scaled positive or negative stamps of the top-left grey template.

    Input structure:
    - The background is the dominant color.
    - The top-left 3x3 corner contains a grey template pattern mixed with background cells.
    - Elsewhere the grid contains separated monochrome square markers: solid 1x1/2x2/3x3 blocks or hollow 3x3 square frames.

    Rule:
    - Erase the template itself from the output.
    - For each solid n x n marker, use its top-left corner as an anchor and stamp the 3x3 template enlarged by factor n.
    - Grey template cells become the marker color for solid markers, while hollow markers use the negative image and color the template-background cells instead.
    - Clip any part of a stamp that extends beyond the grid boundary.

    Key insight:
    - Each marker encodes three instructions at once: side length gives the scale, solid versus hollow switches positive versus negative, and marker color is the paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    assert bg != Color.GREY
    template_mask = _extract_template_mask(g, bg)

    marker_colors = [int(c) for c in np.unique(g) if c not in {bg, Color.GREY}]
    markers = [
        _classify_marker(g, comp, bg)
        for comp in find_components(
            g,
            bg=bg,
            colors=marker_colors,
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
    ]
    assert markers, "expected at least one square marker"

    return _render_output(g.shape, bg, template_mask, markers)


def _sample_template(rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        mask = rng.integers(0, 2, size=(3, 3), endpoint=False).astype(bool)
        count = int(mask.sum())
        inv = ~mask
        if not (2 <= count <= 7):
            continue
        if int(mask.any(axis=0).sum()) < 2 or int(mask.any(axis=1).sum()) < 2:
            continue
        if int(inv.any(axis=0).sum()) < 2 or int(inv.any(axis=1).sum()) < 2:
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate template")


def _visible_box(shape: tuple[int, int], r: int, c: int, n: int) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    return int(r), int(c), min(int(h), int(r + 3 * n)), min(int(w), int(c + 3 * n))


def _mark_occupied(occupied: np.ndarray, box: tuple[int, int, int, int]) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = box
    occupied[max(0, r0 - VISIBLE_PADDING) : min(h, r1 + VISIBLE_PADDING),
             max(0, c0 - VISIBLE_PADDING) : min(w, c1 + VISIBLE_PADDING)] = True


def _marker_patch(n: int, color: int, bg: int, negative: bool) -> np.ndarray:
    patch = np.full((n, n), int(color), dtype=int)
    if negative:
        assert n == 3
        patch[1, 1] = int(bg)
    return patch


def _visible_stamp_has_structure(template_mask: np.ndarray, n: int, negative: bool, vis_h: int, vis_w: int) -> bool:
    patch = _stamp_patch(template_mask, n, 1, 0, negative)[:vis_h, :vis_w]
    colored = patch == 1
    return bool(colored.any() and (~colored).any())


def _anchor_ranges(shape: tuple[int, int], n: int, clipped: bool, mode: str | None) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, shape)
    full = 3 * n
    if full > h or full > w:
        return None

    if not clipped:
        return (0, h - full), (0, w - full)

    min_visible = 2 * n
    clip_r = (h - full + 1, h - min_visible)
    clip_c = (w - full + 1, w - min_visible)
    full_r = (0, h - full)
    full_c = (0, w - full)

    if mode == "right":
        ranges = (full_r, clip_c)
    elif mode == "bottom":
        ranges = (clip_r, full_c)
    elif mode == "corner":
        ranges = (clip_r, clip_c)
    else:
        return None

    (r0, r1), (c0, c1) = ranges
    if r0 > r1 or c0 > c1:
        return None
    return ranges


def _sample_anchor(
    rng: np.random.Generator,
    shape: tuple[int, int],
    n: int,
    clipped: bool,
) -> tuple[int, int] | None:
    if not clipped:
        ranges = _anchor_ranges(shape, n, False, None)
        assert ranges is not None
        (r0, r1), (c0, c1) = ranges
        return int(rng.integers(r0, r1 + 1)), int(rng.integers(c0, c1 + 1))

    modes = ["right", "bottom", "corner"]
    rng.shuffle(modes)
    feasible = [mode for mode in modes if _anchor_ranges(shape, n, True, mode) is not None]
    if not feasible:
        return None

    mode = feasible[int(rng.integers(len(feasible)))]
    (r0, r1), (c0, c1) = _anchor_ranges(shape, n, True, mode)
    return int(rng.integers(r0, r1 + 1)), int(rng.integers(c0, c1 + 1))


def _place_markers(
    rng: np.random.Generator,
    shape: tuple[int, int],
    bg: int,
    template_mask: np.ndarray,
    specs: list[dict[str, int | bool]],
) -> list[dict[str, int | bool]] | None:
    h, w = map(int, shape)
    occupied = np.zeros((h, w), dtype=bool)
    occupied[:RESERVED_CORNER, :RESERVED_CORNER] = True

    placements: list[dict[str, int | bool]] = []
    for spec in sorted(specs, key=lambda item: (int(item["n"]), int(bool(item["clipped"]))), reverse=True):
        placed = None
        for _ in range(400):
            anchor = _sample_anchor(rng, shape, int(spec["n"]), bool(spec["clipped"]))
            if anchor is None:
                return None
            r, c = anchor
            box = _visible_box(shape, r, c, int(spec["n"]))
            r0, c0, r1, c1 = box
            if occupied[max(0, r0 - VISIBLE_PADDING) : min(h, r1 + VISIBLE_PADDING),
                        max(0, c0 - VISIBLE_PADDING) : min(w, c1 + VISIBLE_PADDING)].any():
                continue

            vis_h = int(r1 - r0)
            vis_w = int(c1 - c0)
            negative = bool(spec["negative"])
            if not _visible_stamp_has_structure(template_mask, int(spec["n"]), negative, vis_h, vis_w):
                continue
            if bool(spec["clipped"]) != (vis_h < 3 * int(spec["n"]) or vis_w < 3 * int(spec["n"])):
                continue

            placed = dict(spec)
            placed["r"] = int(r)
            placed["c"] = int(c)
            _mark_occupied(occupied, box)
            placements.append(placed)
            break
        if placed is None:
            return None
    return placements


def _build_specs(rng: np.random.Generator, bg: int) -> list[dict[str, int | bool]]:
    base = [
        {"n": 1, "negative": False, "clipped": False},
        {"n": 2, "negative": False, "clipped": False},
        {"n": 3, "negative": False, "clipped": False},
        {"n": 3, "negative": True, "clipped": False},
    ]
    clip_index = 2 if bool(rng.integers(0, 2)) else 3
    base[clip_index]["clipped"] = True

    if bool(rng.integers(0, 2)):
        extra_negative = bool(rng.integers(0, 3) == 0)
        base.append({"n": 3 if extra_negative else int(rng.integers(1, 4)),
                     "negative": extra_negative,
                     "clipped": False})

    colors = [int(c) for c in rng.choice([c for c in AVAILABLE_COLORS if c != bg], size=len(base), replace=False)]
    rng.shuffle(colors)
    specs = []
    for spec, color in zip(base, colors):
        item = dict(spec)
        item["color"] = int(color)
        specs.append(item)
    return specs


def _render_input(
    shape: tuple[int, int],
    bg: int,
    template_mask: np.ndarray,
    placements: list[dict[str, int | bool]],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE][template_mask] = Color.GREY
    for marker in placements:
        patch = _marker_patch(int(marker["n"]), int(marker["color"]), int(bg), bool(marker["negative"]))
        paste(grid, patch, int(marker["r"]), int(marker["c"]))
    return grid


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    template_mask: np.ndarray,
    placements: list[dict[str, int | bool]],
    bg: int,
) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(inp, out):
        return False
    if not any(bool(item["negative"]) for item in placements):
        return False
    if {int(item["n"]) for item in placements} != {1, 2, 3}:
        return False
    if not any(bool(item["clipped"]) for item in placements):
        return False

    all_positive = _render_output(inp.shape, bg, template_mask, placements, negative_override=False)
    if np.array_equal(all_positive, out):
        return False

    unit_scale = _render_output(inp.shape, bg, template_mask, placements, scale_override=1)
    if np.array_equal(unit_scale, out):
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX))
    w = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX))
    bg = int(rng.choice(AVAILABLE_COLORS))
    template_mask = _sample_template(rng)
    specs = _build_specs(rng, bg)
    placements = _place_markers(rng, (h, w), bg, template_mask, specs)
    if placements is None:
        return None

    inp = _render_input((h, w), bg, template_mask, placements)
    out = _render_output((h, w), bg, template_mask, placements)
    if not _semantic_checks(inp, out, template_mask, placements, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
