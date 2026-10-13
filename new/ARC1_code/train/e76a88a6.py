from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, bbox_h, bbox_w, components_mask, crop_bbox, find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the multicolor template into every solid grey rectangle of matching size.

    Input structure:
    - A dominant background contains one connected, fully occupied non-grey template.
    - One or more solid grey rectangles have the same height and width as the template.

    Rule:
    - Leave the background and original colored template unchanged.
    - Replace each grey rectangle with an exact copy of the template.

    Key insight:
    - Grey marks blank target regions; the non-grey foreground bbox is the pattern
      that fills every target rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    grey = Color.GREY
    assert bg != grey, "expected grey target rectangles on a separate background"

    template_mask = (g != bg) & (g != grey)
    template_components = components_mask(template_mask, connectivity=4)
    assert len(template_components) == 1, "expected one connected non-grey template"

    template_bbox = bbox(template_mask)
    assert template_bbox is not None, "expected a non-grey template"
    template = crop_bbox(g, template_bbox)
    assert np.all((template != bg) & (template != grey)), "template bbox must be fully occupied"

    grey_targets = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[grey],
        return_format="dict",
    )
    assert grey_targets, "expected at least one grey target rectangle"

    out = g.copy()
    th, tw = template.shape
    for target in grey_targets:
        target_bbox = target["bbox"]
        r0, c0, r1, c1 = target_bbox
        assert bbox_h(target_bbox) == th and bbox_w(target_bbox) == tw, (
            "grey target must match template dimensions"
        )
        assert target["area"] == th * tw, "grey target must be a solid rectangle"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == grey), "grey target bbox must be solid"
        out[r0 : r1 + 1, c0 : c1 + 1] = template

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
GRID_SHAPE = (10, 10)
BACKGROUND_COLOR = Color.BLACK
TARGET_COLOR = Color.GREY
TEMPLATE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Object counts and sizes.
TARGET_COUNTS = [2, 3]
TEMPLATE_DIM_OPTIONS = [(3, 3), (3, 4), (4, 3), (4, 4)]
THREE_TARGET_EXCLUDED_DIM = (4, 4)
SLOT_GAP = 1

# Template pattern sampling.
TEMPLATE_STYLES = ["row_frontier", "col_frontier", "corner_regions"]
FRONTIER_STEPS = [-1, 0, 1]
BOOLEAN_CHOICES = [False, True]
CORNER_JITTER_RANGE = (-0.45, 0.45)

GENERATION_RETRIES = 300


def _frontier_cuts(count: int, limit: int) -> list[int]:
    cut = random.randint(1, limit - 1)
    cuts = []
    for _ in range(count):
        cut = max(1, min(limit - 1, cut + random.choice(FRONTIER_STEPS)))
        cuts.append(cut)
    if random.choice(BOOLEAN_CHOICES):
        cuts.reverse()
    return cuts


def _row_frontier_template(th: int, tw: int, a: int, b: int) -> np.ndarray:
    cuts = np.array(_frontier_cuts(th, tw))[:, None]
    return np.where(np.arange(tw) < cuts, a, b).astype(int)


def _col_frontier_template(th: int, tw: int, a: int, b: int) -> np.ndarray:
    cuts = np.array(_frontier_cuts(tw, th))[None, :]
    return np.where(np.arange(th)[:, None] < cuts, a, b).astype(int)


def _corner_region_template(th: int, tw: int, a: int, b: int) -> np.ndarray:
    rr, cc = np.indices((th, tw))
    corners = [(0, 0), (0, tw - 1), (th - 1, 0), (th - 1, tw - 1)]
    first, second = random.sample(corners, 2)
    jitter = np.random.uniform(*CORNER_JITTER_RANGE, size=(th, tw))
    d_first = np.abs(rr - first[0]) + np.abs(cc - first[1]) + jitter
    d_second = np.abs(rr - second[0]) + np.abs(cc - second[1]) - jitter
    return np.where(d_first <= d_second, a, b).astype(int)


def _force_asymmetric_anchors(template: np.ndarray, a: int, b: int) -> np.ndarray:
    th, tw = template.shape
    lr_pairs = [((r, c), (r, tw - 1 - c)) for r in range(th) for c in range(tw // 2)]
    ud_pairs = [((r, c), (th - 1 - r, c)) for r in range(th // 2) for c in range(tw)]
    random.shuffle(lr_pairs)
    random.shuffle(ud_pairs)

    for lr_left, lr_right in lr_pairs:
        for ud_top, ud_bottom in ud_pairs:
            lr_vals = (a, b) if random.choice(BOOLEAN_CHOICES) else (b, a)
            ud_vals = (a, b) if random.choice(BOOLEAN_CHOICES) else (b, a)
            assignments: dict[tuple[int, int], int] = {}
            ok = True
            for cell, value in (
                (lr_left, lr_vals[0]),
                (lr_right, lr_vals[1]),
                (ud_top, ud_vals[0]),
                (ud_bottom, ud_vals[1]),
            ):
                old = assignments.get(cell)
                if old is not None and old != value:
                    ok = False
                    break
                assignments[cell] = value
            if ok:
                for (r, c), value in assignments.items():
                    template[r, c] = value
                assert not np.array_equal(template, np.fliplr(template))
                assert not np.array_equal(template, np.flipud(template))
                return template

    raise AssertionError("could not assign asymmetric template anchors")


def _gen_template(th: int, tw: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, colors)
    style = random.choice(TEMPLATE_STYLES)
    if style == "row_frontier":
        template = _row_frontier_template(th, tw, a, b)
    elif style == "col_frontier":
        template = _col_frontier_template(th, tw, a, b)
    else:
        template = _corner_region_template(th, tw, a, b)
    if random.choice(BOOLEAN_CHOICES):
        template = np.where(template == a, b, a)
    return _force_asymmetric_anchors(template, a, b)


def _gen_slots_clear(first: tuple[int, int], second: tuple[int, int], th: int, tw: int) -> bool:
    r0, c0 = first
    r1, c1 = second
    return (
        r0 + th + SLOT_GAP <= r1
        or r1 + th + SLOT_GAP <= r0
        or c0 + tw + SLOT_GAP <= c1
        or c1 + tw + SLOT_GAP <= c0
    )


def _gen_slots(th: int, tw: int, total: int) -> list[tuple[int, int]] | None:
    grid_h, grid_w = GRID_SHAPE
    candidates = [(r, c) for r in range(grid_h - th + 1) for c in range(grid_w - tw + 1)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []

    def search(pool: list[tuple[int, int]]) -> bool:
        if len(chosen) == total:
            return True
        valid = [p for p in pool if all(_gen_slots_clear(p, q, th, tw) for q in chosen)]
        if len(valid) < total - len(chosen):
            return False
        random.shuffle(valid)
        for p in valid:
            chosen.append(p)
            next_pool = [q for q in pool if q != p]
            if search(next_pool):
                return True
            chosen.pop()
        return False

    if not search(candidates):
        return None
    random.shuffle(chosen)
    return chosen


def _render_pair(template: np.ndarray, slots: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    th, tw = template.shape
    out = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    for r, c in slots:
        paste(out, template, r, c)

    inp = out.copy()
    grey_patch = np.full((th, tw), TARGET_COLOR, dtype=int)
    for r, c in slots[1:]:
        paste(inp, grey_patch, r, c)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    target_count = random.choice(TARGET_COUNTS)
    dim_options = list(TEMPLATE_DIM_OPTIONS)
    if target_count == 3:
        dim_options.remove(THREE_TARGET_EXCLUDED_DIM)
    th, tw = random.choice(dim_options)

    colors = tuple(random.sample(TEMPLATE_COLORS, 2))
    template = _gen_template(th, tw, colors)
    slots = _gen_slots(th, tw, target_count + 1)
    if slots is None:
        return None

    inp, out = _render_pair(template, slots)
    if np.bincount(inp.ravel(), minlength=10).argmax() != BACKGROUND_COLOR:
        return None
    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e76a88a6 input")
