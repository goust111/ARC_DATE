from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GEN_ATTEMPTS = 400
SECTORS = ("L", "R", "U", "D", "UL", "UR", "DL", "DR")


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int)).astype(int)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _boxes_conflict(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _rect_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            box = (min(rows), min(cols), max(rows), max(cols))
            boxes.append(box)
    return boxes


def _transform_for_slot(motif: np.ndarray, box: tuple[int, int, int, int], source_box: tuple[int, int, int, int]) -> np.ndarray:
    sr0, sc0, sr1, sc1 = source_box
    tr0, tc0, tr1, tc1 = box
    row_overlap = _intervals_overlap(sr0, sr1, tr0, tr1)
    col_overlap = _intervals_overlap(sc0, sc1, tc0, tc1)
    assert not (row_overlap and col_overlap), "slot must be offset on at least one axis"
    if row_overlap:
        return np.fliplr(motif)
    if col_overlap:
        return np.flipud(motif)
    return np.rot90(motif, 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the colored source motif into each black slot with a direction-selected mirror and a slot-selected scale.

    Input structure:
    - One non-black background surrounds a single fully occupied colored rectangular motif.
    - Elsewhere, one or more solid black rectangular slots appear without touching the motif.
    - Each slot height and width are the same integer multiple of the motif height and width.

    Rule:
    - Crop the colored motif from its tight bounding box.
    - For every black slot, compute its integer scale factor from the slot size relative to the motif size.
    - If the slot overlaps the motif's rows, paste a left-right mirror of the motif scaled by that factor; if it overlaps the motif's columns, paste an up-down mirror scaled by that factor; otherwise paste the 180-degree rotation scaled by that factor.

    Key insight:
    - The slot does not encode the transform by color or markings; its relative position to the motif determines which mirror combination to stamp, while the slot size determines the upscale factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    assert bg != BLACK, "background must be non-black"

    source_mask = (g != bg) & (g != BLACK)
    source_box = _bbox(source_mask)
    assert source_box is not None, "expected one colored motif"
    motif = _crop(g, source_box)
    assert np.all((motif != bg) & (motif != BLACK)), "motif bbox must be fully occupied"

    slot_boxes = _rect_components(g == BLACK)
    assert slot_boxes, "expected at least one black slot"

    out = g.copy()
    mh, mw = motif.shape
    for box in slot_boxes:
        slot = _crop(g, box)
        assert np.all(slot == BLACK), "each slot component must be a solid black rectangle"
        sh, sw = slot.shape
        assert sh % mh == 0 and sw % mw == 0, "slot dimensions must be integer multiples of motif dimensions"
        factor_h = sh // mh
        factor_w = sw // mw
        assert factor_h == factor_w, "slot must use a uniform scale factor"
        transformed = _transform_for_slot(motif, box, source_box)
        scaled = _scale_sprite(transformed, factor_h)
        _paste(out, scaled, box[0], box[1])

    return out


def _make_motif(h: int, w: int, palette: list[int]) -> np.ndarray:
    for _ in range(120):
        num_colors = random.randint(2, min(4, len(palette)))
        colors = random.sample(palette, num_colors)
        motif = np.array(random.choices(colors, k=h * w), dtype=int).reshape(h, w)
        if len(set(map(int, motif.ravel()))) < 2:
            continue
        variants = [motif, np.fliplr(motif), np.flipud(motif), np.rot90(motif, 2)]
        distinct = True
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                if np.array_equal(variants[i], variants[j]):
                    distinct = False
                    break
            if not distinct:
                break
        if distinct:
            return motif
    raise RuntimeError("failed to sample an asymmetric motif")


def _diagonal_sector(horizontal: str, vertical: str) -> str:
    return {
        ("L", "U"): "DR",
        ("L", "D"): "UR",
        ("R", "U"): "DL",
        ("R", "D"): "UL",
    }[(horizontal, vertical)]


def _sector_sides(sector: str) -> tuple[str | None, str | None]:
    vertical = "N" if "U" in sector else "S" if "D" in sector else None
    horizontal = "W" if "L" in sector else "E" if "R" in sector else None
    return vertical, horizontal


def _slot_ranges(
    sector: str,
    slot_h: int,
    slot_w: int,
    gap: int,
    motif_box: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    margins: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = grid_shape
    top_margin, left_margin, bottom_margin, right_margin = margins
    mr0, mc0, mr1, mc1 = motif_box
    v_side, h_side = _sector_sides(sector)

    if v_side == "N":
        r_lo = top_margin
        r_hi = mr0 - gap - slot_h
    elif v_side == "S":
        r_lo = mr1 + gap + 1
        r_hi = h - bottom_margin - slot_h
    else:
        r_lo = max(top_margin, mr0 - slot_h + 1)
        r_hi = min(h - bottom_margin - slot_h, mr1)

    if h_side == "W":
        c_lo = left_margin
        c_hi = mc0 - gap - slot_w
    elif h_side == "E":
        c_lo = mc1 + gap + 1
        c_hi = w - right_margin - slot_w
    else:
        c_lo = max(left_margin, mc0 - slot_w + 1)
        c_hi = min(w - right_margin - slot_w, mc1)

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (r_lo, r_hi), (c_lo, c_hi)


def _required_extents(slot_specs: list[dict[str, object]]) -> tuple[int, int, int, int]:
    up = down = left = right = 0
    for spec in slot_specs:
        sector = spec["sector"]
        gap = int(spec["gap"])
        slot_h, slot_w = spec["shape"]
        v_side, h_side = _sector_sides(str(sector))
        if v_side == "N":
            up = max(up, slot_h + gap)
        if v_side == "S":
            down = max(down, slot_h + gap)
        if h_side == "W":
            left = max(left, slot_w + gap)
        if h_side == "E":
            right = max(right, slot_w + gap)
    return up, down, left, right


def _build_slot_specs(motif: np.ndarray) -> list[dict[str, object]]:
    h, w = motif.shape
    horizontal = random.choice(["L", "R"])
    vertical = random.choice(["U", "D"])
    sectors = [horizontal, vertical, _diagonal_sector(horizontal, vertical)]
    if random.random() < 0.45:
        extras = [sector for sector in SECTORS if sector not in sectors]
        sectors.append(random.choice(extras))

    max_k = max(2, min(4, 12 // max(h, w)))
    scales = [1, random.randint(2, max_k), random.randint(1, max_k)]
    random.shuffle(scales)
    while len(scales) < len(sectors):
        scales.append(random.randint(1, max_k))

    specs: list[dict[str, object]] = []
    for sector, k in zip(sectors, scales):
        specs.append(
            {
                "sector": sector,
                "k": int(k),
                "gap": random.randint(1, 3),
                "shape": (h * int(k), w * int(k)),
            }
        )
    if not any(int(spec["k"]) > 1 for spec in specs):
        chosen = specs[random.randrange(len(specs))]
        chosen["k"] = 2
        chosen["shape"] = (h * 2, w * 2)
    return specs


def _place_slots(
    motif_box: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    margins: tuple[int, int, int, int],
    slot_specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    placed_boxes = [motif_box]
    ordered = sorted(slot_specs, key=lambda spec: spec["shape"][0] * spec["shape"][1], reverse=True)
    for spec in ordered:
        slot_h, slot_w = spec["shape"]
        ranges = _slot_ranges(
            str(spec["sector"]),
            int(slot_h),
            int(slot_w),
            int(spec["gap"]),
            motif_box,
            grid_shape,
            margins,
        )
        if ranges is None:
            return None
        (r_lo, r_hi), (c_lo, c_hi) = ranges
        placed = False
        for _ in range(120):
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            box = (r0, c0, r0 + int(slot_h) - 1, c0 + int(slot_w) - 1)
            if any(_boxes_conflict(box, other, pad=1) for other in placed_boxes):
                continue
            spec["box"] = box
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return slot_specs


def _render_example(
    bg: int,
    motif: np.ndarray,
    motif_box: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    slot_specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid_shape
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    _paste(inp, motif, motif_box[0], motif_box[1])
    _paste(out, motif, motif_box[0], motif_box[1])
    for spec in slot_specs:
        box = spec["box"]
        sector = str(spec["sector"])
        k = int(spec["k"])
        r0, c0, r1, c1 = box
        inp[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        patch = _scale_sprite(_transform_for_slot(motif, box, motif_box), k)
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        _paste(out, patch, r0, c0)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(1, 9)
    motif_h = random.randint(2, 4)
    motif_w = random.randint(2, 4)
    motif_palette = [color for color in range(1, 10) if color != bg]
    motif = _make_motif(motif_h, motif_w, motif_palette)

    slot_specs = _build_slot_specs(motif)
    up, down, left, right = _required_extents(slot_specs)
    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    grid_h = top_margin + up + motif_h + down + bottom_margin
    grid_w = left_margin + left + motif_w + right + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    motif_r0 = top_margin + up
    motif_c0 = left_margin + left
    motif_box = (motif_r0, motif_c0, motif_r0 + motif_h - 1, motif_c0 + motif_w - 1)
    margins = (top_margin, left_margin, bottom_margin, right_margin)
    placed_specs = _place_slots(motif_box, (grid_h, grid_w), margins, slot_specs)
    if placed_specs is None:
        return None

    inp, out = _render_example(bg, motif, motif_box, (grid_h, grid_w), placed_specs)
    if _most_common_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
