import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 250


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _crop_mask(mask.astype(bool))
    variants = []
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(_mask_key(_crop_mask(rot)))
        variants.append(_mask_key(_crop_mask(np.fliplr(rot))))
    return min(variants)


def _tight_component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _hole_mask(component_mask: np.ndarray) -> np.ndarray:
    bg_mask = ~component_mask
    holes: list[np.ndarray] = []
    for comp in _components(bg_mask):
        if any(
            r == 0 or c == 0 or r == component_mask.shape[0] - 1 or c == component_mask.shape[1] - 1
            for r, c in comp
        ):
            continue
        hole = np.zeros_like(component_mask, dtype=bool)
        for r, c in comp:
            hole[r, c] = True
        holes.append(hole)
    assert len(holes) == 1, "each grey component must contain exactly one enclosed hole"
    return _crop_mask(holes[0])


def _component_signature(coords: list[tuple[int, int]]) -> tuple[str, ...]:
    return _shape_signature(_hole_mask(_tight_component_mask(coords)))


def _legend_side(grey_components: list[list[tuple[int, int]]]) -> str:
    left = [comp for comp in grey_components if any(c == 0 for _r, c in comp)]
    top = [comp for comp in grey_components if any(r == 0 for r, _c in comp)]
    if left and not top:
        return "left"
    if top and not left:
        return "top"
    if left:
        return "left"
    assert top, "expected edge-touching legend templates"
    return "top"


def _legend_color(grid: np.ndarray, comp: list[tuple[int, int]], side: str, bg: int) -> int:
    h, w = grid.shape
    counts: Counter[int] = Counter()
    for r, c in comp:
        nr, nc = (r, c + 1) if side == "left" else (r + 1, c)
        if 0 <= nr < h and 0 <= nc < w:
            value = int(grid[nr, nc])
            if value not in (bg, GREY):
                counts[value] += 1
    assert counts, "legend template must touch a colored block on the inward side"
    return counts.most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by matching its hole shape to a border legend template.

    Input structure:
    - The background is black.
    - A legend sits on the left edge or the top edge; each legend entry contains a grey hollow template and an adjacent solid color block.
    - Elsewhere, several disconnected grey objects each contain exactly one enclosed black hole.

    Rule:
    - Extract the enclosed hole shape from every grey object and every grey legend template.
    - Normalize each hole shape up to rotation and reflection, then use the legend template with the same normalized hole as a lookup key.
    - Recolor all grey pixels of each non-legend object with the color from its matched legend block, while leaving holes and the legend unchanged.

    Key insight:
    - Object identity is encoded by the geometry of the enclosed hole, not by the outer grey silhouette.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    grey_components = _components(g == GREY)
    assert grey_components, "expected grey components"

    side = _legend_side(grey_components)
    legend_components = [
        comp
        for comp in grey_components
        if (side == "left" and any(c == 0 for _r, c in comp))
        or (side == "top" and any(r == 0 for r, _c in comp))
    ]
    assert legend_components, "legend must contain edge-touching grey templates"

    mapping: dict[tuple[str, ...], int] = {}
    for comp in legend_components:
        sig = _component_signature(comp)
        color = _legend_color(g, comp, side, bg)
        assert sig not in mapping, "legend hole shapes must be unique"
        mapping[sig] = color

    out = g.copy()
    for comp in grey_components:
        is_legend = (side == "left" and any(c == 0 for _r, c in comp)) or (
            side == "top" and any(r == 0 for r, _c in comp)
        )
        if is_legend:
            continue
        sig = _component_signature(comp)
        assert sig in mapping, "every object hole must appear in the legend"
        for r, c in comp:
            out[r, c] = mapping[sig]

    return out


def _random_connected_mask(rng: random.Random, h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    cells = [start]
    frontier = set()
    while len(cells) < area:
        r, c = rng.choice(cells)
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
        if not frontier:
            return _random_connected_mask(rng, h, w, area)
        nr, nc = rng.choice(tuple(frontier))
        frontier.discard((nr, nc))
        mask[nr, nc] = True
        cells.append((nr, nc))
    return mask


def _random_hole_shape(rng: random.Random, used: set[tuple[str, ...]]) -> tuple[np.ndarray, tuple[str, ...]]:
    for _ in range(200):
        h = rng.randint(1, 4)
        w = rng.randint(1, 4)
        area = rng.randint(1, h * w)
        hole = _random_connected_mask(rng, h, w, area)
        sig = _shape_signature(hole)
        if sig not in used:
            return hole, sig
    raise RuntimeError("failed to sample a distinct hole shape")


def _random_dihedral(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _build_hollow_shape(
    hole: np.ndarray,
    rng: random.Random,
    pad_min: int,
    pad_max: int,
    max_tabs: int,
) -> np.ndarray:
    hole = _random_dihedral(hole, rng)
    hh, ww = hole.shape
    top = rng.randint(pad_min, pad_max)
    bottom = rng.randint(pad_min, pad_max)
    left = rng.randint(pad_min, pad_max)
    right = rng.randint(pad_min, pad_max)

    base_h = hh + top + bottom
    base_w = ww + left + right
    margin = 2
    canvas = np.zeros((base_h + 2 * margin, base_w + 2 * margin), dtype=bool)
    br = bc = margin
    canvas[br : br + base_h, bc : bc + base_w] = True
    canvas[br + top : br + top + hh, bc + left : bc + left + ww][hole] = False

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[: rng.randint(0, max_tabs)]:
        if side in ("top", "bottom"):
            thickness = rng.randint(1, margin)
            width = rng.randint(max(1, base_w // 3), base_w)
            c0 = bc + rng.randint(0, base_w - width)
            if side == "top":
                canvas[br - thickness : br, c0 : c0 + width] = True
            else:
                canvas[br + base_h : br + base_h + thickness, c0 : c0 + width] = True
        else:
            thickness = rng.randint(1, margin)
            height = rng.randint(max(1, base_h // 3), base_h)
            r0 = br + rng.randint(0, base_h - height)
            if side == "left":
                canvas[r0 : r0 + height, bc - thickness : bc] = True
            else:
                canvas[r0 : r0 + height, bc + base_w : bc + base_w + thickness] = True

    return _crop_mask(canvas)


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _paint_shape(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    rows, cols = np.where(mask)
    grid[r0 + rows, c0 + cols] = int(color)


def _reserve(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[max(0, r0 - 1) : min(occ.shape[0], r0 + h + 1), max(0, c0 - 1) : min(occ.shape[1], c0 + w + 1)] = True


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return not occ[
        max(0, r0 - 1) : min(occ.shape[0], r0 + h + 1),
        max(0, c0 - 1) : min(occ.shape[1], c0 + w + 1),
    ].any()


def _left_entry(template: np.ndarray, color: int, thickness: int) -> np.ndarray:
    h, w = template.shape
    patch = np.full((h, w + thickness), BG, dtype=int)
    patch[:, w:] = int(color)
    rows, cols = np.where(template)
    patch[rows, cols] = GREY
    return patch


def _top_entry(template: np.ndarray, color: int, thickness: int) -> np.ndarray:
    h, w = template.shape
    patch = np.full((h + thickness, w), BG, dtype=int)
    patch[h:, :] = int(color)
    rows, cols = np.where(template)
    patch[rows, cols] = GREY
    return patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(["left", "top"])
    key_count = rng.randint(2, 3)
    colors = rng.sample(PALETTE_COLORS, key_count)

    used_sigs: set[tuple[str, ...]] = set()
    legend_entries: list[np.ndarray] = []
    object_specs: list[tuple[np.ndarray, int]] = []
    expected_objects: list[tuple[np.ndarray, int, int, int]] = []

    for color in colors:
        hole, sig = _random_hole_shape(rng, used_sigs)
        used_sigs.add(sig)

        legend_template = _build_hollow_shape(hole, rng, pad_min=1, pad_max=2, max_tabs=2)
        block_thickness = rng.randint(1, 2)
        legend_entries.append(
            _left_entry(legend_template, color, block_thickness)
            if side == "left"
            else _top_entry(legend_template, color, block_thickness)
        )

        for _ in range(rng.randint(1, 2)):
            obj = _build_hollow_shape(hole, rng, pad_min=1, pad_max=3, max_tabs=3)
            if max(obj.shape) > 10:
                return None
            object_specs.append((obj, color))

    gap = rng.randint(1, 2)
    if side == "left":
        legend_h = sum(entry.shape[0] for entry in legend_entries) + gap * (len(legend_entries) - 1)
        legend_w = max(entry.shape[1] for entry in legend_entries)
        if legend_h > 28 or legend_w > 8:
            return None
        h_min = max(18, legend_h + 2)
        w_min = max(18, legend_w + 8)
        if h_min > 30 or w_min > 30:
            return None
        height = rng.randint(h_min, 30)
        width = rng.randint(w_min, 30)
    else:
        legend_h = max(entry.shape[0] for entry in legend_entries)
        legend_w = sum(entry.shape[1] for entry in legend_entries) + gap * (len(legend_entries) - 1)
        if legend_h > 8 or legend_w > 28:
            return None
        h_min = max(18, legend_h + 8)
        w_min = max(18, legend_w + 2)
        if h_min > 30 or w_min > 30:
            return None
        height = rng.randint(h_min, 30)
        width = rng.randint(w_min, 30)

    grid = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    if side == "left":
        start_r = rng.randint(1, height - legend_h - 1)
        cursor = start_r
        for entry in legend_entries:
            _place_patch(grid, entry, cursor, 0)
            cursor += entry.shape[0] + gap
        occ[:, : min(width, legend_w + 2)] = True
    else:
        start_c = rng.randint(1, width - legend_w - 1)
        cursor = start_c
        for entry in legend_entries:
            _place_patch(grid, entry, 0, cursor)
            cursor += entry.shape[1] + gap
        occ[: min(height, legend_h + 2), :] = True

    rng.shuffle(object_specs)
    object_specs.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
    for mask, color in object_specs:
        h, w = mask.shape
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            if side == "left":
                if height - h - 1 < 1 or width - w - 1 < legend_w + 2:
                    return None
                r0 = rng.randint(1, height - h - 1)
                c0 = rng.randint(legend_w + 2, width - w - 1)
            else:
                if height - h - 1 < legend_h + 2 or width - w - 1 < 1:
                    return None
                r0 = rng.randint(legend_h + 2, height - h - 1)
                c0 = rng.randint(1, width - w - 1)
            if not _can_place(occ, r0, c0, h, w):
                continue
            _paint_shape(grid, mask, GREY, r0, c0)
            _reserve(occ, r0, c0, h, w)
            expected_objects.append((mask, r0, c0, color))
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if side == "left":
        if not np.array_equal(out[:, :legend_w + 1], grid[:, :legend_w + 1]):
            return None
    else:
        if not np.array_equal(out[:legend_h + 1, :], grid[:legend_h + 1, :]):
            return None

    for mask, r0, c0, color in expected_objects:
        rows, cols = np.where(mask)
        if not np.all(out[r0 + rows, c0 + cols] == int(color)):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
