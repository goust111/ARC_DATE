import random

import numpy as np


BG = 0
FENCE_COLOR = 1
INNER_COLORS = (2, 3, 4, 6, 7, 8)
TEMPLATE_COLOR = 5
SEP_COLOR = 9
DIRECTIONS = ("top", "right", "bottom", "left")
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 300

MOTIFS = (
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
)


def _most_common_color(g: np.ndarray) -> int:
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_connected(mask: np.ndarray) -> bool:
    return np.any(mask) and len(_components(mask.astype(bool))) == 1


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _distinct_rotations(mask: np.ndarray) -> int:
    return len({_mask_signature(np.rot90(mask, k)) for k in range(4)})


def _rotate_for_direction(mask: np.ndarray, direction: str) -> np.ndarray:
    turns = {"top": 0, "right": 3, "bottom": 2, "left": 1}[direction]
    return np.rot90(mask, turns)


def _find_fences(g: np.ndarray, bg: int) -> tuple[list[dict], int]:
    fences = []
    for coords in _components(g == FENCE_COLOR):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        if h < 4 or w < 4 or h != w:
            continue

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        fence_mask = patch == FENCE_COLOR
        if np.any(fence_mask & ~border):
            continue

        missing = np.argwhere(border & ~fence_mask)
        if missing.shape[0] != 1:
            continue
        mr, mc = map(int, missing[0])
        if (mr in (0, h - 1)) and (mc in (0, w - 1)):
            continue
        if int(patch[mr, mc]) != bg:
            continue

        interior = patch[1:-1, 1:-1]
        colors = set(map(int, np.unique(interior))) - {bg}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        shape_mask = interior == color
        if not _is_connected(shape_mask):
            continue

        if mr == 0:
            direction = "top"
        elif mr == h - 1:
            direction = "bottom"
        elif mc == 0:
            direction = "left"
        else:
            direction = "right"

        fences.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "color": color,
                "shape": shape_mask.copy(),
            }
        )

    assert fences, "expected at least one broken rectangular fence"
    size = fences[0]["shape"].shape[0]
    assert all(fence["shape"].shape == (size, size) for fence in fences)
    return fences, int(size)


def _parse_templates(g: np.ndarray, size: int, bg: int) -> dict[tuple[tuple[int, ...], ...], np.ndarray]:
    comps = _components(g == SEP_COLOR)
    assert len(comps) == 1, "expected one maroon template strip"
    r0, c0, r1, c1 = _bbox(comps[0])
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h == size + 2, "template strip height must match fence interior size"
    assert (w - 1) % (size + 1) == 0, "template strip width must decompose into equal cells"
    count = (w - 1) // (size + 1)
    assert count >= 2, "expected multiple templates"

    assert np.all(patch[0, :] == SEP_COLOR) and np.all(patch[-1, :] == SEP_COLOR)
    assert np.all(patch[:, 0] == SEP_COLOR) and np.all(patch[:, -1] == SEP_COLOR)

    sep_mask = np.zeros_like(patch, dtype=bool)
    sep_mask[0, :] = True
    sep_mask[-1, :] = True
    for i in range(count + 1):
        sep_col = i * (size + 1)
        sep_mask[:, sep_col] = True
        assert np.all(patch[:, sep_col] == SEP_COLOR)
    assert np.all(patch[sep_mask] == SEP_COLOR)
    assert np.all(patch[~sep_mask] != SEP_COLOR)

    templates: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for i in range(count):
        c_start = 1 + i * (size + 1)
        sub = patch[1 : size + 1, c_start : c_start + size]
        assert set(map(int, np.unique(sub))) <= {bg, TEMPLATE_COLOR}
        mask = sub == TEMPLATE_COLOR
        assert mask.any() and _is_connected(mask), "template must contain one connected grey shape"
        sig = _canonical_signature(mask)
        assert sig not in templates, "template shapes must be unique up to rotation"
        templates[sig] = mask.copy()
    return templates


def _stamp_origin(fence_bbox: tuple[int, int, int, int], size: int, direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = fence_bbox
    if direction == "top":
        return r0 - size, c0 + 1
    if direction == "bottom":
        return r1 + 1, c0 + 1
    if direction == "left":
        return r0 + 1, c0 - size
    return r0 + 1, c1 + 1


def _render_stamps(
    grid: np.ndarray,
    fences: list[dict],
    templates: dict[tuple[tuple[int, ...], ...], np.ndarray],
    size: int,
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for fence in fences:
        sig = _canonical_signature(fence["shape"])
        assert sig in templates, "each fence shape must match one template"
        base_mask = templates[sig]
        stamp_mask = _rotate_for_direction(base_mask, fence["direction"])
        sr, sc = _stamp_origin(fence["bbox"], size, fence["direction"])
        assert 0 <= sr and 0 <= sc and sr + size <= h and sc + size <= w
        region_in = grid[sr : sr + size, sc : sc + size]
        region_out = out[sr : sr + size, sc : sc + size]
        assert np.all(region_in[stamp_mask] == BG), "stamp area must be empty in the input"
        region_out[stamp_mask] = fence["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fenced inner shape to its grey template, then stamp that template outside the fence gap.

    Input structure:
    - The background is black.
    - Several blue one-pixel-thick square fences appear on the grid; each fence is a rectangle border with exactly one missing non-corner border pixel.
    - Inside every fence there is one connected colored shape on a same-sized black interior.
    - A maroon-framed template strip contains several equal square cells; each cell shows one connected grey shape on black, and templates are separated by maroon columns.

    Rule:
    - Read the colored shape inside each fence as a same-sized binary mask and match it to the unique template with the same shape up to 90-degree rotations.
    - Determine which side of the fence contains the missing border pixel.
    - Copy the matched template outside that side of the fence, flush against the fence, recoloring the grey template pixels with the fence's interior color.
    - Rotate the copied template according to the gap direction: top keeps the shown orientation, right/bottom/left use the corresponding quarter-turns.

    Key insight:
    - Shape identity comes from rotation-invariant template matching, but the placed stamp orientation does not come from the current interior rotation; it is set independently by the fence gap direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    assert bg == BG, "generated task uses black as the background"
    fences, size = _find_fences(g, bg)
    templates = _parse_templates(g, size, bg)
    return _render_stamps(g, fences, templates, size)


def _motif_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _sample_template_masks(size: int, count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    used = set()
    for _ in range(2000):
        motif = random.choice(MOTIFS)
        mh, mw = _motif_dims(motif)
        if mh > size or mw > size:
            continue
        top = random.randint(0, size - mh)
        left = random.randint(0, size - mw)
        mask = np.zeros((size, size), dtype=bool)
        for r, c in motif:
            mask[top + r, left + c] = True
        mask = np.rot90(mask, random.randrange(4))
        if _distinct_rotations(mask) < 2:
            continue
        sig = _canonical_signature(mask)
        if sig in used:
            continue
        used.add(sig)
        masks.append(mask.copy())
        if len(masks) == count:
            return masks
    raise RuntimeError("failed to sample distinct template masks")


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], margin: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - margin),
        max(0, box[1] - margin),
        min(h - 1, box[2] + margin),
        min(w - 1, box[3] + margin),
    )


def _draw_template_strip(grid: np.ndarray, masks: list[np.ndarray], corner: str) -> tuple[int, int, int, int]:
    size = masks[0].shape[0]
    count = len(masks)
    strip_h = size + 2
    strip_w = count * (size + 1) + 1
    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - strip_w
    elif corner == "bl":
        r0, c0 = h - strip_h, 0
    else:
        r0, c0 = h - strip_h, w - strip_w

    strip = np.full((strip_h, strip_w), BG, dtype=int)
    strip[0, :] = SEP_COLOR
    strip[-1, :] = SEP_COLOR
    for i in range(count + 1):
        strip[:, i * (size + 1)] = SEP_COLOR
    for i, mask in enumerate(masks):
        c_start = 1 + i * (size + 1)
        strip[1 : size + 1, c_start : c_start + size][mask] = TEMPLATE_COLOR
    grid[r0 : r0 + strip_h, c0 : c0 + strip_w] = strip
    return r0, c0, r0 + strip_h - 1, c0 + strip_w - 1


def _fence_union_bbox(r0: int, c0: int, size: int, direction: str) -> tuple[int, int, int, int]:
    outer = size + 2
    r1 = r0 + outer - 1
    c1 = c0 + outer - 1
    if direction == "top":
        return r0 - size, c0, r1, c1
    if direction == "bottom":
        return r0, c0, r1 + size, c1
    if direction == "left":
        return r0, c0 - size, r1, c1
    return r0, c0, r1, c1 + size


def _sample_fence_position(
    grid_shape: tuple[int, int],
    size: int,
    direction: str,
    reserved: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = grid_shape
    outer = size + 2
    if direction == "top":
        r_range = (size, h - outer)
        c_range = (0, w - outer)
    elif direction == "bottom":
        r_range = (0, h - (outer + size))
        c_range = (0, w - outer)
    elif direction == "left":
        r_range = (0, h - outer)
        c_range = (size, w - outer)
    else:
        r_range = (0, h - outer)
        c_range = (0, w - (outer + size))

    if r_range[0] > r_range[1] or c_range[0] > c_range[1]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(r_range[0], r_range[1])
        c0 = random.randint(c_range[0], c_range[1])
        union = _fence_union_bbox(r0, c0, size, direction)
        if any(_box_overlap(union, box) for box in reserved):
            continue
        return r0, c0
    return None


def _draw_fence(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_mask: np.ndarray,
    direction: str,
    color: int,
    gap_index: int,
) -> tuple[int, int, int, int]:
    size = inner_mask.shape[0]
    outer = size + 2
    patch = np.full((outer, outer), BG, dtype=int)
    patch[0, :] = FENCE_COLOR
    patch[-1, :] = FENCE_COLOR
    patch[:, 0] = FENCE_COLOR
    patch[:, -1] = FENCE_COLOR
    if direction == "top":
        patch[0, gap_index] = BG
    elif direction == "bottom":
        patch[-1, gap_index] = BG
    elif direction == "left":
        patch[gap_index, 0] = BG
    else:
        patch[gap_index, -1] = BG
    patch[1:-1, 1:-1][inner_mask] = color
    grid[r0 : r0 + outer, c0 : c0 + outer] = patch
    return r0, c0, r0 + outer - 1, c0 + outer - 1


def _choose_template_indices(template_count: int, fence_count: int) -> list[int]:
    if fence_count <= template_count and random.random() < 0.7:
        return random.sample(range(template_count), fence_count)
    indices = [random.randrange(template_count) for _ in range(fence_count)]
    if fence_count > 1 and len(set(indices)) == 1:
        indices[1] = (indices[0] + 1) % template_count
    return indices


def _build_expected_output(inp: np.ndarray, fence_specs: list[dict], templates: list[np.ndarray], size: int) -> np.ndarray:
    out = inp.copy()
    for spec in fence_specs:
        mask = _rotate_for_direction(templates[spec["template_index"]], spec["direction"])
        sr, sc = _stamp_origin(spec["bbox"], size, spec["direction"])
        out[sr : sr + size, sc : sc + size][mask] = spec["color"]
    return out


def _generate_candidate() -> np.ndarray | None:
    size = random.choice((4, 5))
    template_count = random.choice((3, 4))
    template_width = template_count * (size + 1) + 1

    h = random.randint(size * 4 + 5, 30)
    w = random.randint(max(25, template_width), 30)
    if random.random() < 0.5:
        h, w = w, h
    h = min(h, 30)
    w = min(w, 30)
    if h < size * 4 + 5 or w < template_width:
        return None

    max_fences = 2 if template_width >= 25 or min(h, w) < 27 else 3
    fence_count = random.choice((2, max_fences)) if max_fences == 3 else 2

    templates = _sample_template_masks(size, template_count)
    grid = np.full((h, w), BG, dtype=int)
    corner = random.choice(("tl", "tr", "bl", "br"))
    template_box = _draw_template_strip(grid, templates, corner)

    directions = [random.choice(DIRECTIONS) for _ in range(fence_count)]
    colors = random.sample(INNER_COLORS, fence_count)
    template_indices = _choose_template_indices(template_count, fence_count)
    inner_rotations = [random.randrange(4) for _ in range(fence_count)]
    if fence_count > 1 and all(turns == 0 for turns in inner_rotations):
        inner_rotations[0] = random.choice((1, 2, 3))

    reserved = [_expand_box(template_box, 1, h, w)]
    fence_specs = []
    for idx in range(fence_count):
        direction = directions[idx]
        pos = _sample_fence_position((h, w), size, direction, reserved)
        if pos is None:
            return None
        r0, c0 = pos
        bbox = _draw_fence(
            grid,
            r0,
            c0,
            np.rot90(templates[template_indices[idx]], inner_rotations[idx]),
            direction,
            colors[idx],
            random.randint(1, size),
        )
        reserved.append(_expand_box(_fence_union_bbox(r0, c0, size, direction), 1, h, w))
        fence_specs.append(
            {
                "bbox": bbox,
                "direction": direction,
                "color": colors[idx],
                "template_index": template_indices[idx],
            }
        )

    expected = _build_expected_output(grid, fence_specs, templates, size)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
