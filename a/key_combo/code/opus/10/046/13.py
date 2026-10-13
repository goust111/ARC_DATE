import numpy as np


BACKGROUND = 0
TEMPLATE_COLOR = 1
REFERENCE_COLOR = 2
DIVIDER_COLOR = 5
N_SLOTS = 4
SCALE_FACTOR = 2
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, k=-(turns % 4))


def _full_divider_columns(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER_COLOR)]


def _region_spans_from_dividers(grid: np.ndarray) -> list[tuple[int, int]]:
    divider_cols = set(_full_divider_columns(grid))
    spans: list[tuple[int, int]] = []
    start = None
    for c in range(grid.shape[1]):
        if c in divider_cols:
            if start is not None:
                spans.append((start, c))
                start = None
        elif start is None:
            start = c
    if start is not None:
        spans.append((start, grid.shape[1]))
    return spans


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _parse_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    spans = _region_spans_from_dividers(g)
    divider_cols = _full_divider_columns(g)
    assert len(spans) == N_SLOTS + 1, f"expected {N_SLOTS + 1} regions, found {len(spans)}"

    slot_spans = spans[:N_SLOTS]
    ref_span = spans[-1]

    template_slot = g[:, slot_spans[0][0] : slot_spans[0][1]]
    template_mask = template_slot == TEMPLATE_COLOR
    template_box = _bbox(template_mask)
    assert template_box is not None, "first slot must contain a blue template"
    anchor_r, anchor_c, _, _ = template_box
    base_mask = _crop_mask(template_mask)

    reference_region = g[:, ref_span[0] : ref_span[1]]
    reference_mask = reference_region == REFERENCE_COLOR
    reference_box = _bbox(reference_mask)
    assert reference_box is not None, "reference region must contain a red shape"
    ref_small = _crop_mask(reference_mask)
    scaled_ref = _scale_mask(ref_small, SCALE_FACTOR)

    rotations = [_rotate_cw(base_mask, k) for k in range(N_SLOTS)]
    matches = [np.array_equal(rot, scaled_ref) for rot in rotations]
    assert sum(matches) == 1, "reference must match exactly one rotated template"

    slot_widths = [c1 - c0 for c0, c1 in slot_spans]
    assert len(set(slot_widths)) == 1, "main slots must have equal width"
    slot_height = g.shape[0]
    for rot in rotations:
        assert anchor_r + rot.shape[0] <= slot_height
    for (c0, c1), rot in zip(slot_spans, rotations):
        assert anchor_c + rot.shape[1] <= c1 - c0

    return {
        "grid": g,
        "divider_cols": divider_cols,
        "slot_spans": slot_spans,
        "reference_span": ref_span,
        "anchor": (anchor_r, anchor_c),
        "rotations": rotations,
        "matches": matches,
    }


def _build_output(parsed: dict) -> np.ndarray:
    g = parsed["grid"]
    out = np.full_like(g, BACKGROUND)
    out[:, parsed["divider_cols"]] = DIVIDER_COLOR
    anchor_r, anchor_c = parsed["anchor"]

    for idx, ((c0, _c1), mask) in enumerate(zip(parsed["slot_spans"], parsed["rotations"])):
        color = REFERENCE_COLOR if parsed["matches"][idx] else TEMPLATE_COLOR
        _paste_mask(out, mask, anchor_r, c0 + anchor_c, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the first-slot template across four divider-separated slots, match the scaled reference shape, and recolor the matching slot.

    Input structure:
    - Vertical gray divider columns split the canvas into four equal-width main slots and one right-side reference region.
    - The first slot contains a blue 4-connected template on black; the remaining three main slots are black.
    - The reference region contains one red connected shape whose 2x upscaled mask matches one rotated version of the template.

    Rule:
    - Detect the divider-separated regions and crop the blue template from the first slot.
    - Place clockwise rotations of that cropped template into slots 1..4 using the same slot-local anchor as the original.
    - Upscale the red reference mask by a factor of 2 and compare it against the four rotated template masks.
    - Recolor the uniquely matching slot from blue to red, keep the other slots blue, preserve the gray dividers, and clear the reference region to black.

    Key insight:
    - The reference object is not copied directly: it becomes a 2x shape query that selects one member of the slotwise rotation sequence.
    """
    parsed = _parse_puzzle(np.asarray(grid, dtype=int))
    return _build_output(parsed)


def _expected_output(
    shape: tuple[int, int],
    divider_cols: list[int],
    slot_spans: list[tuple[int, int]],
    anchor: tuple[int, int],
    rotations: list[np.ndarray],
    target_slot: int,
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    out[:, divider_cols] = DIVIDER_COLOR
    anchor_r, anchor_c = anchor
    for idx, ((c0, _c1), mask) in enumerate(zip(slot_spans, rotations)):
        color = REFERENCE_COLOR if idx == target_slot else TEMPLATE_COLOR
        _paste_mask(out, mask, anchor_r, c0 + anchor_c, color)
    return out


def _canonical_small_shape() -> np.ndarray:
    return np.array([[1, 1], [1, 0]], dtype=bool)


def _generate_attempt() -> np.ndarray | None:
    slot_width = int(np.random.choice([5, 6]))
    ref_width = int(np.random.randint(2, 5 if slot_width == 5 else 3))
    height = int(np.random.randint(6, 11))
    total_width = N_SLOTS * slot_width + N_SLOTS + ref_width
    if total_width > 30 or height > 30:
        return None

    base_small = _rotate_cw(_canonical_small_shape(), int(np.random.randint(4)))
    base_large = _scale_mask(base_small, SCALE_FACTOR)
    rotations = [_rotate_cw(base_large, k) for k in range(N_SLOTS)]

    if any(np.array_equal(rotations[i], rotations[j]) for i in range(N_SLOTS) for j in range(i + 1, N_SLOTS)):
        return None

    template_h, template_w = base_large.shape
    if template_h > height or template_w > slot_width:
        return None

    anchor_r = int(np.random.randint(0, height - template_h + 1))
    anchor_c = int(np.random.randint(0, slot_width - template_w + 1))
    ref_h, ref_w = base_small.shape
    if ref_w > ref_width or ref_h > height:
        return None
    ref_r = int(np.random.randint(0, height - ref_h + 1))
    ref_c = int(np.random.randint(0, ref_width - ref_w + 1))
    target_slot = int(np.random.randint(N_SLOTS))

    grid = np.full((height, total_width), BACKGROUND, dtype=int)
    divider_cols: list[int] = []
    slot_spans: list[tuple[int, int]] = []
    c = 0
    for _ in range(N_SLOTS):
        slot_spans.append((c, c + slot_width))
        c += slot_width
        divider_cols.append(c)
        grid[:, c] = DIVIDER_COLOR
        c += 1
    ref_span = (c, c + ref_width)

    _paste_mask(grid, rotations[0], anchor_r, slot_spans[0][0] + anchor_c, TEMPLATE_COLOR)
    ref_small = _rotate_cw(base_small, target_slot)
    _paste_mask(grid, ref_small, ref_r, ref_span[0] + ref_c, REFERENCE_COLOR)

    scaled_ref = _scale_mask(ref_small, SCALE_FACTOR)
    matches = [np.array_equal(rot, scaled_ref) for rot in rotations]
    if matches.count(True) != 1 or not matches[target_slot]:
        return None

    expected = _expected_output(
        shape=grid.shape,
        divider_cols=divider_cols,
        slot_spans=slot_spans,
        anchor=(anchor_r, anchor_c),
        rotations=rotations,
        target_slot=target_slot,
    )
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    if np.any(solved[:, ref_span[0] : ref_span[1]] != BACKGROUND):
        return None

    red_slots = 0
    for idx, (c0, c1) in enumerate(slot_spans):
        slot = solved[:, c0:c1]
        colors = set(np.unique(slot).tolist()) - {BACKGROUND}
        if idx == target_slot:
            if colors != {REFERENCE_COLOR}:
                return None
            red_slots += 1
        else:
            if colors != {TEMPLATE_COLOR}:
                return None
    if red_slots != 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
