import numpy as np


SPRITE_COLORS = np.array([1, 2, 3, 4], dtype=int)
CONTROL_COLORS = np.array([5, 6, 7, 8, 9], dtype=int)
SPRITE_SHAPES = ((3, 3), (4, 4))
TEMPLATE_SHAPES = ((3, 3),)
GEN_ATTEMPTS = 800
MASK_CACHE = {}
SEEN_OUTPUTS = set()


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in neighbors:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _is_vertical(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[:, ::-1]))


def _is_horizontal(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[::-1, :]))


def _is_tight(mask: np.ndarray) -> bool:
    h, w = mask.shape
    return (
        mask.any()
        and mask[0].any()
        and mask[h - 1].any()
        and mask[:, 0].any()
        and mask[:, w - 1].any()
    )


def _render_output(template_mask: np.ndarray, control: np.ndarray) -> np.ndarray:
    h, w = template_mask.shape
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(control[r, c])
            if color == 0:
                continue
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            block[template_mask] = color
    return out


def _build_mask_cache() -> None:
    if MASK_CACHE:
        return

    for h, w in SPRITE_SHAPES:
        vertical: list[np.ndarray] = []
        horizontal: list[np.ndarray] = []
        asymmetric: list[np.ndarray] = []
        total = h * w
        for bits in range(1, 1 << total):
            count = bits.bit_count()
            if count < 4 or count > 7:
                continue
            flat = np.fromiter(((bits >> i) & 1 for i in range(total)), dtype=int, count=total)
            mask = flat.reshape(h, w).astype(bool)
            if not _is_tight(mask):
                continue
            if len(_components(mask, connectivity=8)) != 1:
                continue
            vertical_ok = _is_vertical(mask)
            horizontal_ok = _is_horizontal(mask)
            if count <= 6 and vertical_ok:
                vertical.append(mask.copy())
            if count <= 6 and horizontal_ok:
                horizontal.append(mask.copy())
            if not vertical_ok and not horizontal_ok:
                asymmetric.append(mask.copy())

        if (h, w) in TEMPLATE_SHAPES:
            assert vertical, f"no vertical templates for {(h, w)}"
            assert horizontal, f"no horizontal templates for {(h, w)}"
        assert asymmetric, f"no asymmetric distractors for {(h, w)}"
        MASK_CACHE[(h, w)] = {
            "vertical": vertical,
            "horizontal": horizontal,
            "asymmetric": asymmetric,
        }


def _parse_split(grid: np.ndarray, split: int) -> tuple[np.ndarray, np.ndarray] | None:
    left = grid[:, :split]
    right = grid[:, split:]
    if not np.any(left) or not np.any(right):
        return None

    components = _components(left != 0, connectivity=8)
    if not (3 <= len(components) <= 4):
        return None

    symmetric_masks: list[np.ndarray] = []
    for coords in components:
        sprite = _mask_from_coords(coords)
        h, w = sprite.shape
        if (h, w) not in SPRITE_SHAPES:
            return None
        pixels = int(sprite.sum())
        if not (4 <= pixels <= 7):
            return None
        colors = np.unique(left[coords[:, 0], coords[:, 1]])
        if len(colors) != 1 or int(colors[0]) == 0:
            return None
        if _is_vertical(sprite) or _is_horizontal(sprite):
            symmetric_masks.append(sprite)

    if len(symmetric_masks) != 1:
        return None

    control_mask = right != 0
    if not control_mask.any():
        return None
    control = _crop_mask(right)
    template_mask = symmetric_masks[0]
    if control.shape != template_mask.shape:
        return None
    if not np.array_equal(control != 0, template_mask):
        return None

    nz = control[control != 0]
    if not (4 <= len(nz) <= 6):
        return None
    if len(np.unique(nz)) < 2:
        return None

    return template_mask, control


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parses: dict[tuple[tuple[int, int], bytes, tuple[int, int], bytes], tuple[np.ndarray, np.ndarray]] = {}
    for split in range(1, grid.shape[1]):
        parsed = _parse_split(grid, split)
        if parsed is None:
            continue
        template_mask, control = parsed
        sig = (template_mask.shape, template_mask.tobytes(), control.shape, control.tobytes())
        parses[sig] = (template_mask, control)
    assert len(parses) == 1, "expected exactly one valid sprite/control interpretation"
    return next(iter(parses.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the unique reflection-symmetric sprite as a stamp template, expand it into a self-stamped lattice, and recolor each stamped copy by a matching control cell.

    Input structure:
    - The grid has a black background with 3-4 small monochrome sprites on the left and one sparse control matrix on the right.
    - Each sprite has a tight 3x3 or 4x4 bounding box and 4-7 foreground pixels.
    - Exactly one left-side sprite is symmetric under a horizontal or vertical reflection.
    - The right-side control matrix has the same bounding-box shape as that symmetric sprite, with colors only where the template has foreground.

    Rule:
    - Test every left-side sprite after cropping to its tight bounding box and keep the unique one with horizontal or vertical reflection symmetry.
    - Treat that cropped sprite as both stencil and stamp: build an output of size (H*H) x (W*W), split into H x W blocks of size H x W.
    - For each non-black template cell at position (r, c), paste the whole template mask into the corresponding block.
    - Recolor every pasted template copy uniformly with the control-matrix color at the same index (r, c).
    - Omit all original input content; the output contains only the recolored stamped lattice.

    Key insight:
    - Symmetry does not identify the final output directly; it identifies which sprite becomes the stamp template, and that template geometry then fixes the lattice positions that the control matrix recolors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    template_mask, control = _parse_scene(g)
    return _render_output(template_mask, control)


def _sample_template() -> tuple[np.ndarray, str]:
    _build_mask_cache()
    shape = TEMPLATE_SHAPES[int(np.random.randint(len(TEMPLATE_SHAPES)))]
    symmetry = str(np.random.choice(["vertical", "horizontal"]))
    masks = MASK_CACHE[shape][symmetry]
    return masks[int(np.random.randint(len(masks)))].copy(), symmetry


def _sample_distractor(shape: tuple[int, int], target_count: int | None = None) -> np.ndarray:
    _build_mask_cache()
    candidates = MASK_CACHE[shape]["asymmetric"]
    if target_count is not None:
        filtered = [mask for mask in candidates if int(mask.sum()) == target_count]
        if filtered:
            candidates = filtered
    return candidates[int(np.random.randint(len(candidates)))].copy()


def _choose_distractors(template_mask: np.ndarray, total: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    template_shape = tuple(template_mask.shape)
    template_count = int(template_mask.sum())
    same_shape = _sample_distractor(template_shape, target_count=template_count)
    masks.append(same_shape)
    while len(masks) < total - 1:
        shape = SPRITE_SHAPES[int(np.random.randint(len(SPRITE_SHAPES)))]
        distractor = _sample_distractor(shape)
        masks.append(distractor)
    return masks


def _sample_control(template_mask: np.ndarray) -> np.ndarray:
    h, w = template_mask.shape
    control = np.zeros((h, w), dtype=int)
    coords = np.argwhere(template_mask)
    k = len(coords)
    color_count = int(np.random.randint(2, min(len(CONTROL_COLORS), k) + 1))
    palette = np.random.choice(CONTROL_COLORS, size=color_count, replace=False)
    assigned = np.empty(k, dtype=int)
    assigned[:color_count] = palette
    assigned[color_count:] = np.random.choice(palette, size=k - color_count, replace=True)
    np.random.shuffle(assigned)
    for (r, c), color in zip(coords, assigned):
        control[int(r), int(c)] = int(color)
    return control


def _can_place(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int], padding: int = 2) -> bool:
    h, w = shape
    hh, ww = occupied.shape
    if top < 1 or left < 1 or top + h >= hh or left + w >= ww:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(hh, top + h + padding)
    c1 = min(ww, left + w + padding)
    return not occupied[r0:r1, c0:c1].any()


def _paint_sprite(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = int(color)


def _place_sprites(grid_h: int, left_width: int, masks: list[np.ndarray]) -> list[tuple[int, int]]:
    occupied = np.zeros((grid_h, left_width), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * len(masks)
    order = np.random.permutation(len(masks))
    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        candidates = [(r, c) for r in range(1, grid_h - h) for c in range(1, left_width - w)]
        np.random.shuffle(candidates)
        for top, left in candidates:
            if _can_place(occupied, top, left, (h, w), padding=2):
                occupied[top : top + h, left : left + w] = True
                placements[idx] = (int(top), int(left))
                break
        else:
            raise ValueError("failed to place sprites")
    return placements


def _valid_full_scene(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    colors = np.unique(expected[expected != 0])
    if len(colors) < 2:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    template_mask, _symmetry = _sample_template()
    control = _sample_control(template_mask)
    expected = _render_output(template_mask, control)
    out_sig = (expected.shape, expected.tobytes())
    if out_sig in SEEN_OUTPUTS:
        return None

    sprite_count = int(np.random.randint(3, 5))
    distractors = _choose_distractors(template_mask, sprite_count)
    sprites = distractors + [template_mask]
    sprite_colors = np.random.choice(SPRITE_COLORS, size=sprite_count, replace=False)
    template_index = int(np.random.randint(sprite_count))
    sprites[template_index], sprites[-1] = sprites[-1], sprites[template_index]

    grid_h = int(np.random.randint(18, 27))
    grid_w = int(np.random.randint(20, 27))
    border_right = int(np.random.randint(1, 3))
    group_gap = int(np.random.randint(5, 8))
    ctl_h, ctl_w = control.shape
    ctl_left = grid_w - ctl_w - border_right
    left_width = ctl_left - group_gap
    if left_width < 9 or grid_h <= ctl_h + 2:
        return None

    try:
        placements = _place_sprites(grid_h, left_width, sprites)
    except ValueError:
        return None

    ctl_top = int(np.random.randint(1, grid_h - ctl_h))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, (mask, color, (top, left)) in enumerate(zip(sprites, sprite_colors, placements)):
        _paint_sprite(grid, top, left, mask, int(color))
    grid[ctl_top : ctl_top + ctl_h, ctl_left : ctl_left + ctl_w] = control

    # Semantic validations beyond simple closure.
    try:
        parsed_template, parsed_control = _parse_scene(grid)
    except AssertionError:
        return None
    if not np.array_equal(parsed_template, template_mask):
        return None
    if not np.array_equal(parsed_control, control):
        return None
    left_components = _components(grid[:, :ctl_left - 1] != 0, connectivity=8)
    if not (3 <= len(left_components) <= 4):
        return None
    symmetric_count = 0
    for coords in left_components:
        sprite = _mask_from_coords(coords)
        if _is_vertical(sprite) or _is_horizontal(sprite):
            symmetric_count += 1
    if symmetric_count != 1:
        return None
    if not _valid_full_scene(grid, expected):
        return None

    SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
