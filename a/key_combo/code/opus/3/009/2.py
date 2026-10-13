from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

try:
    from arc_utils import (
        Color,
        bbox_coords,
        components_mask,
        crop_bbox,
        find_components,
        paste,
        random_free_location,
    )
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_anchor_embed",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    components_mask = _ARC_UTILS.components_mask
    crop_bbox = _ARC_UTILS.crop_bbox
    find_components = _ARC_UTILS.find_components
    paste = _ARC_UTILS.paste
    random_free_location = _ARC_UTILS.random_free_location


BODY_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.TEAL,
    Color.MAROON,
]
SIZE_OPTIONS = [
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 5),
]
MAX_GENERATE_ATTEMPTS = 400


def _size_signature(h: int, w: int) -> tuple[int, int]:
    return tuple(sorted((int(h), int(w))))


def _anchor_offset(patch: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(patch == Color.ORANGE)
    assert coords.shape == (1, 2)
    return int(coords[0, 0]), int(coords[0, 1])


def _rotations_that_fit(
    patch: np.ndarray, target_shape: tuple[int, int]
) -> list[tuple[int, np.ndarray]]:
    target_shape = tuple(map(int, target_shape))
    out: list[tuple[int, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(patch, k)
        if tuple(rotated.shape) == target_shape:
            out.append((k, rotated))
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any() and len(components_mask(mask, connectivity=4)) == 1)


def _edge_anchor_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        on_edge = r in (0, h - 1) or c in (0, w - 1)
        is_corner = r in (0, h - 1) and c in (0, w - 1)
        if on_edge and not is_corner:
            out.append((r, c))
    return out


def _largest_grey_template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
        return_format="dict",
    )
    assert comps
    comps.sort(key=lambda comp: int(comp["area"]), reverse=True)
    if len(comps) > 1:
        assert int(comps[0]["area"]) > int(comps[1]["area"])
    return tuple(map(int, comps[0]["bbox"]))


def _extract_cavities(
    grid: np.ndarray, template_bbox: tuple[int, int, int, int]
) -> list[dict]:
    tr0, tc0, _tr1, _tc1 = template_bbox
    template = crop_bbox(grid, template_bbox)
    mask = template != Color.GREY
    cavities: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = template[r0 : r1 + 1, c0 : c1 + 1]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert int(coords.shape[0]) == height * width
        assert np.all((patch == Color.BLACK) | (patch == Color.ORANGE))
        anchor_rel = _anchor_offset(patch)
        cavities.append(
            {
                "r": int(tr0 + r0),
                "c": int(tc0 + c0),
                "h": height,
                "w": width,
                "anchor_rel": anchor_rel,
            }
        )
    cavities.sort(key=lambda cavity: (cavity["r"], cavity["c"]))
    return cavities


def _extract_sprites(
    grid: np.ndarray, template_bbox: tuple[int, int, int, int]
) -> list[dict]:
    r0, c0, r1, c1 = template_bbox
    masked = np.array(grid, copy=True)
    masked[r0 : r1 + 1, c0 : c1 + 1] = Color.BLACK
    comps = find_components(
        masked,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    sprites: list[dict] = []
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(grid, bbox)
        colors = {int(v) for v in patch.flatten() if int(v) != Color.BLACK}
        assert Color.GREY not in colors
        assert Color.ORANGE in colors
        assert 2 <= len(colors) <= 3
        area = int((patch != Color.BLACK).sum())
        assert 8 <= area <= 20
        sprites.append(
            {
                "bbox": bbox,
                "patch": patch,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "area": area,
            }
        )
    sprites.sort(key=lambda sprite: (sprite["bbox"][0], sprite["bbox"][1]))
    return sprites


def _size_matches(cavity: dict, sprite: dict) -> bool:
    return _size_signature(cavity["h"], cavity["w"]) == _size_signature(
        sprite["h"], sprite["w"]
    )


def _unique_size_matching(cavities: list[dict], sprites: list[dict]) -> list[int]:
    order = sorted(
        range(len(cavities)),
        key=lambda idx: sum(_size_matches(cavities[idx], sprite) for sprite in sprites),
    )
    current = [-1] * len(cavities)
    used = [False] * len(sprites)
    solutions: list[list[int]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(current.copy())
            return
        cavity_idx = order[pos]
        for sprite_idx, sprite in enumerate(sprites):
            if used[sprite_idx] or not _size_matches(cavities[cavity_idx], sprite):
                continue
            used[sprite_idx] = True
            current[cavity_idx] = sprite_idx
            dfs(pos + 1)
            current[cavity_idx] = -1
            used[sprite_idx] = False

    dfs(0)
    assert len(solutions) == 1
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each anchored sprite until its anchor aligns with the matching cavity anchor, then embed it into the grey template.

    Input structure:
    - The background is black, with one large grey template region containing several rectangular black cavities.
    - Each cavity contains exactly one orange anchor pixel on its inner edge, and each outside multicolor sprite also contains exactly one orange anchor pixel.
    - Every sprite is a 4-connected non-grey object whose bounding box matches exactly one cavity up to rotation.

    Rule:
    - Find the grey template as the largest grey connected component and inspect the non-grey rectangles inside its bounding box as cavities.
    - Pair each outside sprite to the unique cavity with the same bounding-box size up to swapping height and width.
    - For each matched pair, test the four rotations of the sprite and choose the unique one whose rotated orange anchor offset equals the cavity anchor offset.
    - Paste the rotated sprite patch into the cavity while leaving the rest of the grid unchanged.

    Key insight:
    - Size identifies which sprite belongs to which cavity, but the orange anchor position is what disambiguates the rotation within that size-compatible pair.
    """
    g = np.asarray(grid, dtype=int)
    template_bbox = _largest_grey_template_bbox(g)
    cavities = _extract_cavities(g, template_bbox)
    sprites = _extract_sprites(g, template_bbox)
    assert cavities and len(cavities) == len(sprites)

    matching = _unique_size_matching(cavities, sprites)
    out = np.array(g, copy=True)
    for cavity_idx, sprite_idx in enumerate(matching):
        cavity = cavities[cavity_idx]
        sprite = sprites[sprite_idx]
        fits: list[np.ndarray] = []
        for _k, rotated in _rotations_that_fit(sprite["patch"], (cavity["h"], cavity["w"])):
            if _anchor_offset(rotated) == cavity["anchor_rel"]:
                fits.append(rotated)
        assert len(fits) == 1
        paste(out, fits[0], cavity["r"], cavity["c"])
    return out


def _generate_mask(h: int, w: int) -> np.ndarray | None:
    area = int(h * w)
    low = max(8, int(np.ceil(area * 0.55)))
    high = min(20, area)
    if low > high:
        low = high
    for _ in range(80):
        target = random.randint(low, high)
        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > target:
            removable: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                candidate = mask.copy()
                candidate[int(r), int(c)] = False
                if not _touches_all_sides(candidate):
                    continue
                if not _is_connected(candidate):
                    continue
                removable.append((int(r), int(c)))
            if not removable:
                break
            rr, cc = random.choice(removable)
            mask[rr, cc] = False
        if (
            8 <= int(mask.sum()) <= 20
            and _touches_all_sides(mask)
            and _is_connected(mask)
            and _edge_anchor_candidates(mask)
        ):
            return mask
    return None


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    occupied = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    anchor = random.choice(_edge_anchor_candidates(mask))
    patch[anchor] = Color.ORANGE
    body_coords = [coord for coord in occupied if coord != anchor]
    body_color_count = random.choice([1, 2, 2])
    body_colors = random.sample(BODY_COLORS, body_color_count)
    if body_color_count == 1:
        for r, c in body_coords:
            patch[r, c] = body_colors[0]
        return patch

    seed_a, seed_b = random.sample(body_coords, 2)
    for r, c in body_coords:
        da = abs(r - seed_a[0]) + abs(c - seed_a[1])
        db = abs(r - seed_b[0]) + abs(c - seed_b[1])
        if da == db:
            patch[r, c] = random.choice(body_colors)
        elif da < db:
            patch[r, c] = body_colors[0]
        else:
            patch[r, c] = body_colors[1]
    patch[seed_a] = body_colors[0]
    patch[seed_b] = body_colors[1]
    return patch


def _generate_sprite_patch(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        mask = _generate_mask(h, w)
        if mask is None:
            continue
        patch = _colorize_mask(mask)
        colors = {int(v) for v in patch.flatten() if int(v) != Color.BLACK}
        if Color.ORANGE in colors and 2 <= len(colors) <= 3:
            return patch
    return None


def _choose_size_triplet() -> list[tuple[int, int]]:
    square_options = [size for size in SIZE_OPTIONS if size[0] == size[1]]
    nonsquare_options = [size for size in SIZE_OPTIONS if size[0] != size[1]]
    square_count = random.choice([0, 1, 1, 2])
    chosen = random.sample(square_options, square_count) + random.sample(
        nonsquare_options, 3 - square_count
    )
    random.shuffle(chosen)
    return chosen


def _place_cavity_rectangles(
    template_h: int, template_w: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    blocked = np.zeros((template_h, template_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    for idx in order:
        h, w = sizes[idx]
        candidates: list[tuple[int, int]] = []
        for r in range(1, template_h - h):
            for c in range(1, template_w - w):
                rr0 = max(0, r - 1)
                cc0 = max(0, c - 1)
                rr1 = min(template_h, r + h + 1)
                cc1 = min(template_w, c + w + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        positions[idx] = (r, c)
        blocked[max(0, r - 1) : min(template_h, r + h + 1), max(0, c - 1) : min(template_w, c + w + 1)] = True
    return [pos for pos in positions if pos is not None]


def _plan_sprite_for_cavity(cavity_shape: tuple[int, int]) -> dict | None:
    ch, cw = map(int, cavity_shape)
    sprite_shape = random.choice([(ch, cw), (cw, ch)]) if ch != cw else (ch, cw)
    patch = _generate_sprite_patch(*sprite_shape)
    if patch is None:
        return None
    fit_variants = []
    for k, rotated in _rotations_that_fit(patch, (ch, cw)):
        fit_variants.append((k, rotated, _anchor_offset(rotated)))
    if len(fit_variants) < 2:
        return None
    anchor_positions = [item[2] for item in fit_variants]
    if len(set(anchor_positions)) != len(anchor_positions):
        return None
    _chosen_k, filled_patch, anchor_rel = random.choice(fit_variants)
    return {
        "display_patch": patch,
        "filled_patch": filled_patch,
        "anchor_rel": anchor_rel,
    }


def _semantic_validate(inp: np.ndarray, expected: np.ndarray) -> None:
    template_bbox = _largest_grey_template_bbox(inp)
    cavities = _extract_cavities(inp, template_bbox)
    sprites = _extract_sprites(inp, template_bbox)

    assert len(cavities) == 3
    assert len(sprites) == 3
    assert any(cavity["h"] != cavity["w"] for cavity in cavities)
    assert np.any(inp != expected)

    matching = _unique_size_matching(cavities, sprites)
    for cavity_idx, sprite_idx in enumerate(matching):
        cavity = cavities[cavity_idx]
        sprite = sprites[sprite_idx]
        fit_rotations = _rotations_that_fit(sprite["patch"], (cavity["h"], cavity["w"]))
        assert len(fit_rotations) >= 2
        anchor_matching = [
            k
            for k, rotated in fit_rotations
            if _anchor_offset(rotated) == cavity["anchor_rel"]
        ]
        assert len(anchor_matching) == 1

    solved = main(inp)
    assert np.array_equal(solved, expected)


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(20, 26)
    grid_w = random.randint(20, 26)
    template_h = random.randint(10, 15)
    template_w = random.randint(10, 15)
    if template_h >= grid_h - 3 or template_w >= grid_w - 3:
        return None

    sizes = _choose_size_triplet()
    max_h = max(h for h, _w in sizes)
    max_w = max(w for _h, w in sizes)
    if template_h < max_h + 3 or template_w < max_w + 3:
        return None

    cavity_positions = _place_cavity_rectangles(template_h, template_w, sizes)
    if cavity_positions is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    row_slack = grid_h - template_h - 2
    col_slack = grid_w - template_w - 2
    template_r = 1 + random.randint(0, max(0, row_slack))
    template_c = 1 + random.randint(0, max(0, col_slack))
    grid[template_r : template_r + template_h, template_c : template_c + template_w] = Color.GREY

    sprite_plans = []
    for size in sizes:
        plan = _plan_sprite_for_cavity(size)
        if plan is None:
            return None
        sprite_plans.append(plan)

    filled = grid.copy()
    for (cavity_r, cavity_c), size, plan in zip(cavity_positions, sizes, sprite_plans):
        ch, cw = size
        cavity = np.zeros((ch, cw), dtype=int)
        cavity[plan["anchor_rel"]] = Color.ORANGE
        paste(grid, cavity, template_r + cavity_r, template_c + cavity_c)
        paste(filled, plan["filled_patch"], template_r + cavity_r, template_c + cavity_c)

    occupancy = np.zeros_like(grid)
    occupancy[
        template_r : template_r + template_h,
        template_c : template_c + template_w,
    ] = Color.MAROON

    sprite_order = list(range(3))
    random.shuffle(sprite_order)
    for idx in sprite_order:
        patch = sprite_plans[idx]["display_patch"]
        blocker = np.full(patch.shape, Color.BLUE, dtype=int)
        try:
            loc = random_free_location(
                occupancy,
                blocker,
                transparent=Color.BLACK,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, blocker, loc[0], loc[1])
        paste(grid, patch, loc[0], loc[1])
        paste(filled, patch, loc[0], loc[1])

    _semantic_validate(grid, filled)
    return grid, filled


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid anchored-rotation cavity instance")
