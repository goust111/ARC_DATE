from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import components_mask, crop_bbox, find_components, get_bg, paste


BG = 0
FRAME = 5
MARKER_PALETTE = (1, 2, 3, 4)
SPRITE_PALETTE = (6, 7, 8, 9)
FRAME_OUTER = 18
FRAME_SLOTS = ((4, 5), (7, 12), (11, 6), (13, 13))
AREA_TO_SHAPES = {
    1: ((1, 1),),
    3: ((1, 3), (3, 1)),
    5: ((1, 5),),
    9: ((3, 3),),
    15: ((3, 5), (5, 3)),
}
ORNAMENT_RECT = {2: (1, 2), 4: (2, 2), 6: (2, 3), 8: (2, 4)}
ROT_TO_HINT_AREA = {0: 4, 1: 1, 2: 2, 3: 3}
HINT_MASKS = {
    1: np.array([[1]], dtype=bool),
    2: np.array([[1, 1]], dtype=bool),
    3: np.array([[1, 1], [1, 0]], dtype=bool),
    4: np.array([[1, 1], [1, 1]], dtype=bool),
}
MAX_TRIES = 400


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate_cw(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -int(k))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    return crop_bbox(mask.astype(int), _bbox_of_mask(mask)).astype(bool)


def _swap_two_colors(patch: np.ndarray) -> np.ndarray:
    cols = sorted(int(v) for v in np.unique(patch) if int(v) != BG)
    assert len(cols) == 2
    a, b = cols
    out = patch.copy()
    out[patch == a] = b
    out[patch == b] = a
    return out


def _component_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    keep = np.zeros_like(patch, dtype=bool)
    keep[coords[:, 0] - r0, coords[:, 1] - c0] = True
    patch[~keep] = BG
    return patch


def _frame_and_cavities(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict], set[int]]:
    frame_comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[FRAME])
    assert frame_comps, "expected a gray frame"
    frame = max(frame_comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
    sub = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    black = sub == BG

    cavities = []
    for comp in components_mask(black, connectivity=4):
        rs = comp[:, 0]
        cs = comp[:, 1]
        if int(rs.min()) == 0 or int(cs.min()) == 0 or int(rs.max()) == sub.shape[0] - 1 or int(cs.max()) == sub.shape[1] - 1:
            continue
        lr0, lc0 = int(rs.min()), int(cs.min())
        lr1, lc1 = int(rs.max()), int(cs.max())
        center_r = int((lr0 + lr1) // 2)
        center_c = int((lc0 + lc1) // 2)
        marker_top = int(sub[0, center_c])
        marker_left = int(sub[center_r, 0])
        assert marker_top == marker_left and marker_top in MARKER_PALETTE
        local_mask = np.zeros((lr1 - lr0 + 1, lc1 - lc0 + 1), dtype=bool)
        local_mask[rs - lr0, cs - lc0] = True
        cavities.append(
            {
                "bbox": (fr0 + lr0, fc0 + lc0, fr0 + lr1, fc0 + lc1),
                "mask": local_mask,
                "area": int(comp.shape[0]),
                "marker_color": marker_top,
                "center": (fr0 + center_r, fc0 + center_c),
            }
        )
    cavities.sort(key=lambda item: (int(item["area"]), item["center"][0], item["center"][1]))
    colors = {int(item["marker_color"]) for item in cavities}
    assert 3 <= len(cavities) <= 4
    assert len(colors) == len(cavities)
    return (fr0, fc0, fr1, fc1), cavities, colors


def _collect_hints(grid: np.ndarray, marker_colors: set[int], frame_bbox: tuple[int, int, int, int]) -> dict[int, dict]:
    fr0, fc0, fr1, fc1 = frame_bbox
    hints: dict[int, dict] = {}
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=sorted(marker_colors)):
        coords = comp["coords"]
        on_frame_border = all(
            ((int(r) == fr0 and fc0 <= int(c) <= fc1) or (int(c) == fc0 and fr0 <= int(r) <= fr1))
            for r, c in coords
        )
        if on_frame_border:
            continue
        color = int(comp["color"])
        assert color not in hints
        hints[color] = {"area": int(comp["area"]), "coords": coords}
    assert set(hints) == set(marker_colors)
    return hints


def _collect_sprites(grid: np.ndarray, marker_colors: set[int]) -> list[dict]:
    sprite_colors = sorted(set(int(v) for v in np.unique(grid)) - {BG, FRAME} - set(marker_colors))
    assert sprite_colors
    sprites = []
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        colors=sprite_colors,
    ):
        patch = _component_patch(grid, comp["coords"])
        colors = sorted(int(v) for v in np.unique(patch) if int(v) != BG)
        assert len(colors) == 2
        counts = {color: int(np.sum(patch == color)) for color in colors}
        sprites.append({"patch": patch, "colors": colors, "counts": counts, "coords": comp["coords"]})
    assert 3 <= len(sprites) <= 4
    return sprites


def _match_sprites_to_cavities(sprites: list[dict], cavities: list[dict]) -> list[tuple[dict, dict, int]]:
    areas = [int(cavity["area"]) for cavity in cavities]
    matches = []
    used = set()
    for cavity in cavities:
        target = int(cavity["area"])
        cand = []
        for idx, sprite in enumerate(sprites):
            for color, count in sprite["counts"].items():
                if int(count) == target:
                    cand.append((idx, int(color)))
        assert len(cand) == 1, "area matching must be unique"
        idx, fill_color = cand[0]
        assert idx not in used
        used.add(idx)
        matches.append((sprites[idx], cavity, fill_color))
    assert len(used) == len(sprites) == len(cavities)
    assert sorted(areas) == sorted(int(sprite["counts"][fill]) for sprite, _cavity, fill in matches)
    return matches


def _place_transformed_sprite(out: np.ndarray, cavity: dict, patch: np.ndarray, fill_color: int, rot_k: int) -> None:
    fill_mask = patch == int(fill_color)
    rot_patch = _rotate_cw(patch, rot_k)
    rot_fill = _rotate_cw(fill_mask.astype(int), rot_k).astype(bool)
    cavity_mask = np.asarray(cavity["mask"], dtype=bool)

    if np.array_equal(_crop_mask(rot_fill), cavity_mask):
        final_patch = rot_patch
        final_fill = rot_fill
    else:
        flip_patch = np.fliplr(rot_patch)
        flip_fill = np.fliplr(rot_fill)
        assert np.array_equal(_crop_mask(flip_fill), cavity_mask), "sprite must fit after optional flip"
        final_patch = _swap_two_colors(flip_patch)
        final_fill = flip_fill

    pr0, pc0, _pr1, _pc1 = _bbox_of_mask(final_fill)
    cr0, cc0, _cr1, _cc1 = cavity["bbox"]
    paste(out, final_patch, cr0 - pr0, cc0 - pc0, transparent=BG)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each scattered two-color sprite into the frame cavity whose area matches one of its color regions, using border-marker colors to select the rotation hint.

    Input structure:
    - Black background surrounds one large gray rectangular frame on the right, with several black rectangular cavities inside it.
    - A colored pixel on the frame top border and a same-colored pixel on the left border cross at each cavity center, giving that cavity a unique marker color.
    - The left side contains 3-4 disconnected two-color sprites and 3-4 small monochrome hint sprites whose colors reuse the frame marker colors.

    Rule:
    - For each cavity, use the top/left border color intersection to read its marker color.
    - Match that cavity to the unique two-color sprite whose one color-region area equals the cavity area; that color is the fill region.
    - Find the hint sprite of the same marker color, let its pixel count be P, rotate the whole matched sprite clockwise by (P mod 4) quarter-turns, and align the fill-region bbox top-left to the cavity bbox top-left.
    - If the rotated fill region still does not match the cavity shape, horizontally flip the sprite and swap its two colors, then place it; erase all original scattered sprites and hints.

    Key insight:
    - Area identifies which sprite belongs to which cavity, the border-marker color identifies which hint controls that cavity, and the hint pixel count determines the final sprite orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert int(bg) == BG

    frame_bbox, cavities, marker_colors = _frame_and_cavities(g)
    hints = _collect_hints(g, marker_colors, frame_bbox)
    sprites = _collect_sprites(g, marker_colors)
    matches = _match_sprites_to_cavities(sprites, cavities)

    out = g.copy()
    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BG
    for hint in hints.values():
        out[hint["coords"][:, 0], hint["coords"][:, 1]] = BG

    for sprite, cavity, fill_color in matches:
        marker_color = int(cavity["marker_color"])
        rot_k = int(hints[marker_color]["area"] % 4)
        _place_transformed_sprite(out, cavity, sprite["patch"], fill_color, rot_k)

    return out


def _sprite_patch(fill_hw: tuple[int, int], fill_color: int, other_color: int, ornament_area: int) -> tuple[np.ndarray, np.ndarray]:
    fh, fw = map(int, fill_hw)
    oh, ow = ORNAMENT_RECT[int(ornament_area)]
    fill_r0 = 2
    patch_h = fill_r0 + fh
    patch_w = max(fw + 3, ow + 2)
    fill_c0 = random.randint(1, patch_w - fw - 1)
    low = max(0, fill_c0 - ow + 1)
    high = min(patch_w - ow, fill_c0 + fw - 1)
    orn_c0 = random.randint(low, high)
    orn_r0 = fill_r0 - oh

    patch = np.zeros((patch_h, patch_w), dtype=int)
    fill_mask = np.zeros_like(patch, dtype=bool)
    patch[fill_r0 : fill_r0 + fh, fill_c0 : fill_c0 + fw] = int(fill_color)
    fill_mask[fill_r0 : fill_r0 + fh, fill_c0 : fill_c0 + fw] = True
    patch[orn_r0 : orn_r0 + oh, orn_c0 : orn_c0 + ow] = int(other_color)
    return patch, fill_mask


def _coords_from_patch(patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    return coords + np.array([int(r0), int(c0)])


def _chebyshev_touches(a: np.ndarray, b: np.ndarray, gap: int = 1) -> bool:
    dr = np.abs(a[:, None, 0] - b[None, :, 0])
    dc = np.abs(a[:, None, 1] - b[None, :, 1])
    return bool(np.any((dr <= gap) & (dc <= gap)))


def _random_place(grid: np.ndarray, patch: np.ndarray, row_range: range, col_range: range, existing: list[np.ndarray]) -> tuple[int, int] | None:
    candidates = [(r, c) for r in row_range for c in col_range]
    random.shuffle(candidates)
    ph, pw = patch.shape
    h, w = grid.shape
    for r0, c0 in candidates:
        if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
            continue
        coords = _coords_from_patch(patch, r0, c0)
        if np.any(grid[coords[:, 0], coords[:, 1]] != BG):
            continue
        if any(_chebyshev_touches(coords, other, gap=1) for other in existing):
            continue
        return int(r0), int(c0)
    return None


def _build_frame(cavity_specs: list[dict]) -> np.ndarray:
    frame = np.full((FRAME_OUTER, FRAME_OUTER), FRAME, dtype=int)
    for spec in cavity_specs:
        r0, c0, r1, c1 = spec["local_bbox"]
        frame[r0 : r1 + 1, c0 : c1 + 1] = BG
        frame[0, spec["center"][1]] = int(spec["marker_color"])
        frame[spec["center"][0], 0] = int(spec["marker_color"])
    return frame


def _apply_expected(scene: dict) -> np.ndarray:
    out = scene["input"].copy()
    for sprite in scene["sprites"]:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BG
    for hint in scene["hints"]:
        out[hint["coords"][:, 0], hint["coords"][:, 1]] = BG
    for cavity, sprite in zip(sorted(scene["cavities"], key=lambda x: x["area"]), sorted(scene["sprites"], key=lambda x: x["area"])):
        _place_transformed_sprite(out, cavity, sprite["patch"], sprite["fill_color"], sprite["rot_k"])
    return out


def _semantic_checks(scene: dict, expected: np.ndarray) -> None:
    cavities = sorted(scene["cavities"], key=lambda x: x["area"])
    sprites = sorted(scene["sprites"], key=lambda x: x["area"])
    assert len(cavities) == len(sprites) == len(scene["hints"])
    assert len({int(c["area"]) for c in cavities}) == len(cavities)
    assert len({int(s["rot_k"]) for s in sprites}) >= 2

    cavity_areas = {int(c["area"]) for c in cavities}
    for sprite in sprites:
        counts = sprite["color_counts"]
        hits = [color for color, count in counts.items() if int(count) in cavity_areas]
        assert hits == [int(sprite["fill_color"])]

    flex_idx = {int(sprite["area"]): idx for idx, sprite in enumerate(sprites)}
    assert 1 in flex_idx and 9 in flex_idx
    a, b = flex_idx[1], flex_idx[9]
    wrong = scene["input"].copy()
    for sprite in scene["sprites"]:
        wrong[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BG
    for hint in scene["hints"]:
        wrong[hint["coords"][:, 0], hint["coords"][:, 1]] = BG
    bad_rot = {sprite["marker_color"]: sprite["rot_k"] for sprite in sprites}
    bad_rot[sprites[a]["marker_color"]], bad_rot[sprites[b]["marker_color"]] = bad_rot[sprites[b]["marker_color"]], bad_rot[sprites[a]["marker_color"]]
    for cavity, sprite in zip(cavities, sprites):
        _place_transformed_sprite(wrong, cavity, sprite["patch"], sprite["fill_color"], bad_rot[int(cavity["marker_color"])])
    assert not np.array_equal(wrong, expected), "marker-driven hint selection must matter"


def _make_scene() -> dict | None:
    n = random.randint(3, 4)
    grid_h = random.randint(24, 28)
    grid_w = random.randint(27, 30)
    right_margin = random.randint(1, 2)
    frame_r0 = random.randint(2, grid_h - FRAME_OUTER - 2)
    frame_c0 = grid_w - FRAME_OUTER - right_margin
    if frame_c0 < 10:
        return None

    optional_areas = [3, 5, 15]
    areas = [1, 9] + random.sample(optional_areas, n - 2)
    slot_ids = random.sample(range(len(FRAME_SLOTS)), n)
    marker_colors = random.sample(MARKER_PALETTE, n)
    rot_values = random.sample(range(4), n)
    ornament_areas = random.sample(sorted(ORNAMENT_RECT), n)
    color_pairs = random.sample(
        [(a, b) for a in SPRITE_PALETTE for b in SPRITE_PALETTE if a != b],
        n,
    )

    cavity_specs = []
    sprite_specs = []
    for area, slot_id, marker_color, rot_k, ornament_area, (fill_color, other_color) in zip(
        areas, slot_ids, marker_colors, rot_values, ornament_areas, color_pairs
    ):
        shape = random.choice(AREA_TO_SHAPES[int(area)])
        ch, cw = shape
        center = FRAME_SLOTS[int(slot_id)]
        local_r0 = int(center[0] - ch // 2)
        local_c0 = int(center[1] - cw // 2)
        local_r1 = local_r0 + ch - 1
        local_c1 = local_c0 + cw - 1
        if local_r0 <= 1 or local_c0 <= 1 or local_r1 >= FRAME_OUTER - 2 or local_c1 >= FRAME_OUTER - 2:
            return None

        pre_shape = (ch, cw) if rot_k % 2 == 0 else (cw, ch)
        patch, fill_mask = _sprite_patch(pre_shape, fill_color, other_color, ornament_area)
        color_counts = {fill_color: int(np.sum(patch == fill_color)), other_color: int(np.sum(patch == other_color))}
        if color_counts[fill_color] != int(area) or color_counts[other_color] in areas:
            return None

        cavity_specs.append(
            {
                "area": int(area),
                "local_bbox": (local_r0, local_c0, local_r1, local_c1),
                "bbox": (frame_r0 + local_r0, frame_c0 + local_c0, frame_r0 + local_r1, frame_c0 + local_c1),
                "mask": np.ones((ch, cw), dtype=bool),
                "marker_color": int(marker_color),
                "center": center,
            }
        )
        sprite_specs.append(
            {
                "patch": patch,
                "fill_mask": fill_mask,
                "fill_color": int(fill_color),
                "other_color": int(other_color),
                "area": int(area),
                "marker_color": int(marker_color),
                "rot_k": int(rot_k),
                "color_counts": color_counts,
            }
        )

    coords_by_area = {spec["area"]: spec["local_bbox"] for spec in cavity_specs}
    if len(coords_by_area) != len(cavity_specs):
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    paste(grid, _build_frame(cavity_specs), frame_r0, frame_c0)

    occupied: list[np.ndarray] = []
    hints = []
    sprites = []
    left_max_col = frame_c0 - 1
    if left_max_col < 7:
        return None

    for spec in sprite_specs:
        patch = spec["patch"]
        pos = _random_place(
            grid,
            patch,
            range(1, grid_h - patch.shape[0]),
            range(1, left_max_col - patch.shape[1]),
            occupied,
        )
        if pos is None:
            return None
        r0, c0 = pos
        paste(grid, patch, r0, c0, transparent=BG)
        coords = _coords_from_patch(patch, r0, c0)
        spec["coords"] = coords
        occupied.append(coords)
        sprites.append(spec)

    for spec in sprite_specs:
        hint_area = int(ROT_TO_HINT_AREA[int(spec["rot_k"])])
        mask = HINT_MASKS[hint_area]
        patch = np.where(mask, int(spec["marker_color"]), BG)
        pos = _random_place(
            grid,
            patch,
            range(1, grid_h - patch.shape[0]),
            range(1, left_max_col - patch.shape[1]),
            occupied,
        )
        if pos is None:
            return None
        r0, c0 = pos
        paste(grid, patch, r0, c0, transparent=BG)
        coords = _coords_from_patch(patch, r0, c0)
        occupied.append(coords)
        hints.append({"color": int(spec["marker_color"]), "area": hint_area, "coords": coords})

    scene = {"input": grid, "cavities": cavity_specs, "sprites": sprites, "hints": hints}
    expected = _apply_expected(scene)

    out_coords = []
    temp = np.zeros_like(grid, dtype=bool)
    for cavity, sprite in zip(sorted(cavity_specs, key=lambda x: x["area"]), sorted(sprites, key=lambda x: x["area"])):
        work = np.zeros_like(grid)
        _place_transformed_sprite(work, cavity, sprite["patch"], sprite["fill_color"], sprite["rot_k"])
        coords = np.argwhere(work != BG)
        if coords.size == 0:
            return None
        if out_coords and any(_chebyshev_touches(coords, other, gap=0) for other in out_coords):
            return None
        out_coords.append(coords)
        temp[coords[:, 0], coords[:, 1]] = True

    frame_bbox, cavities, marker_set = _frame_and_cavities(grid)
    hints_found = _collect_hints(grid, marker_set, frame_bbox)
    sprites_found = _collect_sprites(grid, marker_set)
    if len(cavities) != n or len(hints_found) != n or len(sprites_found) != n:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    _semantic_checks(scene, expected)
    return scene


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        scene = _make_scene()
        if scene is not None:
            return scene["input"]
    raise RuntimeError("failed to generate a valid nested-triple-encoding scene")
