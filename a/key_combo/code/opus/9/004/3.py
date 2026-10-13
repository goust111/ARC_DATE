from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, is_contiguous, neighbors4


SPRITE_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW)
DOMINO_COLOR = Color.RED
MIRROR_COLOR = Color.ORANGE
BG_COLOR = Color.BLACK

GRID_MIN = 12
GRID_MAX = 18
GROUP_COUNT_CHOICES = (2, 3)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
PADDING = 1


def _component_coords_set(comp: dict) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in comp["coords"]}


def _is_domino(comp: dict) -> bool:
    if int(comp["color"]) != DOMINO_COLOR or int(comp["area"]) != 2:
        return False
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (r0 == r1 and c1 == c0 + 1) or (c0 == c1 and r1 == r0 + 1)


def _domino_orientation(comp: dict) -> str:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if r0 == r1 and c1 == c0 + 1:
        return "horizontal"
    if c0 == c1 and r1 == r0 + 1:
        return "vertical"
    raise AssertionError("red marker must be a horizontal or vertical domino")


def _touches_sprite(sprite: dict, domino: dict) -> bool:
    sprite_cells = _component_coords_set(sprite)
    for r, c in domino["coords"]:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in sprite_cells:
                return True
    return False


def _pair_sprites_and_dominos(grid: np.ndarray) -> list[tuple[dict, dict]]:
    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    sprites = [comp for comp in components if int(comp["color"]) not in {bg, DOMINO_COLOR}]
    dominos = [comp for comp in components if int(comp["color"]) == DOMINO_COLOR]

    assert 2 <= len(sprites) <= 3, "expected two or three colored sprites"
    assert len(dominos) == len(sprites), "expected one red domino per sprite"
    assert {int(comp["color"]) for comp in sprites}.issubset(set(SPRITE_COLORS))
    assert len({int(comp["color"]) for comp in sprites}) == len(sprites), "sprite colors must be distinct"
    assert all(_is_domino(comp) for comp in dominos), "every red component must be a domino"

    pairings: list[tuple[dict, dict]] = []
    used_dominos: set[int] = set()
    for sprite in sorted(sprites, key=lambda comp: tuple(map(int, comp["bbox"]))):
        candidates = [idx for idx, domino in enumerate(dominos) if _touches_sprite(sprite, domino)]
        assert len(candidates) == 1, "each sprite must touch exactly one red domino"
        used_dominos.add(candidates[0])
        pairings.append((sprite, dominos[candidates[0]]))

    assert len(used_dominos) == len(dominos), "each domino must belong to exactly one sprite"
    return pairings


def _reflected_coords(sprite: dict, domino: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, domino["bbox"])
    coords: list[tuple[int, int]] = []
    if _domino_orientation(domino) == "horizontal":
        axis_sum = c0 + c1
        for r, c in sprite["coords"]:
            coords.append((int(r), int(axis_sum - int(c))))
    else:
        axis_sum = r0 + r1
        for r, c in sprite["coords"]:
            coords.append((int(axis_sum - int(r)), int(c)))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored sprite across the mirror axis encoded by its adjacent red domino marker.

    Input structure:
    - A black-background grid contains two or three distinct monochrome 4-connected sprites in blue, green, or yellow.
    - Each sprite touches exactly one red 2-pixel domino marker by a shared edge, and different sprite-marker groups stay separated.
    - A red domino is either horizontal or vertical, and its position keeps the sprite's reflected copy fully inside the grid without overlaps.

    Rule:
    - Find the unique red domino adjacent to each sprite.
    - A horizontal domino defines a vertical mirror axis through the midpoint of the domino's two columns; a vertical domino defines a horizontal mirror axis through the midpoint of its two rows.
    - Reflect every sprite pixel across that axis, draw the reflected copy in orange, erase all red domino pixels to black, and keep the original sprite unchanged.

    Key insight:
    - The domino orientation does not indicate the reflection direction directly; it encodes the perpendicular mirror axis that determines where the orange copy appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    pairings = _pair_sprites_and_dominos(g)

    all_sprite_cells = set()
    all_domino_cells = set()
    for sprite, domino in pairings:
        all_sprite_cells |= _component_coords_set(sprite)
        all_domino_cells |= _component_coords_set(domino)

    out = g.copy()
    out[out == DOMINO_COLOR] = bg
    painted_mirrors: set[tuple[int, int]] = set()
    H, W = map(int, g.shape)

    for sprite, domino in pairings:
        sprite_cells = _component_coords_set(sprite)
        mirror_cells = _reflected_coords(sprite, domino)
        mirror_set = set(mirror_cells)

        assert len(mirror_set) == int(sprite["area"]), "reflection must preserve area"
        assert all(0 <= r < H and 0 <= c < W for r, c in mirror_cells), "mirror copy must stay in bounds"
        assert not (mirror_set & all_domino_cells), "mirror copy must not overlap any domino"
        assert not (mirror_set & (all_sprite_cells - sprite_cells)), "mirror copy must not overlap another sprite"
        assert not (mirror_set & painted_mirrors), "different mirror copies must stay disjoint"

        painted_mirrors |= mirror_set
        for r, c in mirror_cells:
            out[r, c] = MIRROR_COLOR

    return out


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _build_sprite_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for h, w in ((2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)):
        total = h * w
        for area in range(3, min(7, total) + 1):
            for cells in itertools.combinations(range(total), area):
                mask = np.zeros((h, w), dtype=bool)
                for cell in cells:
                    mask[cell // w, cell % w] = True
                if not _touches_all_bbox_sides(mask):
                    continue
                if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
                    continue
                if not is_contiguous(mask.astype(int), background=0, connectivity=4):
                    continue
                masks.append(mask)
    assert masks, "expected at least one valid sprite mask"
    return masks


SPRITE_MASKS = _build_sprite_masks()


def _paint_coords(patch: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        patch[int(r), int(c)] = int(color)


def _make_group(mask: np.ndarray, color: int, mode: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, mask.shape)
    sprite_local = [tuple(map(int, coord)) for coord in np.argwhere(mask)]

    if mode == "lr":
        patch_h, patch_w = h, 2 * w + 2
        domino_left = w
        sprite_r = 0
        sprite_c = w + 2 if side == "left" else 0
        boundary_col = 0 if side == "left" else w - 1
        contact_rows = [int(r) for r in np.where(mask[:, boundary_col])[0]]
        domino_r = random.choice(contact_rows)
        domino_cells = [(domino_r, domino_left), (domino_r, domino_left + 1)]
    else:
        patch_h, patch_w = 2 * h + 2, w
        domino_top = h
        sprite_r = h + 2 if side == "up" else 0
        sprite_c = 0
        boundary_row = 0 if side == "up" else h - 1
        contact_cols = [int(c) for c in np.where(mask[boundary_row])[0]]
        domino_r = domino_top
        domino_c = random.choice(contact_cols)
        domino_cells = [(domino_r, domino_c), (domino_r + 1, domino_c)]

    sprite_cells = [(sprite_r + r, sprite_c + c) for r, c in sprite_local]
    sprite_comp = {
        "coords": np.array(sprite_cells, dtype=int),
        "area": len(sprite_cells),
        "color": color,
        "bbox": (
            min(r for r, _ in sprite_cells),
            min(c for _, c in sprite_cells),
            max(r for r, _ in sprite_cells),
            max(c for _, c in sprite_cells),
        ),
    }
    domino_comp = {
        "coords": np.array(domino_cells, dtype=int),
        "area": 2,
        "color": DOMINO_COLOR,
        "bbox": (
            min(r for r, _ in domino_cells),
            min(c for _, c in domino_cells),
            max(r for r, _ in domino_cells),
            max(c for _, c in domino_cells),
        ),
    }
    mirror_cells = _reflected_coords(sprite_comp, domino_comp)

    inp = np.full((patch_h, patch_w), BG_COLOR, dtype=int)
    out = np.full_like(inp, BG_COLOR)
    _paint_coords(inp, sprite_cells, color)
    _paint_coords(inp, domino_cells, DOMINO_COLOR)
    _paint_coords(out, sprite_cells, color)
    _paint_coords(out, mirror_cells, MIRROR_COLOR)
    return inp, out


def _can_place(reserved: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return not bool(reserved[top : top + h, left : left + w].any())


def _reserve(reserved: np.ndarray, top: int, left: int, shape: tuple[int, int], padding: int) -> None:
    h, w = map(int, shape)
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(reserved.shape[0], top + h + padding)
    c1 = min(reserved.shape[1], left + w + padding)
    reserved[r0:r1, c0:c1] = True


def _paste_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = dst[top : top + h, left : left + w]
    mask = patch != BG_COLOR
    region[mask] = patch[mask]


def _choose_orientations(n_groups: int) -> list[str]:
    orientations = ["lr", "ud"]
    while len(orientations) < n_groups:
        orientations.append(random.choice(["lr", "ud"]))
    random.shuffle(orientations)
    return orientations


def _canvas_bounds(group_shapes: list[tuple[int, int]]) -> tuple[int, int]:
    max_h = max(h for h, _ in group_shapes)
    max_w = max(w for _, w in group_shapes)
    total_h = sum(h for h, _ in group_shapes) + PADDING * (len(group_shapes) + 1)
    total_w = sum(w for _, w in group_shapes) + PADDING * (len(group_shapes) + 1)
    min_h = max(GRID_MIN, max_h + 2, min(total_h, GRID_MAX))
    min_w = max(GRID_MIN, max_w + 2, min(total_w, GRID_MAX))
    return min_h, min_w


def _place_groups(patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    shapes = [patch.shape for patch, _ in patches]
    min_h, min_w = _canvas_bounds(shapes)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    H = random.randint(min_h, GRID_MAX)
    W = random.randint(min_w, GRID_MAX)
    reserved = np.zeros((H, W), dtype=bool)
    inp = np.full((H, W), BG_COLOR, dtype=int)
    out = np.full((H, W), BG_COLOR, dtype=int)

    order = list(range(len(patches)))
    order.sort(key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1], reverse=True)
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch_in, patch_out = patches[idx]
        ph, pw = patch_in.shape
        if ph > H or pw > W:
            return None

        candidates = [(r, c) for r in range(H - ph + 1) for c in range(W - pw + 1)]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates[:PLACEMENT_ATTEMPTS]:
            if not _can_place(reserved, top, left, patch_in.shape):
                continue
            placements[idx] = (top, left)
            _reserve(reserved, top, left, patch_in.shape, PADDING)
            placed = True
            break
        if not placed:
            return None

    for idx, (top, left) in placements.items():
        patch_in, patch_out = patches[idx]
        _paste_patch(inp, patch_in, top, left)
        _paste_patch(out, patch_out, top, left)

    return inp, out


def _validate_generated(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape:
        return False
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if out.ndim != 2 or max(out.shape) > 30:
        return False

    pairings = _pair_sprites_and_dominos(inp)
    orientations = {_domino_orientation(domino) for _, domino in pairings}
    if orientations != {"horizontal", "vertical"}:
        return False

    if np.any(out == DOMINO_COLOR):
        return False
    if np.count_nonzero(out == MIRROR_COLOR) != sum(int(sprite["area"]) for sprite, _ in pairings):
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.choice(GROUP_COUNT_CHOICES)
    colors = random.sample(list(SPRITE_COLORS), n_groups)
    modes = _choose_orientations(n_groups)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color, mode in zip(colors, modes):
        mask = random.choice(SPRITE_MASKS)
        side = random.choice(["left", "right"]) if mode == "lr" else random.choice(["up", "down"])
        patches.append(_make_group(mask, color, mode, side))

    placed = _place_groups(patches)
    if placed is None:
        return None
    inp, out = placed
    if not _validate_generated(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-axis domino puzzle")
