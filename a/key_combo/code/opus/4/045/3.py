from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, neighbors4, random_free_location


BASE_SHAPES: dict[str, np.ndarray] = {
    "block2": np.array([[1, 1], [1, 1]], dtype=bool),
    "rect23": np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    "rect24": np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool),
    "corner8": np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
}

GRID_MIN = 18
GRID_MAX = 24
BOX_H_RANGE = (10, 14)
BOX_W_RANGE = (11, 15)
GEN_ATTEMPTS = 400
SPRITE_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.GREY]


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.asarray(coords, dtype=int)
    if coords.size == 0:
        return ()
    mins = coords.min(axis=0)
    norm = coords - mins
    return tuple(sorted((int(r), int(c)) for r, c in norm))


def _patch_signature(patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _shape_signature(np.argwhere(patch))


def _rotation_signatures(coords: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    patch = _coords_to_patch(coords)
    return {_patch_signature(np.rot90(patch, k)) for k in range(4)}


def _coords_to_patch(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = True
    return patch


def _clean_supported_pixels(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == Color.BLACK:
                continue
            support = 0
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == color:
                    support += 1
            if support < 2:
                out[r, c] = Color.BLACK
    return out


def _collect_holes(crop: np.ndarray) -> list[np.ndarray]:
    hole_grid = np.where(crop == Color.BLACK, Color.RED, Color.BLACK)
    comps = find_components(
        hole_grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.RED],
        return_format="dict",
    )
    h, w = crop.shape
    holes = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        touches_border = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        if not touches_border:
            holes.append(coords)
    return holes


def _collect_sprites(grid: np.ndarray) -> list[dict]:
    mask = (grid != Color.BLACK) & (grid != Color.BLUE)
    sprite_grid = np.where(mask, Color.RED, Color.BLACK)
    comps = find_components(
        sprite_grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.RED],
        return_format="dict",
    )
    sprites = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "cleaned sprite must be monochrome"
        color = next(iter(colors))
        sprites.append(
            {
                "color": color,
                "coords": coords,
                "rot_sigs": _rotation_signatures(coords),
            }
        )
    return sprites


def _solve(grid: np.ndarray, *, clean: bool) -> np.ndarray:
    g = _clean_supported_pixels(grid) if clean else np.asarray(grid, dtype=int).copy()
    blue_comps = find_components(
        g,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.BLUE],
        return_format="dict",
    )
    assert blue_comps, "expected a blue object"
    blue_comps.sort(key=lambda comp: int(comp["area"]), reverse=True)
    blue = blue_comps[0]
    if clean:
        assert len(blue_comps) == 1, "expected exactly one cleaned blue object"
    target = crop_bbox(g, blue["bbox"])
    holes = _collect_holes(target)
    sprites = _collect_sprites(g)
    out = target.copy()

    for hole in holes:
        sig = _shape_signature(hole)
        matches = [sprite for sprite in sprites if sig in sprite["rot_sigs"]]
        if clean:
            assert len(matches) == 1, "each cleaned hole must match exactly one cleaned sprite"
        if len(matches) == 1:
            out[hole[:, 0], hole[:, 1]] = matches[0]["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean support-poor noise, crop the blue holed object, and fill each cleaned hole with the color of its rotation-matching cleaned sprite.

    Input structure:
    - Black background contains one large blue 4-connected object whose bbox encloses 2 or 3 black holes.
    - Outside that object are 2 or 3 smaller monochrome sprites in distinct colors.
    - Magenta noise pixels may sit inside holes, on sprite edges, or as isolated clutter, and blue may have single-cell spurs.

    Rule:
    - Erase every non-black pixel whose same-color 4-neighbor count is below 2, including blue and non-blue pixels.
    - Crop the bbox of the remaining blue object and treat each internal black 4-connected component as a hole.
    - For each remaining non-blue 4-connected sprite, compare its normalized shape against each hole under the four right-angle rotations.
    - Fill a hole with the matched sprite color; unmatched holes stay black, and return only the cropped blue-object subgrid.

    Key insight:
    - The visible raw contours are misleading because support-poor magenta and blue pixels distort both hole topology and sprite outlines; only the cleaned shapes match.
    """
    return _solve(np.asarray(grid, dtype=int), clean=True)


def _canonical_shape_key(patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_patch_signature(np.rot90(patch, k)) for k in range(4))


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_holes(box_h: int, box_w: int, patches: list[tuple[str, np.ndarray]]) -> list[dict] | None:
    order = patches[:]
    random.shuffle(order)
    placed: list[dict] = []
    for name, patch in order:
        h, w = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(1, box_h - h)
            for c0 in range(1, box_w - w)
        ]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            cur_box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_boxes_far_enough(cur_box, prev["bbox"], gap=1) for prev in placed):
                placed.append({"name": name, "patch": patch, "top": r0, "left": c0, "bbox": cur_box})
                break
        else:
            return None
    return placed


def _hole_boundary_cells(crop: np.ndarray, hole_coords: np.ndarray) -> list[tuple[int, int]]:
    hole_set = {tuple(map(int, coord)) for coord in hole_coords}
    boundary = []
    h, w = crop.shape
    for r, c in hole_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in hole_set and int(crop[nr, nc]) == Color.BLUE:
                boundary.append((r, c))
                break
    return boundary


def _build_sprite_patch(core_patch: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((core_patch.shape[0] + 2, core_patch.shape[1] + 2), dtype=int)
    patch[1:-1, 1:-1][core_patch] = color
    occupied = {tuple(map(int, coord)) for coord in np.argwhere(patch == color)}
    attachments = []
    for r, c in sorted(occupied):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and patch[nr, nc] == Color.BLACK:
                attachments.append((nr, nc))
    primary = random.choice(attachments)
    patch[primary] = Color.PINK

    if random.random() < 0.45:
        secondaries = []
        for nr, nc in neighbors4(*primary):
            if (
                0 <= nr < patch.shape[0]
                and 0 <= nc < patch.shape[1]
                and patch[nr, nc] == Color.BLACK
                and (nr, nc) not in occupied
            ):
                secondaries.append((nr, nc))
        if secondaries:
            patch[random.choice(secondaries)] = Color.PINK
    return patch


def _add_blue_spikes(grid: np.ndarray, top: int, left: int, box_h: int, box_w: int) -> int:
    candidates = []
    for c in range(left + 1, left + box_w - 1):
        candidates.append((top - 1, c))
        candidates.append((top + box_h, c))
    for r in range(top + 1, top + box_h - 1):
        candidates.append((r, left - 1))
        candidates.append((r, left + box_w))
    candidates = [
        (r, c)
        for r, c in candidates
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == Color.BLACK
    ]
    random.shuffle(candidates)
    spike_count = random.choice([1, 2])
    chosen = candidates[:spike_count]
    for r, c in chosen:
        grid[r, c] = Color.BLUE
    return len(chosen)


def _place_noise_group(grid: np.ndarray, length: int) -> bool:
    h, w = grid.shape
    dirs = [(0, 1), (1, 0)]
    candidates = []
    for r in range(h):
        for c in range(w):
            for dr, dc in dirs:
                cells = [(r + k * dr, c + k * dc) for k in range(length)]
                if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells):
                    continue
                if any(grid[rr, cc] != Color.BLACK for rr, cc in cells):
                    continue
                blocked = False
                for rr, cc in cells:
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and grid[nr, nc] != Color.BLACK:
                            blocked = True
                            break
                    if blocked:
                        break
                if not blocked:
                    candidates.append(cells)
    if not candidates:
        return False
    for rr, cc in random.choice(candidates):
        grid[rr, cc] = Color.PINK
    return True


def _expected_output(box_h: int, box_w: int, hole_infos: list[dict]) -> np.ndarray:
    out = np.full((box_h, box_w), Color.BLUE, dtype=int)
    for info in hole_infos:
        out[info["coords"][:, 0], info["coords"][:, 1]] = info["color"]
    return out


def _validate_instance(grid: np.ndarray, expected: np.ndarray, n_holes: int) -> bool:
    cleaned = _clean_supported_pixels(grid)
    if not np.any(grid == Color.PINK) or np.any(cleaned == Color.PINK):
        return False
    if np.sum(grid == Color.BLUE) <= np.sum(cleaned == Color.BLUE):
        return False

    blue_comps = find_components(
        cleaned,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.BLUE],
        return_format="dict",
    )
    if len(blue_comps) != 1:
        return False
    target = crop_bbox(cleaned, blue_comps[0]["bbox"])
    holes = _collect_holes(target)
    sprites = _collect_sprites(cleaned)
    if len(holes) != n_holes or len(sprites) != n_holes:
        return False

    for hole in holes:
        hole_sig = _shape_signature(hole)
        if sum(hole_sig in sprite["rot_sigs"] for sprite in sprites) != 1:
            return False

    if not np.array_equal(main(grid), expected):
        return False

    try:
        raw_out = _solve(grid, clean=False)
    except AssertionError:
        raw_out = None
    if raw_out is not None and np.array_equal(raw_out, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    box_h = random.randint(*BOX_H_RANGE)
    box_w = random.randint(*BOX_W_RANGE)
    if box_h + 6 > grid_h or box_w + 6 > grid_w:
        return None

    n_holes = random.choice([2, 3])
    shape_names = random.sample(list(BASE_SHAPES), n_holes)
    if "corner8" not in shape_names and random.random() < 0.75:
        shape_names[0] = "corner8"
        if len(set(shape_names)) != n_holes:
            return None

    hole_patches = []
    rel_rots = []
    for name in shape_names:
        hole_rot = random.randrange(4)
        sprite_rot = random.randrange(4)
        rel_rots.append((hole_rot - sprite_rot) % 4)
        hole_patches.append((name, np.rot90(BASE_SHAPES[name], hole_rot)))
    if all(rot == 0 for rot in rel_rots):
        return None

    placements = _place_holes(box_h, box_w, hole_patches)
    if placements is None:
        return None

    hole_infos = []
    clean_crop = np.full((box_h, box_w), Color.BLUE, dtype=int)
    colors = random.sample(SPRITE_COLORS, n_holes)
    for placement, color in zip(placements, colors):
        patch = placement["patch"]
        coords = np.argwhere(patch) + np.array([placement["top"], placement["left"]])
        clean_crop[coords[:, 0], coords[:, 1]] = Color.BLACK
        hole_infos.append({"name": placement["name"], "coords": coords.astype(int), "color": color})

    grid = np.zeros((grid_h, grid_w), dtype=int)
    obj_top = random.randint(2, grid_h - box_h - 3)
    obj_left = random.randint(2, grid_w - box_w - 3)
    grid[obj_top : obj_top + box_h, obj_left : obj_left + box_w] = clean_crop

    if _add_blue_spikes(grid, obj_top, obj_left, box_h, box_w) == 0:
        return None

    hole_noise_count = 0
    for info in hole_infos:
        boundary = _hole_boundary_cells(clean_crop, info["coords"])
        if not boundary:
            return None
        rr, cc = random.choice(boundary)
        grid[obj_top + rr, obj_left + cc] = Color.PINK
        hole_noise_count += 1
    if hole_noise_count == 0:
        return None

    core_rotations = {name: random.randrange(4) for name in shape_names}
    sprite_name_to_color = {info["name"]: info["color"] for info in hole_infos}
    for name in shape_names:
        core_patch = np.rot90(BASE_SHAPES[name], core_rotations[name])
        sprite_patch = _build_sprite_patch(core_patch, sprite_name_to_color[name])
        try:
            top, left = random_free_location(
                grid,
                sprite_patch,
                transparent=Color.BLACK,
                border=1,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        mask = sprite_patch != Color.BLACK
        grid[top : top + sprite_patch.shape[0], left : left + sprite_patch.shape[1]][mask] = sprite_patch[mask]

    for _ in range(random.choice([1, 2, 3])):
        _place_noise_group(grid, random.choice([1, 2]))

    expected = _expected_output(box_h, box_w, hole_infos)
    if not _validate_instance(grid, expected, n_holes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
