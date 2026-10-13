from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG_COLOR = 0
HALO_COLOR = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}
CANONICAL_TAIL_DIR = (0, 1)

SPRITE_SHAPES = [
    ("BB.", "BAB"),
    ("BB.", "BAB", "BB."),
    ("BBB.", "BABB"),
    ("BBB.", "BABB", "BBB."),
]

GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (18, 24)
SPRITE_COUNT_RANGE = (2, 4)
GENERATION_ATTEMPTS = 500
PLACEMENT_PADDING = 1


def _rotate_dir_ccw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _rotation_steps(src: tuple[int, int], dst: tuple[int, int]) -> int:
    cur = src
    for steps in range(4):
        if cur == dst:
            return steps
        cur = _rotate_dir_ccw(cur)
    raise AssertionError(f"cannot rotate {src} into {dst}")


def _rotate_patch_anchor(
    patch: np.ndarray, anchor: tuple[int, int], steps: int
) -> tuple[np.ndarray, tuple[int, int]]:
    steps %= 4
    h, w = patch.shape
    ar, ac = anchor
    if steps == 0:
        return patch.copy(), (ar, ac)
    if steps == 1:
        return np.rot90(patch, 1), (w - 1 - ac, ar)
    if steps == 2:
        return np.rot90(patch, 2), (h - 1 - ar, w - 1 - ac)
    return np.rot90(patch, 3), (ac, h - 1 - ar)


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int) -> np.ndarray:
    patch = np.zeros((len(shape_rows), len(shape_rows[0])), dtype=int)
    for r, row in enumerate(shape_rows):
        for c, ch in enumerate(row):
            if ch == "B":
                patch[r, c] = body
            elif ch == "A":
                patch[r, c] = anchor
    return patch


def _occupied_rel_coords(patch: np.ndarray, bg: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch != bg)]


def _occupied_abs_coords(
    patch: np.ndarray, top: int, left: int, bg: int
) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in _occupied_rel_coords(patch, bg)]


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], radius: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _can_place_cells(
    shape: tuple[int, int], blocked: np.ndarray, cells: list[tuple[int, int]]
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _find_anchor(patch: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    colors = [int(v) for v in patch[patch != bg]]
    counts = Counter(colors)
    min_count = min(counts.values())
    anchor_colors = [color for color, count in counts.items() if count == min_count]
    assert min_count == 1 and len(anchor_colors) == 1, "sprite must contain one unique least-frequency anchor"
    anchor_color = anchor_colors[0]
    anchor_coords = np.argwhere(patch == anchor_color)
    assert anchor_coords.shape == (1, 2), "anchor color must occupy exactly one pixel"
    ar, ac = map(int, anchor_coords[0])
    return anchor_color, (ar, ac)


def _tail_direction(patch: np.ndarray, bg: int) -> tuple[int, int]:
    cells = set(_occupied_rel_coords(patch, bg))
    leaves = []
    for r, c in cells:
        neighbors = [(r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in cells]
        if len(neighbors) == 1:
            leaves.append(((r, c), neighbors[0]))
    assert len(leaves) == 1, "sprite must have exactly one single-pixel tail"
    (tail_r, tail_c), (nbr_r, nbr_c) = leaves[0]
    return tail_r - nbr_r, tail_c - nbr_c


def _find_halos(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    halos: list[dict] = []
    used_centers: set[tuple[int, int]] = set()
    used_ring: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center_color = int(grid[r, c])
            if center_color in (bg, HALO_COLOR):
                continue
            diagonals = [int(grid[r + dr, c + dc]) for dr in (-1, 1) for dc in (-1, 1)]
            if any(value != HALO_COLOR for value in diagonals):
                continue
            gap_dirs = [(dr, dc) for dr, dc in DIRS if int(grid[r + dr, c + dc]) == bg]
            if len(gap_dirs) != 1:
                continue
            gap_dir = gap_dirs[0]
            other_cardinals = [d for d in DIRS if d != gap_dir]
            if any(int(grid[r + dr, c + dc]) != HALO_COLOR for dr, dc in other_cardinals):
                continue
            ring = [
                (r + dr, c + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (dr, dc) != gap_dir
            ]
            assert (r, c) not in used_centers, "halo centers must be unique"
            assert not (set(ring) & used_ring), "halos must not share ring cells"
            used_centers.add((r, c))
            used_ring.update(ring)
            halos.append(
                {
                    "center": (r, c),
                    "color": center_color,
                    "gap_dir": gap_dir,
                    "ring": ring,
                }
            )
    assert halos, "expected at least one halo marker"
    halo_colors = [int(halo["color"]) for halo in halos]
    assert len(halo_colors) == len(set(halo_colors)), "halo center colors must be unique"
    return halos


def _erase_halos(grid: np.ndarray, halos: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for halo in halos:
        cr, cc = halo["center"]
        out[cr, cc] = bg
        for rr, cc2 in halo["ring"]:
            out[rr, cc2] = bg
    return out


def _sprite_records(grid: np.ndarray, bg: int) -> list[dict]:
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert components, "expected multicolor sprites"
    records = []
    for comp in components:
        assert int(comp["area"]) >= 2, "source sprites must be multi-pixel objects"
        patch = crop_bbox(grid, comp["bbox"])
        anchor_color, anchor = _find_anchor(patch, bg)
        tail_dir = _tail_direction(patch, bg)
        records.append(
            {
                "patch": patch,
                "anchor_color": anchor_color,
                "anchor": anchor,
                "tail_dir": tail_dir,
            }
        )
    anchor_colors = [int(record["anchor_color"]) for record in records]
    assert len(anchor_colors) == len(set(anchor_colors)), "sprite anchor colors must be unique"
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each sprite so its tail points toward the halo gap, then move its anchor onto the halo center.

    Input structure:
    - A dominant background contains several multicolor sprites, each with one unique least-frequency anchor pixel.
    - Separate landing markers consist of a colored center pixel surrounded by a gray halo that is missing exactly one cardinal ring cell.
    - Each halo center color matches exactly one sprite anchor color, and each sprite has one single-pixel tail that fixes its current facing.

    Rule:
    - Identify each sprite by its unique anchor color and match it to the halo whose center has the same color.
    - Rotate the sprite by 0/90/180/270 degrees until the tail points in the same direction as the halo gap relative to the halo center.
    - Erase the original sprites and the full halos, then place the rotated sprites so their anchor pixels land on the matched halo centers.

    Key insight:
    - The halo carries three independent signals at once: center color chooses the sprite, the missing gray cell chooses its final orientation, and the halo center gives the final anchor position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    halos = _find_halos(g, bg)
    sprite_grid = _erase_halos(g, halos, bg)
    sprites = _sprite_records(sprite_grid, bg)

    assert len(sprites) == len(halos), "sprite and halo counts must match"
    halo_by_color = {int(halo["color"]): halo for halo in halos}
    assert set(halo_by_color) == {int(sprite["anchor_color"]) for sprite in sprites}

    out = np.full_like(g, bg)
    for sprite in sprites:
        halo = halo_by_color[int(sprite["anchor_color"])]
        steps = _rotation_steps(sprite["tail_dir"], halo["gap_dir"])
        rotated_patch, rotated_anchor = _rotate_patch_anchor(sprite["patch"], sprite["anchor"], steps)
        top = int(halo["center"][0] - rotated_anchor[0])
        left = int(halo["center"][1] - rotated_anchor[1])
        h, w = rotated_patch.shape
        assert 0 <= top and 0 <= left and top + h <= g.shape[0] and left + w <= g.shape[1]
        mask = rotated_patch != bg
        assert not np.any(out[top : top + h, left : left + w][mask] != bg), "landed sprites must not overlap"
        paste(out, rotated_patch, top, left, transparent=bg)

    return out


def _sample_orientation_lists(n_sprites: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    while True:
        target_dirs = [DIRS[random.randrange(4)] for _ in range(n_sprites)]
        if n_sprites == 1 or len(set(target_dirs)) >= 2:
            break
    while True:
        source_dirs = [DIRS[random.randrange(4)] for _ in range(n_sprites)]
        deltas = [(_rotation_steps(src, dst)) for src, dst in zip(source_dirs, target_dirs)]
        if any(delta != 0 for delta in deltas) and (n_sprites == 1 or len(set(deltas)) >= 2):
            return source_dirs, target_dirs


def _build_halo_patch(center_color: int, gap_dir: tuple[int, int], bg: int) -> np.ndarray:
    patch = np.full((3, 3), HALO_COLOR, dtype=int)
    patch[1, 1] = center_color
    patch[1 + gap_dir[0], 1 + gap_dir[1]] = bg
    return patch


def _place_halo_with_output(
    input_grid: np.ndarray,
    blocked_input: np.ndarray,
    blocked_output: np.ndarray,
    final_patch: np.ndarray,
    final_anchor: tuple[int, int],
    halo_patch: np.ndarray,
    bg: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = input_grid.shape
    centers = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(centers)
    final_rel = _occupied_rel_coords(final_patch, bg)
    halo_rel = _occupied_rel_coords(halo_patch, bg)
    for center_r, center_c in centers:
        final_top = center_r - final_anchor[0]
        final_left = center_c - final_anchor[1]
        final_cells = [(final_top + r, final_left + c) for r, c in final_rel]
        if not _can_place_cells(input_grid.shape, blocked_output, final_cells):
            continue
        halo_top = center_r - 1
        halo_left = center_c - 1
        halo_cells = [(halo_top + r, halo_left + c) for r, c in halo_rel]
        if not _can_place_cells(input_grid.shape, blocked_input, halo_cells):
            continue
        paste(input_grid, halo_patch, halo_top, halo_left, transparent=bg)
        _mark_blocked(blocked_input, halo_cells, PLACEMENT_PADDING)
        _mark_blocked(blocked_output, final_cells, PLACEMENT_PADDING)
        return (center_r, center_c), (final_top, final_left)
    return None


def _place_source_sprite(
    input_grid: np.ndarray,
    blocked_input: np.ndarray,
    source_patch: np.ndarray,
    bg: int,
) -> tuple[int, int] | None:
    h, w = input_grid.shape
    ph, pw = source_patch.shape
    tops = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(tops)
    rel = _occupied_rel_coords(source_patch, bg)
    for top, left in tops:
        cells = [(top + r, left + c) for r, c in rel]
        if not _can_place_cells(input_grid.shape, blocked_input, cells):
            continue
        paste(input_grid, source_patch, top, left, transparent=bg)
        _mark_blocked(blocked_input, cells, PLACEMENT_PADDING)
        return top, left
    return None


def _translate_without_rotation(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray | None:
    naive = np.full(shape, bg, dtype=int)
    for spec in specs:
        patch = spec["source_patch"]
        anchor = spec["source_anchor"]
        center_r, center_c = spec["center"]
        top = center_r - anchor[0]
        left = center_c - anchor[1]
        ph, pw = patch.shape
        if top < 0 or left < 0 or top + ph > shape[0] or left + pw > shape[1]:
            return None
        mask = patch != bg
        if np.any(naive[top : top + ph, left : left + pw][mask] != bg):
            return None
        paste(naive, patch, top, left, transparent=bg)
    return naive


def _has_nearest_center_mismatch(specs: list[dict]) -> bool:
    centers = [tuple(map(int, spec["center"])) for spec in specs]
    for idx, spec in enumerate(specs):
        ar, ac = spec["source_anchor_abs"]
        dists = [abs(ar - cr) + abs(ac - cc) for cr, cc in centers]
        if dists[idx] != min(dists):
            return True
    return False


def _canonical_sprite_specs(n_sprites: int) -> list[dict]:
    anchor_palette = [1, 2, 3, 4, 6, 7, 8, 9]
    anchor_colors = random.sample(anchor_palette, n_sprites)
    body_pool = [color for color in anchor_palette if color not in anchor_colors]
    specs = []
    for anchor_color in anchor_colors:
        body_color = random.choice(body_pool)
        shape_rows = random.choice(SPRITE_SHAPES)
        patch = _render_sprite(shape_rows, body_color, anchor_color)
        anchor_found, anchor = _find_anchor(patch, BG_COLOR)
        tail_dir = _tail_direction(patch, BG_COLOR)
        assert anchor_found == anchor_color
        assert tail_dir == CANONICAL_TAIL_DIR
        specs.append(
            {
                "anchor_color": anchor_color,
                "body_color": body_color,
                "canonical_patch": patch,
                "canonical_anchor": anchor,
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)
    input_grid = np.full((height, width), BG_COLOR, dtype=int)
    expected = np.full_like(input_grid, BG_COLOR)
    blocked_input = np.zeros_like(input_grid, dtype=bool)
    blocked_output = np.zeros_like(input_grid, dtype=bool)

    specs = _canonical_sprite_specs(n_sprites)
    source_dirs, target_dirs = _sample_orientation_lists(n_sprites)

    placements: list[dict] = []
    order = list(range(n_sprites))
    random.shuffle(order)

    for idx in order:
        spec = specs[idx]
        final_steps = _rotation_steps(CANONICAL_TAIL_DIR, target_dirs[idx])
        final_patch, final_anchor = _rotate_patch_anchor(
            spec["canonical_patch"], spec["canonical_anchor"], final_steps
        )
        halo_patch = _build_halo_patch(spec["anchor_color"], target_dirs[idx], BG_COLOR)
        placed = _place_halo_with_output(
            input_grid,
            blocked_input,
            blocked_output,
            final_patch,
            final_anchor,
            halo_patch,
            BG_COLOR,
        )
        if placed is None:
            return None
        center, final_top_left = placed
        placements.append(
            {
                "idx": idx,
                "center": center,
                "target_dir": target_dirs[idx],
                "final_patch": final_patch,
                "final_anchor": final_anchor,
                "final_top_left": final_top_left,
                "anchor_color": spec["anchor_color"],
            }
        )

    placements.sort(key=lambda item: item["idx"])

    for idx, placement in enumerate(placements):
        spec = specs[idx]
        source_steps = _rotation_steps(CANONICAL_TAIL_DIR, source_dirs[idx])
        source_patch, source_anchor = _rotate_patch_anchor(
            spec["canonical_patch"], spec["canonical_anchor"], source_steps
        )
        top_left = _place_source_sprite(input_grid, blocked_input, source_patch, BG_COLOR)
        if top_left is None:
            return None
        top, left = top_left
        placement["source_patch"] = source_patch
        placement["source_anchor"] = source_anchor
        placement["source_top_left"] = top_left
        placement["source_anchor_abs"] = (top + source_anchor[0], left + source_anchor[1])

    for placement in placements:
        top, left = placement["final_top_left"]
        paste(expected, placement["final_patch"], top, left, transparent=BG_COLOR)

    if not _has_nearest_center_mismatch(placements):
        return None

    naive = _translate_without_rotation(placements, input_grid.shape, BG_COLOR)
    if naive is not None and np.array_equal(naive, expected):
        return None

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-compass input")
