from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, paste


SIDE_ORDER = ("top", "bottom", "left", "right")
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400
BLOCK_COUNT_RANGE = (3, 7)
GRID_HEIGHT_RANGE = (13, 24)
GRID_WIDTH_RANGE = (13, 24)


def _uniform_non_black(values: np.ndarray) -> int | None:
    vals = [int(v) for v in np.ravel(values)]
    if not vals:
        return None
    first = vals[0]
    if first == Color.BLACK:
        return None
    if all(v == first for v in vals):
        return first
    return None


def _parse_walls(grid: np.ndarray) -> tuple[dict[str, int], dict[int, str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3

    walls: dict[str, int] = {}
    top = _uniform_non_black(g[0, 1 : w - 1])
    bottom = _uniform_non_black(g[h - 1, 1 : w - 1])
    left = _uniform_non_black(g[1 : h - 1, 0])
    right = _uniform_non_black(g[1 : h - 1, w - 1])
    if top is not None:
        walls["top"] = top
    if bottom is not None:
        walls["bottom"] = bottom
    if left is not None:
        walls["left"] = left
    if right is not None:
        walls["right"] = right

    assert len(walls) >= 2, "expected at least two edge walls"
    assert len(set(walls.values())) == len(walls), "wall colors must be distinct"
    by_color = {color: side for side, color in walls.items()}
    return walls, by_color


def _distance_to_wall(r0: int, c0: int, h: int, w: int, side: str, shape: tuple[int, int]) -> int:
    grid_h, grid_w = shape
    if side == "top":
        return int(r0 - 1)
    if side == "bottom":
        return int(grid_h - r0 - h - 1)
    if side == "left":
        return int(c0 - 1)
    if side == "right":
        return int(grid_w - c0 - w - 1)
    raise ValueError(f"unknown side: {side}")


def _marker_local(patch: np.ndarray, marker_color: int) -> tuple[int, int]:
    coords = np.argwhere(patch == int(marker_color))
    assert coords.shape == (1, 2)
    return int(coords[0, 0]), int(coords[0, 1])


def _block_signature(block: dict) -> tuple:
    patch = np.asarray(block["patch"], dtype=int)
    h, w = patch.shape
    return (
        int(block["r0"]),
        int(block["c0"]),
        h,
        w,
        str(block["side"]),
        int(block["distance"]),
        int(block["fill_color"]),
        int(block["marker_color"]),
        _marker_local(patch, int(block["marker_color"])),
    )


def _parse_blocks(grid: np.ndarray, wall_by_color: dict[int, str]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    interior = g.copy()
    interior[0, :] = Color.BLACK
    interior[-1, :] = Color.BLACK
    interior[:, 0] = Color.BLACK
    interior[:, -1] = Color.BLACK
    objects = find_components(
        interior,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert objects, "expected at least one block"

    blocks: list[dict] = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        h, w = patch.shape
        assert int(obj["area"]) == int(h * w), "each block must be a solid rectangle"

        cols = colors_present(patch, exclude={Color.BLACK})
        assert len(cols) == 2, "each block must use fill color plus one marker color"
        counts = Counter(map(int, patch.ravel()))
        marker_candidates = [col for col in cols if counts[int(col)] == 1]
        assert len(marker_candidates) == 1, "expected exactly one marker pixel per block"

        marker_color = int(marker_candidates[0])
        assert marker_color in wall_by_color, "marker must claim an existing wall color"
        fill_color = int(cols[0] if cols[1] == marker_color else cols[1])

        blocks.append(
            {
                "coords": obj["coords"],
                "r0": int(r0),
                "c0": int(c0),
                "patch": patch,
                "side": wall_by_color[marker_color],
                "distance": _distance_to_wall(int(r0), int(c0), h, w, wall_by_color[marker_color], g.shape),
                "fill_color": fill_color,
                "marker_color": marker_color,
            }
        )
    return blocks


def _can_place(base: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    grid_h, grid_w = base.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid_h or c0 + w > grid_w:
        return False
    sub = base[r0 : r0 + h, c0 : c0 + w]
    return not bool(np.any((patch != Color.BLACK) & (sub != Color.BLACK)))


def _stop_reason(base: np.ndarray, patch: np.ndarray, r0: int, c0: int, step: tuple[int, int]) -> str:
    dr, dc = step
    nr = int(r0 + dr)
    nc = int(c0 + dc)
    h, w = patch.shape
    grid_h, grid_w = base.shape
    if nr < 0 or nc < 0 or nr + h > grid_h or nc + w > grid_w:
        return "boundary"

    sub = base[nr : nr + h, nc : nc + w]
    overlap = (patch != Color.BLACK) & (sub != Color.BLACK)
    assert overlap.any(), "stop reason should only be queried at a blocked step"
    rr, cc = np.nonzero(overlap)
    abs_r = nr + rr
    abs_c = nc + cc
    on_border = (abs_r == 0) | (abs_r == grid_h - 1) | (abs_c == 0) | (abs_c == grid_w - 1)
    return "wall" if bool(np.all(on_border)) else "block"


def _clear_blocks(grid: np.ndarray, blocks: list[dict]) -> np.ndarray:
    base = np.asarray(grid, dtype=int).copy()
    for block in blocks:
        coords = np.asarray(block["coords"], dtype=int)
        base[coords[:, 0], coords[:, 1]] = Color.BLACK
    return base


def _settle_blocks(base_grid: np.ndarray, blocks: list[dict], *, track: bool = False) -> tuple[np.ndarray, list[dict]]:
    distances = [int(block["distance"]) for block in blocks]
    assert len(set(distances)) == len(distances), "generator guarantees a unique nearest-first order"

    out = np.asarray(base_grid, dtype=int).copy()
    records: list[dict] = []
    for block in sorted(blocks, key=lambda item: int(item["distance"])):
        patch = np.asarray(block["patch"], dtype=int)
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        step = SIDE_TO_STEP[str(block["side"])]

        moved = 0
        while _can_place(out, patch, r0 + step[0], c0 + step[1]):
            r0 += step[0]
            c0 += step[1]
            moved += 1

        reason = _stop_reason(out, patch, r0, c0, step)
        paste(out, patch, r0, c0, transparent=Color.BLACK, clip=False)
        if track:
            records.append(
                {
                    "side": str(block["side"]),
                    "distance": int(block["distance"]),
                    "start": (int(block["r0"]), int(block["c0"])),
                    "end": (r0, c0),
                    "steps": moved,
                    "stop": reason,
                }
            )
    return out, records


def _resolve(grid: np.ndarray, *, track: bool = False) -> tuple[np.ndarray, list[dict], dict[str, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    walls, wall_by_color = _parse_walls(g)
    blocks = _parse_blocks(g, wall_by_color)
    base = _clear_blocks(g, blocks)
    out, records = _settle_blocks(base, blocks, track=track)
    return out, records, walls, blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked rectangle toward its claimed edge wall, settling blocks nearest to their own wall first.

    Input structure:
    - The background is black, and at least two border sides carry one-cell-thick solid edge walls in distinct colors.
    - Inside the border are several solid rectangular blocks; each block has one differently colored marker pixel embedded inside it.
    - Every marker color matches exactly one wall color, which identifies that block's target side.

    Rule:
    - Detect each interior rectangle and read its target direction from the embedded marker color.
    - Order all blocks globally by current distance to their own claimed wall, with the nearest block settling first.
    - Move each block rigidly in a straight line until one more step would hit the claimed wall, the grid boundary, or an already settled block; keep the walls unchanged.

    Key insight:
    - Gravity is block-specific rather than global: the marker color chooses the destination wall, and earlier settled blocks become obstacles for later blocks even when they move in different directions.
    """
    out, _records, _walls, _blocks = _resolve(np.asarray(grid, dtype=int), track=False)
    return out


def _draw_walls(grid: np.ndarray, walls: dict[str, int]) -> None:
    for side in SIDE_ORDER:
        if side not in walls:
            continue
        color = int(walls[side])
        if side == "top":
            grid[0, :] = color
        elif side == "bottom":
            grid[-1, :] = color
        elif side == "left":
            grid[:, 0] = color
        else:
            grid[:, -1] = color


def _make_patch(h: int, w: int, fill_color: int, marker_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.full((h, w), int(fill_color), dtype=int)
    marker_r = random.randrange(h)
    marker_c = random.randrange(w)
    patch[marker_r, marker_c] = int(marker_color)
    return patch, (marker_r, marker_c)


def _rect_is_clear(occ: np.ndarray, r0: int, c0: int, h: int, w: int, *, pad: int = 1) -> bool:
    grid_h, grid_w = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(grid_h, r0 + h + pad)
    cc1 = min(grid_w, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _candidate_positions(
    shape: tuple[int, int],
    side: str,
    h: int,
    w: int,
    used_distances: set[int],
    occ: np.ndarray,
) -> list[tuple[int, int, int]]:
    grid_h, grid_w = shape
    candidates: list[tuple[int, int, int]] = []
    for r0 in range(1, grid_h - h):
        for c0 in range(1, grid_w - w):
            if not _rect_is_clear(occ, r0, c0, h, w, pad=1):
                continue
            dist = _distance_to_wall(r0, c0, h, w, side, shape)
            if dist < 1 or dist in used_distances:
                continue
            candidates.append((r0, c0, dist))
    return candidates


def _random_shape() -> tuple[int, int]:
    for _ in range(30):
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        if h * w > 1:
            return h, w
    return 2, 2


def _all_wall_subsets(force_cross: bool) -> list[tuple[str, ...]]:
    sides = list(SIDE_ORDER)
    out: list[tuple[str, ...]] = []
    for mask in range(1 << len(sides)):
        subset = tuple(side for i, side in enumerate(sides) if mask & (1 << i))
        if len(subset) < 2:
            continue
        if force_cross:
            has_vertical = any(side in {"top", "bottom"} for side in subset)
            has_horizontal = any(side in {"left", "right"} for side in subset)
            if not (has_vertical and has_horizontal):
                continue
        out.append(subset)
    return out


def _build_input_grid(shape: tuple[int, int], walls: dict[str, int], blocks: list[dict]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    _draw_walls(grid, walls)
    for block in blocks:
        paste(grid, np.asarray(block["patch"], dtype=int), int(block["r0"]), int(block["c0"]), transparent=Color.BLACK)
    return grid


def _add_block_spec(
    blocks: list[dict],
    occ: np.ndarray,
    r0: int,
    c0: int,
    patch: np.ndarray,
    side: str,
    fill_color: int,
    marker_color: int,
    shape: tuple[int, int],
) -> None:
    h, w = patch.shape
    blocks.append(
        {
            "coords": np.argwhere(np.ones((h, w), dtype=bool)) + np.array([r0, c0]),
            "r0": int(r0),
            "c0": int(c0),
            "patch": patch.copy(),
            "side": str(side),
            "distance": _distance_to_wall(r0, c0, h, w, side, shape),
            "fill_color": int(fill_color),
            "marker_color": int(marker_color),
        }
    )
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _try_place_coupled_pair(
    shape: tuple[int, int],
    walls: dict[str, int],
    fill_palette: list[int],
    blocks: list[dict],
    occ: np.ndarray,
    used_distances: set[int],
) -> bool:
    vertical_sides = [side for side in walls if side in {"top", "bottom"}]
    horizontal_sides = [side for side in walls if side in {"left", "right"}]
    if not vertical_sides or not horizontal_sides:
        return False

    grid_h, grid_w = shape
    for _ in range(80):
        leader_side = random.choice(vertical_sides)
        follower_side = random.choice(horizontal_sides)
        h_a = random.randint(2, 4)
        w_a = random.randint(2, 4)
        d_a_max = (grid_h - h_a - 2) if leader_side in {"top", "bottom"} else (grid_w - w_a - 2)
        if d_a_max < 2:
            continue
        d_a = random.randint(2, min(4, d_a_max))
        r_a = 1 + d_a if leader_side == "top" else grid_h - h_a - 1 - d_a
        c_a_min = 1
        c_a_max = grid_w - w_a - 1
        if c_a_min > c_a_max:
            continue
        c_a = random.randint(c_a_min, c_a_max)
        if not _rect_is_clear(occ, r_a, c_a, h_a, w_a, pad=1):
            continue

        h_b, w_b = _random_shape()
        if h_b * w_b <= 1:
            continue
        r_af = 1 if leader_side == "top" else grid_h - h_a - 1
        c_af = c_a
        overlap_rows = [
            r_b
            for r_b in range(1, grid_h - h_b)
            if max(r_b, r_af) <= min(r_b + h_b - 1, r_af + h_a - 1)
        ]
        random.shuffle(overlap_rows)

        for r_b in overlap_rows:
            if follower_side == "right":
                c_b_final = c_af - w_b
                if c_b_final < 1:
                    continue
                min_d_b = max(d_a + 1, grid_w - c_af)
                max_d_b = grid_w - w_b - 2
                if min_d_b > max_d_b:
                    continue
                d_b_choices = [d for d in range(min_d_b, max_d_b + 1) if d not in used_distances and d != d_a]
                if not d_b_choices:
                    continue
                d_b = random.choice(d_b_choices)
                c_b = grid_w - w_b - 1 - d_b
                if c_b >= c_b_final:
                    continue
            else:
                c_b_final = c_af + w_a
                if c_b_final + w_b - 1 > grid_w - 2:
                    continue
                min_d_b = max(d_a + 1, c_b_final)
                max_d_b = grid_w - w_b - 2
                if min_d_b > max_d_b:
                    continue
                d_b_choices = [d for d in range(min_d_b, max_d_b + 1) if d not in used_distances and d != d_a]
                if not d_b_choices:
                    continue
                d_b = random.choice(d_b_choices)
                c_b = 1 + d_b
                if c_b <= c_b_final:
                    continue

            if not _rect_is_clear(occ, r_b, c_b, h_b, w_b, pad=1):
                continue

            fill_a = random.choice(fill_palette)
            fill_b = random.choice(fill_palette)
            patch_a, _ = _make_patch(h_a, w_a, fill_a, walls[leader_side])
            patch_b, _ = _make_patch(h_b, w_b, fill_b, walls[follower_side])

            _add_block_spec(blocks, occ, r_a, c_a, patch_a, leader_side, fill_a, walls[leader_side], shape)
            _add_block_spec(blocks, occ, r_b, c_b, patch_b, follower_side, fill_b, walls[follower_side], shape)
            used_distances.add(d_a)
            used_distances.add(d_b)
            return True
    return False


def _try_place_random_block(
    shape: tuple[int, int],
    walls: dict[str, int],
    fill_palette: list[int],
    blocks: list[dict],
    occ: np.ndarray,
    used_distances: set[int],
    preferred_side: str | None = None,
) -> str | None:
    side_pool = [preferred_side] if preferred_side is not None else list(walls)
    for _ in range(60):
        side = random.choice(side_pool)
        h, w = _random_shape()
        candidates = _candidate_positions(shape, side, h, w, used_distances, occ)
        if not candidates:
            if preferred_side is not None:
                return None
            continue
        r0, c0, dist = random.choice(candidates)
        fill_color = random.choice(fill_palette)
        patch, _ = _make_patch(h, w, fill_color, walls[side])
        _add_block_spec(blocks, occ, r0, c0, patch, side, fill_color, walls[side], shape)
        used_distances.add(dist)
        return side
    return None


def _validate_scene(grid: np.ndarray, walls: dict[str, int], blocks: list[dict], *, expect_coupling: bool) -> bool:
    try:
        parsed_walls, by_color = _parse_walls(grid)
        if parsed_walls != walls:
            return False

        parsed_blocks = _parse_blocks(grid, by_color)
        if sorted(_block_signature(block) for block in parsed_blocks) != sorted(_block_signature(block) for block in blocks):
            return False

        if len({int(block["distance"]) for block in blocks}) != len(blocks):
            return False
        if len({str(block["side"]) for block in blocks}) < 2:
            return False

        expected_base = _clear_blocks(grid, blocks)
        expected, expected_records = _settle_blocks(expected_base, blocks, track=True)
        actual, actual_records, _walls, _parsed = _resolve(grid, track=True)
        if not np.array_equal(expected, actual):
            return False
        if [rec["end"] for rec in expected_records] != [rec["end"] for rec in actual_records]:
            return False
        if max(rec["steps"] for rec in expected_records) <= 0:
            return False
        if expect_coupling and not any(rec["stop"] == "block" for rec in expected_records):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    force_cross = random.random() < 0.65
    shape = (
        random.randint(*GRID_HEIGHT_RANGE),
        random.randint(*GRID_WIDTH_RANGE),
    )

    wall_sides = random.choice(_all_wall_subsets(force_cross))
    wall_colors = random.sample(list(ALL_COLORS), len(wall_sides))
    walls = {side: color for side, color in zip(wall_sides, wall_colors)}
    fill_palette = [color for color in ALL_COLORS if color not in wall_colors]

    occ = np.zeros(shape, dtype=bool)
    blocks: list[dict] = []
    used_distances: set[int] = set()
    used_sides: list[str] = []

    coupled = False
    if force_cross:
        coupled = _try_place_coupled_pair(shape, walls, fill_palette, blocks, occ, used_distances)
        if not coupled:
            return None
        used_sides.extend([str(blocks[-2]["side"]), str(blocks[-1]["side"])])

    target_block_count = random.randint(*BLOCK_COUNT_RANGE)
    if coupled:
        target_block_count = max(target_block_count, 4)

    required_sides = list(random.sample(list(walls), min(2, len(walls))))
    for side in used_sides:
        if side in required_sides:
            required_sides.remove(side)

    while len(blocks) < target_block_count:
        preferred = required_sides[0] if required_sides else None
        placed_side = _try_place_random_block(shape, walls, fill_palette, blocks, occ, used_distances, preferred_side=preferred)
        if placed_side is None and preferred is not None:
            return None
        if placed_side is None:
            return None
        if required_sides and placed_side == required_sides[0]:
            required_sides.pop(0)

    grid = _build_input_grid(shape, walls, blocks)
    if not _validate_scene(grid, walls, blocks, expect_coupling=coupled):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid directional multi-gravity ARC input")
