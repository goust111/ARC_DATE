from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    candidates = [Path("/home/lixinyu/snap/new/solvers/arc_utils.py")]
    candidates.extend(parent / "new" / "solvers" / "arc_utils.py" for parent in here.parents)
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("_arc_utils_task_003_8", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError("could not locate new/solvers/arc_utils.py")


_arc_utils = _load_arc_utils()
Color = _arc_utils.Color
bbox = _arc_utils.bbox
colors_present = _arc_utils.colors_present
crop_bbox = _arc_utils.crop_bbox
find_components = _arc_utils.find_components
in_bounds = _arc_utils.in_bounds
paste = _arc_utils.paste


BG = Color.BLACK
BOUNDARY = Color.RED
COMPANION = Color.GREEN
SEPARATOR = Color.GREY
SPRITE_COLORS = (Color.BLUE, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
GEN_ATTEMPTS = 500


def _rot_cw(sprite: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(sprite, -int(quarter_turns))


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEPARATOR)]
    assert len(rows) == 1, f"expected exactly one full grey separator row, got {rows}"
    return int(rows[0])


def _extract_sprite(top: np.ndarray) -> np.ndarray:
    palette = colors_present(top, exclude={BG, SEPARATOR})
    assert len(palette) == 1, f"expected exactly one sprite color above separator, got {palette}"
    color = int(palette[0])
    comps = find_components(top, bg=BG, connectivity=4, monochromatic=True, colors=[color])
    assert len(comps) == 1, f"expected one top sprite component, got {len(comps)}"
    bb = bbox(top == color)
    assert bb is not None
    return crop_bbox(top, bb)


def _period_markers(lower: np.ndarray) -> tuple[tuple[int, int], int, int]:
    red = np.argwhere(lower == BOUNDARY)
    assert red.size, "missing red boundary markers"
    left = sorted((int(r), int(c)) for r, c in red if int(c) == 0)
    top = sorted((int(r), int(c)) for r, c in red if int(r) == 0)
    assert len(left) == 2, f"expected two left-edge red markers, got {left}"
    assert len(top) == 2, f"expected two top-edge red markers, got {top}"
    sv = abs(left[1][0] - left[0][0])
    sh = abs(top[1][1] - top[0][1])
    assert sv > 0 and sh > 0, f"periods must be positive, got sv={sv}, sh={sh}"
    anchor = min(((int(r), int(c)) for r, c in red), key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    return anchor, sv, sh


def _placement_points(shape: tuple[int, int], anchor: tuple[int, int], sv: int, sh: int) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    r0, c0 = map(int, anchor)
    points = []
    r = r0
    while r < height:
        c = c0
        while c < width:
            points.append((r, c))
            c += sh
        r += sv
    return points


def _companion_rotation(lower: np.ndarray, point: tuple[int, int]) -> int | None:
    r, c = map(int, point)
    found = []
    for rotation, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if in_bounds(lower.shape, nr, nc) and int(lower[nr, nc]) == COMPANION:
            found.append(rotation)
    assert len(found) <= 1, f"placement point {point} has multiple companions"
    return found[0] if found else None


def _analyze_input(grid: np.ndarray) -> dict:
    sep = _find_separator_row(grid)
    top = grid[:sep, :]
    lower = grid[sep + 1 :, :]
    sprite = _extract_sprite(top)
    anchor, sv, sh = _period_markers(lower)
    points = _placement_points(lower.shape, anchor, sv, sh)

    green = {tuple(map(int, rc)) for rc in np.argwhere(lower == COMPANION)}
    rotations: dict[tuple[int, int], int] = {}
    for point in points:
        rotation = _companion_rotation(lower, point)
        if rotation is not None:
            rotations[point] = rotation

    for cell in green:
        hits = 0
        for point in points:
            if abs(point[0] - cell[0]) + abs(point[1] - cell[1]) == 1:
                hits += 1
        assert hits == 1, f"green companion {cell} must belong to exactly one grid point"

    return {
        "separator": sep,
        "top": top,
        "lower": lower,
        "sprite": sprite,
        "anchor": anchor,
        "sv": sv,
        "sh": sh,
        "points": points,
        "rotations": rotations,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use red boundary-point periods and green neighbor directions to rotate and tile one sprite on the lower canvas.

    Input structure:
    - A black grid is split into a top template band and a lower canvas by one full grey separator row.
    - The top band contains exactly one non-black unit sprite in a single color.
    - On the lower canvas, red points on the left edge and top edge define the vertical and horizontal lattice periods, and some lattice points have one green 4-neighbor companion.

    Rule:
    - Read the top sprite as its tight bounding-box pattern.
    - In the lower canvas, use the two left-edge red points to get the vertical step, the two top-edge red points to get the horizontal step, and the red point closest to the canvas top-left as the lattice origin.
    - Generate all lattice points by repeated period steps; for each lattice point with one adjacent green companion, rotate the sprite according to the companion direction (up=0, right=90 clockwise, down=180, left=270 clockwise) and paste it with the lattice point as the sprite bbox's top-left anchor, clipping at the canvas boundary.
    - Remove all red and green markers to black, while keeping the top template band and grey separator row unchanged.

    Key insight:
    - The red markers do not mark every placement directly: they encode the 2D placement lattice by period, and each green companion compresses the rotation instruction into one adjacent cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _analyze_input(g)
    lower_out = info["lower"].copy()
    lower_out[(lower_out == BOUNDARY) | (lower_out == COMPANION)] = BG

    sprite = info["sprite"]
    for point, rotation in info["rotations"].items():
        r, c = point
        rotated = _rot_cw(sprite, rotation)
        paste(lower_out, rotated, r, c, transparent=BG, clip=True)

    return np.vstack(
        [
            info["top"].copy(),
            np.full((1, g.shape[1]), SEPARATOR, dtype=int),
            lower_out,
        ]
    )


def _random_sprite(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        size = rng.choice((3, 4))
        target_area = rng.randint(size + 1, size * size - 1)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not mask[-1].any() or not mask[:, -1].any():
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric sprite")


def _choose_dimensions(
    rng: random.Random, sprite_shape: tuple[int, int], row_count: int, col_count: int
) -> tuple[int, int, int, int]:
    sprite_h, sprite_w = map(int, sprite_shape)
    sv_min = max(3, sprite_h + 1)
    sh_min = max(3, sprite_w + 1)
    sv = rng.randint(sv_min, sv_min + 2)
    sh = rng.randint(sh_min, sh_min + 2)

    last_r = (row_count - 1) * sv
    last_c = (col_count - 1) * sh
    clip_row = rng.random() < 0.4
    clip_col = rng.random() < 0.4

    if clip_row:
        main_h = rng.randint(last_r + 1, last_r + sprite_h - 1)
    else:
        main_h = last_r + sprite_h + rng.randint(0, 2)
    if clip_col:
        main_w = rng.randint(last_c + 1, last_c + sprite_w - 1)
    else:
        main_w = last_c + sprite_w + rng.randint(0, 2)

    if main_h + sprite_h + 1 > 30 or main_w > 30:
        raise ValueError("sampled grid exceeds ARC bounds")
    return sv, sh, main_h, main_w


def _allowed_dirs(point: tuple[int, int], lower_shape: tuple[int, int], red_cells: set[tuple[int, int]]) -> list[int]:
    r, c = point
    allowed = []
    for direction, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if in_bounds(lower_shape, nr, nc) and (nr, nc) not in red_cells:
            allowed.append(direction)
    return allowed


def _assign_companions(
    rng: random.Random,
    points: list[tuple[int, int]],
    lower_shape: tuple[int, int],
    red_cells: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    total = len(points)
    assert total >= 4
    occupied_count = rng.randint(3, total - 1)

    allowed_map = {point: _allowed_dirs(point, lower_shape, red_cells) for point in points}
    for _ in range(200):
        chosen = rng.sample(points, occupied_count)
        if len({r for r, _ in chosen}) < 2 or len({c for _, c in chosen}) < 2:
            continue
        union = set().union(*(set(allowed_map[p]) for p in chosen))
        if len(union) < 2:
            continue

        target_distinct = min(len(union), occupied_count, 4 if occupied_count >= 4 and rng.random() < 0.7 else 3)
        target_dirs = rng.sample(sorted(union), target_distinct)
        assignment: dict[tuple[int, int], int] = {}
        remaining = list(chosen)
        ok = True
        for direction in target_dirs:
            candidates = [point for point in remaining if direction in allowed_map[point]]
            if not candidates:
                ok = False
                break
            point = rng.choice(candidates)
            assignment[point] = direction
            remaining.remove(point)
        if not ok:
            continue
        for point in remaining:
            assignment[point] = rng.choice(allowed_map[point])
        if len(set(assignment.values())) < 2:
            continue
        return assignment
    raise RuntimeError("failed to assign non-degenerate companions")


def _build_case(rng: random.Random) -> tuple[np.ndarray, dict]:
    sprite_mask = _random_sprite(rng)
    sprite_color = int(rng.choice(SPRITE_COLORS))
    sprite = sprite_mask.astype(int) * sprite_color
    row_count = rng.randint(2, 3)
    col_count = rng.randint(2, 3)
    sv, sh, main_h, main_w = _choose_dimensions(rng, sprite.shape, row_count, col_count)

    lower = np.zeros((main_h, main_w), dtype=int)
    red_cells = {(0, 0), (sv, 0), (0, sh)}
    for r, c in red_cells:
        lower[r, c] = BOUNDARY

    points = _placement_points(lower.shape, (0, 0), sv, sh)
    assignment = _assign_companions(rng, points, lower.shape, red_cells)
    for (r, c), direction in assignment.items():
        dr, dc = DIRS[direction]
        lower[r + dr, c + dc] = COMPANION

    top_h = sprite.shape[0] + rng.randint(0, 1)
    grid = np.zeros((top_h + 1 + main_h, main_w), dtype=int)
    sprite_col0 = rng.randint(0, main_w - sprite.shape[1])
    sprite_row0 = rng.randint(0, top_h - sprite.shape[0])
    patch = grid[sprite_row0 : sprite_row0 + sprite.shape[0], sprite_col0 : sprite_col0 + sprite.shape[1]]
    patch[sprite_mask] = sprite_color
    grid[top_h, :] = SEPARATOR
    grid[top_h + 1 :, :] = lower

    meta = {
        "sprite": sprite,
        "top_h": top_h,
        "main_h": main_h,
        "main_w": main_w,
        "sv": sv,
        "sh": sh,
        "points": points,
        "assignment": assignment,
        "sprite_pos": (sprite_row0, sprite_col0),
    }
    return grid, meta


def _expected_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    top_h = int(meta["top_h"])
    lower = out[top_h + 1 :, :]
    lower[(lower == BOUNDARY) | (lower == COMPANION)] = BG
    sprite = meta["sprite"]
    for point, rotation in meta["assignment"].items():
        paste(lower, _rot_cw(sprite, rotation), point[0], point[1], transparent=BG, clip=True)
    return out


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    expected = _expected_output(grid, meta)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    info = _analyze_input(grid)
    if info["sv"] != meta["sv"] or info["sh"] != meta["sh"]:
        return False
    if info["rotations"] != meta["assignment"]:
        return False
    if len(info["rotations"]) < 3:
        return False
    if len(info["points"]) - len(info["rotations"]) < 1:
        return False
    if len({r for r, _ in info["rotations"]}) < 2 or len({c for _, c in info["rotations"]}) < 2:
        return False

    sprite_mask = meta["sprite"] != BG
    if len({np.rot90(sprite_mask, k).tobytes() for k in range(4)}) != 4:
        return False
    if len(set(info["rotations"].values())) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        try:
            grid, meta = _build_case(rng)
        except (RuntimeError, ValueError):
            continue
        if _validate_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid periodic-rotation tiling input")
