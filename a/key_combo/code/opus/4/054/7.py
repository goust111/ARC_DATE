from __future__ import annotations

import numpy as np


BG = 0
BODY = 5
MARKER_POS_BY_TURNS = {
    0: (0, 1),  # up -> 0 deg
    1: (1, 2),  # right -> 90 deg clockwise
    2: (2, 1),  # down -> 180 deg
    3: (1, 0),  # left -> 270 deg clockwise
}
TURNS_BY_MARKER_POS = {pos: turns for turns, pos in MARKER_POS_BY_TURNS.items()}
MARKER_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
FEATURE_PATTERNS = (
    ((0, 0),),
    ((0, 2),),
    ((2, 0),),
    ((2, 2),),
    ((1, 1), (0, 0)),
    ((1, 1), (0, 2)),
    ((1, 1), (2, 0)),
    ((1, 1), (2, 2)),
    ((0, 0), (0, 2)),
    ((0, 0), (2, 0)),
    ((0, 2), (2, 2)),
    ((2, 0), (2, 2)),
)
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _rotate_clockwise(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, k=(-int(turns)) % 4)


def _stack(patches: list[np.ndarray]) -> np.ndarray:
    out = np.full((3 * len(patches), 3), BG, dtype=int)
    for i, patch in enumerate(patches):
        out[3 * i : 3 * i + 3, :] = patch
    return out


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    sprites = []
    for coords in _components_mask(grid != BG):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert patch.shape == (3, 3), f"expected 3x3 sprite, got {patch.shape}"
        assert np.all(patch != BG), "sprite must be a filled 3x3 block with no background holes"

        edge_marks = [pos for pos in TURNS_BY_MARKER_POS if int(patch[pos]) != BODY]
        assert len(edge_marks) == 1, "expected exactly one non-grey edge midpoint direction marker"
        marker_pos = edge_marks[0]
        turns = TURNS_BY_MARKER_POS[marker_pos]
        marker_color = int(patch[marker_pos])
        assert marker_color not in (BG, BODY)

        feature_positions = [
            (r, c)
            for r in range(3)
            for c in range(3)
            if (r, c) != marker_pos and int(patch[r, c]) != BODY
        ]
        assert 1 <= len(feature_positions) <= 2, "expected one or two extra feature pixels"
        feature_colors = {int(patch[r, c]) for r, c in feature_positions}
        assert len(feature_colors) == 1, "all extra feature pixels must share one color"
        feature_color = next(iter(feature_colors))
        assert feature_color not in (BG, BODY, marker_color)

        sprites.append(
            {
                "bbox": bbox,
                "patch": patch,
                "turns": turns,
                "marker_color": marker_color,
                "feature_color": feature_color,
                "feature_positions": tuple(feature_positions),
                "dist": int(r0 + c0),
            }
        )

    assert 3 <= len(sprites) <= 5, "expected 3 to 5 sprites"
    marker_colors = [sprite["marker_color"] for sprite in sprites]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be unique across sprites"
    dists = [sprite["dist"] for sprite in sprites]
    assert len(set(dists)) == len(dists), "sprite Manhattan distances must be unique"
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each direction-marked 3x3 sprite, then stack the rotated sprites by original Manhattan distance to the top-left corner.

    Input structure:
    - A black background contains 3 to 5 separated 3x3 sprites.
    - Each sprite is a filled grey 3x3 block with exactly one non-grey edge midpoint marker.
    - Each sprite also has one or two extra non-grey feature pixels of a second color, making its rotated appearance distinguishable.

    Rule:
    - Read the marker position inside each 3x3 sprite: top means 0 degrees, right means 90 degrees clockwise,
      bottom means 180 degrees, and left means 270 degrees clockwise.
    - Rotate the entire 3x3 sprite by that amount.
    - Compute each sprite's original Manhattan distance to (0, 0) using the sprite bbox top-left corner.
    - Sort the rotated sprites by that distance from small to large, then stack them tightly from top to bottom in a 3-column output.

    Key insight:
    - The marker is not an output color code; it is a direction code that determines a geometric rotation before the distance-based stacking order is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = _extract_sprites(g)
    sprites.sort(key=lambda sprite: sprite["dist"])
    rotated = [_rotate_clockwise(sprite["patch"], sprite["turns"]) for sprite in sprites]
    return _stack(rotated)


def _sample_directions(count: int) -> list[int]:
    base = [0, 1, 2, 3]
    RNG.shuffle(base)
    if count == 3:
        dirs = [0, base[1], base[2]]
    elif count == 4:
        dirs = base[:]
    else:
        dirs = base[:] + [int(RNG.integers(0, 4))]
    RNG.shuffle(dirs)
    return dirs


def _make_patch(marker_color: int, turns: int, feature_color: int, feature_cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.full((3, 3), BODY, dtype=int)
    patch[MARKER_POS_BY_TURNS[int(turns)]] = int(marker_color)
    for r, c in feature_cells:
        patch[r, c] = int(feature_color)
    return patch


def _can_place(existing: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = candidate
    for rr, cc in existing:
        if abs(r - rr) < 5 and abs(c - cc) < 5:
            return False
    return True


def _sample_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    RNG.shuffle(candidates)
    positions: list[tuple[int, int]] = []
    used_dists = set()
    for candidate in candidates:
        dist = candidate[0] + candidate[1]
        if dist in used_dists:
            continue
        if _can_place(positions, candidate):
            positions.append(candidate)
            used_dists.add(dist)
            if len(positions) == count:
                return positions
    return None


def _spec_output(specs: list[dict], order_key, rotated: bool) -> np.ndarray:
    ordered = sorted(specs, key=order_key)
    patches = [spec["rotated"] if rotated else spec["patch"] for spec in ordered]
    return _stack(patches)


def _validate_instance(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        extracted = _extract_sprites(grid)
    except AssertionError:
        return False

    if len(extracted) != len(specs):
        return False

    truth = _spec_output(specs, order_key=lambda spec: spec["dist"], rotated=True)
    if not np.array_equal(main(grid), truth):
        return False

    no_rotation = _spec_output(specs, order_key=lambda spec: spec["dist"], rotated=False)
    row_sorted = _spec_output(specs, order_key=lambda spec: (spec["r"], spec["c"]), rotated=True)
    col_sorted = _spec_output(specs, order_key=lambda spec: (spec["c"], spec["r"]), rotated=True)
    if np.array_equal(truth, no_rotation):
        return False
    if np.array_equal(truth, row_sorted):
        return False
    if np.array_equal(truth, col_sorted):
        return False

    row_order = [spec["id"] for spec in sorted(specs, key=lambda spec: (spec["r"], spec["c"]))]
    col_order = [spec["id"] for spec in sorted(specs, key=lambda spec: (spec["c"], spec["r"]))]
    dist_order = [spec["id"] for spec in sorted(specs, key=lambda spec: spec["dist"])]
    if row_order == dist_order or col_order == dist_order:
        return False

    turns = [spec["turns"] for spec in specs]
    if 0 not in turns or not any(turn != 0 for turn in turns):
        return False
    if len(set(turns)) < min(3, len(specs)):
        return False

    rotated_signatures = {tuple(spec["rotated"].reshape(-1).tolist()) for spec in specs}
    if len(rotated_signatures) != len(specs):
        return False

    for spec in specs:
        if spec["turns"] != 0 and np.array_equal(spec["patch"], spec["rotated"]):
            return False

    bboxes = [(spec["r"], spec["c"], spec["r"] + 2, spec["c"] + 2) for spec in specs]
    for i, (r0, c0, r1, c1) in enumerate(bboxes):
        for rr0, cc0, rr1, cc1 in bboxes[i + 1 :]:
            row_gap = max(0, max(rr0 - r1 - 1, r0 - rr1 - 1))
            col_gap = max(0, max(cc0 - c1 - 1, c0 - cc1 - 1))
            if row_gap < 2 and col_gap < 2:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.integers(3, 6))
    h = int(RNG.integers(15, 21))
    w = int(RNG.integers(15, 21))
    positions = _sample_positions(h, w, count)
    if positions is None:
        return None

    marker_colors = [int(c) for c in RNG.choice(MARKER_COLORS, size=count, replace=False)]
    directions = _sample_directions(count)
    grid = np.full((h, w), BG, dtype=int)
    specs = []

    for idx, ((r, c), marker_color, turns) in enumerate(zip(positions, marker_colors, directions)):
        feature_color_choices = [int(color) for color in MARKER_COLORS if int(color) != marker_color]
        feature_color = int(feature_color_choices[int(RNG.integers(len(feature_color_choices)))])

        pattern_order = list(range(len(FEATURE_PATTERNS)))
        RNG.shuffle(pattern_order)
        patch = None
        rotated = None
        chosen_pattern = None
        for pattern_idx in pattern_order:
            chosen_pattern = FEATURE_PATTERNS[pattern_idx]
            candidate = _make_patch(marker_color, turns, feature_color, chosen_pattern)
            candidate_rot = _rotate_clockwise(candidate, turns)
            if turns != 0 and np.array_equal(candidate, candidate_rot):
                continue
            patch = candidate
            rotated = candidate_rot
            break
        if patch is None or rotated is None or chosen_pattern is None:
            return None

        grid[r : r + 3, c : c + 3] = patch
        specs.append(
            {
                "id": idx,
                "r": int(r),
                "c": int(c),
                "dist": int(r + c),
                "turns": int(turns),
                "patch": patch,
                "rotated": rotated,
                "marker_color": int(marker_color),
                "feature_color": int(feature_color),
                "pattern": chosen_pattern,
            }
        )

    if not _validate_instance(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid direction-rotation-stacking ARC input")
