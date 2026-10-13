from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

_ROTATION_TO_AREAS = {
    0: (4, 8),
    1: (5,),
    2: (6,),
    3: (3, 7),
}
_GEN_RETRY_LIMIT = 500
_GRID_SIZE_RANGE = (10, 12)
_SEED_COUNT_CHOICES = (2, 3, 4)
_SPRITE_COLORS = (GREY,)
_HORIZONTAL_COLORS = (GREEN, BLUE)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _rotationally_asymmetric(mask: np.ndarray) -> bool:
    base = _crop_mask(mask)
    for k in (1, 2, 3):
        other = _crop_mask(np.rot90(base, k))
        if other.shape == base.shape and np.array_equal(other, base):
            return False
    return True


def _build_sprite_catalog() -> dict[int, list[np.ndarray]]:
    catalog: dict[int, list[np.ndarray]] = {area: [] for area in range(3, 9)}
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for area in range(3, 9):
        local_seen: set[tuple[tuple[int, ...], ...]] = set()
        for cells in combinations(range(16), area):
            full = np.zeros((4, 4), dtype=bool)
            for idx in cells:
                full[idx // 4, idx % 4] = True
            cropped = _crop_mask(full)
            h, w = cropped.shape
            if h < 2 or w < 2:
                continue
            if not _connected4(cropped):
                continue
            if not _rotationally_asymmetric(cropped):
                continue
            key = _mask_key(cropped)
            if key in local_seen or key in seen:
                continue
            local_seen.add(key)
            seen.add(key)
            catalog[area].append(cropped.astype(bool))
    return catalog


_SPRITE_CATALOG = _build_sprite_catalog()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> np.ndarray:
    out = grid.copy()
    h, w = mask.shape
    out[r0 : r0 + h, c0 : c0 + w][mask] = int(color)
    return out


def _apply_rays(rotated: np.ndarray) -> np.ndarray:
    g = np.asarray(rotated, dtype=int)
    h, w = g.shape
    grey = g == GREY
    out = np.zeros_like(g)

    for r, c in np.argwhere((g == GREEN) | (g == BLUE)):
        color = int(g[r, c])
        left_hits = np.where(grey[int(r), : int(c)])[0]
        right_hits = np.where(grey[int(r), int(c) + 1 :])[0]
        c0 = int(left_hits.max() + 1) if len(left_hits) else 0
        c1 = int(c + 1 + right_hits.min()) if len(right_hits) else w
        out[int(r), c0:c1] = color

    for r, c in np.argwhere(g == RED):
        up_hits = np.where(grey[: int(r), int(c)])[0]
        down_hits = np.where(grey[int(r) + 1 :, int(c)])[0]
        r0 = int(up_hits.max() + 1) if len(up_hits) else 0
        r1 = int(r + 1 + down_hits.min()) if len(down_hits) else h
        out[r0:r1, int(c)] = RED

    out[grey] = GREY
    return out


def _ray_geometry(rotated: np.ndarray, r: int, c: int) -> tuple[str, int, int]:
    g = np.asarray(rotated, dtype=int)
    if int(g[r, c]) == RED:
        grey_rows = np.where(g[:, int(c)] == GREY)[0]
        above = grey_rows[grey_rows < int(r)]
        below = grey_rows[grey_rows > int(r)]
        top = int(above.max() + 1) if len(above) else 0
        bottom = int(below.min()) if len(below) else g.shape[0]
        return "v", top, bottom
    grey_cols = np.where(g[int(r), :] == GREY)[0]
    left = grey_cols[grey_cols < int(c)]
    right = grey_cols[grey_cols > int(c)]
    start = int(left.max() + 1) if len(left) else 0
    stop = int(right.min()) if len(right) else g.shape[1]
    return "h", start, stop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the grid by a grey sprite's pixel count, then extend color-coded seed rays that stop at the rotated sprite.

    Input structure:
    - The grid is a square black canvas containing one grey 4-connected sprite near the center and 2-4 single-pixel seeds.
    - Seed colors are red, green, or blue, and no seed touches the grey sprite.
    - The grey sprite is the only obstacle object and its foreground size is between 3 and 8 pixels.

    Rule:
    - Count the grey sprite pixels; let R be that count modulo 4.
    - Rotate the entire grid clockwise by R quarter-turns.
    - On the rotated grid, each red seed fills its column, while each green or blue seed fills its row.
    - A ray stops immediately before the first grey sprite pixel in that direction and never overwrites grey.
    - If vertical red and horizontal green/blue rays cross, the crossing cell is red.

    Key insight:
    - The same grey sprite serves two roles at once: its area chooses the rotation, and its rotated shape then changes which rays are truncated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    grey_count = int(np.count_nonzero(g == GREY))
    rotation = grey_count % 4
    rotated = np.rot90(g, -rotation)
    return _apply_rays(rotated)


def _sprite_positions(n: int, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    center = (n - 1) / 2.0
    positions = []
    for r0 in range(1, n - h):
        for c0 in range(1, n - w):
            sprite_center_r = r0 + (h - 1) / 2.0
            sprite_center_c = c0 + (w - 1) / 2.0
            if abs(sprite_center_r - center) <= 1.5 and abs(sprite_center_c - center) <= 1.5:
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _seed_ok(r: int, c: int, forbidden: np.ndarray, used: list[tuple[int, int]]) -> bool:
    if forbidden[int(r), int(c)]:
        return False
    for ur, uc in used:
        if max(abs(int(r) - int(ur)), abs(int(c) - int(uc))) <= 1:
            return False
    return True


def _enumerate_open_red(forbidden: np.ndarray, sprite_mask: np.ndarray, used: list[tuple[int, int]]) -> list[tuple[int, int]]:
    n = forbidden.shape[0]
    cols = [c for c in range(n) if not sprite_mask[:, c].any()]
    out = []
    for c in cols:
        for r in range(n):
            if _seed_ok(r, c, forbidden, used):
                out.append((r, c))
    random.shuffle(out)
    return out


def _enumerate_blocked_red(forbidden: np.ndarray, sprite_mask: np.ndarray, used: list[tuple[int, int]]) -> list[tuple[int, int]]:
    n = forbidden.shape[0]
    out = []
    for c in range(n):
        rows = np.where(sprite_mask[:, c])[0]
        if len(rows) == 0:
            continue
        top = int(rows.min())
        bottom = int(rows.max())
        for r in range(0, top):
            if _seed_ok(r, c, forbidden, used):
                out.append((r, c))
        for r in range(bottom + 1, n):
            if _seed_ok(r, c, forbidden, used):
                out.append((r, c))
    random.shuffle(out)
    return out


def _enumerate_open_horizontal(
    forbidden: np.ndarray,
    sprite_mask: np.ndarray,
    used: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    n = forbidden.shape[0]
    rows = [r for r in range(n) if not sprite_mask[r, :].any()]
    out = []
    for r in rows:
        for c in range(n):
            if _seed_ok(r, c, forbidden, used):
                out.append((r, c))
    random.shuffle(out)
    return out


def _enumerate_blocked_horizontal(
    forbidden: np.ndarray,
    sprite_mask: np.ndarray,
    used: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    n = forbidden.shape[0]
    out = []
    for r in range(n):
        cols = np.where(sprite_mask[r, :])[0]
        if len(cols) == 0:
            continue
        left = int(cols.min())
        right = int(cols.max())
        for c in range(0, left):
            if _seed_ok(r, c, forbidden, used):
                out.append((r, c))
        for c in range(right + 1, n):
            if _seed_ok(r, c, forbidden, used):
                out.append((r, c))
    random.shuffle(out)
    return out


def _pick_open_cross_pair(
    forbidden: np.ndarray,
    sprite_mask: np.ndarray,
    used: list[tuple[int, int]],
    row_color: int,
) -> list[tuple[int, int, int]] | None:
    n = forbidden.shape[0]
    open_cols = [c for c in range(n) if not sprite_mask[:, c].any()]
    open_rows = [r for r in range(n) if not sprite_mask[r, :].any()]
    random.shuffle(open_cols)
    random.shuffle(open_rows)
    for col in open_cols:
        for row in open_rows:
            red_rows = [r for r in range(n) if r != row and _seed_ok(r, col, forbidden, used)]
            random.shuffle(red_rows)
            for red_row in red_rows:
                used2 = used + [(red_row, col)]
                row_cols = [c for c in range(n) if c != col and _seed_ok(row, c, forbidden, used2)]
                random.shuffle(row_cols)
                if row_cols:
                    return [(red_row, col, RED), (row, row_cols[0], row_color)]
    return None


def _pick_seed_by_role(
    role: str,
    forbidden: np.ndarray,
    sprite_mask: np.ndarray,
    used: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if role == "open_red":
        return _enumerate_open_red(forbidden, sprite_mask, used)
    if role == "blocked_red":
        return _enumerate_blocked_red(forbidden, sprite_mask, used)
    if role == "open_horizontal":
        return _enumerate_open_horizontal(forbidden, sprite_mask, used)
    if role == "blocked_horizontal":
        return _enumerate_blocked_horizontal(forbidden, sprite_mask, used)
    raise ValueError(role)


def _seed_signature(seeds: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((int(r), int(c), int(color)) for r, c, color in seeds))


def _build_scene(n: int, sprite_mask: np.ndarray) -> np.ndarray | None:
    scene = np.zeros((n, n), dtype=int)
    placements = _sprite_positions(n, sprite_mask)
    for r0, c0 in placements:
        sprite = _place_mask(scene, sprite_mask, GREY, r0, c0)
        obstacle = sprite == GREY
        forbidden = _dilate8(obstacle)
        seed_count = random.choice(_SEED_COUNT_CHOICES)
        used: list[tuple[int, int]] = []
        seeds: list[tuple[int, int, int]] = []

        if seed_count >= 3:
            row_color = random.choice(_HORIZONTAL_COLORS)
            cross_pair = _pick_open_cross_pair(forbidden, obstacle, used, row_color)
            if cross_pair is None:
                continue
            for r, c, color in cross_pair:
                seeds.append((r, c, color))
                used.append((r, c))

            blocked_options = [
                (RED, "blocked_red"),
                (GREEN, "blocked_horizontal"),
                (BLUE, "blocked_horizontal"),
            ]
            random.shuffle(blocked_options)
            placed_blocked = False
            for color, role in blocked_options:
                candidates = _pick_seed_by_role(role, forbidden, obstacle, used)
                for r, c in candidates:
                    if color == RED and any(uc == c and scene[ur, uc] == RED for ur, uc in used):
                        continue
                    seeds.append((r, c, color))
                    used.append((r, c))
                    placed_blocked = True
                    break
                if placed_blocked:
                    break
            if not placed_blocked:
                continue
        else:
            blocked_options = [
                (RED, "blocked_red"),
                (GREEN, "blocked_horizontal"),
                (BLUE, "blocked_horizontal"),
            ]
            open_options = [
                (RED, "open_red"),
                (GREEN, "open_horizontal"),
                (BLUE, "open_horizontal"),
            ]
            random.shuffle(blocked_options)
            random.shuffle(open_options)
            placed_blocked = False
            for color, role in blocked_options:
                for r, c in _pick_seed_by_role(role, forbidden, obstacle, used):
                    seeds.append((r, c, color))
                    used.append((r, c))
                    placed_blocked = True
                    break
                if placed_blocked:
                    break
            if not placed_blocked:
                continue

            placed_open = False
            for color, role in open_options:
                for r, c in _pick_seed_by_role(role, forbidden, obstacle, used):
                    if color == RED and any(prev_c == c for _, prev_c, prev_color in seeds if prev_color == RED):
                        continue
                    if color in _HORIZONTAL_COLORS and any(prev_r == r and prev_color in _HORIZONTAL_COLORS for prev_r, _, prev_color in seeds):
                        continue
                    seeds.append((r, c, color))
                    used.append((r, c))
                    placed_open = True
                    break
                if placed_open:
                    break
            if not placed_open:
                continue

        while len(seeds) < seed_count:
            missing_horizontal = [color for color in _HORIZONTAL_COLORS if color not in {seed[2] for seed in seeds}]
            extra_specs = []
            if missing_horizontal:
                extra_specs.append((missing_horizontal[0], "open_horizontal"))
                extra_specs.append((missing_horizontal[0], "blocked_horizontal"))
            extra_specs.extend(
                [
                    (RED, "open_red"),
                    (RED, "blocked_red"),
                    (GREEN, "open_horizontal"),
                    (GREEN, "blocked_horizontal"),
                    (BLUE, "open_horizontal"),
                    (BLUE, "blocked_horizontal"),
                ]
            )
            random.shuffle(extra_specs)
            placed_extra = False
            for color, role in extra_specs:
                for r, c in _pick_seed_by_role(role, forbidden, obstacle, used):
                    if color == RED and any(prev_c == c for _, prev_c, prev_color in seeds if prev_color == RED):
                        continue
                    if color in _HORIZONTAL_COLORS and any(prev_r == r and prev_color in _HORIZONTAL_COLORS for prev_r, _, prev_color in seeds):
                        continue
                    seeds.append((r, c, color))
                    used.append((r, c))
                    placed_extra = True
                    break
                if placed_extra:
                    break
            if not placed_extra:
                break

        if len(seeds) != seed_count:
            continue
        out = sprite.copy()
        for r, c, color in seeds:
            out[int(r), int(c)] = int(color)
        if len(set((r, c) for r, c, _ in seeds)) != len(seeds):
            continue
        return out
    return None


def _rotate_input_from_scene(scene: np.ndarray, rotation: int) -> np.ndarray:
    return np.rot90(scene, rotation).copy()


def _validate_scene(scene: np.ndarray, rotation: int) -> bool:
    obstacle = scene == GREY
    if int(obstacle.sum()) not in range(3, 9):
        return False
    if not _connected4(obstacle):
        return False
    if not _rotationally_asymmetric(_crop_mask(obstacle)):
        return False

    seeds = [(int(r), int(c), int(scene[r, c])) for r, c in np.argwhere((scene == RED) | (scene == GREEN) | (scene == BLUE))]
    if not (2 <= len(seeds) <= 4):
        return False

    forbidden = _dilate8(obstacle)
    for r, c, _ in seeds:
        if forbidden[r, c] and not obstacle[r, c]:
            return False

    blocked = 0
    unblocked = 0
    for r, c, color in seeds:
        orient, start, stop = _ray_geometry(scene, r, c)
        if orient == "v":
            open_full = not obstacle[:, c].any()
            length = stop - start
        else:
            open_full = not obstacle[r, :].any()
            length = stop - start
        if length < 3:
            return False
        if open_full:
            unblocked += 1
        else:
            blocked += 1

    if blocked < 1 or unblocked < 1:
        return False

    output = _apply_rays(scene)
    if np.array_equal(output, scene):
        return False
    if np.array_equal(output, _apply_rays(np.where(scene == GREY, scene, 0))):
        return False

    input_grid = _rotate_input_from_scene(scene, rotation)
    if not np.array_equal(main(input_grid), output):
        return False

    wrong = 0
    for k in range(4):
        candidate_scene = np.rot90(input_grid, -k)
        candidate_output = _apply_rays(candidate_scene)
        if k != rotation and not np.array_equal(candidate_output, output):
            wrong += 1
    if wrong < 2:
        return False

    if np.array_equal(output, np.rot90(input_grid, -rotation)):
        return False

    has_red = any(color == RED for _, _, color in seeds)
    has_horizontal = any(color in _HORIZONTAL_COLORS for _, _, color in seeds)
    if has_red and has_horizontal:
        red_cols = [c for _, c, color in seeds if color == RED]
        horiz_rows = [r for r, _, color in seeds if color in _HORIZONTAL_COLORS]
        if any(output[r, c] == RED for r in horiz_rows for c in red_cols if not obstacle[r, c]):
            return True
    return len(seeds) == 2


def _attempt_generate() -> np.ndarray | None:
    n = random.randint(*_GRID_SIZE_RANGE)
    rotation = random.randint(0, 3)
    area = random.choice(_ROTATION_TO_AREAS[rotation])
    sprite_mask = random.choice(_SPRITE_CATALOG[area]).copy()
    scene = _build_scene(n, sprite_mask)
    if scene is None:
        return None
    if not _validate_scene(scene, rotation):
        return None
    input_grid = _rotate_input_from_scene(scene, rotation)
    if int(np.count_nonzero(input_grid == GREY)) % 4 != rotation:
        return None
    return input_grid.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-ray ARC input")
