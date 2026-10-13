from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

TL, TR, BL, BR = 0, 1, 2, 3
TERRITORY_OUTWARD_DIRS = {
    TL: {UP, LEFT},
    TR: {UP, RIGHT},
    BL: {DOWN, LEFT},
    BR: {DOWN, RIGHT},
}

BASE_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
)

PAIR_COUNT_CHOICES = (3, 4)
CELL_SIDE_RANGE = (9, 11)
GAP_RANGE = (3, 4)
MOVE_LEN_RANGE = (1, 4)
ARM_LEN_RANGE = (1, 3)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    comps: list[dict] = []
    for color_raw in np.unique(g):
        color = int(color_raw)
        if color == bg:
            continue
        mask = g == color
        visited = np.zeros((H, W), dtype=bool)
        starts = np.argwhere(mask)
        for start in starts:
            sr, sc = map(int, start)
            if visited[sr, sc]:
                continue
            stack = [(sr, sc)]
            visited[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in DIRS:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _analyze_incomplete_cross(coords: np.ndarray) -> dict | None:
    cells = {(int(r), int(c)) for r, c in coords}
    matches: list[dict] = []
    for center in cells:
        cr, cc = center
        if not all(r == cr or c == cc for r, c in cells):
            continue
        lengths: dict[tuple[int, int], int] = {}
        total = 1
        for dr, dc in DIRS:
            length = 0
            rr = cr + dr
            cc2 = cc + dc
            while (rr, cc2) in cells:
                length += 1
                rr += dr
                cc2 += dc
            lengths[(dr, dc)] = length
            total += length
        if total != len(cells):
            continue
        missing = [d for d in DIRS if lengths[d] == 0]
        if len(missing) != 1:
            continue
        missing_dir = missing[0]
        distance = int(lengths[OPPOSITE[missing_dir]])
        if distance <= 0:
            continue
        matches.append(
            {
                "center": (cr, cc),
                "missing": missing_dir,
                "distance": distance,
                "lengths": lengths,
            }
        )
    if len(matches) != 1:
        return None
    return matches[0]


def _min_manhattan(a: np.ndarray, b: np.ndarray) -> int:
    diff = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    return int(diff.min())


def _min_chebyshev(a: np.ndarray, b: np.ndarray) -> int:
    diff = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2)
    return int(diff.min())


def _pair_markers_to_sprites(markers: list[dict], sprites: list[dict]) -> list[int]:
    n = len(markers)
    assert n == len(sprites) and n > 0
    costs = tuple(
        tuple(_min_manhattan(marker["coords"], sprite["coords"]) for sprite in sprites)
        for marker in markers
    )

    @lru_cache(maxsize=None)
    def dp(i: int, used_mask: int) -> tuple[int, tuple[int, ...], int]:
        if i == n:
            return 0, (), 1
        best_cost: int | None = None
        best_assign: tuple[int, ...] = ()
        ways = 0
        for j in range(n):
            if used_mask & (1 << j):
                continue
            sub_cost, sub_assign, sub_ways = dp(i + 1, used_mask | (1 << j))
            total = int(costs[i][j]) + sub_cost
            candidate = (j,) + sub_assign
            if best_cost is None or total < best_cost:
                best_cost = total
                best_assign = candidate
                ways = sub_ways
            elif total == best_cost:
                if candidate < best_assign:
                    best_assign = candidate
                ways = min(2, ways + sub_ways)
        assert best_cost is not None
        return best_cost, best_assign, ways

    _best_cost, assignment, ways = dp(0, 0)
    assert ways == 1, "marker-sprite pairing must be unambiguous"
    return list(assignment)


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for turns in range(4):
            variant = np.rot90(base, turns)
            key = variant.tobytes()
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


ALL_SPRITE_VARIANTS = [variant for base in BASE_SHAPES for variant in _mask_variants(base)]


def _marker_mask(missing: tuple[int, int], distance: int, arm_a: int, arm_b: int) -> np.ndarray:
    lengths = {d: 0 for d in DIRS}
    lengths[missing] = 0
    lengths[OPPOSITE[missing]] = int(distance)
    orthogonal = [d for d in DIRS if d not in (missing, OPPOSITE[missing])]
    lengths[orthogonal[0]] = int(arm_a)
    lengths[orthogonal[1]] = int(arm_b)

    cells = [(0, 0)]
    for dr, dc in DIRS:
        for step in range(1, lengths[(dr, dc)] + 1):
            cells.append((dr * step, dc * step))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = [(r - min_r, c - min_c) for r, c in cells]
    H = max(r for r, _ in norm) + 1
    W = max(c for _, c in norm) + 1
    mask = np.zeros((H, W), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _territory_boxes(cell_h: int, cell_w: int, gap_h: int, gap_w: int) -> tuple[tuple[int, int, int, int], ...]:
    top = 0
    left = 0
    mid_r = cell_h + gap_h
    mid_c = cell_w + gap_w
    return (
        (top, left, cell_h - 1, cell_w - 1),
        (top, mid_c, cell_h - 1, mid_c + cell_w - 1),
        (mid_r, left, mid_r + cell_h - 1, cell_w - 1),
        (mid_r, mid_c, mid_r + cell_h - 1, mid_c + cell_w - 1),
    )


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    return np.argwhere(mask).astype(int) + np.array([top, left], dtype=int)


def _translate_visible(coords: np.ndarray, dr: int, dc: int, amount: int, shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    shifted = coords + np.array([dr * amount, dc * amount], dtype=int)
    keep = (
        (0 <= shifted[:, 0])
        & (shifted[:, 0] < H)
        & (0 <= shifted[:, 1])
        & (shifted[:, 1] < W)
    )
    return shifted[keep]


def _origin_candidates(
    territory: tuple[int, int, int, int],
    sprite_mask: np.ndarray,
    direction: tuple[int, int],
    amount: int,
    clip: bool,
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = territory
    H, W = grid_shape
    h, w = sprite_mask.shape
    dr, dc = direction
    rows = range(r0, r1 - h + 2)
    cols = range(c0, c1 - w + 2)
    candidates: list[tuple[int, int]] = []
    for top in rows:
        for left in cols:
            if not clip:
                target_top = top + dr * amount
                target_left = left + dc * amount
                if not (r0 <= target_top and target_top + h - 1 <= r1):
                    continue
                if not (c0 <= target_left and target_left + w - 1 <= c1):
                    continue
            else:
                if dr < 0 and not (top - amount < 0 and top + h - 1 - amount >= 0):
                    continue
                if dr > 0 and not (top + amount <= H - 1 and top + amount + h - 1 >= H):
                    continue
                if dc < 0 and not (left - amount < 0 and left + w - 1 - amount >= 0):
                    continue
                if dc > 0 and not (left + amount <= W - 1 and left + amount + w - 1 >= W):
                    continue
            candidates.append((top, left))
    return candidates


def _choose_sprite_mask(direction: tuple[int, int], clip: bool) -> np.ndarray | None:
    choices = ALL_SPRITE_VARIANTS[:]
    random.shuffle(choices)
    for mask in choices:
        if clip:
            extent = mask.shape[0] if direction in (UP, DOWN) else mask.shape[1]
            if extent < 2:
                continue
        return mask.copy()
    return None


def _build_pair_spec(
    territory: tuple[int, int, int, int],
    direction: tuple[int, int],
    clip: bool,
    grid_shape: tuple[int, int],
    color: int,
) -> dict | None:
    for _ in range(120):
        sprite_mask = _choose_sprite_mask(direction, clip)
        if sprite_mask is None:
            continue
        h, w = sprite_mask.shape
        if not clip:
            slack = (territory[2] - territory[0] + 1 - h) if direction in (UP, DOWN) else (territory[3] - territory[1] + 1 - w)
            if slack < 1:
                continue
            max_amount = min(MOVE_LEN_RANGE[1], slack)
            amount = random.randint(MOVE_LEN_RANGE[0], max_amount)
        else:
            amount = random.randint(*MOVE_LEN_RANGE)

        arm_a = random.randint(*ARM_LEN_RANGE)
        arm_b = random.randint(*ARM_LEN_RANGE)
        marker_mask = _marker_mask(direction, amount, arm_a, arm_b)
        th = territory[2] - territory[0] + 1
        tw = territory[3] - territory[1] + 1
        if marker_mask.shape[0] > th or marker_mask.shape[1] > tw:
            continue

        candidates = _origin_candidates(territory, sprite_mask, direction, amount, clip, grid_shape)
        if not candidates:
            continue
        top, left = random.choice(candidates)
        sprite_coords = _coords_from_mask(sprite_mask, top, left)
        moved_coords = _translate_visible(sprite_coords, direction[0], direction[1], amount, grid_shape)
        if clip:
            if not (0 < moved_coords.shape[0] < sprite_coords.shape[0]):
                continue
        else:
            if moved_coords.shape[0] != sprite_coords.shape[0]:
                continue

        return {
            "direction": direction,
            "clip": clip,
            "distance": amount,
            "color": int(color),
            "sprite_mask": sprite_mask,
            "sprite_top_left": (top, left),
            "sprite_coords": sprite_coords,
            "moved_coords": moved_coords,
            "marker_mask": marker_mask,
            "territory": territory,
        }
    return None


def _marker_candidates(spec: dict, all_specs: list[dict]) -> list[tuple[int, int, int]]:
    territory = spec["territory"]
    marker_mask = spec["marker_mask"]
    own_sprite = spec["sprite_coords"]
    r0, c0, r1, c1 = territory
    mh, mw = marker_mask.shape
    candidates: list[tuple[int, int, int]] = []
    for top in range(r0, r1 - mh + 2):
        for left in range(c0, c1 - mw + 2):
            marker_coords = _coords_from_mask(marker_mask, top, left)
            if _min_chebyshev(marker_coords, own_sprite) < 2:
                continue
            own_dist = _min_manhattan(marker_coords, own_sprite)
            if not (2 <= own_dist <= 6):
                continue
            other_dists = [
                _min_manhattan(marker_coords, other["sprite_coords"])
                for other in all_specs
                if other is not spec
            ]
            margin = min(other_dists) - own_dist if other_dists else 9
            if margin < 2:
                continue
            candidates.append((margin, top, left))
    return candidates


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _render_instance(shape: tuple[int, int], bg: int, marker_color: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        _paint_coords(inp, spec["sprite_coords"], spec["color"])
        _paint_coords(inp, spec["marker_coords"], marker_color)
        if spec["moved_coords"].size:
            _paint_coords(out, spec["moved_coords"], spec["color"])
    return inp, out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    comps = _find_monochrome_components(inp, _get_bg(inp))
    markers = [comp for comp in comps if _analyze_incomplete_cross(comp["coords"]) is not None]
    sprites = [comp for comp in comps if _analyze_incomplete_cross(comp["coords"]) is None]
    if len(markers) != len(specs) or len(sprites) != len(specs):
        return False

    for i, spec in enumerate(specs):
        own = _min_manhattan(spec["marker_coords"], spec["sprite_coords"])
        others = [
            _min_manhattan(spec["marker_coords"], other["sprite_coords"])
            for j, other in enumerate(specs)
            if i != j
        ]
        if others and own >= min(others):
            return False
        if spec["clip"]:
            if not (0 < spec["moved_coords"].shape[0] < spec["sprite_coords"].shape[0]):
                return False
        else:
            if spec["moved_coords"].shape[0] != spec["sprite_coords"].shape[0]:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite along the missing-arm direction encoded by its paired incomplete cross marker.

    Input structure:
    - The background is the most common color.
    - Foreground components are separated monochrome sprites plus one-pixel-thick incomplete cross markers.
    - Every marker has exactly one junction, exactly three present arms, and sits near exactly one sprite.

    Rule:
    - For each incomplete cross, find the missing arm direction.
    - Read the opposite arm length L, erase all markers, and translate the paired nearby sprite by L cells in the missing direction.
    - Clip translated sprite pixels that leave the grid; all remaining background stays unchanged.

    Key insight:
    - The cross does not point to a target location: the missing arm gives the motion direction, and the opposite arm length gives the motion distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _find_monochrome_components(g, bg)
    markers: list[dict] = []
    sprites: list[dict] = []
    for comp in components:
        marker_info = _analyze_incomplete_cross(comp["coords"])
        if marker_info is None:
            sprites.append(comp)
        else:
            markers.append({**comp, **marker_info})

    assert markers, "expected at least one incomplete cross marker"
    assert len(markers) == len(sprites), "every marker must pair with exactly one sprite"

    assignment = _pair_markers_to_sprites(markers, sprites)

    out = g.copy()
    for comp in components:
        rr = comp["coords"][:, 0]
        cc = comp["coords"][:, 1]
        out[rr, cc] = bg

    painted = np.zeros(g.shape, dtype=bool)
    H, W = g.shape
    for marker, sprite_idx in zip(markers, assignment):
        sprite = sprites[sprite_idx]
        dr, dc = marker["missing"]
        amount = int(marker["distance"])
        shifted = sprite["coords"] + np.array([dr * amount, dc * amount], dtype=int)
        keep = (
            (0 <= shifted[:, 0])
            & (shifted[:, 0] < H)
            & (0 <= shifted[:, 1])
            & (shifted[:, 1] < W)
        )
        visible = shifted[keep]
        if visible.size == 0:
            continue
        rr = visible[:, 0]
        cc = visible[:, 1]
        assert not painted[rr, cc].any(), "translated sprites must not overlap"
        out[rr, cc] = int(sprite["color"])
        painted[rr, cc] = True

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        cell_h = random.randint(*CELL_SIDE_RANGE)
        cell_w = random.randint(*CELL_SIDE_RANGE)
        gap_h = random.randint(*GAP_RANGE)
        gap_w = random.randint(*GAP_RANGE)
        H = 2 * cell_h + gap_h
        W = 2 * cell_w + gap_w
        if max(H, W) > 30:
            continue

        pair_count = random.choice(PAIR_COUNT_CHOICES)
        territories = _territory_boxes(cell_h, cell_w, gap_h, gap_w)
        active = random.sample((TL, TR, BL, BR), pair_count)
        directions = random.sample(DIRS, pair_count)

        clip_dir = random.choice(directions) if random.random() < 0.6 else None
        mapping: dict[int, tuple[int, int]] = {}
        remaining_dirs = directions[:]
        remaining_territories = active[:]
        if clip_dir is not None:
            clip_territories = [t for t in active if clip_dir in TERRITORY_OUTWARD_DIRS[t]]
            clip_territory = random.choice(clip_territories)
            mapping[clip_territory] = clip_dir
            remaining_dirs.remove(clip_dir)
            remaining_territories.remove(clip_territory)
        random.shuffle(remaining_dirs)
        random.shuffle(remaining_territories)
        for territory_idx, direction in zip(remaining_territories, remaining_dirs):
            mapping[territory_idx] = direction

        palette = list(range(10))
        bg = random.choice(palette)
        marker_color = random.choice([c for c in palette if c != bg])
        sprite_colors = random.sample([c for c in palette if c not in {bg, marker_color}], pair_count)

        specs: list[dict] = []
        ok = True
        for territory_idx, color in zip(active, sprite_colors):
            direction = mapping[territory_idx]
            clip = clip_dir is not None and direction == clip_dir and territory_idx in mapping and mapping[territory_idx] == clip_dir
            spec = _build_pair_spec(territories[territory_idx], direction, clip, (H, W), color)
            if spec is None:
                ok = False
                break
            specs.append(spec)
        if not ok:
            continue

        for spec in specs:
            candidates = _marker_candidates(spec, specs)
            if not candidates:
                ok = False
                break
            best_margin = max(candidate[0] for candidate in candidates)
            pool = [candidate for candidate in candidates if candidate[0] >= best_margin - 1]
            _, top, left = random.choice(pool)
            spec["marker_top_left"] = (top, left)
            spec["marker_coords"] = _coords_from_mask(spec["marker_mask"], top, left)
        if not ok:
            continue

        inp, expected = _render_instance((H, W), bg, marker_color, specs)
        if _validate_semantics(inp, expected, specs):
            return inp

    raise RuntimeError("failed to generate a valid phantom-arm ARC input")
