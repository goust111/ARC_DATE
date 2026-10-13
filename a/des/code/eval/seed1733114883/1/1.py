from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
LENS_COLOR_POOL = [1, 2, 3, 5, 6, 8, 9]
SPRITE_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300
MAX_GRID_HEIGHT = 24
MAX_GRID_WIDTH = 30


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _monochromatic_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            components.append({"color": color, "coords": np.array(coords, dtype=int)})

    return components


def _find_lenses(grid: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray]:
    h, w = grid.shape
    lens_mask = np.zeros((h, w), dtype=bool)
    lenses: list[dict[str, int]] = []

    for comp in _monochromatic_components(grid):
        coords = comp["coords"]
        color = int(comp["color"])
        rows = coords[:, 0]
        cols = coords[:, 1]
        area = len(coords)

        if len(set(map(int, rows))) == 1:
            row = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            if area >= 3 and area == c1 - c0 + 1:
                lens_mask[coords[:, 0], coords[:, 1]] = True
                lenses.append(
                    {
                        "orientation": 0,
                        "anchor": row,
                        "start": c0,
                        "end": c1,
                        "color": color,
                    }
                )
        elif len(set(map(int, cols))) == 1:
            col = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            if area >= 3 and area == r1 - r0 + 1:
                lens_mask[coords[:, 0], coords[:, 1]] = True
                lenses.append(
                    {
                        "orientation": 1,
                        "anchor": col,
                        "start": r0,
                        "end": r1,
                        "color": color,
                    }
                )

    assert lenses, "expected at least one lens"
    lens_colors = [int(lens["color"]) for lens in lenses]
    assert len(set(lens_colors)) == len(lens_colors), "lens colors must be unique"
    return lenses, lens_mask


def _find_sprites(grid: np.ndarray, lens_mask: np.ndarray) -> list[np.ndarray]:
    sprite_mask = (grid != BLACK) & ~lens_mask
    sprites = _mask_components(sprite_mask)
    assert sprites, "expected at least one sprite"

    for coords in sprites:
        assert 2 <= len(coords) <= 4, "sprites must have 2-4 pixels"
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) >= 2, "sprites must be multi-color"

    return sprites


def _mirror_point(lens: dict[str, int], r: int, c: int) -> tuple[int, int]:
    if int(lens["orientation"]) == 0:
        return 2 * int(lens["anchor"]) - int(r), int(c)
    return int(r), 2 * int(lens["anchor"]) - int(c)


def _candidate_distance(
    grid: np.ndarray, sprite: np.ndarray, lens: dict[str, int]
) -> int | None:
    h, w = grid.shape
    sprite_set = {tuple(map(int, rc)) for rc in sprite}
    rows = sprite[:, 0]
    cols = sprite[:, 1]

    if int(lens["orientation"]) == 0:
        row = int(lens["anchor"])
        c0 = int(lens["start"])
        c1 = int(lens["end"])
        if int(cols.min()) < c0 or int(cols.max()) > c1:
            return None
        if int(rows.max()) < row:
            step_to_lens = 1
        elif int(rows.min()) > row:
            step_to_lens = -1
        else:
            return None

        for r_raw, c_raw in sprite:
            r = int(r_raw)
            c = int(c_raw)
            rr, cc = _mirror_point(lens, r, c)
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            if int(grid[rr, cc]) != BLACK:
                return None

            for pr in range(r + step_to_lens, row, step_to_lens):
                if int(grid[pr, c]) != BLACK and (pr, c) not in sprite_set:
                    return None

            step_to_target = 1 if rr > row else -1
            for pr in range(row + step_to_target, rr, step_to_target):
                if int(grid[pr, c]) != BLACK:
                    return None

        return int(np.min(np.abs(rows - row)))

    col = int(lens["anchor"])
    r0 = int(lens["start"])
    r1 = int(lens["end"])
    if int(rows.min()) < r0 or int(rows.max()) > r1:
        return None
    if int(cols.max()) < col:
        step_to_lens = 1
    elif int(cols.min()) > col:
        step_to_lens = -1
    else:
        return None

    for r_raw, c_raw in sprite:
        r = int(r_raw)
        c = int(c_raw)
        rr, cc = _mirror_point(lens, r, c)
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if int(grid[rr, cc]) != BLACK:
            return None

        for pc in range(c + step_to_lens, col, step_to_lens):
            if int(grid[r, pc]) != BLACK and (r, pc) not in sprite_set:
                return None

        step_to_target = 1 if cc > col else -1
        for pc in range(col + step_to_target, cc, step_to_target):
            if int(grid[r, pc]) != BLACK:
                return None

    return int(np.min(np.abs(cols - col)))


def _scene_matches(grid: np.ndarray) -> list[tuple[np.ndarray, dict[str, int]]]:
    lenses, lens_mask = _find_lenses(grid)
    sprites = _find_sprites(grid, lens_mask)
    matches: list[tuple[np.ndarray, dict[str, int]]] = []

    for sprite in sprites:
        candidates: list[tuple[int, dict[str, int]]] = []
        for lens in lenses:
            distance = _candidate_distance(grid, sprite, lens)
            if distance is not None:
                candidates.append((distance, lens))

        assert candidates, "each sprite must see a compatible lens"
        min_distance = min(distance for distance, _lens in candidates)
        best = [lens for distance, lens in candidates if distance == min_distance]
        assert len(best) == 1, "nearest compatible lens must be unique"
        matches.append((sprite, best[0]))

    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each sprite across its nearest lens and recolor the projection.

    Rule
    - Detect every monochrome horizontal or vertical lens.
    - For each multi-color sprite, find the unique nearest compatible lens along
      the lens normal.
    - Mirror the sprite across that lens.
    - If a sprite pixel matches the lens color, project it as yellow; otherwise
      project it as the lens color.
    - Keep the original scene unchanged and only draw the mirrored projection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    out = g.copy()
    for sprite, lens in _scene_matches(g):
        lens_color = int(lens["color"])
        for r_raw, c_raw in sprite:
            r = int(r_raw)
            c = int(c_raw)
            rr, cc = _mirror_point(lens, r, c)
            color = YELLOW if int(g[r, c]) == lens_color else lens_color
            assert int(out[rr, cc]) in {BLACK, color}, "projections must land on empty target cells"
            out[rr, cc] = color

    return out


def _random_polyomino(size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    normalized = {(r - min_r, c - min_c) for r, c in cells}
    max_r = max(r for r, _c in normalized)
    max_c = max(c for _r, c in normalized)

    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in normalized:
        mask[r, c] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randint(0, 3))
    return mask.copy()


def _sprite_colors(size: int, lens_color: int, include_match: bool) -> list[int]:
    others = [color for color in SPRITE_COLOR_POOL if color != lens_color]
    if include_match:
        distinct = 2 if size <= 3 else random.choice([2, 3])
        chosen = [lens_color] + random.sample(others, distinct - 1)
    else:
        distinct = 2 if size <= 2 else random.randint(2, min(3, size))
        chosen = random.sample(others, distinct)

    counts = {color: 1 for color in chosen}
    remaining = size - len(chosen)
    while remaining > 0:
        options = [color for color, count in counts.items() if count < 2]
        counts[random.choice(options)] += 1
        remaining -= 1

    colors: list[int] = []
    for color, count in counts.items():
        colors.extend([int(color)] * int(count))
    random.shuffle(colors)

    if include_match:
        assert lens_color in colors and any(color != lens_color for color in colors)
    else:
        assert lens_color not in colors
    return colors


def _build_horizontal_scene() -> np.ndarray | None:
    sprite_count = random.randint(2, 4)
    lens_colors = random.sample(LENS_COLOR_POOL, sprite_count)

    match_total = sprite_count // 2
    if sprite_count % 2 == 1 and random.random() < 0.5:
        match_total += 1
    match_flags = [True] * match_total + [False] * (sprite_count - match_total)
    random.shuffle(match_flags)

    specs: list[dict[str, object]] = []
    for idx in range(sprite_count):
        mask = _random_polyomino(random.randint(2, 4))
        sh, sw = mask.shape
        dist = random.randint(2, 4)
        reach = dist + sh - 1
        lens_len = max(3, sw) + random.randint(0, 1)
        slot_w = lens_len + random.randint(0, 1)
        colors = _sprite_colors(int(mask.sum()), int(lens_colors[idx]), bool(match_flags[idx]))
        specs.append(
            {
                "mask": mask,
                "height": sh,
                "width": sw,
                "dist": dist,
                "reach": reach,
                "lens_len": lens_len,
                "slot_w": slot_w,
                "side_above": random.random() < 0.5,
                "lens_color": int(lens_colors[idx]),
                "colors": colors,
            }
        )

    gaps = [random.randint(1, 2) for _ in range(sprite_count - 1)]
    left_margin = random.randint(0, 1)
    right_margin = random.randint(0, 1)
    width = left_margin + right_margin + sum(int(spec["slot_w"]) for spec in specs) + sum(gaps)
    height_min = max(2 * int(spec["reach"]) + 1 for spec in specs)
    height = random.randint(height_min, height_min + 4)

    if width > MAX_GRID_WIDTH or height > MAX_GRID_HEIGHT:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    cursor = left_margin
    for idx, spec in enumerate(specs):
        slot_w = int(spec["slot_w"])
        lens_len = int(spec["lens_len"])
        lens_color = int(spec["lens_color"])
        mask = np.asarray(spec["mask"], dtype=bool)
        sh = int(spec["height"])
        sw = int(spec["width"])
        dist = int(spec["dist"])
        reach = int(spec["reach"])

        lens_c0 = cursor + random.randint(0, slot_w - lens_len)
        lens_row = random.randint(reach, height - 1 - reach)
        grid[lens_row, lens_c0 : lens_c0 + lens_len] = lens_color

        sprite_c0 = lens_c0 + random.randint(0, lens_len - sw)
        if bool(spec["side_above"]):
            sprite_r0 = lens_row - dist - sh + 1
        else:
            sprite_r0 = lens_row + dist

        coords = np.argwhere(mask)
        colors = list(spec["colors"])
        assert len(coords) == len(colors)
        for color, (dr_raw, dc_raw) in zip(colors, coords):
            dr = int(dr_raw)
            dc = int(dc_raw)
            grid[sprite_r0 + dr, sprite_c0 + dc] = int(color)

        cursor += slot_w
        if idx < len(gaps):
            cursor += int(gaps[idx])

    return grid


def _validated_scene() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_horizontal_scene()
        if grid is None:
            continue

        if random.random() < 0.5:
            grid = grid.T.copy()
        if random.random() < 0.5:
            grid = np.flipud(grid).copy()
        if random.random() < 0.5:
            grid = np.fliplr(grid).copy()

        matches = _scene_matches(grid)
        has_match = False
        has_diff = False
        total_sprite_pixels = 0
        for sprite, lens in matches:
            lens_color = int(lens["color"])
            total_sprite_pixels += len(sprite)
            for r, c in sprite:
                if int(grid[r, c]) == lens_color:
                    has_match = True
                else:
                    has_diff = True

        if not (has_match and has_diff):
            continue

        out = main(grid)
        if int(np.count_nonzero(out != grid)) != total_sprite_pixels:
            continue
        return grid

    raise RuntimeError("failed to generate a valid lens projection scene")


def generate_input() -> np.ndarray:
    return _validated_scene()
