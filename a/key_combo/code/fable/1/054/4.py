from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ORTHO4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_l_sprite(
    grid: np.ndarray, coords: list[tuple[int, int]], bg: int
) -> dict[str, object] | None:
    del bg
    if len(coords) < 5:
        return None

    color_counts = Counter(int(grid[r, c]) for r, c in coords)
    if len(color_counts) != 2:
        return None

    marker_colors = [color for color, count in color_counts.items() if count == 1]
    if len(marker_colors) != 1:
        return None

    marker_color = marker_colors[0]
    body_color = next(color for color in color_counts if color != marker_color)
    if color_counts[body_color] < 4:
        return None

    bend = next((r, c) for r, c in coords if int(grid[r, c]) == marker_color)
    br, bc = bend
    body = {(r, c) for r, c in coords if int(grid[r, c]) == body_color}
    if any(r != br and c != bc for r, c in body):
        return None

    row_cells = sorted(c for r, c in body if r == br)
    col_cells = sorted(r for r, c in body if c == bc)
    if len(row_cells) < 2 or len(col_cells) < 2:
        return None
    if any(c == bc for c in row_cells) or any(r == br for r in col_cells):
        return None

    h_signs = {1 if c > bc else -1 for c in row_cells}
    v_signs = {1 if r > br else -1 for r in col_cells}
    if len(h_signs) != 1 or len(v_signs) != 1:
        return None

    h_sign = h_signs.pop()
    v_sign = v_signs.pop()
    expected_row = {bc + h_sign * d for d in range(1, len(row_cells) + 1)}
    expected_col = {br + v_sign * d for d in range(1, len(col_cells) + 1)}
    expected_body = {(br, c) for c in expected_row} | {(r, bc) for r in expected_col}
    if body != expected_body:
        return None

    pixels = [(r, c, int(grid[r, c])) for r, c in coords]
    return {
        "body_color": int(body_color),
        "marker_color": int(marker_color),
        "bend": (int(br), int(bc)),
        "pixels": pixels,
        "coords": [(int(r), int(c)) for r, c in coords],
        "shape": (int(v_sign), int(h_sign), len(col_cells), len(row_cells)),
    }


def _find_cycles(edges: dict[int, int]) -> list[list[int]]:
    cycles: list[list[int]] = []
    done: set[int] = set()
    for start in edges:
        if start in done:
            continue
        path: list[int] = []
        first_seen: dict[int, int] = {}
        cur = start
        while cur in edges and cur not in first_seen and cur not in done:
            first_seen[cur] = len(path)
            path.append(cur)
            cur = edges[cur]
        done.update(path)
        if cur in first_seen:
            cycle = path[first_seen[cur] :]
            if len(cycle) >= 2:
                cycles.append(cycle)
    return cycles


def _extract_scene(
    grid: np.ndarray,
) -> tuple[int, list[dict[str, object]], dict[int, int], list[list[int]], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components4(g != bg)
    sprites: list[dict[str, object]] = []
    others: list[list[tuple[int, int]]] = []
    for comp in comps:
        sprite = _parse_l_sprite(g, comp, bg)
        if sprite is None:
            others.append(comp)
        else:
            sprites.append(sprite)

    body_to_idx = {int(sprite["body_color"]): idx for idx, sprite in enumerate(sprites)}
    edges: dict[int, int] = {}
    for idx, sprite in enumerate(sprites):
        target = body_to_idx.get(int(sprite["marker_color"]))
        if target is not None and target != idx:
            edges[idx] = int(target)
    return bg, sprites, edges, _find_cycles(edges), others


def _sprite_spec(
    body_color: int, marker_color: int, v_sign: int, h_sign: int, v_len: int, h_len: int
) -> dict[str, object]:
    rel_pixels = [(0, 0, int(marker_color))]
    rel_pixels.extend((v_sign * d, 0, int(body_color)) for d in range(1, v_len + 1))
    rel_pixels.extend((0, h_sign * d, int(body_color)) for d in range(1, h_len + 1))
    rows = [dr for dr, _dc, _color in rel_pixels]
    cols = [dc for _dr, dc, _color in rel_pixels]
    return {
        "body_color": int(body_color),
        "marker_color": int(marker_color),
        "rel_pixels": rel_pixels,
        "min_r": min(rows),
        "max_r": max(rows),
        "min_c": min(cols),
        "max_c": max(cols),
    }


def _abs_pixels(sprite: dict[str, object], bend: tuple[int, int]) -> list[tuple[int, int, int]]:
    br, bc = bend
    return [
        (br + int(dr), bc + int(dc), int(color))
        for dr, dc, color in sprite["rel_pixels"]  # type: ignore[index]
    ]


def _cells_only(pixels: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c, _color in pixels}


def _fits_without_touch(
    pixels: list[tuple[int, int, int]], shape: tuple[int, int], occupied: set[tuple[int, int]]
) -> bool:
    h, w = shape
    for r, c, _color in pixels:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for dr, dc in ORTHO4:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _candidate_bends(sprite: dict[str, object], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0 = -int(sprite["min_r"])
    r1 = h - 1 - int(sprite["max_r"])
    c0 = -int(sprite["min_c"])
    c1 = w - 1 - int(sprite["max_c"])
    bends = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(bends)
    return bends


def _paint_pixels(grid: np.ndarray, pixels: list[tuple[int, int, int]]) -> None:
    for r, c, color in pixels:
        grid[r, c] = int(color)


def _place_cycle_positions(
    cycle_sprites: list[dict[str, object]], shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    k = len(cycle_sprites)
    candidates = [_candidate_bends(sprite, shape) for sprite in cycle_sprites]
    positions: list[tuple[int, int] | None] = [None] * k

    def backtrack(
        idx: int, occ_in: set[tuple[int, int]], occ_out: set[tuple[int, int]]
    ) -> tuple[list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
        if idx == k:
            assert positions[0] is not None
            closing = _abs_pixels(cycle_sprites[-1], positions[0])
            if not _fits_without_touch(closing, shape, occ_out):
                return None
            final_positions = [pos for pos in positions if pos is not None]
            return final_positions, occ_in, occ_out | _cells_only(closing)

        sprite = cycle_sprites[idx]
        for bend in candidates[idx]:
            own_pixels = _abs_pixels(sprite, bend)
            if not _fits_without_touch(own_pixels, shape, occ_in):
                continue

            new_occ_out = occ_out
            if idx > 0:
                prev_pixels = _abs_pixels(cycle_sprites[idx - 1], bend)
                if not _fits_without_touch(prev_pixels, shape, occ_out):
                    continue
                new_occ_out = occ_out | _cells_only(prev_pixels)

            positions[idx] = bend
            result = backtrack(idx + 1, occ_in | _cells_only(own_pixels), new_occ_out)
            if result is not None:
                return result
            positions[idx] = None
        return None

    return backtrack(0, set(), set())


def _place_static_sprites(
    sprites: list[dict[str, object]],
    shape: tuple[int, int],
    occ_in: set[tuple[int, int]],
    occ_out: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    positions: list[tuple[int, int]] = []
    cur_in = set(occ_in)
    cur_out = set(occ_out)
    for sprite in sprites:
        placed = False
        for bend in _candidate_bends(sprite, shape):
            pixels = _abs_pixels(sprite, bend)
            if _fits_without_touch(pixels, shape, cur_in) and _fits_without_touch(
                pixels, shape, cur_out
            ):
                coords = _cells_only(pixels)
                positions.append(bend)
                cur_in |= coords
                cur_out |= coords
                placed = True
                break
        if not placed:
            return None
    return positions, cur_in, cur_out


def _place_noise_pixels(
    noise_colors: list[int],
    shape: tuple[int, int],
    occ_in: set[tuple[int, int]],
    occ_out: set[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    noise: list[tuple[int, int, int]] = []
    cur_in = set(occ_in)
    cur_out = set(occ_out)
    for color in noise_colors:
        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        placed = False
        for r, c in cells:
            pixels = [(r, c, int(color))]
            if _fits_without_touch(pixels, shape, cur_in) and _fits_without_touch(
                pixels, shape, cur_out
            ):
                noise.append((r, c, int(color)))
                cur_in.add((r, c))
                cur_out.add((r, c))
                placed = True
                break
        if not placed:
            return None
    return noise, cur_in, cur_out


def _expected_output(
    grid: np.ndarray,
    bg: int,
    cycle_sprites: list[dict[str, object]],
    cycle_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for sprite, bend in zip(cycle_sprites, cycle_positions):
        for r, c, _color in _abs_pixels(sprite, bend):
            out[r, c] = int(bg)
    for idx, sprite in enumerate(cycle_sprites):
        target_bend = cycle_positions[(idx + 1) % len(cycle_sprites)]
        _paint_pixels(out, _abs_pixels(sprite, target_bend))
    return out


def _wrong_sequential_output(
    grid: np.ndarray,
    bg: int,
    cycle_sprites: list[dict[str, object]],
    cycle_positions: list[tuple[int, int]],
) -> np.ndarray | None:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    cur_positions = list(cycle_positions)
    for idx, sprite in enumerate(cycle_sprites):
        for r, c, _color in _abs_pixels(sprite, cur_positions[idx]):
            out[r, c] = int(bg)
        target_bend = cur_positions[(idx + 1) % len(cycle_sprites)]
        moved_pixels = _abs_pixels(sprite, target_bend)
        if any(not (0 <= r < h and 0 <= c < w) for r, c, _color in moved_pixels):
            return None
        _paint_pixels(out, moved_pixels)
        cur_positions[idx] = target_bend
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each marker-linked L sprite onto the next sprite's former corner position.

    Input structure:
    - The background is the dominant color.
    - Foreground objects include several 4-connected two-color sprites: each sprite is a one-pixel marker at the L corner plus two same-color arms, with unique body colors.
    - Some marker colors equal another sprite's body color and therefore define pointers; extra sprites may have marker colors that match no body color, and isolated noise pixels may use unrelated colors.

    Rule:
    - Interpret a sprite as pointing to the unique sprite whose body color matches its marker color.
    - Find the closed pointer cycle and treat every sprite on that cycle as moving simultaneously.
    - Erase the cycle sprites from their old cells, then translate each one so its L corner lands on the pointed sprite's corner in the input snapshot, while keeping shape, orientation, body color, and embedded marker unchanged.

    Key insight:
    - The destination anchors are other sprites' old corner cells, so the move must be computed from the unchanged input snapshot rather than by chasing already-moved targets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, sprites, edges, cycles, _others = _extract_scene(g)
    assert cycles, "expected at least one pointer cycle"

    out = g.copy()
    moving = {idx for cycle in cycles for idx in cycle}
    for idx in moving:
        for r, c, _color in sprites[idx]["pixels"]:  # type: ignore[index]
            out[int(r), int(c)] = int(bg)

    for cycle in cycles:
        cycle_set = set(cycle)
        for idx in cycle:
            target = edges[idx]
            assert target in cycle_set
            sr, sc = sprites[idx]["bend"]  # type: ignore[index]
            tr, tc = sprites[target]["bend"]  # type: ignore[index]
            dr = int(tr) - int(sr)
            dc = int(tc) - int(sc)
            moved_pixels = [
                (int(r) + dr, int(c) + dc, int(color))
                for r, c, color in sprites[idx]["pixels"]  # type: ignore[index]
            ]
            _paint_pixels(out, moved_pixels)

    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(23, 30)
    w = random.randint(23, 30)
    bg = random.choice(PALETTE)
    usable = [color for color in PALETTE if color != bg]

    cycle_len = random.choice([2, 3, 4])
    max_decoys = min(2, 5 - cycle_len)
    decoy_len = random.randint(1, max_decoys)
    total_sprites = cycle_len + decoy_len

    body_colors = random.sample(usable, total_sprites)
    cycle_body = body_colors[:cycle_len]
    decoy_body = body_colors[cycle_len:]
    remaining = [color for color in usable if color not in body_colors]
    if len(remaining) < decoy_len + 1:
        return None

    decoy_markers = random.sample(remaining, decoy_len)
    remaining = [color for color in remaining if color not in decoy_markers]
    noise_count = random.randint(1, min(3, len(remaining)))
    noise_colors = random.sample(remaining, noise_count)

    cycle_sprites = [
        _sprite_spec(
            body_color=cycle_body[idx],
            marker_color=cycle_body[(idx + 1) % cycle_len],
            v_sign=random.choice([-1, 1]),
            h_sign=random.choice([-1, 1]),
            v_len=random.randint(2, 3),
            h_len=random.randint(2, 3),
        )
        for idx in range(cycle_len)
    ]
    decoy_sprites = [
        _sprite_spec(
            body_color=body_color,
            marker_color=marker_color,
            v_sign=random.choice([-1, 1]),
            h_sign=random.choice([-1, 1]),
            v_len=random.randint(2, 3),
            h_len=random.randint(2, 3),
        )
        for body_color, marker_color in zip(decoy_body, decoy_markers)
    ]

    row_slots = list(range(3, h - 3, 8))
    col_slots = list(range(3, w - 3, 8))
    anchors = [(r, c) for r in row_slots for c in col_slots]
    if len(anchors) < total_sprites:
        return None
    random.shuffle(anchors)

    cycle_positions = anchors[:cycle_len]
    decoy_positions = anchors[cycle_len:total_sprites]

    occ_in: set[tuple[int, int]] = set()
    occ_out: set[tuple[int, int]] = set()
    for sprite, bend in zip(cycle_sprites, cycle_positions):
        occ_in |= _cells_only(_abs_pixels(sprite, bend))
    for idx, sprite in enumerate(cycle_sprites):
        occ_out |= _cells_only(_abs_pixels(sprite, cycle_positions[(idx + 1) % cycle_len]))
    for sprite, bend in zip(decoy_sprites, decoy_positions):
        coords = _cells_only(_abs_pixels(sprite, bend))
        occ_in |= coords
        occ_out |= coords

    noise_result = _place_noise_pixels(noise_colors, (h, w), occ_in, occ_out)
    if noise_result is None:
        return None
    noise_pixels, _occ_in, _occ_out = noise_result

    grid = np.full((h, w), int(bg), dtype=int)
    for sprite, bend in zip(cycle_sprites, cycle_positions):
        _paint_pixels(grid, _abs_pixels(sprite, bend))
    for sprite, bend in zip(decoy_sprites, decoy_positions):
        _paint_pixels(grid, _abs_pixels(sprite, bend))
    for r, c, color in noise_pixels:
        grid[r, c] = int(color)

    expected = _expected_output(grid, bg, cycle_sprites, cycle_positions)
    produced = main(grid)
    if np.array_equal(produced, grid):
        return None
    if not np.array_equal(produced, expected):
        return None
    wrong = _wrong_sequential_output(grid, bg, cycle_sprites, cycle_positions)
    if wrong is not None and np.array_equal(wrong, expected):
        return None

    parsed_bg, sprites, edges, cycles, others = _extract_scene(grid)
    if int(parsed_bg) != int(bg):
        return None
    if len(sprites) != total_sprites:
        return None
    if len(cycles) != 1 or len(cycles[0]) != cycle_len:
        return None
    cycle_nodes = set(cycles[0])
    if set(edges) != cycle_nodes:
        return None
    if sum(1 for comp in others if len(comp) == 1) != noise_count:
        return None
    if len(others) != noise_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid old-site rotation puzzle")
