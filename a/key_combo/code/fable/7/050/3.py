import itertools
import random

import numpy as np


BG = 0
PALETTE = list(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        out[r - r0, c - c0] = int(grid[r, c])
    return out


def _is_domino(coords: list[tuple[int, int]]) -> bool:
    return len(coords) == 2 and abs(coords[0][0] - coords[1][0]) + abs(coords[0][1] - coords[1][1]) == 1


def _find_internal_dominoes(sprite: np.ndarray, bg: int) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    dominoes: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in np.unique(sprite):
        color = int(color)
        if color == bg:
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite == color)]
        if _is_domino(coords):
            dominoes[color] = tuple(sorted(coords))
    return dominoes


def _extract_scene(grid: np.ndarray):
    bg = _get_bg(grid)
    comps = _components(grid != bg)
    sprite_comps = [comp for comp in comps if len(comp) > 2]
    anchor_comps = [comp for comp in comps if len(comp) == 2]

    assert len(sprite_comps) == 1, "expected exactly one connected sprite body"
    assert len(anchor_comps) >= 1, "expected at least one external domino anchor"
    assert len(sprite_comps) + len(anchor_comps) == len(comps), "all foreground components must be the sprite or domino anchors"

    sprite_coords = sprite_comps[0]
    sprite = _crop_component(grid, sprite_coords, bg)
    sprite_dominoes = _find_internal_dominoes(sprite, bg)

    anchors = []
    for comp in anchor_comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "anchor components must be monochrome"
        color = next(iter(colors))
        assert _is_domino(comp), "each anchor must be a 2-cell domino"
        assert color in sprite_dominoes, "anchor color must match a unique domino inside the sprite"
        anchors.append({"color": color, "coords": tuple(sorted(comp))})

    return bg, sprite, sprite_dominoes, anchors


def _candidate_placements(
    grid: np.ndarray,
    sprite: np.ndarray,
    color: int,
    anchor_coords: tuple[tuple[int, int], tuple[int, int]],
    bg: int,
):
    h, w = grid.shape
    anchor_set = set(anchor_coords)
    candidates = []
    seen = set()

    for turns in range(4):
        rotated = np.rot90(sprite, turns)
        domino = [tuple(map(int, rc)) for rc in np.argwhere(rotated == color)]
        if not _is_domino(domino):
            continue
        occupied = [(int(r), int(c), int(rotated[r, c])) for r, c in np.argwhere(rotated != bg)]
        for perm in itertools.permutations(anchor_coords):
            dr = perm[0][0] - domino[0][0]
            dc = perm[0][1] - domino[0][1]
            shifted_domino = {(domino[0][0] + dr, domino[0][1] + dc), (domino[1][0] + dr, domino[1][1] + dc)}
            if shifted_domino != anchor_set:
                continue

            placed = []
            ok = True
            for sr, sc, value in occupied:
                rr = sr + dr
                cc = sc + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                existing = int(grid[rr, cc])
                if existing != bg and (rr, cc) not in anchor_set:
                    ok = False
                    break
                placed.append((rr, cc, value))
            if not ok:
                continue

            key = (turns, dr, dc)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "turns": turns,
                    "offset": (dr, dc),
                    "placed": tuple(sorted(placed)),
                }
            )
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the multicolor sprite onto each matching external domino anchor by rotating it until the paired internal domino lands exactly on the anchor.

    Input structure:
    - The grid has a black background, one connected multicolor sprite, and several isolated monochrome 2-cell domino anchors.
    - Some sprite colors appear exactly twice inside the sprite as adjacent cells, forming unique internal dominoes.
    - Each external anchor uses one of those domino colors exactly once, and different anchors use different colors.

    Rule:
    - For each external domino anchor, locate the sprite's internal domino of the same color.
    - Rotate the whole sprite by 0, 90, 180, or 270 degrees and translate it so that the internal domino exactly overlaps the external anchor.
    - Stamp every transformed sprite onto the grid, keeping the original sprite and covering each anchor with its matched copy.

    Key insight:
    - The anchor domino encodes both which internal feature to match and which rigid pose the whole sprite must take.
    - The correct copy is the unique rotation-and-translation that lets the paired domino coincide with the anchor without colliding with any other existing foreground object.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprite, _sprite_dominoes, anchors = _extract_scene(g)

    out = g.copy()
    for anchor in anchors:
        candidates = _candidate_placements(g, sprite, anchor["color"], anchor["coords"], bg)
        assert len(candidates) == 1, "each anchor must determine a unique rigid placement"
        for r, c, value in candidates[0]["placed"]:
            out[r, c] = value
    return out


def _grow_connected_mask(height: int, width: int, area: int, rng: random.Random) -> set[tuple[int, int]]:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < area:
        if not frontier:
            return set()
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _adjacent_pairs(cells: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    for r, c in cells:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) in cells:
                pairs.append(((r, c), (nr, nc)))
    return pairs


def _choose_disjoint_pairs(
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    needed: int,
    rng: random.Random,
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    for _ in range(80):
        order = pairs[:]
        rng.shuffle(order)
        chosen = []
        used = set()
        for a, b in order:
            if a in used or b in used:
                continue
            chosen.append((a, b))
            used.add(a)
            used.add(b)
            if len(chosen) == needed:
                return chosen
    return None


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(list(cells))
    return {(r - r0, c - c0) for r, c in cells}


def _make_sprite(rng: random.Random, anchor_count: int) -> tuple[np.ndarray, list[int]]:
    for _ in range(300):
        height = rng.randint(4, 6)
        width = rng.randint(4, 6)
        min_area = 2 * anchor_count + 4
        max_area = min(height * width, min_area + 4)
        area = rng.randint(min_area, max_area)

        cells = _grow_connected_mask(height, width, area, rng)
        if len(cells) != area:
            continue
        normalized = _normalize_cells(cells)
        rows = {r for r, _ in normalized}
        cols = {c for _, c in normalized}
        if len(rows) < 2 or len(cols) < 2:
            continue
        pairs = _adjacent_pairs(normalized)
        chosen_pairs = _choose_disjoint_pairs(pairs, anchor_count, rng)
        if chosen_pairs is None:
            continue

        used_cells = {cell for pair in chosen_pairs for cell in pair}
        leftovers = [cell for cell in normalized if cell not in used_cells]
        if len(leftovers) < 3:
            continue

        colors = rng.sample(PALETTE, anchor_count + 2)
        anchor_colors = colors[:anchor_count]
        body_color = colors[anchor_count]
        accent_color = colors[anchor_count + 1]

        r0, c0, r1, c1 = _bbox(list(normalized))
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        for r, c in normalized:
            sprite[r, c] = body_color

        accent_cell = rng.choice(leftovers)
        sprite[accent_cell] = accent_color

        for color, pair in zip(anchor_colors, chosen_pairs):
            for r, c in pair:
                sprite[r, c] = color

        if len(_find_internal_dominoes(sprite, BG)) != anchor_count:
            continue
        if any(np.array_equal(sprite, np.rot90(sprite, turns)) for turns in (1, 2, 3)):
            continue
        return sprite, anchor_colors
    raise RuntimeError("failed to construct a valid sprite")


def _rects_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int,
) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah - 1 + margin < br
        or br + bh - 1 + margin < ar
        or ac + aw - 1 + margin < bc
        or bc + bw - 1 + margin < ac
    )


def _sample_position(
    grid_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    placed_rects: list[tuple[int, int, int, int]],
    rng: random.Random,
    margin: int,
    prefer_edge: bool,
) -> tuple[int, int] | None:
    h, w = grid_shape
    sh, sw = sprite_shape
    if sh > h or sw > w:
        return None

    row_choices = list(range(0, h - sh + 1))
    col_choices = list(range(0, w - sw + 1))
    if prefer_edge:
        edge_rows = [0, h - sh] if h - sh > 0 else [0]
        edge_cols = [0, w - sw] if w - sw > 0 else [0]
        candidates = [(r, c) for r in edge_rows for c in col_choices] + [(r, c) for r in row_choices for c in edge_cols]
        rng.shuffle(candidates)
        candidates += [(r, c) for r in row_choices for c in col_choices]
    else:
        candidates = [(r, c) for r in row_choices for c in col_choices]
        rng.shuffle(candidates)

    seen = set()
    for r, c in candidates:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        rect = (r, c, sh, sw)
        if any(_rects_too_close(rect, other, margin) for other in placed_rects):
            continue
        return r, c
    return None


def _stamp(sprite: np.ndarray, top: int, left: int) -> set[tuple[int, int, int]]:
    cells = set()
    for r, c in np.argwhere(sprite != BG):
        cells.add((top + int(r), left + int(c), int(sprite[r, c])))
    return cells


def _anchor_key(anchor_coords: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted(anchor_coords))


def _layout_attempt(rng: random.Random) -> np.ndarray | None:
    anchor_count = 3 if rng.random() < 0.55 else 2
    sprite, anchor_colors = _make_sprite(rng, anchor_count)
    turns_list = [rng.randrange(4) for _ in range(anchor_count)]
    if all(turns == 0 for turns in turns_list):
        turns_list[rng.randrange(anchor_count)] = rng.choice([1, 2, 3])

    rotated_shapes = [np.rot90(sprite, turns).shape for turns in turns_list]
    max_h = max(sprite.shape[0], *(shape[0] for shape in rotated_shapes))
    max_w = max(sprite.shape[1], *(shape[1] for shape in rotated_shapes))
    height = rng.randint(max(18, max_h + 8), 28)
    width = rng.randint(max(18, max_w + 8), 28)

    placed_rects: list[tuple[int, int, int, int]] = []
    orig_pos = _sample_position((height, width), sprite.shape, placed_rects, rng, margin=2, prefer_edge=False)
    if orig_pos is None:
        return None
    placed_rects.append((orig_pos[0], orig_pos[1], sprite.shape[0], sprite.shape[1]))

    copy_positions = []
    for idx, shape in enumerate(rotated_shapes):
        prefer_edge = idx == 0 or turns_list[idx] in (0, 2)
        pos = _sample_position((height, width), shape, placed_rects, rng, margin=2, prefer_edge=prefer_edge)
        if pos is None:
            return None
        placed_rects.append((pos[0], pos[1], shape[0], shape[1]))
        copy_positions.append(pos)

    grid = np.full((height, width), BG, dtype=int)
    orig_r, orig_c = orig_pos
    for r, c in np.argwhere(sprite != BG):
        grid[orig_r + int(r), orig_c + int(c)] = int(sprite[r, c])

    expected_by_anchor = {}
    for color, turns, (top, left) in zip(anchor_colors, turns_list, copy_positions):
        rotated = np.rot90(sprite, turns)
        domino = [tuple(map(int, rc)) for rc in np.argwhere(rotated == color)]
        if not _is_domino(domino):
            return None
        anchor_coords = tuple(sorted(((top + domino[0][0], left + domino[0][1]), (top + domino[1][0], left + domino[1][1]))))
        for r, c in anchor_coords:
            if grid[r, c] != BG:
                return None
            grid[r, c] = color
        expected_by_anchor[anchor_coords] = _stamp(rotated, top, left)

    comps = _components(grid != BG)
    if len(comps) != anchor_count + 1:
        return None
    if sum(len(comp) > 2 for comp in comps) != 1:
        return None
    if sum(len(comp) == 2 for comp in comps) != anchor_count:
        return None

    bg, parsed_sprite, _dominoes, anchors = _extract_scene(grid)
    if parsed_sprite.shape != sprite.shape or not np.array_equal(parsed_sprite, sprite):
        return None

    for anchor in anchors:
        anchor_key = _anchor_key(anchor["coords"])
        expected = expected_by_anchor.get(anchor_key)
        if expected is None:
            return None
        candidates = _candidate_placements(grid, sprite, anchor["color"], anchor["coords"], bg)
        if len(candidates) != 1:
            return None
        if set(candidates[0]["placed"]) != expected:
            return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _layout_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-anchor sprite grid")
