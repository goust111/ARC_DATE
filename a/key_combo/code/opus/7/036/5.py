from __future__ import annotations

from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
LEGEND_H = 2
MAIN_COLORS = np.array([1, 2, 3, 4], dtype=int)
ACCENT_COLORS = np.array([6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MAX_RETRIES = 400


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _rotate_offset(dr: int, dc: int, turns: int) -> tuple[int, int]:
    turns %= 4
    if turns == 0:
        return int(dr), int(dc)
    if turns == 1:
        return int(dc), int(-dr)
    if turns == 2:
        return int(-dr), int(-dc)
    return int(-dc), int(dr)


def _decode_legend_cell(cell: np.ndarray, color: int) -> int:
    mask = cell == int(color)
    if bool(mask.all()):
        return 2
    if np.array_equal(mask, np.array([[True, False], [True, False]])):
        return 0
    if np.array_equal(mask, np.array([[False, True], [False, True]])):
        return 1
    if int(mask.sum()) == 1 and bool(mask[0].any()):
        return 3
    raise AssertionError("invalid legend cell encoding")


def _parse_legend(legend: np.ndarray) -> dict[int, int]:
    assert legend.shape[0] == LEGEND_H, "legend height must be exactly two rows"
    best: dict[int, int] | None = None
    for phase in range(4):
        mapping: dict[int, int] = {}
        valid = True
        for c0 in range(phase, legend.shape[1] - 1, 4):
            cell = legend[:, c0 : c0 + 2]
            colors = [int(v) for v in np.unique(cell) if int(v) != GREY]
            if not colors:
                continue
            if len(colors) != 1:
                valid = False
                break
            color = colors[0]
            if color in mapping:
                valid = False
                break
            try:
                mapping[color] = _decode_legend_cell(cell, color)
            except AssertionError:
                valid = False
                break
        if valid and 2 <= len(mapping) <= 3:
            if best is None or len(mapping) > len(best):
                best = mapping
    assert best is not None, "failed to parse legend"
    return best


def _major_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values if int(v) not in (BLACK, GREY))
    assert counts, "sprite must contain non-gray artwork"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "major color must be unique"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top legend, rotate each anchored sprite by its color-mapped angle, and place it on the paired target anchor.

    Input structure:
    - The top 2 rows are a gray legend band whose colored 2-column cells encode 0/90/180/270-degree clockwise rotations.
    - Below the legend is a black background containing 2-3 multicolor 4-connected sprites; each sprite has one gray anchor pixel and one dominant non-gray color from the legend.
    - The main area also contains the same number of isolated gray target-anchor pixels on the background.

    Rule:
    - Remove the 2-row legend band in the output.
    - For each sprite, identify its dominant non-gray color, look up the corresponding rotation in the legend, and rotate the sprite clockwise around its gray anchor.
    - Sort source sprites by anchor column and sort isolated target anchors by column; pair them in that order.
    - Erase the original sprites and the target anchors, then draw only the rotated non-gray pixels so that each rotated sprite's anchor coincides with its paired target anchor.

    Key insight:
    - The legend encodes sprite rotation angle rather than motion direction, so the color lookup must happen before anchor-aligned placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > LEGEND_H, "expected a 2D grid with a 2-row legend"

    legend_map = _parse_legend(g[:LEGEND_H, :])
    base = g[LEGEND_H:, :]

    sources: list[dict] = []
    targets: list[tuple[int, int]] = []
    for coords in _find_components(base != BLACK):
        vals = base[coords[:, 0], coords[:, 1]]
        if coords.shape[0] == 1:
            assert int(vals[0]) == GREY, "single-cell components must be gray targets"
            targets.append((int(coords[0, 0]), int(coords[0, 1])))
            continue

        anchor_coords = coords[vals == GREY]
        assert anchor_coords.shape == (1, 2), "each sprite must contain exactly one gray anchor"
        ar, ac = map(int, anchor_coords[0])
        major = _major_color(vals)
        assert major in legend_map, "sprite major color must appear in the legend"

        art: list[tuple[int, int, int]] = []
        for r, c in coords:
            color = int(base[int(r), int(c)])
            if color == GREY:
                continue
            art.append((int(r) - ar, int(c) - ac, color))
        assert art, "sprite artwork cannot be empty"
        sources.append(
            {
                "anchor": (ar, ac),
                "major": major,
                "turns": int(legend_map[major]),
                "art": art,
            }
        )

    assert 2 <= len(sources) <= 3, "expected 2-3 source sprites"
    assert len(targets) == len(sources), "source and target counts must match"

    sources.sort(key=lambda item: (int(item["anchor"][1]), int(item["anchor"][0])))
    targets.sort(key=lambda rc: (int(rc[1]), int(rc[0])))

    out = np.zeros_like(base)
    for source, (tr, tc) in zip(sources, targets, strict=True):
        for dr, dc, color in source["art"]:
            rr_off, cc_off = _rotate_offset(int(dr), int(dc), int(source["turns"]))
            rr = int(tr + rr_off)
            cc = int(tc + cc_off)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "placed sprite must stay in bounds"
            assert int(out[rr, cc]) == BLACK, "placed sprites must not overlap"
            out[rr, cc] = int(color)

    return out


def _art_key(art: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((int(dr), int(dc), int(color)) for dr, dc, color in art))


def _rotate_art(art: list[tuple[int, int, int]], turns: int) -> list[tuple[int, int, int]]:
    rotated = []
    for dr, dc, color in art:
        rr, cc = _rotate_offset(int(dr), int(dc), int(turns))
        rotated.append((rr, cc, int(color)))
    return rotated


def _shape_offsets(full_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in full_cells)


def _grow_connected_shape(rng: np.random.Generator, total: int) -> set[tuple[int, int]] | None:
    cells = {(0, 0)}
    while len(cells) < int(total):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if abs(nr) > 2 or abs(nc) > 2 or (nr, nc) in cells:
                    continue
                frontier.add((nr, nc))
        if not frontier:
            return None
        chosen = list(frontier)[int(rng.integers(len(frontier)))]
        cells.add(chosen)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if max(rows) == min(rows) or max(cols) == min(cols):
        return None
    return cells


def _generate_sprite(
    rng: np.random.Generator,
    main_color: int,
    turns: int,
) -> dict | None:
    for _ in range(80):
        total = int(rng.integers(4, 9))
        full_cells = _grow_connected_shape(rng, total)
        if full_cells is None:
            continue

        non_anchor = [(r, c) for r, c in full_cells if (r, c) != (0, 0)]
        if len(non_anchor) < 3:
            continue

        minority_count = 1 if len(non_anchor) <= 4 else int(rng.integers(1, 3))
        minority_count = min(minority_count, len(non_anchor) - 1)
        chosen_idx = rng.choice(len(non_anchor), size=minority_count, replace=False)
        minority_cells = {non_anchor[int(i)] for i in np.atleast_1d(chosen_idx)}

        accent_count = min(minority_count, int(rng.integers(1, min(2, minority_count) + 1)))
        accent_colors = list(map(int, rng.choice(ACCENT_COLORS, size=accent_count, replace=False)))

        art: list[tuple[int, int, int]] = []
        accent_i = 0
        for r, c in sorted(non_anchor):
            color = int(main_color)
            if (r, c) in minority_cells:
                color = int(accent_colors[accent_i % len(accent_colors)])
                accent_i += 1
            art.append((int(r), int(c), color))

        if turns != 0 and _art_key(_rotate_art(art, turns)) == _art_key(art):
            continue

        return {
            "main_color": int(main_color),
            "turns": int(turns),
            "art": art,
            "full_offsets": _shape_offsets(full_cells),
        }
    return None


def _anchor_range(offsets: list[tuple[int, int]], h: int, w: int) -> tuple[range, range]:
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)
    return range(-min_r, h - max_r), range(-min_c, w - max_c)


def _cells_at(anchor: tuple[int, int], offsets: list[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + dr), int(ac + dc)) for dr, dc in offsets}


def _art_cells_at(anchor: tuple[int, int], art: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + dr), int(ac + dc)) for dr, dc, _ in art}


def _halo(cells: set[tuple[int, int]], h: int, w: int, dirs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        out.add((int(r), int(c)))
        for dr, dc in dirs:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _place_sources(
    rng: np.random.Generator,
    sprites: list[dict],
    h: int,
    w: int,
) -> list[dict] | None:
    placed: list[dict | None] = [None] * len(sprites)
    occupied_halo: set[tuple[int, int]] = set()
    anchor_cols: set[int] = set()
    order = list(map(int, rng.permutation(len(sprites))))

    for idx in order:
        sprite = sprites[idx]
        candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
        ranges = _anchor_range(sprite["full_offsets"], h, w)
        for ar in ranges[0]:
            for ac in ranges[1]:
                if int(ac) in anchor_cols:
                    continue
                cells = _cells_at((int(ar), int(ac)), sprite["full_offsets"])
                if cells & occupied_halo:
                    continue
                candidates.append((int(ar), int(ac), cells))
        if not candidates:
            return None
        ar, ac, cells = candidates[int(rng.integers(len(candidates)))]
        occupied_halo |= _halo(cells, h, w, DIR8)
        anchor_cols.add(int(ac))
        placed[idx] = {
            **sprite,
            "anchor": (int(ar), int(ac)),
            "source_cells": cells,
        }

    return [item for item in placed if item is not None]


def _place_targets(
    rng: np.random.Generator,
    ordered_sources: list[dict],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    source_cells = set().union(*(source["source_cells"] for source in ordered_sources))
    forbidden = _halo(source_cells, h, w, DIR4)
    occupied_output: set[tuple[int, int]] = set()
    targets: list[tuple[int, int]] = []
    prev_col = -1

    for source in ordered_sources:
        rotated_art = _rotate_art(source["art"], int(source["turns"]))
        art_offsets = [(0, 0), *[(int(dr), int(dc)) for dr, dc, _ in rotated_art]]
        row_range, col_range = _anchor_range(art_offsets, h, w)
        candidates: list[tuple[int, int]] = []
        for tr in row_range:
            for tc in col_range:
                if int(tc) <= prev_col:
                    continue
                if (int(tr), int(tc)) in forbidden or (int(tr), int(tc)) in occupied_output:
                    continue
                art_cells = _art_cells_at((int(tr), int(tc)), rotated_art)
                if art_cells & occupied_output:
                    continue
                if any(target in art_cells for target in targets):
                    continue
                candidates.append((int(tr), int(tc)))
        if not candidates:
            return None
        target = candidates[int(rng.integers(len(candidates)))]
        targets.append(target)
        prev_col = int(target[1])
        forbidden |= _halo({target}, h, w, DIR4)
        occupied_output |= _art_cells_at(target, rotated_art)

    return targets


def _legend_cell(turns: int, color: int, rng: np.random.Generator) -> np.ndarray:
    cell = np.full((2, 2), GREY, dtype=int)
    if turns == 0:
        cell[:, 0] = int(color)
    elif turns == 1:
        cell[:, 1] = int(color)
    elif turns == 2:
        cell[:, :] = int(color)
    else:
        cell[0, int(rng.integers(0, 2))] = int(color)
    return cell


def _build_input(
    h: int,
    w: int,
    phase: int,
    legend_entries: list[tuple[int, int]],
    sources: list[dict],
    targets: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.zeros((h + LEGEND_H, w), dtype=int)
    grid[:LEGEND_H, :] = GREY

    for idx, (color, turns) in enumerate(legend_entries):
        c0 = int(phase + 4 * idx)
        grid[:LEGEND_H, c0 : c0 + 2] = _legend_cell(int(turns), int(color), rng)

    base = grid[LEGEND_H:, :]
    for source in sources:
        ar, ac = source["anchor"]
        base[int(ar), int(ac)] = GREY
        for dr, dc, color in source["art"]:
            base[int(ar + dr), int(ac + dc)] = int(color)
    for tr, tc in targets:
        base[int(tr), int(tc)] = GREY

    return grid


def _build_output(h: int, w: int, ordered_sources: list[dict], targets: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for source, target in zip(ordered_sources, targets, strict=True):
        tr, tc = target
        for dr, dc, color in _rotate_art(source["art"], int(source["turns"])):
            rr = int(tr + dr)
            cc = int(tc + dc)
            out[rr, cc] = int(color)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(2, 4))
    colors = list(map(int, rng.choice(MAIN_COLORS, size=count, replace=False)))
    turns = list(map(int, rng.choice(np.array([0, 1, 2, 3], dtype=int), size=count, replace=False)))
    phase = int(rng.integers(0, 4))

    min_w = max(14, int(phase + 4 * (count - 1) + 2))
    w = int(rng.integers(min_w, 21))
    h = int(rng.integers(11, 17))

    sprites: list[dict] = []
    for color, turn in zip(colors, turns, strict=True):
        sprite = _generate_sprite(rng, int(color), int(turn))
        if sprite is None:
            return None
        sprites.append(sprite)

    sources = _place_sources(rng, sprites, h, w)
    if sources is None:
        return None
    ordered_sources = sorted(sources, key=lambda item: (int(item["anchor"][1]), int(item["anchor"][0])))

    targets = _place_targets(rng, ordered_sources, h, w)
    if targets is None:
        return None

    legend_entries = [(int(color), int(turn)) for color, turn in zip(colors, turns, strict=True)]
    rng.shuffle(legend_entries)

    inp = _build_input(h, w, phase, legend_entries, sources, targets, rng)
    out = _build_output(h, w, ordered_sources, targets)

    if inp.shape[0] > 30 or inp.shape[1] > 30 or out.shape[0] > 30 or out.shape[1] > 30:
        return None
    if int(np.count_nonzero(inp[LEGEND_H:, :] == GREY)) != 2 * count:
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, out):
        return None
    if np.array_equal(inp[LEGEND_H:, :], out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
