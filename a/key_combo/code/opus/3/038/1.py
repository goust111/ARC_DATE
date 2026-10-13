from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if _TOOLBOX.is_dir() and str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, bbox_coords, find_components, paste, random_free_location


BG = Color.BLACK
PALETTE = list(range(1, 10))
GEN_MAX_ATTEMPTS = 500


def _sorted_component_cells(coords: np.ndarray) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords)


def _extract_domino_rules(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    dominoes: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    domino_mask = np.zeros(grid.shape, dtype=bool)
    components = find_components(grid, bg=BG, connectivity=4, monochromatic=False)

    for comp in components:
        if int(comp["area"]) != 2:
            continue
        coords = _sorted_component_cells(comp["coords"])
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 2:
            continue
        src_cell, dst_cell = coords
        src = int(grid[src_cell])
        dst = int(grid[dst_cell])
        top_r = min(src_cell[0], dst_cell[0])
        left_c = min(src_cell[1], dst_cell[1])
        dominoes.append((src_cell, dst_cell, top_r, left_c))

    dominoes.sort(key=lambda item: (item[2], item[3]))
    rules: list[tuple[int, int]] = []
    for src_cell, dst_cell, _top_r, _left_c in dominoes:
        rules.append((int(grid[src_cell]), int(grid[dst_cell])))
        domino_mask[src_cell] = True
        domino_mask[dst_cell] = True
    return rules, domino_mask


def _apply_chain(color: int, rules: list[tuple[int, int]]) -> tuple[int, int]:
    current = int(color)
    steps = 0
    for src, dst in rules:
        if current == int(src):
            current = int(dst)
            steps += 1
    return current, steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor large foreground objects by reading scattered two-color dominoes as an ordered substitution chain.

    Input structure:
    - The grid has black background plus several isolated 4-connected foreground
      components.
    - Some components are exactly two-cell, two-color dominoes; the remaining
      components are larger foreground objects or isolated distractor pixels.
    - Each domino's row-major first cell and second cell encode a source color
      and a target color, and the dominoes are ordered from top to bottom.

    Rule:
    - Find every 4-connected foreground component of area 2 whose two cells have
      different colors.
    - For each such domino, read the row-major earlier cell as `source` and the
      later cell as `target`, then sort all dominoes by their top row to obtain
      an ordered replacement chain.
    - For every non-domino foreground pixel, scan that chain in order and replace
      its color whenever it matches the current rule's source; finally erase all
      domino pixels to black.

    Key insight:
    - The dominoes are not data to keep; they are spatially scattered legend
      entries whose vertical order controls a non-commutative chain of color
      rewrites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rules, domino_mask = _extract_domino_rules(g)
    assert rules, "expected at least one two-color domino"

    out = g.copy()
    targets = np.argwhere((g != BG) & (~domino_mask))
    for r, c in targets:
        final_color, _steps = _apply_chain(int(g[r, c]), rules)
        out[int(r), int(c)] = final_color

    out[domino_mask] = BG
    return out


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _composition(total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[random.randrange(parts)] += 1
    return shares


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set(_neighbors4(mask.shape, *start))

    while int(mask.sum()) < int(area):
        frontier = {cell for cell in frontier if not mask[cell]}
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        mask[cell] = True
        frontier.update(_neighbors4(mask.shape, *cell))

    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    if min(cropped.shape) < 2:
        return None
    return cropped


def _colorize_mask(
    mask: np.ndarray,
    available_colors: list[int],
    *,
    required_colors: list[int] | None = None,
    force_multicolor: bool = False,
) -> np.ndarray:
    coords = _sorted_component_cells(np.argwhere(mask))
    required = []
    for color in required_colors or []:
        if color not in required:
            required.append(int(color))

    if force_multicolor:
        min_colors = max(2, len(required))
    else:
        min_colors = max(1, len(required))
    max_colors = min(4, len(available_colors), len(coords))
    if max_colors < min_colors:
        min_colors = max_colors
    n_colors = random.randint(min_colors, max_colors)

    chosen = required[:]
    extras = [color for color in available_colors if color not in chosen]
    random.shuffle(extras)
    chosen.extend(extras[: max(0, n_colors - len(chosen))])
    if len(chosen) < n_colors:
        for color in available_colors:
            if len(chosen) >= n_colors:
                break
            if color not in chosen:
                chosen.append(color)
    chosen = chosen[:n_colors]
    random.shuffle(chosen)

    counts = [1] * len(chosen)
    for idx, extra in enumerate(_composition(len(coords) - len(chosen), len(chosen))):
        counts[idx] += extra

    patch = np.zeros(mask.shape, dtype=int)
    start = 0
    for color, count in zip(chosen, counts):
        for r, c in coords[start : start + count]:
            patch[r, c] = int(color)
        start += count
    return patch


def _make_object_patch(
    object_colors: list[int],
    *,
    required_colors: list[int] | None = None,
    force_multicolor: bool = False,
) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        max_area = min(11, h * w)
        area = random.randint(5, max_area)
        mask = _grow_connected_mask(h, w, area)
        if mask is None:
            continue
        return _colorize_mask(
            mask,
            object_colors,
            required_colors=required_colors,
            force_multicolor=force_multicolor,
        )
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, *, border: int = 0, padding: int = 1) -> tuple[int, int]:
    r0, c0 = random_free_location(
        grid,
        patch,
        transparent=BG,
        border=border,
        padding=padding,
        padding_connectivity=8,
    )
    paste(grid, patch, r0, c0, transparent=BG)
    return int(r0), int(c0)


def _is_isolated_placement(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in cell_set:
                    continue
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG:
                    return False
    return True


def _domino_candidates(grid: np.ndarray, patch_shape: tuple[int, int], used_top_rows: set[int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = map(int, patch_shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        if r0 in used_top_rows:
            continue
        for c0 in range(w - pw + 1):
            cells = [
                (r0 + r, c0 + c)
                for r in range(ph)
                for c in range(pw)
            ]
            if _is_isolated_placement(grid, cells):
                candidates.append((r0, c0))
    return candidates


def _place_dominoes(grid: np.ndarray, rules: list[tuple[int, int]]) -> bool:
    used_top_rows: set[int] = set()
    for src, dst in rules:
        orientations = [(1, 2), (2, 1)]
        random.shuffle(orientations)
        placed = False
        for ph, pw in orientations:
            candidates = _domino_candidates(grid, (ph, pw), used_top_rows)
            if not candidates:
                continue
            r0, c0 = random.choice(candidates)
            patch = np.zeros((ph, pw), dtype=int)
            patch[0, 0] = int(src)
            patch[ph - 1, pw - 1] = int(dst)
            paste(grid, patch, r0, c0, transparent=BG)
            used_top_rows.add(int(r0))
            placed = True
            break
        if not placed:
            return False
    return True


def _build_rules(count: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        chain_len = 3 if count >= 4 and random.random() < 0.55 else 2
        palette_size = random.randint(chain_len + 2, min(7, len(PALETTE)))
        colors = random.sample(PALETTE, palette_size)
        chain = random.sample(colors, chain_len + 1)
        rules = [(chain[i], chain[i + 1]) for i in range(chain_len)]

        while len(rules) < count:
            src = random.choice(colors)
            dst_options = [color for color in colors if color != src and (src, color) not in rules]
            if not dst_options:
                break
            rules.append((src, random.choice(dst_options)))

        if len(rules) != count:
            continue
        if not any(_apply_chain(src, rules)[1] >= 2 for src, _dst in rules):
            continue
        if any(_apply_chain(src, rules)[0] == src for src, _dst in rules):
            continue
        return rules
    return None


def _object_color_pool(rules: list[tuple[int, int]]) -> list[int]:
    used = []
    for src, dst in rules:
        if src not in used:
            used.append(int(src))
        if dst not in used:
            used.append(int(dst))
    return used


def _count_non_domino_changes(grid: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    _rules, domino_mask = _extract_domino_rules(grid)
    changed = int(np.count_nonzero((grid != out) & (~domino_mask)))
    multi_step = 0
    rules, _mask = _extract_domino_rules(grid)
    for r, c in np.argwhere((grid != BG) & (~domino_mask)):
        _final, steps = _apply_chain(int(grid[r, c]), rules)
        if steps >= 2:
            multi_step += 1
    return changed, multi_step


def _generate_one() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    grid = np.zeros((h, w), dtype=int)

    domino_count = random.randint(3, 5)
    rules = _build_rules(domino_count)
    if rules is None:
        return None
    object_colors = _object_color_pool(rules)

    object_count = 1 if random.random() < 0.45 else 2
    chain_heads = [src for src, _dst in rules if _apply_chain(src, rules)[1] >= 2]
    if not chain_heads:
        return None
    chain_head = random.choice(chain_heads)
    chain_mid, _ = _apply_chain(chain_head, rules[:1])
    required = [chain_head]
    if chain_mid != chain_head:
        required.append(chain_mid)

    first_patch = _make_object_patch(
        object_colors,
        required_colors=required,
        force_multicolor=True,
    )
    if first_patch is None:
        return None
    _place_patch(grid, first_patch, border=1, padding=1)

    for _ in range(object_count - 1):
        patch = _make_object_patch(
            object_colors,
            required_colors=[],
            force_multicolor=random.random() < 0.65,
        )
        if patch is None:
            return None
        _place_patch(grid, patch, border=1, padding=1)

    distractor_count = random.randint(0, 3)
    unrelated_colors = [color for color in PALETTE if color not in object_colors]
    for _ in range(distractor_count):
        if unrelated_colors and random.random() < 0.45:
            color = random.choice(unrelated_colors)
        else:
            color = random.choice(object_colors)
        patch = np.array([[color]], dtype=int)
        try:
            _place_patch(grid, patch, border=0, padding=1)
        except ValueError:
            break

    if not _place_dominoes(grid, rules):
        return None

    components = find_components(grid, bg=BG, connectivity=4, monochromatic=False)
    dominoes = [comp for comp in components if int(comp["area"]) == 2 and len({int(grid[r, c]) for r, c in comp["coords"]}) == 2]
    large = [comp for comp in components if int(comp["area"]) >= 5]
    singles = [comp for comp in components if int(comp["area"]) == 1]
    if len(dominoes) != domino_count or len(large) != object_count:
        return None
    if len(singles) > distractor_count:
        return None

    out = main(grid)
    changed, multi_step = _count_non_domino_changes(grid, out)
    if changed == 0 or multi_step == 0:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 038/1")
