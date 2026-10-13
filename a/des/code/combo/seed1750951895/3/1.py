from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 7  # orange
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"coords": coords, "area": len(coords)})
    return comps


def _analyze_sprite(grid: np.ndarray, comp: dict, bg: int) -> dict:
    coords = [(int(r), int(c)) for r, c in comp["coords"]]
    assert 3 <= len(coords) <= 5

    colors = Counter(int(grid[r, c]) for r, c in coords)
    assert len(colors) == 2
    marker_colors = [color for color, count in colors.items() if count == 1]
    assert len(marker_colors) == 1
    marker = int(marker_colors[0])
    body = next(color for color, count in colors.items() if count > 1)

    occupied = set(coords)
    bends = []
    for r, c in coords:
        if not any((r + dr, c + dc) in occupied for dr, dc in [(-1, 0), (1, 0)]):
            continue
        if not any((r + dr, c + dc) in occupied for dr, dc in [(0, -1), (0, 1)]):
            continue
        if all(rr == r or cc == c for rr, cc in occupied):
            bends.append((r, c))
    assert len(bends) == 1
    bend = bends[0]

    return {
        "coords": coords,
        "body": int(body),
        "marker": marker,
        "bend": bend,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move each marked L-sprite to the same-colored control pixel and rotate it 180 degrees about its bend."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    controls_by_color: dict[int, list[tuple[int, int]]] = {}
    sprites = []
    for comp in _find_components(g, bg):
        if int(comp["area"]) == 1:
            r, c = comp["coords"][0]
            controls_by_color.setdefault(int(g[r, c]), []).append((int(r), int(c)))
        else:
            sprites.append(_analyze_sprite(g, comp, bg))

    for sprite in sprites:
        matches = controls_by_color.get(sprite["marker"], [])
        if len(matches) != 1:
            continue

        target = matches[0]
        for r, c in sprite["coords"]:
            out[r, c] = bg
        out[target] = bg

        br, bc = sprite["bend"]
        tr, tc = target
        for r, c in sprite["coords"]:
            nr = tr - (r - br)
            nc = tc - (c - bc)
            out[nr, nc] = int(g[r, c])

    return out


def _shape_cells(vlen: int, hlen: int, vdir: int, hdir: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    for i in range(1, vlen):
        cells.add((i * vdir, 0))
    for i in range(1, hlen):
        cells.add((0, i * hdir))
    return sorted(cells)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _can_place_input(grid: np.ndarray, blocked: np.ndarray, cells: list[tuple[int, int]], top: int, left: int) -> bool:
    h, w = grid.shape
    for dr, dc in cells:
        r = top + dr
        c = left + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _can_place_control(grid: np.ndarray, blocked: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = pos
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w) or blocked[r, c]:
        return False
    for dr, dc in CARDINALS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
            return False
    return True


def _output_cells(sprite: dict, target: tuple[int, int]) -> list[tuple[int, int]]:
    tr, tc = target
    return [(tr - dr, tc - dc) for dr, dc in sprite["rel_cells"]]


def _sample_target(
    h: int,
    w: int,
    sprite: dict,
    output_blocked: np.ndarray,
    input_blocked: np.ndarray,
    grid: np.ndarray,
) -> tuple[int, int] | None:
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)
    for target in candidates:
        cells = _output_cells(sprite, target)
        ok = True
        for r, c in cells:
            if not (0 <= r < h and 0 <= c < w):
                ok = False
                break
            if output_blocked[r, c]:
                ok = False
                break
        if not ok or not _can_place_control(grid, input_blocked, target):
            continue
        return target
    return None


def _make_sprite_spec(body: int, marker: int) -> dict:
    total = random.choice([3, 4, 5])
    choices = []
    for vlen in range(2, 5):
        for hlen in range(2, 5):
            if vlen + hlen - 1 == total:
                choices.append((vlen, hlen))
    vlen, hlen = random.choice(choices)
    vdir = random.choice([-1, 1])
    hdir = random.choice([-1, 1])
    rel_cells = _shape_cells(vlen, hlen, vdir, hdir)
    marker_rel = random.choice(rel_cells)
    return {
        "body": body,
        "marker": marker,
        "rel_cells": rel_cells,
        "marker_rel": marker_rel,
    }


def _place_input_sprites(h: int, w: int, specs: list[dict]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for spec in specs:
        candidates = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(candidates)
        blocked = _dilate4(occupied)
        placed = False
        for br, bc in candidates:
            cells = [(br + dr, bc + dc) for dr, dc in spec["rel_cells"]]
            if not _can_place_input(grid, blocked, spec["rel_cells"], br, bc):
                continue
            for r, c in cells:
                color = spec["marker"] if (r - br, c - bc) == spec["marker_rel"] else spec["body"]
                grid[r, c] = color
                occupied[r, c] = True
            spec["bend"] = (br, bc)
            spec["cells"] = cells
            placed = True
            break
        if not placed:
            return None
    return grid


def _choose_match_pattern(count: int) -> list[bool]:
    mode = random.choice(["all_matched", "mixed"])
    if mode == "all_matched":
        return [True] * count
    matched_count = random.randint(1, count - 1)
    flags = [True] * matched_count + [False] * (count - matched_count)
    random.shuffle(flags)
    return flags


def _gen_attempt() -> np.ndarray | None:
    branch = random.choice(["all", "mixed"])
    sprite_count = random.randint(1, 4) if branch == "all" else random.randint(2, 5)
    h = random.randint(12, 22)
    w = random.randint(12, 22)

    palette = [c for c in range(10) if c != BG]
    if sprite_count * 2 > len(palette):
        return None
    random.shuffle(palette)
    marker_colors = palette[:sprite_count]
    body_colors = palette[sprite_count : 2 * sprite_count]

    specs = [_make_sprite_spec(body, marker) for body, marker in zip(body_colors, marker_colors)]
    grid = _place_input_sprites(h, w, specs)
    if grid is None:
        return None

    matched = _choose_match_pattern(sprite_count) if branch == "mixed" else [True] * sprite_count
    if branch == "mixed" and (all(matched) or not any(matched)):
        return None

    input_blocked = _dilate4(grid != BG)
    output_blocked = np.zeros((h, w), dtype=bool)
    for spec, is_matched in zip(specs, matched):
        if not is_matched:
            for r, c in spec["cells"]:
                output_blocked[r, c] = True

    matched_specs = [spec for spec, is_matched in zip(specs, matched) if is_matched]
    random.shuffle(matched_specs)
    for spec in matched_specs:
        target = _sample_target(h, w, spec, output_blocked, input_blocked, grid)
        if target is None:
            return None
        tr, tc = target
        grid[tr, tc] = spec["marker"]
        input_blocked = _dilate4(grid != BG)
        for r, c in _output_cells(spec, target):
            output_blocked[r, c] = True

    actual = main(grid)
    if actual.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
