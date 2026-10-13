from __future__ import annotations

from collections import Counter
import random

import numpy as np


COLORS = list(range(10))
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _terminal(color: int, mapping: dict[int, int]) -> int:
    cur = int(color)
    seen: set[int] = set()
    while cur in mapping:
        assert cur not in seen, "color-pair legend must be acyclic"
        seen.add(cur)
        cur = int(mapping[cur])
    return cur


def _path_edges(color: int, mapping: dict[int, int]) -> list[tuple[int, int]]:
    cur = int(color)
    seen: set[int] = set()
    edges: list[tuple[int, int]] = []
    while cur in mapping:
        assert cur not in seen, "color-pair legend must be acyclic"
        seen.add(cur)
        nxt = int(mapping[cur])
        edges.append((cur, nxt))
        cur = nxt
    return edges


def _extract_scene(grid: np.ndarray):
    bg = _background(grid)
    components = _components(grid != bg)
    assert components, "expected foreground objects and color-pair legends"

    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    objects: list[dict] = []

    for coords in components:
        values = [int(grid[int(r), int(c)]) for r, c in coords]
        cells = [(int(r), int(c)) for r, c in coords]
        is_vertical_pair = False
        if len(cells) == 2:
            (r0, c0), (r1, c1) = sorted(cells)
            is_vertical_pair = c0 == c1 and r1 == r0 + 1 and int(grid[r0, c0]) != int(grid[r1, c1])

        if is_vertical_pair:
            (r0, c0), (r1, c1) = sorted(cells)
            src = int(grid[r0, c0])
            dst = int(grid[r1, c1])
            assert src != bg and dst != bg, "legend colors must be foreground colors"
            assert src not in mapping, "a color can be the source of at most one legend pair"
            mapping[src] = dst
            legend_cells.add((r0, c0))
            legend_cells.add((r1, c1))
            continue

        unique = set(values)
        assert len(unique) == 1, "non-legend components must be single-color objects"
        assert len(cells) >= 2, "objects must contain more than one cell"
        objects.append({"color": int(values[0]), "coords": coords})

    assert mapping, "expected at least one vertical color-pair legend"
    assert objects, "expected at least one recolorable object"
    assert len(set(mapping.values())) == len(mapping.values()), "legend pairs must form disjoint chains, not merges"

    legend_colors = set(mapping) | set(mapping.values())
    for obj in objects:
        assert int(obj["color"]) in legend_colors, "object colors must belong to a legend chain"
    for src in mapping:
        _terminal(src, mapping)

    return bg, mapping, objects, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object by following the vertical color-pair legend to the terminal color.

    Input structure:
    - The background is the dominant color.
    - Foreground includes isolated vertical 1x2 legend pairs; the upper cell maps to the lower cell.
    - All other foreground components are separated multi-cell single-color objects.
    - The legend edges form one or more acyclic color chains with no repeated source color.

    Rule:
    - Read every vertical two-color pair as a directed color mapping from upper color to lower color.
    - For each non-legend object, repeatedly replace its color through the mapping until the color is no longer a source.
    - Recolor the whole object to that terminal color while leaving all legend pixels unchanged.

    Key insight:
    - The legend is not a one-step lookup: intermediate target colors can also be sources, so every object color must be traced to its chain endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, mapping, objects, _legend_cells = _extract_scene(g)
    out = g.copy()
    for obj in objects:
        dst = _terminal(int(obj["color"]), mapping)
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = dst
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (6, 9)
GRID_SIDE_RANGE = (18, 30)
OBJECT_AREA_RANGE = (3, 10)
OBJECT_MAX_H = 4
OBJECT_MAX_W = 5
PLACE_PADDING = 1


def _sample_chains(bg: int) -> list[list[int]]:
    colors = [c for c in COLORS if c != int(bg)]
    random.shuffle(colors)
    first_len = random.randint(3, 5)
    max_second_len = min(4, len(colors) - first_len)
    second_len = random.randint(2, max_second_len)
    return [colors[:first_len], colors[first_len:first_len + second_len]]


def _chain_mapping(chains: list[list[int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for chain in chains:
        for src, dst in zip(chain, chain[1:]):
            mapping[int(src)] = int(dst)
    return mapping


def _random_polyomino() -> np.ndarray:
    target = random.randint(*OBJECT_AREA_RANGE)
    cells: set[tuple[int, int]] = {(0, 0)}
    for _ in range(300):
        if len(cells) >= target:
            break
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(NEIGHBORS4)
        candidate = (r + dr, c + dc)
        trial = cells | {candidate}
        rs = [p[0] for p in trial]
        cs = [p[1] for p in trial]
        if max(rs) - min(rs) + 1 <= OBJECT_MAX_H and max(cs) - min(cs) + 1 <= OBJECT_MAX_W:
            cells = trial

    if len(cells) < 3:
        cells = {(0, 0), (0, 1), (1, 0)}

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    ph, pw = mask.shape
    if r0 < 1 or c0 < 1 or r0 + ph > h - 1 or c0 + pw > w - 1:
        return False
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if bool(occupied[rr, cc]):
            return False
        ar0 = max(0, rr - PLACE_PADDING)
        ar1 = min(h, rr + PLACE_PADDING + 1)
        ac0 = max(0, cc - PLACE_PADDING)
        ac1 = min(w, cc + PLACE_PADDING + 1)
        if bool(np.any(occupied[ar0:ar1, ac0:ac1])):
            return False
    return True


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    ph, pw = mask.shape
    positions = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
    random.shuffle(positions)

    for r0, c0 in positions:
        if not _can_place(occupied, mask, r0, c0):
            continue
        placed: list[tuple[int, int]] = []
        for dr, dc in np.argwhere(mask):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            grid[rr, cc] = int(values[int(dr), int(dc)])
            occupied[rr, cc] = True
            placed.append((rr, cc))
        return placed
    return None


def _object_colors(chains: list[list[int]], n_objects: int) -> list[int]:
    required = [
        chains[0][0],
        chains[0][1],
        chains[0][-1],
        chains[1][0],
        chains[1][-1],
    ]
    if len(chains[1]) >= 3:
        required.append(chains[1][1])

    palette = [c for chain in chains for c in chain]
    while len(required) < n_objects:
        required.append(random.choice(palette))
    random.shuffle(required)
    return [int(c) for c in required[:n_objects]]


def _expected_output(grid: np.ndarray, objects: list[dict], mapping: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        dst = _terminal(int(obj["color"]), mapping)
        for r, c in obj["cells"]:
            out[int(r), int(c)] = dst
    return out


def _assert_semantic_sample(
    grid: np.ndarray,
    expected: np.ndarray,
    mapping: dict[int, int],
    objects: list[dict],
    legend_cells: set[tuple[int, int]],
) -> None:
    parsed_bg, parsed_mapping, parsed_objects, parsed_legend_cells = _extract_scene(grid)
    assert parsed_bg == _background(grid)
    assert parsed_mapping == mapping
    assert parsed_legend_cells == legend_cells
    assert len(parsed_objects) == len(objects)

    assert np.array_equal(main(grid), expected)
    assert not np.array_equal(grid, expected), "sample must contain at least one recolored object"

    object_colors = [int(obj["color"]) for obj in objects]
    path_lengths = [len(_path_edges(color, mapping)) for color in object_colors]
    assert any(length >= 2 for length in path_lengths), "sample must require following a multi-step chain"
    assert any(length == 0 for length in path_lengths), "sample must include an already-terminal object"
    assert len({_terminal(color, mapping) for color in object_colors}) >= 2, "sample must have more than one chain endpoint"

    for r, c in legend_cells:
        assert int(grid[r, c]) == int(expected[r, c]), "legend pixels must be preserved"

    used_edges = set()
    for color in object_colors:
        used_edges.update(_path_edges(color, mapping))
    assert set(mapping.items()) <= used_edges, "every legend edge must affect at least one object path"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    bg = random.choice(COLORS)
    chains = _sample_chains(bg)
    mapping = _chain_mapping(chains)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    obj_colors = _object_colors(chains, n_objects)

    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    legend_cells: set[tuple[int, int]] = set()

    patches: list[dict] = []
    for src, dst in mapping.items():
        values = np.array([[int(src)], [int(dst)]], dtype=int)
        patches.append({"kind": "legend", "values": values, "mask": np.ones((2, 1), dtype=bool), "src": src, "dst": dst})

    for color in obj_colors:
        mask = _random_polyomino()
        values = np.full(mask.shape, int(bg), dtype=int)
        values[mask] = int(color)
        patches.append({"kind": "object", "values": values, "mask": mask, "color": int(color)})

    random.shuffle(patches)
    for patch in patches:
        cells = _place_patch(grid, occupied, patch["values"], patch["mask"])
        if cells is None:
            return None
        if patch["kind"] == "legend":
            legend_cells.update(cells)
        else:
            objects.append({"color": int(patch["color"]), "cells": cells})

    expected = _expected_output(grid, objects, mapping)
    try:
        _assert_semantic_sample(grid, expected, mapping, objects, legend_cells)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-chain input")
