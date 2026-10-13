from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

import numpy as np


BACKGROUND = 0
SOURCE_COLORS = (1, 2, 3)
MULTI_HIT_COLOR = 4
SPRITE_COLOR = 5

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (18, 24)
MAX_ATTEMPTS = 400
PLACEMENT_RESTARTS = 120
SEARCH_BRANCH_LIMIT = 60


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[int, int]] = set()
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        cells: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _sprite_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for coords in _component_coords(grid == SPRITE_COLOR):
        box = _bbox(coords)
        comps.append({"coords": coords, "bbox": box, "size": int(len(coords))})
    return comps


def _source_info(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    sources: list[dict] = []
    for c in range(w):
        color = int(grid[h - 1, c])
        if color != bg and color != SPRITE_COLOR:
            sources.append({"row": h - 1, "col": c, "color": color, "diag": h - 1 + c})
    return sources


def _analyze_hits(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sources = _source_info(g, bg)
    source_diags = {src["diag"]: idx for idx, src in enumerate(sources)}
    sprites = _sprite_components(g)
    for sprite in sprites:
        hit_indices: set[int] = set()
        for r, c in sprite["coords"]:
            idx = source_diags.get(int(r + c))
            if idx is not None:
                hit_indices.add(idx)
        sprite["hits"] = tuple(sorted(hit_indices))
    return sources, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray sprite by how many upward-right diagonal source rays pass through it.

    Input structure:
    - A mostly black grid contains 2 or 3 single-pixel colored sources on the bottom row.
    - The interior contains several disconnected gray sprites, each 4-connected, size 4-8, and bounded by at most 3x3.
    - Sources use visible colors while sprites are uniformly gray.

    Rule:
    - Treat every bottom-row source pixel as launching a 45-degree ray upward-right, one row up and one column right per step, continuing to the grid edge without being blocked.
    - For each gray sprite, count how many distinct source rays pass through at least one of its pixels.
    - Leave sprites with count 0 gray, recolor sprites with count 1 to that unique source color, and recolor sprites with count 2 or more yellow.
    - Keep the source pixels themselves unchanged, and do not draw the rays.

    Key insight:
    - The diagonal rays are invisible probes: only their sprite-crossing counts matter, and multiple hits collapse to one shared yellow outcome.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sources, sprites = _analyze_hits(g)

    assert 2 <= len(sources) <= 3, "expected 2 or 3 distinct bottom-row sources"
    source_colors = [src["color"] for src in sources]
    assert len(set(source_colors)) == len(source_colors), "source colors must be distinct"

    for sprite in sprites:
        hits = sprite["hits"]
        coords = sprite["coords"]
        if len(hits) == 0:
            continue
        if len(hits) == 1:
            color = int(sources[hits[0]]["color"])
        else:
            color = MULTI_HIT_COLOR
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask)


def _enumerate_masks() -> list[dict]:
    found: dict[tuple[tuple[int, ...], ...], dict] = {}
    for cell_count in range(4, 9):
        for cells in combinations(range(9), cell_count):
            mask = np.zeros((3, 3), dtype=bool)
            for k in cells:
                mask[k // 3, k % 3] = True
            if not _is_connected(mask):
                continue
            cropped = _crop_mask(mask)
            if cropped.shape[0] > 3 or cropped.shape[1] > 3:
                continue
            key = _mask_key(cropped)
            if key in found:
                continue
            coords = np.argwhere(cropped)
            found[key] = {
                "mask": cropped,
                "coords": coords,
                "shape": cropped.shape,
                "diag_offsets": tuple(sorted({int(r + c) for r, c in coords})),
                "size": int(coords.shape[0]),
            }
    masks = list(found.values())
    masks.sort(key=lambda item: (item["shape"][0] * item["shape"][1], item["size"], item["shape"]))
    return masks


MASK_LIBRARY = _enumerate_masks()


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _sample_source_layout(h: int, w: int) -> list[dict] | None:
    n_sources = random.choice((2, 3))
    if n_sources == 2:
        gaps = [random.randint(2, 4)]
    else:
        gaps = [random.randint(2, 3), random.randint(2, 3)]
    total_span = sum(gaps)
    max_start = min(6, w - 8 - total_span)
    if max_start < 0:
        return None
    start = random.randint(0, max_start)
    cols = [start]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    colors = random.sample(list(SOURCE_COLORS), n_sources)
    return [
        {"row": h - 1, "col": col, "color": color, "diag": h - 1 + col}
        for col, color in zip(cols, colors)
    ]


def _candidate_placements(h: int, w: int, sources: list[dict]) -> dict[frozenset[int], list[dict]]:
    source_diags = [src["diag"] for src in sources]
    candidates: dict[frozenset[int], list[dict]] = {}
    max_bottom = h - 3
    for mask_data in MASK_LIBRARY:
        mh, mw = mask_data["shape"]
        max_r0 = max_bottom - mh
        if max_r0 < 0:
            continue
        for r0 in range(max_r0 + 1):
            for c0 in range(w - mw + 1):
                global_diags = {r0 + c0 + d for d in mask_data["diag_offsets"]}
                hit_set = frozenset(
                    idx for idx, diag in enumerate(source_diags) if diag in global_diags
                )
                coords = mask_data["coords"] + np.array([r0, c0], dtype=int)
                bbox = (r0, c0, r0 + mh - 1, c0 + mw - 1)
                min_delta = min(abs(diag - gd) for gd in global_diags for diag in source_diags)
                entry = {
                    "bbox": bbox,
                    "coords": coords,
                    "mask": mask_data["mask"],
                    "size": mask_data["size"],
                    "hits": hit_set,
                    "min_delta": int(min_delta),
                }
                candidates.setdefault(hit_set, []).append(entry)
    return candidates


def _required_hit_sets(n_sources: int) -> list[frozenset[int]]:
    required = [frozenset()]
    required.extend(frozenset({idx}) for idx in range(n_sources))
    multi_options = [frozenset(combo) for r in range(2, n_sources + 1) for combo in combinations(range(n_sources), r)]
    required.append(random.choice(multi_options))
    extra_count = random.choice((0, 1))
    for _ in range(extra_count):
        pool = [frozenset({idx}) for idx in range(n_sources)] + multi_options + [frozenset()]
        required.append(random.choice(pool))
    return required


def _candidate_order(hit_set: frozenset[int], options: list[dict]) -> list[dict]:
    items = options[:]
    random.shuffle(items)
    if len(hit_set) == 0:
        near = [item for item in items if item["min_delta"] == 1]
        if near:
            items = near + [item for item in items if item["min_delta"] != 1]
    elif len(hit_set) >= 2:
        items.sort(key=lambda item: (-item["size"], item["min_delta"]))
    return items[: min(len(items), SEARCH_BRANCH_LIMIT)]


def _pick_placements(candidates: dict[frozenset[int], list[dict]], hit_sets: list[frozenset[int]]) -> list[dict] | None:
    for hit_set in hit_sets:
        if not candidates.get(hit_set):
            return None

    reqs = sorted(hit_sets, key=lambda hs: (len(candidates.get(hs, [])), -len(hs)))
    chosen: list[dict] = []

    def search(i: int) -> bool:
        if i == len(reqs):
            return True
        hit_set = reqs[i]
        for cand in _candidate_order(hit_set, candidates[hit_set]):
            if all(_bbox_gap_ok(cand["bbox"], prev["bbox"]) for prev in chosen):
                chosen.append(cand)
                if search(i + 1):
                    return True
                chosen.pop()
        return False

    return chosen[:] if search(0) else None


def _render_grid(h: int, w: int, sources: list[dict], sprites: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for sprite in sprites:
        coords = sprite["coords"]
        grid[coords[:, 0], coords[:, 1]] = SPRITE_COLOR
    for src in sources:
        grid[src["row"], src["col"]] = src["color"]
    return grid


def _validate_generated(grid: np.ndarray, planned_hits: list[frozenset[int]]) -> bool:
    sources, sprites = _analyze_hits(grid)
    if not (2 <= len(sources) <= 3):
        return False
    if len(set(src["color"] for src in sources)) != len(sources):
        return False
    if len(sprites) != len(planned_hits):
        return False

    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        if sprite["size"] < 4 or sprite["size"] > 8:
            return False
        if r1 - r0 + 1 > 3 or c1 - c0 + 1 > 3:
            return False

    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            if not _bbox_gap_ok(sprites[i]["bbox"], sprites[j]["bbox"]):
                return False

    hit_multiset = Counter(frozenset(sprite["hits"]) for sprite in sprites)
    if hit_multiset != Counter(planned_hits):
        return False

    if not any(len(sprite["hits"]) == 0 for sprite in sprites):
        return False
    if not any(len(sprite["hits"]) == 1 for sprite in sprites):
        return False
    if not any(len(sprite["hits"]) >= 2 for sprite in sprites):
        return False

    singleton_colors = {
        int(sources[sprite["hits"][0]]["color"])
        for sprite in sprites
        if len(sprite["hits"]) == 1
    }
    if len(singleton_colors) < min(2, len(sources)):
        return False

    ray_hits = Counter(idx for sprite in sprites for idx in sprite["hits"])
    if any(ray_hits[idx] == 0 for idx in range(len(sources))):
        return False

    no_hit_near_ray = any(
        sprite["hits"] == tuple()
        and min(abs(int(r + c) - src["diag"]) for r, c in sprite["coords"] for src in sources) == 1
        for sprite in sprites
    )
    if not no_hit_near_ray:
        return False

    expected = _expected_output_semantic(grid)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(grid, actual):
        return False
    if len(set(np.unique(actual)) - {BACKGROUND}) < 3:
        return False
    return True


def _expected_output_semantic(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sources, sprites = _analyze_hits(g)
    for sprite in sprites:
        hit_count = len(sprite["hits"])
        if hit_count == 1:
            color = int(sources[sprite["hits"][0]]["color"])
            out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = color
        elif hit_count >= 2:
            out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = MULTI_HIT_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    sources = _sample_source_layout(h, w)
    if sources is None:
        return None

    candidates = _candidate_placements(h, w, sources)
    hit_sets = _required_hit_sets(len(sources))
    placements = None
    for _ in range(PLACEMENT_RESTARTS):
        placements = _pick_placements(candidates, hit_sets)
        if placements is not None:
            break
    if placements is None:
        return None

    grid = _render_grid(h, w, sources, placements)
    if _validate_generated(grid, [sprite["hits"] for sprite in placements]):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray sprite puzzle")
