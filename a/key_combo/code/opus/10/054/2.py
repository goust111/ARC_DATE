from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

BODY_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
GEN_RETRY_LIMIT = 500
PLACEMENT_RETRY_LIMIT = 200
GRID_MIN = 18
GRID_MAX = 24
MAX_OFFSET = 3

_BASE_TEMPLATES = (
    {(0, 0), (0, 1), (1, 0), (-1, 0)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, -1), (-1, -1)},
    {(0, 0), (1, 0), (0, -1), (1, -1)},
    {(0, 0), (0, 1), (-1, 0), (-1, 1)},
)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_output(coords: list[tuple[int, int]], values: dict[tuple[int, int], int]) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(arr)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rc in coords:
        out[rc[0] - r0, rc[1] - c0] = values[rc]
    return out


def _extract_sprites(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _components(grid != BLACK)
    sprites: list[dict[str, object]] = []
    for comp in comps:
        vals = [int(grid[r, c]) for r, c in comp]
        counts = Counter(vals)
        assert counts[GREY] == 1, "each sprite must have exactly one gray anchor"
        body_colors = [color for color in counts if color not in (BLACK, GREY)]
        assert len(body_colors) == 1, "each sprite must have exactly one body color"
        color = int(body_colors[0])
        anchor_abs = tuple(map(int, comp[np.argmax(np.array(vals) == GREY)]))
        body_offsets: set[tuple[int, int]] = set()
        for r, c in comp:
            v = int(grid[r, c])
            if v in (BLACK, GREY):
                continue
            body_offsets.add((int(r - anchor_abs[0]), int(c - anchor_abs[1])))
        assert 3 <= len(body_offsets) <= 7, "body must have 3-7 pixels"
        full_offsets = body_offsets | {(0, 0)}
        assert 4 <= len(full_offsets) <= 8
        sprites.append(
            {
                "color": color,
                "anchor_abs": anchor_abs,
                "body_offsets": body_offsets,
                "full_offsets": full_offsets,
                "bbox": _bbox_from_coords(comp),
            }
        )
    assert 3 <= len(sprites) <= 5, "expected 3-5 separated sprites"
    return sprites


def _parity_render(body_sets: list[set[tuple[int, int]]]) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    coverage: Counter[tuple[int, int]] = Counter()
    for body in body_sets:
        for rc in body:
            coverage[rc] += 1
    assert coverage, "at least one body pixel is required"
    coords = list(coverage)
    values = {
        rc: (GREEN if count % 2 == 1 else RED)
        for rc, count in coverage.items()
        if count > 0
    }
    return _normalize_output(coords, values), dict(coverage)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align separated anchor sprites, combine their body layers by parity, and crop the composite.

    Input structure:
    - Black background contains 3-5 separated 4-connected sprites.
    - Each sprite has exactly one gray anchor pixel and 3-7 body pixels of one other color.
    - Different sprites may have different shapes, but each is handled independently by its own anchor.

    Rule:
    - Isolate each sprite and record every body pixel relative to that sprite's gray anchor.
    - Translate all sprites so their gray anchors coincide on one common point.
    - Count, for every aligned location, how many sprites contribute a non-gray body pixel there:
      odd counts become green, positive even counts become red, zero stays black.
    - Ignore gray anchor pixels themselves when counting, then crop to the smallest non-black box.

    Key insight:
    - The gray pixels are registration marks; only after anchor alignment can the multi-layer overlap
      counts be evaluated, and the final color is determined by coverage parity rather than by source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sprites = _extract_sprites(g)
    out, _coverage = _parity_render([sprite["body_offsets"] for sprite in sprites])
    return out


def _transform_point(rc: tuple[int, int], mode: int) -> tuple[int, int]:
    r, c = rc
    if mode == 0:
        return r, c
    if mode == 1:
        return r, -c
    if mode == 2:
        return -r, c
    if mode == 3:
        return -r, -c
    if mode == 4:
        return c, r
    if mode == 5:
        return c, -r
    if mode == 6:
        return -c, r
    return -c, -r


def _grow_shape(
    seed: set[tuple[int, int]],
    target_size: int,
    shared_freq: Counter[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    cells = set(seed)
    while len(cells) < target_size:
        candidates: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cells:
                    continue
                if max(abs(nr), abs(nc)) > MAX_OFFSET:
                    continue
                candidates.append((nr, nc))
        if not candidates:
            return None
        unique_candidates = sorted(set(candidates))
        scored = []
        for cand in unique_candidates:
            overlap_penalty = 4 * shared_freq[cand]
            extent_penalty = max(abs(cand[0]), abs(cand[1]))
            scored.append((overlap_penalty + extent_penalty, random.random(), cand))
        best_score = min(score for score, _rand, _cand in scored)
        pool = [cand for score, _rand, cand in scored if score <= best_score + 1]
        cells.add(random.choice(pool))
    return cells


def _full_to_body(full_offsets: set[tuple[int, int]]) -> set[tuple[int, int]]:
    body = set(full_offsets)
    body.remove((0, 0))
    return body


def _sprite_anchor_in_bbox(full_offsets: set[tuple[int, int]]) -> tuple[int, int]:
    arr = np.asarray(list(full_offsets), dtype=int)
    r0, c0, _r1, _c1 = _bbox_from_coords(arr)
    return int(-r0), int(-c0)


def _render_top_left_variant(full_sets: list[set[tuple[int, int]]]) -> np.ndarray:
    aligned_body_sets: list[set[tuple[int, int]]] = []
    for full in full_sets:
        arr = np.asarray(list(full), dtype=int)
        r0, c0, _r1, _c1 = _bbox_from_coords(arr)
        body = {(r - r0, c - c0) for r, c in full if (r, c) != (0, 0)}
        aligned_body_sets.append(body)
    out, _coverage = _parity_render(aligned_body_sets)
    return out


def _binary_overlap_variant(body_sets: list[set[tuple[int, int]]]) -> np.ndarray:
    _out, coverage = _parity_render(body_sets)
    values = {
        rc: (GREEN if count == 1 else RED)
        for rc, count in coverage.items()
        if count > 0
    }
    return _normalize_output(list(values), values)


def _shape_is_connected(full_offsets: set[tuple[int, int]]) -> bool:
    mask = np.zeros((2 * MAX_OFFSET + 3, 2 * MAX_OFFSET + 3), dtype=bool)
    shift = MAX_OFFSET + 1
    for r, c in full_offsets:
        mask[r + shift, c + shift] = True
    comps = _components(mask)
    return len(comps) == 1


def _shape_to_sprite(full_offsets: set[tuple[int, int]], color: int) -> np.ndarray:
    arr = np.asarray(list(full_offsets), dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(arr)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in full_offsets:
        sprite[r - r0, c - c0] = GREY if (r, c) == (0, 0) else color
    return sprite


def _place_sprites(sprites: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    order = sorted(range(len(sprites)), key=lambda idx: sprites[idx].size, reverse=True)
    for idx in order:
        sprite = sprites[idx]
        sh, sw = sprite.shape
        if sh + 2 > h or sw + 2 > w:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRY_LIMIT):
            r0 = random.randint(1, h - sh - 1)
            c0 = random.randint(1, w - sw - 1)
            cells = [(r0 + r, c0 + c) for r, c in np.argwhere(sprite != BLACK)]
            halo = {
                (rr, cc)
                for r, c in cells
                for rr in range(r - 1, r + 2)
                for cc in range(c - 1, c + 2)
                if 0 <= rr < h and 0 <= cc < w
            }
            if any(cell in occupied for cell in halo):
                continue
            for r, c in np.argwhere(sprite != BLACK):
                grid[r0 + r, c0 + c] = int(sprite[r, c])
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None
    return grid


def _build_shapes() -> list[set[tuple[int, int]]] | None:
    n_sprites = random.randint(3, 5)
    transform = random.randint(0, 7)
    template_indices = list(range(3))
    if n_sprites >= 4:
        template_indices.append(random.choice([3, 4]))
    if n_sprites == 5:
        choices = [3, 4]
        random.shuffle(choices)
        template_indices.append(choices[0] if template_indices[-1] != choices[0] else choices[1])

    full_sets = []
    for idx in template_indices:
        transformed = {_transform_point(rc, transform) for rc in _BASE_TEMPLATES[idx]}
        full_sets.append(transformed)

    shared_freq: Counter[tuple[int, int]] = Counter()
    for full in full_sets:
        for rc in full:
            if rc != (0, 0):
                shared_freq[rc] += 1

    grown_sets: list[set[tuple[int, int]]] = []
    for full in full_sets:
        target_size = random.randint(len(full), 8)
        grown = _grow_shape(full, target_size, shared_freq)
        if grown is None or not _shape_is_connected(grown):
            return None
        if not (4 <= len(grown) <= 8):
            return None
        body = _full_to_body(grown)
        if not (3 <= len(body) <= 7):
            return None
        grown_sets.append(grown)
        for rc in body:
            shared_freq[rc] += 1
    return grown_sets


def _semantic_checks(full_sets: list[set[tuple[int, int]]], grid: np.ndarray) -> bool:
    body_sets = [_full_to_body(full) for full in full_sets]
    expected, coverage = _parity_render(body_sets)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    counts = list(coverage.values())
    if not any(count == 1 for count in counts):
        return False
    if not any(count >= 2 and count % 2 == 0 for count in counts):
        return False
    if not any(count >= 3 and count % 2 == 1 for count in counts):
        return False

    if not (np.any(expected == GREEN) and np.any(expected == RED)):
        return False

    top_left_variant = _render_top_left_variant(full_sets)
    if np.array_equal(expected, top_left_variant):
        return False

    binary_variant = _binary_overlap_variant(body_sets)
    if np.array_equal(expected, binary_variant):
        return False

    anchor_positions = {_sprite_anchor_in_bbox(full) for full in full_sets}
    if len(anchor_positions) < 2:
        return False

    overlap_pixels = sum(count > 1 for count in counts)
    if overlap_pixels == 0 or overlap_pixels > max(4, len(counts) // 2):
        return False

    extracted = _extract_sprites(grid)
    if len(extracted) != len(full_sets):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    full_sets = _build_shapes()
    if full_sets is None:
        return None

    colors = random.sample(list(BODY_COLORS), len(full_sets))
    sprites = [_shape_to_sprite(full, color) for full, color in zip(full_sets, colors)]

    for _ in range(PLACEMENT_RETRY_LIMIT):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        grid = _place_sprites(sprites, (h, w))
        if grid is None:
            continue
        if _semantic_checks(full_sets, grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-parity overlay input")
