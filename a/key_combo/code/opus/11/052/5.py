from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
OCCLUDER = 1
SPRITE = 2
KEY_COLORS = (3, 4, 6, 7, 8, 9)
SLOT = 5
GAP = 1
LATTICE = 5
GRID = LATTICE * SLOT + (LATTICE - 1) * GAP
MAX_ATTEMPTS = 400


SPRITE_TEMPLATES = [
    "11111/10101/10011/11111/00000",
    "11111/11001/10111/11111/00000",
    "11111/11101/10011/11111/00000",
    "11111/10111/11001/11111/00000",
    "11111/10101/10111/11111/00000",
    "11111/10101/11101/11111/00000",
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), rel


def _slot_origin(br: int, bc: int) -> tuple[int, int]:
    step = SLOT + GAP
    return br * step, bc * step


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _sprite_mask_from_text(text: str) -> np.ndarray:
    rows = text.split("/")
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _interior_holes(sprite_mask: np.ndarray) -> list[np.ndarray]:
    holes = []
    for coords in _components(~sprite_mask):
        if np.any((coords[:, 0] == 0) | (coords[:, 0] == sprite_mask.shape[0] - 1) | (coords[:, 1] == 0) | (coords[:, 1] == sprite_mask.shape[1] - 1)):
            continue
        holes.append(coords)
    return holes


def _find_occluder(grid: np.ndarray, bg: int) -> tuple[int, str, int]:
    h, w = grid.shape
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if len(coords) != area:
                continue
            if c0 == 0 and c1 == w - 1 and r1 - r0 + 1 == SLOT:
                candidates.append((color, "row", r0 // (SLOT + GAP)))
            if r0 == 0 and r1 == h - 1 and c1 - c0 + 1 == SLOT:
                candidates.append((color, "col", c0 // (SLOT + GAP)))
    assert len(candidates) == 1
    return candidates[0]


def _extract_sprite_mask(grid: np.ndarray, bg: int, sprite_color: int, occluder_color: int) -> np.ndarray:
    masks = []
    for br in range(LATTICE):
        for bc in range(LATTICE):
            r0, c0 = _slot_origin(br, bc)
            block = grid[r0 : r0 + SLOT, c0 : c0 + SLOT]
            if np.any(block == occluder_color):
                continue
            colors = set(map(int, np.unique(block)))
            if colors <= {int(bg), int(sprite_color)} and np.any(block == sprite_color):
                masks.append(block == sprite_color)
    assert masks
    sprite_mask = masks[0]
    for mask in masks[1:]:
        assert np.array_equal(mask, sprite_mask)
    return sprite_mask


def _key_shape_to_color(grid: np.ndarray, bg: int, occluder_color: int, sprite_color: int) -> dict:
    mapping = {}
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg), int(occluder_color), int(sprite_color)}):
        comps = _components(grid == color)
        assert len(comps) == 1
        mapping[_shape_key(comps[0])] = int(color)
    return mapping


def _render_lattice(sprite_mask: np.ndarray, hole_colors: dict | None = None, *, only_slots: set[tuple[int, int]] | None = None) -> np.ndarray:
    out = np.full((GRID, GRID), BG, dtype=int)
    sprite_patch = np.where(sprite_mask, SPRITE, BG)
    holes = _interior_holes(sprite_mask)
    for br in range(LATTICE):
        for bc in range(LATTICE):
            if not sprite_mask[br, bc]:
                continue
            if only_slots is not None and (br, bc) not in only_slots:
                continue
            r0, c0 = _slot_origin(br, bc)
            _paste(out, sprite_patch, r0, c0)
            if hole_colors is None:
                continue
            for hole in holes:
                color = hole_colors[_shape_key(hole)]
                rr = hole[:, 0] + r0
                cc = hole[:, 1] + c0
                out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the occluded self-similar sprite lattice, then color every sprite hole by matching it to an external key shape.

    Input structure:
    - A 29x29 grid contains a 5x5 lattice of 5x5 slots separated by single black gaps.
    - Every occupied slot holds the same red 5x5 sprite; the sprite's own red mask is also the coarse lattice mask, so slot (r,c) is occupied iff sprite cell (r,c) is red.
    - One full slot row or one full slot column is hidden by a solid blue bar.
    - The sprite contains two enclosed background holes of different shapes, and two small monochrome key objects of other colors appear in the otherwise empty bottom slot-row.

    Rule:
    - Recover the full coarse lattice by reading the sprite mask from any visible red copy and using that mask as the slot-occupancy map.
    - Remove the blue bar and redraw the red sprite in every required slot, including the hidden ones.
    - Normalize each internal hole shape of the sprite and match it to the uniquely equal normalized key shape.
    - Fill that hole in every sprite copy with the matched key color, then erase the keys by returning only the restored lattice.

    Key insight:
    - The same 5x5 sprite encodes two independent clues at once: its red pixels tell you which slots should exist, while its enclosed background holes tell you which external key colors belong inside every copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID, GRID)

    bg = _get_bg(g)
    occluder_color, _axis, _hidden_idx = _find_occluder(g, bg)

    fg_colors = sorted(set(map(int, np.unique(g))) - {int(bg), int(occluder_color)})
    counts = {color: int(np.sum(g == color)) for color in fg_colors}
    sprite_color = max(counts, key=counts.get)

    sprite_mask = _extract_sprite_mask(g, bg, sprite_color, occluder_color)
    hole_shapes = _interior_holes(sprite_mask)
    assert len(hole_shapes) == 2
    assert len({_shape_key(hole) for hole in hole_shapes}) == len(hole_shapes)

    shape_to_color = _key_shape_to_color(g, bg, occluder_color, sprite_color)
    assert len(shape_to_color) == len(hole_shapes)
    for hole in hole_shapes:
        assert _shape_key(hole) in shape_to_color

    return _render_lattice(sprite_mask, hole_colors=shape_to_color)


def _build_key_specs(sprite_mask: np.ndarray, axis: str, hidden_idx: int):
    holes = _interior_holes(sprite_mask)
    colors = random.sample(KEY_COLORS, len(holes))
    blocked_col = hidden_idx if axis == "col" else None
    free_cols = [c for c in range(LATTICE) if c != blocked_col]
    key_cols = random.sample(free_cols, len(holes))
    order = list(range(len(holes)))
    random.shuffle(order)

    key_specs = []
    shape_to_color = {}
    for slot_col, hole_idx, color in zip(key_cols, order, colors):
        hole = holes[hole_idx]
        hr0, hc0, hr1, hc1 = _bbox(hole)
        h = hr1 - hr0 + 1
        w = hc1 - hc0 + 1
        max_dr = SLOT - h
        max_dc = SLOT - w
        dr_lo = 1 if max_dr >= 2 else 0
        dc_lo = 1 if max_dc >= 2 else 0
        dr_hi = max_dr - 1 if max_dr >= 2 else max_dr
        dc_hi = max_dc - 1 if max_dc >= 2 else max_dc
        dr = random.randint(dr_lo, dr_hi)
        dc = random.randint(dc_lo, dc_hi)
        patch = np.full((SLOT, SLOT), BG, dtype=int)
        rel = hole - np.array([hr0, hc0])
        patch[rel[:, 0] + dr, rel[:, 1] + dc] = int(color)
        key_specs.append((slot_col, patch))
        shape_to_color[_shape_key(hole)] = int(color)
    return key_specs, shape_to_color


def _candidate_hidden_indices(sprite_mask: np.ndarray, axis: str) -> list[int]:
    if axis == "row":
        lines = [sprite_mask[r] for r in range(LATTICE - 1)]
    else:
        lines = [sprite_mask[:, c] for c in range(LATTICE)]
    return [
        idx
        for idx, line in enumerate(lines)
        if np.any(line) and np.any(~line)
    ]


def _make_input_from_mask(sprite_mask: np.ndarray, axis: str, hidden_idx: int):
    key_specs, shape_to_color = _build_key_specs(sprite_mask, axis, hidden_idx)
    out = _render_lattice(sprite_mask, hole_colors=shape_to_color)
    inp = _render_lattice(sprite_mask, hole_colors=None)
    for slot_col, patch in key_specs:
        r0, c0 = _slot_origin(LATTICE - 1, slot_col)
        _paste(inp, patch, r0, c0)

    line_start = hidden_idx * (SLOT + GAP)
    if axis == "row":
        inp[line_start : line_start + SLOT, :] = OCCLUDER
    else:
        inp[:, line_start : line_start + SLOT] = OCCLUDER
    return inp, out, shape_to_color


def _validate_instance(inp: np.ndarray, out: np.ndarray, sprite_mask: np.ndarray, axis: str, hidden_idx: int, shape_to_color: dict) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(inp, out):
        return False

    hidden_slots = {
        (br, bc)
        for br in range(LATTICE)
        for bc in range(LATTICE)
        if sprite_mask[br, bc] and ((axis == "row" and br == hidden_idx) or (axis == "col" and bc == hidden_idx))
    }
    visible_slots = {
        (br, bc)
        for br in range(LATTICE)
        for bc in range(LATTICE)
        if sprite_mask[br, bc] and (br, bc) not in hidden_slots
    }
    if not hidden_slots or not visible_slots:
        return False

    restored_unfilled = _render_lattice(sprite_mask, hole_colors=None)
    filled_visible_only = _render_lattice(sprite_mask, hole_colors=shape_to_color, only_slots=visible_slots)
    if np.array_equal(restored_unfilled, out):
        return False
    if np.array_equal(filled_visible_only, out):
        return False

    holes = _interior_holes(sprite_mask)
    if len(holes) != 2 or len({_shape_key(h) for h in holes}) != 2:
        return False
    if set(shape_to_color) != {_shape_key(h) for h in holes}:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sprite_mask = _sprite_mask_from_text(random.choice(SPRITE_TEMPLATES))
        axis = random.choice(("row", "col"))
        hidden_candidates = _candidate_hidden_indices(sprite_mask, axis)
        if not hidden_candidates:
            continue
        hidden_idx = random.choice(hidden_candidates)
        inp, out, shape_to_color = _make_input_from_mask(sprite_mask, axis, hidden_idx)
        if _validate_instance(inp, out, sprite_mask, axis, hidden_idx, shape_to_color):
            return inp
    raise RuntimeError("failed to generate a valid input")
