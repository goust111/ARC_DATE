from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BASE_HOOK = ((0, 0), (1, 0), (2, 0), (2, 1))
GEN_RETRIES = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_key(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _c in shape) + 1
    w = max(c for _r, c in shape) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _mask_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _shape_variants(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    base = _shape_mask(_shape_key(cells))
    variants: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for flip in (False, True):
            cur = np.fliplr(rot) if flip else rot
            key = _mask_shape(cur)
            if key not in seen:
                seen.add(key)
                variants.append(key)
    return variants


HOOK_VARIANTS = tuple(_shape_variants(BASE_HOOK))


def _boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _find_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        border = _boundary(r0, c0, r1, c1)
        if set(comp) != border:
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == GREY):
            continue
        frames.append({"bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda x: x["bbox"])
    assert frames, "expected at least one grey rectangular window frame"
    return frames


def _match_gaps_to_fragments(gaps: list[dict], fragments: list[dict]) -> dict[int, int]:
    assert len(gaps) == len(fragments), "gap and fragment counts must match"
    options: list[list[int]] = []
    for gap in gaps:
        opts = [
            i
            for i, frag in enumerate(fragments)
            if gap["shape"] in frag["variants"]
        ]
        assert opts, "each gap must be matched by a transformed same-color fragment"
        options.append(opts)

    order = sorted(range(len(gaps)), key=lambda i: (len(options[i]), gaps[i]["shape"]))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        gi = order[pos]
        for fi in sorted(options[gi], key=lambda j: fragments[j]["bbox"]):
            if fi in used:
                continue
            used.add(fi)
            assignment[gi] = fi
            if rec(pos + 1):
                return True
            used.remove(fi)
            del assignment[gi]
        return False

    assert rec(0), "expected a unique complete gap-to-fragment matching"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move transformed same-color scraps into the black holes inside grey window frames.

    Input structure:
    - Black is the background and Grey draws separated one-pixel-thick rectangular window frames.
    - Each frame interior contains exactly one non-Black, non-Grey pattern color with one or more Black 4-connected gaps.
    - Outside all frames, small 4-connected scraps of the same colors are scattered on the Black background.

    Rule:
    - For each frame, use the unique interior pattern color to select its outside scraps.
    - Treat every Black component inside the frame interior as a missing shape.
    - Rotate and/or horizontally flip each same-color outside scrap until its shape matches one missing shape.
    - Fill the matched missing cells with the pattern color and erase the original outside scrap cells to Black.

    Key insight:
    - Color assigns scraps to windows, while the D4 shape match reveals which rotated or reflected scrap completes each hidden part.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    frames = _find_frames(g)
    blocked = np.zeros(g.shape, dtype=bool)
    gaps_by_color: dict[int, list[dict]] = {}

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = {int(v) for v in np.unique(interior)} - {BLACK, GREY}
        assert len(colors) == 1, "each window interior must have one pattern color"
        color = colors.pop()

        gap_components = _components(interior == BLACK)
        assert gap_components, "each window must contain at least one black gap"
        for comp in gap_components:
            global_cells = [(r0 + 1 + r, c0 + 1 + c) for r, c in comp]
            gaps_by_color.setdefault(color, []).append(
                {"cells": global_cells, "shape": _shape_key(comp)}
            )

    for color, gaps in gaps_by_color.items():
        fragments: list[dict] = []
        for comp in _components((g == color) & (~blocked)):
            fragments.append(
                {
                    "cells": comp,
                    "bbox": _bbox(comp),
                    "variants": _shape_variants(comp),
                }
            )
        assignment = _match_gaps_to_fragments(gaps, fragments)

        for gi, gap in enumerate(gaps):
            for r, c in gap["cells"]:
                out[r, c] = int(color)
            frag = fragments[assignment[gi]]
            for r, c in frag["cells"]:
                out[r, c] = BLACK

    return out


# --------------- generator ---------------


def _patch_from_shape(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = max(r for r, _c in shape) + 1
    w = max(c for _r, c in shape) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _window_patch(h: int, w: int, color: int, hole_shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[1:-1, 1:-1] = int(color)

    sh = max(r for r, _c in hole_shape) + 1
    sw = max(c for _r, c in hole_shape) + 1
    inner_h, inner_w = h - 2, w - 2
    rr = random.randint(0, inner_h - sh)
    cc = random.randint(0, inner_w - sw)
    for r, c in hole_shape:
        patch[1 + rr + r, 1 + cc + c] = BLACK
    return patch


def _expanded_box(box: tuple[int, int, int, int], gap: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - gap, c0 - gap, r1 + gap, c1 + gap


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _random_origin(
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    *,
    gap: int = 1,
    border: int = 1,
) -> tuple[int, int] | None:
    h, w = shape
    ph, pw = patch_shape
    positions = [
        (r0, c0)
        for r0 in range(border, h - ph - border + 1)
        for c0 in range(border, w - pw - border + 1)
    ]
    random.shuffle(positions)
    for r0, c0 in positions:
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        padded = _expanded_box(box, gap)
        if any(_boxes_overlap(padded, old) for old in boxes):
            continue
        return r0, c0
    return None


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BLACK
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    n_windows = random.randint(2, 4)
    colors = random.sample(COLORS, n_windows)

    grid = np.zeros((h, w), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    fragments: list[np.ndarray] = []

    for color in colors:
        placed = False
        for _ in range(80):
            ph = random.randint(5, 8)
            pw = random.randint(5, 8)
            inner_h, inner_w = ph - 2, pw - 2
            hole_choices = [
                shape
                for shape in HOOK_VARIANTS
                if max(r for r, _c in shape) < inner_h and max(c for _r, c in shape) < inner_w
            ]
            if not hole_choices:
                continue
            hole_shape = random.choice(hole_choices)
            patch = _window_patch(ph, pw, color, hole_shape)
            origin = _random_origin(grid.shape, patch.shape, boxes, gap=1, border=1)
            if origin is None:
                continue
            r0, c0 = origin
            _paste(grid, patch, r0, c0)
            boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

            fragment_shape = random.choice(HOOK_VARIANTS)
            fragments.append(_patch_from_shape(fragment_shape, color))
            placed = True
            break
        if not placed:
            return None

    random.shuffle(fragments)
    for patch in fragments:
        origin = _random_origin(grid.shape, patch.shape, boxes, gap=1, border=1)
        if origin is None:
            return None
        r0, c0 = origin
        _paste(grid, patch, r0, c0)
        boxes.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
