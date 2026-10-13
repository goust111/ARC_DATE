"""Reconstruct the occluded side of a transpose-symmetric square by a learned recolor involution.

Input structure
The input is a square grid. One non-dominant color forms exactly one solid rectangular patch strictly on one side of the main diagonal and fully occludes that region.
Aside from that patch, every off-diagonal pair (i,j) and (j,i) is related by one global color involution: some colors swap in pairs and other colors map to themselves; diagonal cells are all fixed-color values.

Rule
- Identify the unique solid rectangle color component that lies entirely above or entirely below the main diagonal and treat it as the occluder.
- From all off-diagonal visible pairs that do not involve the occluder color, infer the global involution mapping f by enforcing f(a)=b and f(b)=a for each pair (a,b).
- For each cell inside the occluded rectangle, read the color at its transpose position and place f(that color) into the output at the same rectangle-relative position.
- Return only the restored rectangle.

Key insight
The symmetry holds only after recoloring, so direct copying of transpose values is wrong; the missing patch is obtained by taking its transpose values and passing them through a hidden self-consistent color map read from the rest of the grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, bbox, colors_present, components_mask


Grid = NDArray[np.integer]


def _find_cover_candidates(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w, "input grid must be square"

    candidates: list[tuple[int, int, int, int, int]] = []
    for color in colors_present(g):
        mask = g == color
        comps = components_mask(mask, connectivity=4)
        for comp in comps:
            area = int(comp.shape[0])
            if area < 4:
                continue
            r0, c0 = comp.min(axis=0)
            r1, c1 = comp.max(axis=0)
            r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
            rect_h, rect_w = r1 - r0 + 1, c1 - c0 + 1
            if area != rect_h * rect_w:
                continue

            rows = comp[:, 0] - comp[:, 1]
            if np.all(rows < 0) or np.all(rows > 0):
                pass
            else:
                continue
            if rect_h == 1 or rect_w == 1:
                continue
            if area > (h * w) // 2:
                continue

            candidates.append((int(color), r0, c0, rect_h, rect_w))

    return candidates


def _infer_involution(grid: np.ndarray, cover_color: int) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    n, m = g.shape
    assert n == m, "input grid must be square"

    mapping: dict[int, int] = {}

    for d in range(n):
        color = int(g[d, d])
        if color == cover_color:
            continue
        if color in mapping:
            assert mapping[color] == color, f"diagonal color {color} must be fixed"
        mapping[color] = color

    for r in range(n):
        for c in range(r + 1, n):
            a = int(g[r, c])
            b = int(g[c, r])
            if a == cover_color or b == cover_color:
                continue
            if a == b:
                if a in mapping:
                    assert mapping[a] == a, "same-color diagonal and off-diagonal pairs must map to themselves"
                else:
                    mapping[a] = a
                continue

            if a in mapping:
                assert mapping[a] == b, f"inconsistent mapping for {a}: {mapping[a]} vs {b}"
            else:
                mapping[a] = b

            if b in mapping:
                assert mapping[b] == a, f"inconsistent mapping for {b}: {mapping[b]} vs {a}"
            else:
                mapping[b] = a

    for a, b in list(mapping.items()):
        if a == cover_color or b == cover_color:
            continue
        assert mapping.get(b, b) == a, f"mapping is not an involution for {a} -> {b}"

    return mapping


def _restore_hidden_region(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D input"
    h, w = g.shape
    assert h == w, "expected square input"

    candidates = _find_cover_candidates(g)
    assert len(candidates) == 1, f"expected exactly one occluder rectangle candidate, got {len(candidates)}"
    cover_color, r0, c0, rect_h, rect_w = candidates[0]

    mapping = _infer_involution(g, cover_color)

    out = np.empty((rect_h, rect_w), dtype=int)
    for rr in range(rect_h):
        for cc in range(rect_w):
            src = int(g[c0 + cc, r0 + rr])
            assert src != cover_color, "transpose source cannot be hidden"
            assert src in mapping, f"no mapping entry for source color {src}"
            out[rr, cc] = mapping[src]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct the hidden rectangle by transpose pairing + recolor involution.

    Input structure:
    - A square grid with one solid rectangular occluder color located strictly above or below the main diagonal.
    - Other cells follow a global transpose color involution: off-diagonal pairs (i,j)/(j,i) are related by the same map, and diagonal cells are fixed points.

    Rule:
    - Find the unique solid rectangular occluder component off the diagonal.
    - Infer the involution from all visible off-diagonal pairs that do not involve the occluder color.
    - For each occluded cell, recolor the value from its transpose via the inferred involution and emit only the restored rectangle.

    Key insight:
    - The picture is symmetric only after recoloring; solving requires discovering that color map from visible pairs before transposing back.
    """
    return _restore_hidden_region(np.asarray(grid, dtype=int))


def _build_base_grid(
    size: int,
    mapping: dict[int, int],
    fixed_colors: list[int],
) -> np.ndarray:
    g = np.empty((size, size), dtype=int)
    n = int(size)

    for d in range(n):
        g[d, d] = int(np.random.choice(fixed_colors))

    colors = list(mapping.keys())
    for r in range(n):
        for c in range(r + 1, n):
            src = int(np.random.choice(colors))
            dst = mapping[src]
            g[r, c] = src
            g[c, r] = dst
    return g


def _sample_involution_colors() -> tuple[dict[int, int], list[int]]:
    pool = [int(c) for c in Color.NOT_BLACK]
    np.random.shuffle(pool)
    color_count = int(np.random.randint(4, 8))
    colors = pool[:color_count]

    possible_pairs = [k for k in range(1, color_count // 2 + 1) if color_count - 2 * k >= 1]
    pair_count = int(np.random.choice(possible_pairs))
    fixed_count = color_count - 2 * pair_count

    mapping_colors = colors[:]
    np.random.shuffle(mapping_colors)
    fixed_colors = mapping_colors[:fixed_count]
    pair_colors = mapping_colors[fixed_count:]

    mapping: dict[int, int] = {int(c): int(c) for c in fixed_colors}
    for i in range(0, len(pair_colors), 2):
        a, b = int(pair_colors[i]), int(pair_colors[i + 1])
        mapping[a] = b
        mapping[b] = a

    return mapping, list(dict.fromkeys(fixed_colors))


def _sample_cover_rect(size: int, *, side: str, max_thickness: int = 8) -> tuple[int, int, int, int]:
    assert side in {"upper", "lower"}
    for _ in range(128):
        rect_h = int(np.random.randint(2, max_thickness + 1))
        rect_w = int(np.random.randint(2, max_thickness + 1))
        if rect_h >= size or rect_w >= size:
            continue

        if side == "upper":
            r0 = int(np.random.randint(0, size - rect_h))
            c0_min = r0 + rect_h
            c0_max = size - rect_w
            if c0_min <= c0_max:
                c0 = int(np.random.randint(c0_min, c0_max + 1))
                return r0, c0, rect_h, rect_w
        else:
            c0 = int(np.random.randint(0, size - rect_w))
            r0_min = c0 + rect_w
            r0_max = size - rect_h
            if r0_min <= r0_max:
                r0 = int(np.random.randint(r0_min, r0_max + 1))
                return r0, c0, rect_h, rect_w

    raise RuntimeError("failed to sample a valid occluder rectangle")


def _expected_region(
    base: np.ndarray,
    rect: tuple[int, int, int, int],
    mapping: dict[int, int],
) -> np.ndarray:
    r0, c0, rect_h, rect_w = rect
    out = np.empty((rect_h, rect_w), dtype=int)
    for rr in range(rect_h):
        for cc in range(rect_w):
            out[rr, cc] = mapping[int(base[c0 + cc, r0 + rr])]
    return out


def _pick_cover_color(mapping: dict[int, int]) -> int:
    candidates = [c for c in Color.NOT_BLACK if c not in mapping]
    if not candidates:
        candidates = [int(c) for c in range(1, 10) if c not in mapping]
    return int(np.random.choice(candidates))


def generate_input() -> np.ndarray:
    for _ in range(800):
        size = int(np.random.randint(12, 25))

        mapping, fixed_colors = _sample_involution_colors()
        if len(fixed_colors) < 1:
            continue

        side = "upper" if int(np.random.randint(2)) == 0 else "lower"
        try:
            rect = _sample_cover_rect(size, side=side)
        except RuntimeError:
            continue
        r0, c0, rect_h, rect_w = rect

        base = _build_base_grid(size, mapping, fixed_colors)
        cover_color = _pick_cover_color(mapping)

        inp = base.copy()
        inp[r0 : r0 + rect_h, c0 : c0 + rect_w] = cover_color

        # Ensure the target solver can recover and the candidate is unique.
        try:
            out = _restore_hidden_region(inp)
            expected = _expected_region(base, rect, mapping)
        except Exception:
            continue

        if not np.array_equal(out, expected):
            continue

        # Ensure uniqueness and geometry assumptions are not degenerate.
        candidates = _find_cover_candidates(inp)
        if len(candidates) != 1:
            continue
        cand = candidates[0]
        if cand[0] == cover_color and cand[1:]==(r0,c0,rect_h,rect_w):
            pass
        else:
            continue

        out_h = int(out.shape[0])
        out_w = int(out.shape[1])
        if not (2 <= out_h <= 8 and 2 <= out_w <= 8):
            continue

        return inp

    raise RuntimeError("failed to generate valid input for 4.py")
