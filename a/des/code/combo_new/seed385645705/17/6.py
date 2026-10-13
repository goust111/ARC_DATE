from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
ARC_COLORS = tuple(range(10))
CANVAS_MIN = 12
CANVAS_MAX = 28
RECT_SIDE_MIN = 4
RECT_SIDE_MAX = 9
MAX_OBJECTS = 4
MAX_GENERATION_ATTEMPTS = 300
CORNERS = ("tl", "tr", "bl", "br")


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _diag_signature(coords: np.ndarray) -> tuple[str, int] | None:
    if coords.shape[0] < 2:
        return None
    sums = coords[:, 0] + coords[:, 1]
    if np.all(sums == sums[0]):
        pts = coords[np.argsort(coords[:, 0])]
        if np.all(np.diff(pts[:, 0]) == 1) and np.all(np.diff(pts[:, 1]) == -1):
            return ("sum", int(sums[0]))
    diffs = coords[:, 0] - coords[:, 1]
    if np.all(diffs == diffs[0]):
        pts = coords[np.argsort(coords[:, 0])]
        if np.all(np.diff(pts[:, 0]) == 1) and np.all(np.diff(pts[:, 1]) == 1):
            return ("diff", int(diffs[0]))
    return None


def _find_grey_color(sub: np.ndarray) -> int:
    colors = [int(v) for v in np.unique(sub)]
    if GREY in colors:
        sig = _diag_signature(np.argwhere(sub == GREY))
        if sig is not None:
            return GREY
    candidates: list[int] = []
    for color in colors:
        sig = _diag_signature(np.argwhere(sub == color))
        if sig is not None:
            candidates.append(color)
    assert len(candidates) == 1, "expected a unique diagonal crease color"
    return int(candidates[0])


def _to_normalized(r: int, c: int, shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return r, c
    if corner == "tr":
        return r, w - 1 - c
    if corner == "bl":
        return h - 1 - r, c
    if corner == "br":
        return h - 1 - r, w - 1 - c
    raise AssertionError(f"unknown corner {corner}")


def _from_normalized(r: int, c: int, shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return r, c
    if corner == "tr":
        return r, w - 1 - c
    if corner == "bl":
        return h - 1 - r, c
    if corner == "br":
        return h - 1 - r, w - 1 - c
    raise AssertionError(f"unknown corner {corner}")


def _describe_component(sub: np.ndarray) -> tuple[str, int, int, int]:
    grey_color = _find_grey_color(sub)
    grey_coords = np.argwhere(sub == grey_color)
    sig = _diag_signature(grey_coords)
    assert sig is not None, "crease must be a straight diagonal chain"
    h, w = sub.shape
    touches_top = bool(np.any(grey_coords[:, 0] == 0))
    touches_bottom = bool(np.any(grey_coords[:, 0] == h - 1))
    touches_left = bool(np.any(grey_coords[:, 1] == 0))
    touches_right = bool(np.any(grey_coords[:, 1] == w - 1))

    if sig[0] == "sum":
        if touches_top and touches_left:
            corner = "tl"
        else:
            assert touches_bottom and touches_right, "sum-diagonal crease must touch TL or BR edges"
            corner = "br"
    else:
        if touches_top and touches_right:
            corner = "tr"
        else:
            assert touches_bottom and touches_left, "diff-diagonal crease must touch TR or BL edges"
            corner = "bl"

    corner_rc = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }[corner]
    cover_color = int(sub[corner_rc])
    assert cover_color != grey_color, "crease cannot occupy the triangle corner"

    other_colors = sorted(set(map(int, np.unique(sub))) - {grey_color, cover_color})
    assert len(other_colors) == 1, "component must contain exactly base, cover, and crease colors"
    base_color = int(other_colors[0])

    grey_norm = np.asarray([_to_normalized(int(r), int(c), sub.shape, corner) for r, c in grey_coords], dtype=int)
    assert np.all(grey_norm[:, 0] + grey_norm[:, 1] == grey_norm[0, 0] + grey_norm[0, 1])
    depth = int(grey_norm[0, 0] + grey_norm[0, 1])
    return corner, depth, base_color, cover_color


def _unfold_component(sub: np.ndarray) -> np.ndarray:
    corner, depth, base_color, cover_color = _describe_component(sub)
    h, w = sub.shape
    out = sub.copy()
    cover_mask = sub == cover_color
    grey_mask = sub == _find_grey_color(sub)

    cover_norm = np.asarray(
        [_to_normalized(int(r), int(c), sub.shape, corner) for r, c in np.argwhere(cover_mask)],
        dtype=int,
    )
    assert cover_norm.size > 0, "triangle must contain visible cover cells"
    assert np.all(cover_norm[:, 0] + cover_norm[:, 1] < depth), "cover must lie on one side of the crease"

    out[cover_mask | grey_mask] = base_color
    for nr, nc in cover_norm:
        rr = int(depth - nc)
        cc = int(depth - nr)
        tr, tc = _from_normalized(rr, cc, (h, w), corner)
        if 0 <= tr < h and 0 <= tc < w:
            out[tr, tc] = cover_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold each rectangular envelope by flipping its triangular cover across the marked diagonal crease.

    Input structure:
    - The background is the most common color.
    - Several separated non-background objects are solid axis-aligned rectangles.
    - Inside each rectangle, one base color fills the whole area, one second color covers a corner as a right triangle, and a grey diagonal chain marks that triangle's hypotenuse.

    Rule:
    - Process each non-background rectangle independently.
    - Use the grey diagonal's orientation and the edges it touches to identify which corner holds the folded triangle.
    - Remove the original triangle and the grey crease back to the base color, then reflect every cover-color pixel across the grey diagonal onto the opposite side of that crease; reflected pixels outside the rectangle are clipped.

    Key insight:
    - The grey cells are not part of the decoration; they are the fold axis, so the visible cover triangle must be mirrored across that diagonal rather than translated or recolored in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    comps = _find_components(g != bg)
    assert comps, "expected at least one envelope rectangle"

    out = g.copy()
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert coords.shape[0] == sub.shape[0] * sub.shape[1], "each envelope must occupy a full rectangle"
        out[r0 : r1 + 1, c0 : c1 + 1] = _unfold_component(sub)
    return out


def _make_envelope(height: int, width: int, depth: int, corner: str, base: int, cover: int) -> np.ndarray:
    rect = np.full((height, width), base, dtype=int)
    for r in range(height):
        for c in range(width):
            nr, nc = _to_normalized(r, c, rect.shape, corner)
            if nr + nc < depth:
                rect[r, c] = cover
            elif nr + nc == depth:
                rect[r, c] = GREY
    return rect


def _expanded_bbox(top: int, left: int, height: int, width: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h - 1, top + height)
    c1 = min(w - 1, left + width)
    return r0, c0, r1, c1


def _bg_is_unique_majority(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[bg]) > int(np.max(np.delete(counts, bg)))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(CANVAS_MIN, CANVAS_MAX)
    w = rng.randint(CANVAS_MIN, CANVAS_MAX)
    bg = rng.choice([c for c in ARC_COLORS if c != GREY])
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    max_by_area = max(1, min(MAX_OBJECTS, (h * w) // 110 + 1))
    target_objects = rng.randint(1, max_by_area)
    corners = list(CORNERS)
    rng.shuffle(corners)
    while len(corners) < target_objects:
        corners.append(rng.choice(CORNERS))

    palette = [c for c in ARC_COLORS if c not in (bg, GREY)]
    placed = 0
    total_area = 0
    for i in range(target_objects):
        corner = corners[i]
        success = False
        for _ in range(60):
            rect_h = rng.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, h - 2))
            rect_w = rng.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, w - 2))
            if total_area + rect_h * rect_w > int(h * w * 0.42):
                continue
            max_depth = min(rect_h, rect_w) - 2
            if max_depth < 2:
                continue
            depth = rng.randint(2, max_depth)
            base, cover = rng.sample(palette, 2)
            top = rng.randint(0, h - rect_h)
            left = rng.randint(0, w - rect_w)
            er0, ec0, er1, ec1 = _expanded_bbox(top, left, rect_h, rect_w, (h, w))
            if np.any(occupied[er0 : er1 + 1, ec0 : ec1 + 1]):
                continue

            patch = _make_envelope(rect_h, rect_w, depth, corner, base, cover)
            grid[top : top + rect_h, left : left + rect_w] = patch
            occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
            total_area += rect_h * rect_w
            placed += 1
            success = True
            break
        if not success:
            continue

    if placed == 0:
        return None
    if not _bg_is_unique_majority(grid, bg):
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid envelope grid after {MAX_GENERATION_ATTEMPTS} attempts")
