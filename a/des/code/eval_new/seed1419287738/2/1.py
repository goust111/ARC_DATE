from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
KINDS = ("H", "V", "B")
GEN_RETRIES = 300
PLACE_RETRIES = 200
PATTERN_RETRIES = 200
OBJECT_RETRIES = 120
PALETTE = tuple(color for color in range(10) if color not in (BG_COLOR, FRAME_COLOR))


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _classify_symmetry(mask: np.ndarray) -> str:
    cropped = _crop_true(np.asarray(mask, dtype=bool))
    hsym = np.array_equal(cropped, cropped[::-1, :])
    vsym = np.array_equal(cropped, cropped[:, ::-1])
    if hsym and vsym:
        return "B"
    if hsym:
        return "H"
    if vsym:
        return "V"
    raise AssertionError("seed pattern does not match any required symmetry class")


def _is_rectangular_border(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expect = np.zeros((h, w), dtype=bool)
    expect[0, :] = True
    expect[-1, :] = True
    expect[:, 0] = True
    expect[:, -1] = True
    return np.array_equal(patch, expect)


def _box_has_bg_moat(grid: np.ndarray, bg: int, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0], r0 + h + 1)
    cc1 = min(grid.shape[1], c0 + w + 1)
    region = grid[rr0:rr1, cc0:cc1]
    keep = np.ones(region.shape, dtype=bool)
    keep[r0 - rr0 : r0 - rr0 + h, c0 - cc0 : c0 - cc0 + w] = False
    return bool(np.all(region[keep] == bg))


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], dict[str, tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    for c0 in (0, w - 2):
        if c0 < 0 or c0 + 1 >= w:
            continue
        for r0 in range(h - 2):
            sub = grid[r0 : r0 + 3, c0 : c0 + 2]
            if sub.shape != (3, 2) or np.any(sub == bg):
                continue
            if np.any(sub[:, 0] == sub[:, 1]):
                continue
            if not _box_has_bg_moat(grid, bg, r0, c0, 3, 2):
                continue
            candidates.append((r0, c0, sub.copy()))
    assert len(candidates) == 1, f"expected one edge legend, found {len(candidates)}"
    r0, c0, sub = candidates[0]
    mapping = {
        "H": (int(sub[0, 0]), int(sub[0, 1])),
        "V": (int(sub[1, 0]), int(sub[1, 1])),
        "B": (int(sub[2, 0]), int(sub[2, 1])),
    }
    assert len({pair[0] for pair in mapping.values()}) == 3
    return (r0, c0, r0 + 2, c0 + 1), mapping


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    best_color = None
    best_frames: list[dict[str, object]] = []
    ambiguous = False
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        frames: list[dict[str, object]] = []
        for comp in _components(grid == color):
            box = _bbox(comp)
            if not _is_rectangular_border(comp, box):
                continue
            r0, c0, r1, c1 = box
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = set(int(v) for v in np.unique(inner) if int(v) != bg)
            if len(inner_colors) != 1:
                continue
            pattern_color = next(iter(inner_colors))
            pattern_mask = inner == pattern_color
            if not pattern_mask.any():
                continue
            frames.append(
                {
                    "bbox": box,
                    "pattern_color": int(pattern_color),
                    "pattern_mask": pattern_mask.copy(),
                }
            )
        if len(frames) > len(best_frames):
            best_color = int(color)
            best_frames = frames
            ambiguous = False
        elif frames and len(frames) == len(best_frames):
            ambiguous = True
    assert best_frames and best_color is not None and not ambiguous, "frame color must be uniquely identifiable"
    return best_color, best_frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor outside objects according to the symmetry class of same-colored seed patterns inside gray frames.

    Input structure:
    - The background is black.
    - Several gray rectangular frames appear in the grid; each frame contains one monochrome seed pattern.
    - Scattered outside the frames are monochrome 4-connected objects, and an edge legend lists three source->target color pairs in a vertical 3x2 block.

    Rule:
    - Read the legend from top to bottom as the mappings for horizontal-only symmetry, vertical-only symmetry, and dual-axis symmetry.
    - For each frame, crop its seed pattern to its tight bbox and classify whether it is symmetric only across the horizontal axis, only across the vertical axis, or across both axes.
    - Use the corresponding legend row, verify that its source color matches the seed color, and recolor every outside object of that source color to the row's target color.
    - Keep the frames, the seed patterns, and the legend unchanged.

    Key insight:
    - The framed pattern is not copied anywhere; it acts only as a symmetry classifier that selects which legend row controls the recoloring of matching outside objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    legend_box, legend_map = _find_legend(g, bg)
    _, frames = _find_frames(g, bg)

    protected = np.zeros_like(g, dtype=bool)
    lr0, lc0, lr1, lc1 = legend_box
    protected[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    recolor_plan: dict[int, int] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        protected[r0 : r1 + 1, c0 : c1 + 1] = True

        pattern_color = int(frame["pattern_color"])  # type: ignore[index]
        pattern_mask = np.asarray(frame["pattern_mask"], dtype=bool)  # type: ignore[index]
        kind = _classify_symmetry(pattern_mask)
        source, target = legend_map[kind]
        assert pattern_color == source, "legend source color must match the seed color for its symmetry row"
        if source in recolor_plan:
            assert recolor_plan[source] == target
        recolor_plan[source] = target

    out = g.copy()
    for source, target in recolor_plan.items():
        repaint = (g == source) & ~protected
        assert repaint.any(), f"expected at least one outside object of color {source}"
        out[repaint] = target

    return out


def _normalize_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _rotate_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return _normalize_cells((c, -r) for r, c in cells)


def _reflect_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return _normalize_cells((r, -c) for r, c in cells)


def _random_orientation(cells: Iterable[tuple[int, int]], rng: random.Random) -> tuple[tuple[int, int], ...]:
    pts = tuple(cells)
    if rng.random() < 0.5:
        pts = _reflect_cells(pts)
    for _ in range(rng.randrange(4)):
        pts = _rotate_cells(pts)
    return _normalize_cells(pts)


def _orbits_for_kind(h: int, w: int, kind: str) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if kind == "H":
                orbit = tuple(sorted({(r, c), (h - 1 - r, c)}))
            elif kind == "V":
                orbit = tuple(sorted({(r, c), (r, w - 1 - c)}))
            else:
                orbit = tuple(sorted({(r, c), (h - 1 - r, c), (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))
            if orbit not in seen:
                seen.add(orbit)
                out.append(orbit)
    return out


def _make_seed_pattern(kind: str, rng: random.Random) -> np.ndarray:
    for _ in range(PATTERN_RETRIES):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        mask = np.zeros((h, w), dtype=bool)
        for orbit in _orbits_for_kind(h, w, kind):
            if rng.random() < 0.48:
                for r, c in orbit:
                    mask[r, c] = True
        if not mask.any():
            continue
        mask = _crop_true(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        area = int(mask.sum())
        if area < 4 or area >= mask.size:
            continue
        if len(_components(mask)) != 1:
            continue
        if _classify_symmetry(mask) != kind:
            continue
        return mask.astype(bool)
    raise RuntimeError(f"failed to build a {kind} seed pattern")


def _make_object_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    for _ in range(OBJECT_RETRIES):
        target_area = rng.randint(2, 6)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(sorted(frontier)))
        norm = _random_orientation(_normalize_cells(cells), rng)
        rows = [r for r, _ in norm]
        cols = [c for _, c in norm]
        if max(rows) + 1 <= 4 and max(cols) + 1 <= 4:
            return norm
    raise RuntimeError("failed to build a small outside object")


def _placeable(blocked: np.ndarray, cells: Iterable[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: Iterable[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        rr0 = max(0, r - pad)
        cc0 = max(0, c - pad)
        rr1 = min(h, r + pad + 1)
        cc1 = min(w, c + pad + 1)
        blocked[rr0:rr1, cc0:cc1] = True


def _rect_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    chosen = rng.sample(PALETTE, 6)
    sources = {"H": chosen[0], "V": chosen[1], "B": chosen[2]}
    targets = {"H": chosen[3], "V": chosen[4], "B": chosen[5]}
    leftovers = [color for color in PALETTE if color not in chosen]

    legend_side = rng.choice(("left", "right"))
    legend_r0 = rng.randint(1, h - 4)
    legend_c0 = 0 if legend_side == "left" else w - 2
    legend_rows = [sources["H"], sources["V"], sources["B"]]
    legend_targets = [targets["H"], targets["V"], targets["B"]]
    for idx, (src, tgt) in enumerate(zip(legend_rows, legend_targets)):
        grid[legend_r0 + idx, legend_c0] = src
        grid[legend_r0 + idx, legend_c0 + 1] = tgt
    _mark_blocked(blocked, _rect_cells(legend_r0, legend_c0, 3, 2), pad=1)

    protected = np.zeros_like(grid, dtype=bool)
    protected[legend_r0 : legend_r0 + 3, legend_c0 : legend_c0 + 2] = True

    frame_specs: list[tuple[str, int, tuple[int, int, int, int]]] = []
    for kind in rng.sample(list(KINDS), len(KINDS)):
        pattern = _make_seed_pattern(kind, rng)
        ph, pw = pattern.shape
        oh, ow = ph + 2, pw + 2
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = rng.randint(1, h - oh - 1)
            c0 = rng.randint(1, w - ow - 1)
            box_cells = _rect_cells(r0, c0, oh, ow)
            if not _placeable(blocked, box_cells):
                continue
            grid[r0, c0 : c0 + ow] = FRAME_COLOR
            grid[r0 + oh - 1, c0 : c0 + ow] = FRAME_COLOR
            grid[r0 : r0 + oh, c0] = FRAME_COLOR
            grid[r0 : r0 + oh, c0 + ow - 1] = FRAME_COLOR
            inner = grid[r0 + 1 : r0 + 1 + ph, c0 + 1 : c0 + 1 + pw]
            inner[pattern] = sources[kind]
            _mark_blocked(blocked, box_cells, pad=1)
            protected[r0 : r0 + oh, c0 : c0 + ow] = True
            frame_specs.append((kind, sources[kind], (r0, c0, r0 + oh - 1, c0 + ow - 1)))
            placed = True
            break
        if not placed:
            return None

    outside_counts: dict[int, int] = {sources[kind]: 0 for kind in KINDS}
    for kind in KINDS:
        count = rng.randint(1, 3)
        color = sources[kind]
        for _ in range(count):
            shape = _make_object_shape(rng)
            sh = max(r for r, _ in shape) + 1
            sw = max(c for _, c in shape) + 1
            placed = False
            for _ in range(PLACE_RETRIES):
                r0 = rng.randint(1, h - sh - 1)
                c0 = rng.randint(1, w - sw - 1)
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if not _placeable(blocked, cells):
                    continue
                for r, c in cells:
                    grid[r, c] = color
                _mark_blocked(blocked, cells, pad=1)
                outside_counts[color] += 1
                placed = True
                break
            if not placed:
                return None

    distractor_colors = rng.sample(leftovers, rng.randint(0, min(2, len(leftovers))))
    for color in distractor_colors:
        for _ in range(rng.randint(1, 2)):
            shape = _make_object_shape(rng)
            sh = max(r for r, _ in shape) + 1
            sw = max(c for _, c in shape) + 1
            placed = False
            for _ in range(PLACE_RETRIES):
                r0 = rng.randint(1, h - sh - 1)
                c0 = rng.randint(1, w - sw - 1)
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if not _placeable(blocked, cells):
                    continue
                for r, c in cells:
                    grid[r, c] = color
                _mark_blocked(blocked, cells, pad=1)
                placed = True
                break
            if not placed:
                return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert np.array_equal(out[protected], grid[protected])
    for kind in KINDS:
        source = sources[kind]
        target = targets[kind]
        repaint = (grid == source) & ~protected
        assert repaint.any() and outside_counts[source] > 0
        assert np.all(out[repaint] == target)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
