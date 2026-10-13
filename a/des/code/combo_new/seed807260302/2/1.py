from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DEPTHS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 500


def _components(g: np.ndarray) -> list[dict]:
    """Return same-color 4-connected foreground components."""
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color == BG or seen[sr, sc]:
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    expected_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected_area:
        return False

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    actual = {tuple(map(int, xy)) for xy in comp["coords"]}
    return actual == expected


def _inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 < ir0 and ir1 < or1 and oc0 < ic0 and ic1 < oc1


def _shape_key(comp: dict) -> tuple[bytes, tuple[int, int]]:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask.tobytes(), tuple(map(int, mask.shape))


def _legend(g: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    """Read bottom-row 1x2 swatches from left to right."""
    bottom = g[-1]
    colors: list[int] = []
    cells: set[tuple[int, int]] = set()
    c = 0
    width = int(g.shape[1])
    while c < width:
        if int(bottom[c]) == BG:
            c += 1
            continue
        color = int(bottom[c])
        start = c
        while c < width and int(bottom[c]) == color:
            c += 1
        if c - start == 2:
            colors.append(color)
            cells.add((int(g.shape[0] - 1), start))
            cells.add((int(g.shape[0] - 1), start + 1))
    assert len(colors) >= 3, "expected at least three 1x2 legend swatches"
    return colors, cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor free objects by matching their shapes to framed core objects and using the nesting-depth legend.

    Input structure:
    - Black background with monochrome 4-connected objects.
    - Some objects are enclosed by one-pixel rectangular frame rings, with up to three nested frame layers.
    - The bottom row contains left-to-right 1x2 color swatches for nesting depths 1, 2, and 3.

    Rule:
    - For each innermost non-frame object inside frames, count how many rectangular frame rings contain it.
    - Build a mapping from that object's exact translated shape to the legend color for the counted depth.
    - Recolor matching same-shape objects that are outside all frames, without rotating or flipping the shape.
    - Leave frames, framed core objects, unmatched objects, and the bottom legend unchanged.

    Key insight:
    - Frame nesting depth is a visible index into the bottom color legend; the resulting shape-to-color map is then applied to matching free objects.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend_colors, legend_cells = _legend(g)

    comps = _components(g)
    frames = [comp for comp in comps if _is_frame(comp)]
    nonframes = [
        comp
        for comp in comps
        if not _is_frame(comp) and not set(map(tuple, comp["coords"])).issubset(legend_cells)
    ]

    shape_to_color: dict[tuple[bytes, tuple[int, int]], int] = {}
    comp_depths: list[tuple[dict, int]] = []
    for comp in nonframes:
        depth = sum(1 for frame in frames if _inside(comp["bbox"], frame["bbox"]))
        comp_depths.append((comp, int(depth)))
        if depth > 0:
            assert depth <= len(legend_colors), "missing legend color for a nesting depth"
            key = _shape_key(comp)
            color = int(legend_colors[depth - 1])
            if key in shape_to_color:
                assert shape_to_color[key] == color, "shape maps to two different depths"
            shape_to_color[key] = color

    for comp, depth in comp_depths:
        if depth != 0:
            continue
        key = _shape_key(comp)
        if key not in shape_to_color:
            continue
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(shape_to_color[key])

    return out


# --------------- generator ---------------


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                cell = (int(rr), int(cc))
                if cell not in seen:
                    seen.add(cell)
                    q.append(cell)
    return len(seen) == int(mask.sum())


def _mask_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    return np.asarray(mask, dtype=bool).tobytes(), tuple(map(int, mask.shape))


def _grow_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        max_area = min(h * w, 8)
        min_area = min(max(3, max(h, w)), max_area)
        if min_area > max_area:
            continue
        area = int(rng.integers(min_area, max_area + 1))

        cells = {(int(rng.integers(0, h)), int(rng.integers(0, w)))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                        frontier.append(nb)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != area:
            continue

        rs, cs = zip(*cells)
        r0, c0 = min(rs), min(cs)
        r1, c1 = max(rs), max(cs)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        if mask.shape == (1, 2) or mask.shape == (2, 1):
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if _is_connected(mask):
            return mask
    raise RuntimeError("failed to build a connected mask")


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _framed_patch(mask: np.ndarray, depth: int, frame_color: int, core_color: int) -> np.ndarray:
    mh, mw = map(int, mask.shape)
    patch = np.zeros((mh + 4 * depth, mw + 4 * depth), dtype=int)
    for layer in range(1, depth + 1):
        offset = 2 * (depth - layer)
        h = mh + 4 * layer
        w = mw + 4 * layer
        r0 = c0 = offset
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        patch[r0, c0 : c1 + 1] = int(frame_color)
        patch[r1, c0 : c1 + 1] = int(frame_color)
        patch[r0 : r1 + 1, c0] = int(frame_color)
        patch[r0 : r1 + 1, c1] = int(frame_color)
    core_r = core_c = 2 * depth
    patch[core_r : core_r + mh, core_c : core_c + mw][mask] = int(core_color)
    return patch


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, usable_rows: int) -> bool:
    h, w = map(int, patch.shape)
    if r0 < 1 or c0 < 1 or r0 + h + 1 > usable_rows or c0 + w + 1 > grid.shape[1]:
        return False
    padded = grid[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1]
    return not bool(np.any(padded != BG))


def _place_random(
    rng: np.random.Generator,
    grid: np.ndarray,
    patch: np.ndarray,
    usable_rows: int,
) -> tuple[int, int] | None:
    h, w = map(int, patch.shape)
    positions = [
        (r, c)
        for r in range(1, usable_rows - h)
        for c in range(1, grid.shape[1] - w)
        if _can_place(grid, patch, r, c, usable_rows)
    ]
    if not positions:
        return None
    r0, c0 = positions[int(rng.integers(0, len(positions)))]
    mask = patch != BG
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    return int(r0), int(c0)


def _draw_legend(grid: np.ndarray, legend_colors: list[int], rng: np.random.Generator) -> None:
    width = int(grid.shape[1])
    total = 2 * len(legend_colors) + len(legend_colors) - 1
    start = int(rng.integers(1, width - total))
    c = start
    for color in legend_colors:
        grid[-1, c : c + 2] = int(color)
        c += 3


def _build_expected(inp: np.ndarray) -> np.ndarray | None:
    try:
        return main(inp)
    except Exception:
        return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(24, 31))
    width = int(rng.integers(24, 31))
    usable_rows = height - 2
    grid = np.zeros((height, width), dtype=int)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    legend_colors = colors[:3]
    _draw_legend(grid, legend_colors, rng)

    shape_keys: set[tuple[bytes, tuple[int, int]]] = set()
    records: list[dict] = []
    for depth in DEPTHS:
        for _ in range(200):
            mask = _grow_mask(rng)
            key = _mask_key(mask)
            if key not in shape_keys:
                shape_keys.add(key)
                break
        else:
            return None

        frame_color = int(rng.choice(colors))
        core_choices = [c for c in colors if c != frame_color]
        core_color = int(rng.choice(core_choices))
        free_choices = [c for c in colors if c != legend_colors[depth - 1]]
        free_color = int(rng.choice(free_choices))
        records.append(
            {
                "depth": int(depth),
                "mask": mask,
                "frame": _framed_patch(mask, int(depth), frame_color, core_color),
                "free": _patch_from_mask(mask, free_color),
            }
        )

    distractor_count = int(rng.integers(0, 3))
    distractors: list[np.ndarray] = []
    for _ in range(distractor_count):
        for _attempt in range(100):
            mask = _grow_mask(rng)
            if _mask_key(mask) not in shape_keys:
                shape_keys.add(_mask_key(mask))
                color = int(rng.choice(colors))
                distractors.append(_patch_from_mask(mask, color))
                break

    patches: list[np.ndarray] = []
    for record in records:
        patches.append(record["frame"])
        patches.append(record["free"])
    patches.extend(distractors)
    rng.shuffle(patches)
    patches.sort(key=lambda p: int(p.shape[0] * p.shape[1]), reverse=True)

    for patch in patches:
        if _place_random(rng, grid, patch, usable_rows) is None:
            return None

    expected = _build_expected(grid)
    if expected is None:
        return None
    if expected.shape != grid.shape or max(expected.shape) > 30:
        return None
    if np.array_equal(expected, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame legend input")
