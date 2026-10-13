from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
NOISE = 7
RECT_COLORS = (1, 3, 4, 6)
SIDE_STEP = 2
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _row_first_color(grid: np.ndarray, row: int, colors: set[int]) -> int:
    for value in map(int, grid[row]):
        if value in colors:
            return value
    for color in RECT_COLORS:
        if color in colors:
            return color
    raise AssertionError("expected at least one tied rectangle color")


def _repair_noise(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for coords in _component_coords(g == NOISE):
        if coords.shape[0] > 3:
            continue
        counts: dict[int, int] = {}
        total = 0
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    color = int(g[nr, nc])
                    if color in RECT_COLORS:
                        counts[color] = counts.get(color, 0) + 1
                        total += 1
        if total >= 2:
            best = max(counts.values())
            tied = {color for color, count in counts.items() if count == best}
            if len(tied) == 1:
                fill_color = next(iter(tied))
            else:
                rows = sorted({int(r) for r, _c in coords})
                fill_color = next((color for row in rows for color in [_row_first_color(g, row, tied)] if color in tied), None)
                if fill_color is None:
                    fill_color = _row_first_color(g, int(coords[:, 0].min()), tied)
            out[coords[:, 0], coords[:, 1]] = fill_color
        else:
            out[coords[:, 0], coords[:, 1]] = BG
    return out


def _draw_centered_border(out: np.ndarray, size: int, color: int) -> None:
    n = out.shape[0]
    inset = (n - size) // 2
    lo = inset
    hi = lo + size - 1
    out[lo, lo : hi + 1] = color
    out[hi, lo : hi + 1] = color
    out[lo : hi + 1, lo] = color
    out[lo : hi + 1, hi] = color


def _build_output(layers: list[tuple[int, int]]) -> np.ndarray:
    n = max(side for side, _color in layers)
    out = np.full((n, n), BG, dtype=int)
    for side, color in sorted(layers, key=lambda item: -item[0]):
        _draw_centered_border(out, int(side), int(color))

    inner_side, inner_color = min(layers, key=lambda item: item[0])
    if inner_side >= 3:
        inset = (n - inner_side) // 2
        out[inset + 1 : inset + inner_side - 1, inset + 1 : inset + inner_side - 1] = inner_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair orange noise on separated rectangles, then reorder the restored sizes into centered concentric square frames.

    Input structure:
    - A black background contains 3-4 solid rectangles in distinct colors chosen from blue, green, yellow, and magenta.
    - Orange pixels appear in 4-connected components of area at most 3, either covering rectangle edge cells or sitting just outside rectangles.
    - Rectangle groups are spatially separated; after valid noise repair each color still forms a single rectangle-like bbox.

    Rule:
    - Find each orange 4-connected component of size at most 3 and count all rectangle-colored 4-neighbor contacts around that whole component.
    - If a component has at least two such contacts, recolor the entire component to the majority touching rectangle color; ties use the first tied rectangle color encountered on the component's rows, otherwise erase the component to black.
    - After repairing all noise, compute for each rectangle color the longer side of its bbox, sort colors by that side descending, and draw centered square borders of those sizes into a square output whose side equals the largest repaired side.
    - After all borders are drawn, fill the innermost square's interior with the innermost color.

    Key insight:
    - Orange edge occlusions can hide an entire extreme row or column of a thin rectangle, so the visible bbox before repair can give the wrong ring size and even the wrong layer order.
    """
    repaired = _repair_noise(np.asarray(grid, dtype=int))

    layers: list[tuple[int, int, int, int]] = []
    for color in RECT_COLORS:
        coords = np.argwhere(repaired == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        side = max(r1 - r0 + 1, c1 - c0 + 1)
        layers.append((side, color, r0, c0))

    assert layers, "expected at least one rectangle color"
    ordered = [(side, color) for side, color, _r0, _c0 in sorted(layers, key=lambda item: (-item[0], item[2], item[3]))]
    return _build_output(ordered)


@dataclass(frozen=True)
class _ObjectSpec:
    color: int
    side: int
    patch: np.ndarray
    box_h: int
    box_w: int
    hidden_edges: int
    has_delete_cluster: bool


def _make_object(color: int, side: int, orientation: str, hidden_edges: int, force_delete: bool) -> _ObjectSpec:
    rect_h = side if orientation == "vertical" else 3
    rect_w = 3 if orientation == "vertical" else side

    delete_side = None
    if force_delete or (hidden_edges > 0 and random.random() < 0.45):
        if orientation == "vertical":
            delete_side = random.choice(["left", "right"])
        else:
            delete_side = random.choice(["top", "bottom"])

    box_h = rect_h + int(delete_side in {"top", "bottom"})
    box_w = rect_w + int(delete_side in {"left", "right"})
    rect_top = 1 if delete_side == "top" else 0
    rect_left = 1 if delete_side == "left" else 0

    patch = np.full((box_h, box_w), BG, dtype=int)
    patch[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = color

    if orientation == "vertical":
        edge_rows = [rect_top, rect_top + rect_h - 1]
        if hidden_edges == 1:
            edge_rows = [random.choice(edge_rows)]
        for rr in edge_rows[:hidden_edges]:
            patch[rr, rect_left : rect_left + rect_w] = NOISE
    else:
        edge_cols = [rect_left, rect_left + rect_w - 1]
        if hidden_edges == 1:
            edge_cols = [random.choice(edge_cols)]
        for cc in edge_cols[:hidden_edges]:
            patch[rect_top : rect_top + rect_h, cc] = NOISE

    if delete_side is not None:
        if orientation == "vertical":
            length = random.randint(1, min(3, rect_h - 2 if rect_h >= 3 else 1))
            start_min = rect_top + 1
            start_max = rect_top + rect_h - length - 1
            start = random.randint(start_min, start_max)
            cc = rect_left - 1 if delete_side == "left" else rect_left + rect_w
            patch[start : start + length, cc] = NOISE
        else:
            length = random.randint(1, min(3, rect_w - 2 if rect_w >= 3 else 1))
            start_min = rect_left + 1
            start_max = rect_left + rect_w - length - 1
            start = random.randint(start_min, start_max)
            rr = rect_top - 1 if delete_side == "top" else rect_top + rect_h
            patch[rr, start : start + length] = NOISE

    return _ObjectSpec(
        color=color,
        side=side,
        patch=patch,
        box_h=box_h,
        box_w=box_w,
        hidden_edges=hidden_edges,
        has_delete_cluster=delete_side is not None,
    )


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    target = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    target[mask] = patch[mask]


def _choose_sides(count: int) -> list[int]:
    if count == 3:
        max_side = random.choice([7, 8, 9, 10, 11])
    else:
        max_side = random.choice([9, 10, 11])
    return [max_side - SIDE_STEP * idx for idx in range(count)]


def _visible_side(grid: np.ndarray, color: int) -> int:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return max(r1 - r0 + 1, c1 - c0 + 1)


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([3, 4])
    sides = _choose_sides(count)
    colors = random.sample(list(RECT_COLORS), count)

    hidden_counts = [2] + [random.choice([0, 1]) for _ in range(count - 1)]
    if all(value > 0 for value in hidden_counts):
        hidden_counts[random.randint(1, count - 1)] = 0

    orientations = [random.choice(["vertical", "horizontal"]) for _ in range(count)]
    specs = [
        _make_object(
            color=colors[idx],
            side=sides[idx],
            orientation=orientations[idx],
            hidden_edges=min(hidden_counts[idx], 2 if sides[idx] >= 5 else 1),
            force_delete=hidden_counts[idx] == 0,
        )
        for idx in range(count)
    ]

    if not any(spec.has_delete_cluster for spec in specs):
        force_idx = max(range(1, count), key=lambda idx: specs[idx].side)
        specs[force_idx] = _make_object(
            color=colors[force_idx],
            side=sides[force_idx],
            orientation=orientations[force_idx],
            hidden_edges=min(hidden_counts[force_idx], 2 if sides[force_idx] >= 5 else 1),
            force_delete=True,
        )

    h = random.randint(max(20, sides[0] + 9), 26)
    w = random.randint(max(20, sides[0] + 9), 26)
    grid = np.full((h, w), BG, dtype=int)

    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(count), key=lambda idx: specs[idx].box_h * specs[idx].box_w, reverse=True)
    positions: dict[int, tuple[int, int]] = {}

    for idx in order:
        spec = specs[idx]
        candidates = [
            (r, c)
            for r in range(h - spec.box_h + 1)
            for c in range(w - spec.box_w + 1)
        ]
        random.shuffle(candidates)
        chosen = None
        for top, left in candidates:
            box = (top, left, top + spec.box_h - 1, left + spec.box_w - 1)
            if any(_boxes_too_close(box, other) for other in placed):
                continue
            chosen = (top, left)
            placed.append(box)
            positions[idx] = chosen
            break
        if chosen is None:
            return None

    for idx, spec in enumerate(specs):
        top, left = positions[idx]
        _paste_patch(grid, spec.patch, top, left)

    repaired = _repair_noise(grid)
    repaired_layers: list[tuple[int, int]] = []
    for color, side in zip(colors, sides):
        coords = np.argwhere(repaired == color)
        if coords.size == 0:
            return None
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        repaired_side = max(r1 - r0 + 1, c1 - c0 + 1)
        if repaired_side != side:
            return None
        repaired_layers.append((side, color))

    if not any(_visible_side(grid, color) != side for color, side in zip(colors, sides)):
        return None

    if sum(spec.hidden_edges == 2 for spec in specs) == 0:
        return None

    orange_components = _component_coords(grid == NOISE)
    if any(comp.shape[0] > 3 for comp in orange_components):
        return None

    for color in colors:
        if not any(
            grid[r, c] == NOISE
            for r, c in np.argwhere(grid == NOISE)
            if any(
                0 <= nr < h and 0 <= nc < w and repaired[nr, nc] == color
                for nr, nc in _neighbors4(int(r), int(c))
            )
        ):
            return None

    expected = _build_output(sorted(repaired_layers, key=lambda item: -item[0]))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
