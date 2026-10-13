from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SIGNAL_COLORS = {RED, GREEN}
RESERVED_COLORS = {RED, GREEN, YELLOW, GREY}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def monochrome_components(grid: np.ndarray, active_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not bool(active_mask[r, c]) or bool(seen[r, c]):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(active_mask[nr, nc]) and not bool(seen[nr, nc]):
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": coords_arr})
    return comps


def _detect_axis(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, int, int]] = []
    for r in range(h):
        color = int(g[r, 0])
        if color != bg and np.all(g[r, :] == color):
            candidates.append(("h", r, color))
    for c in range(w):
        color = int(g[0, c])
        if color != bg and np.all(g[:, c] == color):
            candidates.append(("v", c, color))

    assert len(candidates) == 1, "expected a unique full mirror axis"
    return candidates[0]


def _extract_side(grid: np.ndarray, axis_kind: str, axis_idx: int, side_name: str) -> np.ndarray:
    if axis_kind == "v":
        return grid[:, :axis_idx] if side_name == "left" else grid[:, axis_idx + 1 :]
    assert axis_kind == "h"
    return grid[:axis_idx, :] if side_name == "top" else grid[axis_idx + 1 :, :]


def _globalize_coords(
    coords: np.ndarray, axis_kind: str, axis_idx: int, side_name: str
) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    if axis_kind == "v":
        if side_name == "right":
            out[:, 1] += axis_idx + 1
    else:
        assert axis_kind == "h"
        if side_name == "bottom":
            out[:, 0] += axis_idx + 1
    return out


def _side_profile(side: np.ndarray, bg: int) -> dict:
    comps = monochrome_components(side, side != bg)
    sizes = [int(comp["coords"].shape[0]) for comp in comps]
    colors = [int(comp["color"]) for comp in comps]
    return {
        "components": comps,
        "nonempty": bool(comps),
        "has_shape": any(size > 1 for size in sizes),
        "all_singletons": all(size == 1 for size in sizes),
        "all_signal_colors": all(color in SIGNAL_COLORS for color in colors),
    }


def _locate_sides(grid: np.ndarray, axis_kind: str, axis_idx: int, bg: int) -> tuple[str, str]:
    if axis_kind == "v":
        first, second = "left", "right"
    else:
        first, second = "top", "bottom"

    first_profile = _side_profile(_extract_side(grid, axis_kind, axis_idx, first), bg)
    second_profile = _side_profile(_extract_side(grid, axis_kind, axis_idx, second), bg)

    first_is_template = first_profile["has_shape"]
    second_is_signal = second_profile["nonempty"] and second_profile["all_singletons"] and second_profile["all_signal_colors"]
    if first_is_template and second_is_signal:
        return first, second

    second_is_template = second_profile["has_shape"]
    first_is_signal = first_profile["nonempty"] and first_profile["all_singletons"] and first_profile["all_signal_colors"]
    if second_is_template and first_is_signal:
        return second, first

    raise AssertionError("expected one shape side and one singleton signal side")


def _mirror_coords(coords: np.ndarray, axis_kind: str, axis_idx: int) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    if axis_kind == "v":
        out[:, 1] = 2 * axis_idx - out[:, 1]
    else:
        assert axis_kind == "h"
        out[:, 0] = 2 * axis_idx - out[:, 0]
    return out


def _coords_on_side(coords: np.ndarray, axis_kind: str, axis_idx: int, side_name: str) -> bool:
    if axis_kind == "v":
        cols = coords[:, 1]
        if side_name == "left":
            return bool(np.all(cols < axis_idx))
        return bool(np.all(cols > axis_idx))
    rows = coords[:, 0]
    if side_name == "top":
        return bool(np.all(rows < axis_idx))
    return bool(np.all(rows > axis_idx))


def _clear_side(grid: np.ndarray, axis_kind: str, axis_idx: int, side_name: str, fill: int) -> None:
    if axis_kind == "v":
        if side_name == "left":
            grid[:, :axis_idx] = fill
        else:
            grid[:, axis_idx + 1 :] = fill
        return
    assert axis_kind == "h"
    if side_name == "top":
        grid[:axis_idx, :] = fill
    else:
        grid[axis_idx + 1 :, :] = fill


def _final_color(base_color: int, hit_colors: set[int]) -> int:
    if RED in hit_colors and GREEN in hit_colors:
        return YELLOW
    if RED in hit_colors:
        return RED
    if GREEN in hit_colors:
        return GREEN
    return base_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the template-side shapes across the grey axis, then recolor each mirrored
    shape based on which red/green signal pixels land inside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    axis_kind, axis_idx, axis_color = _detect_axis(g, bg)
    assert axis_color == GREY, "mirror axis must be grey"

    template_side, signal_side = _locate_sides(g, axis_kind, axis_idx, bg)
    template = _extract_side(g, axis_kind, axis_idx, template_side)
    comps = monochrome_components(template, template != bg)
    assert comps, "template side must contain shapes"

    out = g.copy()
    _clear_side(out, axis_kind, axis_idx, signal_side, bg)

    h, w = g.shape
    for comp in comps:
        global_coords = _globalize_coords(comp["coords"], axis_kind, axis_idx, template_side)
        mirrored = _mirror_coords(global_coords, axis_kind, axis_idx)
        assert mirrored.size > 0
        assert np.all((0 <= mirrored[:, 0]) & (mirrored[:, 0] < h))
        assert np.all((0 <= mirrored[:, 1]) & (mirrored[:, 1] < w))
        assert _coords_on_side(mirrored, axis_kind, axis_idx, signal_side)

        hit_colors = {
            int(g[r, c])
            for r, c in mirrored
            if int(g[r, c]) in SIGNAL_COLORS
        }
        color = _final_color(int(comp["color"]), hit_colors)
        out[mirrored[:, 0], mirrored[:, 1]] = color

    return out


BASE_MASKS = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
]

COLOR_POOL = [0, 1, 6, 7, 8, 9]
GEN_RETRIES = 400
PADDING = 1


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, ...], ...]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        for variant in (rot, np.fliplr(rot)):
            key = tuple(tuple(int(v) for v in row) for row in variant.astype(int))
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
    return out


TRANSFORMS = [_unique_transforms(mask) for mask in BASE_MASKS]


def _sample_statuses(rng: random.Random) -> list[int]:
    num_shapes = rng.choice([3, 4])
    if num_shapes == 4:
        statuses = [0, 1, 2, 3]
    else:
        omitted = rng.randrange(4)
        statuses = [status for status in range(4) if status != omitted]
    rng.shuffle(statuses)
    return statuses


def _sample_layout(
    rng: random.Random,
) -> tuple[int, int, str, int, str, tuple[int, int, int, int]]:
    axis_kind = rng.choice(["v", "h"])
    if axis_kind == "v":
        h = rng.randint(11, 16)
        half_w = rng.randint(6, 9)
        w = 2 * half_w + 1
        axis_idx = half_w
        template_side = rng.choice(["left", "right"])
        bbox = (0, 0, h - 1, axis_idx - 1) if template_side == "left" else (0, axis_idx + 1, h - 1, w - 1)
        return h, w, axis_kind, axis_idx, template_side, bbox

    w = rng.randint(11, 16)
    half_h = rng.randint(6, 9)
    h = 2 * half_h + 1
    axis_idx = half_h
    template_side = rng.choice(["top", "bottom"])
    bbox = (0, 0, axis_idx - 1, w - 1) if template_side == "top" else (axis_idx + 1, 0, h - 1, w - 1)
    return h, w, axis_kind, axis_idx, template_side, bbox


def _place_shape(
    blocked: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    mh, mw = mask.shape
    origins = [
        (r, c)
        for r in range(r0, r1 - mh + 2)
        for c in range(c0, c1 - mw + 2)
    ]
    rng.shuffle(origins)
    for r, c in origins:
        rr0 = max(0, r - PADDING)
        cc0 = max(0, c - PADDING)
        rr1 = min(blocked.shape[0], r + mh + PADDING)
        cc1 = min(blocked.shape[1], c + mw + PADDING)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        blocked[rr0:rr1, cc0:cc1] = True
        return r, c
    return None


def _mask_coords(mask: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    cells = np.argwhere(mask)
    cells = cells.astype(int)
    cells[:, 0] += origin[0]
    cells[:, 1] += origin[1]
    return cells


def _signal_side(template_side: str) -> str:
    return {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }[template_side]


def _shape_signal_requirements(status: int) -> list[int]:
    if status == 0:
        return []
    if status == 1:
        return [RED]
    if status == 2:
        return [GREEN]
    assert status == 3
    return [RED, GREEN]


def _non_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1


def _pick_signal_cells(
    mirrored_coords: np.ndarray,
    colors: list[int],
    existing: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int]] | None:
    coords = [tuple(map(int, rc)) for rc in mirrored_coords]
    rng.shuffle(coords)
    if not colors:
        return []

    if len(colors) == 1:
        for cell in coords:
            if all(_non_adjacent(cell, other) for other in existing):
                return [(cell[0], cell[1], colors[0])]
        return None

    assert len(colors) == 2
    pairs = [(a, b) for a in coords for b in coords if a != b]
    rng.shuffle(pairs)
    for first, second in pairs:
        if not _non_adjacent(first, second):
            continue
        if not all(_non_adjacent(first, other) and _non_adjacent(second, other) for other in existing):
            continue
        return [
            (first[0], first[1], colors[0]),
            (second[0], second[1], colors[1]),
        ]
    return None


def _build_expected(
    inp: np.ndarray,
    shapes: list[dict],
    axis_kind: str,
    axis_idx: int,
    signal_side: str,
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    _clear_side(out, axis_kind, axis_idx, signal_side, bg)
    for shape in shapes:
        out[shape["mirrored"][:, 0], shape["mirrored"][:, 1]] = int(shape["final_color"])
    return out


def _validate_signal_side(grid: np.ndarray, axis_kind: str, axis_idx: int, signal_side: str, bg: int) -> bool:
    side = _extract_side(grid, axis_kind, axis_idx, signal_side)
    comps = monochrome_components(side, side != bg)
    return bool(comps) and all(int(comp["coords"].shape[0]) == 1 for comp in comps) and all(
        int(comp["color"]) in SIGNAL_COLORS for comp in comps
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w, axis_kind, axis_idx, template_side, template_bbox = _sample_layout(rng)
    signal_side = _signal_side(template_side)

    bg = rng.choice(COLOR_POOL)
    template_colors = [color for color in COLOR_POOL if color != bg]
    statuses = _sample_statuses(rng)
    shape_indices = rng.sample(range(len(BASE_MASKS)), k=len(statuses))
    rng.shuffle(template_colors)

    grid = np.full((h, w), bg, dtype=int)
    if axis_kind == "v":
        grid[:, axis_idx] = GREY
    else:
        grid[axis_idx, :] = GREY

    blocked = np.zeros((h, w), dtype=bool)
    if axis_kind == "v":
        blocked[:, axis_idx] = True
    else:
        blocked[axis_idx, :] = True

    shapes: list[dict] = []
    for color, status, shape_idx in zip(template_colors, statuses, shape_indices):
        mask = rng.choice(TRANSFORMS[shape_idx])
        origin = _place_shape(blocked, template_bbox, mask, rng)
        if origin is None:
            return None
        coords = _mask_coords(mask, origin)
        grid[coords[:, 0], coords[:, 1]] = color
        mirrored = _mirror_coords(coords, axis_kind, axis_idx)
        if not _coords_on_side(mirrored, axis_kind, axis_idx, signal_side):
            return None
        shapes.append(
            {
                "color": int(color),
                "status": int(status),
                "coords": coords,
                "mirrored": mirrored,
                "final_color": _final_color(int(color), set(_shape_signal_requirements(int(status)))),
            }
        )

    existing_signals: set[tuple[int, int]] = set()
    for shape in shapes:
        signal_cells = _pick_signal_cells(
            shape["mirrored"],
            _shape_signal_requirements(int(shape["status"])),
            existing_signals,
            rng,
        )
        if signal_cells is None:
            return None
        for r, c, color in signal_cells:
            if (r, c) in existing_signals:
                return None
            grid[r, c] = color
            existing_signals.add((r, c))

    if not _validate_signal_side(grid, axis_kind, axis_idx, signal_side, bg):
        return None

    expected = _build_expected(grid, shapes, axis_kind, axis_idx, signal_side, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
