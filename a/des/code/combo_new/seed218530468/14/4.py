from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
GEN_RETRIES = 300
PLACE_RETRIES = 150
GRID_SIDE_RANGE = (18, 31)
OBJECT_COUNT_RANGE = (2, 5)


def _shape(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=int)


UP_TEMPLATES = (
    _shape(
        (
            "#...",
            "#.##",
            "###.",
        )
    ),
    _shape(
        (
            "#..#",
            "#.##",
            "###.",
        )
    ),
    _shape(
        (
            "...#",
            "##.#",
            ".###",
        )
    ),
    _shape(
        (
            "#..#",
            "##.#",
            ".###",
        )
    ),
    _shape(
        (
            "#....",
            "#.###",
            "###..",
        )
    ),
)


DIR_TO_ROT = {"up": 0, "left": 1, "down": 2, "right": 3}
DIRECTIONS = ("up", "down", "left", "right")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _side_notch_candidates(mask: np.ndarray) -> list[str]:
    h, w = mask.shape
    bg = ~mask
    found: list[str] = []

    if h >= 2 and w >= 4:
        for c in range(1, w - 2):
            if bg[0, c] and bg[0, c + 1] and bg[1, c]:
                found.append("up")
            if bg[0, c] and bg[0, c + 1] and bg[1, c + 1]:
                found.append("up")
            if bg[h - 1, c] and bg[h - 1, c + 1] and bg[h - 2, c]:
                found.append("down")
            if bg[h - 1, c] and bg[h - 1, c + 1] and bg[h - 2, c + 1]:
                found.append("down")

    if h >= 4 and w >= 2:
        for r in range(1, h - 2):
            if bg[r, 0] and bg[r + 1, 0] and bg[r, 1]:
                found.append("left")
            if bg[r, 0] and bg[r + 1, 0] and bg[r + 1, 1]:
                found.append("left")
            if bg[r, w - 1] and bg[r + 1, w - 1] and bg[r, w - 2]:
                found.append("right")
            if bg[r, w - 1] and bg[r + 1, w - 1] and bg[r + 1, w - 2]:
                found.append("right")

    return found


def _direction_from_patch(patch: np.ndarray) -> str:
    candidates = _side_notch_candidates(patch.astype(bool))
    if not candidates:
        raise ValueError("shape is missing a side-opening L-notch")
    directions = set(candidates)
    if len(directions) != 1:
        raise ValueError(f"ambiguous notch directions: {sorted(directions)}")
    return candidates[0]


def _step(direction: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if direction == "up":
        return (-h, 0)
    if direction == "down":
        return (h, 0)
    if direction == "left":
        return (0, -w)
    if direction == "right":
        return (0, w)
    raise ValueError(direction)


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _component_patch(grid: np.ndarray, comp: dict[str, object]) -> np.ndarray:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
    return (grid[r0 : r1 + 1, c0 : c1 + 1] == color).astype(int)


def _template_cells(patch: np.ndarray, origin: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = origin
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(patch == 1):
        cells.add((r0 + int(r), c0 + int(c)))
    return cells


def _stamped_cells(
    patch: np.ndarray,
    origin: tuple[int, int],
    direction: str,
    canvas_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = canvas_shape
    base = _template_cells(patch, origin)
    count = int(patch.sum())
    step_r, step_c = _step(direction, patch.shape)
    cells: set[tuple[int, int]] = set()
    for k in range(count + 1):
        dr = k * step_r
        dc = k * step_c
        for r, c in base:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
    return cells


def _expand(cells: set[tuple[int, int]], canvas_shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = canvas_shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly stamp each colored shape in the direction encoded by its embedded notch.

    Input structure:
    - The background is black.
    - The grid contains several separated monochrome 4-connected shapes, each with its own color.
    - Inside each shape's tight bounding box there is exactly one 3-cell background L-notch whose side opening points up, down, left, or right.

    Rule:
    - For each colored shape, detect the opening direction of its internal L-notch.
    - Let N be the number of colored cells in that shape.
    - Keep the original shape and add N translated copies along the notch direction.
    - Each translation step is exactly the shape bbox height for vertical motion or bbox width for horizontal motion, and out-of-bounds copy pixels are clipped.

    Key insight:
    - The missing 3-cell notch is not damage; it is a built-in arrow that tells both which sprite to repeat and which way to stamp it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for comp in _find_components(g):
        color = int(comp["color"])
        coords = comp["coords"]  # type: ignore[index]
        patch = _component_patch(g, comp)
        direction = _direction_from_patch(patch)
        copies = int(coords.shape[0])
        step_r, step_c = _step(direction, patch.shape)

        for k in range(1, copies + 1):
            dr = k * step_r
            dc = k * step_c
            for r, c in coords:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color

    return out


def _sample_patch(rng: np.random.Generator, direction: str) -> np.ndarray:
    base = UP_TEMPLATES[int(rng.integers(len(UP_TEMPLATES)))]
    return np.rot90(base, DIR_TO_ROT[direction]).astype(int)


def _placement_ranges(
    direction: str,
    patch_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = patch_shape
    H, W = canvas_shape
    r_min = 0
    r_max = H - h
    c_min = 0
    c_max = W - w

    if direction == "up":
        r_min = h
    elif direction == "down":
        r_max = H - (2 * h)
    elif direction == "left":
        c_min = w
    elif direction == "right":
        c_max = W - (2 * w)

    return r_min, r_max, c_min, c_max


def _full_visible_stamps(
    direction: str,
    patch_shape: tuple[int, int],
    origin: tuple[int, int],
    canvas_shape: tuple[int, int],
    copies: int,
) -> int:
    h, w = patch_shape
    H, W = canvas_shape
    step_r, step_c = _step(direction, patch_shape)
    count = 0
    for k in range(copies + 1):
        r0 = origin[0] + k * step_r
        c0 = origin[1] + k * step_c
        if 0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W:
            count += 1
    return count


def _render_input(canvas_shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(canvas_shape, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        cells = spec["base_cells"]  # type: ignore[index]
        _paint_cells(grid, cells, color)
    return grid


def _render_output(canvas_shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = _render_input(canvas_shape, specs)
    for spec in specs:
        color = int(spec["color"])
        cells = spec["all_cells"]  # type: ignore[index]
        _paint_cells(out, cells, color)
    return out


def _place_spec(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    patch: np.ndarray,
    color: int,
    direction: str,
    reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    r_min, r_max, c_min, c_max = _placement_ranges(direction, patch.shape, canvas_shape)
    if r_min > r_max or c_min > c_max:
        return None

    copies = int(patch.sum())
    for _ in range(PLACE_RETRIES):
        r0 = int(rng.integers(r_min, r_max + 1))
        c0 = int(rng.integers(c_min, c_max + 1))
        origin = (r0, c0)
        all_cells = _stamped_cells(patch, origin, direction, canvas_shape)
        if len(all_cells) <= copies:
            continue
        if _full_visible_stamps(direction, patch.shape, origin, canvas_shape, copies) < 2:
            continue
        if any(cell in reserved for cell in all_cells):
            continue
        return {
            "color": int(color),
            "direction": direction,
            "patch": patch,
            "origin": origin,
            "base_cells": _template_cells(patch, origin),
            "all_cells": all_cells,
        }
    return None


def _validate_specs(canvas_shape: tuple[int, int], specs: list[dict[str, object]]) -> bool:
    inp = _render_input(canvas_shape, specs)
    expected = _render_output(canvas_shape, specs)
    comps = _find_components(inp)
    if len(comps) != len(specs):
        return False

    actual: dict[int, tuple[str, int]] = {}
    for comp in comps:
        color = int(comp["color"])
        patch = _component_patch(inp, comp)
        actual[color] = (_direction_from_patch(patch), int(comp["coords"].shape[0]))  # type: ignore[index]

    for spec in specs:
        color = int(spec["color"])
        direction = str(spec["direction"])
        count = int(spec["patch"].sum())  # type: ignore[index]
        if actual.get(color) != (direction, count):
            return False

    return np.array_equal(main(inp), expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*GRID_SIDE_RANGE))
    W = int(rng.integers(*GRID_SIDE_RANGE))
    n_objects = int(rng.integers(*OBJECT_COUNT_RANGE))
    colors = rng.choice(PALETTE, size=n_objects, replace=False)
    directions = list(rng.permutation(np.array(DIRECTIONS, dtype=object))[:n_objects])

    specs_seed: list[dict[str, object]] = []
    for color, direction in zip(colors.tolist(), directions):
        patch = _sample_patch(rng, str(direction))
        footprint = int(patch.sum()) * (patch.shape[0] + patch.shape[1])
        specs_seed.append(
            {
                "color": int(color),
                "direction": str(direction),
                "patch": patch,
                "priority": footprint,
            }
        )

    specs_seed.sort(key=lambda item: int(item["priority"]), reverse=True)

    placed: list[dict[str, object]] = []
    reserved: set[tuple[int, int]] = set()
    for seed in specs_seed:
        spec = _place_spec(
            rng=rng,
            canvas_shape=(H, W),
            patch=seed["patch"],  # type: ignore[arg-type]
            color=int(seed["color"]),
            direction=str(seed["direction"]),
            reserved=reserved,
        )
        if spec is None:
            return None
        placed.append(spec)
        reserved |= _expand(spec["all_cells"], (H, W))  # type: ignore[arg-type]

    if not _validate_specs((H, W), placed):
        return None

    return _render_input((H, W), placed)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-direction input")
