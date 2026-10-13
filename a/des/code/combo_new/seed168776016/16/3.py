from __future__ import annotations

from collections import deque
import math

import numpy as np


BG = 0
PREFERRED_OUTPUT_COLOR = 8
INPUT_COLOR_POOL = tuple(color for color in range(1, 10) if color != PREFERRED_OUTPUT_COLOR)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator settings.
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
INNER_DIM_MIN = 2
INNER_DIM_MAX = 6
PLACEMENT_PADDING = 1
PLACEMENT_BORDER = 1
GRID_SIDE_MAX = 30
GENERATION_ATTEMPTS = 400


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _shape_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _component_info(grid: np.ndarray, color: int, coords: np.ndarray) -> dict:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(coords.shape[0])
    mask = _shape_mask(coords, box)

    solid = bool(area == h * w and np.all(mask))

    frame = False
    if h >= 3 and w >= 3:
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        frame = bool(np.array_equal(mask, border) and np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG))

    return {
        "color": int(color),
        "coords": coords,
        "bbox": box,
        "h": h,
        "w": w,
        "area": area,
        "is_solid": solid,
        "is_frame": frame,
    }


def _choose_output_color(grid: np.ndarray) -> int:
    used = set(map(int, np.asarray(grid, dtype=int).ravel()))
    if PREFERRED_OUTPUT_COLOR not in used:
        return int(PREFERRED_OUTPUT_COLOR)
    for color in range(1, 10):
        if color not in used:
            return int(color)
    raise AssertionError("expected at least one unused non-background color")


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    pairs: list[dict] = []

    for color in sorted(set(map(int, g.ravel())) - {BG}):
        components = _connected_components(g == color)
        infos = [_component_info(g, color, coords) for coords in components]
        frames = [info for info in infos if info["is_frame"]]
        fillers = [info for info in infos if info["is_solid"]]

        assert len(infos) == 2, "each active color must form exactly one frame and one filler"
        assert len(frames) == 1, "expected one hollow frame per color"
        assert len(fillers) == 1, "expected one solid filler per color"

        frame = frames[0]
        filler = fillers[0]
        inner_h = int(frame["h"] - 2)
        inner_w = int(frame["w"] - 2)
        filler_dims = (int(filler["h"]), int(filler["w"]))

        assert filler["area"] == inner_h * inner_w, "filler area must match frame cavity area"
        assert filler_dims in {(inner_h, inner_w), (inner_w, inner_h)}, "filler must fit directly or after rotation"

        pairs.append(
            {
                "color": int(color),
                "frame_bbox": frame["bbox"],
                "filler_coords": filler["coords"],
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )

    assert pairs, "expected at least one frame-filler pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each color-matched solid rectangle into its hollow frame and recolor all inserted rectangles to one shared new color.

    Input structure:
    - The background is black.
    - Several distinct colors each appear in exactly two disconnected rectangular objects: a one-pixel-thick hollow frame and a solid rectangle.
    - Every frame encloses a black rectangular cavity, and the same-color solid rectangle has the same area as that cavity.
    - The solid rectangle's dimensions either already match the cavity or are the cavity dimensions swapped, corresponding to a 90-degree rotation.

    Rule:
    - Pair each solid rectangle with the hollow frame of the same color.
    - Remove the solid rectangle from its original location.
    - Fill the paired frame cavity completely; when the rectangle dimensions are swapped, interpret this as rotating it before insertion.
    - Keep frame borders unchanged and recolor every inserted rectangle to the same fixed new color that is absent from the input.

    Key insight:
    - Color gives the pairing, while the cavity dimensions explain whether the movable rectangle is already aligned or conceptually needs a quarter-turn before docking.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    output_color = _choose_output_color(g)

    for pair in _extract_pairs(g):
        filler = pair["filler_coords"]
        out[filler[:, 0], filler[:, 1]] = BG

        r0, c0, r1, c1 = pair["frame_bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = output_color

    return out


def _frame_patch(color: int, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _dimension_choices() -> list[tuple[int, int]]:
    return [
        (h, w)
        for h in range(INNER_DIM_MIN, INNER_DIM_MAX + 1)
        for w in range(INNER_DIM_MIN, INNER_DIM_MAX + 1)
        if h != w
    ]


def _sample_specs(rng: np.random.Generator) -> list[dict]:
    n_pairs = int(rng.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX + 1))
    colors = [int(x) for x in rng.choice(np.array(INPUT_COLOR_POOL, dtype=int), size=n_pairs, replace=False)]

    rotation_flags = [False, True]
    while len(rotation_flags) < n_pairs:
        rotation_flags.append(bool(rng.integers(0, 2)))
    rng.shuffle(rotation_flags)

    dims_pool = _dimension_choices()
    specs: list[dict] = []
    for color, needs_rotation in zip(colors, rotation_flags):
        inner_h, inner_w = dims_pool[int(rng.integers(0, len(dims_pool)))]
        filler_h, filler_w = (inner_w, inner_h) if needs_rotation else (inner_h, inner_w)
        specs.append(
            {
                "color": int(color),
                "needs_rotation": bool(needs_rotation),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "frame_patch": _frame_patch(int(color), int(inner_h), int(inner_w)),
                "filler_patch": np.full((filler_h, filler_w), int(color), dtype=int),
            }
        )
    return specs


def _candidate_positions(occupied: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, occupied.shape)
    ph, pw = map(int, patch.shape)
    positions: list[tuple[int, int]] = []

    start_r = PLACEMENT_BORDER
    end_r = h - ph - PLACEMENT_BORDER
    start_c = PLACEMENT_BORDER
    end_c = w - pw - PLACEMENT_BORDER
    if end_r < start_r or end_c < start_c:
        return positions

    for r0 in range(start_r, end_r + 1):
        for c0 in range(start_c, end_c + 1):
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            rr1 = min(h, r0 + ph + PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            cc1 = min(w, c0 + pw + PLACEMENT_PADDING)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((int(r0), int(c0)))

    return positions


def _choose_canvas_size(rng: np.random.Generator, patches: list[np.ndarray]) -> tuple[int, int] | None:
    heights = [int(p.shape[0]) for p in patches]
    widths = [int(p.shape[1]) for p in patches]
    max_h = max(heights)
    max_w = max(widths)
    padded_area = sum((h + 2 * PLACEMENT_PADDING) * (w + 2 * PLACEMENT_PADDING) for h, w in zip(heights, widths))

    min_side = max(max_h + 2 * PLACEMENT_BORDER, max_w + 2 * PLACEMENT_BORDER, 10)
    base_side = int(math.ceil(math.sqrt(float(padded_area))))
    min_side = max(min_side, base_side - 2)
    if min_side > GRID_SIDE_MAX:
        return None

    for _ in range(60):
        grid_h = int(rng.integers(min_side, GRID_SIDE_MAX + 1))
        grid_w = int(rng.integers(min_side, GRID_SIDE_MAX + 1))
        if grid_h * grid_w < int(math.ceil(1.2 * padded_area)):
            continue
        return grid_h, grid_w
    return None


def _place_specs(rng: np.random.Generator, specs: list[dict]) -> tuple[int, int, dict] | None:
    items: list[dict] = []
    for idx, spec in enumerate(specs):
        items.append({"pair": idx, "kind": "frame", "patch": spec["frame_patch"]})
        items.append({"pair": idx, "kind": "filler", "patch": spec["filler_patch"]})

    rng.shuffle(items)
    items.sort(key=lambda item: int(item["patch"].shape[0] * item["patch"].shape[1]), reverse=True)

    patches = [item["patch"] for item in items]
    for _ in range(80):
        canvas_size = _choose_canvas_size(rng, patches)
        if canvas_size is None:
            return None
        grid_h, grid_w = canvas_size
        occupied = np.zeros((grid_h, grid_w), dtype=bool)
        placements: dict[tuple[int, str], tuple[int, int]] = {}
        ok = True

        for item in items:
            candidates = _candidate_positions(occupied, item["patch"])
            if not candidates:
                ok = False
                break
            pos = candidates[int(rng.integers(0, len(candidates)))]
            r0, c0 = pos
            ph, pw = map(int, item["patch"].shape)
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placements[(int(item["pair"]), str(item["kind"]))] = pos

        if ok:
            return grid_h, grid_w, placements

    return None


def _crop_with_random_margin(
    rng: np.random.Generator, inp: np.ndarray, expected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_room = int(r0)
    bottom_room = int(inp.shape[0] - 1 - r1)
    left_room = int(c0)
    right_room = int(inp.shape[1] - 1 - c1)

    top = int(rng.integers(0, min(4, top_room) + 1))
    bottom = int(rng.integers(0, min(4, bottom_room) + 1))
    left = int(rng.integers(0, min(4, left_room) + 1))
    right = int(rng.integers(0, min(4, right_room) + 1))

    rs = int(r0 - top)
    re = int(r1 + bottom + 1)
    cs = int(c0 - left)
    ce = int(c1 + right + 1)
    return inp[rs:re, cs:ce].copy(), expected[rs:re, cs:ce].copy()


def _render_case(
    specs: list[dict], grid_h: int, grid_w: int, placements: dict[tuple[int, str], tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)

    for idx, spec in enumerate(specs):
        frame_r, frame_c = placements[(idx, "frame")]
        filler_r, filler_c = placements[(idx, "filler")]
        _paste_nonzero(inp, spec["frame_patch"], frame_r, frame_c)
        _paste_nonzero(inp, spec["filler_patch"], filler_r, filler_c)

    expected[:] = inp
    for idx, spec in enumerate(specs):
        frame_r, frame_c = placements[(idx, "frame")]
        filler_r, filler_c = placements[(idx, "filler")]
        filler_h, filler_w = map(int, spec["filler_patch"].shape)
        expected[filler_r : filler_r + filler_h, filler_c : filler_c + filler_w] = BG
        expected[
            frame_r + 1 : frame_r + 1 + int(spec["inner_h"]),
            frame_c + 1 : frame_c + 1 + int(spec["inner_w"]),
        ] = PREFERRED_OUTPUT_COLOR

    return inp, expected


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _sample_specs(rng)
    placed = _place_specs(rng, specs)
    if placed is None:
        return None

    grid_h, grid_w, placements = placed
    inp, expected = _render_case(specs, grid_h, grid_w, placements)
    inp, expected = _crop_with_random_margin(rng, inp, expected)

    if PREFERRED_OUTPUT_COLOR in set(map(int, inp.ravel())):
        return None
    if inp.ndim != 2 or max(inp.shape) > GRID_SIDE_MAX:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
