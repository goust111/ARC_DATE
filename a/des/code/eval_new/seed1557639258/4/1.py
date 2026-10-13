from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _components(np.asarray(grid == int(color), dtype=bool))


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(int(r0), int(c)) for c in range(int(c0), int(c1) + 1)]
    cells.extend((int(r1), int(c)) for c in range(int(c0), int(c1) + 1))
    cells.extend((int(r), int(c0)) for r in range(int(r0) + 1, int(r1)))
    cells.extend((int(r), int(c1)) for r in range(int(r0) + 1, int(r1)))
    return cells


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict[str, object]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _color_components(g, int(color)):
            if not comp:
                continue
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
                continue

            boundary = _boundary_cells(int(r0), int(c0), int(r1), int(c1))
            if len(comp) != len(boundary) - 1:
                continue

            marker_cells: list[tuple[int, int, int]] = []
            valid = True
            for br, bc in boundary:
                value = int(g[br, bc])
                if value == int(color):
                    continue
                if value == int(bg):
                    valid = False
                    break
                marker_cells.append((int(br), int(bc), int(value)))
            if not valid or len(marker_cells) != 1:
                continue

            mr, mc, seed_color = marker_cells[0]
            side = None
            if mr == r0 and c0 < mc < c1:
                side = "top"
            elif mr == r1 and c0 < mc < c1:
                side = "bottom"
            elif mc == c0 and r0 < mr < r1:
                side = "left"
            elif mc == c1 and r0 < mr < r1:
                side = "right"
            if side is None:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            non_bg_mask = interior != int(bg)
            if not bool(non_bg_mask.any()):
                continue
            if bool(np.any(interior[non_bg_mask] != int(seed_color))):
                continue

            seed_comps = _components(interior == int(seed_color))
            if len(seed_comps) != 1:
                continue
            seed_rel = seed_comps[0]
            if not (1 <= len(seed_rel) <= 4):
                continue

            seed_abs = [(int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr, cc in seed_rel]
            seed_rows = [r for r, _c in seed_abs]
            seed_cols = [c for _r, c in seed_abs]
            if side in ("top", "bottom") and not (min(seed_cols) <= mc <= max(seed_cols)):
                continue
            if side in ("left", "right") and not (min(seed_rows) <= mr <= max(seed_rows)):
                continue

            candidates.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "frame_color": int(color),
                    "seed_color": int(seed_color),
                    "marker": (int(mr), int(mc)),
                    "side": side,
                    "seed": seed_abs,
                }
            )

    candidates.sort(key=lambda obj: tuple(int(v) for v in obj["bbox"]))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each seed shape out of its rectangular frame by opening a matching gap on the marked side.

    Input structure:
    - The background is a dominant solid color.
    - Several one-pixel-thick rectangular frames of one color sit on the grid.
    - Inside each frame there is one small connected seed shape of another color.
    - Exactly one border cell of each frame is replaced by a same-color marker on one side of the frame.

    Rule:
    - Read the marker side to determine the push direction for that frame.
    - Remove the interior seed and carve a border gap on that same side whose span matches the seed's projection perpendicular to the push direction.
    - Translate the whole seed straight outward until it is fully outside the frame and flush against the opened side.

    Key insight:
    - The marker chooses which wall opens, while the seed's own projected span fixes how wide the opening must be for a straight push through the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()
    h, w = g.shape

    for frame in _find_frames(g, bg):
        seed = [(int(r), int(c)) for r, c in frame["seed"]]
        side = str(frame["side"])
        color = int(frame["seed_color"])
        r0, c0, r1, c1 = (int(v) for v in frame["bbox"])
        seed_rows = [r for r, _c in seed]
        seed_cols = [c for _r, c in seed]

        for r, c in seed:
            out[r, c] = int(bg)

        dr = 0
        dc = 0
        if side == "top":
            out[r0, min(seed_cols) : max(seed_cols) + 1] = int(bg)
            dr = int(r0 - 1 - max(seed_rows))
        elif side == "bottom":
            out[r1, min(seed_cols) : max(seed_cols) + 1] = int(bg)
            dr = int(r1 + 1 - min(seed_rows))
        elif side == "left":
            out[min(seed_rows) : max(seed_rows) + 1, c0] = int(bg)
            dc = int(c0 - 1 - max(seed_cols))
        elif side == "right":
            out[min(seed_rows) : max(seed_rows) + 1, c1] = int(bg)
            dc = int(c1 + 1 - min(seed_cols))

        for r, c in seed:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(color)

    return out


BASE_SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
]


def _normalize_shape(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _c in cells]
    cols = [int(c) for _r, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((int(c), int(-r)) for r, c in cells))


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_shape(tuple((int(r), int(-c)) for r, c in cells))


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    cur = _normalize_shape(cells)
    for _ in range(4):
        variants.add(cur)
        variants.add(_flip_shape(cur))
        cur = _rotate_shape(cur)
    return sorted(variants)


def _sample_shape() -> tuple[tuple[int, int], ...]:
    base = random.choice(BASE_SHAPES)
    return random.choice(_shape_variants(base))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return int(max(rows) + 1), int(max(cols) + 1)


def _make_object(side: str, frame_color: int, seed_color: int) -> dict[str, object] | None:
    shape = _sample_shape()
    seed_h, seed_w = _shape_size(shape)

    min_frame_h = max(6, int(seed_h + 4))
    min_frame_w = max(6, int(seed_w + 4))
    max_frame_h = min(10, int(seed_h + 7))
    max_frame_w = min(10, int(seed_w + 7))
    if min_frame_h > max_frame_h or min_frame_w > max_frame_w:
        return None

    frame_h = random.randint(min_frame_h, max_frame_h)
    frame_w = random.randint(min_frame_w, max_frame_w)
    seed_top = random.randint(2, frame_h - seed_h - 2)
    seed_left = random.randint(2, frame_w - seed_w - 2)

    if side in ("top", "bottom"):
        marker = (0 if side == "top" else frame_h - 1, random.randint(seed_left, seed_left + seed_w - 1))
    else:
        marker = (random.randint(seed_top, seed_top + seed_h - 1), 0 if side == "left" else frame_w - 1)

    return {
        "side": side,
        "frame_color": int(frame_color),
        "seed_color": int(seed_color),
        "shape": shape,
        "frame_h": int(frame_h),
        "frame_w": int(frame_w),
        "seed_top": int(seed_top),
        "seed_left": int(seed_left),
        "marker_local": (int(marker[0]), int(marker[1])),
        "top_margin": int(seed_h if side == "top" else 0),
        "bottom_margin": int(seed_h if side == "bottom" else 0),
        "left_margin": int(seed_w if side == "left" else 0),
        "right_margin": int(seed_w if side == "right" else 0),
    }


def _seed_cells(spec: dict[str, object], *, origin: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    r0 = int(spec.get("r0", 0) if origin is None else origin[0])
    c0 = int(spec.get("c0", 0) if origin is None else origin[1])
    top = int(spec["seed_top"])
    left = int(spec["seed_left"])
    shape = tuple(spec["shape"])
    return [(int(r0 + top + dr), int(c0 + left + dc)) for dr, dc in shape]


def _reserve_bbox(spec: dict[str, object], r0: int, c0: int) -> tuple[int, int, int, int]:
    top_margin = int(spec["top_margin"])
    bottom_margin = int(spec["bottom_margin"])
    left_margin = int(spec["left_margin"])
    right_margin = int(spec["right_margin"])
    frame_h = int(spec["frame_h"])
    frame_w = int(spec["frame_w"])
    return (
        int(r0 - top_margin - 1),
        int(c0 - left_margin - 1),
        int(r0 + frame_h + bottom_margin),
        int(c0 + frame_w + right_margin),
    )


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return _ranges_overlap(int(a[0]), int(a[2]), int(b[0]), int(b[2])) and _ranges_overlap(
        int(a[1]), int(a[3]), int(b[1]), int(b[3])
    )


def _place_objects(specs: list[dict[str, object]], height: int, width: int) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    occupied: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda s: -(int(s["frame_h"]) * int(s["frame_w"]))):
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        top_margin = int(spec["top_margin"])
        bottom_margin = int(spec["bottom_margin"])
        left_margin = int(spec["left_margin"])
        right_margin = int(spec["right_margin"])

        r_min = int(top_margin + 1)
        r_max = int(height - frame_h - bottom_margin - 1)
        c_min = int(left_margin + 1)
        c_max = int(width - frame_w - right_margin - 1)
        if r_min > r_max or c_min > c_max:
            return None

        found = False
        for _ in range(250):
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            box = _reserve_bbox(spec, int(r0), int(c0))
            if any(_bbox_overlap(box, prev) for prev in occupied):
                continue
            placed_spec = dict(spec)
            placed_spec["r0"] = int(r0)
            placed_spec["c0"] = int(c0)
            occupied.append(box)
            placed.append(placed_spec)
            found = True
            break
        if not found:
            return None

    return placed


def _draw_input(height: int, width: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        frame_color = int(spec["frame_color"])
        seed_color = int(spec["seed_color"])
        marker_r, marker_c = (int(v) for v in spec["marker_local"])

        r1 = int(r0 + frame_h - 1)
        c1 = int(c0 + frame_w - 1)
        grid[r0, c0 : c1 + 1] = int(frame_color)
        grid[r1, c0 : c1 + 1] = int(frame_color)
        grid[r0 : r1 + 1, c0] = int(frame_color)
        grid[r0 : r1 + 1, c1] = int(frame_color)
        grid[r0 + marker_r, c0 + marker_c] = int(seed_color)

        for sr, sc in _seed_cells(spec):
            grid[sr, sc] = int(seed_color)
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for spec in specs:
        side = str(spec["side"])
        seed_color = int(spec["seed_color"])
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        r1 = int(r0 + frame_h - 1)
        c1 = int(c0 + frame_w - 1)
        seed = _seed_cells(spec)
        rows = [r for r, _c in seed]
        cols = [c for _r, c in seed]

        for r, c in seed:
            out[r, c] = 0

        dr = 0
        dc = 0
        if side == "top":
            out[r0, min(cols) : max(cols) + 1] = 0
            dr = int(r0 - 1 - max(rows))
        elif side == "bottom":
            out[r1, min(cols) : max(cols) + 1] = 0
            dr = int(r1 + 1 - min(rows))
        elif side == "left":
            out[min(rows) : max(rows) + 1, c0] = 0
            dc = int(c0 - 1 - max(cols))
        else:
            out[min(rows) : max(rows) + 1, c1] = 0
            dc = int(c1 + 1 - min(cols))

        for r, c in seed:
            out[r + dr, c + dc] = int(seed_color)
    return out


def _sample_canvas(specs: list[dict[str, object]]) -> tuple[int, int]:
    min_h = max(int(spec["frame_h"]) + int(spec["top_margin"]) + int(spec["bottom_margin"]) + 2 for spec in specs)
    min_w = max(int(spec["frame_w"]) + int(spec["left_margin"]) + int(spec["right_margin"]) + 2 for spec in specs)
    reserve_area = sum(
        (int(spec["frame_h"]) + int(spec["top_margin"]) + int(spec["bottom_margin"]) + 2)
        * (int(spec["frame_w"]) + int(spec["left_margin"]) + int(spec["right_margin"]) + 2)
        for spec in specs
    )

    for _ in range(80):
        height = random.randint(max(16, min_h), 30)
        width = random.randint(max(16, min_w), 30)
        if height * width >= int(reserve_area * 1.2):
            return int(height), int(width)
    return 30, 30


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice([2, 2, 3, 3, 4])
    sides = random.sample(SIDES, k=n_objects)
    colors = random.sample(list(range(1, 10)), 2 * n_objects)

    specs: list[dict[str, object]] = []
    for i, side in enumerate(sides):
        spec = _make_object(side, int(colors[2 * i]), int(colors[2 * i + 1]))
        if spec is None:
            return None
        specs.append(spec)

    height, width = _sample_canvas(specs)
    placed = _place_objects(specs, int(height), int(width))
    if placed is None:
        return None

    inp = _draw_input(int(height), int(width), placed)
    expected = _expected_output(inp, placed)

    frames = _find_frames(inp, 0)
    if len(frames) != n_objects:
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
