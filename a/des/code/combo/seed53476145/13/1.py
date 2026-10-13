from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400
GRID_SIDE_RANGE = (20, 28)
OBJECT_COUNT_CHOICES = (3, 4)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return comps


def _shape_library() -> list[np.ndarray]:
    shapes: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], np.ndarray] = {}
    for bits in range(1 << 9):
        area = int(bits.bit_count())
        if not 3 <= area <= 6:
            continue

        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue

        seen = {tuple(map(int, coords[0]))}
        stack = [tuple(map(int, coords[0]))]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < 3 and 0 <= nc < 3 and mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        if len(seen) != area:
            continue

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        key = (
            tuple(map(int, local.shape)),
            tuple(map(tuple, np.argwhere(local))),
        )
        shapes[key] = local.copy()

    return list(shapes.values())


SHAPE_LIBRARY = _shape_library()


def _sprite_local(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    local[rel[:, 0], rel[:, 1]] = int(comp["color"])
    return local


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each small monochrome sprite according to the count of same-color singleton markers
    aligned with its bounding-box rows and columns, then erase the markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    components = _find_components(g, bg=BG)
    sprites: list[dict] = []
    markers_by_color: dict[int, list[tuple[int, int]]] = {}

    for comp in components:
        area = int(comp["area"])
        color = int(comp["color"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers_by_color.setdefault(color, []).append((r, c))
            continue

        r0, c0, r1, c1 = comp["bbox"]
        sprite_h = int(r1 - r0 + 1)
        sprite_w = int(c1 - c0 + 1)
        assert 3 <= area <= 6, "sprite area must be 3..6"
        assert sprite_h <= 3 and sprite_w <= 3, "sprite bbox must fit in 3x3"
        sprites.append(comp)

    assert sprites, "expected at least one sprite"

    sprites_by_color: dict[int, list[dict]] = {}
    for sprite in sprites:
        sprites_by_color.setdefault(int(sprite["color"]), []).append(sprite)

    for color, markers in markers_by_color.items():
        assert color in sprites_by_color, "every marker color must have a matching sprite"
        for mr, mc in markers:
            attached = 0
            for sprite in sprites_by_color[color]:
                r0, c0, r1, c1 = sprite["bbox"]
                if r0 <= mr <= r1 or c0 <= mc <= c1:
                    attached += 1
            assert attached >= 1, "every marker must align with at least one same-color sprite"

    out = np.full_like(g, BG)

    for sprite in sprites:
        color = int(sprite["color"])
        r0, c0, _r1, _c1 = sprite["bbox"]
        local = _sprite_local(sprite)
        sprite_h, sprite_w = map(int, local.shape)

        horiz = 0
        vert = 0
        for mr, mc in markers_by_color.get(color, []):
            if r0 <= mr <= r0 + sprite_h - 1:
                horiz += 1
            if c0 <= mc <= c0 + sprite_w - 1:
                vert += 1

        kh = horiz + 1 if horiz else 1
        kv = vert + 1 if vert else 1
        expanded = np.repeat(np.repeat(local, kv, axis=0), kh, axis=1)
        mask = expanded != BG

        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(h, r0 + expanded.shape[0])
        cc1 = min(w, c0 + expanded.shape[1])
        if rr0 >= rr1 or cc0 >= cc1:
            continue

        submask = mask[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]
        region = out[rr0:rr1, cc0:cc1]
        region[submask] = color

    return out


def _case_list(n: int) -> list[str]:
    cases = ["none", "h", "v", "hv"]
    random.shuffle(cases)
    return cases[:n]


def _marker_offsets(
    sprite_h: int,
    sprite_w: int,
    h_rows: list[int],
    h_side: str | None,
    v_cols: list[int],
    v_side: str | None,
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    if h_side == "left":
        cols = [-2 - 2 * i for i in range(len(h_rows))]
    elif h_side == "right":
        cols = [sprite_w + 1 + 2 * i for i in range(len(h_rows))]
    else:
        cols = []
    for row, col in zip(h_rows, cols):
        coords.append((int(row), int(col)))

    if v_side == "up":
        rows = [-2 - 2 * i for i in range(len(v_cols))]
    elif v_side == "down":
        rows = [sprite_h + 1 + 2 * i for i in range(len(v_cols))]
    else:
        rows = []
    for row, col in zip(rows, v_cols):
        coords.append((int(row), int(col)))

    assert len(coords) == len(set(coords))
    return coords


def _separated(cells: set[tuple[int, int]], existing: set[tuple[int, int]], margin: int) -> bool:
    for r, c in cells:
        for er, ec in existing:
            if max(abs(r - er), abs(c - ec)) <= margin:
                return False
    return True


def _build_spec(color: int, case: str, clip_choice: str | None) -> dict:
    mask = random.choice(SHAPE_LIBRARY).copy()
    sprite_h, sprite_w = map(int, mask.shape)
    h_count = random.randint(1, 3) if case in {"h", "hv"} else 0
    v_count = random.randint(1, 3) if case in {"v", "hv"} else 0
    kh = h_count + 1 if h_count else 1
    kv = v_count + 1 if v_count else 1

    clip_x = clip_choice in {"x", "both"}
    clip_y = clip_choice in {"y", "both"}
    h_side = None
    v_side = None
    if h_count:
        h_side = "left" if clip_x else random.choice(["left", "right"])
    if v_count:
        v_side = "up" if clip_y else random.choice(["up", "down"])

    h_rows = [random.randrange(sprite_h) for _ in range(h_count)]
    v_cols = [random.randrange(sprite_w) for _ in range(v_count)]
    marker_cells = _marker_offsets(sprite_h, sprite_w, h_rows, h_side, v_cols, v_side)
    sprite_cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    expanded_mask = np.repeat(np.repeat(mask, kv, axis=0), kh, axis=1)
    expanded_cells = [tuple(map(int, cell)) for cell in np.argwhere(expanded_mask)]

    return {
        "color": int(color),
        "case": case,
        "mask": mask,
        "sprite_h": int(sprite_h),
        "sprite_w": int(sprite_w),
        "kh": int(kh),
        "kv": int(kv),
        "clip_x": bool(clip_x),
        "clip_y": bool(clip_y),
        "sprite_cells": sprite_cells,
        "marker_cells": marker_cells,
        "expanded_mask": expanded_mask,
        "expanded_cells": expanded_cells,
    }


def _choose_clip_map(cases: list[str]) -> dict[int, str | None]:
    clip_map = {i: None for i in range(len(cases))}
    stretchable = [i for i, case in enumerate(cases) if case != "none"]
    if not stretchable or random.random() >= 0.25:
        return clip_map

    idx = random.choice(stretchable)
    case = cases[idx]
    if case == "h":
        clip_map[idx] = "x"
    elif case == "v":
        clip_map[idx] = "y"
    else:
        clip_map[idx] = random.choice(["x", "x", "y", "y", "both"])
    return clip_map


def _anchor_candidates(
    spec: dict,
    height: int,
    width: int,
    input_used: set[tuple[int, int]],
    output_used: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    input_cells = spec["sprite_cells"] + spec["marker_cells"]
    min_r = min(r for r, _ in input_cells)
    max_r = max(r for r, _ in input_cells)
    min_c = min(c for _, c in input_cells)
    max_c = max(c for _, c in input_cells)

    exp_h, exp_w = map(int, spec["expanded_mask"].shape)
    sprite_h = int(spec["sprite_h"])
    sprite_w = int(spec["sprite_w"])
    candidates: list[tuple[int, int]] = []

    for top in range(-min_r, height - max_r):
        if spec["clip_y"]:
            if top + sprite_h > height or top + exp_h <= height:
                continue
        else:
            if top + exp_h > height:
                continue

        for left in range(-min_c, width - max_c):
            if spec["clip_x"]:
                if left + sprite_w > width or left + exp_w <= width:
                    continue
            else:
                if left + exp_w > width:
                    continue

            abs_input = {(top + r, left + c) for r, c in input_cells}
            if not _separated(abs_input, input_used, margin=1):
                continue

            abs_output = {
                (top + r, left + c)
                for r, c in spec["expanded_cells"]
                if 0 <= top + r < height and 0 <= left + c < width
            }
            if not abs_output:
                continue
            if not _separated(abs_output, output_used, margin=0):
                continue

            candidates.append((top, left))

    return candidates


def _render_input(height: int, width: int, placements: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for item in placements:
        top = int(item["top"])
        left = int(item["left"])
        color = int(item["spec"]["color"])
        for r, c in item["spec"]["sprite_cells"]:
            grid[top + r, left + c] = color
        for r, c in item["spec"]["marker_cells"]:
            grid[top + r, left + c] = color
    return grid


def _render_expected(height: int, width: int, placements: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for item in placements:
        top = int(item["top"])
        left = int(item["left"])
        color = int(item["spec"]["color"])
        for r, c in item["spec"]["expanded_cells"]:
            rr = top + r
            cc = left + c
            if 0 <= rr < height and 0 <= cc < width:
                grid[rr, cc] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    cases = _case_list(n_objects)
    colors = random.sample(PALETTE, n_objects)
    clip_map = _choose_clip_map(cases)

    specs = [
        _build_spec(color, case, clip_map[i])
        for i, (color, case) in enumerate(zip(colors, cases))
    ]
    specs.sort(
        key=lambda spec: (
            int(spec["clip_x"]) + int(spec["clip_y"]),
            len(spec["expanded_cells"]),
            len(spec["marker_cells"]),
        ),
        reverse=True,
    )

    placements: list[dict] = []
    input_used: set[tuple[int, int]] = set()
    output_used: set[tuple[int, int]] = set()

    for spec in specs:
        candidates = _anchor_candidates(spec, height, width, input_used, output_used)
        if not candidates:
            return None
        top, left = random.choice(candidates)

        abs_input = {(top + r, left + c) for r, c in spec["sprite_cells"] + spec["marker_cells"]}
        abs_output = {
            (top + r, left + c)
            for r, c in spec["expanded_cells"]
            if 0 <= top + r < height and 0 <= left + c < width
        }
        input_used |= abs_input
        output_used |= abs_output
        placements.append({"spec": spec, "top": int(top), "left": int(left)})

    inp = _render_input(height, width, placements)
    expected = _render_expected(height, width, placements)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_ATTEMPTS} attempts")
