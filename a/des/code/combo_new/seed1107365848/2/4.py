from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True

            components.append(
                {
                    "index": len(components),
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "mask": mask,
                }
            )

    return components


def _column_midpoint(mask: np.ndarray, col: int) -> int:
    rows = np.flatnonzero(mask[:, col])
    return int((int(rows.min()) + int(rows.max())) // 2)


def _row_midpoint(mask: np.ndarray, row: int) -> int:
    cols = np.flatnonzero(mask[row, :])
    return int((int(cols.min()) + int(cols.max())) // 2)


def _horizontal_candidate(a: dict, b: dict) -> dict | None:
    if a["mask"].shape != b["mask"].shape:
        return None

    if int(a["c1"]) + 1 < int(b["c0"]):
        left, right = a, b
    elif int(b["c1"]) + 1 < int(a["c0"]):
        left, right = b, a
    else:
        return None

    if not np.array_equal(left["mask"], np.fliplr(right["mask"])):
        return None

    left_mid = int(left["r0"]) + _column_midpoint(left["mask"], int(left["mask"].shape[1]) - 1)
    right_mid = int(right["r0"]) + _column_midpoint(right["mask"], 0)
    if left_mid != right_mid:
        return None

    return {
        "members": (int(left["index"]), int(right["index"])),
        "kind": "H",
        "left": int(left["index"]),
        "right": int(right["index"]),
        "row": left_mid,
    }


def _vertical_candidate(a: dict, b: dict) -> dict | None:
    if a["mask"].shape != b["mask"].shape:
        return None

    if int(a["r1"]) + 1 < int(b["r0"]):
        top, bottom = a, b
    elif int(b["r1"]) + 1 < int(a["r0"]):
        top, bottom = b, a
    else:
        return None

    if not np.array_equal(top["mask"], np.flipud(bottom["mask"])):
        return None

    top_mid = int(top["c0"]) + _row_midpoint(top["mask"], int(top["mask"].shape[0]) - 1)
    bottom_mid = int(bottom["c0"]) + _row_midpoint(bottom["mask"], 0)
    if top_mid != bottom_mid:
        return None

    return {
        "members": (int(top["index"]), int(bottom["index"])),
        "kind": "V",
        "top": int(top["index"]),
        "bottom": int(bottom["index"]),
        "col": top_mid,
    }


def _pair_components(components: list[dict]) -> list[dict]:
    count = len(components)
    options_by_index: dict[int, list[dict]] = {i: [] for i in range(count)}

    for i in range(count):
        for j in range(i + 1, count):
            horizontal = _horizontal_candidate(components[i], components[j])
            vertical = _vertical_candidate(components[i], components[j])
            for option in (horizontal, vertical):
                if option is None:
                    continue
                for member in option["members"]:
                    options_by_index[int(member)].append(option)

    used: set[int] = set()

    def search(chosen: list[dict]) -> list[dict] | None:
        if len(used) == count:
            return list(chosen)

        remaining = [idx for idx in range(count) if idx not in used]
        pivot = min(
            remaining,
            key=lambda idx: sum(
                1 for option in options_by_index[idx] if option["members"][0] not in used and option["members"][1] not in used
            ),
        )
        candidates = [
            option
            for option in options_by_index[pivot]
            if option["members"][0] not in used and option["members"][1] not in used
        ]
        for option in candidates:
            a, b = option["members"]
            used.add(int(a))
            used.add(int(b))
            chosen.append(option)
            result = search(chosen)
            if result is not None:
                return result
            chosen.pop()
            used.remove(int(a))
            used.remove(int(b))
        return None

    solution = search([])
    assert solution is not None, "components must admit a complete mirror pairing"
    return solution


def _winner_color(comp_a: dict, comp_b: dict) -> int:
    winner = min(
        (comp_a, comp_b),
        key=lambda comp: (-int(comp["area"]), int(comp["r0"]), int(comp["c0"])),
    )
    return int(winner["color"])


def _paint_component(grid: np.ndarray, comp: dict, color: int) -> None:
    coords = comp["coords"]
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _draw_pair_line(grid: np.ndarray, pair: dict, components: list[dict], color: int) -> None:
    if pair["kind"] == "H":
        left = components[int(pair["left"])]
        right = components[int(pair["right"])]
        row = int(pair["row"])
        for col in range(int(left["c1"]), int(right["c0"]) + 1):
            grid[row, col] = int(color)
        return

    top = components[int(pair["top"])]
    bottom = components[int(pair["bottom"])]
    col = int(pair["col"])
    for row in range(int(top["r1"]), int(bottom["r0"]) + 1):
        grid[row, col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair mirrored shapes, recolor each pair to the larger partner's color, and connect them.

    Input structure:
    - The background is black.
    - The grid contains an even number of monochromatic 4-connected foreground shapes.
    - Every shape uses a unique color, and each pair of shapes is related by reflection.
    - A pair is either arranged left/right with a horizontal mirror relation or top/bottom
      with a vertical mirror relation, with black cells between the two shapes.

    Rule:
    - Find the reflection partner of every shape by matching exact mirrored masks.
    - For each pair, choose the shared output color from the larger-area shape; if the
      areas tie, choose the shape with the smaller row index.
    - Recolor both shapes to that shared color.
    - Draw a 1-pixel-wide straight connector in the shared color: horizontal for left/right
      pairs between boundary-column midpoints, vertical for top/bottom pairs between
      boundary-row midpoints.

    Key insight:
    - Color is only an identity marker: the real pairing cue is geometric mirror symmetry,
      and the final color must be decided only after each mirrored pair has been found.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert components and len(components) % 2 == 0

    pairs = _pair_components(components)
    out = g.copy()
    for pair in pairs:
        idx_a, idx_b = pair["members"]
        comp_a = components[int(idx_a)]
        comp_b = components[int(idx_b)]
        color = _winner_color(comp_a, comp_b)
        _paint_component(out, comp_a, color)
        _paint_component(out, comp_b, color)
        _draw_pair_line(out, pair, components, color)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[random.randrange(height), random.randrange(width)] = True

    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    return _crop_mask(mask)


def _mirror_family_key(mask: np.ndarray) -> tuple:
    variants = [
        mask,
        np.fliplr(mask),
        np.flipud(mask),
        np.flipud(np.fliplr(mask)),
    ]
    return min((int(v.shape[0]), int(v.shape[1]), v.tobytes()) for v in variants)


def _random_asymmetric_mask(used_keys: set[tuple]) -> np.ndarray | None:
    for _ in range(120):
        raw_h = random.randint(3, 7)
        raw_w = random.randint(3, 7)
        max_area = min(raw_h * raw_w - 1, int(np.floor(raw_h * raw_w * 0.72)))
        min_area = max(5, raw_h + raw_w - 3)
        if min_area > max_area:
            continue

        target_area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(raw_h, raw_w, target_area)
        h, w = mask.shape
        area = int(mask.sum())
        if h < 3 or w < 3:
            continue
        if area >= h * w:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue

        key = _mirror_family_key(mask)
        if key in used_keys:
            continue
        return mask
    return None


def _build_pair_spec(mask: np.ndarray, orientation: str, colors: tuple[int, int]) -> dict:
    base = mask.copy()
    if orientation == "H" and random.random() < 0.5:
        base = np.flipud(base)
    if orientation == "V" and random.random() < 0.5:
        base = np.fliplr(base)

    if orientation == "H":
        left_mask = base if random.random() < 0.5 else np.fliplr(base)
        right_mask = np.fliplr(left_mask)
        h, w = left_mask.shape
        gap = random.randint(1, 5)
        return {
            "orientation": "H",
            "group_h": h,
            "group_w": 2 * w + gap,
            "shapes": [
                {"mask": left_mask, "color": int(colors[0]), "offset_r": 0, "offset_c": 0},
                {"mask": right_mask, "color": int(colors[1]), "offset_r": 0, "offset_c": w + gap},
            ],
        }

    top_mask = base if random.random() < 0.5 else np.flipud(base)
    bottom_mask = np.flipud(top_mask)
    h, w = top_mask.shape
    gap = random.randint(1, 5)
    return {
        "orientation": "V",
        "group_h": 2 * h + gap,
        "group_w": w,
        "shapes": [
            {"mask": top_mask, "color": int(colors[0]), "offset_r": 0, "offset_c": 0},
            {"mask": bottom_mask, "color": int(colors[1]), "offset_r": h + gap, "offset_c": 0},
        ],
    }


def _choose_grid_shape(specs: list[dict]) -> tuple[int, int] | None:
    padded_area = sum((int(spec["group_h"]) + 2) * (int(spec["group_w"]) + 2) for spec in specs)
    min_h = max(max(int(spec["group_h"]) for spec in specs), int(np.ceil(np.sqrt(padded_area * 0.7))))
    if min_h > 30:
        return None
    height = random.randint(min_h, 30)

    min_w = max(max(int(spec["group_w"]) for spec in specs), int(np.ceil(padded_area / height)))
    if min_w > 30:
        return None
    width = random.randint(min_w, 30)
    return height, width


def _region_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    row0 = max(0, r0 - margin)
    col0 = max(0, c0 - margin)
    row1 = min(occupied.shape[0], r0 + h + margin)
    col1 = min(occupied.shape[1], c0 + w + margin)
    return not bool(occupied[row0:row1, col0:col1].any())


def _mark_region(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> None:
    row0 = max(0, r0 - margin)
    col0 = max(0, c0 - margin)
    row1 = min(occupied.shape[0], r0 + h + margin)
    col1 = min(occupied.shape[1], c0 + w + margin)
    occupied[row0:row1, col0:col1] = True


def _place_specs(specs: list[dict], height: int, width: int) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = sorted(
        range(len(specs)),
        key=lambda idx: int(specs[idx]["group_h"]) * int(specs[idx]["group_w"]),
        reverse=True,
    )

    for idx in order:
        group_h = int(specs[idx]["group_h"])
        group_w = int(specs[idx]["group_w"])
        if group_h > height or group_w > width:
            return None

        candidates = [(r, c) for r in range(height - group_h + 1) for c in range(width - group_w + 1)]
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            if not _region_is_free(occupied, r0, c0, group_h, group_w, margin=1):
                continue
            placements[idx] = (r0, c0)
            _mark_region(occupied, r0, c0, group_h, group_w, margin=1)
            placed = True
            break
        if not placed:
            return None

    return placements


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _mask_winner_color(shape_a: dict, shape_b: dict) -> int:
    area_a = int(shape_a["mask"].sum())
    area_b = int(shape_b["mask"].sum())
    winner = min(
        (
            (area_a, int(shape_a["r0"]), int(shape_a["c0"]), int(shape_a["color"])),
            (area_b, int(shape_b["r0"]), int(shape_b["c0"]), int(shape_b["color"])),
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return int(winner[3])


def _render_expected(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        shape_a, shape_b = pair["shapes"]
        color = _mask_winner_color(shape_a, shape_b)
        _paint_mask(out, shape_a["mask"], int(shape_a["r0"]), int(shape_a["c0"]), color)
        _paint_mask(out, shape_b["mask"], int(shape_b["r0"]), int(shape_b["c0"]), color)

        if pair["orientation"] == "H":
            if int(shape_a["c0"]) < int(shape_b["c0"]):
                left, right = shape_a, shape_b
            else:
                left, right = shape_b, shape_a
            row = int(left["r0"]) + _column_midpoint(left["mask"], int(left["mask"].shape[1]) - 1)
            left_col = int(left["c0"]) + int(left["mask"].shape[1]) - 1
            right_col = int(right["c0"])
            out[row, left_col : right_col + 1] = int(color)
            continue

        if int(shape_a["r0"]) < int(shape_b["r0"]):
            top, bottom = shape_a, shape_b
        else:
            top, bottom = shape_b, shape_a
        col = int(top["c0"]) + _row_midpoint(top["mask"], int(top["mask"].shape[0]) - 1)
        top_row = int(top["r0"]) + int(top["mask"].shape[0]) - 1
        bottom_row = int(bottom["r0"])
        out[top_row : bottom_row + 1, col] = int(color)

    return out


def _attempt_generate() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    colors = random.sample(range(1, 10), 2 * pair_count)

    used_keys: set[tuple] = set()
    specs: list[dict] = []
    for pair_idx in range(pair_count):
        mask = _random_asymmetric_mask(used_keys)
        if mask is None:
            return None
        used_keys.add(_mirror_family_key(mask))
        orientation = random.choice(["H", "V"])
        pair_colors = (colors[2 * pair_idx], colors[2 * pair_idx + 1])
        specs.append(_build_pair_spec(mask, orientation, pair_colors))

    shape = _choose_grid_shape(specs)
    if shape is None:
        return None
    height, width = shape

    placements = _place_specs(specs, height, width)
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    placed_pairs: list[dict] = []
    for idx, spec in enumerate(specs):
        base_r, base_c = placements[idx]
        pair_shapes: list[dict] = []
        for shape_spec in spec["shapes"]:
            r0 = base_r + int(shape_spec["offset_r"])
            c0 = base_c + int(shape_spec["offset_c"])
            mask = shape_spec["mask"]
            color = int(shape_spec["color"])
            _paint_mask(grid, mask, r0, c0, color)
            pair_shapes.append({"mask": mask, "color": color, "r0": r0, "c0": c0})
        placed_pairs.append({"orientation": spec["orientation"], "shapes": pair_shapes})

    if len(_find_components(grid)) != 2 * pair_count:
        return None

    expected = _render_expected(grid, placed_pairs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-pair ARC input")
