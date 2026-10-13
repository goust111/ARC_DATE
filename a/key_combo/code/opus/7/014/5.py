from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, np.unique(grid))))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _reflect(coord: tuple[int, int], orientation: str, axis2: int) -> tuple[int, int]:
    r, c = coord
    if orientation == "vertical":
        return r, axis2 - c
    return axis2 - r, c


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sorted_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda rc: (rc[0], rc[1]))


def _patch_axes(patch: np.ndarray) -> list[tuple[str, int]]:
    h, w = patch.shape
    axes: list[tuple[str, int]] = []
    for axis2 in range(0, 2 * w - 1):
        ok = True
        for r in range(h):
            for c in range(w):
                rr, cc = _reflect((r, c), "vertical", axis2)
                if not _in_bounds(patch.shape, rr, cc) or int(patch[r, c]) != int(patch[rr, cc]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            axes.append(("vertical", axis2))
    for axis2 in range(0, 2 * h - 1):
        ok = True
        for r in range(h):
            for c in range(w):
                rr, cc = _reflect((r, c), "horizontal", axis2)
                if not _in_bounds(patch.shape, rr, cc) or int(patch[r, c]) != int(patch[rr, cc]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            axes.append(("horizontal", axis2))
    return axes


def _candidate_from_axis(
    grid: np.ndarray,
    color: int,
    visible_body: set[tuple[int, int]],
    visible_grey: set[tuple[int, int]],
    orientation: str,
    axis2: int,
) -> dict | None:
    h, w = grid.shape
    visible_types = {coord: 1 for coord in visible_body}
    visible_types.update({coord: 2 for coord in visible_grey})

    full_body = set(visible_body)
    full_grey = set(visible_grey)
    hidden_coords: set[tuple[int, int]] = set()
    hidden_grey = 0

    for coord, kind in visible_types.items():
        rr, cc = _reflect(coord, orientation, axis2)
        if not _in_bounds(grid.shape, rr, cc):
            return None
        reflected = (rr, cc)
        if reflected in visible_types:
            if visible_types[reflected] != kind:
                return None
        else:
            target = int(grid[rr, cc])
            if target == BG or target == color or target == GREY:
                return None
            hidden_coords.add(reflected)
            if kind == 2:
                hidden_grey += 1
        if kind == 1:
            full_body.add(reflected)
        else:
            full_grey.add(reflected)

    if not full_grey or hidden_grey == 0:
        return None

    full_mask = full_body | full_grey
    area = len(full_mask)
    if area < 8 or area > 20:
        return None

    r0, c0, r1, c1 = _bbox(full_mask)
    if orientation == "vertical":
        grey_rows = {r for r, _ in full_grey}
        grey_cols = sorted(c for _, c in full_grey)
        if len(grey_rows) != 1:
            return None
        crack_index = next(iter(grey_rows))
        if grey_cols != list(range(min(grey_cols), max(grey_cols) + 1)):
            return None
        if min(grey_cols) != c0 or max(grey_cols) != c1:
            return None
        upper = {coord for coord in full_mask if coord[0] < crack_index}
        lower = {coord for coord in full_mask if coord[0] > crack_index}
        if not upper or not lower:
            return None
        crack_orientation = "horizontal"
    else:
        grey_cols = {c for _, c in full_grey}
        grey_rows = sorted(r for r, _ in full_grey)
        if len(grey_cols) != 1:
            return None
        crack_index = next(iter(grey_cols))
        if grey_rows != list(range(min(grey_rows), max(grey_rows) + 1)):
            return None
        if min(grey_rows) != r0 or max(grey_rows) != r1:
            return None
        left = {coord for coord in full_mask if coord[1] < crack_index}
        right = {coord for coord in full_mask if coord[1] > crack_index}
        if not left or not right:
            return None
        crack_orientation = "vertical"

    score = (
        hidden_grey,
        len(hidden_coords),
        len(full_grey),
        area,
        -(r1 - r0 + c1 - c0),
    )
    return {
        "color": color,
        "axis_orientation": orientation,
        "axis2": axis2,
        "crack_orientation": crack_orientation,
        "crack_index": crack_index,
        "full_body": full_body,
        "full_grey": full_grey,
        "full_mask": full_mask,
        "hidden_coords": hidden_coords,
        "hidden_grey": hidden_grey,
        "score": score,
    }


def _all_candidates_for_color(grid: np.ndarray, color: int) -> list[dict]:
    visible_body = {tuple(map(int, coord)) for coord in np.argwhere(grid == color)}
    if not visible_body:
        return []

    grey_neighbors: set[tuple[int, int]] = set()
    for r, c in visible_body:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
                grey_neighbors.add((nr, nc))
    if not grey_neighbors:
        return []

    h, w = grid.shape
    candidates: list[dict] = []
    for axis2 in range(0, 2 * w - 1):
        cand = _candidate_from_axis(grid, color, visible_body, grey_neighbors, "vertical", axis2)
        if cand is not None:
            candidates.append(cand)
    for axis2 in range(0, 2 * h - 1):
        cand = _candidate_from_axis(grid, color, visible_body, grey_neighbors, "horizontal", axis2)
        if cand is not None:
            candidates.append(cand)
    return candidates


def _infer_fractured_object(grid: np.ndarray, color: int) -> dict | None:
    candidates = _all_candidates_for_color(grid, color)
    if not candidates:
        return None
    candidates.sort(
        key=lambda cand: (
            cand["score"],
            cand["axis_orientation"] == "vertical",
            -cand["axis2"],
        ),
        reverse=True,
    )
    return candidates[0]


def _separate_object(candidate: dict) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    crack_index = int(candidate["crack_index"])
    for r, c in candidate["full_mask"]:
        if candidate["crack_orientation"] == "horizontal":
            if r <= crack_index:
                out.add((r - 1, c))
            if r >= crack_index:
                out.add((r + 1, c))
        else:
            if c <= crack_index:
                out.add((r, c - 1))
            if c >= crack_index:
                out.add((r, c + 1))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete occluded symmetric shapes to reveal hidden grey fracture markers, then split the restored shapes apart.

    Input structure:
    - The grid has black background, one fully visible top occluder, and one or two lower colored shapes.
    - Each lower shape uses one body color plus a grey one-cell fracture line; the visible part is still sufficient to infer one horizontal or vertical symmetry axis.
    - The top occluder hides part of the lower shape, including part of its grey fracture line, so the fracture is incomplete in the input.

    Rule:
    - For each colored shape that visibly touches grey, infer the symmetry axis that lets every visible cell reflect either to the same shape or to an occluding non-background cell.
    - Reflect the visible body cells and visible grey cells across that axis to restore the hidden half of the lower shape.
    - Treat the restored grey line as the fracture line, erase its marker color, duplicate that center line into both halves, and move the two halves one cell away from the fracture.
    - Shapes with no such grey-linked fracture are unchanged; moved lower pixels remain hidden wherever the unchanged top occluder still covers them.

    Key insight:
    - The grey split marker is only partially visible in the input, so the split cannot be inferred directly; symmetry completion must happen first, and the recovered fracture line then determines the separation direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    out = np.full(g.shape, bg, dtype=int)

    moving: list[dict] = []
    static_colors: list[int] = []
    for color in _colors_present(g, exclude={bg, GREY}):
        candidate = _infer_fractured_object(g, color)
        if candidate is None:
            static_colors.append(color)
        else:
            moving.append(candidate)

    for candidate in moving:
        for r, c in _separate_object(candidate):
            if _in_bounds(g.shape, r, c):
                out[r, c] = int(candidate["color"])

    for color in static_colors:
        out[g == color] = color
    return out


def _make_profile_patch(color: int, orientation: str) -> dict | None:
    for _ in range(120):
        axis_on_cell = bool(random.getrandbits(1))
        allowed = [1, 3, 5, 7] if axis_on_cell else [2, 4, 6]
        width = random.choice([value for value in allowed if value <= 7])
        height = random.randint(4, 6)
        crack_row = random.randint(1, height - 2)
        widths = [random.choice([value for value in allowed if value <= width]) for _ in range(height)]
        widths[crack_row] = width
        if widths == widths[::-1]:
            continue
        if sum(widths) < 8 or sum(widths) > 20:
            continue

        patch = np.zeros((height, width), dtype=int)
        for r, row_width in enumerate(widths):
            c0 = (width - row_width) // 2
            c1 = c0 + row_width
            patch[r, c0:c1] = GREY if r == crack_row else color

        axes = _patch_axes(patch)
        if axes != [("vertical", width - 1)]:
            continue

        if orientation == "horizontal":
            patch = patch.T.copy()
            return {
                "patch": patch,
                "body_color": color,
                "axis_orientation": "horizontal",
                "axis2_local": patch.shape[0] - 1,
                "crack_orientation": "vertical",
                "crack_index_local": crack_row,
            }
        return {
            "patch": patch,
            "body_color": color,
            "axis_orientation": "vertical",
            "axis2_local": width - 1,
            "crack_orientation": "horizontal",
            "crack_index_local": crack_row,
        }
    return None


def _place_bottom_shapes(shapes: list[dict], grid_shape: tuple[int, int], orientation: str) -> bool:
    h, w = grid_shape
    if orientation == "vertical":
        max_width = max(shape["patch"].shape[1] for shape in shapes)
        common_c0 = random.randint(2, w - max_width - 3)
        total_h = sum(shape["patch"].shape[0] for shape in shapes)
        min_gap = 4
        need_h = total_h + min_gap * (len(shapes) - 1) + 4
        if need_h > h:
            return False
        cursor = random.randint(2, h - (total_h + min_gap * (len(shapes) - 1)) - 2)
        for shape in shapes:
            patch_h, patch_w = shape["patch"].shape
            shape["r0"] = cursor
            shape["c0"] = common_c0
            cursor += patch_h + min_gap
    else:
        max_height = max(shape["patch"].shape[0] for shape in shapes)
        common_r0 = random.randint(2, h - max_height - 3)
        total_w = sum(shape["patch"].shape[1] for shape in shapes)
        min_gap = 4
        need_w = total_w + min_gap * (len(shapes) - 1) + 4
        if need_w > w:
            return False
        cursor = random.randint(2, w - (total_w + min_gap * (len(shapes) - 1)) - 2)
        for shape in shapes:
            patch_h, patch_w = shape["patch"].shape
            shape["r0"] = common_r0
            shape["c0"] = cursor
            cursor += patch_w + min_gap
    return True


def _hidden_side_intersection(shapes: list[dict], orientation: str, side: str) -> list[int]:
    sets: list[set[int]] = []
    for shape in shapes:
        patch_h, patch_w = shape["patch"].shape
        if orientation == "vertical":
            local_axis2 = int(shape["axis2_local"])
            if side == "right":
                cols = {
                    shape["c0"] + c
                    for c in range(patch_w)
                    if 2 * c > local_axis2
                }
            else:
                cols = {
                    shape["c0"] + c
                    for c in range(patch_w)
                    if 2 * c < local_axis2
                }
            sets.append(cols)
        else:
            local_axis2 = int(shape["axis2_local"])
            if side == "bottom":
                rows = {
                    shape["r0"] + r
                    for r in range(patch_h)
                    if 2 * r > local_axis2
                }
            else:
                rows = {
                    shape["r0"] + r
                    for r in range(patch_h)
                    if 2 * r < local_axis2
                }
            sets.append(rows)
    common = set.intersection(*sets)
    return sorted(common)


def _choose_occluder(shapes: list[dict], grid_shape: tuple[int, int], orientation: str) -> dict | None:
    h, w = grid_shape
    if orientation == "vertical":
        side = random.choice(["left", "right"])
        common_cols = _hidden_side_intersection(shapes, orientation, side)
        if not common_cols:
            return None
        width = 1 if len(common_cols) == 1 else random.choice([1, 2])
        starts = [
            idx for idx in range(len(common_cols) - width + 1)
            if common_cols[idx : idx + width] == list(range(common_cols[idx], common_cols[idx] + width))
        ]
        if not starts:
            return None
        start = random.choice(starts)
        cols = common_cols[start : start + width]
        c0 = cols[0]
        c1 = cols[-1]
        r0 = max(0, min(shape["r0"] + 1 for shape in shapes) - random.randint(0, 1))
        r1 = min(h - 1, max(shape["r0"] + shape["patch"].shape[0] - 2 for shape in shapes) + random.randint(0, 1))
        if r1 - r0 + 1 < 3:
            return None
        return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "side": side}
    side = random.choice(["top", "bottom"])
    common_rows = _hidden_side_intersection(shapes, orientation, side)
    if not common_rows:
        return None
    height = 1 if len(common_rows) == 1 else random.choice([1, 2])
    starts = [
        idx for idx in range(len(common_rows) - height + 1)
        if common_rows[idx : idx + height] == list(range(common_rows[idx], common_rows[idx] + height))
    ]
    if not starts:
        return None
    start = random.choice(starts)
    rows = common_rows[start : start + height]
    r0 = rows[0]
    r1 = rows[-1]
    c0 = max(0, min(shape["c0"] + 1 for shape in shapes) - random.randint(0, 1))
    c1 = min(w - 1, max(shape["c0"] + shape["patch"].shape[1] - 2 for shape in shapes) + random.randint(0, 1))
    if c1 - c0 + 1 < 3:
        return None
    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "side": side}


def _draw_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            value = int(patch[r, c])
            if value != BG:
                grid[r0 + r, c0 + c] = value


def _render_input(grid_shape: tuple[int, int], shapes: list[dict], occluder: dict, top_color: int) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for shape in shapes:
        _draw_patch(grid, shape["patch"], int(shape["r0"]), int(shape["c0"]))
    grid[occluder["r0"] : occluder["r1"] + 1, occluder["c0"] : occluder["c1"] + 1] = top_color
    return grid


def _global_candidate_from_shape(shape: dict) -> dict:
    patch = shape["patch"]
    full_body: set[tuple[int, int]] = set()
    full_grey: set[tuple[int, int]] = set()
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            value = int(patch[r, c])
            if value == BG:
                continue
            coord = (shape["r0"] + r, shape["c0"] + c)
            if value == GREY:
                full_grey.add(coord)
            else:
                full_body.add(coord)
    return {
        "color": int(shape["body_color"]),
        "axis_orientation": shape["axis_orientation"],
        "axis2": (
            shape["c0"] + shape["axis2_local"]
            if shape["axis_orientation"] == "vertical"
            else shape["r0"] + shape["axis2_local"]
        ),
        "crack_orientation": shape["crack_orientation"],
        "crack_index": (
            shape["r0"] + shape["crack_index_local"]
            if shape["crack_orientation"] == "horizontal"
            else shape["c0"] + shape["crack_index_local"]
        ),
        "full_body": full_body,
        "full_grey": full_grey,
        "full_mask": full_body | full_grey,
    }


def _render_output(grid_shape: tuple[int, int], shapes: list[dict], occluder: dict, top_color: int) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    for shape in shapes:
        candidate = _global_candidate_from_shape(shape)
        for r, c in _separate_object(candidate):
            if _in_bounds(grid_shape, r, c):
                out[r, c] = int(shape["body_color"])
    out[occluder["r0"] : occluder["r1"] + 1, occluder["c0"] : occluder["c1"] + 1] = top_color
    return out


def _restored_scene(grid_shape: tuple[int, int], shapes: list[dict], occluder: dict, top_color: int) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for shape in shapes:
        candidate = _global_candidate_from_shape(shape)
        for r, c in candidate["full_body"]:
            grid[r, c] = int(shape["body_color"])
        for r, c in candidate["full_grey"]:
            grid[r, c] = GREY
    grid[occluder["r0"] : occluder["r1"] + 1, occluder["c0"] : occluder["c1"] + 1] = top_color
    return grid


def _visible_grey_and_hidden_stats(inp: np.ndarray, shape: dict) -> tuple[int, int, int]:
    patch = shape["patch"]
    visible_grey = 0
    hidden_grey = 0
    hidden_body = 0
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            value = int(patch[r, c])
            if value == BG:
                continue
            gr = shape["r0"] + r
            gc = shape["c0"] + c
            seen = int(inp[gr, gc])
            if value == GREY:
                if seen == GREY:
                    visible_grey += 1
                else:
                    hidden_grey += 1
            elif seen != value:
                hidden_body += 1
    return visible_grey, hidden_grey, hidden_body


def _validate_instance(inp: np.ndarray, expected: np.ndarray, shapes: list[dict], occluder: dict, top_color: int) -> bool:
    if np.array_equal(inp, expected):
        return False
    restored = _restored_scene(inp.shape, shapes, occluder, top_color)
    if np.array_equal(restored, expected):
        return False

    for shape in shapes:
        visible_grey, hidden_grey, hidden_body = _visible_grey_and_hidden_stats(inp, shape)
        if visible_grey == 0 or hidden_grey == 0 or hidden_body == 0:
            return False
        inferred = _infer_fractured_object(inp, int(shape["body_color"]))
        if inferred is None:
            return False
        truth = _global_candidate_from_shape(shape)
        if inferred["axis_orientation"] != truth["axis_orientation"]:
            return False
        if inferred["crack_orientation"] != truth["crack_orientation"]:
            return False
        if inferred["full_mask"] != truth["full_mask"]:
            return False
        candidates = _all_candidates_for_color(inp, int(shape["body_color"]))
        if len(candidates) < 1:
            return False
        top_score = max(cand["score"] for cand in candidates)
        if sum(cand["score"] == top_score for cand in candidates) != 1:
            return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    for shape in shapes:
        moved = _separate_object(_global_candidate_from_shape(shape))
        visible_after = sum(
            1
            for r, c in moved
            if _in_bounds(inp.shape, r, c) and not (occluder["r0"] <= r <= occluder["r1"] and occluder["c0"] <= c <= occluder["c1"])
        )
        if visible_after == 0:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    grid_shape = (random.randint(16, 22), random.randint(16, 22))
    n_bottom = random.choice([1, 2])
    orientation = random.choice(["vertical", "horizontal"])

    colors = random.sample(PALETTE, n_bottom + 1)
    top_color = colors[-1]
    shapes: list[dict] = []
    for color in colors[:-1]:
        shape = _make_profile_patch(color, orientation)
        if shape is None:
            return None
        shapes.append(shape)

    if not _place_bottom_shapes(shapes, grid_shape, orientation):
        return None
    occluder = _choose_occluder(shapes, grid_shape, orientation)
    if occluder is None:
        return None

    inp = _render_input(grid_shape, shapes, occluder, top_color)
    expected = _render_output(grid_shape, shapes, occluder, top_color)
    if not _validate_instance(inp, expected, shapes, occluder, top_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-fracture puzzle")
