from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
INPUT_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    expanded = (a[0] - margin, a[1] - margin, a[2] + margin, a[3] + margin)
    return _boxes_overlap(expanded, b)


def _mirror_bbox(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0, width - 1 - c1, r1, width - 1 - c0)


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    rects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords_list: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords_list.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            coords = np.asarray(coords_list, dtype=int)
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert int(coords.shape[0]) == height * width, "each object must be a solid rectangle"
            assert np.all(patch == color), "each rectangle must be monochromatic"
            rects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "height": height,
                    "width": width,
                }
            )

    rects.sort(key=lambda rect: rect["bbox"])
    return rects


def _shift_rectangles(rects: list[dict], grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    target_mask = np.zeros((h, w), dtype=bool)
    shifted: list[dict] = []

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        rh = int(rect["height"])
        rw = int(rect["width"])
        dst_r0 = r0 - rh
        dst_r1 = r1 - rh
        assert dst_r0 >= 0, "each rectangle must have enough headroom to move upward"
        assert np.all(g[dst_r0:r0, c0 : c1 + 1] == BLACK), "the h rows above each rectangle must be empty"
        assert not np.any(target_mask[dst_r0 : dst_r1 + 1, c0 : c1 + 1]), "shifted rectangles must not overlap"
        target_mask[dst_r0 : dst_r1 + 1, c0 : c1 + 1] = True
        shifted.append(
            {
                "color": int(rect["color"]),
                "bbox": (dst_r0, c0, dst_r1, c1),
                "height": rh,
                "width": rw,
            }
        )

    return shifted


def _paired_indices(shifted: list[dict], width: int) -> set[int]:
    by_box = {tuple(rect["bbox"]): idx for idx, rect in enumerate(shifted)}
    paired: set[int] = set()
    for idx, rect in enumerate(shifted):
        mirror = _mirror_bbox(tuple(rect["bbox"]), width)
        partner = by_box.get(mirror)
        if partner is None or partner == idx:
            continue
        other = shifted[partner]
        if rect["height"] == other["height"] and rect["width"] == other["width"]:
            paired.add(idx)
            paired.add(partner)
    return paired


def _render_output(shifted: list[dict], paired: set[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx in paired:
        r0, c0, r1, c1 = shifted[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every rectangle upward by its own height, then keep only vertically mirrored rectangle pairs in blue.

    Input structure:
    - The grid has a black background and an even number of columns.
    - Foreground objects are separated monochromatic solid rectangles.
    - Each rectangle has exactly as many empty rows above it as its own height, so it can move upward safely.

    Rule:
    - Shift each rectangle upward by a distance equal to its height and erase its original position.
    - On the shifted layout, compare rectangles by their full bounding boxes across the vertical midline.
    - If two shifted rectangles have the same size and are horizontal mirror images of each other, keep both and recolor them blue.
    - Erase every shifted rectangle that does not belong to such a mirror pair.

    Key insight:
    - The symmetry test is not performed on the input layout: each rectangle first moves by an amount tied to its own height, and only the shifted arrangement determines which objects survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    height, width = g.shape
    assert width % 2 == 0, "the mirror axis requires an even-width grid"
    assert int(np.min(g)) >= 0

    rects = _find_rectangles(g)
    assert rects, "expected at least one foreground rectangle"

    shifted = _shift_rectangles(rects, g)
    paired = _paired_indices(shifted, width)
    return _render_output(shifted, paired, (height, width))


def _spec_boxes(spec: dict) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    dest_box = (spec["dest_r0"], spec["c0"], spec["dest_r0"] + spec["h"] - 1, spec["c0"] + spec["w"] - 1)
    input_box = (spec["dest_r0"] + spec["h"], spec["c0"], spec["dest_r0"] + 2 * spec["h"] - 1, spec["c0"] + spec["w"] - 1)
    return dest_box, input_box


def _centered_on_axis(width: int, c0: int, rect_w: int) -> bool:
    return c0 == width - c0 - rect_w


def _can_add_specs(existing: list[dict], new_specs: list[dict], height: int, width: int) -> bool:
    all_specs = existing + new_specs

    for spec in new_specs:
        dest_box, input_box = _spec_boxes(spec)
        if dest_box[0] < 0 or dest_box[2] >= height or dest_box[1] < 0 or dest_box[3] >= width:
            return False
        if input_box[0] < 0 or input_box[2] >= height or input_box[1] < 0 or input_box[3] >= width:
            return False
        if _centered_on_axis(width, spec["c0"], spec["w"]):
            return False

    for i, spec_i in enumerate(all_specs):
        dest_i, input_i = _spec_boxes(spec_i)
        for j in range(i + 1, len(all_specs)):
            dest_j, input_j = _spec_boxes(all_specs[j])
            if _boxes_too_close(dest_i, dest_j, margin=1):
                return False
            if _boxes_too_close(input_i, input_j, margin=1):
                return False
            if _boxes_overlap(dest_i, input_j) or _boxes_overlap(input_i, dest_j):
                return False

    return True


def _make_pair_specs(height: int, width: int) -> list[dict] | None:
    for _ in range(80):
        rect_h = random.randint(1, 4)
        rect_w = random.randint(2, 5)
        max_left = (width - 2 * rect_w - 1) // 2
        if max_left < 0:
            continue
        c0 = random.randint(0, max_left)
        dest_r0 = random.randint(0, height - 2 * rect_h)
        mirror_c0 = width - c0 - rect_w
        return [
            {"h": rect_h, "w": rect_w, "dest_r0": dest_r0, "c0": c0, "role": "pair"},
            {"h": rect_h, "w": rect_w, "dest_r0": dest_r0, "c0": mirror_c0, "role": "pair"},
        ]
    return None


def _make_input_mirror_foil_specs(height: int, width: int) -> list[dict] | None:
    for _ in range(120):
        rect_w = random.randint(2, 5)
        h1, h2 = random.sample((1, 2, 3, 4), 2)
        max_left = (width - 2 * rect_w - 1) // 2
        if max_left < 0:
            continue
        top_min = max(h1, h2)
        top_max = height - max(h1, h2)
        if top_min > top_max:
            continue
        input_top = random.randint(top_min, top_max)
        c0 = random.randint(0, max_left)
        mirror_c0 = width - c0 - rect_w
        return [
            {"h": h1, "w": rect_w, "dest_r0": input_top - h1, "c0": c0, "role": "foil"},
            {"h": h2, "w": rect_w, "dest_r0": input_top - h2, "c0": mirror_c0, "role": "foil"},
        ]
    return None


def _make_stray_spec(height: int, width: int) -> dict | None:
    for _ in range(120):
        rect_h = random.randint(1, 4)
        rect_w = random.randint(2, 5)
        c0 = random.randint(0, width - rect_w)
        if _centered_on_axis(width, c0, rect_w):
            continue
        dest_r0 = random.randint(0, height - 2 * rect_h)
        return {"h": rect_h, "w": rect_w, "dest_r0": dest_r0, "c0": c0, "role": "stray"}
    return None


def _build_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    colors = random.sample(INPUT_COLORS, len(specs))
    for spec, color in zip(specs, colors, strict=True):
        spec["color"] = int(color)
        _, input_box = _spec_boxes(spec)
        r0, c0, r1, c1 = input_box
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _shifted_specs(specs: list[dict]) -> list[dict]:
    return [
        {
            "color": int(spec["color"]),
            "bbox": _spec_boxes(spec)[0],
            "height": int(spec["h"]),
            "width": int(spec["w"]),
            "role": spec["role"],
        }
        for spec in specs
    ]


def _has_input_mirror_foil(rects: list[dict], width: int) -> bool:
    by_signature: dict[tuple[int, int, int, int], list[dict]] = {}
    for rect in rects:
        r0, c0, _r1, _c1 = rect["bbox"]
        key = (r0, rect["width"], c0, width - c0 - rect["width"])
        by_signature.setdefault(key, []).append(rect)

    for rect in rects:
        r0, c0, _r1, _c1 = rect["bbox"]
        mirror_c0 = width - c0 - rect["width"]
        for other in rects:
            if other is rect:
                continue
            if other["bbox"][0] != r0:
                continue
            if other["bbox"][1] != mirror_c0:
                continue
            if other["width"] != rect["width"]:
                continue
            if other["height"] != rect["height"]:
                return True
    return False


def _validate_generated_case(grid: np.ndarray, specs: list[dict], require_foil: bool) -> np.ndarray:
    h, w = grid.shape
    assert 4 <= len(specs) <= 6
    assert w % 2 == 0

    rects = _find_rectangles(grid)
    assert len(rects) == len(specs), "component count mismatch"
    assert len({rect["color"] for rect in rects}) == len(rects), "input rectangle colors must be distinct"

    shifted = _shift_rectangles(rects, grid)
    paired = _paired_indices(shifted, w)
    assert paired, "there must be at least one surviving mirror pair"
    assert len(paired) < len(rects), "there must be at least one erased rectangle"
    expected = _render_output(shifted, paired, (h, w))

    intended_shifted = _shifted_specs(specs)
    intended_paired = _paired_indices(intended_shifted, w)
    assert intended_paired == paired, "constructed roles and actual pairing must agree"
    intended_output = _render_output(intended_shifted, intended_paired, (h, w))
    assert np.array_equal(expected, intended_output)

    if require_foil:
        assert _has_input_mirror_foil(rects, w), "expected a misleading mirror-in-input foil pair"

    actual = main(grid)
    assert np.array_equal(actual, expected)
    return expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(16, 22)
        width = random.choice((12, 14, 16, 18))
        pair_count = random.choice((1, 2))
        total_count = random.randint(max(4, 2 * pair_count + 1), min(6, 2 * pair_count + 2))
        remaining = total_count - 2 * pair_count
        use_foil = remaining >= 2 and random.random() < 0.7

        specs: list[dict] = []
        ok = True

        for _pair_index in range(pair_count):
            placed = False
            for _ in range(120):
                pair_specs = _make_pair_specs(height, width)
                if pair_specs is not None and _can_add_specs(specs, pair_specs, height, width):
                    specs.extend(pair_specs)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        if use_foil:
            placed = False
            for _ in range(160):
                foil_specs = _make_input_mirror_foil_specs(height, width)
                if foil_specs is not None and _can_add_specs(specs, foil_specs, height, width):
                    specs.extend(foil_specs)
                    remaining -= 2
                    placed = True
                    break
            if not placed:
                continue

        while remaining > 0:
            stray = _make_stray_spec(height, width)
            if stray is None:
                ok = False
                break
            if _can_add_specs(specs, [stray], height, width):
                specs.append(stray)
                remaining -= 1
        if not ok:
            continue

        grid = _build_input(specs, (height, width))
        try:
            _validate_generated_case(grid, specs, require_foil=use_foil)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 5")
