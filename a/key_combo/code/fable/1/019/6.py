from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
CARDINAL = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
BOX_COUNT_CHOICES = (2, 2, 3)
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat).argmax())


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _translate_bbox(
    bbox: tuple[int, int, int, int], dr: int, dc: int, steps: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr * steps, c0 + dc * steps, r1 + dr * steps, c1 + dc * steps


def _strictly_inside(bbox: tuple[int, int, int, int], coord: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = coord
    return r0 < r < r1 and c0 < c < c1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_hollow_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if comp["area"] != 2 * h + 2 * w - 4:
        return False
    actual = {tuple(map(int, rc)) for rc in comp["coords"]}
    expected = set(_perimeter_coords(comp["bbox"]))
    return actual == expected


def _scene_from_grid(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    components = _find_components(g, bg)

    boxes: list[dict] = []
    singletons: list[dict] = []
    for comp in components:
        if comp["area"] == 1:
            r, c = map(int, comp["coords"][0])
            singletons.append({"coord": (r, c), "color": int(comp["color"])})
        else:
            assert _is_hollow_rect(comp), "all multi-pixel objects must be intact hollow rectangles"
            boxes.append(
                {
                    "color": int(comp["color"]),
                    "bbox": comp["bbox"],
                    "wall": [tuple(map(int, rc)) for rc in comp["coords"]],
                }
            )

    boxes.sort(key=lambda box: box["bbox"])
    assert boxes, "expected at least one hollow rectangle box"

    loose: list[dict] = []
    cargo_by_box: list[list[dict]] = [[] for _ in boxes]
    for pix in singletons:
        owners = [i for i, box in enumerate(boxes) if _strictly_inside(box["bbox"], pix["coord"])]
        assert len(owners) <= 1, "singletons may belong to at most one box interior"
        if owners:
            cargo_by_box[owners[0]].append(pix)
        else:
            loose.append(pix)

    for box, cargo in zip(boxes, cargo_by_box):
        assert cargo, "each box must contain cargo pixels"
        coords = np.asarray([pix["coord"] for pix in cargo], dtype=float)
        r0, c0, r1, c1 = box["bbox"]
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        bias_r = float(coords[:, 0].mean() - center_r)
        bias_c = float(coords[:, 1].mean() - center_c)
        if abs(bias_r) > abs(bias_c):
            assert abs(bias_r) > 1e-6
            move = (1 if bias_r > 0 else -1, 0)
        else:
            assert abs(bias_c) > 1e-6
            move = (0, 1 if bias_c > 0 else -1)
        box["cargo"] = cargo
        box["step"] = len(cargo)
        box["move"] = move
        box["new_bbox"] = _translate_bbox(box["bbox"], move[0], move[1], box["step"])

    return bg, boxes, loose


def main(grid: np.ndarray) -> np.ndarray:
    """
    Throw each hollow box toward its cargo-heavy side, leaving the cargo behind and recoloring any loose pixels caught inside the landing box.

    Input structure:
    - The background is black, with several monochrome one-cell-thick hollow rectangles of distinct colors.
    - Each rectangle contains several colored singleton cargo pixels; additional singleton loose pixels are scattered elsewhere.
    - For every box, the cargo centroid is strictly biased only up, down, left, or right relative to the box center.

    Rule:
    - For each box, infer the translation direction from the side toward which its cargo centroid is biased, and set the translation distance to the number of cargo pixels in that box.
    - Remove the box from its old location, keep all original cargo pixels fixed in place, and redraw the box at the translated location.
    - Any loose singleton strictly inside the translated box interior is recolored to the box color; loose pixels outside all landing interiors stay unchanged.

    Key insight:
    - The cargo encodes two hidden controls at once: its off-center mass picks the direction, while its pixel count picks the distance, and the old cargo must remain while only newly covered loose pixels change color.
    """
    g = np.asarray(grid, dtype=int)
    bg, boxes, loose = _scene_from_grid(g)

    out = g.copy()
    for box in boxes:
        for r, c in box["wall"]:
            out[r, c] = bg

    for box in boxes:
        for r, c in _perimeter_coords(box["new_bbox"]):
            out[r, c] = int(box["color"])

    for pix in loose:
        for box in boxes:
            if _strictly_inside(box["new_bbox"], pix["coord"]):
                r, c = pix["coord"]
                out[r, c] = int(box["color"])
                break

    return out


def _sample_box_shape(direction: str) -> dict:
    if direction in ("L", "R"):
        inner_h = 5
        inner_w = random.choice((3, 5))
        step = inner_w + 2 + random.randint(0, 1)
    else:
        inner_h = random.choice((3, 5))
        inner_w = 5
        step = inner_h + 2 + random.randint(0, 1)
    return {
        "direction": direction,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
        "step": step,
    }


def _cargo_local_coords(inner_h: int, inner_w: int, direction: str, step: int) -> list[tuple[int, int]]:
    if direction in ("L", "R"):
        anchor_c = inner_w - 1 if direction == "R" else 0
        support_c = inner_w - 2 if direction == "R" else 1
        center_r = inner_h // 2
        coords = [(r, anchor_c) for r in range(inner_h)]
        remaining = step - inner_h
        if remaining in (1, 3):
            coords.append((center_r, support_c))
        if remaining >= 2:
            coords.extend([(center_r - 1, support_c), (center_r + 1, support_c)])
    else:
        anchor_r = inner_h - 1 if direction == "D" else 0
        support_r = inner_h - 2 if direction == "D" else 1
        center_c = inner_w // 2
        coords = [(anchor_r, c) for c in range(inner_w)]
        remaining = step - inner_w
        if remaining in (1, 3):
            coords.append((support_r, center_c))
        if remaining >= 2:
            coords.extend([(support_r, center_c - 1), (support_r, center_c + 1)])
    assert len(coords) == step
    return coords


def _capture_local_coords(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    candidates = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r + c) % 2 == 0]
    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) != 1 for rr, cc in picked):
            picked.append(cell)
            if len(picked) == count:
                break
    assert len(picked) == count
    return picked


def _origin_candidates(h: int, w: int, spec: dict) -> list[dict]:
    dr, dc = CARDINAL[spec["direction"]]
    oh, ow, step = spec["outer_h"], spec["outer_w"], spec["step"]
    out: list[dict] = []
    for r0 in range(h - oh + 1):
        for c0 in range(w - ow + 1):
            nr = r0 + dr * step
            nc = c0 + dc * step
            if not (0 <= nr <= h - oh and 0 <= nc <= w - ow):
                continue
            old_bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            new_bbox = (nr, nc, nr + oh - 1, nc + ow - 1)
            corridor = (
                min(old_bbox[0], new_bbox[0]),
                min(old_bbox[1], new_bbox[1]),
                max(old_bbox[2], new_bbox[2]),
                max(old_bbox[3], new_bbox[3]),
            )
            out.append(
                {
                    "old_bbox": old_bbox,
                    "new_bbox": new_bbox,
                    "reserve_bbox": (
                        max(0, corridor[0] - 1),
                        max(0, corridor[1] - 1),
                        min(h - 1, corridor[2] + 1),
                        min(w - 1, corridor[3] + 1),
                    ),
                }
            )
    random.shuffle(out)
    return out


def _reserve_free(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not mask[r0 : r1 + 1, c0 : c1 + 1].any()


def _place_specs(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    order = sorted(range(len(specs)), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"] * specs[i]["step"], reverse=True)
    candidates = {i: _origin_candidates(h, w, specs[i]) for i in order}
    if any(not candidates[i] for i in order):
        return None

    def search(pos: int, reserved: np.ndarray, placed: list[dict | None]) -> list[dict] | None:
        if pos == len(order):
            return [item for item in placed if item is not None]
        idx = order[pos]
        for candidate in candidates[idx]:
            if not _reserve_free(reserved, candidate["reserve_bbox"]):
                continue
            new_reserved = reserved.copy()
            r0, c0, r1, c1 = candidate["reserve_bbox"]
            new_reserved[r0 : r1 + 1, c0 : c1 + 1] = True
            placed_next = placed.copy()
            merged = dict(specs[idx])
            merged.update(candidate)
            placed_next[idx] = merged
            solved = search(pos + 1, new_reserved, placed_next)
            if solved is not None:
                return solved
        return None

    return search(0, np.zeros((h, w), dtype=bool), [None] * len(specs))


def _absolute_inner_coords(old_bbox: tuple[int, int, int, int], locals_: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = old_bbox
    return [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in locals_]


def _assign_colors(coords: list[tuple[int, int]], allowed: list[int]) -> dict[tuple[int, int], int]:
    random.shuffle(allowed)
    assigned: dict[tuple[int, int], int] = {}
    coord_set = set(coords)
    for coord in sorted(coords):
        neighbor_colors = {assigned[nbr] for nbr in _neighbors4(*coord) if nbr in coord_set and nbr in assigned}
        options = [color for color in allowed if color not in neighbor_colors]
        assert options, "expected at least one singleton color not used by a 4-neighbor"
        assigned[coord] = random.choice(options)
    return assigned


def _render_input(h: int, w: int, boxes: list[dict], uncaptured: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for box in boxes:
        for r, c in _perimeter_coords(box["old_bbox"]):
            grid[r, c] = int(box["color"])
        for pix in box["cargo_pixels"]:
            r, c = pix["coord"]
            grid[r, c] = int(pix["color"])
        for pix in box["captured_loose"]:
            r, c = pix["coord"]
            grid[r, c] = int(pix["color"])
    for pix in uncaptured:
        r, c = pix["coord"]
        grid[r, c] = int(pix["color"])
    return grid


def _render_expected(inp: np.ndarray, boxes: list[dict]) -> np.ndarray:
    out = inp.copy()
    for box in boxes:
        for r, c in _perimeter_coords(box["old_bbox"]):
            out[r, c] = BACKGROUND
    for box in boxes:
        for r, c in _perimeter_coords(box["new_bbox"]):
            out[r, c] = int(box["color"])
        for pix in box["captured_loose"]:
            r, c = pix["coord"]
            out[r, c] = int(box["color"])
    return out


def _free_loose_cells(h: int, w: int, boxes: list[dict]) -> list[tuple[int, int]]:
    blocked = np.zeros((h, w), dtype=bool)
    for box in boxes:
        r0, c0, r1, c1 = box["reserve_bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if not blocked[r, c]]
    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in picked):
            picked.append(cell)
    return picked


def _augment_specs_with_pixels(boxes: list[dict], box_colors: list[int], singleton_palette: list[int]) -> None:
    for box, color in zip(boxes, box_colors):
        box["color"] = color
        cargo_local = _cargo_local_coords(box["inner_h"], box["inner_w"], box["direction"], box["step"])
        cargo_abs = _absolute_inner_coords(box["old_bbox"], cargo_local)
        cargo_colors = _assign_colors(cargo_abs, singleton_palette.copy())
        box["cargo_pixels"] = [{"coord": coord, "color": cargo_colors[coord]} for coord in cargo_abs]

        capture_count = box["capture_count"]
        capture_local = _capture_local_coords(box["inner_h"], box["inner_w"], capture_count)
        capture_abs = _absolute_inner_coords(box["new_bbox"], capture_local)
        capture_colors = _assign_colors(capture_abs, singleton_palette.copy())
        box["captured_loose"] = [{"coord": coord, "color": capture_colors[coord]} for coord in capture_abs]


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, boxes: list[dict], uncaptured: list[dict]) -> None:
    solved = main(inp)
    assert np.array_equal(solved, expected), "solver output must match the generator's explicit expected output"

    bg, parsed_boxes, parsed_loose = _scene_from_grid(inp)
    assert bg == BACKGROUND
    assert len(parsed_boxes) == len(boxes)
    assert len(parsed_loose) == sum(len(box["captured_loose"]) for box in boxes) + len(uncaptured)

    by_color = {box["color"]: box for box in boxes}
    for parsed in parsed_boxes:
        spec = by_color[int(parsed["color"])]
        assert parsed["bbox"] == spec["old_bbox"]
        assert parsed["new_bbox"] == spec["new_bbox"]
        assert parsed["step"] == spec["step"]
        assert tuple(parsed["move"]) == CARDINAL[spec["direction"]]
        for r, c in _perimeter_coords(spec["new_bbox"]):
            assert int(inp[r, c]) == BACKGROUND, "landing wall cells must be empty in the input"
        for pix in spec["cargo_pixels"]:
            assert not _strictly_inside(spec["new_bbox"], pix["coord"]), "old cargo must stay exposed after the box moves"
        for pix in spec["captured_loose"]:
            r, c = pix["coord"]
            assert _strictly_inside(spec["new_bbox"], pix["coord"])
            assert int(inp[r, c]) != int(spec["color"])
            assert int(expected[r, c]) == int(spec["color"])

    assert any(box["captured_loose"] for box in boxes), "at least one loose pixel should be captured"
    assert uncaptured, "need visible uncaptured loose pixels as counterexamples"
    for pix in uncaptured:
        r, c = pix["coord"]
        assert int(inp[r, c]) == int(expected[r, c]), "uncaptured loose pixels must remain unchanged"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_boxes = random.choice(BOX_COUNT_CHOICES)
        h = random.randint(22 if n_boxes == 3 else 18, 30)
        w = random.randint(22 if n_boxes == 3 else 18, 30)

        directions = random.sample(tuple(CARDINAL), n_boxes)
        specs = [_sample_box_shape(direction) for direction in directions]
        for spec in specs:
            spec["capture_count"] = random.choice((0, 1, 1, 2))
        if not any(spec["capture_count"] > 0 for spec in specs):
            specs[0]["capture_count"] = 1

        boxes = _place_specs(h, w, specs)
        if boxes is None:
            continue

        box_colors = random.sample(PALETTE, n_boxes)
        singleton_palette = [color for color in PALETTE if color not in box_colors]
        _augment_specs_with_pixels(boxes, box_colors, singleton_palette)

        free_cells = _free_loose_cells(h, w, boxes)
        if len(free_cells) < 2:
            continue
        need_uncaptured = random.randint(2, min(5, len(free_cells)))
        uncaptured_coords = free_cells[:need_uncaptured]
        uncaptured_colors = _assign_colors(uncaptured_coords, singleton_palette.copy())
        uncaptured = [{"coord": coord, "color": uncaptured_colors[coord]} for coord in uncaptured_coords]

        inp = _render_input(h, w, boxes, uncaptured)
        expected = _render_expected(inp, boxes)

        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        try:
            _semantic_checks(inp, expected, boxes, uncaptured)
        except AssertionError:
            continue
        return inp

    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
