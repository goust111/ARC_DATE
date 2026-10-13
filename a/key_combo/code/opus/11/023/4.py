from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 2
BODY_COLORS = (1, 3)
GEN_ATTEMPTS = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _foreground_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components_from_mask(np.asarray(grid, dtype=int) != BG)


def _paint_shifted(out: np.ndarray, grid: np.ndarray, coords: np.ndarray, dr: int, dc: int) -> None:
    h, w = out.shape
    for r, c in coords:
        rr = int(r) + int(dr)
        cc = int(c) + int(dc)
        if 0 <= rr < h and 0 <= cc < w:
            color = int(grid[int(r), int(c)])
            assert color in BODY_COLORS
            assert int(out[rr, cc]) in (BG, color)
            out[rr, cc] = color


def _parse_object(grid: np.ndarray, coords: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bbox = _bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    height = _bbox_h(bbox)
    width = _bbox_w(bbox)
    assert height != width, "object bbox must be non-square to define the crack axis"

    vals = {int(g[int(r), int(c)]) for r, c in coords}
    assert MARKER in vals, "each object must include a red crack"
    body_colors = vals - {BG, MARKER}
    assert len(body_colors) == 1, "each object must use exactly one body color"
    body = int(next(iter(body_colors)))
    assert body in BODY_COLORS

    marker_coords = np.array([(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == MARKER], dtype=int)
    body_coords = np.array([(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == body], dtype=int)
    assert marker_coords.size > 0 and body_coords.size > 0

    if width > height:
        marker_cols = {int(c) for _r, c in marker_coords}
        assert len(marker_cols) == 1, "wide object must have a vertical crack line"
        crack_pos = int(next(iter(marker_cols)))
        rows = [int(r) for r, _c in marker_coords]
        assert sorted(rows) == list(range(r0, r1 + 1)), "vertical crack must span the full bbox height"
        move_axis = "horizontal"
    else:
        marker_rows = {int(r) for r, _c in marker_coords}
        assert len(marker_rows) == 1, "tall object must have a horizontal crack line"
        crack_pos = int(next(iter(marker_rows)))
        cols = [int(c) for _r, c in marker_coords]
        assert sorted(cols) == list(range(c0, c1 + 1)), "horizontal crack must span the full bbox width"
        move_axis = "vertical"

    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    patch[patch == MARKER] = BG
    parts_local = _components_from_mask(patch == body)
    assert len(parts_local) == 2, "erasing the crack must leave exactly two pieces"

    parts = []
    for local in parts_local:
        abs_coords = local + np.array([r0, c0], dtype=int)
        part_bbox = _bbox_coords(abs_coords)
        parts.append({"coords": abs_coords, "bbox": part_bbox})

    if move_axis == "horizontal":
        parts.sort(key=lambda part: int(part["bbox"][1]))
        assert int(parts[0]["bbox"][3]) < crack_pos < int(parts[1]["bbox"][1])
        part_sizes = (_bbox_w(parts[0]["bbox"]), _bbox_w(parts[1]["bbox"]))
    else:
        parts.sort(key=lambda part: int(part["bbox"][0]))
        assert int(parts[0]["bbox"][2]) < crack_pos < int(parts[1]["bbox"][0])
        part_sizes = (_bbox_h(parts[0]["bbox"]), _bbox_h(parts[1]["bbox"]))

    return {
        "bbox": bbox,
        "body": body,
        "move_axis": move_axis,
        "crack_pos": crack_pos,
        "parts": parts,
        "part_sizes": part_sizes,
    }


def _parse_objects(grid: np.ndarray) -> list[dict]:
    objects = [_parse_object(np.asarray(grid, dtype=int), coords) for coords in _foreground_components(grid)]
    assert 2 <= len(objects) <= 3, "expected 2-3 fractured objects"
    return objects


def _transform(grid: np.ndarray, mode: str = "own") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = np.full_like(g, BG)

    for obj in _parse_objects(g):
        first, second = obj["parts"]
        size_a, size_b = obj["part_sizes"]
        if mode == "own":
            dist_a, dist_b = size_a, size_b
        elif mode == "fixed1":
            dist_a = dist_b = 1
        elif mode == "shared_max":
            dist_a = dist_b = max(size_a, size_b)
        else:
            raise ValueError(f"unknown transform mode: {mode}")

        if obj["move_axis"] == "horizontal":
            _paint_shifted(out, g, first["coords"], 0, -dist_a)
            _paint_shifted(out, g, second["coords"], 0, dist_b)
        else:
            _paint_shifted(out, g, first["coords"], -dist_a, 0)
            _paint_shifted(out, g, second["coords"], dist_b, 0)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Erase each red crack, split the object into two pieces, and move each piece outward by its own span.

    Input structure:
    - The grid has black background and 2-3 disconnected 8-connected objects.
    - Each object uses one body color (blue or green) plus a red crack line.
    - Every object has a non-square bbox; wide objects contain a full-height vertical crack, and tall objects contain a full-width horizontal crack.

    Rule:
    - Remove all red crack pixels.
    - The remaining body pixels split into exactly two 8-connected pieces.
    - If the crack is vertical, move the left piece left by its own bbox width and the right piece right by its own bbox width.
    - If the crack is horizontal, move the top piece up by its own bbox height and the bottom piece down by its own bbox height.
    - Pixels shifted outside the canvas are discarded.

    Key insight:
    - The shift distance is local to each split half, not fixed and not shared by the whole original object.
    """
    return _transform(grid, mode="own")


def _corner_notch_mask(height: int, width: int, corners: tuple[str, ...]) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if height < 3 or width < 3 or random.random() < 0.35:
        return mask

    corner = random.choice(corners)
    notch_h = random.randint(1, height - 1)
    notch_w = random.randint(1, width - 1)

    if corner == "tl":
        mask[:notch_h, :notch_w] = False
    elif corner == "tr":
        mask[:notch_h, width - notch_w :] = False
    elif corner == "bl":
        mask[height - notch_h :, :notch_w] = False
    elif corner == "br":
        mask[height - notch_h :, width - notch_w :] = False
    else:
        raise ValueError(corner)
    return mask


def _make_wide_template(force_asymmetry: bool) -> dict | None:
    for _ in range(80):
        height = random.randint(2, 6)
        left_w = random.randint(2, 6)
        right_w = random.randint(2, 6)
        if force_asymmetry and left_w == right_w:
            continue
        total_w = left_w + 1 + right_w
        if total_w <= height or total_w > 12:
            continue

        color = random.choice(BODY_COLORS)
        left_mask = _corner_notch_mask(height, left_w, ("tl", "bl"))
        right_mask = _corner_notch_mask(height, right_w, ("tr", "br"))

        in_pixels: list[tuple[int, int, int]] = []
        left_out: list[tuple[int, int, int]] = []
        right_out: list[tuple[int, int, int]] = []

        for r, c in np.argwhere(left_mask):
            rr = int(r)
            cc = int(c)
            in_pixels.append((rr, cc, color))
            left_out.append((rr, cc - left_w, color))
        for r in range(height):
            in_pixels.append((r, left_w, MARKER))
        for r, c in np.argwhere(right_mask):
            rr = int(r)
            cc = int(c)
            in_pixels.append((rr, left_w + 1 + cc, color))
            right_out.append((rr, left_w + 1 + cc + right_w, color))

        return {
            "orientation": "wide",
            "input_h": height,
            "input_w": total_w,
            "in_pixels": tuple(in_pixels),
            "out_groups": (tuple(left_out), tuple(right_out)),
            "out_pixels": tuple(left_out + right_out),
            "part_sizes": (left_w, right_w),
            "asymmetric": left_w != right_w,
        }
    return None


def _make_tall_template(force_asymmetry: bool) -> dict | None:
    for _ in range(80):
        width = random.randint(2, 6)
        top_h = random.randint(2, 6)
        bottom_h = random.randint(2, 6)
        if force_asymmetry and top_h == bottom_h:
            continue
        total_h = top_h + 1 + bottom_h
        if total_h <= width or total_h > 12:
            continue

        color = random.choice(BODY_COLORS)
        top_mask = _corner_notch_mask(top_h, width, ("tl", "tr"))
        bottom_mask = _corner_notch_mask(bottom_h, width, ("bl", "br"))

        in_pixels: list[tuple[int, int, int]] = []
        top_out: list[tuple[int, int, int]] = []
        bottom_out: list[tuple[int, int, int]] = []

        for r, c in np.argwhere(top_mask):
            rr = int(r)
            cc = int(c)
            in_pixels.append((rr, cc, color))
            top_out.append((rr - top_h, cc, color))
        for c in range(width):
            in_pixels.append((top_h, c, MARKER))
        for r, c in np.argwhere(bottom_mask):
            rr = int(r)
            cc = int(c)
            in_pixels.append((top_h + 1 + rr, cc, color))
            bottom_out.append((top_h + 1 + rr + bottom_h, cc, color))

        return {
            "orientation": "tall",
            "input_h": total_h,
            "input_w": width,
            "in_pixels": tuple(in_pixels),
            "out_groups": (tuple(top_out), tuple(bottom_out)),
            "out_pixels": tuple(top_out + bottom_out),
            "part_sizes": (top_h, bottom_h),
            "asymmetric": top_h != bottom_h,
        }
    return None


def _sample_orientations(count: int) -> list[str]:
    if count == 2:
        choices = [
            ["wide", "tall"],
            ["wide", "tall"],
            ["wide", "tall"],
            ["wide", "wide"],
            ["tall", "tall"],
        ]
        out = random.choice(choices)[:]
        random.shuffle(out)
        return out

    out = ["wide", "tall", random.choice(["wide", "tall"])]
    random.shuffle(out)
    return out


def _abs_pixels(spec: dict, r0: int, c0: int, key: str) -> list[tuple[int, int, int]]:
    return [(r0 + dr, c0 + dc, color) for dr, dc, color in spec[key]]


def _input_clear(input_occ: set[tuple[int, int]], abs_in: list[tuple[int, int, int]]) -> bool:
    for r, c, _color in abs_in:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in input_occ:
                return False
        if (r, c) in input_occ:
            return False
    return True


def _output_clear(output_occ: set[tuple[int, int]], visible: list[tuple[int, int, int]]) -> bool:
    for r, c, _color in visible:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in output_occ:
                return False
        if (r, c) in output_occ:
            return False
    return True


def _candidate_positions(
    spec: dict,
    shape: tuple[int, int],
    input_occ: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
    require_clip: bool,
) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    total_out = len(spec["out_pixels"])
    out_groups = spec["out_groups"]

    row_order = list(range(h - spec["input_h"] + 1))
    col_order = list(range(w - spec["input_w"] + 1))
    random.shuffle(row_order)
    random.shuffle(col_order)

    for r0 in row_order:
        for c0 in col_order:
            abs_in = _abs_pixels(spec, r0, c0, "in_pixels")
            if not _input_clear(input_occ, abs_in):
                continue

            abs_out = _abs_pixels(spec, r0, c0, "out_pixels")
            clipped = False
            visible: list[tuple[int, int, int]] = []
            blocked = False
            for rr, cc, color in abs_out:
                if 0 <= rr < h and 0 <= cc < w:
                    if (rr, cc) in output_occ:
                        blocked = True
                        break
                    visible.append((rr, cc, color))
                else:
                    clipped = True
            if blocked:
                continue
            if require_clip and not clipped:
                continue
            if not require_clip and clipped:
                continue
            if require_clip and len(visible) * 5 < total_out * 3:
                continue
            if require_clip:
                for group in out_groups:
                    visible_group = sum(1 for rr, cc, _color in _abs_pixels({"out_pixels": group}, r0, c0, "out_pixels") if 0 <= rr < h and 0 <= cc < w)
                    if visible_group == 0 or visible_group * 2 < len(group):
                        blocked = True
                        break
                if blocked:
                    continue
            if not _output_clear(output_occ, visible):
                continue

            out.append((r0, c0))
    return out


def _place_specs(specs: list[dict], shape: tuple[int, int], clip_index: int | None) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    input_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()

    for idx, spec in enumerate(specs):
        require_clip = clip_index == idx
        candidates = _candidate_positions(spec, shape, input_occ, output_occ, require_clip)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)

        for rr, cc, color in _abs_pixels(spec, r0, c0, "in_pixels"):
            grid[rr, cc] = int(color)
            input_occ.add((rr, cc))
        for rr, cc, _color in _abs_pixels(spec, r0, c0, "out_pixels"):
            if 0 <= rr < h and 0 <= cc < w:
                output_occ.add((rr, cc))
    return grid


def _validate_case(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert not np.array_equal(inp, out)
    assert not np.any(out == MARKER), "output must not keep crack markers"

    objects = _parse_objects(inp)
    assert 2 <= len(objects) <= 3
    assert any(obj["part_sizes"][0] != obj["part_sizes"][1] for obj in objects), "need at least one asymmetric split"

    move_sizes = {size for obj in objects for size in obj["part_sizes"]}
    assert len(move_sizes) >= 2, "samples must not collapse to a single global shift distance"

    expected = _transform(inp, mode="own")
    try:
        fixed1 = _transform(inp, mode="fixed1")
    except AssertionError:
        fixed1 = None
    try:
        shared = _transform(inp, mode="shared_max")
    except AssertionError:
        shared = None
    assert np.array_equal(out, expected)
    assert fixed1 is None or not np.array_equal(out, fixed1), "sample degenerates to a fixed 1-cell split"
    assert shared is None or not np.array_equal(out, shared), "sample degenerates to a shared-distance split"

    out_comps = _foreground_components(out)
    assert len(out_comps) == 2 * len(objects), "each split half should remain a separate visible object"


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((2, 2, 3))
    shape = (random.randint(14, 22), random.randint(16, 24))
    orientations = _sample_orientations(count)

    specs: list[dict] = []
    force_asym_index = random.randrange(count)
    for idx, orientation in enumerate(orientations):
        force_asym = idx == force_asym_index
        spec = _make_wide_template(force_asym) if orientation == "wide" else _make_tall_template(force_asym)
        if spec is None:
            return None
        specs.append(spec)

    if len({color for spec in specs for _dr, _dc, color in spec["in_pixels"] if color in BODY_COLORS}) == 1 and random.random() < 0.5:
        target = random.randrange(len(specs))
        old = next(color for _dr, _dc, color in specs[target]["in_pixels"] if color in BODY_COLORS)
        new = BODY_COLORS[0] if old == BODY_COLORS[1] else BODY_COLORS[1]
        specs[target] = {
            **specs[target],
            "in_pixels": tuple((dr, dc, new if color == old else color) for dr, dc, color in specs[target]["in_pixels"]),
            "out_pixels": tuple((dr, dc, new if color == old else color) for dr, dc, color in specs[target]["out_pixels"]),
        }

    clip_index = random.randrange(count) if random.random() < 0.25 else None
    inp = _place_specs(specs, shape, clip_index)
    if inp is None:
        return None

    out = main(inp)
    _validate_case(inp, out)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
