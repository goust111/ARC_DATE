from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 800


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[...] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert int(connectivity) == 4, "this task only uses 4-connected components"
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r0 = int(r)
        c0 = int(c)
        if bool(seen[r0, c0]):
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((int(cr), int(cc)))
            for dr, dc in DIR4:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _perimeter_coords(h: int, w: int) -> list[tuple[int, int]]:
    assert h >= 2 and w >= 2
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, int(c)))
    for r in range(1, h - 1):
        coords.append((int(r), w - 1))
    for c in range(w - 1, -1, -1):
        coords.append((h - 1, int(c)))
    for r in range(h - 2, 0, -1):
        coords.append((int(r), 0))
    return coords


def _corner_indices(h: int, w: int) -> list[int]:
    ring = _perimeter_coords(h, w)
    corner_set = {(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)}
    return [idx for idx, coord in enumerate(ring) if coord in corner_set]


def _ordered_cyclic_run(indices: list[int], ring_len: int) -> list[int]:
    ordered = sorted(set(map(int, indices)))
    assert len(ordered) == len(indices), "duplicate perimeter indices are not allowed"
    if len(ordered) == ring_len:
        return ordered
    breaks = [i for i in range(len(ordered)) if (ordered[(i + 1) % len(ordered)] - ordered[i]) % ring_len != 1]
    assert len(breaks) == 1, "border pixels of one color must form one cyclic run"
    start_pos = (breaks[0] + 1) % len(ordered)
    return ordered[start_pos:] + ordered[:start_pos]


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == BG
    h, w = map(int, g.shape)
    ring = _perimeter_coords(h, w)
    ring_len = len(ring)
    border_mask = np.zeros_like(g, dtype=bool)
    for r, c in ring:
        border_mask[int(r), int(c)] = True

    border_colors = {int(g[r, c]) for r, c in ring if int(g[r, c]) != bg}
    assert border_colors, "expected at least one colored perimeter segment"
    assert set(colors_present(g, exclude={bg})) == border_colors

    segments: list[dict] = []
    for color in border_colors:
        border_indices = [idx for idx, (r, c) in enumerate(ring) if int(g[r, c]) == int(color)]
        ordered = _ordered_cyclic_run(border_indices, ring_len)
        object_mask = (g == int(color)) & (~border_mask)
        comps = components_mask(object_mask, connectivity=4)
        assert len(comps) == 1, "each border color must have exactly one matching interior object"
        coords = comps[0]
        box = bbox_coords(coords)
        bbox_area = int((box[2] - box[0] + 1) * (box[3] - box[1] + 1))
        segments.append(
            {
                "color": int(color),
                "start": int(ordered[0]),
                "length": int(len(ordered)),
                "ordered_indices": [int(x) for x in ordered],
                "area": int(coords.shape[0]),
                "bbox": box,
                "bbox_area": bbox_area,
                "object_coords": coords.astype(int),
                "irregular": int(coords.shape[0]) < bbox_area,
            }
        )

    return {
        "shape": (h, w),
        "ring": ring,
        "ring_len": ring_len,
        "corner_indices": _corner_indices(h, w),
        "segments": segments,
    }


def _render_shifted(scene: dict, shift_by_color: dict[int, int]) -> np.ndarray | None:
    out = np.full(scene["shape"], BG, dtype=int)
    ring = scene["ring"]
    ring_len = int(scene["ring_len"])
    for seg in scene["segments"]:
        shift = int(shift_by_color[int(seg["color"])]) % ring_len
        for idx in seg["ordered_indices"]:
            new_idx = int((int(idx) + shift) % ring_len)
            r, c = ring[new_idx]
            if int(out[r, c]) != BG:
                return None
            out[r, c] = int(seg["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored border segment clockwise by the pixel area of the interior object with the same color.

    Input structure:
    - The background is black.
    - The outermost perimeter contains several non-black contiguous color segments, with one color used per segment.
    - Inside the border, each segment color appears exactly once again as a solid 4-connected object away from the perimeter.

    Rule:
    - Read the full outer border as one clockwise cyclic path.
    - For each color, count the pixels of the interior object of that color.
    - Shift that color's perimeter segment clockwise by exactly that many border cells, and clear every interior object.
    - Leave all other cells black.

    Key insight:
    - The inner object is only a numeric instruction for its same-colored border segment; the border must be handled as one loop, not as four independent sides.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    shift_by_color = {int(seg["color"]): int(seg["area"]) for seg in scene["segments"]}
    out = _render_shifted(scene, shift_by_color)
    assert out is not None, "translated border segments must remain disjoint"
    return out


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [int(total)]
    bars = sorted(rng.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for bar in bars + [total + parts - 1]:
        out.append(int(bar - prev - 1))
        prev = int(bar)
    return out


def _indices_overlap_free(intervals: list[list[int]], ring_len: int, *, require_gap: bool) -> bool:
    owner = np.full(int(ring_len), -1, dtype=int)
    for seg_id, interval in enumerate(intervals):
        for idx in interval:
            if int(owner[int(idx)]) != -1:
                return False
            owner[int(idx)] = int(seg_id)
    if require_gap:
        for idx in range(int(ring_len)):
            a = int(owner[idx])
            b = int(owner[(idx + 1) % ring_len])
            if a != -1 and b != -1 and a != b:
                return False
    return True


def _segment_spans_corner(indices: list[int], corner_indices: list[int], ring_len: int) -> bool:
    cells = set(map(int, indices))
    for corner in corner_indices:
        if corner in cells and (corner - 1) % ring_len in cells and (corner + 1) % ring_len in cells:
            return True
    return False


def _sample_segment_layout(
    h: int,
    w: int,
    colors: list[int],
    areas: list[int],
    rng: random.Random,
) -> list[dict] | None:
    ring_len = len(_perimeter_coords(h, w))
    corner_indices = _corner_indices(h, w)
    count = len(colors)

    for _ in range(400):
        lengths: list[int] = []
        corner_target = rng.randrange(count)
        for i, area in enumerate(areas):
            choices = [length for length in range(2, 6) if length != int(area)]
            if i == corner_target:
                choices = [length for length in choices if length >= 3]
            if not choices:
                break
            lengths.append(int(rng.choice(choices)))
        if len(lengths) != count:
            continue
        if sum(lengths) + count > ring_len:
            continue

        order = list(range(count))
        rng.shuffle(order)
        gaps = [1 + extra for extra in _random_composition(ring_len - sum(lengths) - count, count, rng)]
        offset = rng.randrange(ring_len)
        pos = int(offset)
        segments: list[dict] = []
        for gap, obj_idx in zip(gaps, order):
            start = int(pos % ring_len)
            length = int(lengths[obj_idx])
            area = int(areas[obj_idx])
            segments.append(
                {
                    "color": int(colors[obj_idx]),
                    "start": start,
                    "length": length,
                    "area": area,
                    "input_indices": [int((start + step) % ring_len) for step in range(length)],
                    "output_indices": [int((start + area + step) % ring_len) for step in range(length)],
                }
            )
            pos += int(length + gap)

        input_intervals = [seg["input_indices"] for seg in segments]
        output_intervals = [seg["output_indices"] for seg in segments]
        if not _indices_overlap_free(input_intervals, ring_len, require_gap=True):
            continue
        if not _indices_overlap_free(output_intervals, ring_len, require_gap=True):
            continue
        if not (
            any(_segment_spans_corner(seg["input_indices"], corner_indices, ring_len) for seg in segments)
            or any(_segment_spans_corner(seg["output_indices"], corner_indices, ring_len) for seg in segments)
        ):
            continue
        return segments
    return None


def _orient_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    k = rng.randrange(4)
    return np.rot90(out, k).copy()


def _grow_polyomino(area: int, rng: random.Random, *, irregular_required: bool) -> np.ndarray | None:
    for _ in range(160):
        cells = {(0, 0)}
        while len(cells) < int(area):
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != int(area):
            continue

        coords = np.asarray(sorted(cells), dtype=int)
        coords[:, 0] -= int(coords[:, 0].min())
        coords[:, 1] -= int(coords[:, 1].min())
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        if max(mask.shape) > 6:
            continue
        if irregular_required and (int(mask.sum()) == int(mask.size) or min(mask.shape) == 1):
            continue
        return _orient_mask(mask, rng)
    return None


def _make_object_mask(area: int, rng: random.Random, *, irregular_required: bool) -> np.ndarray | None:
    area = int(area)
    if irregular_required:
        return _grow_polyomino(area, rng, irregular_required=True)

    if rng.random() < 0.45:
        rect_choices = [(h, area // h) for h in range(1, area + 1) if area % h == 0 and max(h, area // h) <= 5]
        rect_choices = [dims for dims in rect_choices if dims[0] * dims[1] == area]
        if rect_choices:
            h, w = rng.choice(rect_choices)
            return _orient_mask(np.ones((int(h), int(w)), dtype=bool), rng)

    mask = _grow_polyomino(area, rng, irregular_required=False)
    if mask is not None:
        return mask
    return None


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = blocked.shape
    for r, c in np.argwhere(mask):
        rr = int(top + r)
        cc = int(left + c)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_objects(grid: np.ndarray, specs: list[dict], rng: random.Random) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:2, :] = True
    blocked[-2:, :] = True
    blocked[:, :2] = True
    blocked[:, -2:] = True

    for spec in sorted(specs, key=lambda item: int(item["mask"].sum()), reverse=True):
        mask = spec["mask"]
        patch_h, patch_w = mask.shape
        if patch_h > h - 4 or patch_w > w - 4:
            return False

        positions = [(r, c) for r in range(2, h - patch_h - 1) for c in range(2, w - patch_w - 1)]
        rng.shuffle(positions)
        placed = False
        for top, left in positions:
            coords = np.argwhere(mask)
            if any(bool(blocked[int(top + r), int(left + c)]) for r, c in coords):
                continue
            patch = np.full(mask.shape, BG, dtype=int)
            patch[mask] = int(spec["color"])
            paste(grid, patch, int(top), int(left), transparent=BG)
            _mark_blocked(blocked, mask, int(top), int(left))
            placed = True
            break
        if not placed:
            return False
    return True


def _scene_output(scene: dict) -> np.ndarray | None:
    return _render_shifted(scene, {int(seg["color"]): int(seg["area"]) for seg in scene["segments"]})


def _wrong_length_output(scene: dict) -> np.ndarray | None:
    return _render_shifted(scene, {int(seg["color"]): int(seg["length"]) for seg in scene["segments"]})


def _wrong_bbox_area_output(scene: dict) -> np.ndarray | None:
    return _render_shifted(scene, {int(seg["color"]): int(seg["bbox_area"]) for seg in scene["segments"]})


def _wrong_order_pair_output(scene: dict) -> np.ndarray | None:
    border_order = sorted(scene["segments"], key=lambda seg: (int(seg["start"]), int(seg["color"])))
    object_order = sorted(scene["segments"], key=lambda seg: (int(seg["bbox"][0]), int(seg["bbox"][1]), int(seg["color"])))
    shift_by_color = {
        int(border_seg["color"]): int(object_seg["area"])
        for border_seg, object_seg in zip(border_order, object_order)
    }
    return _render_shifted(scene, shift_by_color)


def _scene_is_semantic(scene: dict, out: np.ndarray) -> bool:
    segments = scene["segments"]
    ring_len = int(scene["ring_len"])
    if len(segments) < 2:
        return False
    if len({int(seg["area"]) for seg in segments}) != len(segments):
        return False
    if len({int(seg["color"]) for seg in segments}) != len(segments):
        return False
    if len({int(seg["length"]) for seg in segments}) == 1:
        return False
    if not any(bool(seg["irregular"]) for seg in segments):
        return False
    if not any(int(seg["area"]) != int(seg["length"]) for seg in segments):
        return False
    if np.array_equal(out, np.full(scene["shape"], BG, dtype=int)):
        return False

    border_order_colors = [int(seg["color"]) for seg in sorted(segments, key=lambda seg: int(seg["start"]))]
    object_order_colors = [int(seg["color"]) for seg in sorted(segments, key=lambda seg: (int(seg["bbox"][0]), int(seg["bbox"][1])))]
    if border_order_colors == object_order_colors:
        return False

    if not (
        any(_segment_spans_corner(seg["ordered_indices"], scene["corner_indices"], ring_len) for seg in segments)
        or any(
            _segment_spans_corner(
                [int((idx + seg["area"]) % ring_len) for idx in seg["ordered_indices"]],
                scene["corner_indices"],
                ring_len,
            )
            for seg in segments
        )
    ):
        return False

    wrong_length = _wrong_length_output(scene)
    if wrong_length is not None and np.array_equal(wrong_length, out):
        return False
    wrong_bbox = _wrong_bbox_area_output(scene)
    if wrong_bbox is not None and np.array_equal(wrong_bbox, out):
        return False
    wrong_order = _wrong_order_pair_output(scene)
    if wrong_order is not None and np.array_equal(wrong_order, out):
        return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 4)
    base = 7 + 2 * count
    h = rng.randint(base, min(18, base + 4))
    w = rng.randint(base, min(18, base + 4))
    colors = rng.sample(PALETTE, count)
    areas = rng.sample(list(range(2, 10)), count)
    if not any(area >= 4 for area in areas):
        return None

    segments = _sample_segment_layout(h, w, colors, areas, rng)
    if segments is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    ring = _perimeter_coords(h, w)
    for seg in segments:
        for idx in seg["input_indices"]:
            r, c = ring[int(idx)]
            grid[r, c] = int(seg["color"])

    irregular_color = int(rng.choice([color for color, area in zip(colors, areas) if int(area) >= 4]))
    specs: list[dict] = []
    for color, area in zip(colors, areas):
        mask = _make_object_mask(int(area), rng, irregular_required=int(color) == irregular_color)
        if mask is None:
            return None
        specs.append({"color": int(color), "area": int(area), "mask": mask})

    if not _place_objects(grid, specs, rng):
        return None

    try:
        scene = _extract_scene(grid)
        out = _scene_output(scene)
        if out is None:
            return None
        if not _scene_is_semantic(scene, out):
            return None
        if not np.array_equal(main(grid), out):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid perimeter-command-ring input")
