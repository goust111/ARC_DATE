from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(10))

MAX_GENERATION_ATTEMPTS = 400
SHAPE_RETRIES = 200
LAYOUT_RETRIES = 120
PANEL_HEIGHT_RANGE = (8, 14)
PANEL_WIDTH_RANGE = (9, 16)
OBJECT_COUNT_RANGE = (3, 5)
PLACEMENT_GAP = 1
MAX_STEP = 3


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = _bbox_from_coords(arr)
            patch = np.full((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), bg, dtype=int)
            patch[arr[:, 0] - bbox[0], arr[:, 1] - bbox[1]] = color
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "patch": patch,
                    "mask": patch != bg,
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _object_key(obj: dict, bg: int) -> tuple[int, tuple[int, int], bytes]:
    mask = obj["patch"] != bg
    return int(obj["color"]), tuple(mask.shape), mask.tobytes()


def _reflect_position(pos: int, low: int, high: int) -> int:
    low = int(low)
    high = int(high)
    pos = int(pos)
    if low == high:
        return low
    span = high - low
    period = 2 * span
    offset = (pos - low) % period
    if offset <= span:
        return low + offset
    return high - (offset - span)


def _split_frames(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    assert h % 2 == 1, "input must contain two equal panels separated by one divider row"
    mid = h // 2
    divider = g[mid]
    assert np.all(divider == divider[0]), "middle row must be a solid divider bar"
    top = g[:mid].copy()
    bottom = g[mid + 1 :].copy()
    bg_top = _get_bg(top)
    bg_bottom = _get_bg(bottom)
    assert bg_top == bg_bottom, "both panels must share one background color"
    assert int(divider[0]) != bg_top, "divider bar must differ from the panel background"
    return top, bottom, int(bg_top)


def _objects_by_key(panel: np.ndarray, bg: int) -> dict[tuple[int, tuple[int, int], bytes], dict]:
    objects = _find_components(panel, bg)
    assert objects, "expected at least one foreground object"
    by_key: dict[tuple[int, tuple[int, int], bytes], dict] = {}
    for obj in objects:
        key = _object_key(obj, bg)
        assert key not in by_key, "each color-shape pair must be unique inside one panel"
        obj = dict(obj)
        obj["key"] = key
        by_key[key] = obj
    return by_key


def _project_once(origin: tuple[int, int], delta: tuple[int, int], limit: tuple[int, int]) -> tuple[int, int]:
    r, c = origin
    dr, dc = delta
    max_r, max_c = limit
    return (
        _reflect_position(r + dr, 0, max_r),
        _reflect_position(c + dc, 0, max_c),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _origin_box(origin: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = origin
    h, w = shape
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.tobytes()


def _shape_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        if rng.random() < 0.1:
            return np.array([[True]], dtype=bool)
        h = rng.randint(1, 4)
        w = rng.randint(1, 4)
        max_area = min(6, h * w)
        min_area = 2 if max_area >= 2 else 1
        target = rng.randint(min_area, max_area)
        mask = np.zeros((h, w), dtype=bool)
        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < target:
            frontier = set()
            for rr, cc in np.argwhere(mask):
                rr = int(rr)
                cc = int(cc)
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(tuple(frontier))
            mask[nr, nc] = True
        if int(mask.sum()) != target:
            continue
        mask = _crop_bool(mask)
        if mask.shape[0] > 4 or mask.shape[1] > 4:
            continue
        return mask
    raise RuntimeError("failed to sample a connected shape")


def _sample_shapes(count: int, rng: random.Random) -> list[np.ndarray]:
    used: set[tuple[tuple[int, int], bytes]] = set()
    shapes: list[np.ndarray] = []
    while len(shapes) < count:
        shape = _random_shape(rng)
        sig = _shape_signature(shape)
        if sig in used:
            continue
        used.add(sig)
        shapes.append(shape)
    return shapes


def _sample_colors(count: int, bg: int, divider: int, rng: random.Random) -> list[int]:
    available = [c for c in ALL_COLORS if c not in {bg, divider}]
    if count >= 3 and rng.random() < 0.7:
        palette = rng.sample(available, rng.randint(2, min(4, count)))
        repeated = rng.choice(palette)
        colors = [repeated, repeated]
        while len(colors) < count:
            colors.append(rng.choice(palette))
        rng.shuffle(colors)
        return colors
    if rng.random() < 0.6:
        return rng.sample(available, count)
    palette = rng.sample(available, rng.randint(2, min(4, count)))
    return [rng.choice(palette) for _ in range(count)]


def _sample_roles(count: int, rng: random.Random) -> list[str]:
    roles = ["free", "free"]
    if rng.random() < 0.65:
        roles.append("bounce")
    else:
        roles.append("free")
    if len(roles) < count and rng.random() < 0.45:
        roles.append("still")
    while len(roles) < count:
        roles.append(rng.choices(["free", "bounce", "still"], weights=[4, 3, 2], k=1)[0])
    rng.shuffle(roles)
    return roles[:count]


def _enumerate_candidates(spec: dict, panel_shape: tuple[int, int], rng: random.Random) -> list[dict]:
    panel_h, panel_w = panel_shape
    h, w = spec["shape"].shape
    max_r = panel_h - h
    max_c = panel_w - w
    role = spec["role"]
    candidates: list[dict] = []

    if role == "still":
        for r2 in range(max_r + 1):
            for c2 in range(max_c + 1):
                origin = (r2, c2)
                box = _origin_box(origin, (h, w))
                candidates.append(
                    {
                        "top_origin": origin,
                        "bottom_origin": origin,
                        "out_origin": origin,
                        "top_box": box,
                        "bottom_box": box,
                        "out_box": box,
                        "delta": (0, 0),
                        "bounce": False,
                    }
                )
        rng.shuffle(candidates)
        return candidates

    row_steps = range(-min(MAX_STEP, max_r), min(MAX_STEP, max_r) + 1)
    col_steps = range(-min(MAX_STEP, max_c), min(MAX_STEP, max_c) + 1)
    vectors = [(dr, dc) for dr in row_steps for dc in col_steps if not (dr == 0 and dc == 0)]
    rng.shuffle(vectors)

    for dr, dc in vectors:
        for r2 in range(max_r + 1):
            r1 = r2 - dr
            if not (0 <= r1 <= max_r):
                continue
            row_bounce = not (0 <= r2 + dr <= max_r)
            for c2 in range(max_c + 1):
                c1 = c2 - dc
                if not (0 <= c1 <= max_c):
                    continue
                col_bounce = not (0 <= c2 + dc <= max_c)
                bounced = row_bounce or col_bounce
                if role == "free" and bounced:
                    continue
                if role == "bounce" and not bounced:
                    continue
                top_origin = (r1, c1)
                bottom_origin = (r2, c2)
                out_origin = _project_once(bottom_origin, (dr, dc), (max_r, max_c))
                candidates.append(
                    {
                        "top_origin": top_origin,
                        "bottom_origin": bottom_origin,
                        "out_origin": out_origin,
                        "top_box": _origin_box(top_origin, (h, w)),
                        "bottom_box": _origin_box(bottom_origin, (h, w)),
                        "out_box": _origin_box(out_origin, (h, w)),
                        "delta": (dr, dc),
                        "bounce": bounced,
                    }
                )

    rng.shuffle(candidates)
    return candidates


def _candidate_fits(candidate: dict, frame_boxes: list[list[tuple[int, int, int, int]]]) -> bool:
    return not any(
        _boxes_overlap(candidate[name], other, gap=PLACEMENT_GAP)
        for name, boxes in zip(("top_box", "bottom_box", "out_box"), frame_boxes)
        for other in boxes
    )


def _backtrack_place(
    ordered: list[tuple[int, dict, list[dict]]],
    frame_boxes: list[list[tuple[int, int, int, int]]],
    placements: list[dict | None],
    depth: int = 0,
) -> bool:
    if depth == len(ordered):
        return True

    spec_index, spec, candidates = ordered[depth]
    for candidate in candidates:
        if not _candidate_fits(candidate, frame_boxes):
            continue
        placements[spec_index] = candidate
        frame_boxes[0].append(candidate["top_box"])
        frame_boxes[1].append(candidate["bottom_box"])
        frame_boxes[2].append(candidate["out_box"])
        if _backtrack_place(ordered, frame_boxes, placements, depth + 1):
            return True
        frame_boxes[0].pop()
        frame_boxes[1].pop()
        frame_boxes[2].pop()
        placements[spec_index] = None
    return False


def _render_panel(specs: list[dict], placements: list[dict], frame_name: str, panel_shape: tuple[int, int], bg: int) -> np.ndarray:
    panel = np.full(panel_shape, bg, dtype=int)
    for spec, placement in zip(specs, placements):
        origin = placement[frame_name]
        _paste(panel, spec["patch"], origin[0], origin[1], transparent=bg)
    return panel


def _analyze_input(grid: np.ndarray) -> list[dict]:
    top, bottom, bg = _split_frames(grid)
    top_by_key = _objects_by_key(top, bg)
    bottom_by_key = _objects_by_key(bottom, bg)
    assert set(top_by_key) == set(bottom_by_key)
    info = []
    for key, bottom_obj in bottom_by_key.items():
        top_obj = top_by_key[key]
        dr = int(bottom_obj["bbox"][0] - top_obj["bbox"][0])
        dc = int(bottom_obj["bbox"][1] - top_obj["bbox"][1])
        h, w = bottom_obj["patch"].shape
        max_r = top.shape[0] - h
        max_c = top.shape[1] - w
        r2 = int(bottom_obj["bbox"][0])
        c2 = int(bottom_obj["bbox"][1])
        info.append(
            {
                "key": key,
                "color": int(bottom_obj["color"]),
                "delta": (dr, dc),
                "bounce": not (0 <= r2 + dr <= max_r and 0 <= c2 + dc <= max_c),
                "stationary": dr == 0 and dc == 0,
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate each lower-frame object into a bounced third frame using its own measured translation.

    Input structure:
    - The grid has odd height and is split by one solid divider row into equal upper and lower panels.
    - Both panels share one background color and contain the same separated monochrome objects.
    - Object identity is the unique pair of its color and tight-bbox shape; colors may repeat across different shapes.

    Rule:
    - Match every upper-panel object with the lower-panel object that has the same color and cropped shape mask.
    - Read the upper-to-lower displacement from the two bboxes' top-left corners.
    - Return one panel-sized grid where each lower-panel object is moved one more step by that same vector, reflecting independently on the row and column borders when the next step would leave the panel.

    Key insight:
    - The two panels are consecutive frames, so motion must be inferred per object from cross-frame correspondence rather than by applying one global shift; the third frame replays each object's own vector with wall reflection.
    """
    top, bottom, bg = _split_frames(grid)
    top_by_key = _objects_by_key(top, bg)
    bottom_by_key = _objects_by_key(bottom, bg)
    assert set(top_by_key) == set(bottom_by_key), "top and bottom panels must contain the same color-shape set"

    out = np.full(top.shape, bg, dtype=int)
    occupied = np.zeros(top.shape, dtype=bool)
    for key, bottom_obj in bottom_by_key.items():
        top_obj = top_by_key[key]
        dr = int(bottom_obj["bbox"][0] - top_obj["bbox"][0])
        dc = int(bottom_obj["bbox"][1] - top_obj["bbox"][1])

        patch = bottom_obj["patch"]
        mask = patch != bg
        h, w = patch.shape
        max_r = top.shape[0] - h
        max_c = top.shape[1] - w
        origin = (int(bottom_obj["bbox"][0]), int(bottom_obj["bbox"][1]))
        nr, nc = _project_once(origin, (dr, dc), (max_r, max_c))

        region = occupied[nr : nr + h, nc : nc + w]
        assert not np.any(region & mask), "predicted third-frame objects must not overlap"
        region |= mask
        _paste(out, patch, nr, nc, transparent=bg)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_h = rng.randint(*PANEL_HEIGHT_RANGE)
    panel_w = rng.randint(*PANEL_WIDTH_RANGE)
    assert 2 * panel_h + 1 <= 30 and panel_w <= 30

    bg = rng.choice(ALL_COLORS)
    divider = rng.choice([c for c in ALL_COLORS if c != bg])
    count = rng.randint(*OBJECT_COUNT_RANGE)
    shapes = _sample_shapes(count, rng)
    colors = _sample_colors(count, bg, divider, rng)
    roles = _sample_roles(count, rng)
    specs = [
        {
            "shape": shape,
            "patch": _shape_patch(shape, color, bg),
            "color": color,
            "role": role,
            "area": int(shape.sum()),
        }
        for shape, color, role in zip(shapes, colors, roles)
    ]

    for _ in range(LAYOUT_RETRIES):
        candidates_per_spec = [_enumerate_candidates(spec, (panel_h, panel_w), rng) for spec in specs]
        if any(not candidates for candidates in candidates_per_spec):
            return None
        ordered = sorted(
            [
                (idx, spec, candidates)
                for idx, (spec, candidates) in enumerate(zip(specs, candidates_per_spec))
            ],
            key=lambda item: (len(item[2]), -item[1]["area"]),
        )
        placements: list[dict | None] = [None] * len(specs)
        if not _backtrack_place(ordered, [[], [], []], placements):
            continue
        resolved = [p for p in placements if p is not None]
        vectors = [tuple(p["delta"]) for p in resolved]
        if len(set(vectors)) < 2:
            continue
        if all(v == (0, 0) for v in vectors):
            continue
        if not any(p["bounce"] for p in resolved) and any(spec["role"] == "bounce" for spec in specs):
            continue

        top = _render_panel(specs, resolved, "top_origin", (panel_h, panel_w), bg)
        bottom = _render_panel(specs, resolved, "bottom_origin", (panel_h, panel_w), bg)
        expected = _render_panel(specs, resolved, "out_origin", (panel_h, panel_w), bg)

        full = np.full((2 * panel_h + 1, panel_w), bg, dtype=int)
        full[:panel_h] = top
        full[panel_h] = divider
        full[panel_h + 1 :] = bottom

        try:
            produced = main(full)
        except AssertionError:
            continue
        if not np.array_equal(produced, expected):
            continue

        info = _analyze_input(full)
        if len(info) != count:
            continue
        if len({item["key"] for item in info}) != count:
            continue
        if len({item["delta"] for item in info}) < 2:
            continue
        if any(spec["role"] == "still" for spec in specs) and not any(item["stationary"] for item in info):
            continue
        if any(spec["role"] == "bounce" for spec in specs) and not any(item["bounce"] for item in info):
            continue
        return full
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-frame extrapolation input")
