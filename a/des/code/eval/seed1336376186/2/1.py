from __future__ import annotations

from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
NUM_FRAME_CHOICES = (2, 3)
FRAME_INNER_H_PAD_RANGE = (2, 6)
FRAME_INNER_W_PAD_RANGE = (2, 7)
FRAME_PLACE_PADDING = 1
TEMPLATE_PLACE_PADDING = 1
MAX_GENERATION_ATTEMPTS = 300


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values).reshape(-1), return_counts=True)
    return int(vals[int(counts.argmax())])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    rows = arr[:, 0]
    cols = arr[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize_cells(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    seq = list(cells)
    min_r = min(r for r, _ in seq)
    min_c = min(c for _, c in seq)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in seq))


def _shape_key(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    arr = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), rel


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    dst[int(r0) : int(r0 + h), int(c0) : int(c0 + w)] = patch


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(len(coords)) != 2 * (h + w) - 4:
        return False
    s = {(int(r), int(c)) for r, c in coords}
    for r in range(r0, r1 + 1):
        if (int(r), int(c0)) not in s or (int(r), int(c1)) not in s:
            return False
    for c in range(c0, c1 + 1):
        if (int(r0), int(c)) not in s or (int(r1), int(c)) not in s:
            return False
    for r, c in s:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _components(grid != int(bg))
    frames: list[dict] = []
    for comp in comps:
        colors = {int(grid[int(r), int(c)]) for r, c in comp}
        if len(colors) != 1:
            continue
        if not _is_hollow_rectangle(comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if int(r1 - r0 + 1) < 5 or int(c1 - c0 + 1) < 5:
            continue
        inner = grid[int(r0 + 1) : int(r1), int(c0 + 1) : int(c1)]
        inner_comps = _components(inner != int(bg))
        if len(inner_comps) != 1:
            continue
        frag_rel = inner_comps[0]
        if not (3 <= int(len(frag_rel)) <= 6):
            continue
        frag_colors = {int(inner[int(r), int(c)]) for r, c in frag_rel}
        if len(frag_colors) != 1:
            continue
        fr0, fc0, fr1, fc1 = _bbox(frag_rel)
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "frame_color": int(next(iter(colors))),
                "fragment_color": int(next(iter(frag_colors))),
                "fragment_key": _shape_key(frag_rel),
                "fragment_bbox_abs": (
                    int(r0 + 1 + fr0),
                    int(c0 + 1 + fc0),
                    int(r0 + 1 + fr1),
                    int(c0 + 1 + fc1),
                ),
            }
        )
    return frames


def _find_templates(grid: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    outside = grid != int(bg)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[int(r0) : int(r1 + 1), int(c0) : int(c1 + 1)] = False

    templates: list[dict] = []
    for comp in _components(outside):
        area = int(len(comp))
        if not (3 <= area <= 6):
            continue
        counts = Counter(int(grid[int(r), int(c)]) for r, c in comp)
        if len(counts) != 2:
            continue
        marker_colors = [col for col, cnt in counts.items() if int(cnt) == 1]
        body_colors = [col for col, cnt in counts.items() if int(cnt) == area - 1]
        if len(marker_colors) != 1 or len(body_colors) != 1:
            continue
        r0, c0, _r1, _c1 = _bbox(comp)
        rel = np.array([(int(r - r0), int(c - c0)) for r, c in comp], dtype=int)
        marker_color = int(marker_colors[0])
        marker_rel = next((int(r - r0), int(c - c0)) for r, c in comp if int(grid[int(r), int(c)]) == marker_color)
        templates.append(
            {
                "coords": comp,
                "shape_key": _shape_key(rel),
                "body_color": int(body_colors[0]),
                "marker_color": marker_color,
                "marker_rel": marker_rel,
            }
        )
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's internal grey fragment to an outside template of the same shape.
    The template's uniquely recolored pixel gives a row inside the frame: background pixels
    above that row use the template body color, and background pixels on/under that row use
    the marker color. Frame borders and grey fragments stay; templates are erased.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frames = _find_frames(g, bg)
    templates = _find_templates(g, bg, frames)

    template_by_shape: dict[tuple[int, int, tuple[tuple[int, int], ...]], dict] = {}
    for tmpl in templates:
        key = tmpl["shape_key"]
        assert key not in template_by_shape, "template shapes must be unique"
        template_by_shape[key] = tmpl

    out = g.copy()

    for tmpl in templates:
        for r, c in tmpl["coords"]:
            out[int(r), int(c)] = int(bg)

    for frame in frames:
        key = frame["fragment_key"]
        assert key in template_by_shape, "every frame fragment must have a matching template"
        tmpl = template_by_shape[key]
        frag_r0 = int(frame["fragment_bbox_abs"][0])
        divider_row = int(frag_r0 + tmpl["marker_rel"][0])
        r0, c0, r1, c1 = frame["bbox"]
        for r in range(int(r0 + 1), int(r1)):
            fill = int(tmpl["body_color"]) if int(r) < divider_row else int(tmpl["marker_color"])
            row = out[int(r), int(c0 + 1) : int(c1)]
            row[row == int(bg)] = fill

    return out


def _build_shape_library() -> list[dict]:
    shapes_by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 7):
        prev = shapes_by_size[int(size - 1)]
        current: set[tuple[tuple[int, int], ...]] = set()
        for shape in prev:
            cell_set = set(shape)
            for r, c in shape:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if (nr, nc) in cell_set:
                        continue
                    current.add(_normalize_cells(cell_set | {(nr, nc)}))
        shapes_by_size[int(size)] = current

    library: list[dict] = []
    for size in range(3, 7):
        for shape in sorted(shapes_by_size[int(size)]):
            arr = np.array(shape, dtype=int)
            h = int(arr[:, 0].max()) + 1
            w = int(arr[:, 1].max()) + 1
            library.append(
                {
                    "coords": shape,
                    "h": h,
                    "w": w,
                    "area": int(size),
                    "key": (h, w, shape),
                }
            )
    return library


SHAPE_LIBRARY = _build_shape_library()


def _sample_distinct_shapes(rng: np.random.Generator, count: int) -> list[dict]:
    indices = rng.choice(len(SHAPE_LIBRARY), size=int(count), replace=False)
    return [SHAPE_LIBRARY[int(i)] for i in indices]


def _sample_colors(rng: np.random.Generator, n_frames: int) -> tuple[int, int, list[int], list[tuple[int, int]]]:
    bg = int(rng.integers(0, 10))
    fragment_color = int(rng.choice([c for c in range(10) if c != bg]))
    remaining = [c for c in range(10) if c not in {bg, fragment_color}]
    frame_colors = [int(x) for x in rng.choice(remaining, size=int(n_frames), replace=False)]

    pairs: list[tuple[int, int]] = []
    used_pairs: set[tuple[int, int]] = set()
    for frame_color in frame_colors:
        candidates = [c for c in remaining if c != int(frame_color)]
        for _ in range(50):
            pair = tuple(int(x) for x in rng.choice(candidates, size=2, replace=False))
            if pair not in used_pairs:
                used_pairs.add(pair)
                pairs.append(pair)
                break
        else:
            raise RuntimeError("failed to sample unique template color pairs")
    return bg, fragment_color, frame_colors, pairs


def _make_frame_spec(
    rng: np.random.Generator,
    shape: dict,
    frame_color: int,
    fragment_color: int,
    body_color: int,
    marker_color: int,
) -> dict:
    placeholder = -1
    frag_h = int(shape["h"])
    frag_w = int(shape["w"])
    inner_h_low = frag_h + FRAME_INNER_H_PAD_RANGE[0]
    inner_h_high = frag_h + FRAME_INNER_H_PAD_RANGE[1]
    inner_w_low = frag_w + FRAME_INNER_W_PAD_RANGE[0]
    inner_w_high = frag_w + FRAME_INNER_W_PAD_RANGE[1]
    inner_h = int(rng.integers(inner_h_low, inner_h_high + 1))
    inner_w = int(rng.integers(inner_w_low, inner_w_high + 1))

    top = int(rng.integers(1, inner_h - frag_h))
    left = int(rng.integers(1, inner_w - frag_w))
    marker_rel = shape["coords"][int(rng.integers(0, len(shape["coords"])))]
    divider_interior_row = int(top + marker_rel[0])

    patch = np.full((inner_h + 2, inner_w + 2), placeholder, dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    for rr, cc in shape["coords"]:
        patch[int(1 + top + rr), int(1 + left + cc)] = int(fragment_color)

    template = np.full((frag_h, frag_w), placeholder, dtype=int)
    for rr, cc in shape["coords"]:
        template[int(rr), int(cc)] = int(body_color)
    template[int(marker_rel[0]), int(marker_rel[1])] = int(marker_color)

    return {
        "shape_key": shape["key"],
        "frame_patch": patch,
        "template_patch": template,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "divider_interior_row": divider_interior_row,
        "body_color": int(body_color),
        "marker_color": int(marker_color),
    }


def _place_items(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    items: list[dict],
) -> list[tuple[int, int]] | None:
    h, w = map(int, canvas_shape)
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(items)
    order = sorted(
        range(len(items)),
        key=lambda i: int(items[i]["reserve_shape"][0] * items[i]["reserve_shape"][1]),
        reverse=True,
    )
    for idx in order:
        item_h, item_w = map(int, items[idx]["reserve_shape"])
        pad = int(items[idx]["padding"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, h - item_h + 1):
            for c0 in range(0, w - item_w + 1):
                rr0 = max(0, int(r0 - pad))
                cc0 = max(0, int(c0 - pad))
                rr1 = min(h, int(r0 + item_h + pad))
                cc1 = min(w, int(c0 + item_w + pad))
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        pick = candidates[int(rng.integers(0, len(candidates)))]
        positions[idx] = pick
        rr0 = max(0, int(pick[0] - pad))
        cc0 = max(0, int(pick[1] - pad))
        rr1 = min(h, int(pick[0] + item_h + pad))
        cc1 = min(w, int(pick[1] + item_w + pad))
        occupied[rr0:rr1, cc0:cc1] = True
    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _make_input_and_output(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = int(rng.choice(NUM_FRAME_CHOICES))
    shapes = _sample_distinct_shapes(rng, n_frames)
    bg, fragment_color, frame_colors, template_pairs = _sample_colors(rng, n_frames)

    specs = [
        _make_frame_spec(
            rng,
            shape=shape,
            frame_color=int(frame_color),
            fragment_color=int(fragment_color),
            body_color=int(pair[0]),
            marker_color=int(pair[1]),
        )
        for shape, frame_color, pair in zip(shapes, frame_colors, template_pairs, strict=False)
    ]

    items: list[dict] = []
    for spec in specs:
        frame_patch = spec["frame_patch"].copy()
        frame_patch[frame_patch < 0] = int(bg)
        spec["frame_patch"] = frame_patch

        template_patch = spec["template_patch"].copy()
        template_patch[template_patch < 0] = int(bg)
        spec["template_patch"] = template_patch

        items.append(
            {
                "kind": "frame",
                "reserve_shape": frame_patch.shape,
                "padding": FRAME_PLACE_PADDING,
            }
        )
        items.append(
            {
                "kind": "template",
                "reserve_shape": template_patch.shape,
                "padding": TEMPLATE_PLACE_PADDING,
            }
        )

    min_h = max(item["reserve_shape"][0] for item in items) + 2
    min_w = max(item["reserve_shape"][1] for item in items) + 2
    total_area = sum(int(item["reserve_shape"][0] * item["reserve_shape"][1]) for item in items)

    for _ in range(50):
        h_lo = max(GRID_H_RANGE[0], int(min_h))
        w_lo = max(GRID_W_RANGE[0], int(min_w))
        if h_lo >= GRID_H_RANGE[1] or w_lo >= GRID_W_RANGE[1]:
            return None
        h = int(rng.integers(h_lo, GRID_H_RANGE[1]))
        w = int(rng.integers(w_lo, GRID_W_RANGE[1]))
        if int(h * w) < int(total_area * 2.2):
            continue
        positions = _place_items(rng, (h, w), items)
        if positions is None:
            continue

        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full((h, w), int(bg), dtype=int)

        for idx, spec in enumerate(specs):
            frame_pos = positions[2 * idx]
            template_pos = positions[2 * idx + 1]
            _paste(inp, spec["frame_patch"], frame_pos[0], frame_pos[1])
            _paste(out, spec["frame_patch"], frame_pos[0], frame_pos[1])
            _paste(inp, spec["template_patch"], template_pos[0], template_pos[1])

            divider_abs = int(frame_pos[0] + 1 + spec["divider_interior_row"])
            inner_left = int(frame_pos[1] + 1)
            inner_right = int(inner_left + spec["inner_w"])
            for r in range(int(frame_pos[0] + 1), int(frame_pos[0] + 1 + spec["inner_h"])):
                fill = int(spec["body_color"]) if int(r) < divider_abs else int(spec["marker_color"])
                row = out[int(r), inner_left:inner_right]
                row[row == int(bg)] = fill

        if _mode_color(inp) != int(bg):
            continue
        if np.array_equal(main(inp), out):
            return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        pair = _make_input_and_output(rng)
        if pair is not None:
            inp, _out = pair
            return inp
    raise RuntimeError("failed to generate a valid input")
