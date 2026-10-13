import random
from collections import defaultdict, deque
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PLACEHOLDER_COLORS = (RED, GREEN)
FRAME_COLOR = GREY
LEGEND_OUTPUT_POOL = (YELLOW, PINK, ORANGE, MAROON)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    out[:, 0] -= int(out[:, 0].min())
    out[:, 1] -= int(out[:, 1].min())
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    _r0, _c0, r1, c1 = _bbox(coords)
    return int(r1 + 1), int(c1 + 1)


def _coords_to_patch(coords: np.ndarray, color: int) -> np.ndarray:
    h, w = _shape_dims(coords)
    patch = np.full((h, w), BLACK, dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME_COLOR)
    assert len(coords) > 0, "expected a gray frame"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame is too small"
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(frame[0, :] == FRAME_COLOR)
    assert np.all(frame[-1, :] == FRAME_COLOR)
    assert np.all(frame[:, 0] == FRAME_COLOR)
    assert np.all(frame[:, -1] == FRAME_COLOR)
    return r0, c0, r1, c1


def _find_top_band_height(inner: np.ndarray) -> int:
    dominant = []
    for row in inner:
        blue = int(np.count_nonzero(row == BLUE))
        teal = int(np.count_nonzero(row == TEAL))
        assert blue + teal > 0, "each interior row must expose band color"
        dominant.append(BLUE if blue >= teal else TEAL)
    split = next((i for i, col in enumerate(dominant) if col == TEAL), None)
    assert split is not None and 0 < split < len(dominant), "expected a blue-to-teal horizontal split"
    assert all(col == BLUE for col in dominant[:split])
    assert all(col == TEAL for col in dominant[split:])
    return int(split)


def _outside_components(grid: np.ndarray, outside: np.ndarray) -> list[dict[str, Any]]:
    comps: list[dict[str, Any]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BLACK, FRAME_COLOR)):
        for coords in _components(outside & (grid == color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(coords)),
                }
            )
    return comps


def _legend_map(comps: list[dict[str, Any]]) -> dict[int, int]:
    singles = {}
    for comp in comps:
        if comp["area"] == 1:
            r, c = map(int, comp["coords"][0])
            singles[(r, c)] = int(comp["color"])
    mapping: dict[int, int] = {}
    for (r, c), src in singles.items():
        below = (r + 1, c)
        if below not in singles:
            continue
        if (r - 1, c) in singles or (r + 2, c) in singles:
            continue
        mapping[int(src)] = int(singles[below])
    return mapping


def _band_background(crop_shape: tuple[int, int], top_h: int) -> np.ndarray:
    h, w = crop_shape
    out = np.full((h, w), BLACK, dtype=int)
    out[0, :] = FRAME_COLOR
    out[-1, :] = FRAME_COLOR
    out[:, 0] = FRAME_COLOR
    out[:, -1] = FRAME_COLOR
    out[1 : 1 + top_h, 1:-1] = BLUE
    out[1 + top_h : -1, 1:-1] = TEAL
    return out


def _band_of_stamp(coords: np.ndarray, top_h: int) -> int:
    top = int(np.count_nonzero(coords[:, 0] <= top_h))
    bottom = int(len(coords) - top)
    assert top != bottom, "band majority must be unique"
    return BLUE if top > bottom else TEAL


def _slide_coords(
    coords: np.ndarray,
    direction: int,
    left_wall: int,
    right_wall: int,
    occupied: np.ndarray,
) -> np.ndarray:
    cur = coords.copy()
    while True:
        nxt = cur.copy()
        nxt[:, 1] += int(direction)
        if int(nxt[:, 1].min()) < left_wall or int(nxt[:, 1].max()) > right_wall:
            return cur
        if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
            return cur
        cur = nxt


def _render_moved_items(
    crop_shape: tuple[int, int],
    top_h: int,
    items: list[dict[str, Any]],
    *,
    reverse_same_band: bool = False,
    ignore_blockers: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = _band_background(crop_shape, top_h)
    occupied = np.zeros(crop_shape, dtype=bool)
    moved_items: list[dict[str, Any]] = []
    left_wall = 1
    right_wall = crop_shape[1] - 2

    for band in (BLUE, TEAL):
        band_items = [item for item in items if item["band"] == band]
        band_items.sort(key=lambda item: (-item["distance"], item["anchor"][0], item["anchor"][1], item["src"]))
        if reverse_same_band and len(band_items) > 1:
            band_items.reverse()
        local_occupied = occupied if not ignore_blockers else np.zeros_like(occupied)
        direction = 1 if band == BLUE else -1
        for item in band_items:
            moved = _slide_coords(item["coords"], direction, left_wall, right_wall, local_occupied)
            if ignore_blockers:
                assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "independent flush collided unexpectedly"
            assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "moved sprites must not overlap"
            occupied[moved[:, 0], moved[:, 1]] = True
            out[moved[:, 0], moved[:, 1]] = int(item["tgt"])
            moved_item = dict(item)
            moved_item["final_coords"] = moved
            moved_items.append(moved_item)
    return out, moved_items


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fr0, fc0, fr1, fc1 = _find_frame_bbox(g)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    inner = crop[1:-1, 1:-1]
    top_h = _find_top_band_height(inner)

    placeholders = []
    for r in range(1, crop.shape[0] - 1):
        for c in range(1, crop.shape[1] - 1):
            color = int(crop[r, c])
            if color in PLACEHOLDER_COLORS:
                placeholders.append((color, r, c))
    assert 2 <= len(placeholders) <= 3, "expected 2-3 placeholders"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _outside_components(g, outside)
    legend = _legend_map(comps)

    sprite_shapes: dict[int, np.ndarray] = {}
    placeholder_colors = {color for color, _r, _c in placeholders}
    for color in placeholder_colors:
        candidates = [comp for comp in comps if comp["color"] == color and comp["area"] > 1]
        assert len(candidates) == 1, f"expected one sprite for color {color}"
        sprite_shapes[color] = _normalize(candidates[0]["coords"])

    items: list[dict[str, Any]] = []
    stamped = np.zeros(crop.shape, dtype=bool)
    for src, r, c in placeholders:
        assert src in legend, f"missing legend entry for color {src}"
        shape = sprite_shapes[src]
        coords = shape.copy()
        coords[:, 0] += int(r)
        coords[:, 1] += int(c)
        assert int(coords[:, 0].min()) >= 1 and int(coords[:, 0].max()) <= crop.shape[0] - 2
        assert int(coords[:, 1].min()) >= 1 and int(coords[:, 1].max()) <= crop.shape[1] - 2
        assert not np.any(stamped[coords[:, 0], coords[:, 1]]), "stamped sprites may not overlap"
        stamped[coords[:, 0], coords[:, 1]] = True
        band = _band_of_stamp(coords, top_h)
        _r0, c0, _r1, c1 = _bbox(coords)
        distance = (crop.shape[1] - 2 - c1) if band == BLUE else (c0 - 1)
        assert distance >= 1, "each sprite must have room to slide"
        items.append(
            {
                "src": int(src),
                "tgt": int(legend[src]),
                "anchor": (int(r), int(c)),
                "coords": coords,
                "band": int(band),
                "distance": int(distance),
            }
        )

    out, moved_items = _render_moved_items(crop.shape, top_h, items)
    if return_meta:
        return {
            "output": out,
            "frame_bbox": (fr0, fc0, fr1, fc1),
            "top_h": top_h,
            "items": items,
            "moved_items": moved_items,
            "crop_shape": crop.shape,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp outside sprites onto placeholder cells inside a split frame, slide them by carrier band, and recolor them via a legend.

    Input structure:
    - A black background contains one gray rectangular frame.
    - Inside the frame, the interior is split into a blue upper band and a teal lower band, with 2-3 red/green single-pixel placeholders.
    - Outside the frame, each placeholder color has one matching connected sprite, and vertical two-pixel legend pairs map source colors to final colors.

    Rule:
    - Detect the frame, recover the blue/teal bands, read the placeholders, the matching outside sprites, and the legend mappings.
    - Stamp each matching sprite with its bbox top-left aligned to the placeholder pixel, keeping the placeholder color.
    - Assign the stamped sprite to the band containing a strict majority of its pixels; blue-band sprites slide right and teal-band sprites slide left.
    - Within one band, the sprite farther from that band's target wall moves first, already moved sprites block later ones, and each final sprite is recolored by the legend while uncovered cells revert to the band colors.

    Key insight:
    - The placeholder color selects the sprite shape and the legend target color, while the placeholder position determines the carrier band and therefore the slide direction, so all clues must be combined.
    """
    return _solve(grid)


BASE_SHAPES = (
    np.array([(0, 0), (0, 1), (1, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)], dtype=int),
)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    h, w = _shape_dims(coords)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(50):
        arr = _coords_to_mask(BASE_SHAPES[rng.randrange(len(BASE_SHAPES))])
        if rng.random() < 0.5:
            arr = np.fliplr(arr)
        arr = np.rot90(arr, k=rng.randrange(4))
        coords = _normalize(np.argwhere(arr))
        if len(coords) in (3, 5):
            h, w = _shape_dims(coords)
            if h <= 3 and w <= 3 and np.any((coords == (0, 0)).all(axis=1)):
                return coords
    raise RuntimeError("failed to sample a sprite shape")


def _row_candidates(shape: np.ndarray, inner_h: int, top_h: int, band: int) -> list[dict[str, Any]]:
    h, _w = _shape_dims(shape)
    out = []
    for row in range(inner_h - h + 1):
        placed_rows = row + shape[:, 0]
        top = int(np.count_nonzero(placed_rows < top_h))
        bottom = int(len(shape) - top)
        if band == BLUE and top > bottom:
            out.append({"row": int(row), "cross": bool(top > 0 and bottom > 0), "anchor_top": bool(row < top_h)})
        elif band == TEAL and bottom > top:
            out.append({"row": int(row), "cross": bool(top > 0 and bottom > 0), "anchor_top": bool(row < top_h)})
    return out


def _shape_interval(shape: np.ndarray, row: int) -> tuple[int, int]:
    h, _w = _shape_dims(shape)
    return int(row), int(row + h - 1)


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _stamp_abs_coords(shape: np.ndarray, row: int, col: int) -> np.ndarray:
    coords = shape.copy()
    coords[:, 0] += int(1 + row)
    coords[:, 1] += int(1 + col)
    return coords


def _column_options(shape: np.ndarray, inner_w: int, band: int) -> list[int]:
    _h, w = _shape_dims(shape)
    if band == BLUE:
        return list(range(0, inner_w - w))
    return list(range(1, inner_w - w + 1))


def _assign_columns(
    rng: random.Random,
    specs: list[dict[str, Any]],
    rows: dict[int, int],
    inner_w: int,
) -> dict[int, int] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]["shape"]), reverse=True)
    cols: dict[int, int] = {}
    occupied: set[tuple[int, int]] = set()

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        options = _column_options(specs[idx]["shape"], inner_w, specs[idx]["band"])
        rng.shuffle(options)
        for col in options:
            coords = _stamp_abs_coords(specs[idx]["shape"], rows[idx], col)
            key = {tuple(map(int, xy)) for xy in coords}
            if key & occupied:
                continue
            cols[idx] = int(col)
            occupied.update(key)
            if backtrack(k + 1):
                return True
            del cols[idx]
            occupied.difference_update(key)
        return False

    return cols if backtrack(0) else None


def _place_patch(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    *,
    pad: int = 1,
) -> bool:
    h, w = patch.shape
    candidates = []
    for r in range(grid.shape[0] - h + 1):
        for c in range(grid.shape[1] - w + 1):
            rr0 = max(0, r - pad)
            cc0 = max(0, c - pad)
            rr1 = min(grid.shape[0], r + h + pad)
            cc1 = min(grid.shape[1], c + w + pad)
            if not np.any(reserved[rr0:rr1, cc0:cc1]):
                candidates.append((r, c))
    if not candidates:
        return False
    r, c = candidates[rng.randrange(len(candidates))]
    mask = patch != BLACK
    grid[r : r + h, c : c + w][mask] = patch[mask]
    reserved[max(0, r - pad) : min(grid.shape[0], r + h + pad), max(0, c - pad) : min(grid.shape[1], c + w + pad)] = True
    return True


def _semantic_checks(meta: dict[str, Any]) -> bool:
    items = meta["items"]
    counts = defaultdict(int)
    for item in items:
        counts[item["band"]] += 1
    active_bands = [band for band, count in counts.items() if count > 1]
    if active_bands:
        normal = {
            (item["src"], item["anchor"]): tuple(map(tuple, item["final_coords"]))
            for item in meta["moved_items"]
        }
        try:
            alt_out, alt_items = _render_moved_items(meta["crop_shape"], meta["top_h"], items, reverse_same_band=True)
        except AssertionError:
            alt_out, alt_items = None, None
        if alt_out is not None and alt_items is not None:
            reversed_map = {
                (item["src"], item["anchor"]): tuple(map(tuple, item["final_coords"]))
                for item in alt_items
            }
            if np.array_equal(alt_out, meta["output"]) or reversed_map == normal:
                return False
        try:
            indep_out, _ = _render_moved_items(meta["crop_shape"], meta["top_h"], items, ignore_blockers=True)
        except AssertionError:
            indep_out = None
        if indep_out is not None and np.array_equal(indep_out, meta["output"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    inner_h = rng.randint(8, 11)
    inner_w = rng.randint(11, 14)
    top_h = rng.randint(3, inner_h - 3)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    red_shape = _random_shape(rng)
    green_shape = _random_shape(rng)
    for _ in range(20):
        if tuple(map(tuple, red_shape)) != tuple(map(tuple, green_shape)):
            break
        green_shape = _random_shape(rng)
    else:
        return None
    shape_by_color = {RED: red_shape, GREEN: green_shape}

    count_pattern = rng.choices([(1, 1), (2, 1), (1, 2)], weights=[2, 5, 5], k=1)[0]
    total = sum(count_pattern)
    colors = [RED] * count_pattern[0] + [GREEN] * count_pattern[1]
    rng.shuffle(colors)

    if total == 2:
        bands = [BLUE, TEAL]
        rng.shuffle(bands)
    else:
        heavy_band = rng.choice([BLUE, TEAL])
        bands = [heavy_band, heavy_band, TEAL if heavy_band == BLUE else BLUE]
        rng.shuffle(bands)

    specs = [{"src": int(src), "shape": shape_by_color[src], "band": int(band)} for src, band in zip(colors, bands)]

    force_cross = rng.random() < 0.65
    majority_counterexample = rng.random() < 0.55
    counterexample_idx = None
    if majority_counterexample:
        teal_idxs = [i for i, spec in enumerate(specs) if spec["band"] == TEAL]
        if teal_idxs:
            counterexample_idx = rng.choice(teal_idxs)

    row_choices: list[list[dict[str, Any]]] = []
    for idx, spec in enumerate(specs):
        options = _row_candidates(spec["shape"], inner_h, top_h, spec["band"])
        if counterexample_idx == idx:
            filtered = [opt for opt in options if opt["cross"] and opt["anchor_top"]]
            if not filtered:
                return None
            options = filtered
        elif force_cross and idx == 0:
            filtered = [opt for opt in options if opt["cross"]]
            if filtered:
                options = filtered
        if not options:
            return None
        row_choices.append(options)

    for _ in range(150):
        rows = {i: row_choices[i][rng.randrange(len(row_choices[i]))]["row"] for i in range(len(specs))}
        repeated_band = next((band for band in (BLUE, TEAL) if sum(spec["band"] == band for spec in specs) > 1), None)
        if repeated_band is not None:
            idxs = [i for i, spec in enumerate(specs) if spec["band"] == repeated_band]
            ivals = [_shape_interval(specs[i]["shape"], rows[i]) for i in idxs]
            if not _intervals_overlap(ivals[0], ivals[1]):
                continue
        cols = _assign_columns(rng, specs, rows, inner_w)
        if cols is None:
            continue

        frame = _band_background((frame_h, frame_w), top_h)
        for idx, spec in enumerate(specs):
            frame[1 + rows[idx], 1 + cols[idx]] = int(spec["src"])

        use_extra_legend = rng.random() < 0.45
        extra_src = rng.choice(LEGEND_OUTPUT_POOL) if use_extra_legend else None
        available_tgts = [c for c in LEGEND_OUTPUT_POOL if c != extra_src]
        if len(available_tgts) < 2 + int(use_extra_legend):
            continue
        chosen_tgts = rng.sample(available_tgts, 2 + int(use_extra_legend))
        legend_pairs = [(RED, chosen_tgts[0]), (GREEN, chosen_tgts[1])]
        if use_extra_legend and extra_src is not None:
            legend_pairs.append((int(extra_src), int(chosen_tgts[2])))
        rng.shuffle(legend_pairs)

        canvas_h = min(30, frame_h + rng.randint(8, 12))
        canvas_w = min(30, frame_w + rng.randint(8, 12))
        if canvas_h <= frame_h + 1 or canvas_w <= frame_w + 1:
            continue
        grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)
        reserved = np.zeros((canvas_h, canvas_w), dtype=bool)
        fr0 = rng.randint(1, canvas_h - frame_h - 1)
        fc0 = rng.randint(1, canvas_w - frame_w - 1)
        grid[fr0 : fr0 + frame_h, fc0 : fc0 + frame_w] = frame
        reserved[max(0, fr0 - 1) : min(canvas_h, fr0 + frame_h + 1), max(0, fc0 - 1) : min(canvas_w, fc0 + frame_w + 1)] = True

        patches = [
            _coords_to_patch(red_shape, RED),
            _coords_to_patch(green_shape, GREEN),
        ] + [np.array([[src], [tgt]], dtype=int) for src, tgt in legend_pairs]
        patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
        ok = True
        for patch in patches:
            if not _place_patch(rng, grid, reserved, patch):
                ok = False
                break
        if not ok:
            continue

        try:
            meta = _solve(grid, return_meta=True)
        except AssertionError:
            continue

        if not _semantic_checks(meta):
            continue
        if counterexample_idx is not None:
            item = next(
                (
                    it
                    for it in meta["items"]
                    if it["anchor"] == (1 + rows[counterexample_idx], 1 + cols[counterexample_idx])
                ),
                None,
            )
            if item is None or item["band"] != TEAL or item["anchor"][0] > top_h:
                continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
