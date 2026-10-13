from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

PALETTE = list(range(10))
WALL_WIDTH = 1
MAX_GEN_ATTEMPTS = 400
HEIGHT_RANGE = (4, 6)
WIDTH_RANGE = (4, 6)
EXTRA_GAP_RANGE = (0, 2)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _hole_stats(local_mask: np.ndarray) -> tuple[int, int]:
    h, w = local_mask.shape
    empty = ~local_mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[Coord] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = empty & ~outside
    return int(holes.sum()), len(_mask_components(holes))


def _component_from_cells(grid: np.ndarray, cells: list[Coord], color: int) -> dict:
    coords = np.asarray(cells, dtype=int)
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    hole_area, hole_components = _hole_stats(local)
    return {
        "color": int(color),
        "coords": coords,
        "bbox": box,
        "mask": local,
        "hole_area": int(hole_area),
        "hole_components": int(hole_components),
    }


def _find_wall(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    wall_color = int(g[0, -1])
    assert np.all(g[:, -1] == wall_color), "last column must be the wall"

    wall_left = g.shape[1] - 1
    while wall_left - 1 >= 0 and np.all(g[:, wall_left - 1] == wall_color):
        wall_left -= 1

    wall_mask = np.zeros_like(g, dtype=bool)
    wall_mask[:, wall_left:] = True
    assert np.all(g[:, wall_left:] == wall_color), "wall must be a solid trailing strip"
    return wall_color, int(wall_left), wall_mask


def _find_objects(grid: np.ndarray, bg: int, wall_mask: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or wall_mask[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and not wall_mask[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(_component_from_cells(g, pts, color))

    return comps


def _group_bands(objects: list[dict]) -> list[dict]:
    objs = sorted(objects, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    bands: list[dict] = []
    for comp in objs:
        r0, _c0, r1, _c1 = comp["bbox"]
        if not bands or int(r0) > int(bands[-1]["r1"]):
            bands.append({"r0": int(r0), "r1": int(r1), "objects": [comp]})
        else:
            bands[-1]["r1"] = max(int(bands[-1]["r1"]), int(r1))
            bands[-1]["objects"].append(comp)
    for band in bands:
        band["objects"].sort(key=lambda comp: comp["bbox"][1])
    return bands


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    wall_color, wall_left, wall_mask = _find_wall(g)
    objects = _find_objects(g, bg, wall_mask)
    bands = _group_bands(objects)
    return {
        "grid": g,
        "bg": int(bg),
        "wall_color": int(wall_color),
        "wall_left": int(wall_left),
        "wall_mask": wall_mask,
        "bands": bands,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each holed object row-band toward the right wall so its own enclosed background area becomes the stopping gap.

    Input structure:
    - The grid has a dominant background and a solid wall occupying the full right edge.
    - One or more horizontal bands contain separate monochrome objects; each object may enclose zero or more background cells.
    - Different bands are vertically separated and do not interact.

    Rule:
    - Process each band independently, ordering its objects from nearest to farthest from the wall.
    - The nearest object slides right until the number of background columns between it and the wall equals its own enclosed background-cell count.
    - Each remaining object then slides right until the number of background columns between it and the already placed object in front equals that moving object's own enclosed background-cell count.

    Key insight:
    - Each gap belongs to the object on its left: identical-looking chains only resolve correctly if you count the enclosed hole area of every object and place the chain from the wall outward.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    g = scene["grid"]
    bg = int(scene["bg"])
    wall_color = int(scene["wall_color"])
    wall_left = int(scene["wall_left"])

    out = np.full_like(g, bg)
    out[:, wall_left:] = wall_color

    for band in scene["bands"]:
        next_left = wall_left
        for obj in reversed(band["objects"]):
            gap = int(obj["hole_area"])
            desired_right = int(next_left - gap - 1)
            shift = int(desired_right - obj["bbox"][3])
            assert shift >= 0, "objects may only slide toward the wall"
            for r, c in obj["coords"]:
                rr = int(r)
                cc = int(c + shift)
                assert out[rr, cc] == bg, "objects must not overlap after sliding"
                out[rr, cc] = int(obj["color"])
            next_left = int(obj["bbox"][1] + shift)

    return out


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _rect_candidates(h: int, w: int, min_area: int, max_area: int) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for r0 in range(1, h - 1):
        for c0 in range(1, w - 1):
            for r1 in range(r0, h - 2 + 1):
                for c1 in range(c0, w - 2 + 1):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if min_area <= area <= max_area:
                        rects.append((int(r0), int(c0), int(r1), int(c1)))
    return rects


def _carve(mask: np.ndarray, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = mask.copy()
    for r0, c0, r1, c1 in rects:
        out[r0 : r1 + 1, c0 : c1 + 1] = False
    return out


def _build_object(height: int, style: str, target_area: int | None = None) -> dict | None:
    for _ in range(120):
        width = random.randint(*WIDTH_RANGE)
        mask = np.ones((height, width), dtype=bool)
        if style == "solid":
            hole_area, hole_components = _hole_stats(mask)
            return {
                "mask": mask,
                "hole_area": int(hole_area),
                "hole_components": int(hole_components),
            }

        if style == "single":
            rects = _rect_candidates(height, width, 2, 4)
            if target_area is not None:
                rects = [rect for rect in rects if _rect_area(rect) == int(target_area)]
            if not rects:
                continue
            rect = random.choice(rects)
            obj = _carve(mask, [rect])
            hole_area, hole_components = _hole_stats(obj)
            if 2 <= hole_area <= 4 and hole_components == 1:
                return {
                    "mask": obj,
                    "hole_area": int(hole_area),
                    "hole_components": int(hole_components),
                }
            continue

        if style == "double":
            rects = _rect_candidates(height, width, 1, 2)
            random.shuffle(rects)
            for i, rect_a in enumerate(rects):
                for rect_b in rects[i + 1 :]:
                    obj = _carve(mask, [rect_a, rect_b])
                    hole_area, hole_components = _hole_stats(obj)
                    if target_area is not None and hole_area != int(target_area):
                        continue
                    if 2 <= hole_area <= 4 and hole_components == 2:
                        return {
                            "mask": obj,
                            "hole_area": int(hole_area),
                            "hole_components": int(hole_components),
                        }
    return None


def _final_width(objects: list[dict]) -> int:
    return int(sum(obj["mask"].shape[1] + obj["hole_area"] for obj in objects))


def _used_width(objects: list[dict], extras: list[int]) -> int:
    return int(sum(obj["mask"].shape[1] + obj["hole_area"] + extra for obj, extra in zip(objects, extras)))


def _assign_band_colors(objects: list[dict], palette: list[int]) -> None:
    colors = random.sample(palette, len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = int(color)


def _generate_feature_band(height: int, palette: list[int]) -> dict | None:
    for _ in range(120):
        shared_area = random.choice([2, 3])
        single = _build_object(height, "single", target_area=shared_area)
        double = _build_object(height, "double", target_area=shared_area)
        solid = _build_object(height, "solid")
        if single is None or double is None or solid is None:
            continue
        objects = [single, double, solid]
        random.shuffle(objects)
        extras = [random.randint(*EXTRA_GAP_RANGE) for _ in objects]
        if not any(extras):
            extras[random.randrange(len(extras))] = random.randint(1, EXTRA_GAP_RANGE[1])
        if _used_width(objects, extras) > 28:
            continue
        _assign_band_colors(objects, palette)
        return {"height": int(height), "objects": objects, "extras": extras}
    return None


def _generate_random_band(height: int, palette: list[int]) -> dict | None:
    for _ in range(180):
        count = random.choices([1, 2, 3], weights=[1, 3, 2], k=1)[0]
        objects: list[dict] = []
        for _obj_idx in range(count):
            style = random.choices(["solid", "single", "double"], weights=[2, 3, 2], k=1)[0]
            target = None
            if style in {"single", "double"}:
                target = random.choice([2, 3, 4])
            obj = _build_object(height, style, target_area=target)
            if obj is None:
                objects = []
                break
            objects.append(obj)
        if not objects:
            continue
        if _final_width(objects) > 24:
            continue
        extras = [random.randint(*EXTRA_GAP_RANGE) for _ in objects]
        if not any(extras):
            extras[random.randrange(len(extras))] = random.randint(1, EXTRA_GAP_RANGE[1])
        if _used_width(objects, extras) > 28:
            continue
        _assign_band_colors(objects, palette)
        return {"height": int(height), "objects": objects, "extras": extras}
    return None


def _band_tops(heights: list[int]) -> tuple[int, list[int], int] | None:
    gaps = [random.randint(1, 3) for _ in range(len(heights) - 1)]
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    total = top_margin + sum(heights) + sum(gaps) + bottom_margin
    if total > 30:
        return None
    tops: list[int] = []
    cur = top_margin
    for idx, h in enumerate(heights):
        tops.append(int(cur))
        cur += h
        if idx < len(gaps):
            cur += gaps[idx]
    return int(total), tops, top_margin


def _render_scene(height: int, width: int, bg: int, wall_color: int, bands: list[dict], *, solved: bool) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    wall_left = width - WALL_WIDTH
    grid[:, wall_left:] = int(wall_color)

    for band in bands:
        next_left = wall_left
        extras = [0] * len(band["objects"]) if solved else band["extras"]
        for obj, extra in reversed(list(zip(band["objects"], extras))):
            gap = int(obj["hole_area"] + extra)
            patch = obj["mask"]
            obj_w = int(patch.shape[1])
            c1 = int(next_left - gap - 1)
            c0 = int(c1 - obj_w + 1)
            r0 = int(band["top"])
            assert c0 >= 1, "left margin must remain positive"
            for rr, cc in np.argwhere(patch):
                ar = int(r0 + rr)
                ac = int(c0 + cc)
                assert grid[ar, ac] == bg
                grid[ar, ac] = int(obj["color"])
            next_left = c0

    return grid


def _scene_features(scene: dict) -> dict:
    objects = [obj for band in scene["bands"] for obj in band["objects"]]
    return {
        "has_zero": any(int(obj["hole_area"]) == 0 for obj in objects),
        "has_big": any(int(obj["hole_area"]) > 1 for obj in objects),
        "same_area_diff_components": any(
            int(a["hole_area"]) == int(b["hole_area"]) > 0 and int(a["hole_components"]) != int(b["hole_components"])
            for idx, a in enumerate(objects)
            for b in objects[idx + 1 :]
        ),
        "varying_chain": any(
            len(band["objects"]) >= 2 and len({int(obj["hole_area"]) for obj in band["objects"]}) > 1
            for band in scene["bands"]
        ),
    }


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    in_scene = _parse_scene(inp)
    out_scene = _parse_scene(out)

    assert inp.shape == out.shape
    assert int(in_scene["bg"]) == int(out_scene["bg"])
    assert int(in_scene["wall_left"]) == int(out_scene["wall_left"])
    assert np.array_equal(inp[:, in_scene["wall_left"] :], out[:, out_scene["wall_left"] :])
    assert len(in_scene["bands"]) == len(out_scene["bands"])

    moved = False
    for in_band, out_band in zip(in_scene["bands"], out_scene["bands"]):
        in_objects = in_band["objects"]
        out_objects = out_band["objects"]
        assert len(in_objects) == len(out_objects)
        for in_obj, out_obj in zip(in_objects, out_objects):
            assert int(in_obj["color"]) == int(out_obj["color"])
            assert np.array_equal(in_obj["mask"], out_obj["mask"])
            assert int(in_obj["hole_area"]) == int(out_obj["hole_area"])
            assert int(in_obj["bbox"][0]) == int(out_obj["bbox"][0])
            assert int(in_obj["bbox"][2]) == int(out_obj["bbox"][2])
            if int(out_obj["bbox"][1]) > int(in_obj["bbox"][1]):
                moved = True

        for idx, out_obj in enumerate(out_objects):
            if idx + 1 == len(out_objects):
                gap = int(out_scene["wall_left"] - out_obj["bbox"][3] - 1)
            else:
                gap = int(out_objects[idx + 1]["bbox"][1] - out_obj["bbox"][3] - 1)
            assert gap == int(out_obj["hole_area"])

    assert moved, "at least one object must actually slide"


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    wall_color = random.choice([color for color in PALETTE if color != bg])
    object_palette = [color for color in PALETTE if color not in {bg, wall_color}]

    n_bands = random.randint(2, 3)
    heights = [random.randint(*HEIGHT_RANGE) for _ in range(n_bands)]
    layout = _band_tops(heights)
    if layout is None:
        return None
    grid_h, tops, _top_margin = layout

    feature_idx = random.randrange(n_bands)
    bands: list[dict] = []
    for idx, band_h in enumerate(heights):
        if idx == feature_idx:
            band = _generate_feature_band(band_h, object_palette)
        else:
            band = _generate_random_band(band_h, object_palette)
        if band is None:
            return None
        band["top"] = int(tops[idx])
        bands.append(band)

    max_used = max(_used_width(band["objects"], band["extras"]) for band in bands)
    left_margin = random.randint(1, 2)
    grid_w = int(max_used + left_margin + WALL_WIDTH)
    if grid_w > 30:
        return None

    inp = _render_scene(grid_h, grid_w, bg, wall_color, bands, solved=False)
    expected = _render_scene(grid_h, grid_w, bg, wall_color, bands, solved=True)

    features = _scene_features(_parse_scene(inp))
    if not all(features.values()):
        return None
    if np.array_equal(inp, expected):
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    _validate_semantics(inp, actual)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
