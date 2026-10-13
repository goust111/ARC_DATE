from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATION_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 80


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            box = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "area": int(coords.shape[0]),
                    "mask": _crop_mask(coords, box),
                }
            )
    return comps


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    assert bg == BG, "expected black background"
    comps = _find_components(g, bg)
    assert len(comps) >= 4, "expected one blueprint and multiple stamps"

    groups: dict[tuple[tuple[int, ...], ...], list[dict]] = {}
    masks: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for comp in comps:
        key = _shape_key(comp["mask"])
        groups.setdefault(key, []).append(comp)
        masks[key] = comp["mask"]

    assert len(groups) == 2, "expected exactly two sprite shapes"
    counts = {key: len(items) for key, items in groups.items()}
    blueprint_keys = [key for key, count in counts.items() if count == 1]
    stamp_keys = [key for key, count in counts.items() if count > 1]
    assert len(blueprint_keys) == 1 and len(stamp_keys) == 1, "uniqueness must determine the roles"

    blueprint = groups[blueprint_keys[0]][0]
    blueprint_mask = masks[blueprint_keys[0]]
    stamp_mask = masks[stamp_keys[0]]
    stamps = sorted(groups[stamp_keys[0]], key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    bp_area = int(blueprint_mask.sum())
    assert bp_area >= 2, "blueprint must have multiple foreground cells"
    assert len(stamps) == bp_area, "stamp count must equal blueprint foreground area"

    return {
        "grid": g,
        "blueprint": blueprint,
        "blueprint_mask": blueprint_mask,
        "stamp_mask": stamp_mask,
        "stamps": stamps,
        "bg": bg,
    }


def _render_output(blueprint_mask: np.ndarray, stamp_mask: np.ndarray, stamp_colors: list[int]) -> np.ndarray:
    bh, bw = blueprint_mask.shape
    sh, sw = stamp_mask.shape
    out = np.zeros((bh * sh, bw * sw), dtype=int)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(blueprint_mask)]
    assert len(cells) == len(stamp_colors)
    for (br, bc), color in zip(cells, stamp_colors):
        block = out[br * sh : (br + 1) * sh, bc * sw : (bc + 1) * sw]
        block[stamp_mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the lone blueprint by stamping the repeated shape into its foreground cells in row-major order.

    Input structure:
    - The background is black.
    - The foreground consists of several small monochrome sprites in exactly two cropped shapes.
    - One shape appears exactly once as a blueprint, while the other shape appears multiple times as colored stamps.
    - The number of stamp copies is exactly the number of foreground cells in the blueprint shape.

    Rule:
    - Identify roles by uniqueness: the single-occurrence shape is the blueprint and the repeated shape is the stamp mask.
    - Sort the stamp copies by their positions in global row-major order.
    - Enumerate the blueprint foreground cells in the blueprint crop in row-major order.
    - Replace the k-th blueprint foreground cell by a copy of the stamp mask colored with the k-th stamp color, and replace every blueprint background cell by an all-black block.

    Key insight:
    - The one-off shape controls the macro layout, while the repeated shape supplies the micro mask; the equality between stamp count and blueprint area reveals the row-major one-to-one correspondence.
    """
    scene = _parse_scene(grid)
    stamp_colors = [int(comp["color"]) for comp in scene["stamps"]]
    return _render_output(scene["blueprint_mask"], scene["stamp_mask"], stamp_colors)


def _random_connected_mask(area_min: int, area_max: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        target = random.randint(area_min, area_max)
        work = np.zeros((6, 6), dtype=bool)
        cells = {(3, 3)}
        work[3, 3] = True
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 6 and 0 <= nc < 6 and not work[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            work[nr, nc] = True
            cells.add((nr, nc))

        coords = np.argwhere(work)
        mask = _crop_mask(coords, _bbox(coords))
        h, w = mask.shape
        if not (2 <= h <= max_h and 2 <= w <= max_w):
            continue
        area = int(mask.sum())
        if area != target or area == h * w:
            continue
        if h == 1 or w == 1:
            continue
        return mask
    raise RuntimeError("failed to sample a connected shape")


def _paint_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = occ.shape
    mh, mw = mask.shape
    for rr, cc in np.argwhere(mask):
        ar = int(r0 + rr)
        ac = int(c0 + cc)
        if not (0 <= ar < h and 0 <= ac < w):
            return False
        for nr in range(ar - pad, ar + pad + 1):
            for nc in range(ac - pad, ac + pad + 1):
                if 0 <= nr < h and 0 <= nc < w and occ[nr, nc]:
                    return False
    return True


def _place_object(canvas: np.ndarray, occ: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = canvas.shape
    mh, mw = mask.shape
    candidates = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _can_place(occ, mask, r0, c0):
            continue
        patch = _paint_patch(mask, color)
        window = canvas[r0 : r0 + mh, c0 : c0 + mw]
        window[mask] = patch[mask]
        occ[r0 : r0 + mh, c0 : c0 + mw][mask] = True
        return True
    return False


def _alt_prefix_blueprint_output(scene: dict) -> np.ndarray:
    shape = scene["blueprint_mask"].shape
    count = len(scene["stamps"])
    prefix = np.zeros(shape, dtype=bool)
    prefix.flat[:count] = True
    return _render_output(prefix, scene["stamp_mask"], [int(comp["color"]) for comp in scene["stamps"]])


def _alt_full_stamp_output(scene: dict) -> np.ndarray:
    full_stamp = np.ones_like(scene["stamp_mask"], dtype=bool)
    return _render_output(scene["blueprint_mask"], full_stamp, [int(comp["color"]) for comp in scene["stamps"]])


def _validate_scene(inp: np.ndarray) -> None:
    scene = _parse_scene(inp)
    blueprint_mask = scene["blueprint_mask"]
    stamp_mask = scene["stamp_mask"]
    stamps = scene["stamps"]
    stamp_colors = [int(comp["color"]) for comp in stamps]

    assert len(set(stamp_colors)) == len(stamp_colors), "stamp colors must be pairwise distinct"
    assert blueprint_mask.shape[0] > 1 and blueprint_mask.shape[1] > 1
    assert stamp_mask.shape[0] > 1 and stamp_mask.shape[1] > 1
    assert int(blueprint_mask.sum()) < blueprint_mask.size, "blueprint must use background blocks"
    assert int(stamp_mask.sum()) < stamp_mask.size, "stamp mask must matter"

    out = main(inp)
    assert out.shape == (
        blueprint_mask.shape[0] * stamp_mask.shape[0],
        blueprint_mask.shape[1] * stamp_mask.shape[1],
    )
    assert max(out.shape) <= 30

    reversed_out = _render_output(blueprint_mask, stamp_mask, list(reversed(stamp_colors)))
    assert not np.array_equal(out, reversed_out), "stamp order must affect the result"
    assert not np.array_equal(out, _alt_prefix_blueprint_output(scene)), "blueprint layout must affect the result"
    assert not np.array_equal(out, _alt_full_stamp_output(scene)), "stamp mask must affect the result"


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        blueprint_mask = _random_connected_mask(area_min=3, area_max=6, max_h=4, max_w=5)
        stamp_mask = _random_connected_mask(area_min=2, area_max=5, max_h=4, max_w=4)
        if _shape_key(blueprint_mask) == _shape_key(stamp_mask):
            continue

        stamp_count = int(blueprint_mask.sum())
        if stamp_count + 1 > len(COLOR_POOL):
            continue
        colors = random.sample(COLOR_POOL, stamp_count + 1)
        blueprint_color = colors[0]
        stamp_colors = colors[1:]

        bh, bw = blueprint_mask.shape
        sh, sw = stamp_mask.shape
        out_h = bh * sh
        out_w = bw * sw
        if max(out_h, out_w) > 30:
            continue

        object_count = 1 + stamp_count
        min_side = max(12, max(bh, bw, sh, sw) + 2)
        max_side = min(24, 12 + object_count * 2)

        built = None
        for _place_try in range(PLACEMENT_RETRIES):
            h = random.randint(min_side, max_side)
            w = random.randint(min_side, max_side)
            canvas = np.zeros((h, w), dtype=int)
            occ = np.zeros((h, w), dtype=bool)

            objects = [(blueprint_mask, blueprint_color)] + [(stamp_mask, color) for color in stamp_colors]
            random.shuffle(objects)
            ok = True
            for mask, color in objects:
                if not _place_object(canvas, occ, mask, color):
                    ok = False
                    break
            if ok:
                built = canvas
                break

        if built is None:
            continue

        try:
            _validate_scene(built)
        except AssertionError:
            continue
        return built

    raise RuntimeError("failed to generate a valid isolated-blueprint scene")
